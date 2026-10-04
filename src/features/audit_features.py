from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
FEATURE_DIR = SCRIPT_PATH.parent
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(FEATURE_DIR),
    )

from feature_registry import (  # noqa: E402
    SAFE_DEFAULT_GROUPS,
    feature_columns,
    registry_records,
)


def load_sample(
    panel_root: Path,
    *,
    columns: list[str],
    end_date: pd.Timestamp,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    files = sorted(
        Path(
            panel_root
        ).glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No enriched panel under {panel_root}"
        )

    available = set(
        pq.read_schema(
            files[0]
        ).names
    )
    required = [
        "date",
        "canonical_security_id",
        *columns,
    ]
    missing = [
        column
        for column in required
        if column not in available
    ]
    if missing:
        raise RuntimeError(
            f"Missing audit columns: {missing}"
        )

    frames: list[
        pd.DataFrame
    ] = []

    for number, path in enumerate(
        files,
        start=1,
    ):
        frame = pd.read_parquet(
            path,
            columns=required,
        )
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.normalize()
        frame = frame.loc[
            frame[
                "date"
            ].le(
                end_date
            )
        ]
        if not frame.empty:
            frames.append(
                frame
            )

        if (
            number % 1000 == 0
            or number == len(files)
        ):
            print(
                f"  scanned {number:,}/"
                f"{len(files):,}",
                flush=True,
            )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    if (
        max_rows > 0
        and len(df) > max_rows
    ):
        df = df.sample(
            n=max_rows,
            random_state=seed,
        )

    return df


def feature_quality(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        finite = values.dropna()

        rows.append({
            "feature": column,
            "rows": int(
                len(values)
            ),
            "non_null": int(
                finite.size
            ),
            "missing_fraction": float(
                values.isna().mean()
            ),
            "unique_non_null": int(
                finite.nunique()
            ),
            "mean": (
                float(
                    finite.mean()
                )
                if len(finite)
                else None
            ),
            "std": (
                float(
                    finite.std()
                )
                if len(finite)
                else None
            ),
            "q01": (
                float(
                    finite.quantile(
                        0.01
                    )
                )
                if len(finite)
                else None
            ),
            "q50": (
                float(
                    finite.quantile(
                        0.50
                    )
                )
                if len(finite)
                else None
            ),
            "q99": (
                float(
                    finite.quantile(
                        0.99
                    )
                )
                if len(finite)
                else None
            ),
        })

    return pd.DataFrame(
        rows
    )


def high_correlation_pairs(
    df: pd.DataFrame,
    columns: list[str],
    *,
    threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    numeric = (
        df[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    corr = numeric.corr(
        method="spearman",
        min_periods=100,
    )

    rows: list[
        dict
    ] = []
    for i, left in enumerate(
        columns
    ):
        for right in columns[
            i + 1:
        ]:
            value = corr.loc[
                left,
                right,
            ]
            if (
                pd.notna(
                    value
                )
                and abs(
                    float(value)
                )
                >= threshold
            ):
                rows.append({
                    "feature_a": left,
                    "feature_b": right,
                    "spearman": float(
                        value
                    ),
                    "abs_spearman": abs(
                        float(value)
                    ),
                })

    pairs = pd.DataFrame(
        rows
    )
    if not pairs.empty:
        pairs = (
            pairs.sort_values(
                "abs_spearman",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return (
        corr,
        pairs,
    )


def annual_coverage(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    frame = df.copy()
    frame[
        "year"
    ] = frame[
        "date"
    ].dt.year

    rows: list[
        dict
    ] = []

    for year, group in (
        frame.groupby(
            "year",
            sort=True,
        )
    ):
        for column in columns:
            rows.append({
                "year": int(
                    year
                ),
                "feature": column,
                "coverage": float(
                    pd.to_numeric(
                        group[
                            column
                        ],
                        errors="coerce",
                    )
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .notna()
                    .mean()
                ),
            })

    return pd.DataFrame(
        rows
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit feature missingness, "
            "redundancy and temporal coverage "
            "without using sealed evaluation "
            "returns."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--panel",
        default=(
            "data/processed/"
            "research_panel_enriched"
        ),
    )
    ap.add_argument(
        "--end-date",
        default="2023-12-31",
        help=(
            "Default deliberately stops before "
            "the sealed 2024-2026 evaluation."
        ),
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
    )
    ap.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    columns = feature_columns(
        SAFE_DEFAULT_GROUPS
    )

    print(
        "Feature audit uses development "
        f"data through {args.end_date} only."
    )

    df = load_sample(
        root
        / args.panel,
        columns=columns,
        end_date=pd.Timestamp(
            args.end_date
        ),
        max_rows=int(
            args.max_rows
        ),
        seed=int(
            args.seed
        ),
    )

    quality = feature_quality(
        df,
        columns,
    )
    corr, pairs = (
        high_correlation_pairs(
            df,
            columns,
            threshold=float(
                args.correlation_threshold
            ),
        )
    )
    coverage = annual_coverage(
        df,
        columns,
    )

    output = (
        root
        / "reports/features/audit"
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality.to_csv(
        output
        / "feature_quality.csv",
        index=False,
    )
    corr.to_csv(
        output
        / "spearman_matrix.csv",
    )
    pairs.to_csv(
        output
        / "high_correlation_pairs.csv",
        index=False,
    )
    coverage.to_csv(
        output
        / "annual_coverage.csv",
        index=False,
    )
    (
        output
        / "registry.json"
    ).write_text(
        json.dumps(
            registry_records(),
            indent=2,
        )
        + "\n"
    )

    constant = quality.loc[
        quality[
            "unique_non_null"
        ].le(1)
    ]
    high_missing = quality.loc[
        quality[
            "missing_fraction"
        ].ge(0.50)
    ]

    print(
        "\n=== FEATURE AUDIT ==="
    )
    print(
        f"Sample rows: {len(df):,}"
    )
    print(
        f"Features: {len(columns):,}"
    )
    print(
        "Constant / degenerate: "
        f"{len(constant):,}"
    )
    print(
        ">=50% missing: "
        f"{len(high_missing):,}"
    )
    print(
        "High-correlation pairs "
        f"(|rho| >= "
        f"{args.correlation_threshold:.2f}): "
        f"{len(pairs):,}"
    )

    if not pairs.empty:
        print(
            "\nTop redundant pairs:"
        )
        print(
            pairs.head(
                30
            ).to_string(
                index=False
            )
        )

    print(
        f"\nOutputs: {output}"
    )


if __name__ == "__main__":
    main()
