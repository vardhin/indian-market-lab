from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_registry(
    root: Path,
) -> pd.DataFrame:
    path = (
        root
        / "reports/features_v2/"
        "feature_registry.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing feature registry: {path}"
        )

    registry = pd.read_csv(
        path
    )
    required = {
        "feature",
        "family",
        "status",
    }
    missing = (
        required
        - set(
            registry.columns
        )
    )
    if missing:
        raise RuntimeError(
            "Feature registry missing columns: "
            f"{sorted(missing)}"
        )

    return registry


def select_sample_files(
    panel_root: Path,
    *,
    max_dates: int,
) -> list[Path]:
    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            "No feature-panel-v2 partitions "
            f"under {panel_root}"
        )

    if (
        max_dates <= 0
        or len(files) <= max_dates
    ):
        return files

    positions = np.linspace(
        0,
        len(files) - 1,
        num=max_dates,
        dtype=int,
    )
    positions = np.unique(
        positions
    )

    return [
        files[int(i)]
        for i in positions
    ]


def load_sample(
    files: list[Path],
    *,
    features: list[str],
) -> pd.DataFrame:
    columns = [
        "date",
        "canonical_security_id",
        "eligible_universe",
        *features,
    ]

    schema = pq.read_schema(
        files[0]
    )
    available = set(
        schema.names
    )
    missing = [
        col
        for col in columns
        if col not in available
    ]
    if missing:
        raise RuntimeError(
            "Feature panel missing columns: "
            f"{missing[:20]}"
        )

    frames = []
    for i, path in enumerate(
        files,
        start=1,
    ):
        frames.append(
            pd.read_parquet(
                path,
                columns=columns,
            )
        )
        if (
            i % 50 == 0
            or i == len(files)
        ):
            print(
                f"  loaded audit sample "
                f"{i:,}/{len(files):,}",
                flush=True,
            )

    df = pd.concat(
        frames,
        ignore_index=True,
    )
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    return df


def numeric_summary(
    df: pd.DataFrame,
    *,
    features: list[str],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    family_map = (
        registry.set_index(
            "feature"
        )[
            "family"
        ].to_dict()
    )

    rows = []
    eligible = (
        df[
            "eligible_universe"
        ].fillna(
            False
        )
    )

    for feature in features:
        values = pd.to_numeric(
            df.loc[
                eligible,
                feature,
            ],
            errors="coerce",
        )
        finite = (
            values.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if finite.empty:
            rows.append({
                "feature": feature,
                "family": family_map.get(
                    feature
                ),
                "observations": 0,
                "non_null_fraction": 0.0,
                "unique_values": 0,
                "std": None,
                "minimum": None,
                "q01": None,
                "median": None,
                "q99": None,
                "maximum": None,
                "near_constant": True,
            })
            continue

        unique_values = int(
            finite.nunique()
        )
        std = float(
            finite.std()
        ) if len(
            finite
        ) > 1 else 0.0

        rows.append({
            "feature": feature,
            "family": family_map.get(
                feature
            ),
            "observations": int(
                len(
                    finite
                )
            ),
            "non_null_fraction": float(
                len(
                    finite
                )
                / max(
                    int(
                        eligible.sum()
                    ),
                    1,
                )
            ),
            "unique_values": (
                unique_values
            ),
            "std": std,
            "minimum": float(
                finite.min()
            ),
            "q01": float(
                finite.quantile(
                    0.01
                )
            ),
            "median": float(
                finite.median()
            ),
            "q99": float(
                finite.quantile(
                    0.99
                )
            ),
            "maximum": float(
                finite.max()
            ),
            "near_constant": bool(
                unique_values <= 1
                or (
                    np.isfinite(
                        std
                    )
                    and abs(
                        std
                    ) < 1e-12
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def correlation_audit(
    df: pd.DataFrame,
    *,
    features: list[str],
    registry: pd.DataFrame,
    threshold: float,
    max_rows: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    eligible = (
        df[
            "eligible_universe"
        ].fillna(
            False
        )
    )

    work = df.loc[
        eligible,
        features,
    ].copy()

    for col in features:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    if (
        max_rows > 0
        and len(
            work
        ) > max_rows
    ):
        work = work.sample(
            n=max_rows,
            random_state=42,
        )

    corr = work.corr(
        method="spearman",
        min_periods=100,
    )

    family_map = (
        registry.set_index(
            "feature"
        )[
            "family"
        ].to_dict()
    )

    rows = []
    cols = list(
        corr.columns
    )

    for i, left in enumerate(
        cols
    ):
        for right in cols[
            i + 1:
        ]:
            value = corr.loc[
                left,
                right,
            ]
            if (
                pd.isna(
                    value
                )
                or abs(
                    float(
                        value
                    )
                )
                < threshold
            ):
                continue

            rows.append({
                "feature_a": left,
                "family_a": family_map.get(
                    left
                ),
                "feature_b": right,
                "family_b": family_map.get(
                    right
                ),
                "spearman": float(
                    value
                ),
                "abs_spearman": abs(
                    float(
                        value
                    )
                ),
            })

    high = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "abs_spearman",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
        if rows
        else pd.DataFrame(
            columns=[
                "feature_a",
                "family_a",
                "feature_b",
                "family_b",
                "spearman",
                "abs_spearman",
            ]
        )
    )

    return corr, high


def family_summary(
    stats: pd.DataFrame,
    high_corr: pd.DataFrame,
) -> pd.DataFrame:
    base = (
        stats.groupby(
            "family",
            sort=True,
        )
        .agg(
            features=(
                "feature",
                "nunique",
            ),
            median_non_null_fraction=(
                "non_null_fraction",
                "median",
            ),
            near_constant_features=(
                "near_constant",
                "sum",
            ),
        )
        .reset_index()
    )

    if high_corr.empty:
        base[
            "high_corr_pair_mentions"
        ] = 0
        return base

    mentions = pd.concat([
        high_corr[
            [
                "family_a"
            ]
        ].rename(
            columns={
                "family_a": "family"
            }
        ),
        high_corr[
            [
                "family_b"
            ]
        ].rename(
            columns={
                "family_b": "family"
            }
        ),
    ])

    counts = (
        mentions.groupby(
            "family"
        )
        .size()
        .rename(
            "high_corr_pair_mentions"
        )
        .reset_index()
    )

    return (
        base.merge(
            counts,
            on="family",
            how="left",
        )
        .fillna({
            "high_corr_pair_mentions": 0,
        })
    )


def run_audit(
    root: Path,
    *,
    max_dates: int,
    max_corr_rows: int,
    corr_threshold: float,
) -> dict:
    root = Path(
        root
    ).resolve()

    panel_root = (
        root
        / "data/processed/"
        "feature_panel_v2"
    )
    reports_root = (
        root
        / "reports/features_v2/audit"
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = load_registry(
        root
    )
    implemented = (
        registry.loc[
            registry[
                "status"
            ].eq(
                "implemented"
            ),
            "feature",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    files = select_sample_files(
        panel_root,
        max_dates=max_dates,
    )

    print(
        f"Auditing {len(files):,} "
        "deterministically spaced dates..."
    )

    sample = load_sample(
        files,
        features=implemented,
    )

    print(
        f"Audit sample rows: "
        f"{len(sample):,}"
    )

    stats = numeric_summary(
        sample,
        features=implemented,
        registry=registry,
    )

    print(
        "Computing sampled Spearman "
        "correlation audit..."
    )
    corr, high_corr = (
        correlation_audit(
            sample,
            features=implemented,
            registry=registry,
            threshold=(
                corr_threshold
            ),
            max_rows=(
                max_corr_rows
            ),
        )
    )

    families = family_summary(
        stats,
        high_corr,
    )

    stats.to_csv(
        reports_root
        / "feature_numeric_summary.csv",
        index=False,
    )
    corr.to_csv(
        reports_root
        / "spearman_matrix.csv",
    )
    high_corr.to_csv(
        reports_root
        / "high_correlation_pairs.csv",
        index=False,
    )
    families.to_csv(
        reports_root
        / "family_summary.csv",
        index=False,
    )

    summary = {
        "sample_dates": int(
            len(
                files
            )
        ),
        "sample_rows": int(
            len(
                sample
            )
        ),
        "implemented_features": int(
            len(
                implemented
            )
        ),
        "near_constant_features": (
            stats.loc[
                stats[
                    "near_constant"
                ],
                "feature",
            ].tolist()
        ),
        "features_below_80pct_eligible_coverage": (
            stats.loc[
                stats[
                    "non_null_fraction"
                ].lt(
                    0.80
                ),
                "feature",
            ].tolist()
        ),
        "high_corr_threshold": float(
            corr_threshold
        ),
        "high_corr_pairs": int(
            len(
                high_corr
            )
        ),
        "output_root": str(
            reports_root
        ),
    }

    (
        reports_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    registry = pd.DataFrame({
        "feature": [
            "a",
            "b",
            "c",
        ],
        "family": [
            "F1",
            "F1",
            "F2",
        ],
        "status": [
            "implemented",
            "implemented",
            "implemented",
        ],
    })

    n = 500
    x = np.arange(
        n,
        dtype=float,
    )
    df = pd.DataFrame({
        "eligible_universe": [
            True
        ] * n,
        "a": x,
        "b": x * 2.0,
        "c": np.sin(
            x
        ),
    })

    corr, high = (
        correlation_audit(
            df,
            features=[
                "a",
                "b",
                "c",
            ],
            registry=registry,
            threshold=0.98,
            max_rows=0,
        )
    )

    assert math.isclose(
        float(
            corr.loc[
                "a",
                "b",
            ]
        ),
        1.0,
        rel_tol=1e-12,
    )
    assert len(
        high
    ) == 1

    print(
        "Feature-panel-v2 audit "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit feature-panel-v2 coverage, "
            "degeneracy and redundancy before "
            "model training."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--max-dates",
        type=int,
        default=252,
    )
    ap.add_argument(
        "--max-corr-rows",
        type=int,
        default=250_000,
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        import math
        self_test()
        return

    summary = run_audit(
        Path(
            args.root
        ),
        max_dates=(
            args.max_dates
        ),
        max_corr_rows=(
            args.max_corr_rows
        ),
        corr_threshold=(
            args.corr_threshold
        ),
    )

    print(
        "\n=== FEATURE V2 AUDIT COMPLETE ==="
    )
    print(
        f"Sample dates:          "
        f"{summary['sample_dates']:,}"
    )
    print(
        f"Sample rows:           "
        f"{summary['sample_rows']:,}"
    )
    print(
        f"Features:              "
        f"{summary['implemented_features']:,}"
    )
    print(
        f"Near-constant:         "
        f"{len(summary['near_constant_features']):,}"
    )
    print(
        f"Coverage <80%:         "
        f"{len(summary['features_below_80pct_eligible_coverage']):,}"
    )
    print(
        f"|Spearman| >= "
        f"{summary['high_corr_threshold']:.2f}: "
        f"{summary['high_corr_pairs']:,}"
    )
    print(
        f"Output:                "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
