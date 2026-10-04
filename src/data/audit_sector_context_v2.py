from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_context(
    root: Path,
) -> pd.DataFrame:
    files = sorted(
        (
            root
            / "data/processed/"
            "sector_context_v2"
        ).glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No sector-context-v2 partitions."
        )

    frames = [
        pd.read_parquet(
            path
        )
        for path in files
    ]

    df = pd.concat(
        frames,
        ignore_index=True,
    )
    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    return df.sort_values(
        [
            "canonical_security_id",
            "date",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def proxy_stability(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    rows = []

    for security, group in df.groupby(
        "canonical_security_id",
        sort=False,
    ):
        group = group.sort_values(
            "date"
        )

        current = group[
            "sector_proxy_name"
        ]
        previous = current.shift(
            1
        )

        comparable = (
            current.notna()
            & previous.notna()
        )

        switches = (
            comparable
            & current.ne(
                previous
            )
        )

        run_id = (
            current.ne(
                previous
            )
            | previous.isna()
        ).cumsum()

        run_lengths = (
            group.loc[
                current.notna()
            ]
            .assign(
                _run=run_id.loc[
                    current.notna()
                ]
            )
            .groupby(
                "_run"
            )
            .size()
        )

        rows.append({
            "canonical_security_id": (
                security
            ),
            "rows": int(
                len(
                    group
                )
            ),
            "comparable_rows": int(
                comparable.sum()
            ),
            "switches": int(
                switches.sum()
            ),
            "switch_fraction": float(
                switches.sum()
                / max(
                    int(
                        comparable.sum()
                    ),
                    1,
                )
            ),
            "median_run_sessions": (
                float(
                    run_lengths.median()
                )
                if len(
                    run_lengths
                )
                else None
            ),
            "median_corr_252d": float(
                pd.to_numeric(
                    group[
                        "sector_proxy_corr_252d"
                    ],
                    errors="coerce",
                ).median()
            ),
            "median_margin_252d": float(
                pd.to_numeric(
                    group[
                        "sector_proxy_corr_margin_252d"
                    ],
                    errors="coerce",
                ).median()
            ),
        })

    frame = pd.DataFrame(
        rows
    )

    overall = {
        "securities": int(
            len(
                frame
            )
        ),
        "median_security_switch_fraction": float(
            frame[
                "switch_fraction"
            ].median()
        ),
        "median_security_run_sessions": float(
            frame[
                "median_run_sessions"
            ].median()
        ),
        "median_security_affinity_margin": float(
            frame[
                "median_margin_252d"
            ].median()
        ),
    }

    return (
        frame,
        overall,
    )


def basket_correlations(
    root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    path = (
        root
        / "data/processed/"
        "sector_benchmarks/"
        "nse_sector_indices.parquet"
    )

    frame = pd.read_parquet(
        path,
        columns=[
            "date",
            "sector_key",
            "return_1d",
        ],
    )

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    wide = frame.pivot(
        index="date",
        columns="sector_key",
        values="return_1d",
    )

    corr = wide.corr(
        method="spearman",
        min_periods=120,
    )

    rows = []
    columns = list(
        corr.columns
    )

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

            if pd.isna(
                value
            ):
                continue

            rows.append({
                "sector_a": left,
                "sector_b": right,
                "spearman": float(
                    value
                ),
                "abs_spearman": abs(
                    float(
                        value
                    )
                ),
            })

    pairs = (
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
    )

    return (
        corr,
        pairs,
    )


def feature_redundancy(
    df: pd.DataFrame,
    *,
    threshold: float,
    max_rows: int,
) -> pd.DataFrame:
    exclude = {
        "date",
        "market_day_index",
        "canonical_security_id",
        "symbol",
        "sector_proxy_name",
    }

    numeric = [
        column
        for column in df.columns
        if column not in exclude
        and pd.api.types.is_numeric_dtype(
            df[
                column
            ]
        )
    ]

    work = df[
        numeric
    ].copy()

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

    rows = []
    columns = list(
        corr.columns
    )

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
                "feature_b": right,
                "spearman": float(
                    value
                ),
                "abs_spearman": abs(
                    float(
                        value
                    )
                ),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "spearman",
                "abs_spearman",
            ]
        )

    return (
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
    )


def run_audit(
    root: Path,
    *,
    corr_threshold: float,
    max_corr_rows: int,
) -> dict:
    root = Path(
        root
    ).resolve()

    reports = (
        root
        / "reports/"
        "sector_context_v2/audit"
    )
    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading sector-context-v2..."
    )
    df = load_context(
        root
    )

    print(
        "Auditing proxy stability..."
    )
    stability, stability_summary = (
        proxy_stability(
            df
        )
    )

    print(
        "Auditing sector-basket overlap..."
    )
    basket_corr, basket_pairs = (
        basket_correlations(
            root
        )
    )

    print(
        "Auditing feature redundancy..."
    )
    redundant = feature_redundancy(
        df,
        threshold=corr_threshold,
        max_rows=max_corr_rows,
    )

    stability.to_csv(
        reports
        / "proxy_stability_by_security.csv",
        index=False,
    )
    basket_corr.to_csv(
        reports
        / "sector_basket_spearman.csv",
    )
    basket_pairs.to_csv(
        reports
        / "sector_basket_pairs.csv",
        index=False,
    )
    redundant.to_csv(
        reports
        / "high_correlation_feature_pairs.csv",
        index=False,
    )

    margin = pd.to_numeric(
        df[
            "sector_proxy_corr_margin_252d"
        ],
        errors="coerce",
    )

    summary = {
        "rows": int(
            len(
                df
            )
        ),
        **stability_summary,
        "proxy_margin_lt_0_01_fraction": float(
            margin.lt(
                0.01
            ).mean()
        ),
        "proxy_margin_lt_0_05_fraction": float(
            margin.lt(
                0.05
            ).mean()
        ),
        "proxy_margin_ge_0_10_fraction": float(
            margin.ge(
                0.10
            ).mean()
        ),
        "highest_sector_basket_pair_spearman": (
            None
            if basket_pairs.empty
            else float(
                basket_pairs.iloc[
                    0
                ][
                    "spearman"
                ]
            )
        ),
        "highest_sector_basket_pair": (
            None
            if basket_pairs.empty
            else [
                str(
                    basket_pairs.iloc[
                        0
                    ][
                        "sector_a"
                    ]
                ),
                str(
                    basket_pairs.iloc[
                        0
                    ][
                        "sector_b"
                    ]
                ),
            ]
        ),
        "high_feature_corr_threshold": float(
            corr_threshold
        ),
        "high_feature_corr_pairs": int(
            len(
                redundant
            )
        ),
    }

    (
        reports
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit market-implied sector "
            "proxy stability, basket overlap "
            "and feature redundancy."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
    )
    ap.add_argument(
        "--max-corr-rows",
        type=int,
        default=200_000,
    )
    args = ap.parse_args()

    summary = run_audit(
        Path(
            args.root
        ),
        corr_threshold=(
            args.corr_threshold
        ),
        max_corr_rows=(
            args.max_corr_rows
        ),
    )

    print(
        "\n=== SECTOR CONTEXT AUDIT COMPLETE ==="
    )
    print(
        "Median proxy switch fraction: "
        f"{summary['median_security_switch_fraction']:.3%}"
    )
    print(
        "Median proxy run:             "
        f"{summary['median_security_run_sessions']:.1f} sessions"
    )
    print(
        "Median affinity margin:       "
        f"{summary['median_security_affinity_margin']:.3f}"
    )
    print(
        "Margin < 0.05:                "
        f"{summary['proxy_margin_lt_0_05_fraction']:.2%}"
    )
    print(
        "High feature-corr pairs:      "
        f"{summary['high_feature_corr_pairs']:,}"
    )


if __name__ == "__main__":
    main()
