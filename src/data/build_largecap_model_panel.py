from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RESEARCH_REQUIRED = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "close",
    "turnover",
    "turnover_median_20d",
    "turnover_median_60d",
    "unsafe_target_window_20d",
    "training_eligible_20d",
    "target_next_open_to_close_20d",
]

PIT_REQUIRED = [
    "date",
    "canonical_security_id",
    "pit_largecap_universe",
    "amfi_market_cap_rank",
    "amfi_average_market_cap_cr",
]

KEYS = [
    "date",
    "canonical_security_id",
]


def partition_map(
    root: Path,
) -> dict[
    pd.Timestamp,
    Path,
]:
    files = sorted(
        root.glob(
            "date=*/data.parquet"
        )
    )

    out: dict[
        pd.Timestamp,
        Path,
    ] = {}

    for path in files:
        name = path.parent.name
        if not name.startswith(
            "date="
        ):
            continue

        date = pd.Timestamp(
            name.split(
                "=",
                1,
            )[
                1
            ]
        ).normalize()

        out[
            date
        ] = path

    return out


def required_columns(
    path: Path,
    columns: list[str],
) -> None:
    schema = pq.read_schema(
        path
    )
    available = set(
        schema.names
    )

    missing = [
        column
        for column in columns
        if column not in available
    ]

    if missing:
        raise RuntimeError(
            f"{path}: missing required "
            f"columns {missing}"
        )


def normalize_keys(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    frame[
        "canonical_security_id"
    ] = (
        frame[
            "canonical_security_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    if "symbol" in frame:
        frame[
            "symbol"
        ] = (
            frame[
                "symbol"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

    return frame


def build_panel(
    root: Path,
) -> dict:
    root = Path(
        root
    ).resolve()

    research_root = (
        root
        / "data/processed/"
        "research_panel"
    )
    feature_root = (
        root
        / "data/processed/"
        "feature_panel_v2"
    )
    pit_root = (
        root
        / "data/processed/"
        "pit_metadata_v2"
    )
    sector_root = (
        root
        / "data/processed/"
        "sector_context_v2"
    )

    output_root = (
        root
        / "data/processed/"
        "largecap_model_panel_v2"
    )
    reports_root = (
        root
        / "reports/"
        "largecap_model_panel_v2"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    research = partition_map(
        research_root
    )
    features = partition_map(
        feature_root
    )
    pit = partition_map(
        pit_root
    )
    sector = partition_map(
        sector_root
    )

    common_dates = sorted(
        set(
            research
        )
        & set(
            features
        )
        & set(
            pit
        )
    )

    if not common_dates:
        raise RuntimeError(
            "No overlapping research/feature/"
            "PIT metadata dates."
        )

    first_paths = (
        research[
            common_dates[
                0
            ]
        ],
        features[
            common_dates[
                0
            ]
        ],
        pit[
            common_dates[
                0
            ]
        ],
    )

    required_columns(
        first_paths[
            0
        ],
        RESEARCH_REQUIRED,
    )
    required_columns(
        first_paths[
            2
        ],
        PIT_REQUIRED,
    )

    feature_schema = (
        pq.read_schema(
            first_paths[
                1
            ]
        )
    )
    feature_columns = [
        column
        for column in feature_schema.names
        if column not in {
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "eligible_universe",
        }
    ]

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_written = 0
    dates_written = 0
    securities: set[
        str
    ] = set()

    coverage_rows = []
    yearly_rows = []

    print(
        f"Building large-cap modeling panel "
        f"across {len(common_dates):,} dates..."
    )

    for number, date in enumerate(
        common_dates,
        start=1,
    ):
        research_frame = (
            normalize_keys(
                pd.read_parquet(
                    research[
                        date
                    ],
                    columns=(
                        RESEARCH_REQUIRED
                        + [
                            column
                            for column
                            in (
                                "gap_return_1d",
                                "intraday_return_1d",
                                "range_pct_1d",
                                "return_1d",
                                "return_3d",
                                "return_5d",
                                "return_10d",
                                "return_20d",
                                "return_60d",
                                "close_to_ma_5d",
                                "close_to_ma_20d",
                                "close_to_ma_60d",
                                "distance_high_5d",
                                "distance_high_20d",
                                "distance_high_60d",
                                "distance_low_5d",
                                "distance_low_20d",
                                "distance_low_60d",
                                "volatility_5d",
                                "volatility_10d",
                                "volatility_20d",
                                "volatility_60d",
                                "turnover_zscore_20d",
                                "volume_zscore_20d",
                                "active_day_ratio_20d",
                            )
                            if column
                            in pq.read_schema(
                                research[
                                    date
                                ]
                            ).names
                        ]
                    ),
                )
            )
        )

        feature_frame = (
            normalize_keys(
                pd.read_parquet(
                    features[
                        date
                    ],
                    columns=[
                        "date",
                        "canonical_security_id",
                        *feature_columns,
                    ],
                )
            )
        )

        pit_frame = normalize_keys(
            pd.read_parquet(
                pit[
                    date
                ],
                columns=PIT_REQUIRED,
            )
        )

        pit_frame = pit_frame.loc[
            pit_frame[
                "pit_largecap_universe"
            ].fillna(
                False
            )
        ].copy()

        if pit_frame.empty:
            continue

        frame = (
            research_frame.merge(
                pit_frame,
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )
        )

        frame = frame.merge(
            feature_frame,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

        sector_path = sector.get(
            date
        )

        if sector_path is not None:
            sector_schema = (
                pq.read_schema(
                    sector_path
                )
            )

            sector_columns = [
                column
                for column
                in sector_schema.names
                if column not in {
                    "date",
                    "market_day_index",
                    "canonical_security_id",
                    "symbol",
                    "amfi_market_cap_rank",
                    "amfi_average_market_cap_cr",
                }
            ]

            sector_frame = (
                normalize_keys(
                    pd.read_parquet(
                        sector_path,
                        columns=[
                            "date",
                            "canonical_security_id",
                            *sector_columns,
                        ],
                    )
                )
            )

            frame = frame.merge(
                sector_frame,
                on=KEYS,
                how="left",
                validate="one_to_one",
            )

        # Preserve the frozen close(t)-knowable research-universe
        # eligibility. PIT large-cap membership is an additional universe
        # restriction, not a replacement for liquidity/history/completeness
        # filters from the research panel.
        frame[
            "eligible_universe"
        ] = (
            frame[
                "eligible_universe"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        frame[
            "training_eligible_20d"
        ] = (
            frame[
                "training_eligible_20d"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        target = pd.to_numeric(
            frame[
                "target_next_open_to_close_20d"
            ],
            errors="coerce",
        )

        safe_target = (
            frame[
                "eligible_universe"
            ]
            & frame[
                "training_eligible_20d"
            ]
            & target.notna()
        )

        frame[
            "target_rank_20d_largecap"
        ] = np.nan

        if int(
            safe_target.sum()
        ) >= 10:
            frame.loc[
                safe_target,
                "target_rank_20d_largecap",
            ] = (
                target.loc[
                    safe_target
                ]
                .rank(
                    method="average",
                    pct=True,
                )
                .astype(
                    "float32"
                )
            )

        # Large-cap-only cross-sectional state.  The original F8
        # sidecar ranks each stock against the whole same-date research
        # panel.  Preserve that as broad-market breadth context, but
        # explicitly add ranks within today's PIT large-cap universe so
        # the two meanings are not conflated.
        largecap_rank_mask = (
            frame[
                "eligible_universe"
            ]
        )

        for source in (
            "return_5d",
            "return_20d",
            "return_60d",
            "turnover",
        ):
            output_name = (
                "largecap_rank_"
                + source
            )
            frame[
                output_name
            ] = np.nan
            frame.loc[
                largecap_rank_mask,
                output_name,
            ] = (
                pd.to_numeric(
                    frame.loc[
                        largecap_rank_mask,
                        source,
                    ],
                    errors="coerce",
                )
                .rank(
                    method="average",
                    pct=True,
                )
                .astype(
                    "float32"
                )
            )

        frame[
            "log_turnover_median_20d"
        ] = np.log1p(
            pd.to_numeric(
                frame[
                    "turnover_median_20d"
                ],
                errors="coerce",
            ).clip(
                lower=0
            )
        )

        frame[
            "log_turnover_median_60d"
        ] = np.log1p(
            pd.to_numeric(
                frame[
                    "turnover_median_60d"
                ],
                errors="coerce",
            ).clip(
                lower=0
            )
        )

        duplicate = frame.duplicated(
            KEYS,
            keep=False,
        )
        if duplicate.any():
            raise RuntimeError(
                f"{date.date()}: duplicate "
                "large-cap modeling keys."
            )

        partition = (
            output_root
            / (
                "date="
                + date.strftime(
                    "%Y-%m-%d"
                )
            )
        )
        partition.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_parquet(
            partition
            / "data.parquet",
            index=False,
            compression="zstd",
        )

        rows_written += len(
            frame
        )
        dates_written += 1

        securities.update(
            frame[
                "canonical_security_id"
            ].astype(
                str
            ).tolist()
        )

        coverage_rows.append({
            "date": date,
            "rows": int(
                len(
                    frame
                )
            ),
            "eligible_rows": int(
                frame[
                    "eligible_universe"
                ].sum()
            ),
            "training_rows": int(
                safe_target.sum()
            ),
            "sector_context_fraction": float(
                frame[
                    "sector_proxy_name"
                ].notna().mean()
            )
            if (
                "sector_proxy_name"
                in frame.columns
            )
            else 0.0,
        })

        yearly_rows.append({
            "year": int(
                date.year
            ),
            "rows": int(
                len(
                    frame
                )
            ),
            "eligible_rows": int(
                frame[
                    "eligible_universe"
                ].sum()
            ),
            "training_rows": int(
                safe_target.sum()
            ),
        })

        if (
            number % 250 == 0
            or number == len(
                common_dates
            )
        ):
            print(
                f"  processed {number:,}/"
                f"{len(common_dates):,}; "
                f"written dates={dates_written:,}",
                flush=True,
            )

    coverage = pd.DataFrame(
        coverage_rows
    )
    coverage.to_csv(
        reports_root
        / "daily_coverage.csv",
        index=False,
    )

    yearly = (
        pd.DataFrame(
            yearly_rows
        )
        .groupby(
            "year",
            sort=True,
        )
        .agg(
            rows=(
                "rows",
                "sum",
            ),
            eligible_rows=(
                "eligible_rows",
                "sum",
            ),
            training_rows=(
                "training_rows",
                "sum",
            ),
            dates=(
                "year",
                "size",
            ),
        )
        .reset_index()
    )

    yearly.to_csv(
        reports_root
        / "yearly_rows.csv",
        index=False,
    )

    summary = {
        "rows": int(
            rows_written
        ),
        "dates": int(
            dates_written
        ),
        "securities": int(
            len(
                securities
            )
        ),
        "first_date": (
            None
            if coverage.empty
            else str(
                pd.Timestamp(
                    coverage[
                        "date"
                    ].min()
                ).date()
            )
        ),
        "last_date": (
            None
            if coverage.empty
            else str(
                pd.Timestamp(
                    coverage[
                        "date"
                    ].max()
                ).date()
            )
        ),
        "target": (
            "next-open(t+1)-to-close(t+20) "
            "percentile rank within PIT "
            "large-cap universe"
        ),
        "target_column": (
            "target_rank_20d_largecap"
        ),
        "cross_section_context": {
            "broad_market": (
                "feature_panel_v2 cross_section_rank_* "
                "computed across all same-date research-panel rows"
            ),
            "largecap_only": (
                "largecap_rank_* recomputed within each "
                "PIT large-cap date partition"
            ),
        },
        "universe": (
            "eligible_universe AND "
            "point-in-time AMFI large_cap"
        ),
        "output_root": str(
            output_root
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build the consolidated point-in-time "
            "large-cap modeling panel from frozen "
            "research, v2 feature, AMFI and sector "
            "sidecars."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    args = ap.parse_args()

    summary = build_panel(
        Path(
            args.root
        )
    )

    print(
        "\n=== LARGECAP MODEL PANEL V2 COMPLETE ==="
    )
    print(
        f"Rows:        "
        f"{summary['rows']:,}"
    )
    print(
        f"Dates:       "
        f"{summary['dates']:,}"
    )
    print(
        f"Securities:  "
        f"{summary['securities']:,}"
    )
    print(
        f"Range:       "
        f"{summary['first_date']} -> "
        f"{summary['last_date']}"
    )
    print(
        f"Output:      "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
