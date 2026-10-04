from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ISIN_RE = re.compile(
    r"^[A-Z]{2}[A-Z0-9]{10}$"
)

RESEARCH_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "isin_resolved",
    "eligible_universe",
]


def _norm_isin(
    value: object,
) -> str | None:
    if value is None:
        return None

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        str(
            value
        ).strip().upper(),
    )

    if not ISIN_RE.match(
        text
    ):
        return None

    return text


def discover_research_files(
    panel_root: Path,
) -> list[Path]:
    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No frozen research-panel "
            f"partitions under {panel_root}"
        )

    schema = pq.read_schema(
        files[0]
    )
    available = set(
        schema.names
    )
    missing = [
        column
        for column in RESEARCH_COLUMNS
        if column not in available
    ]

    if missing:
        raise RuntimeError(
            "Research panel is missing "
            "PIT-metadata columns: "
            f"{missing}"
        )

    return files


def load_amfi(
    root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    processed = (
        root
        / "data/processed/"
        "amfi_market_cap"
    )

    snapshots_path = (
        processed
        / "amfi_market_cap_snapshots.parquet"
    )
    periods_path = (
        processed
        / "amfi_market_cap_periods.parquet"
    )

    if (
        not snapshots_path.is_file()
        or not periods_path.is_file()
    ):
        raise FileNotFoundError(
            "AMFI processed archive not found. "
            "Run src/data/amfi_market_cap.py first."
        )

    snapshots = pd.read_parquet(
        snapshots_path
    )
    periods = pd.read_parquet(
        periods_path
    )

    for column in (
        "measurement_start",
        "measurement_end",
        "effective_from",
        "effective_until_exclusive",
    ):
        if column in snapshots:
            snapshots[
                column
            ] = pd.to_datetime(
                snapshots[
                    column
                ],
                errors="coerce",
            ).dt.normalize()

        if column in periods:
            periods[
                column
            ] = pd.to_datetime(
                periods[
                    column
                ],
                errors="coerce",
            ).dt.normalize()

    snapshots[
        "isin"
    ] = snapshots[
        "isin"
    ].map(
        _norm_isin
    )

    return (
        snapshots,
        periods.sort_values(
            "effective_from"
        ).reset_index(
            drop=True
        ),
    )


def active_period(
    periods: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series | None:
    date = pd.Timestamp(
        date
    ).normalize()

    eligible = periods.loc[
        periods[
            "effective_from"
        ].le(
            date
        )
        & (
            periods[
                "effective_until_exclusive"
            ].isna()
            | periods[
                "effective_until_exclusive"
            ].gt(
                date
            )
        )
    ]

    if eligible.empty:
        return None

    if len(
        eligible
    ) != 1:
        raise RuntimeError(
            "AMFI effective intervals overlap "
            f"for {date.date()}."
        )

    return eligible.iloc[
        0
    ]


def build_metadata(
    root: Path,
) -> dict:
    root = Path(
        root
    ).resolve()

    panel_root = (
        root
        / "data/processed/"
        "research_panel"
    )
    output_root = (
        root
        / "data/processed/"
        "pit_metadata_v2"
    )
    reports_root = (
        root
        / "reports/"
        "pit_metadata_v2"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = (
        discover_research_files(
            panel_root
        )
    )
    snapshots, periods = (
        load_amfi(
            root
        )
    )

    if output_root.exists():
        shutil.rmtree(
            output_root
        )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_by_end: dict[
        pd.Timestamp,
        pd.DataFrame,
    ] = {}

    keep_snapshot_columns = [
        "isin",
        "company_name",
        "nse_symbol",
        "amfi_market_cap_rank",
        "amfi_average_market_cap_cr",
        "amfi_cap_bucket",
        "measurement_start",
        "measurement_end",
        "effective_from",
        "effective_until_exclusive",
        "availability_policy",
    ]

    for end, group in snapshots.groupby(
        "measurement_end",
        sort=False,
    ):
        snapshot_by_end[
            pd.Timestamp(
                end
            ).normalize()
        ] = group[
            keep_snapshot_columns
        ].copy()

    daily_rows: list[
        dict
    ] = []

    print(
        f"Building PIT metadata for "
        f"{len(files):,} research dates..."
    )

    for i, path in enumerate(
        files,
        start=1,
    ):
        frame = pd.read_parquet(
            path,
            columns=RESEARCH_COLUMNS,
        )

        frame[
            "date"
        ] = pd.to_datetime(
            frame[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()

        unique_dates = (
            frame[
                "date"
            ].dropna().unique()
        )
        if len(
            unique_dates
        ) != 1:
            raise RuntimeError(
                f"{path}: expected exactly "
                "one date."
            )

        date = pd.Timestamp(
            unique_dates[
                0
            ]
        ).normalize()

        frame[
            "pit_isin"
        ] = frame[
            "isin_resolved"
        ].map(
            _norm_isin
        )

        missing_isin = (
            frame[
                "pit_isin"
            ].isna()
        )

        canonical = (
            frame.loc[
                missing_isin,
                "canonical_security_id",
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        canonical_isin = (
            canonical.str.replace(
                r"^ISIN:",
                "",
                regex=True,
            )
            .map(
                _norm_isin
            )
        )

        frame.loc[
            missing_isin,
            "pit_isin",
        ] = canonical_isin

        frame[
            "pit_isin_source"
        ] = "isin_resolved"
        frame.loc[
            missing_isin
            & frame[
                "pit_isin"
            ].notna(),
            "pit_isin_source",
        ] = "canonical_security_id"
        frame.loc[
            frame[
                "pit_isin"
            ].isna(),
            "pit_isin_source",
        ] = "unresolved"

        period = active_period(
            periods,
            date,
        )

        if period is None:
            merged = frame.copy()

            merged[
                "amfi_company_name"
            ] = pd.NA
            merged[
                "amfi_nse_symbol"
            ] = pd.NA
            merged[
                "amfi_market_cap_rank"
            ] = pd.Series(
                pd.NA,
                index=merged.index,
                dtype="Int32",
            )
            merged[
                "amfi_average_market_cap_cr"
            ] = float(
                "nan"
            )
            merged[
                "amfi_cap_bucket"
            ] = pd.NA
            merged[
                "amfi_measurement_start"
            ] = pd.NaT
            merged[
                "amfi_measurement_end"
            ] = pd.NaT
            merged[
                "amfi_effective_from"
            ] = pd.NaT
            merged[
                "amfi_effective_until_exclusive"
            ] = pd.NaT
            merged[
                "amfi_availability_policy"
            ] = pd.NA
        else:
            end = pd.Timestamp(
                period[
                    "measurement_end"
                ]
            ).normalize()

            active_snapshot = (
                snapshot_by_end[
                    end
                ]
                .rename(
                    columns={
                        "company_name": (
                            "amfi_company_name"
                        ),
                        "nse_symbol": (
                            "amfi_nse_symbol"
                        ),
                        "measurement_start": (
                            "amfi_measurement_start"
                        ),
                        "measurement_end": (
                            "amfi_measurement_end"
                        ),
                        "effective_from": (
                            "amfi_effective_from"
                        ),
                        "effective_until_exclusive": (
                            "amfi_effective_until_exclusive"
                        ),
                        "availability_policy": (
                            "amfi_availability_policy"
                        ),
                    }
                )
            )

            merged = frame.merge(
                active_snapshot,
                left_on="pit_isin",
                right_on="isin",
                how="left",
                validate="many_to_one",
            )

            if "isin" in merged:
                merged.drop(
                    columns=[
                        "isin"
                    ],
                    inplace=True,
                )

        merged[
            "amfi_classification_known"
        ] = merged[
            "amfi_cap_bucket"
        ].notna()

        merged[
            "amfi_is_largecap"
        ] = (
            merged[
                "amfi_cap_bucket"
            ].eq(
                "large_cap"
            )
            .where(
                merged[
                    "amfi_classification_known"
                ]
            )
            .astype(
                "boolean"
            )
        )
        merged[
            "amfi_is_midcap"
        ] = (
            merged[
                "amfi_cap_bucket"
            ].eq(
                "mid_cap"
            )
            .where(
                merged[
                    "amfi_classification_known"
                ]
            )
            .astype(
                "boolean"
            )
        )
        merged[
            "amfi_is_smallcap"
        ] = (
            merged[
                "amfi_cap_bucket"
            ].eq(
                "small_cap"
            )
            .where(
                merged[
                    "amfi_classification_known"
                ]
            )
            .astype(
                "boolean"
            )
        )

        merged[
            "pit_largecap_universe"
        ] = (
            merged[
                "eligible_universe"
            ].fillna(
                False
            )
            & merged[
                "amfi_is_largecap"
            ].fillna(
                False
            )
        )

        output = merged[
            [
                "date",
                "market_day_index",
                "canonical_security_id",
                "symbol",
                "pit_isin",
                "pit_isin_source",
                "amfi_classification_known",
                "amfi_market_cap_rank",
                "amfi_average_market_cap_cr",
                "amfi_cap_bucket",
                "amfi_is_largecap",
                "amfi_is_midcap",
                "amfi_is_smallcap",
                "pit_largecap_universe",
                "amfi_company_name",
                "amfi_nse_symbol",
                "amfi_measurement_start",
                "amfi_measurement_end",
                "amfi_effective_from",
                "amfi_effective_until_exclusive",
                "amfi_availability_policy",
            ]
        ].copy()

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
        output.to_parquet(
            partition
            / "data.parquet",
            index=False,
            compression="zstd",
        )

        eligible = merged[
            "eligible_universe"
        ].fillna(
            False
        )

        daily_rows.append({
            "date": date,
            "rows": int(
                len(
                    merged
                )
            ),
            "eligible_rows": int(
                eligible.sum()
            ),
            "known_classification_rows": int(
                merged[
                    "amfi_classification_known"
                ].sum()
            ),
            "eligible_known_classification_rows": int(
                (
                    eligible
                    & merged[
                        "amfi_classification_known"
                    ]
                ).sum()
            ),
            "largecap_rows": int(
                merged[
                    "amfi_is_largecap"
                ].fillna(
                    False
                ).sum()
            ),
            "eligible_largecap_rows": int(
                merged[
                    "pit_largecap_universe"
                ].sum()
            ),
            "active_measurement_end": (
                None
                if period is None
                else str(
                    pd.Timestamp(
                        period[
                            "measurement_end"
                        ]
                    ).date()
                )
            ),
        })

        if (
            i % 500 == 0
            or i == len(
                files
            )
        ):
            print(
                f"  wrote {i:,}/"
                f"{len(files):,}",
                flush=True,
            )

    daily = pd.DataFrame(
        daily_rows
    )
    daily.to_csv(
        reports_root
        / "daily_coverage.csv",
        index=False,
    )

    daily[
        "year"
    ] = pd.to_datetime(
        daily[
            "date"
        ]
    ).dt.year

    yearly = (
        daily.groupby(
            "year",
            sort=True,
        )
        .agg(
            dates=(
                "date",
                "nunique",
            ),
            median_eligible_rows=(
                "eligible_rows",
                "median",
            ),
            median_eligible_known=(
                "eligible_known_classification_rows",
                "median",
            ),
            median_eligible_largecap=(
                "eligible_largecap_rows",
                "median",
            ),
            first_date=(
                "date",
                "min",
            ),
            last_date=(
                "date",
                "max",
            ),
        )
        .reset_index()
    )

    yearly[
        "eligible_classification_coverage"
    ] = (
        yearly[
            "median_eligible_known"
        ]
        / yearly[
            "median_eligible_rows"
        ].where(
            yearly[
                "median_eligible_rows"
            ] > 0
        )
    )

    yearly.to_csv(
        reports_root
        / "yearly_coverage.csv",
        index=False,
    )

    first_known = daily.loc[
        daily[
            "known_classification_rows"
        ].gt(
            0
        ),
        "date",
    ]

    summary = {
        "research_dates": int(
            len(
                daily
            )
        ),
        "first_pit_classification_date": (
            None
            if first_known.empty
            else str(
                pd.Timestamp(
                    first_known.min()
                ).date()
            )
        ),
        "last_date": str(
            pd.Timestamp(
                daily[
                    "date"
                ].max()
            ).date()
        ),
        "median_largecap_rows_when_available": float(
            daily.loc[
                daily[
                    "known_classification_rows"
                ].gt(
                    0
                ),
                "largecap_rows",
            ].median()
        ),
        "median_eligible_largecap_rows_when_available": float(
            daily.loc[
                daily[
                    "known_classification_rows"
                ].gt(
                    0
                ),
                "eligible_largecap_rows",
            ].median()
        ),
        "join_key": (
            "resolved ISIN; canonical ISIN fallback only"
        ),
        "symbol_fallback_used": False,
        "availability_policy": (
            "conservative_one_full_month_after_period_end"
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


def self_test() -> None:
    periods = pd.DataFrame({
        "measurement_end": pd.to_datetime([
            "2020-06-30",
            "2020-12-31",
        ]),
        "effective_from": pd.to_datetime([
            "2020-08-01",
            "2021-02-01",
        ]),
        "effective_until_exclusive": pd.to_datetime([
            "2021-02-01",
            None,
        ]),
    })

    assert active_period(
        periods,
        pd.Timestamp(
            "2020-07-31"
        ),
    ) is None

    first = active_period(
        periods,
        pd.Timestamp(
            "2020-08-01"
        ),
    )
    assert first is not None
    assert pd.Timestamp(
        first[
            "measurement_end"
        ]
    ) == pd.Timestamp(
        "2020-06-30"
    )

    second = active_period(
        periods,
        pd.Timestamp(
            "2021-02-01"
        ),
    )
    assert second is not None
    assert pd.Timestamp(
        second[
            "measurement_end"
        ]
    ) == pd.Timestamp(
        "2020-12-31"
    )

    print(
        "PIT metadata v2 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Attach conservative point-in-time "
            "AMFI market-cap classification to "
            "the frozen research panel."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    summary = build_metadata(
        Path(
            args.root
        )
    )

    print(
        "\n=== PIT METADATA V2 COMPLETE ==="
    )
    print(
        f"Research dates:        "
        f"{summary['research_dates']:,}"
    )
    print(
        f"First classification: "
        f"{summary['first_pit_classification_date']}"
    )
    print(
        f"Median large caps:     "
        f"{summary['median_largecap_rows_when_available']:.1f}"
    )
    print(
        f"Median eligible LCs:   "
        f"{summary['median_eligible_largecap_rows_when_available']:.1f}"
    )
    print(
        f"Output:                "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
