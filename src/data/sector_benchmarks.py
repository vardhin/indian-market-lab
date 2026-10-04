from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from index_benchmarks import (
    NSEIndexHistoryClient,
    cache_path,
    chunk_ranges,
    load_or_fetch,
    normalize_record,
    validate_index,
)


CORE_SECTOR_INDICES = [
    "NIFTY AUTO",
    "NIFTY BANK",
    "NIFTY FINANCIAL SERVICES",
    "NIFTY FMCG",
    "NIFTY IT",
    "NIFTY MEDIA",
    "NIFTY METAL",
    "NIFTY PHARMA",
    "NIFTY PRIVATE BANK",
    "NIFTY PSU BANK",
    "NIFTY REALTY",
    "NIFTY OIL & GAS",
]

START = pd.Timestamp(
    "2017-01-01"
)
END = pd.Timestamp(
    "2026-09-30"
)


def safe_name(
    index_name: str,
) -> str:
    return (
        index_name.lower()
        .replace(
            "&",
            "and",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


def build_sector_indices(
    root: Path,
    *,
    indices: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    refresh: bool,
) -> dict:
    root = Path(
        root
    ).resolve()

    raw_root = (
        root
        / "data/raw/"
        "sector_benchmarks"
    )
    output_root = (
        root
        / "data/processed/"
        "sector_benchmarks"
    )
    reports_root = (
        root
        / "reports/"
        "sector_benchmarks"
    )

    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = (
        NSEIndexHistoryClient()
    )

    chunks = chunk_ranges(
        start,
        end,
        days=chunk_days,
    )

    frames: list[
        pd.DataFrame
    ] = []
    summaries: dict[
        str,
        dict
    ] = {}
    failures: dict[
        str,
        str
    ] = {}

    for index_name in indices:
        print(
            f"\n=== {index_name} ==="
        )

        normalized: list[
            dict
        ] = []

        try:
            for number, (
                chunk_start,
                chunk_end,
            ) in enumerate(
                chunks,
                start=1,
            ):
                print(
                    f"[{number:02d}/"
                    f"{len(chunks):02d}] "
                    f"{chunk_start.date()} "
                    f"-> {chunk_end.date()} ",
                    end="",
                    flush=True,
                )

                rows = load_or_fetch(
                    client,
                    index_name,
                    chunk_start,
                    chunk_end,
                    raw_root,
                    refresh=refresh,
                )

                normalized.extend(
                    normalize_record(
                        row,
                        index_name,
                    )
                    for row in rows
                )

                print(
                    f"rows={len(rows):,}"
                )

            frame = pd.DataFrame(
                normalized
            )

            if frame.empty:
                raise RuntimeError(
                    "No rows returned."
                )

            frame[
                "date"
            ] = pd.to_datetime(
                frame[
                    "date"
                ],
                errors="coerce",
            ).dt.normalize()

            for column in (
                "open",
                "high",
                "low",
                "close",
                "shares_traded",
                "turnover_crore",
            ):
                frame[
                    column
                ] = pd.to_numeric(
                    frame[
                        column
                    ],
                    errors="coerce",
                )

            frame = (
                frame.loc[
                    frame[
                        "date"
                    ].notna()
                    & frame[
                        "date"
                    ].between(
                        start,
                        end,
                        inclusive="both",
                    )
                ]
                .drop_duplicates(
                    [
                        "requested_index",
                        "date",
                    ],
                    keep="last",
                )
                .sort_values(
                    "date"
                )
                .reset_index(
                    drop=True
                )
            )

            summary = validate_index(
                frame,
                requested_index=(
                    index_name
                ),
                start=start,
                end=end,
            )

            summaries[
                index_name
            ] = summary

            frame[
                "sector_key"
            ] = safe_name(
                index_name
            )

            for lookback in (
                1,
                2,
                3,
                5,
                20,
                60,
            ):
                frame[
                    f"return_{lookback}d"
                ] = (
                    frame[
                        "close"
                    ]
                    / frame[
                        "close"
                    ].shift(
                        lookback
                    )
                    - 1.0
                )

            frames.append(
                frame
            )

            safe = safe_name(
                index_name
            )
            frame.to_parquet(
                output_root
                / f"{safe}.parquet",
                index=False,
                compression="zstd",
            )

            print(
                "  normalized="
                f"{len(frame):,}, "
                f"{summary['date_min']} "
                f"-> {summary['date_max']}"
            )

        except Exception as exc:
            failures[
                index_name
            ] = str(
                exc
            )
            print(
                "  FAILED: "
                f"{exc}"
            )

    if len(
        frames
    ) < 6:
        raise RuntimeError(
            "Fewer than 6 usable official "
            "sector indices were retrieved; "
            "sector context would be too sparse. "
            f"Failures={failures}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined.to_parquet(
        output_root
        / "nse_sector_indices.parquet",
        index=False,
        compression="zstd",
    )

    summary = {
        "source": (
            "NSE historicalOR indicesHistory"
        ),
        "requested_indices": (
            indices
        ),
        "usable_indices": sorted(
            summaries
        ),
        "failures": failures,
        "date_start": str(
            start.date()
        ),
        "date_end": str(
            end.date()
        ),
        "indices": summaries,
        "combined_rows": int(
            len(
                combined
            )
        ),
        "output": str(
            output_root
            / "nse_sector_indices.parquet"
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
    assert safe_name(
        "NIFTY OIL & GAS"
    ) == (
        "nifty_oil_and_gas"
    )

    assert len(
        CORE_SECTOR_INDICES
    ) >= 10

    print(
        "Sector-benchmark self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download official NSE/NIFTY "
            "sector-index price histories "
            "for point-in-time sector context."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--start",
        default=str(
            START.date()
        ),
    )
    ap.add_argument(
        "--end",
        default=str(
            END.date()
        ),
    )
    ap.add_argument(
        "--chunk-days",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    summary = (
        build_sector_indices(
            Path(
                args.root
            ),
            indices=(
                CORE_SECTOR_INDICES
            ),
            start=pd.Timestamp(
                args.start
            ).normalize(),
            end=pd.Timestamp(
                args.end
            ).normalize(),
            chunk_days=(
                args.chunk_days
            ),
            refresh=args.refresh,
        )
    )

    print(
        "\n=== SECTOR BENCHMARKS COMPLETE ==="
    )
    print(
        f"Usable indices: "
        f"{len(summary['usable_indices'])}"
    )
    print(
        f"Failed indices: "
        f"{len(summary['failures'])}"
    )
    print(
        f"Rows:           "
        f"{summary['combined_rows']:,}"
    )
    print(
        f"Output:         "
        f"{summary['output']}"
    )


if __name__ == "__main__":
    main()
