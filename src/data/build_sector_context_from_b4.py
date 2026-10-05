from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR = SCRIPT_PATH.parent

if str(DATA_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DATA_DIR),
    )

from build_sector_context_v2 import (  # noqa: E402
    compute_security_context,
    load_broad_market_returns,
    load_research_history,
    load_sector_returns,
)


DEFAULT_YEARS = [
    2021,
    2022,
    2023,
]


def load_b4_keys(
    prediction_root: Path,
    *,
    years: list[int],
) -> pd.DataFrame:
    frames = []

    for year in years:
        path = (
            prediction_root
            / f"{year}.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "Missing frozen B4 prediction "
                f"cache: {path}"
            )

        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "market_day_index",
                "canonical_security_id",
                "symbol",
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
        frames.append(
            frame
        )

    keys = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate = keys.duplicated(
        [
            "date",
            "canonical_security_id",
        ],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "Frozen B4 caches contain duplicate "
            "date/security keys."
        )

    return keys.sort_values(
        [
            "date",
            "canonical_security_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def build(
    root: Path,
    *,
    prediction_root: Path,
    years: list[int],
) -> dict:
    root = Path(
        root
    ).resolve()
    prediction_root = Path(
        prediction_root
    ).resolve()

    keys = load_b4_keys(
        prediction_root,
        years=years,
    )

    securities = set(
        keys[
            "canonical_security_id"
        ].astype(
            str
        ).unique()
    )

    first_date = pd.Timestamp(
        keys[
            "date"
        ].min()
    ).normalize()
    last_date = pd.Timestamp(
        keys[
            "date"
        ].max()
    ).normalize()

    history_start = (
        first_date
        - pd.Timedelta(
            days=430
        )
    )

    print(
        "Frozen B4 universe: "
        f"{len(securities):,} securities, "
        f"{len(keys):,} scored date/security rows",
        flush=True,
    )

    history = load_research_history(
        root
        / "data/processed/"
        "research_panel",
        securities=securities,
        start=history_start,
    )

    sector_wide, sector_keys = (
        load_sector_returns(
            root
            / "data/processed/"
            "sector_benchmarks/"
            "nse_sector_indices.parquet"
        )
    )
    broad = load_broad_market_returns(
        root
    )

    history = history.merge(
        sector_wide,
        on="date",
        how="left",
        validate="many_to_one",
    )
    history = history.merge(
        broad,
        on="date",
        how="left",
        validate="many_to_one",
    )

    groups = list(
        history.groupby(
            "canonical_security_id",
            sort=False,
        )
    )

    print(
        "Computing B4 sector dynamics for "
        f"{len(groups):,} securities across "
        f"{len(sector_keys)} sector baskets...",
        flush=True,
    )

    pieces = []

    for i, (
        _,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):
        pieces.append(
            compute_security_context(
                group,
                sector_keys=(
                    sector_keys
                ),
            )
        )

        if (
            i % 10 == 0
            or i == len(
                groups
            )
        ):
            print(
                f"  computed {i:,}/"
                f"{len(groups):,} securities",
                flush=True,
            )

    enriched = pd.concat(
        pieces,
        ignore_index=True,
    )

    enriched[
        "date"
    ] = pd.to_datetime(
        enriched[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    output = enriched.loc[
        enriched[
            "date"
        ].between(
            first_date,
            last_date,
            inclusive="both",
        )
    ].copy()

    output_root = (
        root
        / "data/processed/"
        "sector_context_b4_development"
    )

    if output_root.exists():
        import shutil

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups_by_date = list(
        output.groupby(
            "date",
            sort=True,
        )
    )

    for i, (
        date,
        group,
    ) in enumerate(
        groups_by_date,
        start=1,
    ):
        partition = (
            output_root
            / (
                "date="
                + pd.Timestamp(
                    date
                ).strftime(
                    "%Y-%m-%d"
                )
            )
        )
        partition.mkdir(
            parents=True,
            exist_ok=True,
        )
        group.to_parquet(
            partition
            / "data.parquet",
            index=False,
            compression="zstd",
        )

        if (
            i % 100 == 0
            or i == len(
                groups_by_date
            )
        ):
            print(
                f"  wrote {i:,}/"
                f"{len(groups_by_date):,} dates",
                flush=True,
            )

    summary = {
        "years": [
            int(
                year
            )
            for year in years
        ],
        "securities": int(
            len(
                securities
            )
        ),
        "scored_keys": int(
            len(
                keys
            )
        ),
        "dates": int(
            len(
                groups_by_date
            )
        ),
        "output_root": str(
            output_root
        ),
        "universe_source": (
            "frozen B4_core_plus_F8 prediction cache keys"
        ),
    }

    print(
        "\n=== B4 DEVELOPMENT SECTOR CONTEXT COMPLETE ==="
    )
    print(
        "Securities: "
        f"{summary['securities']:,}"
    )
    print(
        "Dates:      "
        f"{summary['dates']:,}"
    )
    print(
        "Output:     "
        f"{summary['output_root']}"
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--prediction-root",
        default=(
            "reports/ml/"
            "largecap_feature_ablation/"
            "branch/predictions/"
            "B4_core_plus_F8"
        ),
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    prediction_root = (
        root
        / args.prediction_root
        if not Path(
            args.prediction_root
        ).is_absolute()
        else Path(
            args.prediction_root
        )
    )

    build(
        root,
        prediction_root=(
            prediction_root
        ),
        years=list(
            args.years
        ),
    )


if __name__ == "__main__":
    main()
