from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR = SCRIPT_PATH.parent
ML_DIR = SCRIPT_PATH.parents[1] / "ml"

for path in (
    DATA_DIR,
    ML_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )

from controller_v1 import (  # noqa: E402
    B4_FEATURES,
)
from largecap_feature_ablation import (  # noqa: E402
    F5A,
)


DEFAULT_YEARS = [
    2021,
    2022,
    2023,
]

BASE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "open",
    "close",
    "turnover_median_20d",
    "unsafe_target_window_20d",
]


def partition_map(
    root: Path,
) -> dict[pd.Timestamp, Path]:
    output = {}

    for path in sorted(
        root.glob(
            "date=*/data.parquet"
        )
    ):
        name = path.parent.name
        if not name.startswith(
            "date="
        ):
            continue
        output[
            pd.Timestamp(
                name.split(
                    "=",
                    1,
                )[
                    1
                ]
            ).normalize()
        ] = path

    return output


def load_b4(
    root: Path,
    years: list[int],
) -> pd.DataFrame:
    frames = []

    for year in years:
        path = (
            root
            / f"{year}.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen B4 cache: {path}"
            )

        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "market_day_index",
                "canonical_security_id",
                "symbol",
                "score",
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
        frames.append(
            frame
        )

    b4 = pd.concat(
        frames,
        ignore_index=True,
    )

    if b4.duplicated(
        [
            "date",
            "canonical_security_id",
        ],
        keep=False,
    ).any():
        raise RuntimeError(
            "Frozen B4 cache has duplicate keys."
        )

    return b4


def build(
    root: Path,
    *,
    prediction_root: Path,
    years: list[int],
) -> dict:
    b4 = load_b4(
        prediction_root,
        years,
    )

    scored_keys = set(
        zip(
            b4[
                "date"
            ].tolist(),
            b4[
                "canonical_security_id"
            ].astype(
                str
            ).tolist(),
        )
    )
    securities = set(
        b4[
            "canonical_security_id"
        ].astype(
            str
        ).unique()
    )

    first = pd.Timestamp(
        b4[
            "date"
        ].min()
    ).normalize()
    last = pd.Timestamp(
        b4[
            "date"
        ].max()
    ).normalize()

    research = partition_map(
        root
        / "data/processed/"
        "research_panel"
    )
    features = partition_map(
        root
        / "data/processed/"
        "feature_panel_v2"
    )
    sectors = partition_map(
        root
        / "data/processed/"
        "sector_context_b4_development"
    )

    dates = sorted(
        date
        for date in (
            set(
                research
            )
            & set(
                features
            )
            & set(
                sectors
            )
        )
        if (
            first
            <= date
            <= last
        )
    )

    if not dates:
        raise RuntimeError(
            "No overlapping development dates "
            "for research/features/sector context."
        )

    feature_side = [
        column
        for column in B4_FEATURES
        if column not in set(
            F5A
        )
    ]

    first_feature_schema = set(
        pq.read_schema(
            features[
                dates[
                    0
                ]
            ]
        ).names
    )
    missing_feature = [
        column
        for column in feature_side
        if column
        not in first_feature_schema
    ]
    if missing_feature:
        raise RuntimeError(
            "feature_panel_v2 lacks B4 columns: "
            f"{missing_feature}"
        )

    first_sector_schema = set(
        pq.read_schema(
            sectors[
                dates[
                    0
                ]
            ]
        ).names
    )
    missing_sector = [
        column
        for column in F5A
        if column
        not in first_sector_schema
    ]
    if missing_sector:
        raise RuntimeError(
            "B4 sector context lacks F5a columns: "
            f"{missing_sector}"
        )

    output_root = (
        root
        / "data/processed/"
        "largecap_model_panel_v2"
    )
    if output_root.exists():
        shutil.rmtree(
            output_root
        )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = 0

    for number, date in enumerate(
        dates,
        start=1,
    ):
        research_frame = pd.read_parquet(
            research[
                date
            ],
            columns=BASE_COLUMNS,
        )
        research_frame[
            "date"
        ] = pd.to_datetime(
            research_frame[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()
        research_frame[
            "canonical_security_id"
        ] = (
            research_frame[
                "canonical_security_id"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )
        research_frame = research_frame.loc[
            research_frame[
                "canonical_security_id"
            ].isin(
                securities
            )
        ].copy()

        feature_frame = pd.read_parquet(
            features[
                date
            ],
            columns=[
                "date",
                "canonical_security_id",
                *feature_side,
            ],
        )
        feature_frame[
            "date"
        ] = pd.to_datetime(
            feature_frame[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()
        feature_frame[
            "canonical_security_id"
        ] = (
            feature_frame[
                "canonical_security_id"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        sector_frame = pd.read_parquet(
            sectors[
                date
            ],
            columns=[
                "date",
                "canonical_security_id",
                *F5A,
            ],
        )
        sector_frame[
            "date"
        ] = pd.to_datetime(
            sector_frame[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()
        sector_frame[
            "canonical_security_id"
        ] = (
            sector_frame[
                "canonical_security_id"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        frame = research_frame.merge(
            feature_frame,
            on=[
                "date",
                "canonical_security_id",
            ],
            how="left",
            validate="one_to_one",
        )
        frame = frame.merge(
            sector_frame,
            on=[
                "date",
                "canonical_security_id",
            ],
            how="left",
            validate="one_to_one",
        )

        frame[
            "eligible_universe"
        ] = [
            (
                date,
                str(
                    cid
                ),
            )
            in scored_keys
            for cid in frame[
                "canonical_security_id"
            ]
        ]

        for column in B4_FEATURES:
            if column not in frame:
                frame[
                    column
                ] = np.nan

        ordered = [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "eligible_universe",
            "open",
            "close",
            "turnover_median_20d",
            "unsafe_target_window_20d",
            *B4_FEATURES,
        ]
        frame = frame[
            list(
                dict.fromkeys(
                    ordered
                )
            )
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
        frame.to_parquet(
            partition
            / "data.parquet",
            index=False,
            compression="zstd",
        )
        rows += len(
            frame
        )

        if (
            number % 100 == 0
            or number == len(
                dates
            )
        ):
            print(
                f"  wrote {number:,}/"
                f"{len(dates):,} dates; "
                f"rows={rows:,}",
                flush=True,
            )

    # Verify the exact score-key universe is present and eligible.
    checks = []

    for date in dates:
        path = (
            output_root
            / (
                "date="
                + date.strftime(
                    "%Y-%m-%d"
                )
            )
            / "data.parquet"
        )
        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "canonical_security_id",
                "eligible_universe",
            ],
        )
        eligible = frame.loc[
            frame[
                "eligible_universe"
            ].fillna(
                False
            )
        ]
        checks.extend(
            zip(
                pd.to_datetime(
                    eligible[
                        "date"
                    ]
                ).dt.normalize().tolist(),
                eligible[
                    "canonical_security_id"
                ].astype(
                    str
                ).tolist(),
            )
        )

    rebuilt_keys = set(
        checks
    )
    if rebuilt_keys != scored_keys:
        missing = len(
            scored_keys
            - rebuilt_keys
        )
        extra = len(
            rebuilt_keys
            - scored_keys
        )
        raise RuntimeError(
            "Rebuilt development universe does not "
            "match frozen B4 keys: "
            f"missing={missing}, extra={extra}"
        )

    print(
        "\n=== B4 ORACLE DEVELOPMENT PANEL COMPLETE ==="
    )
    print(
        f"Dates:      {len(dates):,}"
    )
    print(
        f"Rows:       {rows:,}"
    )
    print(
        f"B4 keys:    {len(scored_keys):,}"
    )
    print(
        "Eligibility: exact frozen B4 key match ✓"
    )

    return {
        "dates": int(
            len(
                dates
            )
        ),
        "rows": int(
            rows
        ),
        "scored_keys": int(
            len(
                scored_keys
            )
        ),
        "output_root": str(
            output_root
        ),
    }


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
