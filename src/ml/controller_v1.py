from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline


SCRIPT_PATH = Path(__file__).resolve()
ML_DIR = SCRIPT_PATH.parent
SRC_DIR = SCRIPT_PATH.parents[1]
BACKTEST_DIR = SRC_DIR / "backtest"

for path in (
    ML_DIR,
    BACKTEST_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )

from baselines import (  # noqa: E402
    CostProfile,
    buy_execution,
    load_mechanical_events,
    max_affordable_quantity,
    sell_execution,
)
from largecap_feature_ablation import (  # noqa: E402
    BRANCH_BASE_FAMILIES,
    F8,
    FAMILIES,
    feature_columns,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
)


FEATURE_SET = "B4_core_plus_F8"
HORIZON = 20
TOP_K = 5
REBALANCE_SESSIONS = 20

DEVELOPMENT_YEARS = [
    2021,
    2022,
    2023,
]
TRAIN_YEARS = [
    2021,
    2022,
]
VALIDATION_YEARS = [
    2023,
]
CONFIRMATION_YEARS = [
    2024,
    2025,
    2026,
]

B4_FAMILIES = [
    *BRANCH_BASE_FAMILIES,
    "F8_cross_section",
]
B4_FEATURES = feature_columns(
    B4_FAMILIES
)

DYNAMIC_FEATURES = [
    "holding_age_sessions",
    "holding_age_fraction",
    "remaining_sessions",
    "remaining_fraction",
    "entry_signal_score",
    "entry_signal_score_pct",
    "entry_signal_rank",
    "current_score",
    "current_score_pct",
    "current_rank",
    "rank_change_from_signal",
    "score_change_from_signal",
    "score_pct_change_from_signal",
    "unrealized_gross_return",
    "running_peak_return",
    "drawdown_from_peak",
    "best_alt_score",
    "best_alt_score_pct",
    "score_gap_to_best_alt",
    "score_pct_gap_to_best_alt",
    "top5_cutoff_score",
    "score_gap_to_top5_cutoff",
    "eligible_count",
]

CONTROLLER_FEATURES = [
    *B4_FEATURES,
    *DYNAMIC_FEATURES,
]

BASE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "close",
    "turnover_median_20d",
    "unsafe_target_window_20d",
]


def unique(
    values: list[str],
) -> list[str]:
    return list(
        dict.fromkeys(
            values
        )
    )


def prediction_stage(
    stage: str,
) -> str:
    if stage == "development":
        return "branch"
    if stage == "confirmation":
        return "confirmation"
    raise ValueError(
        f"Unknown stage: {stage}"
    )


def years_for_stage(
    stage: str,
) -> list[int]:
    if stage == "development":
        return DEVELOPMENT_YEARS
    if stage == "confirmation":
        return CONFIRMATION_YEARS
    raise ValueError(
        f"Unknown stage: {stage}"
    )


def load_panel(
    root: Path,
    *,
    years: list[int],
) -> pd.DataFrame:
    panel_root = (
        root
        / "data/processed/"
        "largecap_model_panel_v2"
    )
    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            "No large-cap model-panel "
            f"partitions under {panel_root}"
        )

    requested = unique(
        BASE_COLUMNS
        + B4_FEATURES
    )

    schema = pq.read_schema(
        files[
            0
        ]
    )
    available = set(
        schema.names
    )
    missing = [
        column
        for column in requested
        if column not in available
    ]
    if missing:
        raise RuntimeError(
            "Large-cap model panel is "
            "missing controller columns: "
            f"{missing}"
        )

    frames: list[pd.DataFrame] = []

    print(
        "Loading controller panel for "
        f"{min(years)}-{max(years)}..."
    )

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

        if int(
            date.year
        ) not in years:
            continue

        frames.append(
            pd.read_parquet(
                path,
                columns=requested,
            )
        )

        if (
            len(
                frames
            )
            % 250
            == 0
        ):
            print(
                "  loaded "
                f"{len(frames):,} "
                "dates",
                flush=True,
            )

    if not frames:
        raise RuntimeError(
            "No controller panel rows "
            f"for {years}"
        )

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

    df[
        "market_day_index"
    ] = pd.to_numeric(
        df[
            "market_day_index"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    df[
        "canonical_security_id"
    ] = (
        df[
            "canonical_security_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    for column in (
        "eligible_universe",
        "unsafe_target_window_20d",
    ):
        df[
            column
        ] = (
            df[
                column
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    for column in unique(
        [
            "open",
            "close",
            "turnover_median_20d",
            *B4_FEATURES,
        ]
    ):
        df[
            column
        ] = (
            pd.to_numeric(
                df[
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
        )

    duplicate = df.duplicated(
        [
            "market_day_index",
            "canonical_security_id",
        ],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "Controller panel contains "
            "duplicate market-index/security rows."
        )

    return (
        df.sort_values(
            [
                "market_day_index",
                "canonical_security_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def attach_scores(
    df: pd.DataFrame,
    *,
    root: Path,
    stage: str,
    years: list[int],
) -> dict:
    cache_stage = prediction_stage(
        stage
    )
    base = (
        root
        / "reports/ml/"
        "largecap_feature_ablation"
        / cache_stage
        / "predictions"
        / FEATURE_SET
    )

    df[
        "persistent_score"
    ] = np.nan

    keys = [
        "date",
        "canonical_security_id",
    ]

    expected = 0
    attached = 0

    for year in years:
        path = (
            base
            / f"{year}.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "Missing frozen B4 score "
                f"cache: {path}"
            )

        saved = pd.read_parquet(
            path,
            columns=[
                "date",
                "canonical_security_id",
                "score",
            ],
        )
        saved[
            "date"
        ] = pd.to_datetime(
            saved[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()
        saved[
            "canonical_security_id"
        ] = (
            saved[
                "canonical_security_id"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        if saved.duplicated(
            keys,
            keep=False,
        ).any():
            raise RuntimeError(
                "Duplicate frozen B4 "
                f"score keys in {path}"
            )

        mask = (
            df[
                "date"
            ].dt.year.eq(
                year
            )
            & df[
                "eligible_universe"
            ]
        )

        expected += int(
            mask.sum()
        )

        mapping = (
            saved.set_index(
                keys
            )[
                "score"
            ]
        )

        idx = (
            pd.MultiIndex.from_frame(
                df.loc[
                    mask,
                    keys,
                ]
            )
        )

        values = mapping.reindex(
            idx
        ).to_numpy(
            dtype=float
        )

        df.loc[
            mask,
            "persistent_score",
        ] = values

        count = int(
            np.isfinite(
                values
            ).sum()
        )
        attached += count

        print(
            f"  {year}: attached "
            f"{count:,} B4 scores"
        )

    coverage = (
        attached
        / expected
        if expected
        else 0.0
    )

    if coverage < 0.995:
        raise RuntimeError(
            "B4 score coverage is "
            f"only {coverage:.2%}."
        )

    add_daily_ranks(
        df
    )

    return {
        "prediction_stage": (
            cache_stage
        ),
        "expected_rows": int(
            expected
        ),
        "attached_rows": int(
            attached
        ),
        "coverage": float(
            coverage
        ),
    }


def mechanical_action_windows(
    events: pd.DataFrame,
) -> dict[
    str,
    list[pd.Timestamp],
]:
    if events.empty:
        return {}

    out: dict[
        str,
        list[pd.Timestamp],
    ] = {}

    for row in events.itertuples(
        index=False
    ):
        cid = str(
            row.canonical_security_id
        )
        date = pd.Timestamp(
            row.ex_date
        ).normalize()
        out.setdefault(
            cid,
            [],
        ).append(
            date
        )

    for cid in out:
        out[
            cid
        ] = sorted(
            out[
                cid
            ]
        )

    return out


def has_mechanical_action(
    action_dates: dict[
        str,
        list[pd.Timestamp],
    ],
    *,
    cid: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    dates = action_dates.get(
        cid,
        [],
    )
    return any(
        start_date
        <= date
        <= end_date
        for date in dates
    )


def oracle_stop_labels(
    exit_returns: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = np.asarray(
        exit_returns,
        dtype=float,
    )
    n = len(
        values
    )
    hold_best = np.empty(
        n,
        dtype=float,
    )
    advantage = np.empty(
        n,
        dtype=float,
    )
    hold = np.zeros(
        n,
        dtype=bool,
    )

    best_future = -np.inf

    for i in range(
        n - 1,
        -1,
        -1,
    ):
        immediate = float(
            values[
                i
            ]
        )

        if i == n - 1:
            hold_best[
                i
            ] = immediate
            advantage[
                i
            ] = 0.0
            hold[
                i
            ] = False
        else:
            hold_best[
                i
            ] = best_future
            advantage[
                i
            ] = (
                best_future
                - immediate
            )
            hold[
                i
            ] = (
                advantage[
                    i
                ]
                > 0.0
            )

        best_future = max(
            best_future,
            immediate,
        )

    return (
        hold_best,
        advantage,
        hold,
    )


def row_lookup(
    df: pd.DataFrame,
) -> dict[
    tuple[
        int,
        str,
    ],
    int,
]:
    return {
        (
            int(
                row.market_day_index
            ),
            str(
                row.canonical_security_id
            ),
        ): int(
            index
        )
        for index, row
        in df[
            [
                "market_day_index",
                "canonical_security_id",
            ]
        ].iterrows()
    }


def build_teacher(
    root: Path,
    *,
    stage: str,
    slot_budget: float,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> dict:
    years = years_for_stage(
        stage
    )

    df = load_panel(
        root,
        years=years,
    )
    coverage = attach_scores(
        df,
        root=root,
        stage=stage,
        years=years,
    )

    events = load_mechanical_events(
        root
    )
    action_dates = (
        mechanical_action_windows(
            events
        )
    )

    costs = CostProfile(
        brokerage_per_order=(
            brokerage_per_order
        ),
        dp_charge_per_sell=(
            dp_charge_per_sell
        ),
        slippage_bps=(
            slippage_bps
        ),
    )

    indices = sorted(
        df[
            "market_day_index"
        ]
        .dropna()
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    date_by_index = (
        df[
            [
                "market_day_index",
                "date",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "market_day_index"
        )[
            "date"
        ]
        .to_dict()
    )

    lookup = row_lookup(
        df
    )

    signal_positions = list(
        range(
            0,
            len(
                indices
            ),
            REBALANCE_SESSIONS,
        )
    )

    teacher_rows: list[
        dict
    ] = []

    episodes = 0
    skipped_boundary = 0
    skipped_unsafe = 0
    skipped_missing_path = 0
    skipped_action = 0
    skipped_unaffordable = 0

    for signal_position in (
        signal_positions
    ):
        # signal close at S, buy at S+1, controller states through S+20,
        # forced final exit at open S+21.
        if (
            signal_position
            + REBALANCE_SESSIONS
            + 1
            >= len(
                indices
            )
        ):
            skipped_boundary += 1
            continue

        signal_index = indices[
            signal_position
        ]
        entry_index = indices[
            signal_position
            + 1
        ]
        final_state_position = (
            signal_position
            + REBALANCE_SESSIONS
        )
        forced_exit_position = (
            final_state_position
            + 1
        )
        forced_exit_index = indices[
            forced_exit_position
        ]

        signal_date = pd.Timestamp(
            date_by_index[
                signal_index
            ]
        ).normalize()
        forced_exit_date = pd.Timestamp(
            date_by_index[
                forced_exit_index
            ]
        ).normalize()

        # Never let a development episode use next-stage calendar data
        # merely to create its hindsight label.
        if (
            stage
            == "development"
            and int(
                signal_date.year
            )
            == max(
                DEVELOPMENT_YEARS
            )
            and int(
                forced_exit_date.year
            )
            > max(
                DEVELOPMENT_YEARS
            )
        ):
            skipped_boundary += 1
            continue

        day = df.loc[
            df[
                "market_day_index"
            ].eq(
                signal_index
            )
            & df[
                "eligible_universe"
            ]
            & df[
                "persistent_score"
            ].notna()
            & df[
                "persistent_rank"
            ].notna()
        ].sort_values(
            "persistent_rank",
            kind="stable",
        )

        if day.empty:
            continue

        selected = day.head(
            TOP_K
        )

        for signal_row in (
            selected.itertuples(
                index=False
            )
        ):
            cid = str(
                signal_row.canonical_security_id
            )

            if bool(
                signal_row.unsafe_target_window_20d
            ):
                skipped_unsafe += 1
                continue

            entry_key = (
                entry_index,
                cid,
            )
            if entry_key not in lookup:
                skipped_missing_path += 1
                continue

            entry_row = df.iloc[
                lookup[
                    entry_key
                ]
            ]
            quoted_entry = float(
                entry_row[
                    "open"
                ]
            )
            if (
                not math.isfinite(
                    quoted_entry
                )
                or quoted_entry
                <= 0
            ):
                skipped_missing_path += 1
                continue

            quantity = (
                max_affordable_quantity(
                    slot_budget,
                    quoted_entry,
                    costs,
                )
            )
            if quantity <= 0:
                skipped_unaffordable += 1
                continue

            buy = buy_execution(
                quoted_entry,
                quantity,
                costs,
            )
            entry_cash_out = float(
                buy[
                    "cash_out"
                ]
            )
            entry_execution_price = float(
                buy[
                    "execution_price"
                ]
            )

            entry_date = pd.Timestamp(
                entry_row[
                    "date"
                ]
            ).normalize()

            if has_mechanical_action(
                action_dates,
                cid=cid,
                start_date=entry_date,
                end_date=forced_exit_date,
            ):
                # Controller-v1 keeps the oracle exact by excluding the rare
                # split/bonus path rather than pretending raw prices are
                # directly comparable through a quantity-changing action.
                skipped_action += 1
                continue

            state_positions = list(
                range(
                    signal_position
                    + 1,
                    final_state_position
                    + 1,
                )
            )

            path_rows = []
            exit_returns = []
            missing = False
            peak_close = -np.inf

            for state_position in (
                state_positions
            ):
                state_index = indices[
                    state_position
                ]
                exit_index = indices[
                    state_position
                    + 1
                ]

                state_key = (
                    state_index,
                    cid,
                )
                exit_key = (
                    exit_index,
                    cid,
                )

                if (
                    state_key
                    not in lookup
                    or exit_key
                    not in lookup
                ):
                    missing = True
                    break

                state_row = df.iloc[
                    lookup[
                        state_key
                    ]
                ]
                exit_row = df.iloc[
                    lookup[
                        exit_key
                    ]
                ]

                current_close = float(
                    state_row[
                        "close"
                    ]
                )
                next_open = float(
                    exit_row[
                        "open"
                    ]
                )

                if (
                    not math.isfinite(
                        current_close
                    )
                    or current_close
                    <= 0
                    or not math.isfinite(
                        next_open
                    )
                    or next_open
                    <= 0
                ):
                    missing = True
                    break

                sale = sell_execution(
                    next_open,
                    quantity,
                    costs,
                )
                net_return = (
                    float(
                        sale[
                            "cash_in"
                        ]
                    )
                    / entry_cash_out
                    - 1.0
                )
                exit_returns.append(
                    net_return
                )

                peak_close = max(
                    peak_close,
                    current_close,
                )

                current_rank = float(
                    state_row[
                        "persistent_rank"
                    ]
                )
                current_score = float(
                    state_row[
                        "persistent_score"
                    ]
                )
                current_score_pct = float(
                    state_row[
                        "persistent_score_pct"
                    ]
                )

                same_day = df.loc[
                    df[
                        "market_day_index"
                    ].eq(
                        state_index
                    )
                    & df[
                        "eligible_universe"
                    ]
                    & df[
                        "persistent_score"
                    ].notna()
                    & df[
                        "persistent_rank"
                    ].notna()
                    & df[
                        "persistent_score_pct"
                    ].notna()
                ].sort_values(
                    "persistent_rank",
                    kind="stable",
                )

                alternatives = (
                    same_day.loc[
                        same_day[
                            "canonical_security_id"
                        ].astype(
                            str
                        ).ne(
                            cid
                        )
                    ]
                )

                if alternatives.empty:
                    best_alt_score = (
                        current_score
                    )
                    best_alt_score_pct = (
                        current_score_pct
                    )
                else:
                    best_alt = (
                        alternatives.iloc[
                            0
                        ]
                    )
                    best_alt_score = float(
                        best_alt[
                            "persistent_score"
                        ]
                    )
                    best_alt_score_pct = float(
                        best_alt[
                            "persistent_score_pct"
                        ]
                    )

                top5 = same_day.head(
                    TOP_K
                )
                if top5.empty:
                    top5_cutoff = (
                        current_score
                    )
                else:
                    top5_cutoff = float(
                        top5[
                            "persistent_score"
                        ].iloc[
                            -1
                        ]
                    )

                age = int(
                    state_position
                    - (
                        signal_position
                        + 1
                    )
                )
                remaining = int(
                    final_state_position
                    - state_position
                )

                gross_return = (
                    current_close
                    / entry_execution_price
                    - 1.0
                )
                peak_return = (
                    peak_close
                    / entry_execution_price
                    - 1.0
                )
                drawdown_from_peak = (
                    current_close
                    / peak_close
                    - 1.0
                )

                dynamic = {
                    "holding_age_sessions": (
                        float(
                            age
                        )
                    ),
                    "holding_age_fraction": (
                        float(
                            age
                            / max(
                                1,
                                HORIZON - 1,
                            )
                        )
                    ),
                    "remaining_sessions": (
                        float(
                            remaining
                        )
                    ),
                    "remaining_fraction": (
                        float(
                            remaining
                            / max(
                                1,
                                HORIZON - 1,
                            )
                        )
                    ),
                    "entry_signal_score": float(
                        signal_row.persistent_score
                    ),
                    "entry_signal_score_pct": float(
                        signal_row.persistent_score_pct
                    ),
                    "entry_signal_rank": float(
                        signal_row.persistent_rank
                    ),
                    "current_score": (
                        current_score
                    ),
                    "current_score_pct": (
                        current_score_pct
                    ),
                    "current_rank": (
                        current_rank
                    ),
                    "rank_change_from_signal": (
                        current_rank
                        - float(
                            signal_row.persistent_rank
                        )
                    ),
                    "score_change_from_signal": (
                        current_score
                        - float(
                            signal_row.persistent_score
                        )
                    ),
                    "score_pct_change_from_signal": (
                        current_score_pct
                        - float(
                            signal_row.persistent_score_pct
                        )
                    ),
                    "unrealized_gross_return": (
                        float(
                            gross_return
                        )
                    ),
                    "running_peak_return": float(
                        peak_return
                    ),
                    "drawdown_from_peak": float(
                        drawdown_from_peak
                    ),
                    "best_alt_score": float(
                        best_alt_score
                    ),
                    "best_alt_score_pct": float(
                        best_alt_score_pct
                    ),
                    "score_gap_to_best_alt": float(
                        current_score
                        - best_alt_score
                    ),
                    "score_pct_gap_to_best_alt": float(
                        current_score_pct
                        - best_alt_score_pct
                    ),
                    "top5_cutoff_score": float(
                        top5_cutoff
                    ),
                    "score_gap_to_top5_cutoff": float(
                        current_score
                        - top5_cutoff
                    ),
                    "eligible_count": float(
                        len(
                            same_day
                        )
                    ),
                }

                features = {
                    feature: (
                        float(
                            state_row[
                                feature
                            ]
                        )
                        if pd.notna(
                            state_row[
                                feature
                            ]
                        )
                        else np.nan
                    )
                    for feature in (
                        B4_FEATURES
                    )
                }

                path_rows.append({
                    "episode_id": (
                        f"{signal_date.date()}::"
                        f"{cid}"
                    ),
                    "signal_date": (
                        signal_date
                    ),
                    "entry_date": (
                        entry_date
                    ),
                    "state_date": pd.Timestamp(
                        state_row[
                            "date"
                        ]
                    ).normalize(),
                    "canonical_security_id": (
                        cid
                    ),
                    "symbol": str(
                        state_row[
                            "symbol"
                        ]
                    ),
                    "signal_market_index": int(
                        signal_index
                    ),
                    "entry_market_index": int(
                        entry_index
                    ),
                    "state_market_index": int(
                        state_index
                    ),
                    "forced_exit_market_index": int(
                        forced_exit_index
                    ),
                    "quantity": int(
                        quantity
                    ),
                    "entry_execution_price": (
                        entry_execution_price
                    ),
                    "entry_cash_out": (
                        entry_cash_out
                    ),
                    "exit_next_open_net_return": (
                        float(
                            net_return
                        )
                    ),
                    **dynamic,
                    **features,
                })

            if (
                missing
                or len(
                    path_rows
                )
                != HORIZON
            ):
                skipped_missing_path += 1
                continue

            exit_returns_array = (
                np.asarray(
                    exit_returns,
                    dtype=float,
                )
            )

            (
                hold_best,
                advantage,
                oracle_hold,
            ) = oracle_stop_labels(
                exit_returns_array
            )

            for i, row in enumerate(
                path_rows
            ):
                row[
                    "oracle_hold_best_net_return"
                ] = float(
                    hold_best[
                        i
                    ]
                )
                row[
                    "oracle_advantage_return"
                ] = float(
                    advantage[
                        i
                    ]
                )
                row[
                    "oracle_action"
                ] = (
                    "HOLD"
                    if bool(
                        oracle_hold[
                            i
                        ]
                    )
                    else "EXIT"
                )
                row[
                    "oracle_hold"
                ] = bool(
                    oracle_hold[
                        i
                    ]
                )
                row[
                    "oracle_best_net_return"
                ] = float(
                    max(
                        row[
                            "exit_next_open_net_return"
                        ],
                        hold_best[
                            i
                        ],
                    )
                )

            teacher_rows.extend(
                path_rows
            )
            episodes += 1

    teacher = pd.DataFrame(
        teacher_rows
    )

    if teacher.empty:
        raise RuntimeError(
            "Controller teacher is empty."
        )

    teacher[
        "year"
    ] = (
        teacher[
            "state_date"
        ].dt.year.astype(
            int
        )
    )

    output_root = (
        root
        / "data/processed/"
        "controller_v1"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    teacher_path = (
        output_root
        / f"{stage}_teacher.parquet"
    )
    teacher.to_parquet(
        teacher_path,
        index=False,
        compression="zstd",
    )

    reports_root = (
        root
        / "reports/ml/"
        "controller_v1"
        / stage
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    annual = (
        teacher.groupby(
            "year",
            sort=True,
        )
        .agg(
            states=(
                "oracle_hold",
                "size",
            ),
            episodes=(
                "episode_id",
                "nunique",
            ),
            oracle_hold_fraction=(
                "oracle_hold",
                "mean",
            ),
            mean_advantage=(
                "oracle_advantage_return",
                "mean",
            ),
            median_advantage=(
                "oracle_advantage_return",
                "median",
            ),
        )
        .reset_index()
    )
    annual.to_csv(
        reports_root
        / "teacher_annual.csv",
        index=False,
    )

    summary = {
        "stage": stage,
        "years": years,
        "feature_set": (
            FEATURE_SET
        ),
        "selector_feature_count": int(
            len(
                B4_FEATURES
            )
        ),
        "controller_feature_count": int(
            len(
                CONTROLLER_FEATURES
            )
        ),
        "top_k": TOP_K,
        "rebalance_sessions": (
            REBALANCE_SESSIONS
        ),
        "horizon": HORIZON,
        "slot_budget": float(
            slot_budget
        ),
        "cost_profile": asdict(
            costs
        ),
        "score_coverage": (
            coverage
        ),
        "episodes": int(
            episodes
        ),
        "states": int(
            len(
                teacher
            )
        ),
        "oracle_hold_fraction": float(
            teacher[
                "oracle_hold"
            ].mean()
        ),
        "mean_oracle_advantage": float(
            teacher[
                "oracle_advantage_return"
            ].mean()
        ),
        "skipped": {
            "boundary": int(
                skipped_boundary
            ),
            "unsafe": int(
                skipped_unsafe
            ),
            "missing_path": int(
                skipped_missing_path
            ),
            "mechanical_action": int(
                skipped_action
            ),
            "unaffordable": int(
                skipped_unaffordable
            ),
        },
        "teacher_path": str(
            teacher_path
        ),
        "oracle_definition": (
            "At each close, compare net return from EXIT at the next "
            "open against the best net exit available at any later open "
            "within the same 20-session episode. Future information is "
            "used only to create the teacher label, never a predictor."
        ),
    }

    (
        reports_root
        / "teacher_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return summary


def model_specs() -> dict:
    return {
        "histgb": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]),
        "random_forest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=20,
                    max_features=0.7,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]),
        "extra_trees": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=15,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]),
    }


def spearman(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    frame = pd.DataFrame({
        "a": np.asarray(
            a,
            dtype=float,
        ),
        "b": np.asarray(
            b,
            dtype=float,
        ),
    }).dropna()

    if len(
        frame
    ) < 3:
        return float(
            "nan"
        )

    value = frame[
        "a"
    ].corr(
        frame[
            "b"
        ],
        method="spearman",
    )

    return float(
        value
    )


def evaluate_controller(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> dict:
    target = frame[
        "oracle_advantage_return"
    ].to_numpy(
        dtype=float
    )
    pred = np.asarray(
        predictions,
        dtype=float,
    )

    oracle_hold = (
        target > 0.0
    )
    predicted_hold = (
        pred > 0.0
    )

    immediate = frame[
        "exit_next_open_net_return"
    ].to_numpy(
        dtype=float
    )
    hold_best = frame[
        "oracle_hold_best_net_return"
    ].to_numpy(
        dtype=float
    )
    oracle_best = np.maximum(
        immediate,
        hold_best,
    )
    chosen = np.where(
        predicted_hold,
        hold_best,
        immediate,
    )
    regret = (
        oracle_best
        - chosen
    )

    decision = (
        np.abs(
            target
        )
        > 1e-12
    )

    if decision.any():
        action_accuracy = float(
            (
                predicted_hold[
                    decision
                ]
                == oracle_hold[
                    decision
                ]
            ).mean()
        )
    else:
        action_accuracy = float(
            "nan"
        )

    return {
        "rows": int(
            len(
                frame
            )
        ),
        "oracle_hold_fraction": float(
            oracle_hold.mean()
        ),
        "predicted_hold_fraction": float(
            predicted_hold.mean()
        ),
        "mae_advantage": float(
            mean_absolute_error(
                target,
                pred,
            )
        ),
        "rmse_advantage": float(
            math.sqrt(
                mean_squared_error(
                    target,
                    pred,
                )
            )
        ),
        "spearman_advantage": (
            spearman(
                target,
                pred,
            )
        ),
        "action_accuracy_nonzero": (
            action_accuracy
        ),
        "mean_regret": float(
            regret.mean()
        ),
        "median_regret": float(
            np.median(
                regret
            )
        ),
        "mean_regret_bps": float(
            regret.mean()
            * 10_000.0
        ),
        "p95_regret_bps": float(
            np.quantile(
                regret,
                0.95,
            )
            * 10_000.0
        ),
    }


def train_tournament(
    root: Path,
    *,
    teacher_path: Path | None,
) -> dict:
    if teacher_path is None:
        teacher_path = (
            root
            / "data/processed/"
            "controller_v1/"
            "development_teacher.parquet"
        )

    if not teacher_path.is_file():
        raise FileNotFoundError(
            "Development teacher does not exist: "
            f"{teacher_path}. Run --mode build-teacher first."
        )

    teacher = pd.read_parquet(
        teacher_path
    )
    teacher[
        "state_date"
    ] = pd.to_datetime(
        teacher[
            "state_date"
        ],
        errors="coerce",
    ).dt.normalize()

    missing = [
        feature
        for feature in (
            CONTROLLER_FEATURES
        )
        if feature
        not in teacher.columns
    ]
    if missing:
        raise RuntimeError(
            "Teacher is missing controller "
            f"features: {missing}"
        )

    train = teacher.loc[
        teacher[
            "state_date"
        ].dt.year.isin(
            TRAIN_YEARS
        )
    ].copy()

    validation = teacher.loc[
        teacher[
            "state_date"
        ].dt.year.isin(
            VALIDATION_YEARS
        )
    ].copy()

    if (
        train.empty
        or validation.empty
    ):
        raise RuntimeError(
            "Controller train/validation "
            "split is empty."
        )

    X_train = train[
        CONTROLLER_FEATURES
    ]
    y_train = train[
        "oracle_advantage_return"
    ].astype(
        float
    )

    X_validation = validation[
        CONTROLLER_FEATURES
    ]

    reports_root = (
        root
        / "reports/ml/"
        "controller_v1/"
        "development"
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    predictions = []

    print(
        "\n=== CONTROLLER V1 TOURNAMENT ==="
    )
    print(
        f"Train states:      {len(train):,} "
        f"({TRAIN_YEARS})"
    )
    print(
        f"Validation states: {len(validation):,} "
        f"({VALIDATION_YEARS})"
    )
    print(
        "Predictors:        "
        f"{len(CONTROLLER_FEATURES)}"
    )

    for name, model in (
        model_specs().items()
    ):
        print(
            f"\n=== {name} ==="
        )
        model.fit(
            X_train,
            y_train,
        )
        pred = model.predict(
            X_validation
        )

        metrics = (
            evaluate_controller(
                validation,
                pred,
            )
        )

        rows.append({
            "model": name,
            **metrics,
        })

        prediction_frame = validation[
            [
                "episode_id",
                "state_date",
                "canonical_security_id",
                "symbol",
                "oracle_action",
                "oracle_advantage_return",
                "exit_next_open_net_return",
                "oracle_hold_best_net_return",
            ]
        ].copy()
        prediction_frame[
            "model"
        ] = name
        prediction_frame[
            "predicted_advantage"
        ] = pred
        prediction_frame[
            "predicted_action"
        ] = np.where(
            pred > 0.0,
            "HOLD",
            "EXIT",
        )
        predictions.append(
            prediction_frame
        )

        print(
            "  Spearman="
            f"{metrics['spearman_advantage']:+.3f} "
            "action-acc="
            f"{metrics['action_accuracy_nonzero']:.3f} "
            "mean-regret="
            f"{metrics['mean_regret_bps']:.1f} bps "
            "MAE="
            f"{metrics['mae_advantage'] * 10_000.0:.1f} bps"
        )

    leaderboard = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "mean_regret_bps",
                "mae_advantage",
                "spearman_advantage",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    leaderboard[
        "development_rank"
    ] = np.arange(
        1,
        len(
            leaderboard
        )
        + 1,
    )

    leaderboard.to_csv(
        reports_root
        / "controller_leaderboard.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_parquet(
        reports_root
        / "validation_predictions.parquet",
        index=False,
        compression="zstd",
    )

    best_model = str(
        leaderboard.iloc[
            0
        ][
            "model"
        ]
    )

    summary = {
        "train_years": (
            TRAIN_YEARS
        ),
        "validation_years": (
            VALIDATION_YEARS
        ),
        "feature_set": (
            FEATURE_SET
        ),
        "controller_features": int(
            len(
                CONTROLLER_FEATURES
            )
        ),
        "models": list(
            model_specs()
        ),
        "selection_rule": (
            "lowest validation mean hindsight-action regret, "
            "then MAE, then higher Spearman correlation"
        ),
        "best_model": (
            best_model
        ),
        "best_metrics": (
            leaderboard.iloc[
                0
            ].to_dict()
        ),
        "important_limit": (
            "This is one-step teacher imitation diagnostics, not yet "
            "a sequential portfolio backtest. HOLD labels assume the "
            "future oracle can continue acting optimally. The next stage "
            "must run the learned controller recursively through the "
            "historical simulator."
        ),
    }

    (
        reports_root
        / "controller_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=float,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    exits = np.asarray([
        0.01,
        -0.02,
        0.05,
    ])

    (
        hold_best,
        advantage,
        hold,
    ) = oracle_stop_labels(
        exits
    )

    assert np.allclose(
        hold_best,
        [
            0.05,
            0.05,
            0.05,
        ],
    )
    assert np.allclose(
        advantage,
        [
            0.04,
            0.07,
            0.0,
        ],
    )
    assert hold.tolist() == [
        True,
        True,
        False,
    ]

    assert len(
        B4_FEATURES
    ) == 86
    assert len(
        CONTROLLER_FEATURES
    ) == (
        86
        + len(
            DYNAMIC_FEATURES
        )
    )

    forbidden = [
        feature
        for feature in (
            CONTROLLER_FEATURES
        )
        if (
            feature.startswith(
                "oracle_"
            )
            or "target_" in feature
            or "future" in feature
        )
    ]
    assert not forbidden

    print(
        "Controller-v1 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build and evaluate the first finite-horizon "
            "optimal-stopping controller for the frozen "
            "B4 large-cap selector."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--mode",
        choices=[
            "build-teacher",
            "tournament",
        ],
        default="build-teacher",
    )
    ap.add_argument(
        "--stage",
        choices=[
            "development",
            "confirmation",
        ],
        default="development",
        help=(
            "Teacher stage. Tournament currently uses "
            "the development teacher only."
        ),
    )
    ap.add_argument(
        "--teacher-path",
        default=None,
    )
    ap.add_argument(
        "--slot-budget",
        type=float,
        default=10_000.0,
    )
    ap.add_argument(
        "--brokerage-per-order",
        type=float,
        default=15.0,
    )
    ap.add_argument(
        "--dp-charge-per-sell",
        type=float,
        default=0.0,
    )
    ap.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    root = Path(
        args.root
    ).resolve()

    if args.mode == "build-teacher":
        summary = build_teacher(
            root,
            stage=args.stage,
            slot_budget=float(
                args.slot_budget
            ),
            brokerage_per_order=float(
                args.brokerage_per_order
            ),
            dp_charge_per_sell=float(
                args.dp_charge_per_sell
            ),
            slippage_bps=float(
                args.slippage_bps
            ),
        )

        print(
            "\n=== CONTROLLER V1 TEACHER COMPLETE ==="
        )
        print(
            f"Stage:       {summary['stage']}"
        )
        print(
            f"Episodes:    {summary['episodes']:,}"
        )
        print(
            f"States:      {summary['states']:,}"
        )
        print(
            "Oracle HOLD: "
            f"{summary['oracle_hold_fraction']:.2%}"
        )
        print(
            f"Output:      {summary['teacher_path']}"
        )

    else:
        if args.stage != "development":
            raise SystemExit(
                "Tournament selection is development-only. "
                "Do not tune on confirmation."
            )

        teacher_path = (
            Path(
                args.teacher_path
            ).resolve()
            if args.teacher_path
            is not None
            else None
        )

        summary = train_tournament(
            root,
            teacher_path=(
                teacher_path
            ),
        )

        print(
            "\n=== CONTROLLER V1 TOURNAMENT COMPLETE ==="
        )
        print(
            f"Best model: {summary['best_model']}"
        )
        metrics = summary[
            "best_metrics"
        ]
        print(
            "Mean regret: "
            f"{float(metrics['mean_regret_bps']):.1f} bps"
        )
        print(
            "Action acc:  "
            f"{float(metrics['action_accuracy_nonzero']):.3f}"
        )
        print(
            "Spearman:    "
            f"{float(metrics['spearman_advantage']):+.3f}"
        )


if __name__ == "__main__":
    main()
