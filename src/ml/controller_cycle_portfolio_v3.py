from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ML_DIR = SCRIPT_PATH.parent
BACKTEST_DIR = SCRIPT_PATH.parents[1] / "backtest"

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
    build_day_index,
    build_events_by_market_index,
    buy_execution,
    load_mechanical_events,
    max_affordable_quantity,
)
from quant_metrics import (  # noqa: E402
    compute_performance_metrics,
)
from persistent_portfolio import (  # noqa: E402
    close_position,
)
from controller_v1 import (  # noqa: E402
    B4_FEATURES,
    CONTROLLER_FEATURES,
    DEVELOPMENT_YEARS,
    TOP_K,
    REBALANCE_SESSIONS,
    attach_scores,
    load_panel,
)
from controller_fqi_v2 import (  # noqa: E402
    fitted_q_iteration,
    predict_advantage,
)


VALIDATION_YEAR = 2023
MODEL_NAME = "random_forest"
FQI_ITERATIONS = 12
RANDOM_STATE = 242

POLICIES = [
    {
        "name": "base_forced20",
        "threshold": None,
    },
    {
        "name": "rf_m005",
        "threshold": -0.005,
    },
    {
        "name": "rf_000",
        "threshold": 0.0,
    },
    {
        "name": "rf_010",
        "threshold": 0.01,
    },
]


def load_fqi_models(
    root: Path,
):
    teacher_path = (
        root
        / "data/processed/"
        "controller_v1/"
        "development_teacher.parquet"
    )
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing controller teacher: {teacher_path}"
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

    train = teacher.loc[
        teacher[
            "state_date"
        ].dt.year.isin(
            [
                2021,
                2022,
            ]
        )
    ].copy()

    if train.empty:
        raise RuntimeError(
            "FQI training subset is empty."
        )

    print(
        "Training frozen development FQI "
        f"{MODEL_NAME} on {len(train):,} states..."
    )

    (
        exit_model,
        hold_model,
        diagnostics,
    ) = fitted_q_iteration(
        train,
        model_name=MODEL_NAME,
        iterations=FQI_ITERATIONS,
        random_state=RANDOM_STATE,
    )

    return (
        exit_model,
        hold_model,
        diagnostics,
    )


def signal_dates_from_teacher(
    root: Path,
) -> list[pd.Timestamp]:
    path = (
        root
        / "data/processed/"
        "controller_v1/"
        "development_teacher.parquet"
    )
    teacher = pd.read_parquet(
        path,
        columns=[
            "signal_date",
            "state_date",
        ],
    )

    for column in (
        "signal_date",
        "state_date",
    ):
        teacher[
            column
        ] = pd.to_datetime(
            teacher[
                column
            ],
            errors="coerce",
        ).dt.normalize()

    dates = (
        teacher.loc[
            teacher[
                "state_date"
            ].dt.year.eq(
                VALIDATION_YEAR
            ),
            "signal_date",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return [
        pd.Timestamp(
            date
        ).normalize()
        for date in dates
    ]


def controller_state_frame(
    *,
    current_row: pd.Series,
    same_day: pd.DataFrame,
    position: dict,
    current_position_index: int,
    terminal_position_index: int,
) -> pd.DataFrame:
    cid = str(
        current_row[
            "canonical_security_id"
        ]
    )

    current_score = float(
        current_row[
            "persistent_score"
        ]
    )
    current_score_pct = float(
        current_row[
            "persistent_score_pct"
        ]
    )
    current_rank = float(
        current_row[
            "persistent_rank"
        ]
    )

    alternatives = same_day.loc[
        same_day[
            "canonical_security_id"
        ].astype(
            str
        ).ne(
            cid
        )
    ]

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
    top5_cutoff = (
        float(
            top5[
                "persistent_score"
            ].iloc[
                -1
            ]
        )
        if not top5.empty
        else current_score
    )

    current_close = float(
        current_row[
            "close"
        ]
    )

    peak_close = max(
        float(
            position[
                "running_peak_close"
            ]
        ),
        current_close,
    )
    position[
        "running_peak_close"
    ] = peak_close

    age = int(
        current_position_index
        - position[
            "entry_position_index"
        ]
    )

    remaining = int(
        terminal_position_index
        - current_position_index
    )

    effective_entry_price = float(
        position[
            "controller_entry_reference_price"
        ]
    )

    gross_return = (
        current_close
        / effective_entry_price
        - 1.0
    )
    peak_return = (
        peak_close
        / effective_entry_price
        - 1.0
    )
    drawdown_from_peak = (
        current_close
        / peak_close
        - 1.0
    )

    dynamic = {
        "holding_age_sessions": float(
            age
        ),
        "holding_age_fraction": float(
            age
            / max(
                1,
                REBALANCE_SESSIONS
                - 1,
            )
        ),
        "remaining_sessions": float(
            remaining
        ),
        "remaining_fraction": float(
            remaining
            / max(
                1,
                REBALANCE_SESSIONS
                - 1,
            )
        ),
        "entry_signal_score": float(
            position[
                "entry_signal_score"
            ]
        ),
        "entry_signal_score_pct": float(
            position[
                "entry_signal_score_pct"
            ]
        ),
        "entry_signal_rank": float(
            position[
                "entry_signal_rank"
            ]
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
                position[
                    "entry_signal_rank"
                ]
            )
        ),
        "score_change_from_signal": (
            current_score
            - float(
                position[
                    "entry_signal_score"
                ]
            )
        ),
        "score_pct_change_from_signal": (
            current_score_pct
            - float(
                position[
                    "entry_signal_score_pct"
                ]
            )
        ),
        "unrealized_gross_return": float(
            gross_return
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
                current_row[
                    feature
                ]
            )
            if pd.notna(
                current_row[
                    feature
                ]
            )
            else np.nan
        )
        for feature in (
            B4_FEATURES
        )
    }

    state = pd.DataFrame([
        {
            **features,
            **dynamic,
        }
    ])

    return state[
        CONTROLLER_FEATURES
    ]


def run_cycle_backtest(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    signal_dates: list[pd.Timestamp],
    exit_model,
    hold_model,
    threshold: float | None,
    name: str,
    initial_capital: float,
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> dict:
    panel = (
        panel.sort_values(
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

    day_groups, dates = (
        build_day_index(
            panel
        )
    )
    market_indices = sorted(
        day_groups
    )
    position_of_index = {
        market_index: position
        for position, market_index
        in enumerate(
            market_indices
        )
    }
    index_by_date = {
        pd.Timestamp(
            date
        ).normalize(): int(
            market_index
        )
        for market_index, date
        in dates.items()
    }

    signal_indices = [
        index_by_date[
            date
        ]
        for date in signal_dates
        if date in index_by_date
    ]
    signal_indices = sorted(
        signal_indices
    )
    signal_set = set(
        signal_indices
    )

    if not signal_indices:
        raise RuntimeError(
            "No validation signal dates "
            "matched the execution panel."
        )

    terminal_by_signal = {}
    next_entry_by_signal = {}

    for signal_index in signal_indices:
        pos = position_of_index[
            signal_index
        ]
        terminal_pos = (
            pos
            + REBALANCE_SESSIONS
        )
        entry_pos = (
            pos
            + 1
        )
        if (
            terminal_pos
            >= len(
                market_indices
            )
            or entry_pos
            >= len(
                market_indices
            )
        ):
            continue
        terminal_by_signal[
            signal_index
        ] = market_indices[
            terminal_pos
        ]
        next_entry_by_signal[
            signal_index
        ] = market_indices[
            entry_pos
        ]

    events_by_index = (
        build_events_by_market_index(
            events,
            dates,
        )
    )

    cash = float(
        initial_capital
    )
    holdings: dict[
        str,
        dict,
    ] = {}
    pending_sells: dict[
        str,
        str,
    ] = {}
    pending_buys: list[
        dict
    ] = []

    trades = []
    equity_rows = []
    controller_rows = []

    controller_exits = 0
    terminal_exits = 0
    applied_actions = 0
    skipped_unsafe = 0
    failed_entries = 0

    active_signal_index = None
    active_terminal_index = None

    for market_index in (
        market_indices
    ):
        day = panel.loc[
            day_groups[
                market_index
            ]
        ].copy()

        current_date = pd.Timestamp(
            dates[
                market_index
            ]
        ).normalize()

        day_lookup = day.set_index(
            "canonical_security_id",
            drop=False,
        )

        # Apply mapped mechanical quantity changes before the open.
        for (
            cid,
            multiplier,
        ) in events_by_index.get(
            market_index,
            [],
        ):
            position = holdings.get(
                cid
            )
            if position is None:
                continue

            pre = float(
                position[
                    "quantity"
                ]
            )
            post = (
                pre
                * float(
                    multiplier
                )
            )
            position[
                "quantity"
            ] = float(
                math.floor(
                    post
                    + 1e-12
                )
            )
            position[
                "controller_entry_reference_price"
            ] /= float(
                multiplier
            )
            position[
                "running_peak_close"
            ] /= float(
                multiplier
            )
            position[
                "corporate_action_count"
            ] += 1
            position[
                "corporate_action_fraction_discarded"
            ] += (
                post
                - position[
                    "quantity"
                ]
            )
            applied_actions += 1

        # Close orders execute before new-cycle buys.
        for cid, reason in list(
            pending_sells.items()
        ):
            if cid not in holdings:
                del pending_sells[
                    cid
                ]
                continue

            if cid not in day_lookup.index:
                continue

            row = day_lookup.loc[
                cid
            ]
            quoted_open = float(
                row[
                    "open"
                ]
            )
            if (
                not math.isfinite(
                    quoted_open
                )
                or quoted_open
                <= 0
            ):
                continue

            cash, trade = (
                close_position(
                    strategy=name,
                    cid=cid,
                    position=holdings[
                        cid
                    ],
                    quoted_price=quoted_open,
                    current_date=current_date,
                    market_index=(
                        market_index
                    ),
                    costs=costs,
                    cash=cash,
                    reason=reason,
                )
            )
            trades.append(
                trade
            )
            del holdings[
                cid
            ]
            del pending_sells[
                cid
            ]

        if pending_buys:
            still_pending = []

            for order in (
                pending_buys
            ):
                intended = int(
                    order[
                        "entry_market_index"
                    ]
                )

                if intended > market_index:
                    still_pending.append(
                        order
                    )
                    continue

                if intended < market_index:
                    continue

                cid = str(
                    order[
                        "canonical_security_id"
                    ]
                )

                if (
                    cid in holdings
                    or cid
                    not in day_lookup.index
                ):
                    continue

                if bool(
                    order[
                        "unsafe_target"
                    ]
                ):
                    skipped_unsafe += 1
                    continue

                row = day_lookup.loc[
                    cid
                ]
                quoted_open = float(
                    row[
                        "open"
                    ]
                )

                if (
                    not math.isfinite(
                        quoted_open
                    )
                    or quoted_open
                    <= 0
                ):
                    continue

                budget = min(
                    float(
                        order[
                            "slot_budget"
                        ]
                    ),
                    cash,
                )

                quantity = (
                    max_affordable_quantity(
                        budget,
                        quoted_open,
                        costs,
                    )
                )
                if quantity <= 0:
                    failed_entries += 1
                    continue

                execution = (
                    buy_execution(
                        quoted_open,
                        quantity,
                        costs,
                    )
                )
                cash -= float(
                    execution[
                        "cash_out"
                    ]
                )

                holdings[
                    cid
                ] = {
                    "symbol": str(
                        row[
                            "symbol"
                        ]
                    ),
                    "quantity": float(
                        quantity
                    ),
                    "initial_quantity": int(
                        quantity
                    ),
                    "entry_signal_date": (
                        order[
                            "signal_date"
                        ]
                    ),
                    "entry_date": (
                        current_date
                    ),
                    "entry_market_index": int(
                        market_index
                    ),
                    "entry_position_index": int(
                        position_of_index[
                            market_index
                        ]
                    ),
                    "entry_execution_price": float(
                        execution[
                            "execution_price"
                        ]
                    ),
                    "controller_entry_reference_price": float(
                        execution[
                            "execution_price"
                        ]
                    ),
                    "entry_trade_value": float(
                        execution[
                            "trade_value"
                        ]
                    ),
                    "entry_fees": float(
                        execution[
                            "fees"
                        ]
                    ),
                    "entry_cash_out": float(
                        execution[
                            "cash_out"
                        ]
                    ),
                    "entry_score": float(
                        order[
                            "score_pct"
                        ]
                    ),
                    "entry_signal_score": float(
                        order[
                            "score"
                        ]
                    ),
                    "entry_signal_score_pct": float(
                        order[
                            "score_pct"
                        ]
                    ),
                    "entry_signal_rank": float(
                        order[
                            "rank"
                        ]
                    ),
                    "running_peak_close": float(
                        row[
                            "close"
                        ]
                    ),
                    "last_close": float(
                        row[
                            "close"
                        ]
                    ),
                    "corporate_action_count": 0,
                    "corporate_action_fraction_discarded": 0.0,
                }

            pending_buys = (
                still_pending
            )

        # Mark positions.
        for cid, position in (
            holdings.items()
        ):
            if cid not in day_lookup.index:
                continue
            close = float(
                day_lookup.loc[
                    cid,
                    "close",
                ]
            )
            if (
                math.isfinite(
                    close
                )
                and close > 0
            ):
                position[
                    "last_close"
                ] = close

        invested = sum(
            float(
                position[
                    "quantity"
                ]
            )
            * float(
                position[
                    "last_close"
                ]
            )
            for position
            in holdings.values()
        )
        equity = (
            cash
            + invested
        )

        equity_rows.append({
            "date": (
                current_date
            ),
            "market_day_index": int(
                market_index
            ),
            "cash": float(
                cash
            ),
            "invested_market_value": float(
                invested
            ),
            "equity": float(
                equity
            ),
            "open_positions": int(
                len(
                    holdings
                )
            ),
        })

        # At a cycle boundary, force all surviving positions to exit next
        # open and schedule the new top-5 cohort for that same open.
        if (
            market_index
            in signal_set
            and market_index
            in terminal_by_signal
        ):
            if active_signal_index is not None:
                for cid in list(
                    holdings
                ):
                    pending_sells[
                        cid
                    ] = (
                        "cycle_terminal_exit"
                    )
                    terminal_exits += 1

            active_signal_index = (
                market_index
            )
            active_terminal_index = (
                terminal_by_signal[
                    market_index
                ]
            )

            ranked = (
                day.loc[
                    day[
                        "eligible_universe"
                    ].fillna(
                        False
                    )
                    & day[
                        "persistent_score"
                    ].notna()
                    & day[
                        "persistent_rank"
                    ].notna()
                    & day[
                        "persistent_score_pct"
                    ].notna()
                ]
                .sort_values(
                    "persistent_rank",
                    kind="stable",
                )
            )

            selected = ranked.head(
                TOP_K
            )
            target_slot = (
                equity
                / TOP_K
            )

            for row in selected.itertuples(
                index=False
            ):
                pending_buys.append({
                    "canonical_security_id": str(
                        row.canonical_security_id
                    ),
                    "signal_date": (
                        current_date
                    ),
                    "entry_market_index": (
                        next_entry_by_signal[
                            market_index
                        ]
                    ),
                    "slot_budget": float(
                        target_slot
                    ),
                    "score": float(
                        row.persistent_score
                    ),
                    "score_pct": float(
                        row.persistent_score_pct
                    ),
                    "rank": float(
                        row.persistent_rank
                    ),
                    "unsafe_target": bool(
                        row.unsafe_target_window_20d
                    ),
                })

            continue

        # Controller acts after close only inside an active cycle. The cycle
        # boundary itself is handled above as a forced terminal action.
        if (
            threshold is not None
            and holdings
            and active_terminal_index
            is not None
            and market_index
            != active_terminal_index
            and market_index
            < active_terminal_index
        ):
            same_day = (
                day.loc[
                    day[
                        "eligible_universe"
                    ].fillna(
                        False
                    )
                    & day[
                        "persistent_score"
                    ].notna()
                    & day[
                        "persistent_rank"
                    ].notna()
                    & day[
                        "persistent_score_pct"
                    ].notna()
                ]
                .sort_values(
                    "persistent_rank",
                    kind="stable",
                )
            )

            terminal_position_index = (
                position_of_index[
                    active_terminal_index
                ]
            )
            current_position_index = (
                position_of_index[
                    market_index
                ]
            )

            for cid in list(
                holdings
            ):
                if (
                    cid
                    in pending_sells
                    or cid
                    not in day_lookup.index
                ):
                    continue

                row = day_lookup.loc[
                    cid
                ]

                if (
                    not bool(
                        row[
                            "eligible_universe"
                        ]
                    )
                    or pd.isna(
                        row[
                            "persistent_score"
                        ]
                    )
                    or pd.isna(
                        row[
                            "persistent_rank"
                        ]
                    )
                    or pd.isna(
                        row[
                            "persistent_score_pct"
                        ]
                    )
                ):
                    continue

                state = (
                    controller_state_frame(
                        current_row=row,
                        same_day=same_day,
                        position=holdings[
                            cid
                        ],
                        current_position_index=(
                            current_position_index
                        ),
                        terminal_position_index=(
                            terminal_position_index
                        ),
                    )
                )

                (
                    q_exit,
                    q_hold,
                    advantage,
                ) = predict_advantage(
                    state.assign(
                        remaining_sessions=state[
                            "remaining_sessions"
                        ]
                    ),
                    exit_model=exit_model,
                    hold_model=hold_model,
                )

                value = float(
                    advantage[
                        0
                    ]
                )

                controller_rows.append({
                    "date": (
                        current_date
                    ),
                    "canonical_security_id": (
                        cid
                    ),
                    "symbol": str(
                        holdings[
                            cid
                        ][
                            "symbol"
                        ]
                    ),
                    "q_exit": float(
                        q_exit[
                            0
                        ]
                    ),
                    "q_hold": float(
                        q_hold[
                            0
                        ]
                    ),
                    "q_advantage": (
                        value
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "action": (
                        "EXIT"
                        if value
                        <= float(
                            threshold
                        )
                        else "HOLD"
                    ),
                })

                if value <= float(
                    threshold
                ):
                    pending_sells[
                        cid
                    ] = (
                        "controller_exit"
                    )
                    controller_exits += 1

    # Final close liquidation for any residual position.
    if market_indices:
        last_index = market_indices[
            -1
        ]
        final_date = pd.Timestamp(
            dates[
                last_index
            ]
        ).normalize()
        final_day = panel.loc[
            day_groups[
                last_index
            ]
        ]
        final_lookup = (
            final_day.set_index(
                "canonical_security_id",
                drop=False,
            )
        )

        for cid in list(
            holdings
        ):
            position = holdings[
                cid
            ]
            quoted = float(
                final_lookup.loc[
                    cid,
                    "close",
                ]
            ) if (
                cid
                in final_lookup.index
            ) else float(
                position[
                    "last_close"
                ]
            )

            cash, trade = (
                close_position(
                    strategy=name,
                    cid=cid,
                    position=position,
                    quoted_price=quoted,
                    current_date=final_date,
                    market_index=last_index,
                    costs=costs,
                    cash=cash,
                    reason=(
                        "final_liquidation"
                    ),
                )
            )
            trades.append(
                trade
            )
            del holdings[
                cid
            ]

        if equity_rows:
            equity_rows[
                -1
            ][
                "cash"
            ] = float(
                cash
            )
            equity_rows[
                -1
            ][
                "invested_market_value"
            ] = 0.0
            equity_rows[
                -1
            ][
                "equity"
            ] = float(
                cash
            )
            equity_rows[
                -1
            ][
                "open_positions"
            ] = 0

    equity_df = pd.DataFrame(
        equity_rows
    )
    trades_df = pd.DataFrame(
        trades
    )
    controller_df = pd.DataFrame(
        controller_rows
    )

    metrics = (
        compute_performance_metrics(
            equity_df,
            trades_df,
            initial_capital=(
                initial_capital
            ),
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
        )
    )

    metrics.update({
        "policy": name,
        "controller_threshold": (
            threshold
        ),
        "controller_exits": int(
            controller_exits
        ),
        "cycle_terminal_exits": int(
            terminal_exits
        ),
        "applied_corporate_actions": int(
            applied_actions
        ),
        "skipped_unsafe": int(
            skipped_unsafe
        ),
        "failed_entries": int(
            failed_entries
        ),
    })

    return {
        "metrics": metrics,
        "equity": equity_df,
        "trades": trades_df,
        "controller": (
            controller_df
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Matched 2023 portfolio evaluation for "
            "the Controller FQI-v2 HOLD/EXIT policy."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.065,
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
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()

    panel = load_panel(
        root,
        years=[
            VALIDATION_YEAR,
        ],
    )
    attach_scores(
        panel,
        root=root,
        stage="development",
        years=[
            VALIDATION_YEAR,
        ],
    )

    signal_dates = (
        signal_dates_from_teacher(
            root
        )
    )

    print(
        "Matched 2023 controller cycles: "
        f"{len(signal_dates)}"
    )

    (
        exit_model,
        hold_model,
        iteration_diag,
    ) = load_fqi_models(
        root
    )

    costs = CostProfile(
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

    events = load_mechanical_events(
        root
    )

    output_root = (
        root
        / "reports/ml/"
        "controller_cycle_portfolio_v3/"
        "development_2023"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard_rows = []
    equity_frames = []
    trade_frames = []
    controller_frames = []

    for number, policy in enumerate(
        POLICIES,
        start=1,
    ):
        print(
            f"\n[{number}/{len(POLICIES)}] "
            f"{policy['name']}"
        )

        result = run_cycle_backtest(
            panel,
            events,
            signal_dates=(
                signal_dates
            ),
            exit_model=(
                exit_model
            ),
            hold_model=(
                hold_model
            ),
            threshold=(
                policy[
                    "threshold"
                ]
            ),
            name=str(
                policy[
                    "name"
                ]
            ),
            initial_capital=float(
                args.capital
            ),
            costs=costs,
            annual_risk_free_rate=float(
                args.risk_free_rate
            ),
        )

        metrics = result[
            "metrics"
        ]
        leaderboard_rows.append(
            metrics
        )

        equity = result[
            "equity"
        ].copy()
        equity[
            "policy"
        ] = policy[
            "name"
        ]
        equity_frames.append(
            equity
        )

        trades = result[
            "trades"
        ].copy()
        if not trades.empty:
            trades[
                "policy"
            ] = policy[
                "name"
            ]
            trade_frames.append(
                trades
            )

        controller = result[
            "controller"
        ].copy()
        if not controller.empty:
            controller[
                "policy"
            ] = policy[
                "name"
            ]
            controller_frames.append(
                controller
            )

        print(
            "  CAGR="
            f"{float(metrics['cagr']):.2%} "
            "Sharpe="
            f"{float(metrics['sharpe']):.3f} "
            "MDD="
            f"{float(metrics['max_drawdown']):.2%} "
            "turn="
            f"{float(metrics['turnover_multiple']):.1f}x "
            "controller-exits="
            f"{int(metrics['controller_exits'])}"
        )

    leaderboard = pd.DataFrame(
        leaderboard_rows
    )

    base = (
        leaderboard.loc[
            leaderboard[
                "policy"
            ].eq(
                "base_forced20"
            )
        ]
        .iloc[
            0
        ]
    )

    leaderboard[
        "sharpe_delta_vs_base"
    ] = (
        leaderboard[
            "sharpe"
        ]
        - float(
            base[
                "sharpe"
            ]
        )
    )
    leaderboard[
        "cagr_delta_vs_base"
    ] = (
        leaderboard[
            "cagr"
        ]
        - float(
            base[
                "cagr"
            ]
        )
    )
    leaderboard[
        "mdd_delta_vs_base"
    ] = (
        leaderboard[
            "max_drawdown"
        ]
        - float(
            base[
                "max_drawdown"
            ]
        )
    )

    leaderboard = (
        leaderboard.sort_values(
            [
                "sharpe",
                "max_drawdown",
                "cagr",
            ],
            ascending=[
                False,
                False,
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
        output_root
        / "leaderboard.csv",
        index=False,
    )

    pd.concat(
        equity_frames,
        ignore_index=True,
    ).to_parquet(
        output_root
        / "equity_curves.parquet",
        index=False,
        compression="zstd",
    )

    if trade_frames:
        pd.concat(
            trade_frames,
            ignore_index=True,
        ).to_parquet(
            output_root
            / "trades.parquet",
            index=False,
            compression="zstd",
        )

    if controller_frames:
        pd.concat(
            controller_frames,
            ignore_index=True,
        ).to_parquet(
            output_root
            / "controller_decisions.parquet",
            index=False,
            compression="zstd",
        )

    iteration_diag.to_csv(
        output_root
        / "fqi_iteration_diagnostics.csv",
        index=False,
    )

    best = leaderboard.iloc[
        0
    ]

    summary = {
        "year": (
            VALIDATION_YEAR
        ),
        "environment": (
            "matched fresh top-5 20-session cycles; all surviving positions "
            "liquidate at the next-open cycle boundary, exactly matching the "
            "Controller-v1 teacher horizon rather than the persistent "
            "k5_r20_h5 production candidate"
        ),
        "selector": (
            "frozen B4_core_plus_F8"
        ),
        "controller_model": (
            MODEL_NAME
        ),
        "controller_train_years": [
            2021,
            2022,
        ],
        "controller_validation_year": (
            VALIDATION_YEAR
        ),
        "fqi_iterations": (
            FQI_ITERATIONS
        ),
        "policies": (
            POLICIES
        ),
        "cost_profile": asdict(
            costs
        ),
        "signal_dates": int(
            len(
                signal_dates
            )
        ),
        "selection_rule": (
            "highest actual daily portfolio Sharpe on the matched 2023 "
            "development environment, then shallower max drawdown, then "
            "higher CAGR"
        ),
        "best_policy": str(
            best[
                "policy"
            ]
        ),
        "best_metrics": (
            best.to_dict()
        ),
        "next_step": (
            "If the controller improves matched portfolio Sharpe/drawdown, "
            "build a persistent-position controller whose state transitions "
            "continue across rebalance boundaries before opening 2024-2026."
        ),
    }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=float,
        )
        + "\n"
    )

    print(
        "\n=== CONTROLLER CYCLE PORTFOLIO V3 COMPLETE ==="
    )
    print(
        "Best policy: "
        f"{summary['best_policy']}"
    )
    print(
        "Best Sharpe: "
        f"{float(best['sharpe']):.3f}"
    )
    print(
        "Base Sharpe: "
        f"{float(base['sharpe']):.3f}"
    )
    print(
        "Output:      "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
