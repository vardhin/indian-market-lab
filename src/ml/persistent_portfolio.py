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

from classical import (  # noqa: E402
    HORIZON,
    TARGET_COLUMN,
    TRAINING_FLAG,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    add_target_rank,
    build_model,
    fit_model,
    load_ml_panel,
    predict_mask,
)
from baselines import (  # noqa: E402
    CostProfile,
    build_day_index,
    build_events_by_market_index,
    buy_execution,
    load_mechanical_events,
    max_affordable_quantity,
    run_backtest,
    sell_execution,
)
from benchmark_report import (  # noqa: E402
    index_curve,
)
from quant_metrics import (  # noqa: E402
    captioned_metric_rows,
    compute_performance_metrics,
)


DEFAULT_TOP_K = [5, 10]
DEFAULT_HOLD_MULTIPLIERS = [1, 2, 3, 5]
DEFAULT_SCORE_GAPS = [0.0, 0.02, 0.05, 0.10]


def fit_validation_scores(
    df: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
) -> pd.Series:
    cutoff = int(
        df.loc[
            df["date"].le(TRAIN_END),
            "market_day_index",
        ]
        .dropna()
        .max()
    )

    train_mask = (
        df[TRAINING_FLAG]
        & df["target_rank_20d"].notna()
        & df["date"].between(
            TRAIN_START,
            TRAIN_END,
            inclusive="both",
        )
        & df["market_day_index"].le(
            cutoff - HORIZON
        )
    )

    score_mask = (
        df["eligible_universe"]
        & df["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
    )

    print(
        "Fitting validation base model on "
        f"{int(train_mask.sum()):,} rows..."
    )

    model = build_model(
        model_name,
        random_state=random_state,
    )
    fit_model(
        model,
        df,
        train_mask,
    )

    df["persistent_score"] = np.nan
    df.loc[
        score_mask,
        "persistent_score",
    ] = predict_mask(
        model,
        df,
        score_mask,
    )

    return score_mask


def load_test_scores(
    df: pd.DataFrame,
    *,
    root: Path,
) -> pd.Series:
    path = (
        root
        / "reports/ml/portfolio_diagnostics/"
        "walkforward_predictions.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Saved walk-forward predictions are missing. "
            "Run src/ml/portfolio_diagnostics.py first."
        )

    saved = pd.read_parquet(
        path,
        columns=[
            "date",
            "canonical_security_id",
            "walkforward_ml_score",
        ],
    )
    saved["date"] = pd.to_datetime(
        saved["date"],
        errors="coerce",
    ).dt.normalize()
    saved["canonical_security_id"] = (
        saved["canonical_security_id"]
        .astype("string")
        .str.strip()
    )

    keys = [
        "date",
        "canonical_security_id",
    ]
    if saved.duplicated(
        keys,
        keep=False,
    ).any():
        raise RuntimeError(
            "Saved walk-forward predictions contain duplicate keys."
        )

    mapping = (
        saved.set_index(keys)[
            "walkforward_ml_score"
        ]
    )
    key_index = pd.MultiIndex.from_frame(
        df[keys]
    )

    values = mapping.reindex(
        key_index
    ).to_numpy(
        dtype=float
    )

    mask = (
        df["eligible_universe"]
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
    )
    df.loc[
        mask,
        "persistent_score",
    ] = values[
        mask.to_numpy()
    ]

    coverage = float(
        df.loc[
            mask,
            "persistent_score",
        ]
        .notna()
        .mean()
    )
    if coverage < 0.98:
        raise RuntimeError(
            "Walk-forward score coverage is "
            f"only {coverage:.2%}."
        )

    return mask


def add_daily_ranks(
    df: pd.DataFrame,
) -> None:
    df["persistent_rank"] = np.nan
    df["persistent_score_pct"] = np.nan

    mask = (
        df["eligible_universe"]
        & df["persistent_score"].notna()
    )

    view = df.loc[
        mask,
        [
            "date",
            "persistent_score",
            "turnover_median_20d",
            "canonical_security_id",
        ],
    ].copy()

    # Stable deterministic rank for portfolio cutoffs.
    ordered = (
        view.sort_values(
            [
                "date",
                "persistent_score",
                "turnover_median_20d",
                "canonical_security_id",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
            kind="stable",
        )
    )
    ordered[
        "persistent_rank"
    ] = (
        ordered.groupby(
            "date",
            sort=False,
        ).cumcount()
        + 1
    )

    ordered[
        "persistent_score_pct"
    ] = (
        ordered.groupby(
            "date",
            sort=False,
        )["persistent_score"]
        .rank(
            method="average",
            pct=True,
        )
    )

    df.loc[
        ordered.index,
        "persistent_rank",
    ] = ordered[
        "persistent_rank"
    ].to_numpy(
        dtype=float
    )
    df.loc[
        ordered.index,
        "persistent_score_pct",
    ] = ordered[
        "persistent_score_pct"
    ].to_numpy(
        dtype=float
    )


def close_position(
    *,
    strategy: str,
    cid: str,
    position: dict,
    quoted_price: float,
    current_date: pd.Timestamp,
    market_index: int,
    costs: CostProfile,
    cash: float,
    reason: str,
) -> tuple[
    float,
    dict,
]:
    execution = sell_execution(
        quoted_price,
        position["quantity"],
        costs,
    )
    cash += execution["cash_in"]

    total_fees = (
        position["entry_fees"]
        + execution["fees"]
    )
    net_pnl = (
        execution["cash_in"]
        - position["entry_cash_out"]
    )

    trade = {
        "strategy": strategy,
        "symbol": position["symbol"],
        "canonical_security_id": cid,
        "signal_date": position[
            "entry_signal_date"
        ],
        "entry_date": position[
            "entry_date"
        ],
        "exit_date": current_date,
        "entry_market_index": position[
            "entry_market_index"
        ],
        "actual_exit_market_index": (
            market_index
        ),
        "holding_sessions": int(
            market_index
            - position[
                "entry_market_index"
            ]
        ),
        "initial_quantity": position[
            "initial_quantity"
        ],
        "exit_quantity": position[
            "quantity"
        ],
        "entry_price": position[
            "entry_execution_price"
        ],
        "exit_price": execution[
            "execution_price"
        ],
        "entry_trade_value": position[
            "entry_trade_value"
        ],
        "exit_trade_value": execution[
            "trade_value"
        ],
        "entry_fees": position[
            "entry_fees"
        ],
        "exit_fees": execution[
            "fees"
        ],
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "net_return": (
            net_pnl
            / position[
                "entry_cash_out"
            ]
            if position[
                "entry_cash_out"
            ] > 0
            else np.nan
        ),
        "entry_score": position[
            "entry_score"
        ],
        "exit_reason": reason,
        "corporate_action_count": position[
            "corporate_action_count"
        ],
        "corporate_action_fraction_discarded": (
            position[
                "corporate_action_fraction_discarded"
            ]
        ),
        "written_off": False,
    }

    return (
        cash,
        trade,
    )


def run_persistent_backtest(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    name: str,
    initial_capital: float,
    top_k: int,
    hold_rank: int,
    min_score_gap: float,
    rebalance_sessions: int,
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> dict:
    if hold_rank < top_k:
        raise ValueError(
            "hold_rank must be >= top_k"
        )
    if min_score_gap < 0:
        raise ValueError(
            "min_score_gap must be >= 0"
        )

    panel = (
        panel.sort_values(
            [
                "market_day_index",
                "canonical_security_id",
            ]
        )
        .reset_index(drop=True)
    )

    day_groups, dates = build_day_index(
        panel
    )
    market_indices = sorted(
        day_groups
    )
    if not market_indices:
        raise RuntimeError(
            "Persistent backtest has no market days."
        )

    events_by_index = (
        build_events_by_market_index(
            events,
            dates,
        )
    )

    first_index = market_indices[0]
    last_index = market_indices[-1]
    signal_indices = set(
        market_indices[
            ::rebalance_sessions
        ]
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

    trades: list[
        dict
    ] = []
    equity_rows: list[
        dict
    ] = []
    rebalance_rows: list[
        dict
    ] = []

    applied_actions = 0
    unsafe_censors = 0
    retained_decisions = 0
    replacements = 0
    skipped_gap_replacements = 0
    failed_entries = 0

    unsafe_col = (
        "unsafe_target_window_20d"
    )

    for market_index in (
        market_indices
    ):
        day = panel.loc[
            day_groups[
                market_index
            ]
        ].copy()
        day_lookup = day.set_index(
            "canonical_security_id",
            drop=False,
        )
        current_date = dates[
            market_index
        ]

        # Mechanical split/bonus quantity adjustments apply before trading on
        # the mapped ex-date, matching the frozen simulator.
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
                position["quantity"]
            )
            post = (
                pre
                * float(
                    multiplier
                )
            )
            position["quantity"] = float(
                math.floor(
                    post
                    + 1e-12
                )
            )
            position[
                "corporate_action_fraction_discarded"
            ] += (
                post
                - position[
                    "quantity"
                ]
            )
            position[
                "corporate_action_count"
            ] += 1
            applied_actions += 1

        # Orders are decided after the prior signal close and executed at the
        # next market open. Sells execute before buys so released cash can fund
        # replacements.
        if pending_sells:
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
                    row["open"]
                )
                if (
                    not math.isfinite(
                        quoted_open
                    )
                    or quoted_open <= 0
                ):
                    continue

                cash, trade = (
                    close_position(
                        strategy=name,
                        cid=cid,
                        position=(
                            holdings[
                                cid
                            ]
                        ),
                        quoted_price=(
                            quoted_open
                        ),
                        current_date=(
                            current_date
                        ),
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
            still_pending: list[
                dict
            ] = []

            for order in pending_buys:
                if order[
                    "entry_market_index"
                ] > market_index:
                    still_pending.append(
                        order
                    )
                    continue

                cid = order[
                    "canonical_security_id"
                ]

                if cid in holdings:
                    continue

                if len(holdings) >= top_k:
                    still_pending.append(
                        order
                    )
                    continue

                if cid not in day_lookup.index:
                    still_pending.append(
                        order
                    )
                    continue

                if order[
                    "unsafe_target"
                ]:
                    unsafe_censors += 1
                    continue

                row = day_lookup.loc[
                    cid
                ]
                quoted_open = float(
                    row["open"]
                )
                if (
                    not math.isfinite(
                        quoted_open
                    )
                    or quoted_open <= 0
                ):
                    still_pending.append(
                        order
                    )
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

                execution = buy_execution(
                    quoted_open,
                    quantity,
                    costs,
                )
                cash -= execution[
                    "cash_out"
                ]

                holdings[cid] = {
                    "symbol": str(
                        row["symbol"]
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
                    "entry_execution_price": float(
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
                    "last_close": float(
                        row["close"]
                    ),
                    "corporate_action_count": 0,
                    "corporate_action_fraction_discarded": 0.0,
                }

            pending_buys = (
                still_pending
            )

        # Mark holdings with the latest usable close.
        for cid, position in (
            holdings.items()
        ):
            if cid not in day_lookup.index:
                continue
            row = day_lookup.loc[
                cid
            ]
            close = float(
                row["close"]
            )
            if (
                math.isfinite(close)
                and close > 0
            ):
                position[
                    "last_close"
                ] = close

        invested = sum(
            position["quantity"]
            * position["last_close"]
            for position
            in holdings.values()
        )
        equity = (
            cash
            + invested
        )

        equity_rows.append({
            "date": current_date,
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
                len(holdings)
            ),
        })

        if (
            market_index
            not in signal_indices
            or market_index
            >= last_index
        ):
            continue

        ranked = (
            day.loc[
                day[
                    "eligible_universe"
                ].fillna(False)
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
                ascending=True,
            )
            .copy()
        )

        if ranked.empty:
            continue

        rank_lookup = (
            ranked.set_index(
                "canonical_security_id"
            )
        )

        kept: list[str] = []
        mandatory_sell: list[
            str
        ] = []
        censored_slots = 0
        replacement_pool: list[
            str
        ] = []

        # Existing positions that remain inside the hold band are preserved
        # without a sell/rebuy round trip. Ineligible/no-score/unsafe positions
        # are forced out for data-quality or universe reasons.
        for cid in list(
            holdings
        ):
            if cid not in rank_lookup.index:
                mandatory_sell.append(
                    cid
                )
                continue

            row = rank_lookup.loc[
                cid
            ]
            if bool(
                row[
                    unsafe_col
                ]
            ):
                mandatory_sell.append(
                    cid
                )
                unsafe_censors += 1
                censored_slots += 1
                continue

            rank = int(
                row[
                    "persistent_rank"
                ]
            )
            if rank <= hold_rank:
                kept.append(
                    cid
                )
                retained_decisions += 1
            else:
                replacement_pool.append(
                    cid
                )

        for cid in mandatory_sell:
            pending_sells[
                cid
            ] = (
                "forced_universe_or_"
                "data_quality_exit"
            )

        # Candidate buys come only from the fresh buy band. Incumbents already
        # held are excluded. Replacement of a weak incumbent additionally
        # requires the candidate score percentile to exceed the incumbent by
        # min_score_gap.
        buy_band = ranked.loc[
            ranked[
                "persistent_rank"
            ].le(top_k)
        ]

        candidates = [
            str(cid)
            for cid in buy_band[
                "canonical_security_id"
            ].tolist()
            if str(cid)
            not in holdings
        ]

        replacement_pool = sorted(
            replacement_pool,
            key=lambda cid: float(
                rank_lookup.loc[
                    cid,
                    "persistent_score_pct",
                ]
            ),
        )

        accepted_replacements: list[
            tuple[
                str,
                str,
            ]
        ] = []

        for incumbent in (
            replacement_pool
        ):
            if not candidates:
                kept.append(
                    incumbent
                )
                skipped_gap_replacements += 1
                continue

            candidate = (
                candidates[0]
            )
            incumbent_score = float(
                rank_lookup.loc[
                    incumbent,
                    "persistent_score_pct",
                ]
            )
            candidate_score = float(
                rank_lookup.loc[
                    candidate,
                    "persistent_score_pct",
                ]
            )
            gap = (
                candidate_score
                - incumbent_score
            )

            if gap + 1e-12 >= (
                min_score_gap
            ):
                pending_sells[
                    incumbent
                ] = (
                    "rank_buffer_replacement"
                )
                accepted_replacements.append(
                    (
                        incumbent,
                        candidate,
                    )
                )
                candidates.pop(0)
                replacements += 1
            else:
                kept.append(
                    incumbent
                )
                skipped_gap_replacements += 1

        # Initial deployment and vacancies created by mandatory exits are
        # filled from the remaining top-K names. We do not resize retained
        # holdings; this is deliberate turnover avoidance.
        planned_holding_count = (
            len(holdings)
            - len(
                pending_sells
            )
        )
        open_slots = max(
            0,
            top_k
            - planned_holding_count
            - len(
                accepted_replacements
            )
            - censored_slots,
        )

        selected_buys = [
            candidate
            for _, candidate
            in accepted_replacements
        ]
        for candidate in candidates:
            if len(
                selected_buys
            ) >= (
                len(
                    accepted_replacements
                )
                + open_slots
            ):
                break
            selected_buys.append(
                candidate
            )

        target_slot = (
            equity / top_k
        )

        for cid in (
            selected_buys
        ):
            row = rank_lookup.loc[
                cid
            ]
            pending_buys.append({
                "canonical_security_id": (
                    cid
                ),
                "signal_date": (
                    current_date
                ),
                "entry_market_index": (
                    market_index + 1
                ),
                "slot_budget": float(
                    target_slot
                ),
                "score_pct": float(
                    row[
                        "persistent_score_pct"
                    ]
                ),
                "unsafe_target": bool(
                    row[
                        unsafe_col
                    ]
                ),
            })

        rebalance_rows.append({
            "date": current_date,
            "market_day_index": int(
                market_index
            ),
            "holdings_before": int(
                len(holdings)
            ),
            "kept": int(
                len(kept)
            ),
            "mandatory_sells": int(
                len(
                    mandatory_sell
                )
            ),
            "censored_cash_slots": int(
                censored_slots
            ),
            "rank_replacements": int(
                len(
                    accepted_replacements
                )
            ),
            "planned_buys": int(
                len(
                    selected_buys
                )
            ),
            "cash": float(
                cash
            ),
            "equity": float(
                equity
            ),
        })

    # Final liquidation at the final observable close so terminal equity is
    # after exit friction rather than an unrealized mark.
    final_date = dates[
        last_index
    ]
    final_day = panel.loc[
        day_groups[
            last_index
        ]
    ].copy()
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
        if cid in final_lookup.index:
            row = final_lookup.loc[
                cid
            ]
            quoted = float(
                row["close"]
            )
        else:
            quoted = float(
                position[
                    "last_close"
                ]
            )

        if (
            math.isfinite(quoted)
            and quoted > 0
        ):
            cash, trade = (
                close_position(
                    strategy=name,
                    cid=cid,
                    position=(
                        position
                    ),
                    quoted_price=(
                        quoted
                    ),
                    current_date=(
                        final_date
                    ),
                    market_index=(
                        last_index
                    ),
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
        else:
            trades.append({
                "strategy": name,
                "symbol": position[
                    "symbol"
                ],
                "canonical_security_id": (
                    cid
                ),
                "signal_date": position[
                    "entry_signal_date"
                ],
                "entry_date": position[
                    "entry_date"
                ],
                "exit_date": final_date,
                "entry_market_index": (
                    position[
                        "entry_market_index"
                    ]
                ),
                "actual_exit_market_index": (
                    last_index
                ),
                "holding_sessions": int(
                    last_index
                    - position[
                        "entry_market_index"
                    ]
                ),
                "initial_quantity": (
                    position[
                        "initial_quantity"
                    ]
                ),
                "exit_quantity": 0.0,
                "entry_price": position[
                    "entry_execution_price"
                ],
                "exit_price": 0.0,
                "entry_trade_value": (
                    position[
                        "entry_trade_value"
                    ]
                ),
                "exit_trade_value": 0.0,
                "entry_fees": position[
                    "entry_fees"
                ],
                "exit_fees": 0.0,
                "total_fees": position[
                    "entry_fees"
                ],
                "net_pnl": -position[
                    "entry_cash_out"
                ],
                "net_return": -1.0,
                "entry_score": position[
                    "entry_score"
                ],
                "exit_reason": (
                    "final_writeoff"
                ),
                "corporate_action_count": (
                    position[
                        "corporate_action_count"
                    ]
                ),
                "corporate_action_fraction_discarded": (
                    position[
                        "corporate_action_fraction_discarded"
                    ]
                ),
                "written_off": True,
            })

        del holdings[
            cid
        ]

    # Replace the final mark-to-market row with post-liquidation cash.
    if equity_rows:
        equity_rows[-1][
            "cash"
        ] = float(
            cash
        )
        equity_rows[-1][
            "invested_market_value"
        ] = 0.0
        equity_rows[-1][
            "equity"
        ] = float(
            cash
        )
        equity_rows[-1][
            "open_positions"
        ] = 0

    trades_df = pd.DataFrame(
        trades
    )
    equity_df = pd.DataFrame(
        equity_rows
    )
    rebalances_df = pd.DataFrame(
        rebalance_rows
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
        "top_k": int(
            top_k
        ),
        "hold_rank": int(
            hold_rank
        ),
        "min_score_gap": float(
            min_score_gap
        ),
        "rebalance_sessions": int(
            rebalance_sessions
        ),
        "retained_decisions": int(
            retained_decisions
        ),
        "rank_replacements": int(
            replacements
        ),
        "skipped_gap_replacements": int(
            skipped_gap_replacements
        ),
        "unsafe_censors": int(
            unsafe_censors
        ),
        "failed_entries": int(
            failed_entries
        ),
        "applied_corporate_actions": int(
            applied_actions
        ),
    })

    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity": equity_df,
        "rebalances": rebalances_df,
    }


def validation_search(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    top_k_values: list[int],
    hold_multipliers: list[int],
    score_gaps: list[float],
    capital: float,
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> tuple[
    pd.DataFrame,
    dict[int, dict],
]:
    panel = (
        df.loc[
            df["date"].between(
                pd.Timestamp(
                    "2021-01-01"
                ),
                VALIDATION_END,
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows: list[
        dict
    ] = []

    print(
        "\n=== 2021 PERSISTENCE SEARCH ==="
    )

    for top_k in top_k_values:
        for multiplier in (
            hold_multipliers
        ):
            hold_rank = int(
                top_k
                * multiplier
            )
            for gap in score_gaps:
                result = (
                    run_persistent_backtest(
                        panel,
                        events,
                        name=(
                            "validation_persistent"
                        ),
                        initial_capital=(
                            capital
                        ),
                        top_k=top_k,
                        hold_rank=(
                            hold_rank
                        ),
                        min_score_gap=(
                            float(gap)
                        ),
                        rebalance_sessions=(
                            HORIZON
                        ),
                        costs=costs,
                        annual_risk_free_rate=(
                            annual_risk_free_rate
                        ),
                    )
                )
                row = {
                    **result["metrics"],
                }
                rows.append(
                    row
                )
                print(
                    f"k={top_k:2d} "
                    f"hold<={hold_rank:2d} "
                    f"gap={gap:.2f} "
                    f"CAGR={row['cagr']:.2%} "
                    f"Sharpe={row['sharpe']:.3f} "
                    f"turn={row['turnover_multiple']:.1f}x "
                    f"fees=₹{row['total_fees']:,.0f}"
                )

    search = pd.DataFrame(
        rows
    )

    selected: dict[
        int,
        dict,
    ] = {}

    for top_k in top_k_values:
        subset = (
            search.loc[
                search[
                    "top_k"
                ].eq(top_k)
            ]
            .sort_values(
                [
                    "sharpe",
                    "cagr",
                    "turnover_multiple",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
        )
        best = subset.iloc[0]
        selected[
            int(top_k)
        ] = {
            "hold_rank": int(
                best[
                    "hold_rank"
                ]
            ),
            "min_score_gap": float(
                best[
                    "min_score_gap"
                ]
            ),
        }

        print(
            "Selected for "
            f"k={top_k}: "
            f"hold<={selected[top_k]['hold_rank']}, "
            f"gap={selected[top_k]['min_score_gap']:.2f}"
        )

    return (
        search,
        selected,
    )


def index_benchmark_rows(
    root: Path,
    *,
    allowed_dates: set[pd.Timestamp],
    capital: float,
    annual_risk_free_rate: float,
) -> pd.DataFrame:
    path = (
        root
        / "data/processed/index_benchmarks/"
        "nse_price_indices.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Index benchmark data is missing."
        )

    data = pd.read_parquet(
        path
    )
    rows: list[
        dict
    ] = []

    for name in (
        "NIFTY 50",
        "NIFTY 500",
    ):
        source = data.loc[
            data[
                "requested_index"
            ].eq(name)
        ].copy()

        curve = index_curve(
            source,
            initial_capital=capital,
            allowed_dates=(
                allowed_dates
            ),
        )

        metrics = (
            compute_performance_metrics(
                curve[
                    [
                        "date",
                        "equity",
                    ]
                ],
                pd.DataFrame(),
                initial_capital=capital,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
            )
        )
        rows.append({
            "name": name,
            "kind": "price_index",
            "top_k": None,
            "hold_rank": None,
            "min_score_gap": None,
            **metrics,
        })

    return pd.DataFrame(
        rows
    )


def run_test_suite(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    root: Path,
    selected: dict[int, dict],
    top_k_values: list[int],
    capital: float,
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    panel = (
        df.loc[
            df["date"].between(
                TEST_START,
                TEST_END,
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    comparison_rows: list[
        dict
    ] = []
    detail_rows: list[
        dict
    ] = []

    for top_k in top_k_values:
        configs = [
            (
                "buffer_only",
                top_k,
                0.0,
            ),
            (
                "selected_persistence",
                selected[
                    int(top_k)
                ][
                    "hold_rank"
                ],
                selected[
                    int(top_k)
                ][
                    "min_score_gap"
                ],
            ),
        ]

        seen: set[
            tuple[int, float]
        ] = set()

        for label, hold_rank, gap in configs:
            key = (
                int(hold_rank),
                float(gap),
            )
            if key in seen:
                continue
            seen.add(key)

            result = (
                run_persistent_backtest(
                    panel,
                    events,
                    name=label,
                    initial_capital=capital,
                    top_k=top_k,
                    hold_rank=(
                        int(
                            hold_rank
                        )
                    ),
                    min_score_gap=(
                        float(gap)
                    ),
                    rebalance_sessions=(
                        HORIZON
                    ),
                    costs=costs,
                    annual_risk_free_rate=(
                        annual_risk_free_rate
                    ),
                )
            )

            comparison_rows.append({
                "name": (
                    f"{label}_h20_k{top_k}_"
                    f"hold{int(hold_rank)}_"
                    f"gap{float(gap):.2f}"
                ),
                "kind": (
                    "persistent_ml"
                ),
                **result["metrics"],
            })

            details = (
                result[
                    "rebalances"
                ].copy()
            )
            if not details.empty:
                details[
                    "name"
                ] = (
                    f"{label}_k{top_k}"
                )
                detail_rows.append(
                    details
                )

        # Frozen cohort baseline using the exact same test scores.
        cohort = panel.copy()
        cohort[
            "walkforward_score"
        ] = cohort[
            "persistent_score"
        ]
        result = run_backtest(
            cohort,
            events,
            strategy="walkforward_ml_cohort",
            score_column=(
                "walkforward_score"
            ),
            score_direction=1.0,
            initial_capital=capital,
            top_k=top_k,
            holding_sessions=(
                HORIZON
            ),
            costs=costs,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
        )
        comparison_rows.append({
            "name": (
                f"cohort_ml_h20_k{top_k}"
            ),
            "kind": (
                "cohort_ml"
            ),
            **result["metrics"],
        })

        momentum = run_backtest(
            panel,
            events,
            strategy=(
                "momentum_60d"
            ),
            initial_capital=capital,
            top_k=top_k,
            holding_sessions=(
                HORIZON
            ),
            costs=costs,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
        )
        comparison_rows.append({
            "name": (
                f"momentum_60d_h20_k{top_k}"
            ),
            "kind": (
                "deterministic"
            ),
            **momentum["metrics"],
        })

    indices = index_benchmark_rows(
        root,
        allowed_dates=set(
            panel["date"].dropna()
        ),
        capital=capital,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
    )
    comparison_rows.extend(
        indices.to_dict(
            orient="records"
        )
    )

    comparison = (
        pd.DataFrame(
            comparison_rows
        )
        .sort_values(
            [
                "cagr",
                "sharpe",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    details = (
        pd.concat(
            detail_rows,
            ignore_index=True,
        )
        if detail_rows
        else pd.DataFrame()
    )

    return (
        comparison,
        details,
    )


def self_test() -> None:
    dates = pd.bdate_range(
        "2026-01-01",
        periods=65,
    )

    rows: list[
        dict
    ] = []
    for i, day in enumerate(
        dates
    ):
        for j in range(6):
            # S0/S1 remain strongest, while S2/S3 swap slightly. A hold buffer
            # should preserve incumbents and reduce needless replacements.
            score = (
                1.0
                - j * 0.10
                + (
                    0.03
                    if (
                        i >= 20
                        and j == 3
                    )
                    else 0.0
                )
            )
            rows.append({
                "date": day,
                "market_day_index": i,
                "canonical_security_id": (
                    f"S{j}"
                ),
                "symbol": (
                    f"S{j}"
                ),
                "eligible_universe": True,
                "open": (
                    100.0 + i
                ),
                "close": (
                    100.5 + i
                ),
                "turnover_median_20d": (
                    10_000_000.0
                ),
                "return_60d": (
                    0.10
                    + score / 100
                ),
                "persistent_score": score,
                "persistent_rank": (
                    j + 1
                ),
                "persistent_score_pct": (
                    (6 - j) / 6
                ),
                "unsafe_target_window_20d": False,
            })

    panel = pd.DataFrame(
        rows
    )
    events = pd.DataFrame(
        columns=[
            "canonical_security_id",
            "ex_date",
            "share_multiplier",
        ]
    )

    result = run_persistent_backtest(
        panel,
        events,
        name="self_test",
        initial_capital=10_000.0,
        top_k=2,
        hold_rank=4,
        min_score_gap=0.05,
        rebalance_sessions=20,
        costs=CostProfile(
            stt_buy_rate=0.0,
            stt_sell_rate=0.0,
            stamp_buy_rate=0.0,
            sebi_rate=0.0,
            exchange_rate=0.0,
            gst_rate=0.0,
            brokerage_per_order=0.0,
            dp_charge_per_sell=0.0,
            slippage_bps=0.0,
        ),
        annual_risk_free_rate=0.0,
    )

    assert not result[
        "equity"
    ].empty
    assert result[
        "metrics"
    ][
        "ending_equity"
    ] > 0
    assert (
        result["metrics"][
            "retained_decisions"
        ]
        > 0
    )

    print(
        "Persistent-portfolio "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Turnover-aware persistent ML portfolio "
            "with buy/hold rank bands and a score-"
            "improvement replacement hurdle."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--model",
        choices=[
            "ridge",
            "hist_gb",
            "random_forest",
        ],
        default="hist_gb",
    )
    ap.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=(
            DEFAULT_TOP_K
        ),
    )
    ap.add_argument(
        "--hold-multipliers",
        nargs="+",
        type=int,
        default=(
            DEFAULT_HOLD_MULTIPLIERS
        ),
    )
    ap.add_argument(
        "--score-gaps",
        nargs="+",
        type=float,
        default=(
            DEFAULT_SCORE_GAPS
        ),
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
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if any(
        k < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k values must be >= 1"
        )
    if any(
        m < 1
        for m
        in args.hold_multipliers
    ):
        raise SystemExit(
            "--hold-multipliers must be >= 1"
        )
    if any(
        gap < 0
        for gap in args.score_gaps
    ):
        raise SystemExit(
            "--score-gaps must be >= 0"
        )
    if (
        args.risk_free_rate
        <= -1.0
    ):
        raise SystemExit(
            "--risk-free-rate must be > -1"
        )

    root = Path(
        args.root
    ).resolve()

    df = load_ml_panel(
        root
        / "data/processed/research_panel"
    )
    df = add_target_rank(
        df
    )

    fit_validation_scores(
        df,
        model_name=args.model,
        random_state=(
            args.random_state
        ),
    )
    load_test_scores(
        df,
        root=root,
    )
    add_daily_ranks(
        df
    )

    events = load_mechanical_events(
        root
    )
    costs = CostProfile(
        brokerage_per_order=(
            args.brokerage_per_order
        ),
        dp_charge_per_sell=(
            args.dp_charge_per_sell
        ),
        slippage_bps=(
            args.slippage_bps
        ),
    )

    (
        search,
        selected,
    ) = validation_search(
        df,
        events,
        top_k_values=[
            int(x)
            for x in args.top_k
        ],
        hold_multipliers=[
            int(x)
            for x
            in args.hold_multipliers
        ],
        score_gaps=[
            float(x)
            for x
            in args.score_gaps
        ],
        capital=args.capital,
        costs=costs,
        annual_risk_free_rate=(
            args.risk_free_rate
        ),
    )

    comparison, rebalances = (
        run_test_suite(
            df,
            events,
            root=root,
            selected=selected,
            top_k_values=[
                int(x)
                for x
                in args.top_k
            ],
            capital=args.capital,
            costs=costs,
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
        )
    )

    output_root = (
        root
        / "reports/ml/"
        "persistent_portfolio"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    search.to_csv(
        output_root
        / "validation_2021_persistence_search.csv",
        index=False,
    )
    comparison.to_csv(
        output_root
        / "walkforward_persistence_comparison.csv",
        index=False,
    )
    rebalances.to_csv(
        output_root
        / "walkforward_rebalance_log.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    caption_rows: list[
        pd.DataFrame
    ] = []
    for row in comparison.to_dict(
        orient="records"
    ):
        name = str(
            row.get(
                "name",
                ""
            )
        )
        captioned = (
            captioned_metric_rows(
                row
            )
        )
        captioned[
            "name"
        ] = name
        caption_rows.append(
            captioned
        )

    pd.concat(
        caption_rows,
        ignore_index=True,
    ).to_csv(
        output_root
        / "walkforward_persistence_captioned.csv",
        index=False,
    )

    summary = {
        "base_model": args.model,
        "prediction_horizon_sessions": (
            HORIZON
        ),
        "rebalance_sessions": (
            HORIZON
        ),
        "validation_selection": {
            "period": "2021",
            "selection_metric": (
                "net Sharpe per fixed top-k; "
                "CAGR then lower turnover tie-break"
            ),
            "selected_by_top_k": (
                selected
            ),
        },
        "hysteresis_policy": {
            "buy_band": (
                "fresh purchases only from "
                "rank <= top_k"
            ),
            "hold_band": (
                "incumbents retained while "
                "rank <= hold_rank"
            ),
            "replacement_hurdle": (
                "candidate score percentile minus "
                "incumbent score percentile must "
                "be >= min_score_gap"
            ),
            "position_resizing": (
                "retained positions are not resized"
            ),
            "execution": (
                "decide after signal close; "
                "sell then buy next market open"
            ),
        },
        "capital": float(
            args.capital
        ),
        "risk_free_rate_annual": float(
            args.risk_free_rate
        ),
        "cost_profile": asdict(
            costs
        ),
        "outputs": {
            "validation_search": (
                "validation_2021_persistence_search.csv"
            ),
            "comparison": (
                "walkforward_persistence_comparison.csv"
            ),
            "rebalance_log": (
                "walkforward_rebalance_log.csv"
            ),
            "captioned": (
                "walkforward_persistence_captioned.csv"
            ),
        },
    }

    (
        output_root
        / "persistent_portfolio_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    preview = comparison[
        [
            col
            for col in (
                "name",
                "kind",
                "top_k",
                "hold_rank",
                "min_score_gap",
                "ending_equity",
                "cagr",
                "max_drawdown",
                "sharpe",
                "sortino",
                "profit_factor",
                "turnover_multiple",
                "trades",
                "total_fees",
            )
            if col
            in comparison.columns
        ]
    ].copy()

    for col in (
        "cagr",
        "max_drawdown",
    ):
        if col in preview.columns:
            preview[col] = (
                pd.to_numeric(
                    preview[col],
                    errors="coerce",
                )
                * 100.0
            )

    print(
        "\n=== WALK-FORWARD "
        "PERSISTENCE COMPARISON ==="
    )
    print(
        preview.to_string(
            index=False,
            formatters={
                "ending_equity": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:,.2f}"
                    )
                ),
                "cagr": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.2f}%"
                    )
                ),
                "max_drawdown": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.2f}%"
                    )
                ),
                "sharpe": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "sortino": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "profit_factor": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "turnover_multiple": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.1f}x"
                    )
                ),
                "total_fees": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:,.2f}"
                    )
                ),
            },
        )
    )

    print(
        "\n=== PERSISTENT PORTFOLIO "
        "EXPERIMENT COMPLETE ==="
    )
    print(
        f"Outputs: {output_root}"
    )


if __name__ == "__main__":
    main()
