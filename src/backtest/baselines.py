from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


STRATEGIES = {
    "momentum_5d": {
        "column": "return_5d",
        "direction": 1.0,
    },
    "momentum_20d": {
        "column": "return_20d",
        "direction": 1.0,
    },
    "momentum_60d": {
        "column": "return_60d",
        "direction": 1.0,
    },
    "mean_reversion_5d": {
        "column": "return_5d",
        "direction": -1.0,
    },
    "risk_adjusted_momentum_20d": {
        "column": "risk_adjusted_momentum_20d",
        "direction": 1.0,
    },
}


@dataclass(frozen=True)
class CostProfile:
    """
    Generic Indian delivery-equity execution-cost profile.

    Defaults use the current 2026 statutory/exchange rates and a configurable
    flat brokerage assumption. They are intentionally applied CONSTANTLY over
    historical data: this answers "does the historical signal survive today's
    frictions?" rather than pretending 2026 levies existed unchanged since
    2010.

    Rates are decimals applied to executed trade value.
    """

    stt_buy_rate: float = 0.001
    stt_sell_rate: float = 0.001
    stamp_buy_rate: float = 0.00015
    sebi_rate: float = 0.000001
    exchange_rate: float = 307.0 / 10_000_000.0
    gst_rate: float = 0.18
    brokerage_per_order: float = 15.0
    dp_charge_per_sell: float = 0.0
    slippage_bps: float = 5.0


PANEL_BASE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "close",
    "turnover_median_20d",
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
]


def required_panel_columns(
    holding_sessions: int,
) -> list[str]:
    return PANEL_BASE_COLUMNS + [
        f"unsafe_target_window_{holding_sessions}d",
    ]


def load_panel(
    panel_root: Path,
    *,
    holding_sessions: int,
) -> pd.DataFrame:
    panel_root = Path(panel_root)
    files = sorted(
        panel_root.glob("date=*/data.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No research-panel partitions under {panel_root}"
        )

    requested = required_panel_columns(
        holding_sessions
    )
    schema = pq.read_schema(files[0])
    available = set(schema.names)

    missing = [
        c
        for c in requested
        if c not in available
    ]

    if missing:
        raise RuntimeError(
            "Research panel is missing columns required by the backtester: "
            f"{missing}. Rebuild the panel with the current pipeline."
        )

    frames: list[pd.DataFrame] = []

    print(
        f"Loading {len(files):,} research-panel partitions..."
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        frames.append(
            pd.read_parquet(
                path,
                columns=requested,
            )
        )

        if (
            index % 500 == 0
            or index == len(files)
        ):
            print(
                f"  loaded {index:,}/{len(files):,}",
                flush=True,
            )

    panel = pd.concat(
        frames,
        ignore_index=True,
    )

    panel["date"] = pd.to_datetime(
        panel["date"],
        errors="coerce",
    ).dt.normalize()

    panel["market_day_index"] = pd.to_numeric(
        panel["market_day_index"],
        errors="coerce",
    ).astype("Int64")

    for col in (
        "open",
        "close",
        "turnover_median_20d",
        "return_5d",
        "return_20d",
        "return_60d",
        "volatility_20d",
    ):
        panel[col] = pd.to_numeric(
            panel[col],
            errors="coerce",
        )

    panel["canonical_security_id"] = (
        panel["canonical_security_id"]
        .astype("string")
        .str.strip()
    )
    panel["symbol"] = (
        panel["symbol"]
        .astype("string")
        .str.strip()
    )

    duplicate = panel.duplicated(
        [
            "market_day_index",
            "canonical_security_id",
        ],
        keep=False,
    )

    if duplicate.any():
        sample = panel.loc[
            duplicate,
            [
                "date",
                "symbol",
                "canonical_security_id",
            ],
        ].head(20)

        raise RuntimeError(
            "Duplicate security rows in research panel. Sample:\n"
            + sample.to_string(index=False)
        )

    panel["risk_adjusted_momentum_20d"] = (
        panel["return_20d"]
        / panel["volatility_20d"].where(
            panel["volatility_20d"] > 0
        )
    )

    return (
        panel.sort_values(
            [
                "market_day_index",
                "canonical_security_id",
            ]
        )
        .reset_index(drop=True)
    )


def load_mechanical_events(
    root: Path,
) -> pd.DataFrame:
    path = (
        Path(root)
        / "data/processed/corporate_actions/"
        "mechanical_adjustment_events.parquet"
    )

    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "canonical_security_id",
                "ex_date",
                "share_multiplier",
            ]
        )

    events = pd.read_parquet(
        path,
        columns=[
            "canonical_security_id",
            "ex_date",
            "share_multiplier",
        ],
    )

    events["ex_date"] = pd.to_datetime(
        events["ex_date"],
        errors="coerce",
    ).dt.normalize()
    events["share_multiplier"] = pd.to_numeric(
        events["share_multiplier"],
        errors="coerce",
    )

    return events.loc[
        events["canonical_security_id"].notna()
        & events["ex_date"].notna()
        & events["share_multiplier"].gt(0)
    ].copy()


def buy_execution(
    quoted_open: float,
    quantity: float,
    costs: CostProfile,
) -> dict:
    execution_price = float(quoted_open) * (
        1.0 + costs.slippage_bps / 10_000.0
    )
    trade_value = execution_price * quantity

    brokerage = (
        costs.brokerage_per_order
        if quantity > 0
        else 0.0
    )
    exchange = trade_value * costs.exchange_rate
    sebi = trade_value * costs.sebi_rate
    stt = trade_value * costs.stt_buy_rate
    stamp = trade_value * costs.stamp_buy_rate

    # GST treatment is explicit and configurable through the profile. This
    # implementation taxes brokerage + exchange + SEBI service charges.
    gst = costs.gst_rate * (
        brokerage
        + exchange
        + sebi
    )

    fees = (
        brokerage
        + exchange
        + sebi
        + stt
        + stamp
        + gst
    )

    return {
        "execution_price": execution_price,
        "trade_value": trade_value,
        "brokerage": brokerage,
        "exchange_charge": exchange,
        "sebi_charge": sebi,
        "stt": stt,
        "stamp_duty": stamp,
        "gst": gst,
        "dp_charge": 0.0,
        "fees": fees,
        "cash_out": trade_value + fees,
    }


def sell_execution(
    quoted_close: float,
    quantity: float,
    costs: CostProfile,
) -> dict:
    execution_price = float(quoted_close) * (
        1.0 - costs.slippage_bps / 10_000.0
    )
    trade_value = execution_price * quantity

    brokerage = (
        costs.brokerage_per_order
        if quantity > 0
        else 0.0
    )
    exchange = trade_value * costs.exchange_rate
    sebi = trade_value * costs.sebi_rate
    stt = trade_value * costs.stt_sell_rate

    gst = costs.gst_rate * (
        brokerage
        + exchange
        + sebi
    )
    dp = (
        costs.dp_charge_per_sell
        if quantity > 0
        else 0.0
    )

    fees = (
        brokerage
        + exchange
        + sebi
        + stt
        + gst
        + dp
    )

    return {
        "execution_price": execution_price,
        "trade_value": trade_value,
        "brokerage": brokerage,
        "exchange_charge": exchange,
        "sebi_charge": sebi,
        "stt": stt,
        "stamp_duty": 0.0,
        "gst": gst,
        "dp_charge": dp,
        "fees": fees,
        "cash_in": trade_value - fees,
    }


def max_affordable_quantity(
    budget: float,
    quoted_open: float,
    costs: CostProfile,
) -> int:
    if (
        not math.isfinite(float(quoted_open))
        or quoted_open <= 0
        or budget <= 0
    ):
        return 0

    execution_price = float(quoted_open) * (
        1.0 + costs.slippage_bps / 10_000.0
    )

    variable_multiplier = (
        1.0
        + costs.stt_buy_rate
        + costs.stamp_buy_rate
        + costs.exchange_rate
        + costs.sebi_rate
        + costs.gst_rate
        * (
            costs.exchange_rate
            + costs.sebi_rate
        )
    )

    fixed = (
        costs.brokerage_per_order
        * (1.0 + costs.gst_rate)
    )

    if budget <= fixed:
        return 0

    quantity = int(
        math.floor(
            (budget - fixed)
            / (
                execution_price
                * variable_multiplier
            )
        )
    )

    quantity = max(
        0,
        quantity,
    )

    # Guard against floating-point and future cost-profile changes.
    while quantity > 0:
        execution = buy_execution(
            quoted_open,
            quantity,
            costs,
        )
        if execution["cash_out"] <= budget + 1e-9:
            break
        quantity -= 1

    return quantity


def build_day_index(
    panel: pd.DataFrame,
) -> tuple[
    dict[int, pd.Index],
    dict[int, pd.Timestamp],
]:
    groups: dict[int, pd.Index] = {}
    dates: dict[int, pd.Timestamp] = {}

    for market_index, index in panel.groupby(
        "market_day_index",
        sort=True,
    ).groups.items():
        key = int(market_index)
        groups[key] = index

        day_values = panel.loc[
            index,
            "date",
        ].dropna().unique()

        if len(day_values) != 1:
            raise RuntimeError(
                f"Market index {key} maps to {len(day_values)} dates."
            )

        dates[key] = pd.Timestamp(
            day_values[0]
        ).normalize()

    return groups, dates


def build_events_by_market_index(
    events: pd.DataFrame,
    dates: dict[int, pd.Timestamp],
) -> dict[int, list[tuple[str, float]]]:
    if events.empty:
        return {}

    ordered_indices = sorted(dates)
    ordered_dates = pd.Index(
        [dates[i] for i in ordered_indices]
    )

    output: dict[
        int,
        list[tuple[str, float]],
    ] = {}

    for row in events.itertuples(index=False):
        loc = int(
            ordered_dates.searchsorted(
                pd.Timestamp(
                    row.ex_date
                ).normalize(),
                side="left",
            )
        )

        if loc >= len(ordered_indices):
            continue

        market_index = ordered_indices[loc]

        output.setdefault(
            market_index,
            [],
        ).append(
            (
                str(
                    row.canonical_security_id
                ),
                float(
                    row.share_multiplier
                ),
            )
        )

    return output


def strategy_candidates(
    day: pd.DataFrame,
    strategy: str,
    *,
    top_k: int,
) -> pd.DataFrame:
    spec = STRATEGIES[strategy]
    score_col = spec["column"]

    candidates = day.loc[
        day["eligible_universe"].fillna(False)
    ].copy()

    candidates["strategy_score"] = (
        pd.to_numeric(
            candidates[score_col],
            errors="coerce",
        )
        * float(spec["direction"])
    )

    candidates["strategy_score"] = candidates[
        "_score"
    ].replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    candidates = candidates.loc[
        candidates["strategy_score"].notna()
    ].copy()

    if candidates.empty:
        return candidates

    return (
        candidates.sort_values(
            [
                "strategy_score",
                "turnover_median_20d",
                "canonical_security_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def run_backtest(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    strategy: str,
    initial_capital: float,
    top_k: int,
    holding_sessions: int,
    costs: CostProfile,
) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    day_groups, dates = build_day_index(
        panel
    )
    events_by_index = (
        build_events_by_market_index(
            events,
            dates,
        )
    )

    market_indices = sorted(
        day_groups
    )

    if not market_indices:
        raise RuntimeError(
            "Research panel contains no market days."
        )

    first_index = market_indices[0]
    last_index = market_indices[-1]

    cash = float(initial_capital)
    holdings: dict[str, dict] = {}
    pending_orders: list[dict] = []

    trades: list[dict] = []
    equity_rows: list[dict] = []

    skipped_unsafe_target = 0
    failed_entry_no_row = 0
    failed_entry_bad_price = 0
    delayed_exits = 0
    applied_corporate_actions = 0

    unsafe_col = (
        f"unsafe_target_window_{holding_sessions}d"
    )

    for market_index in market_indices:
        day = panel.loc[
            day_groups[market_index]
        ].copy()
        day_lookup = day.set_index(
            "canonical_security_id",
            drop=False,
        )
        current_date = dates[
            market_index
        ]

        # Existing holders receive the mapped split/bonus quantity change
        # before trading on the ex-date. New entries later at today's open are
        # already post-action and therefore do not receive this multiplier.
        for (
            canonical_security_id,
            share_multiplier,
        ) in events_by_index.get(
            market_index,
            [],
        ):
            position = holdings.get(
                canonical_security_id
            )

            if position is None:
                continue

            position["quantity"] *= (
                share_multiplier
            )
            position[
                "corporate_action_count"
            ] += 1
            applied_corporate_actions += 1

        # Execute orders decided after the previous market close.
        if pending_orders:
            still_pending: list[dict] = []

            for order in pending_orders:
                if order[
                    "entry_market_index"
                ] != market_index:
                    still_pending.append(
                        order
                    )
                    continue

                if order[
                    "unsafe_target"
                ]:
                    # This is an EX-POST DATA-QUALITY exclusion, not a live
                    # strategy decision. We do not replace it with a lower
                    # ranked stock; the slot remains cash.
                    skipped_unsafe_target += 1
                    continue

                cid = order[
                    "canonical_security_id"
                ]

                if cid not in day_lookup.index:
                    failed_entry_no_row += 1
                    continue

                row = day_lookup.loc[cid]

                if isinstance(
                    row,
                    pd.DataFrame,
                ):
                    raise RuntimeError(
                        f"Duplicate {cid} on {current_date.date()}"
                    )

                quoted_open = float(
                    row["open"]
                )

                if (
                    not math.isfinite(
                        quoted_open
                    )
                    or quoted_open <= 0
                ):
                    failed_entry_bad_price += 1
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
                    failed_entry_bad_price += 1
                    continue

                execution = buy_execution(
                    quoted_open,
                    quantity,
                    costs,
                )

                if (
                    execution[
                        "cash_out"
                    ]
                    > cash + 1e-8
                ):
                    raise RuntimeError(
                        "Position sizing exceeded available cash."
                    )

                cash -= execution[
                    "cash_out"
                ]

                holdings[cid] = {
                    "canonical_security_id": cid,
                    "symbol": str(
                        row["symbol"]
                    ),
                    "signal_date": order[
                        "signal_date"
                    ],
                    "signal_market_index": order[
                        "signal_market_index"
                    ],
                    "entry_date": current_date,
                    "entry_market_index": market_index,
                    "scheduled_exit_market_index": order[
                        "scheduled_exit_market_index"
                    ],
                    "quantity": float(
                        quantity
                    ),
                    "initial_quantity": int(
                        quantity
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
                        execution["fees"]
                    ),
                    "entry_cash_out": float(
                        execution[
                            "cash_out"
                        ]
                    ),
                    "score": float(
                        order["score"]
                    ),
                    "last_close": float(
                        row["close"]
                    ),
                    "corporate_action_count": 0,
                }

            pending_orders = still_pending

        # Refresh last observable close, then attempt due exits.
        for cid in list(
            holdings
        ):
            position = holdings[cid]

            if cid in day_lookup.index:
                row = day_lookup.loc[cid]

                if isinstance(
                    row,
                    pd.DataFrame,
                ):
                    raise RuntimeError(
                        f"Duplicate {cid} on {current_date.date()}"
                    )

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

            if (
                market_index
                < position[
                    "scheduled_exit_market_index"
                ]
            ):
                continue

            if cid not in day_lookup.index:
                continue

            row = day_lookup.loc[cid]
            quoted_close = float(
                row["close"]
            )

            if (
                not math.isfinite(
                    quoted_close
                )
                or quoted_close <= 0
            ):
                continue

            execution = sell_execution(
                quoted_close,
                position["quantity"],
                costs,
            )

            cash += execution[
                "cash_in"
            ]

            delayed = max(
                0,
                market_index
                - position[
                    "scheduled_exit_market_index"
                ],
            )

            if delayed:
                delayed_exits += 1

            total_fees = (
                position[
                    "entry_fees"
                ]
                + execution["fees"]
            )
            net_pnl = (
                execution["cash_in"]
                - position[
                    "entry_cash_out"
                ]
            )

            trades.append({
                "strategy": strategy,
                "symbol": position[
                    "symbol"
                ],
                "canonical_security_id": cid,
                "signal_date": position[
                    "signal_date"
                ],
                "entry_date": position[
                    "entry_date"
                ],
                "exit_date": current_date,
                "scheduled_exit_market_index": position[
                    "scheduled_exit_market_index"
                ],
                "actual_exit_market_index": market_index,
                "delayed_exit_sessions": delayed,
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
                ),
                "score": position[
                    "score"
                ],
                "corporate_action_count": position[
                    "corporate_action_count"
                ],
                "written_off": False,
            })

            del holdings[cid]

        invested_market_value = sum(
            position["quantity"]
            * position["last_close"]
            for position in holdings.values()
        )

        equity = (
            cash
            + invested_market_value
        )

        equity_rows.append({
            "date": current_date,
            "market_day_index": market_index,
            "cash": cash,
            "invested_market_value": invested_market_value,
            "equity": equity,
            "open_positions": len(
                holdings
            ),
        })

        # Non-overlapping cohort design: wait until every position from the
        # current cohort is closed before generating the next signal.
        if (
            not holdings
            and not pending_orders
            and market_index < last_index
            and (
                market_index
                + holding_sessions
                <= last_index
            )
        ):
            candidates = (
                strategy_candidates(
                    day,
                    strategy,
                    top_k=top_k,
                )
            )

            if candidates.empty:
                continue

            slot_budget = (
                cash / top_k
            )

            for row in candidates.itertuples(
                index=False
            ):
                pending_orders.append({
                    "canonical_security_id": str(
                        row.canonical_security_id
                    ),
                    "symbol": str(
                        row.symbol
                    ),
                    "score": float(
                        row.strategy_score
                    ),
                    "signal_date": current_date,
                    "signal_market_index": market_index,
                    "entry_market_index": market_index + 1,
                    "scheduled_exit_market_index": (
                        market_index
                        + holding_sessions
                    ),
                    "slot_budget": slot_budget,
                    "unsafe_target": bool(
                        getattr(
                            row,
                            unsafe_col,
                        )
                    ),
                })

    written_off_positions = 0

    # If a security never becomes tradeable again before sample end, do not
    # pretend its stale last price is cash. Conservatively write it down to zero.
    if holdings:
        final_date = dates[
            last_index
        ]

        for cid, position in list(
            holdings.items()
        ):
            written_off_positions += 1
            net_pnl = -position[
                "entry_cash_out"
            ]

            trades.append({
                "strategy": strategy,
                "symbol": position[
                    "symbol"
                ],
                "canonical_security_id": cid,
                "signal_date": position[
                    "signal_date"
                ],
                "entry_date": position[
                    "entry_date"
                ],
                "exit_date": final_date,
                "scheduled_exit_market_index": position[
                    "scheduled_exit_market_index"
                ],
                "actual_exit_market_index": None,
                "delayed_exit_sessions": None,
                "initial_quantity": position[
                    "initial_quantity"
                ],
                "exit_quantity": 0.0,
                "entry_price": position[
                    "entry_execution_price"
                ],
                "exit_price": 0.0,
                "entry_trade_value": position[
                    "entry_trade_value"
                ],
                "exit_trade_value": 0.0,
                "entry_fees": position[
                    "entry_fees"
                ],
                "exit_fees": 0.0,
                "total_fees": position[
                    "entry_fees"
                ],
                "net_pnl": net_pnl,
                "net_return": -1.0,
                "score": position[
                    "score"
                ],
                "corporate_action_count": position[
                    "corporate_action_count"
                ],
                "written_off": True,
            })

        holdings.clear()

        if equity_rows:
            equity_rows[-1][
                "invested_market_value"
            ] = 0.0
            equity_rows[-1][
                "equity"
            ] = cash
            equity_rows[-1][
                "open_positions"
            ] = 0

    trades_df = pd.DataFrame(
        trades
    )
    equity_df = pd.DataFrame(
        equity_rows
    )

    metrics = performance_metrics(
        equity_df,
        trades_df,
        initial_capital=initial_capital,
    )

    metrics.update({
        "strategy": strategy,
        "top_k": int(top_k),
        "holding_sessions": int(
            holding_sessions
        ),
        "skipped_unsafe_target_slots": int(
            skipped_unsafe_target
        ),
        "failed_entry_no_row": int(
            failed_entry_no_row
        ),
        "failed_entry_bad_price_or_budget": int(
            failed_entry_bad_price
        ),
        "delayed_exit_trades": int(
            delayed_exits
        ),
        "written_off_positions": int(
            written_off_positions
        ),
        "applied_corporate_actions": int(
            applied_corporate_actions
        ),
    })

    return {
        "metrics": metrics,
        "trades": trades_df,
        "equity": equity_df,
    }


def performance_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> dict:
    if equity.empty:
        return {
            "starting_capital": float(
                initial_capital
            ),
            "ending_equity": float(
                initial_capital
            ),
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "annualized_volatility": 0.0,
            "trades": 0,
            "win_rate": None,
            "total_fees": 0.0,
            "turnover": 0.0,
            "average_exposure": 0.0,
        }

    equity = equity.sort_values(
        "date"
    ).copy()

    ending_equity = float(
        equity["equity"].iloc[-1]
    )
    total_return = (
        ending_equity
        / initial_capital
        - 1.0
    )

    start_date = pd.Timestamp(
        equity["date"].iloc[0]
    )
    end_date = pd.Timestamp(
        equity["date"].iloc[-1]
    )

    years = max(
        (
            end_date - start_date
        ).days
        / 365.25,
        1.0 / 365.25,
    )

    if ending_equity > 0:
        cagr = (
            ending_equity
            / initial_capital
        ) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    running_max = (
        equity["equity"]
        .cummax()
        .replace(0, pd.NA)
    )
    drawdown = (
        equity["equity"]
        / running_max
        - 1.0
    )
    max_drawdown = float(
        drawdown.min()
    )

    daily_returns = (
        equity["equity"]
        .pct_change()
        .replace(
            [
                float("inf"),
                float("-inf"),
            ],
            pd.NA,
        )
        .dropna()
    )

    if (
        len(daily_returns) >= 2
        and float(
            daily_returns.std()
        ) > 0
    ):
        annualized_volatility = (
            float(
                daily_returns.std()
            )
            * math.sqrt(252.0)
        )
        sharpe = (
            float(
                daily_returns.mean()
            )
            / float(
                daily_returns.std()
            )
            * math.sqrt(252.0)
        )
    else:
        annualized_volatility = 0.0
        sharpe = 0.0

    if trades.empty:
        trade_count = 0
        win_rate = None
        total_fees = 0.0
        turnover = 0.0
    else:
        trade_count = int(
            len(trades)
        )
        win_rate = float(
            trades["net_pnl"]
            .gt(0)
            .mean()
        )
        total_fees = float(
            trades["total_fees"]
            .sum()
        )
        turnover = float(
            (
                trades[
                    "entry_trade_value"
                ].fillna(0)
                + trades[
                    "exit_trade_value"
                ].fillna(0)
            ).sum()
        )

    exposure = (
        equity[
            "invested_market_value"
        ]
        / equity["equity"].where(
            equity["equity"] > 0
        )
    )

    return {
        "starting_capital": float(
            initial_capital
        ),
        "ending_equity": (
            ending_equity
        ),
        "total_return": float(
            total_return
        ),
        "cagr": float(cagr),
        "max_drawdown": float(
            max_drawdown
        ),
        "sharpe": float(sharpe),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "trades": trade_count,
        "win_rate": win_rate,
        "total_fees": total_fees,
        "turnover": turnover,
        "average_exposure": float(
            exposure.fillna(0).mean()
        ),
    }


def self_test() -> None:
    dates = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="D",
    )

    rows: list[dict] = []

    for i, day in enumerate(dates):
        rows.append({
            "date": day,
            "market_day_index": i,
            "canonical_security_id": "ISIN:TEST",
            "symbol": "TEST",
            "eligible_universe": True,
            "open": 100.0 + i,
            "close": 100.5 + i,
            "turnover_median_20d": 10_000_000.0,
            "return_5d": 0.05,
            "return_20d": 0.10,
            "return_60d": 0.20,
            "volatility_20d": 0.02,
            "unsafe_target_window_3d": False,
        })

    panel = pd.DataFrame(rows)
    panel[
        "risk_adjusted_momentum_20d"
    ] = (
        panel["return_20d"]
        / panel["volatility_20d"]
    )

    zero_costs = CostProfile(
        stt_buy_rate=0.0,
        stt_sell_rate=0.0,
        stamp_buy_rate=0.0,
        sebi_rate=0.0,
        exchange_rate=0.0,
        gst_rate=0.0,
        brokerage_per_order=0.0,
        dp_charge_per_sell=0.0,
        slippage_bps=0.0,
    )

    result = run_backtest(
        panel,
        pd.DataFrame(
            columns=[
                "canonical_security_id",
                "ex_date",
                "share_multiplier",
            ]
        ),
        strategy="momentum_5d",
        initial_capital=1_000.0,
        top_k=1,
        holding_sessions=3,
        costs=zero_costs,
    )

    trades = result["trades"]

    if trades.empty:
        raise AssertionError(
            "Synthetic backtest produced no trades."
        )

    first = trades.iloc[0]

    assert (
        pd.Timestamp(
            first["entry_date"]
        )
        == dates[1]
    )
    assert (
        pd.Timestamp(
            first["exit_date"]
        )
        == dates[3]
    )
    assert int(
        first["initial_quantity"]
    ) == 9

    buy = buy_execution(
        100.0,
        1,
        CostProfile(
            brokerage_per_order=0.0,
            dp_charge_per_sell=0.0,
            slippage_bps=0.0,
        ),
    )
    assert buy["cash_out"] > 100.0

    print(
        "Baseline backtester self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run deterministic long-only NSE baseline strategies using "
            "point-in-time universe membership, next-session-open entry, "
            "integer initial share quantities and explicit Indian delivery "
            "equity transaction costs."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(
            STRATEGIES
        ),
        choices=list(
            STRATEGIES
        ),
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--holding-sessions",
        type=int,
        choices=[
            1,
            3,
            5,
            10,
            20,
        ],
        default=5,
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
        "--zero-costs",
        action="store_true",
        help=(
            "Run gross-return baseline with all transaction costs/slippage zero."
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.capital <= 0:
        raise SystemExit(
            "--capital must be > 0"
        )
    if args.top_k < 1:
        raise SystemExit(
            "--top-k must be >= 1"
        )
    if args.brokerage_per_order < 0:
        raise SystemExit(
            "--brokerage-per-order must be >= 0"
        )
    if args.dp_charge_per_sell < 0:
        raise SystemExit(
            "--dp-charge-per-sell must be >= 0"
        )
    if args.slippage_bps < 0:
        raise SystemExit(
            "--slippage-bps must be >= 0"
        )

    root = Path(
        args.root
    ).resolve()

    panel = load_panel(
        root
        / "data/processed/research_panel",
        holding_sessions=(
            args.holding_sessions
        ),
    )
    events = load_mechanical_events(
        root
    )

    if args.zero_costs:
        costs = CostProfile(
            stt_buy_rate=0.0,
            stt_sell_rate=0.0,
            stamp_buy_rate=0.0,
            sebi_rate=0.0,
            exchange_rate=0.0,
            gst_rate=0.0,
            brokerage_per_order=0.0,
            dp_charge_per_sell=0.0,
            slippage_bps=0.0,
        )
        cost_profile_name = "zero_costs"
    else:
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
        cost_profile_name = (
            "current_2026_delivery"
        )

    reports_root = (
        root
        / "reports/backtests"
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_rows: list[dict] = []

    for strategy in args.strategies:
        print(
            f"\n=== {strategy} ==="
        )

        result = run_backtest(
            panel,
            events,
            strategy=strategy,
            initial_capital=(
                args.capital
            ),
            top_k=args.top_k,
            holding_sessions=(
                args.holding_sessions
            ),
            costs=costs,
        )

        strategy_root = (
            reports_root
            / strategy
            / (
                f"h{args.holding_sessions}_"
                f"k{args.top_k}_"
                f"{cost_profile_name}"
            )
        )
        strategy_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        result["trades"].to_csv(
            strategy_root
            / "trades.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        result["equity"].to_csv(
            strategy_root
            / "equity_curve.csv",
            index=False,
            date_format="%Y-%m-%d",
        )

        metrics = {
            **result["metrics"],
            "cost_profile": (
                cost_profile_name
            ),
            "cost_parameters": asdict(
                costs
            ),
            "data_quality_policy": (
                "rank using point-in-time eligible_universe; "
                "if a selected trade's future label crosses an unresolved "
                "split/bonus, leave that selected slot in cash and do not "
                "replace it with a lower-ranked stock"
            ),
            "cohort_policy": (
                "non-overlapping; generate a new signal only after every "
                "position from the prior cohort has exited"
            ),
        }

        (
            strategy_root
            / "metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
                default=str,
            )
            + "\n"
        )

        comparison_rows.append(
            metrics
        )

        print(
            f"Ending equity:   ₹"
            f"{metrics['ending_equity']:,.2f}"
        )
        print(
            f"Total return:    "
            f"{metrics['total_return']:.2%}"
        )
        print(
            f"CAGR:            "
            f"{metrics['cagr']:.2%}"
        )
        print(
            f"Max drawdown:    "
            f"{metrics['max_drawdown']:.2%}"
        )
        print(
            f"Sharpe:          "
            f"{metrics['sharpe']:.3f}"
        )
        print(
            f"Trades:          "
            f"{metrics['trades']:,}"
        )
        print(
            f"Total fees:      ₹"
            f"{metrics['total_fees']:,.2f}"
        )
        print(
            "Unsafe slots:    "
            f"{metrics['skipped_unsafe_target_slots']:,}"
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        reports_root
        / (
            f"baseline_comparison_"
            f"h{args.holding_sessions}_"
            f"k{args.top_k}_"
            f"{cost_profile_name}.csv"
        )
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    run_summary = {
        "capital": float(
            args.capital
        ),
        "top_k": int(
            args.top_k
        ),
        "holding_sessions": int(
            args.holding_sessions
        ),
        "strategies": list(
            args.strategies
        ),
        "cost_profile": (
            cost_profile_name
        ),
        "cost_parameters": asdict(
            costs
        ),
        "comparison_csv": str(
            comparison_path
        ),
    }

    summary_path = (
        reports_root
        / (
            f"run_summary_"
            f"h{args.holding_sessions}_"
            f"k{args.top_k}_"
            f"{cost_profile_name}.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            run_summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n=== BASELINE SUITE COMPLETE ==="
    )
    print(
        f"Comparison: {comparison_path}"
    )
    print(
        f"Run summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
