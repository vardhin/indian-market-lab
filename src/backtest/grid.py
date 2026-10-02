from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from baselines import (
    CostProfile,
    load_mechanical_events,
    load_panel,
    run_backtest,
)


DEFAULT_STRATEGIES = [
    "momentum_20d",
    "momentum_60d",
    "risk_adjusted_momentum_20d",
    "risk_adjusted_momentum_60d",
    "liquidity_control",
]

DEFAULT_HOLDS = [5, 10, 20]
DEFAULT_TOP_K = [3, 5, 10]


def cost_profile(
    name: str,
    *,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> CostProfile:
    if name == "zero":
        return CostProfile(
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

    if name == "current":
        return CostProfile(
            brokerage_per_order=(
                brokerage_per_order
            ),
            dp_charge_per_sell=(
                dp_charge_per_sell
            ),
            slippage_bps=slippage_bps,
        )

    raise ValueError(
        f"Unknown cost profile: {name}"
    )


def run_grid(
    *,
    root: Path,
    strategies: list[str],
    holding_sessions: list[int],
    top_k_values: list[int],
    cost_profiles: list[str],
    capital: float,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> pd.DataFrame:
    root = Path(root).resolve()
    panel_root = (
        root
        / "data/processed/research_panel"
    )
    events = load_mechanical_events(
        root
    )

    rows: list[dict] = []

    total = (
        len(strategies)
        * len(holding_sessions)
        * len(top_k_values)
        * len(cost_profiles)
    )
    run_number = 0

    for hold in holding_sessions:
        print(
            f"\nLoading panel for "
            f"holding_sessions={hold}..."
        )

        panel = load_panel(
            panel_root,
            holding_sessions=hold,
        )

        for profile_name in cost_profiles:
            costs = cost_profile(
                profile_name,
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

            for top_k in top_k_values:
                for strategy in strategies:
                    run_number += 1

                    print(
                        f"[{run_number:02d}/"
                        f"{total:02d}] "
                        f"{strategy} | "
                        f"h={hold} | "
                        f"k={top_k} | "
                        f"cost={profile_name}",
                        flush=True,
                    )

                    result = run_backtest(
                        panel,
                        events,
                        strategy=strategy,
                        initial_capital=capital,
                        top_k=top_k,
                        holding_sessions=hold,
                        costs=costs,
                    )

                    metrics = dict(
                        result["metrics"]
                    )

                    rows.append({
                        "strategy": strategy,
                        "holding_sessions": hold,
                        "top_k": top_k,
                        "cost_profile": (
                            profile_name
                        ),
                        "starting_capital": (
                            capital
                        ),
                        "ending_equity": metrics[
                            "ending_equity"
                        ],
                        "total_return": metrics[
                            "total_return"
                        ],
                        "cagr": metrics["cagr"],
                        "max_drawdown": metrics[
                            "max_drawdown"
                        ],
                        "sharpe": metrics[
                            "sharpe"
                        ],
                        "annualized_volatility": (
                            metrics[
                                "annualized_volatility"
                            ]
                        ),
                        "trades": metrics[
                            "trades"
                        ],
                        "win_rate": metrics[
                            "win_rate"
                        ],
                        "total_fees": metrics[
                            "total_fees"
                        ],
                        "turnover": metrics[
                            "turnover"
                        ],
                        "average_exposure": metrics[
                            "average_exposure"
                        ],
                        "unsafe_slots": metrics[
                            "skipped_unsafe_target_slots"
                        ],
                        "failed_entry_no_row": (
                            metrics[
                                "failed_entry_no_row"
                            ]
                        ),
                        "failed_entry_bad_price_or_budget": (
                            metrics[
                                "failed_entry_bad_price_or_budget"
                            ]
                        ),
                        "delayed_exit_trades": (
                            metrics[
                                "delayed_exit_trades"
                            ]
                        ),
                        "written_off_positions": (
                            metrics[
                                "written_off_positions"
                            ]
                        ),
                        "applied_corporate_actions": (
                            metrics[
                                "applied_corporate_actions"
                            ]
                        ),
                    })

    return pd.DataFrame(
        rows
    )


def add_derived_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    results = results.copy()

    # Calmar is undefined when max drawdown is zero.
    drawdown_abs = (
        results["max_drawdown"]
        .abs()
        .where(
            results[
                "max_drawdown"
            ].abs() > 0
        )
    )

    results["calmar"] = (
        results["cagr"]
        / drawdown_abs
    )

    results["fees_to_starting_capital"] = (
        results["total_fees"]
        / results[
            "starting_capital"
        ]
    )

    results["turnover_multiple"] = (
        results["turnover"]
        / results[
            "starting_capital"
        ]
    )

    return results


def build_gross_net_pairs(
    results: pd.DataFrame,
) -> pd.DataFrame:
    if not {
        "current",
        "zero",
    }.issubset(
        set(
            results[
                "cost_profile"
            ].unique()
        )
    ):
        return pd.DataFrame()

    keys = [
        "strategy",
        "holding_sessions",
        "top_k",
    ]

    gross = (
        results.loc[
            results[
                "cost_profile"
            ].eq("zero"),
            keys
            + [
                "ending_equity",
                "cagr",
                "sharpe",
                "max_drawdown",
            ],
        ]
        .rename(
            columns={
                "ending_equity": (
                    "gross_ending_equity"
                ),
                "cagr": "gross_cagr",
                "sharpe": "gross_sharpe",
                "max_drawdown": (
                    "gross_max_drawdown"
                ),
            }
        )
    )

    net = (
        results.loc[
            results[
                "cost_profile"
            ].eq("current"),
            keys
            + [
                "ending_equity",
                "cagr",
                "sharpe",
                "max_drawdown",
                "total_fees",
            ],
        ]
        .rename(
            columns={
                "ending_equity": (
                    "net_ending_equity"
                ),
                "cagr": "net_cagr",
                "sharpe": "net_sharpe",
                "max_drawdown": (
                    "net_max_drawdown"
                ),
            }
        )
    )

    paired = gross.merge(
        net,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    paired[
        "cagr_cost_drag_pp"
    ] = (
        (
            paired["gross_cagr"]
            - paired["net_cagr"]
        )
        * 100.0
    )

    paired[
        "terminal_wealth_cost_drag"
    ] = (
        paired[
            "gross_ending_equity"
        ]
        - paired[
            "net_ending_equity"
        ]
    )

    return paired.sort_values(
        [
            "net_cagr",
            "net_sharpe",
        ],
        ascending=[
            False,
            False,
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a compact deterministic robustness grid over strategy, "
            "holding period, portfolio breadth and cost profile."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
    )
    ap.add_argument(
        "--holding-sessions",
        nargs="+",
        type=int,
        default=DEFAULT_HOLDS,
        choices=[
            1,
            3,
            5,
            10,
            20,
        ],
    )
    ap.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=DEFAULT_TOP_K,
    )
    ap.add_argument(
        "--cost-profiles",
        nargs="+",
        choices=[
            "current",
            "zero",
        ],
        default=["current"],
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
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

    if args.capital <= 0:
        raise SystemExit(
            "--capital must be > 0"
        )

    if any(
        value < 1
        for value in args.top_k
    ):
        raise SystemExit(
            "--top-k values must all be >= 1"
        )

    root = Path(
        args.root
    ).resolve()

    results = run_grid(
        root=root,
        strategies=list(
            args.strategies
        ),
        holding_sessions=list(
            args.holding_sessions
        ),
        top_k_values=list(
            args.top_k
        ),
        cost_profiles=list(
            args.cost_profiles
        ),
        capital=args.capital,
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

    results = add_derived_metrics(
        results
    )

    output_root = (
        root
        / "reports/backtests/grid"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cost_tag = "-".join(
        args.cost_profiles
    )
    hold_tag = "-".join(
        str(x)
        for x in args.holding_sessions
    )
    k_tag = "-".join(
        str(x)
        for x in args.top_k
    )

    stem = (
        f"grid_h{hold_tag}"
        f"_k{k_tag}"
        f"_{cost_tag}"
    )

    results_path = (
        output_root
        / f"{stem}.csv"
    )

    results.sort_values(
        [
            "cost_profile",
            "cagr",
            "sharpe",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).to_csv(
        results_path,
        index=False,
    )

    current = results.loc[
        results[
            "cost_profile"
        ].eq("current")
    ].copy()

    if not current.empty:
        pivot_cagr = (
            current.pivot_table(
                index=[
                    "strategy",
                    "holding_sessions",
                ],
                columns="top_k",
                values="cagr",
                aggfunc="first",
            )
            * 100.0
        )

        pivot_path = (
            output_root
            / f"{stem}_net_cagr_percent.csv"
        )
        pivot_cagr.to_csv(
            pivot_path
        )
    else:
        pivot_path = None

    paired = build_gross_net_pairs(
        results
    )

    if not paired.empty:
        paired_path = (
            output_root
            / f"{stem}_gross_vs_net.csv"
        )
        paired.to_csv(
            paired_path,
            index=False,
        )
    else:
        paired_path = None

    config = {
        "capital": args.capital,
        "strategies": list(
            args.strategies
        ),
        "holding_sessions": list(
            args.holding_sessions
        ),
        "top_k": list(
            args.top_k
        ),
        "cost_profiles": list(
            args.cost_profiles
        ),
        "current_cost_parameters": (
            asdict(
                cost_profile(
                    "current",
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
            )
        ),
        "results_csv": str(
            results_path
        ),
        "net_cagr_pivot_csv": (
            str(pivot_path)
            if pivot_path
            else None
        ),
        "gross_vs_net_csv": (
            str(paired_path)
            if paired_path
            else None
        ),
    }

    config_path = (
        output_root
        / f"{stem}.json"
    )
    config_path.write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n=== GRID COMPLETE ==="
    )
    print(
        f"Runs:       {len(results):,}"
    )
    print(
        f"Results:    {results_path}"
    )

    if pivot_path is not None:
        print(
            f"Net CAGR:   {pivot_path}"
        )

    if paired_path is not None:
        print(
            f"Gross/net:  {paired_path}"
        )

    print(
        "\nTop configurations by CAGR:"
    )

    preview = (
        results.sort_values(
            [
                "cagr",
                "sharpe",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(15)
        .loc[
            :,
            [
                "strategy",
                "holding_sessions",
                "top_k",
                "cost_profile",
                "ending_equity",
                "cagr",
                "max_drawdown",
                "sharpe",
                "trades",
                "total_fees",
            ],
        ]
        .copy()
    )

    preview["cagr"] *= 100.0
    preview[
        "max_drawdown"
    ] *= 100.0

    print(
        preview.to_string(
            index=False,
            formatters={
                "ending_equity": (
                    lambda x: (
                        f"{x:,.2f}"
                    )
                ),
                "cagr": (
                    lambda x: (
                        f"{x:.2f}%"
                    )
                ),
                "max_drawdown": (
                    lambda x: (
                        f"{x:.2f}%"
                    )
                ),
                "sharpe": (
                    lambda x: (
                        f"{x:.3f}"
                    )
                ),
                "total_fees": (
                    lambda x: (
                        f"{x:,.2f}"
                    )
                ),
            },
        )
    )


if __name__ == "__main__":
    main()
