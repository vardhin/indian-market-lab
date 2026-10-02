from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


METRIC_GLOSSARY: dict[str, dict[str, str]] = {
    "starting_capital": {
        "label": "Starting Capital",
        "caption": "Capital available at the beginning of the backtest.",
        "category": "return",
    },
    "ending_equity": {
        "label": "Ending Equity",
        "caption": "Portfolio value at the end of the backtest after modeled costs.",
        "category": "return",
    },
    "net_profit": {
        "label": "Net Profit",
        "caption": "Ending equity minus starting capital.",
        "category": "return",
    },
    "total_return": {
        "label": "Total Return",
        "caption": "Percentage gain or loss over the complete evaluation period.",
        "category": "return",
    },
    "cagr": {
        "label": "CAGR",
        "caption": "Constant annual compound return that would reproduce the observed start-to-end growth.",
        "category": "return",
    },
    "annualized_arithmetic_return": {
        "label": "Annualized Arithmetic Return",
        "caption": "Mean daily return multiplied to a one-year equivalent; unlike CAGR it does not represent geometric compounding.",
        "category": "return",
    },
    "annualized_volatility": {
        "label": "Annualized Volatility",
        "caption": "Annualized standard deviation of daily portfolio returns; higher means a bumpier equity curve.",
        "category": "risk",
    },
    "risk_free_rate_annual": {
        "label": "Risk-Free Rate",
        "caption": "Annual low-risk reference return subtracted when computing Sharpe and Sortino.",
        "category": "risk_adjusted",
    },
    "sharpe": {
        "label": "Sharpe Ratio",
        "caption": "Annualized excess return earned per unit of total return volatility.",
        "category": "risk_adjusted",
    },
    "downside_deviation": {
        "label": "Downside Deviation",
        "caption": "Annualized variability of only returns falling below the risk-free daily return.",
        "category": "risk",
    },
    "sortino": {
        "label": "Sortino Ratio",
        "caption": "Annualized excess return per unit of harmful downside volatility.",
        "category": "risk_adjusted",
    },
    "max_drawdown": {
        "label": "Maximum Drawdown",
        "caption": "Worst peak-to-trough percentage fall in portfolio value.",
        "category": "drawdown",
    },
    "max_drawdown_duration_sessions": {
        "label": "Max Drawdown Duration",
        "caption": "Longest number of market observations spent below the previous high-water mark.",
        "category": "drawdown",
    },
    "max_drawdown_duration_days": {
        "label": "Max Drawdown Duration (Days)",
        "caption": "Longest calendar-time stretch spent below the previous high-water mark.",
        "category": "drawdown",
    },
    "high_water_mark": {
        "label": "High-Water Mark",
        "caption": "Highest portfolio value reached at any point in the evaluation.",
        "category": "drawdown",
    },
    "calmar": {
        "label": "Calmar Ratio",
        "caption": "CAGR divided by the absolute maximum drawdown.",
        "category": "risk_adjusted",
    },
    "recovery_factor": {
        "label": "Recovery Factor",
        "caption": "Net profit divided by the rupee value of the worst drawdown from its peak.",
        "category": "risk_adjusted",
    },
    "var_95": {
        "label": "95% Historical VaR",
        "caption": "Fifth percentile of daily returns; roughly 5% of historical days were worse.",
        "category": "tail_risk",
    },
    "cvar_95": {
        "label": "95% CVaR / Expected Shortfall",
        "caption": "Average daily return on observations at or below the 95% historical VaR threshold.",
        "category": "tail_risk",
    },
    "skewness": {
        "label": "Skewness",
        "caption": "Asymmetry of daily returns; negative values indicate a heavier downside tail.",
        "category": "distribution",
    },
    "excess_kurtosis": {
        "label": "Excess Kurtosis",
        "caption": "Tail-heaviness of daily returns relative to a normal distribution, whose excess kurtosis is zero.",
        "category": "distribution",
    },
    "best_day_return": {
        "label": "Best Day",
        "caption": "Largest single-observation percentage gain in the equity curve.",
        "category": "distribution",
    },
    "worst_day_return": {
        "label": "Worst Day",
        "caption": "Largest single-observation percentage loss in the equity curve.",
        "category": "distribution",
    },
    "positive_day_fraction": {
        "label": "Positive-Day Rate",
        "caption": "Fraction of daily portfolio returns greater than zero.",
        "category": "distribution",
    },
    "trades": {
        "label": "Trades",
        "caption": "Number of completed or written-off positions recorded by the simulator.",
        "category": "trading",
    },
    "win_rate": {
        "label": "Win Rate / Hit Rate",
        "caption": "Fraction of completed trades whose net P&L is positive.",
        "category": "trading",
    },
    "average_win": {
        "label": "Average Win",
        "caption": "Mean net rupee P&L among profitable trades.",
        "category": "trading",
    },
    "average_loss": {
        "label": "Average Loss",
        "caption": "Mean net rupee P&L among losing trades; reported as a negative number.",
        "category": "trading",
    },
    "payoff_ratio": {
        "label": "Payoff / Reward-Risk Ratio",
        "caption": "Average winning-trade profit divided by the absolute average losing-trade loss.",
        "category": "trading",
    },
    "expectancy_per_trade": {
        "label": "Expectancy",
        "caption": "Average net rupee P&L per trade across winners and losers.",
        "category": "trading",
    },
    "profit_factor": {
        "label": "Profit Factor",
        "caption": "Gross profits divided by the absolute value of gross losses.",
        "category": "trading",
    },
    "average_trade_return": {
        "label": "Average Trade Return",
        "caption": "Mean net percentage return across recorded trades.",
        "category": "trading",
    },
    "total_fees": {
        "label": "Transaction Costs",
        "caption": "Total modeled brokerage, taxes, exchange charges and other explicit fees paid by trades.",
        "category": "cost",
    },
    "fees_pct_initial_capital": {
        "label": "Fees / Initial Capital",
        "caption": "Total explicit transaction fees divided by starting capital.",
        "category": "cost",
    },
    "turnover": {
        "label": "Turnover (Rupees)",
        "caption": "Total executed buy plus sell notional across the backtest.",
        "category": "cost",
    },
    "turnover_multiple": {
        "label": "Turnover Multiple",
        "caption": "Total traded notional divided by initial capital.",
        "category": "cost",
    },
    "annualized_turnover_multiple": {
        "label": "Annualized Turnover Multiple",
        "caption": "Turnover multiple divided by evaluation years, showing approximate capital churn per year.",
        "category": "cost",
    },
    "average_exposure": {
        "label": "Average Exposure",
        "caption": "Average fraction of portfolio equity invested in market positions rather than cash.",
        "category": "portfolio",
    },
    "max_exposure": {
        "label": "Maximum Exposure",
        "caption": "Largest observed fraction of portfolio equity invested in market positions.",
        "category": "portfolio",
    },
    "benchmark_total_return": {
        "label": "Benchmark Total Return",
        "caption": "Total return of the aligned benchmark over the same observations.",
        "category": "benchmark",
    },
    "benchmark_cagr": {
        "label": "Benchmark CAGR",
        "caption": "Compound annual growth rate of the aligned benchmark.",
        "category": "benchmark",
    },
    "excess_cagr": {
        "label": "Benchmark Excess CAGR",
        "caption": "Strategy CAGR minus benchmark CAGR over the aligned period.",
        "category": "benchmark",
    },
    "beta": {
        "label": "Beta",
        "caption": "Historical sensitivity of strategy daily returns to benchmark daily returns.",
        "category": "benchmark",
    },
    "alpha_annualized": {
        "label": "CAPM Alpha",
        "caption": "Annualized return unexplained by the risk-free rate and estimated benchmark beta.",
        "category": "benchmark",
    },
    "tracking_error": {
        "label": "Tracking Error",
        "caption": "Annualized volatility of strategy return minus benchmark return.",
        "category": "benchmark",
    },
    "information_ratio": {
        "label": "Information Ratio",
        "caption": "Annualized mean active return divided by tracking error; measures consistency of benchmark outperformance.",
        "category": "benchmark",
    },
    "benchmark_correlation": {
        "label": "Benchmark Correlation",
        "caption": "Linear correlation between strategy and benchmark daily returns.",
        "category": "benchmark",
    },
    "mean_daily_ic": {
        "label": "Mean Daily Rank IC",
        "caption": "Average daily Spearman correlation between model scores and future cross-sectional returns.",
        "category": "ml",
    },
    "icir": {
        "label": "ICIR",
        "caption": "Mean information coefficient divided by its time-series standard deviation.",
        "category": "ml",
    },
    "mean_top_decile_excess": {
        "label": "Top-Decile Excess Return",
        "caption": "Average future return of the model's top score decile minus the eligible-universe average.",
        "category": "ml",
    },
    "top_k_return": {
        "label": "Top-K Return",
        "caption": "Future return achieved by the highest-ranked K securities.",
        "category": "ml",
    },
    "decile_monotonicity": {
        "label": "Decile Monotonicity",
        "caption": "Whether realized future returns improve progressively from low-score to high-score buckets.",
        "category": "ml",
    },
    "rolling_sharpe": {
        "label": "Rolling Sharpe",
        "caption": "Sharpe ratio recomputed on a moving window to reveal changes in risk-adjusted performance through time.",
        "category": "rolling",
    },
    "rolling_drawdown": {
        "label": "Rolling Drawdown",
        "caption": "Drawdown measured through time relative to the running portfolio high-water mark.",
        "category": "rolling",
    },
    "gross_return": {
        "label": "Gross Return",
        "caption": "Strategy return before modeled transaction costs.",
        "category": "cost",
    },
    "net_return": {
        "label": "Net Return",
        "caption": "Strategy return after modeled transaction costs.",
        "category": "cost",
    },
    "slippage": {
        "label": "Slippage",
        "caption": "Modeled difference between quoted market price and assumed execution price.",
        "category": "execution",
    },
    "liquidity": {
        "label": "Liquidity",
        "caption": "Ability to trade a security without materially moving its price; proxied here by point-in-time turnover/activity filters.",
        "category": "execution",
    },
    "market_impact": {
        "label": "Market Impact",
        "caption": "Price movement caused by the strategy's own orders; not explicitly modeled at the current small-capital scale.",
        "category": "execution",
    },
    "capacity": {
        "label": "Capacity",
        "caption": "Capital the strategy could plausibly manage before liquidity and market impact materially erode its edge; not yet estimated.",
        "category": "execution",
    },
    "mae": {
        "label": "Maximum Adverse Excursion",
        "caption": "Worst intratrade move against a position before exit; requires intratrade path tracking and is not yet computed.",
        "category": "trading",
    },
    "mfe": {
        "label": "Maximum Favorable Excursion",
        "caption": "Best intratrade move in a position's favor before exit; requires intratrade path tracking and is not yet computed.",
        "category": "trading",
    },
}


def glossary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": key,
                **value,
            }
            for key, value
            in METRIC_GLOSSARY.items()
        ]
    ).sort_values(
        [
            "category",
            "metric",
        ]
    ).reset_index(
        drop=True
    )


def _clean_returns(
    values: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .astype(float)
    )


def _annual_rf_to_daily(
    annual_rate: float,
) -> float:
    if annual_rate <= -1.0:
        raise ValueError(
            "annual risk-free rate "
            "must be greater than -100%."
        )

    return (
        (1.0 + annual_rate)
        ** (
            1.0
            / TRADING_DAYS_PER_YEAR
        )
        - 1.0
    )


def _drawdown_duration(
    dates: pd.Series,
    values: pd.Series,
) -> tuple[int, int]:
    running_max = (
        values.cummax()
    )
    underwater = (
        values
        < running_max
        * (
            1.0 - 1e-12
        )
    )

    max_sessions = 0
    max_days = 0
    start_position: (
        int | None
    ) = None

    clean_dates = pd.to_datetime(
        dates,
        errors="coerce",
    )

    for position, active in enumerate(
        underwater.to_numpy(
            dtype=bool
        )
    ):
        if active:
            if start_position is None:
                start_position = max(
                    0,
                    position - 1,
                )

            sessions = (
                position
                - start_position
            )
            max_sessions = max(
                max_sessions,
                sessions,
            )

            if (
                not pd.isna(
                    clean_dates.iloc[
                        start_position
                    ]
                )
                and not pd.isna(
                    clean_dates.iloc[
                        position
                    ]
                )
            ):
                days = int(
                    (
                        clean_dates.iloc[
                            position
                        ]
                        - clean_dates.iloc[
                            start_position
                        ]
                    ).days
                )
                max_days = max(
                    max_days,
                    days,
                )
        else:
            start_position = None

    return (
        int(max_sessions),
        int(max_days),
    )


def compute_performance_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None,
    *,
    initial_capital: float,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    if equity.empty:
        return {
            "starting_capital": float(
                initial_capital
            ),
            "ending_equity": float(
                initial_capital
            ),
            "net_profit": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_arithmetic_return": 0.0,
            "annualized_volatility": 0.0,
            "risk_free_rate_annual": float(
                annual_risk_free_rate
            ),
            "sharpe": 0.0,
            "downside_deviation": 0.0,
            "sortino": None,
            "max_drawdown": 0.0,
            "max_drawdown_duration_sessions": 0,
            "max_drawdown_duration_days": 0,
            "high_water_mark": float(
                initial_capital
            ),
            "calmar": None,
            "recovery_factor": None,
            "var_95": None,
            "cvar_95": None,
            "skewness": None,
            "excess_kurtosis": None,
            "best_day_return": None,
            "worst_day_return": None,
            "positive_day_fraction": None,
            "trades": 0,
            "win_rate": None,
            "average_win": None,
            "average_loss": None,
            "payoff_ratio": None,
            "expectancy_per_trade": None,
            "profit_factor": None,
            "average_trade_return": None,
            "total_fees": 0.0,
            "fees_pct_initial_capital": 0.0,
            "turnover": 0.0,
            "turnover_multiple": 0.0,
            "annualized_turnover_multiple": 0.0,
            "average_exposure": 0.0,
            "max_exposure": 0.0,
        }

    curve = (
        equity.sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last",
        )
        .copy()
    )

    values = pd.to_numeric(
        curve["equity"],
        errors="coerce",
    )
    good = (
        values.notna()
        & values.ge(0)
    )
    curve = curve.loc[
        good
    ].reset_index(
        drop=True
    )
    values = pd.to_numeric(
        curve["equity"],
        errors="coerce",
    ).astype(float)

    if curve.empty:
        raise RuntimeError(
            "Equity curve contains "
            "no usable observations."
        )

    ending_equity = float(
        values.iloc[-1]
    )
    net_profit = (
        ending_equity
        - float(
            initial_capital
        )
    )
    total_return = (
        ending_equity
        / float(
            initial_capital
        )
        - 1.0
    )

    start_date = pd.Timestamp(
        curve["date"].iloc[0]
    )
    end_date = pd.Timestamp(
        curve["date"].iloc[-1]
    )
    years = max(
        (
            end_date
            - start_date
        ).days
        / 365.25,
        1.0
        / 365.25,
    )

    if ending_equity > 0:
        cagr = (
            ending_equity
            / float(
                initial_capital
            )
        ) ** (
            1.0 / years
        ) - 1.0
    else:
        cagr = -1.0

    returns = _clean_returns(
        values
    )
    rf_daily = (
        _annual_rf_to_daily(
            annual_risk_free_rate
        )
    )

    annualized_arithmetic_return = (
        float(
            returns.mean()
        )
        * TRADING_DAYS_PER_YEAR
        if len(returns)
        else 0.0
    )

    if (
        len(returns) >= 2
        and float(
            returns.std(
                ddof=1
            )
        ) > 0
    ):
        daily_std = float(
            returns.std(
                ddof=1
            )
        )
        annualized_volatility = (
            daily_std
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
        sharpe = (
            (
                float(
                    returns.mean()
                )
                - rf_daily
            )
            / daily_std
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
    else:
        annualized_volatility = 0.0
        sharpe = 0.0

    if len(returns):
        excess = (
            returns
            - rf_daily
        )
        downside = np.minimum(
            excess.to_numpy(
                dtype=float
            ),
            0.0,
        )
        daily_downside = float(
            np.sqrt(
                np.mean(
                    downside
                    ** 2
                )
            )
        )
        downside_deviation = (
            daily_downside
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
        sortino = (
            (
                float(
                    excess.mean()
                )
                * TRADING_DAYS_PER_YEAR
            )
            / downside_deviation
            if downside_deviation
            > 0
            else None
        )

        var_95 = float(
            returns.quantile(
                0.05
            )
        )
        tail = returns.loc[
            returns <= var_95
        ]
        cvar_95 = (
            float(
                tail.mean()
            )
            if len(tail)
            else None
        )
        skewness = (
            float(
                returns.skew()
            )
            if len(returns) >= 3
            else None
        )
        excess_kurtosis = (
            float(
                returns.kurt()
            )
            if len(returns) >= 4
            else None
        )
        best_day_return = float(
            returns.max()
        )
        worst_day_return = float(
            returns.min()
        )
        positive_day_fraction = float(
            returns.gt(0).mean()
        )
    else:
        downside_deviation = 0.0
        sortino = None
        var_95 = None
        cvar_95 = None
        skewness = None
        excess_kurtosis = None
        best_day_return = None
        worst_day_return = None
        positive_day_fraction = None

    running_max = (
        values.cummax()
        .replace(
            0,
            np.nan,
        )
    )
    drawdown = (
        values
        / running_max
        - 1.0
    )
    max_drawdown = float(
        drawdown.min()
    )
    high_water_mark = float(
        values.max()
    )

    (
        max_dd_sessions,
        max_dd_days,
    ) = _drawdown_duration(
        curve["date"],
        values,
    )

    calmar = (
        float(cagr)
        / abs(
            max_drawdown
        )
        if max_drawdown
        < 0
        else None
    )

    max_drawdown_rupees = (
        high_water_mark
        * abs(
            max_drawdown
        )
    )
    recovery_factor = (
        net_profit
        / max_drawdown_rupees
        if max_drawdown_rupees
        > 0
        else None
    )

    if (
        trades is None
        or trades.empty
    ):
        trade_count = 0
        win_rate = None
        average_win = None
        average_loss = None
        payoff_ratio = None
        expectancy = None
        profit_factor = None
        average_trade_return = None
        total_fees = 0.0
        turnover = 0.0
    else:
        t = trades.copy()
        pnl = pd.to_numeric(
            t.get(
                "net_pnl"
            ),
            errors="coerce",
        ).dropna()

        trade_count = int(
            len(t)
        )
        win_rate = (
            float(
                pnl.gt(0).mean()
            )
            if len(pnl)
            else None
        )

        wins = pnl.loc[
            pnl > 0
        ]
        losses = pnl.loc[
            pnl < 0
        ]

        average_win = (
            float(
                wins.mean()
            )
            if len(wins)
            else None
        )
        average_loss = (
            float(
                losses.mean()
            )
            if len(losses)
            else None
        )
        payoff_ratio = (
            average_win
            / abs(
                average_loss
            )
            if (
                average_win
                is not None
                and average_loss
                is not None
                and average_loss
                != 0
            )
            else None
        )
        expectancy = (
            float(
                pnl.mean()
            )
            if len(pnl)
            else None
        )

        gross_profit = float(
            wins.sum()
        )
        gross_loss = abs(
            float(
                losses.sum()
            )
        )
        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else (
                float("inf")
                if gross_profit > 0
                else None
            )
        )

        net_return = pd.to_numeric(
            t.get(
                "net_return"
            ),
            errors="coerce",
        ).dropna()
        average_trade_return = (
            float(
                net_return.mean()
            )
            if len(net_return)
            else None
        )

        fee_series = pd.to_numeric(
            t.get(
                "total_fees"
            ),
            errors="coerce",
        ).fillna(0)
        total_fees = float(
            fee_series.sum()
        )

        entry_value = pd.to_numeric(
            t.get(
                "entry_trade_value"
            ),
            errors="coerce",
        ).fillna(0)
        exit_value = pd.to_numeric(
            t.get(
                "exit_trade_value"
            ),
            errors="coerce",
        ).fillna(0)
        turnover = float(
            (
                entry_value
                + exit_value
            ).sum()
        )

    turnover_multiple = (
        turnover
        / float(
            initial_capital
        )
        if initial_capital
        else 0.0
    )
    annualized_turnover_multiple = (
        turnover_multiple
        / years
    )
    fees_pct_initial_capital = (
        total_fees
        / float(
            initial_capital
        )
        if initial_capital
        else 0.0
    )

    if (
        "invested_market_value"
        in curve.columns
    ):
        invested = pd.to_numeric(
            curve[
                "invested_market_value"
            ],
            errors="coerce",
        )
        exposure = (
            invested
            / values.where(
                values > 0
            )
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        average_exposure = float(
            exposure.fillna(
                0
            ).mean()
        )
        max_exposure = float(
            exposure.fillna(
                0
            ).max()
        )
    else:
        average_exposure = 1.0
        max_exposure = 1.0

    return {
        "starting_capital": float(
            initial_capital
        ),
        "ending_equity": (
            ending_equity
        ),
        "net_profit": float(
            net_profit
        ),
        "total_return": float(
            total_return
        ),
        "cagr": float(
            cagr
        ),
        "annualized_arithmetic_return": float(
            annualized_arithmetic_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "risk_free_rate_annual": float(
            annual_risk_free_rate
        ),
        "sharpe": float(
            sharpe
        ),
        "downside_deviation": float(
            downside_deviation
        ),
        "sortino": (
            float(
                sortino
            )
            if sortino is not None
            else None
        ),
        "max_drawdown": float(
            max_drawdown
        ),
        "max_drawdown_duration_sessions": int(
            max_dd_sessions
        ),
        "max_drawdown_duration_days": int(
            max_dd_days
        ),
        "high_water_mark": float(
            high_water_mark
        ),
        "calmar": (
            float(
                calmar
            )
            if calmar is not None
            else None
        ),
        "recovery_factor": (
            float(
                recovery_factor
            )
            if recovery_factor
            is not None
            else None
        ),
        "var_95": (
            float(
                var_95
            )
            if var_95
            is not None
            else None
        ),
        "cvar_95": (
            float(
                cvar_95
            )
            if cvar_95
            is not None
            else None
        ),
        "skewness": (
            float(
                skewness
            )
            if skewness
            is not None
            else None
        ),
        "excess_kurtosis": (
            float(
                excess_kurtosis
            )
            if excess_kurtosis
            is not None
            else None
        ),
        "best_day_return": (
            float(
                best_day_return
            )
            if best_day_return
            is not None
            else None
        ),
        "worst_day_return": (
            float(
                worst_day_return
            )
            if worst_day_return
            is not None
            else None
        ),
        "positive_day_fraction": (
            float(
                positive_day_fraction
            )
            if positive_day_fraction
            is not None
            else None
        ),
        "trades": int(
            trade_count
        ),
        "win_rate": (
            float(
                win_rate
            )
            if win_rate is not None
            else None
        ),
        "average_win": (
            float(
                average_win
            )
            if average_win
            is not None
            else None
        ),
        "average_loss": (
            float(
                average_loss
            )
            if average_loss
            is not None
            else None
        ),
        "payoff_ratio": (
            float(
                payoff_ratio
            )
            if payoff_ratio
            is not None
            and math.isfinite(
                payoff_ratio
            )
            else payoff_ratio
        ),
        "expectancy_per_trade": (
            float(
                expectancy
            )
            if expectancy
            is not None
            else None
        ),
        "profit_factor": (
            float(
                profit_factor
            )
            if profit_factor
            is not None
            and math.isfinite(
                profit_factor
            )
            else profit_factor
        ),
        "average_trade_return": (
            float(
                average_trade_return
            )
            if average_trade_return
            is not None
            else None
        ),
        "total_fees": float(
            total_fees
        ),
        "fees_pct_initial_capital": float(
            fees_pct_initial_capital
        ),
        "turnover": float(
            turnover
        ),
        "turnover_multiple": float(
            turnover_multiple
        ),
        "annualized_turnover_multiple": float(
            annualized_turnover_multiple
        ),
        "average_exposure": float(
            average_exposure
        ),
        "max_exposure": float(
            max_exposure
        ),
    }


def benchmark_relative_metrics(
    strategy_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    *,
    strategy_value_col: str = "equity",
    benchmark_value_col: str = "equity",
    annual_risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    left = (
        strategy_curve[
            [
                "date",
                strategy_value_col,
            ]
        ]
        .rename(
            columns={
                strategy_value_col:
                "strategy_value",
            }
        )
        .copy()
    )
    right = (
        benchmark_curve[
            [
                "date",
                benchmark_value_col,
            ]
        ]
        .rename(
            columns={
                benchmark_value_col:
                "benchmark_value",
            }
        )
        .copy()
    )

    left["date"] = pd.to_datetime(
        left["date"],
        errors="coerce",
    ).dt.normalize()
    right["date"] = pd.to_datetime(
        right["date"],
        errors="coerce",
    ).dt.normalize()

    aligned = (
        left.merge(
            right,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .dropna()
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last",
        )
    )

    if len(aligned) < 3:
        return {
            "aligned_observations": int(
                len(aligned)
            ),
            "benchmark_total_return": None,
            "benchmark_cagr": None,
            "excess_cagr": None,
            "beta": None,
            "alpha_annualized": None,
            "tracking_error": None,
            "information_ratio": None,
            "benchmark_correlation": None,
        }

    sr = _clean_returns(
        aligned[
            "strategy_value"
        ]
    )
    br = _clean_returns(
        aligned[
            "benchmark_value"
        ]
    )
    returns = pd.concat(
        [
            sr.rename(
                "strategy"
            ),
            br.rename(
                "benchmark"
            ),
        ],
        axis=1,
    ).dropna()

    first_benchmark = float(
        aligned[
            "benchmark_value"
        ].iloc[0]
    )
    last_benchmark = float(
        aligned[
            "benchmark_value"
        ].iloc[-1]
    )
    benchmark_total_return = (
        last_benchmark
        / first_benchmark
        - 1.0
    )

    start_date = pd.Timestamp(
        aligned[
            "date"
        ].iloc[0]
    )
    end_date = pd.Timestamp(
        aligned[
            "date"
        ].iloc[-1]
    )
    years = max(
        (
            end_date
            - start_date
        ).days
        / 365.25,
        1.0
        / 365.25,
    )
    benchmark_cagr = (
        (
            last_benchmark
            / first_benchmark
        )
        ** (
            1.0 / years
        )
        - 1.0
    )

    first_strategy = float(
        aligned[
            "strategy_value"
        ].iloc[0]
    )
    last_strategy = float(
        aligned[
            "strategy_value"
        ].iloc[-1]
    )
    strategy_cagr = (
        (
            last_strategy
            / first_strategy
        )
        ** (
            1.0 / years
        )
        - 1.0
    )

    if len(returns) < 2:
        beta = None
        alpha = None
        tracking_error = None
        information_ratio = None
        correlation = None
    else:
        benchmark_variance = float(
            returns[
                "benchmark"
            ].var(
                ddof=1
            )
        )
        beta = (
            float(
                returns[
                    "strategy"
                ].cov(
                    returns[
                        "benchmark"
                    ]
                )
            )
            / benchmark_variance
            if benchmark_variance
            > 0
            else None
        )

        rf_daily = (
            _annual_rf_to_daily(
                annual_risk_free_rate
            )
        )
        strategy_excess = (
            returns[
                "strategy"
            ]
            - rf_daily
        )
        benchmark_excess = (
            returns[
                "benchmark"
            ]
            - rf_daily
        )

        alpha = (
            (
                float(
                    strategy_excess.mean()
                )
                - float(beta)
                * float(
                    benchmark_excess.mean()
                )
            )
            * TRADING_DAYS_PER_YEAR
            if beta is not None
            else None
        )

        active = (
            returns[
                "strategy"
            ]
            - returns[
                "benchmark"
            ]
        )
        active_std = float(
            active.std(
                ddof=1
            )
        )
        tracking_error = (
            active_std
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
        information_ratio = (
            float(
                active.mean()
            )
            / active_std
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
            if active_std
            > 0
            else None
        )
        correlation = float(
            returns[
                "strategy"
            ].corr(
                returns[
                    "benchmark"
                ]
            )
        )

    return {
        "aligned_observations": int(
            len(aligned)
        ),
        "benchmark_total_return": float(
            benchmark_total_return
        ),
        "benchmark_cagr": float(
            benchmark_cagr
        ),
        "excess_cagr": float(
            strategy_cagr
            - benchmark_cagr
        ),
        "beta": (
            float(beta)
            if beta is not None
            else None
        ),
        "alpha_annualized": (
            float(alpha)
            if alpha is not None
            else None
        ),
        "tracking_error": (
            float(
                tracking_error
            )
            if tracking_error
            is not None
            else None
        ),
        "information_ratio": (
            float(
                information_ratio
            )
            if information_ratio
            is not None
            else None
        ),
        "benchmark_correlation": (
            float(
                correlation
            )
            if correlation
            is not None
            else None
        ),
    }


def rolling_risk_metrics(
    curve: pd.DataFrame,
    *,
    value_col: str = "equity",
    windows: tuple[int, ...] = (
        126,
        252,
        504,
    ),
    annual_risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    data = (
        curve[
            [
                "date",
                value_col,
            ]
        ]
        .dropna()
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last",
        )
        .copy()
    )
    data[
        value_col
    ] = pd.to_numeric(
        data[
            value_col
        ],
        errors="coerce",
    )
    data = data.loc[
        data[
            value_col
        ].gt(0)
    ].copy()

    values = data[
        value_col
    ]
    returns = values.pct_change()
    rf_daily = (
        _annual_rf_to_daily(
            annual_risk_free_rate
        )
    )

    running_max = (
        values.cummax()
    )
    data[
        "drawdown"
    ] = (
        values
        / running_max
        - 1.0
    )

    for window in windows:
        mean = returns.rolling(
            window,
            min_periods=window,
        ).mean()
        std = returns.rolling(
            window,
            min_periods=window,
        ).std()

        data[
            f"rolling_sharpe_{window}d"
        ] = (
            (
                mean
                - rf_daily
            )
            / std.where(
                std > 0
            )
            * math.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )

        rolling_peak = (
            values.rolling(
                window,
                min_periods=1,
            ).max()
        )
        data[
            f"rolling_drawdown_{window}d"
        ] = (
            values
            / rolling_peak
            - 1.0
        )

    return data


def format_metric_value(
    metric: str,
    value: Any,
) -> str:
    if value is None or (
        isinstance(
            value,
            float,
        )
        and math.isnan(
            value
        )
    ):
        return "n/a"

    percentage_metrics = {
        "total_return",
        "cagr",
        "annualized_arithmetic_return",
        "annualized_volatility",
        "risk_free_rate_annual",
        "downside_deviation",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "best_day_return",
        "worst_day_return",
        "positive_day_fraction",
        "win_rate",
        "average_trade_return",
        "fees_pct_initial_capital",
        "average_exposure",
        "max_exposure",
        "benchmark_total_return",
        "benchmark_cagr",
        "excess_cagr",
        "alpha_annualized",
        "tracking_error",
        "mean_top_decile_excess",
    }

    money_metrics = {
        "starting_capital",
        "ending_equity",
        "net_profit",
        "high_water_mark",
        "average_win",
        "average_loss",
        "expectancy_per_trade",
        "total_fees",
        "turnover",
    }

    if metric in percentage_metrics:
        return (
            f"{float(value) * 100.0:.2f}%"
        )

    if metric in money_metrics:
        return (
            f"₹{float(value):,.2f}"
        )

    if metric in {
        "trades",
        "max_drawdown_duration_sessions",
        "max_drawdown_duration_days",
        "aligned_observations",
    }:
        return f"{int(value):,}"

    return f"{float(value):.3f}"


def captioned_metric_rows(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for metric, value in (
        metrics.items()
    ):
        glossary = (
            METRIC_GLOSSARY.get(
                metric,
                {},
            )
        )

        rows.append({
            "metric": metric,
            "label": (
                glossary.get(
                    "label",
                    metric,
                )
            ),
            "value": value,
            "formatted_value": (
                format_metric_value(
                    metric,
                    value,
                )
            ),
            "caption": (
                glossary.get(
                    "caption",
                    ""
                )
            ),
            "category": (
                glossary.get(
                    "category",
                    "other",
                )
            ),
        })

    return pd.DataFrame(
        rows
    )
