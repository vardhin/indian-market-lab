from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from baselines import (
    CostProfile,
    load_mechanical_events,
    load_panel,
    run_backtest,
)


FROZEN_BASELINES = [
    {
        "name": "liquidity_control_h20_k5",
        "strategy": "liquidity_control",
        "holding_sessions": 20,
        "top_k": 5,
    },
    {
        "name": "momentum_20d_h20_k5",
        "strategy": "momentum_20d",
        "holding_sessions": 20,
        "top_k": 5,
    },
    {
        "name": "momentum_60d_h20_k5",
        "strategy": "momentum_60d",
        "holding_sessions": 20,
        "top_k": 5,
    },
    {
        "name": "momentum_60d_h20_k10",
        "strategy": "momentum_60d",
        "holding_sessions": 20,
        "top_k": 10,
    },
    {
        "name": "risk_adjusted_momentum_60d_h20_k10",
        "strategy": "risk_adjusted_momentum_60d",
        "holding_sessions": 20,
        "top_k": 10,
    },
]


def curve_metrics(
    curve: pd.DataFrame,
    *,
    name: str,
    kind: str,
    value_col: str = "equity",
) -> dict:
    curve = (
        curve.loc[
            curve["date"].notna()
            & curve[value_col].notna()
        ]
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .copy()
    )

    if len(curve) < 2:
        raise RuntimeError(
            f"{name}: insufficient curve rows."
        )

    values = pd.to_numeric(
        curve[value_col],
        errors="coerce",
    )

    good = (
        values.notna()
        & values.gt(0)
    )

    curve = curve.loc[
        good
    ].copy()
    values = values.loc[
        good
    ]

    if len(curve) < 2:
        raise RuntimeError(
            f"{name}: insufficient positive values."
        )

    start_value = float(
        values.iloc[0]
    )
    end_value = float(
        values.iloc[-1]
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
        1.0 / 365.25,
    )

    total_return = (
        end_value
        / start_value
        - 1.0
    )
    cagr = (
        end_value
        / start_value
    ) ** (
        1.0 / years
    ) - 1.0

    returns = (
        values.pct_change()
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
        len(returns) >= 2
        and float(
            returns.std()
        ) > 0
    ):
        annualized_volatility = (
            float(
                returns.std()
            )
            * math.sqrt(252.0)
        )
        sharpe = (
            float(
                returns.mean()
            )
            / float(
                returns.std()
            )
            * math.sqrt(252.0)
        )
    else:
        annualized_volatility = 0.0
        sharpe = 0.0

    running_max = (
        values.cummax()
    )
    drawdown = (
        values
        / running_max
        - 1.0
    )
    max_drawdown = float(
        drawdown.min()
    )

    calmar = (
        cagr
        / abs(
            max_drawdown
        )
        if max_drawdown < 0
        else None
    )

    return {
        "name": name,
        "kind": kind,
        "date_start": str(
            start_date.date()
        ),
        "date_end": str(
            end_date.date()
        ),
        "observations": int(
            len(curve)
        ),
        "start_value": start_value,
        "end_value": end_value,
        "total_return": float(
            total_return
        ),
        "cagr": float(
            cagr
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe": float(
            sharpe
        ),
        "max_drawdown": float(
            max_drawdown
        ),
        "calmar": (
            float(calmar)
            if calmar is not None
            else None
        ),
    }


def index_curve(
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    allowed_dates: set[
        pd.Timestamp
    ],
) -> pd.DataFrame:
    frame = frame.copy()

    frame["date"] = (
        pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.normalize()
    )
    frame["close"] = (
        pd.to_numeric(
            frame["close"],
            errors="coerce",
        )
    )

    frame = (
        frame.loc[
            frame["date"].isin(
                allowed_dates
            )
            & frame["close"].gt(0)
        ]
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if frame.empty:
        raise RuntimeError(
            "Benchmark has no dates in "
            "the research-panel period."
        )

    first_close = float(
        frame["close"].iloc[0]
    )

    frame["equity"] = (
        initial_capital
        * frame["close"]
        / first_close
    )

    return frame[
        [
            "date",
            "equity",
            "close",
        ]
    ].copy()


def calendar_returns(
    curve: pd.DataFrame,
    *,
    name: str,
    value_col: str = "equity",
) -> pd.DataFrame:
    d = (
        curve[
            [
                "date",
                value_col,
            ]
        ]
        .dropna()
        .sort_values("date")
        .copy()
    )

    d["year"] = (
        pd.to_datetime(
            d["date"]
        ).dt.year
    )

    rows: list[
        dict
    ] = []

    for year, group in d.groupby(
        "year",
        sort=True,
    ):
        if len(group) < 2:
            continue

        start = float(
            group[
                value_col
            ].iloc[0]
        )
        end = float(
            group[
                value_col
            ].iloc[-1]
        )

        if start <= 0:
            continue

        rows.append({
            "name": name,
            "year": int(year),
            "return": (
                end / start - 1.0
            ),
            "first_date": str(
                pd.Timestamp(
                    group[
                        "date"
                    ].iloc[0]
                ).date()
            ),
            "last_date": str(
                pd.Timestamp(
                    group[
                        "date"
                    ].iloc[-1]
                ).date()
            ),
        })

    return pd.DataFrame(
        rows
    )


def rolling_return_summary(
    curve: pd.DataFrame,
    *,
    name: str,
    value_col: str = "equity",
) -> list[dict]:
    values = (
        curve[
            [
                "date",
                value_col,
            ]
        ]
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    rows: list[
        dict
    ] = []

    for sessions, label in (
        (
            252,
            "1y",
        ),
        (
            756,
            "3y",
        ),
    ):
        lag = values[
            value_col
        ].shift(sessions)

        returns = (
            values[value_col]
            / lag.where(
                lag > 0
            )
            - 1.0
        ).dropna()

        if returns.empty:
            continue

        rows.append({
            "name": name,
            "window": label,
            "sessions": sessions,
            "observations": int(
                len(returns)
            ),
            "mean": float(
                returns.mean()
            ),
            "median": float(
                returns.median()
            ),
            "minimum": float(
                returns.min()
            ),
            "maximum": float(
                returns.max()
            ),
            "positive_fraction": float(
                returns.gt(0).mean()
            ),
        })

    return rows


def drawdown_episodes(
    curve: pd.DataFrame,
    *,
    name: str,
    value_col: str = "equity",
    top_n: int = 5,
) -> pd.DataFrame:
    d = (
        curve[
            [
                "date",
                value_col,
            ]
        ]
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    values = pd.to_numeric(
        d[value_col],
        errors="coerce",
    )

    running_max = (
        values.cummax()
    )
    drawdown = (
        values
        / running_max
        - 1.0
    )

    in_drawdown = False
    start_idx = 0
    trough_idx = 0
    trough_dd = 0.0

    episodes: list[
        dict
    ] = []

    for i, dd in enumerate(
        drawdown
    ):
        dd = float(dd)

        if (
            dd < 0
            and not in_drawdown
        ):
            in_drawdown = True
            start_idx = max(
                0,
                i - 1,
            )
            trough_idx = i
            trough_dd = dd

        elif (
            in_drawdown
            and dd < trough_dd
        ):
            trough_idx = i
            trough_dd = dd

        if (
            in_drawdown
            and dd >= -1e-12
        ):
            episodes.append({
                "name": name,
                "peak_date": str(
                    pd.Timestamp(
                        d.loc[
                            start_idx,
                            "date",
                        ]
                    ).date()
                ),
                "trough_date": str(
                    pd.Timestamp(
                        d.loc[
                            trough_idx,
                            "date",
                        ]
                    ).date()
                ),
                "recovery_date": str(
                    pd.Timestamp(
                        d.loc[
                            i,
                            "date",
                        ]
                    ).date()
                ),
                "max_drawdown": float(
                    trough_dd
                ),
                "recovered": True,
            })

            in_drawdown = False

    if in_drawdown:
        episodes.append({
            "name": name,
            "peak_date": str(
                pd.Timestamp(
                    d.loc[
                        start_idx,
                        "date",
                    ]
                ).date()
            ),
            "trough_date": str(
                pd.Timestamp(
                    d.loc[
                        trough_idx,
                        "date",
                    ]
                ).date()
            ),
            "recovery_date": None,
            "max_drawdown": float(
                trough_dd
            ),
            "recovered": False,
        })

    if not episodes:
        return pd.DataFrame(
            columns=[
                "name",
                "peak_date",
                "trough_date",
                "recovery_date",
                "max_drawdown",
                "recovered",
            ]
        )

    return (
        pd.DataFrame(
            episodes
        )
        .sort_values(
            "max_drawdown",
            ascending=True,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare frozen deterministic "
            "strategy baselines against "
            "official NSE NIFTY 50 and "
            "NIFTY 500 price indices."
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

    root = Path(
        args.root
    ).resolve()

    index_path = (
        root
        / "data/processed/index_benchmarks/"
        "nse_price_indices.parquet"
    )

    if not index_path.is_file():
        raise SystemExit(
            "Index benchmark dataset "
            "not found. Run:\n"
            "  uv run python "
            "src/data/index_benchmarks.py"
        )

    print(
        "Loading h20 research panel..."
    )
    panel = load_panel(
        root
        / "data/processed/research_panel",
        holding_sessions=20,
    )
    events = load_mechanical_events(
        root
    )

    panel_dates = set(
        pd.to_datetime(
            panel["date"]
        ).dt.normalize()
    )

    index_data = (
        pd.read_parquet(
            index_path
        )
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

    output_root = (
        root
        / "reports/benchmarks"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_rows: list[
        dict
    ] = []
    calendar_frames: list[
        pd.DataFrame
    ] = []
    rolling_rows: list[
        dict
    ] = []
    drawdown_frames: list[
        pd.DataFrame
    ] = []

    strategy_curves: dict[
        str,
        pd.DataFrame,
    ] = {}

    print(
        "\nRunning frozen deterministic "
        "baselines..."
    )

    for number, spec in enumerate(
        FROZEN_BASELINES,
        start=1,
    ):
        print(
            f"[{number}/"
            f"{len(FROZEN_BASELINES)}] "
            f"{spec['name']}"
        )

        result = run_backtest(
            panel,
            events,
            strategy=spec[
                "strategy"
            ],
            initial_capital=(
                args.capital
            ),
            top_k=spec["top_k"],
            holding_sessions=(
                spec[
                    "holding_sessions"
                ]
            ),
            costs=costs,
        )

        curve = (
            result["equity"]
            [
                [
                    "date",
                    "equity",
                ]
            ]
            .copy()
        )

        strategy_curves[
            spec["name"]
        ] = curve

        metric = curve_metrics(
            curve,
            name=spec["name"],
            kind="strategy",
        )

        metric.update({
            "strategy": spec[
                "strategy"
            ],
            "holding_sessions": (
                spec[
                    "holding_sessions"
                ]
            ),
            "top_k": spec[
                "top_k"
            ],
            "trades": result[
                "metrics"
            ]["trades"],
            "total_fees": result[
                "metrics"
            ]["total_fees"],
            "average_exposure": (
                result["metrics"][
                    "average_exposure"
                ]
            ),
        })

        metrics_rows.append(
            metric
        )
        calendar_frames.append(
            calendar_returns(
                curve,
                name=spec["name"],
            )
        )
        rolling_rows.extend(
            rolling_return_summary(
                curve,
                name=spec["name"],
            )
        )
        drawdown_frames.append(
            drawdown_episodes(
                curve,
                name=spec["name"],
            )
        )

    print(
        "\nBuilding market benchmark "
        "curves..."
    )

    index_curves: dict[
        str,
        pd.DataFrame,
    ] = {}

    for index_name in (
        "NIFTY 50",
        "NIFTY 500",
    ):
        source = index_data.loc[
            index_data[
                "requested_index"
            ].eq(index_name)
        ].copy()

        if source.empty:
            raise RuntimeError(
                f"Missing {index_name} "
                "from index benchmark dataset."
            )

        curve = index_curve(
            source,
            initial_capital=(
                args.capital
            ),
            allowed_dates=(
                panel_dates
            ),
        )

        index_curves[
            index_name
        ] = curve

        metric = curve_metrics(
            curve,
            name=index_name,
            kind="price_index",
        )
        metric.update({
            "strategy": None,
            "holding_sessions": None,
            "top_k": None,
            "trades": None,
            "total_fees": 0.0,
            "average_exposure": 1.0,
        })

        metrics_rows.append(
            metric
        )
        calendar_frames.append(
            calendar_returns(
                curve,
                name=index_name,
            )
        )
        rolling_rows.extend(
            rolling_return_summary(
                curve,
                name=index_name,
            )
        )
        drawdown_frames.append(
            drawdown_episodes(
                curve,
                name=index_name,
            )
        )

    cash_curve = pd.DataFrame({
        "date": sorted(
            panel_dates
        ),
        "equity": (
            args.capital
        ),
    })

    cash_metric = curve_metrics(
        cash_curve,
        name="CASH_0PCT",
        kind="cash",
    )
    cash_metric.update({
        "strategy": None,
        "holding_sessions": None,
        "top_k": None,
        "trades": 0,
        "total_fees": 0.0,
        "average_exposure": 0.0,
    })
    metrics_rows.append(
        cash_metric
    )
    calendar_frames.append(
        calendar_returns(
            cash_curve,
            name="CASH_0PCT",
        )
    )
    rolling_rows.extend(
        rolling_return_summary(
            cash_curve,
            name="CASH_0PCT",
        )
    )

    metrics = pd.DataFrame(
        metrics_rows
    )

    benchmark_cagrs = {
        row["name"]: row["cagr"]
        for row in metrics_rows
        if row["name"] in {
            "NIFTY 50",
            "NIFTY 500",
        }
    }

    for benchmark_name, cagr in (
        benchmark_cagrs.items()
    ):
        col = (
            "cagr_excess_vs_"
            + benchmark_name.lower()
            .replace(" ", "_")
        )
        metrics[col] = (
            metrics["cagr"]
            - float(cagr)
        )

    metrics_path = (
        output_root
        / "benchmark_comparison.csv"
    )

    metrics.sort_values(
        [
            "cagr",
            "sharpe",
        ],
        ascending=[
            False,
            False,
        ],
    ).to_csv(
        metrics_path,
        index=False,
    )

    calendar = pd.concat(
        [
            x
            for x in calendar_frames
            if not x.empty
        ],
        ignore_index=True,
    )

    calendar_pivot = (
        calendar.pivot(
            index="year",
            columns="name",
            values="return",
        )
        * 100.0
    )

    calendar_path = (
        output_root
        / "calendar_year_returns_percent.csv"
    )
    calendar_pivot.to_csv(
        calendar_path
    )

    rolling = pd.DataFrame(
        rolling_rows
    )
    rolling_path = (
        output_root
        / "rolling_return_summary.csv"
    )
    rolling.to_csv(
        rolling_path,
        index=False,
    )

    drawdowns = pd.concat(
        [
            x
            for x in drawdown_frames
            if not x.empty
        ],
        ignore_index=True,
    )
    drawdown_path = (
        output_root
        / "largest_drawdowns.csv"
    )
    drawdowns.to_csv(
        drawdown_path,
        index=False,
    )

    # One aligned equity table makes plotting and later report generation easy.
    curves: list[
        pd.DataFrame
    ] = []

    for name, curve in {
        **strategy_curves,
        **index_curves,
        "CASH_0PCT": (
            cash_curve
        ),
    }.items():
        x = curve[
            [
                "date",
                "equity",
            ]
        ].copy()
        x = x.rename(
            columns={
                "equity": name
            }
        )
        curves.append(x)

    aligned = curves[0]

    for curve in curves[1:]:
        aligned = aligned.merge(
            curve,
            on="date",
            how="outer",
            validate="one_to_one",
        )

    aligned = aligned.sort_values(
        "date"
    )

    curves_path = (
        output_root
        / "aligned_equity_curves.csv"
    )
    aligned.to_csv(
        curves_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "benchmark_return_type": (
            "price_index"
        ),
        "benchmark_dividends_included": False,
        "strategy_dividends_included": False,
        "reason_for_price_index": (
            "Current stock backtester does not yet credit cash dividends, "
            "so price-index benchmarks preserve like-for-like return treatment."
        ),
        "initial_capital": float(
            args.capital
        ),
        "strategy_cost_profile": (
            "current_2026_delivery"
        ),
        "strategy_cost_parameters": (
            asdict(costs)
        ),
        "frozen_baselines": (
            FROZEN_BASELINES
        ),
        "outputs": {
            "comparison": str(
                metrics_path
            ),
            "calendar_year_returns": str(
                calendar_path
            ),
            "rolling_returns": str(
                rolling_path
            ),
            "largest_drawdowns": str(
                drawdown_path
            ),
            "aligned_equity_curves": str(
                curves_path
            ),
        },
    }

    summary_path = (
        output_root
        / "benchmark_report_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    preview = (
        metrics.sort_values(
            [
                "cagr",
                "sharpe",
            ],
            ascending=[
                False,
                False,
            ],
        )
        [
            [
                "name",
                "kind",
                "end_value",
                "cagr",
                "max_drawdown",
                "sharpe",
                "calmar",
            ]
        ]
        .copy()
    )

    preview["cagr"] *= 100.0
    preview[
        "max_drawdown"
    ] *= 100.0

    print(
        "\n=== BENCHMARK REPORT COMPLETE ==="
    )
    print(
        preview.to_string(
            index=False,
            formatters={
                "end_value": (
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
                "calmar": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
            },
        )
    )
    print(
        f"\nComparison: {metrics_path}"
    )
    print(
        f"Calendar:   {calendar_path}"
    )
    print(
        f"Rolling:    {rolling_path}"
    )
    print(
        f"Drawdowns:  {drawdown_path}"
    )
    print(
        f"Curves:     {curves_path}"
    )
    print(
        f"Summary:    {summary_path}"
    )


if __name__ == "__main__":
    main()
