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

from quant_metrics import (
    benchmark_relative_metrics,
    captioned_metric_rows,
    compute_performance_metrics,
    glossary_frame,
    rolling_risk_metrics,
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
    annual_risk_free_rate: float = 0.0,
) -> dict:
    clean = (
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

    clean[value_col] = (
        pd.to_numeric(
            clean[value_col],
            errors="coerce",
        )
    )
    clean = clean.loc[
        clean[value_col].gt(0)
    ].copy()

    if len(clean) < 2:
        raise RuntimeError(
            f"{name}: insufficient curve rows."
        )

    first_value = float(
        clean[value_col].iloc[0]
    )
    synthetic = clean[
        [
            "date",
            value_col,
        ]
    ].rename(
        columns={
            value_col: "equity"
        }
    )

    metrics = compute_performance_metrics(
        synthetic,
        pd.DataFrame(),
        initial_capital=first_value,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
    )

    return {
        "name": name,
        "kind": kind,
        "date_start": str(
            pd.Timestamp(
                clean["date"].iloc[0]
            ).date()
        ),
        "date_end": str(
            pd.Timestamp(
                clean["date"].iloc[-1]
            ).date()
        ),
        "observations": int(
            len(clean)
        ),
        "start_value": float(
            metrics["starting_capital"]
        ),
        "end_value": float(
            metrics["ending_equity"]
        ),
        **metrics,
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
    """
    Calendar returns use previous year-end -> current year-end whenever the
    previous year exists in the evaluation window. The first (partial) year is
    reported from the evaluation start and explicitly marked partial.
    """
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
        .copy()
    )

    d["year"] = (
        pd.to_datetime(
            d["date"]
        ).dt.year
    )

    yearly = (
        d.groupby(
            "year",
            sort=True,
        )
        .agg(
            first_date=(
                "date",
                "first",
            ),
            last_date=(
                "date",
                "last",
            ),
            first_value=(
                value_col,
                "first",
            ),
            last_value=(
                value_col,
                "last",
            ),
        )
        .reset_index()
    )

    rows: list[dict] = []
    previous_last: float | None = None

    for row in yearly.itertuples(
        index=False
    ):
        if previous_last is None:
            start_value = float(
                row.first_value
            )
            partial_year = True
        else:
            start_value = float(
                previous_last
            )
            partial_year = False

        end_value = float(
            row.last_value
        )

        if start_value <= 0:
            previous_last = (
                end_value
            )
            continue

        rows.append({
            "name": name,
            "year": int(
                row.year
            ),
            "return": (
                end_value
                / start_value
                - 1.0
            ),
            "partial_year": (
                partial_year
            ),
            "first_date": str(
                pd.Timestamp(
                    row.first_date
                ).date()
            ),
            "last_date": str(
                pd.Timestamp(
                    row.last_date
                ).date()
            ),
        })

        previous_last = (
            end_value
        )

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


def self_test() -> None:
    dates = pd.date_range(
        "2025-01-01",
        periods=800,
        freq="B",
    )
    equity = pd.Series(
        [
            50_000.0
            * (1.0005 ** i)
            for i in range(
                len(dates)
            )
        ]
    )

    curve = pd.DataFrame({
        "date": dates,
        "equity": equity,
    })

    metrics = curve_metrics(
        curve,
        name="TEST",
        kind="synthetic",
    )

    assert (
        metrics["end_value"]
        > metrics["start_value"]
    )
    assert metrics["cagr"] > 0
    assert (
        metrics[
            "max_drawdown"
        ]
        >= -1e-12
    )

    yearly = calendar_returns(
        curve,
        name="TEST",
    )

    assert not yearly.empty
    assert bool(
        yearly.iloc[0][
            "partial_year"
        ]
    )

    rolling = (
        rolling_return_summary(
            curve,
            name="TEST",
        )
    )

    windows = {
        row["window"]
        for row in rolling
    }

    assert "1y" in windows
    assert "3y" in windows

    for required in (
        "sortino",
        "calmar",
        "var_95",
        "cvar_95",
        "max_drawdown_duration_sessions",
    ):
        if required not in metrics:
            raise AssertionError(
                "Missing enriched curve metric: "
                f"{required}"
            )

    print(
        "Benchmark report "
        "self-test: PASS"
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

    eligible_dates = (
        panel.loc[
            panel[
                "eligible_universe"
            ].fillna(False),
            "date",
        ]
        .dropna()
        .sort_values()
    )

    if eligible_dates.empty:
        raise RuntimeError(
            "Research panel contains no "
            "point-in-time eligible rows."
        )

    evaluation_start = (
        pd.Timestamp(
            eligible_dates.iloc[0]
        ).normalize()
    )
    evaluation_end = (
        pd.Timestamp(
            panel["date"].max()
        ).normalize()
    )

    # Exclude the feature warm-up period from EVERY comparator. Before this
    # date the strategy could not legally emit a signal because the historical
    # feature window was not yet available.
    panel = panel.loc[
        panel["date"].between(
            evaluation_start,
            evaluation_end,
            inclusive="both",
        )
    ].copy()

    panel_dates = set(
        pd.to_datetime(
            panel["date"]
        ).dt.normalize()
    )

    print(
        "Evaluation window: "
        f"{evaluation_start.date()} "
        "-> "
        f"{evaluation_end.date()}"
    )

    index_data = (
        pd.read_parquet(
            index_path
        )
    )

    research_dates = pd.Index(
        sorted(
            panel_dates
        )
    )

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

        source_dates = pd.Index(
            pd.to_datetime(
                source["date"],
                errors="coerce",
            )
            .dropna()
            .dt.normalize()
            .unique()
        )

        overlap = research_dates.intersection(
            source_dates
        )

        coverage = (
            len(overlap)
            / len(research_dates)
            if len(research_dates)
            else 0.0
        )

        latest = pd.Timestamp(
            source_dates.max()
        ).normalize()

        end_gap_days = int(
            (
                evaluation_end
                - latest
            ).days
        )

        if (
            coverage < 0.98
            or end_gap_days > 10
        ):
            raise RuntimeError(
                f"{index_name} benchmark coverage is incomplete: "
                f"{len(overlap):,}/{len(research_dates):,} "
                f"research dates ({coverage:.1%}), "
                f"latest index date={latest.date()}, "
                f"evaluation end={evaluation_end.date()}. "
                "Re-run src/data/index_benchmarks.py with the current "
                "small-window downloader before generating this report."
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
    rolling_risk_frames: list[
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

        metric.update(
            result["metrics"]
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
        rolling_risk = (
            rolling_risk_metrics(
                curve
            )
        )
        rolling_risk[
            "name"
        ] = spec["name"]
        rolling_risk_frames.append(
            rolling_risk
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
        rolling_risk = (
            rolling_risk_metrics(
                curve
            )
        )
        rolling_risk[
            "name"
        ] = index_name
        rolling_risk_frames.append(
            rolling_risk
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
    cash_rolling_risk = (
        rolling_risk_metrics(
            cash_curve
        )
    )
    cash_rolling_risk[
        "name"
    ] = "CASH_0PCT"
    rolling_risk_frames.append(
        cash_rolling_risk
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

    rolling_risk = pd.concat(
        rolling_risk_frames,
        ignore_index=True,
    )
    rolling_risk_path = (
        output_root
        / "rolling_risk_metrics.csv"
    )
    rolling_risk.to_csv(
        rolling_risk_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    glossary_path = (
        output_root
        / "metric_glossary.csv"
    )
    glossary_frame().to_csv(
        glossary_path,
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

    relative_rows: list[
        dict
    ] = []

    for strategy_name, strategy_curve in (
        strategy_curves.items()
    ):
        for benchmark_name in (
            "NIFTY 50",
            "NIFTY 500",
        ):
            relative = (
                benchmark_relative_metrics(
                    strategy_curve,
                    index_curves[
                        benchmark_name
                    ],
                )
            )
            relative_rows.append({
                "strategy": strategy_name,
                "benchmark": benchmark_name,
                **relative,
            })

    benchmark_relative = pd.DataFrame(
        relative_rows
    )
    benchmark_relative_path = (
        output_root
        / "benchmark_relative_metrics.csv"
    )
    benchmark_relative.to_csv(
        benchmark_relative_path,
        index=False,
    )

    dashboard_rows: list[
        pd.DataFrame
    ] = []
    for row in (
        metrics.to_dict(
            orient="records"
        )
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
        dashboard_rows.append(
            captioned
        )

    captioned_dashboard = pd.concat(
        dashboard_rows,
        ignore_index=True,
    )
    captioned_dashboard_path = (
        output_root
        / "captioned_metrics_long.csv"
    )
    captioned_dashboard.to_csv(
        captioned_dashboard_path,
        index=False,
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
        "evaluation_start": str(
            evaluation_start.date()
        ),
        "evaluation_end": str(
            evaluation_end.date()
        ),
        "evaluation_start_policy": (
            "first date with any point-in-time eligible security; "
            "feature warm-up period excluded from all comparators"
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
            "rolling_risk_metrics": str(
                rolling_risk_path
            ),
            "benchmark_relative_metrics": str(
                benchmark_relative_path
            ),
            "metric_glossary": str(
                glossary_path
            ),
            "captioned_metrics": str(
                captioned_dashboard_path
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
        "\n=== CORE METRIC JARGON ==="
    )
    for metric in (
        "cagr",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "max_drawdown_duration_sessions",
        "turnover_multiple",
        "profit_factor",
        "expectancy_per_trade",
        "beta",
        "alpha_annualized",
        "information_ratio",
        "tracking_error",
    ):
        info = (
            glossary_frame()
            .loc[
                lambda x: x[
                    "metric"
                ].eq(metric)
            ]
        )
        if info.empty:
            continue
        item = info.iloc[0]
        print(
            f"{item['label']}: "
            f"{item['caption']}"
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
        f"Risk roll:  {rolling_risk_path}"
    )
    print(
        f"Relative:   {benchmark_relative_path}"
    )
    print(
        f"Glossary:   {glossary_path}"
    )
    print(
        f"Captioned:  {captioned_dashboard_path}"
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
