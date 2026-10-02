from __future__ import annotations

import argparse
import json
import math
import sys
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
    load_mechanical_events,
)
from classical import (  # noqa: E402
    add_target_rank,
    load_ml_panel,
)
from nested_generalization import (  # noqa: E402
    annual_metrics_from_result,
    benchmark_rows,
    build_annual_oos_predictions,
    first_aligned_signal_date,
    fixed_schedule,
    run_persistent_strategy,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
)
from quant_metrics import (  # noqa: E402
    benchmark_relative_metrics,
)


DEFAULT_MODELS = [
    "hist_gb",
    "hist_gb_fixed",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
]
DEFAULT_TOP_K = [
    5,
    10,
    15,
    20,
]
TREE_CONSENSUS = [
    "hist_gb_fixed",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
]


def cost_profile(
    *,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> CostProfile:
    return CostProfile(
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


def complete_year_summary(
    annual: pd.DataFrame,
) -> dict:
    if annual.empty:
        return {
            "complete_years": 0,
            "median_annual_cagr": None,
            "q25_annual_cagr": None,
            "positive_cagr_fraction": None,
            "median_annual_sharpe": None,
            "q25_annual_sharpe": None,
            "positive_sharpe_fraction": None,
            "worst_year_drawdown": None,
        }

    complete = annual.loc[
        ~annual[
            "partial_year"
        ].fillna(False)
    ].copy()

    if complete.empty:
        return {
            "complete_years": 0,
            "median_annual_cagr": None,
            "q25_annual_cagr": None,
            "positive_cagr_fraction": None,
            "median_annual_sharpe": None,
            "q25_annual_sharpe": None,
            "positive_sharpe_fraction": None,
            "worst_year_drawdown": None,
        }

    cagr = pd.to_numeric(
        complete["cagr"],
        errors="coerce",
    ).dropna()
    sharpe = pd.to_numeric(
        complete["sharpe"],
        errors="coerce",
    ).dropna()
    drawdown = pd.to_numeric(
        complete["max_drawdown"],
        errors="coerce",
    ).dropna()

    return {
        "complete_years": int(
            complete[
                "year"
            ].nunique()
        ),
        "median_annual_cagr": (
            float(
                cagr.median()
            )
            if len(cagr)
            else None
        ),
        "q25_annual_cagr": (
            float(
                cagr.quantile(
                    0.25
                )
            )
            if len(cagr)
            else None
        ),
        "positive_cagr_fraction": (
            float(
                cagr.gt(0).mean()
            )
            if len(cagr)
            else None
        ),
        "median_annual_sharpe": (
            float(
                sharpe.median()
            )
            if len(sharpe)
            else None
        ),
        "q25_annual_sharpe": (
            float(
                sharpe.quantile(
                    0.25
                )
            )
            if len(sharpe)
            else None
        ),
        "positive_sharpe_fraction": (
            float(
                sharpe.gt(0).mean()
            )
            if len(sharpe)
            else None
        ),
        "worst_year_drawdown": (
            float(
                drawdown.min()
            )
            if len(drawdown)
            else None
        ),
    }


def fixed_policy(
    *,
    top_k: int,
    years: list[int],
) -> dict[int, dict]:
    return fixed_schedule(
        years=years,
        top_k=top_k,
        multiplier=1,
        gap=0.0,
    )


def evaluate_scores(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    model_name: str,
    eval_start_date: pd.Timestamp,
    eval_end_year: int,
    top_k_values: list[int],
    initial_capital: float,
    costs: CostProfile,
    annual_risk_free_rate: float,
    benchmark_curve: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    panel = (
        df.loc[
            df["date"].between(
                eval_start_date,
                pd.Timestamp(
                    f"{eval_end_year}-12-31"
                ),
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    years = list(
        range(
            int(
                eval_start_date.year
            ),
            eval_end_year + 1,
        )
    )

    leaderboard_rows: list[
        dict
    ] = []
    annual_frames: list[
        pd.DataFrame
    ] = []
    equity_frames: list[
        pd.DataFrame
    ] = []

    for top_k in top_k_values:
        result = (
            run_persistent_strategy(
                panel,
                events,
                name=(
                    f"tournament_"
                    f"{model_name}_k{top_k}"
                ),
                capital=(
                    initial_capital
                ),
                top_k=int(
                    top_k
                ),
                schedule=(
                    fixed_policy(
                        top_k=int(
                            top_k
                        ),
                        years=years,
                    )
                ),
                costs=costs,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
            )
        )

        relative = (
            benchmark_relative_metrics(
                result[
                    "equity"
                ],
                benchmark_curve,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
            )
        )

        annual = (
            annual_metrics_from_result(
                result,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                metadata={
                    "model_name": (
                        model_name
                    ),
                    "top_k": int(
                        top_k
                    ),
                },
            )
        )
        annual_frames.append(
            annual
        )

        stability = (
            complete_year_summary(
                annual
            )
        )

        metrics = dict(
            result[
                "metrics"
            ]
        )
        excess_cagr = relative.get(
            "excess_cagr"
        )
        alpha = relative.get(
            "alpha_annualized"
        )
        sharpe = metrics.get(
            "sharpe"
        )

        economic_target = bool(
            excess_cagr is not None
            and sharpe is not None
            and float(
                excess_cagr
            )
            >= 0.05
            and float(
                sharpe
            )
            > 1.0
        )
        alpha_target = bool(
            alpha is not None
            and sharpe is not None
            and float(
                alpha
            )
            >= 0.05
            and float(
                sharpe
            )
            > 1.0
        )

        leaderboard_rows.append({
            "model_name": (
                model_name
            ),
            "top_k": int(
                top_k
            ),
            **metrics,
            **relative,
            **stability,
            "economic_target_excess5_sharpe1": (
                economic_target
            ),
            "factor_alpha5_sharpe1": (
                alpha_target
            ),
        })

        curve = (
            result[
                "equity"
            ][
                [
                    "date",
                    "equity",
                    "cash",
                    "invested_market_value",
                    "open_positions",
                ]
            ]
            .copy()
        )
        curve[
            "model_name"
        ] = model_name
        curve[
            "top_k"
        ] = int(
            top_k
        )
        equity_frames.append(
            curve
        )

        print(
            f"  k={int(top_k):2d} "
            f"CAGR={metrics['cagr']:.2%} "
            f"excess={float(excess_cagr):+.2%} "
            f"alpha={float(alpha):+.2%} "
            f"Sharpe={metrics['sharpe']:.3f} "
            f"DD={metrics['max_drawdown']:.2%}"
        )

    return (
        pd.DataFrame(
            leaderboard_rows
        ),
        pd.concat(
            annual_frames,
            ignore_index=True,
        ),
        pd.concat(
            equity_frames,
            ignore_index=True,
        ),
    )


def model_diagnostic_summary(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    for model_name, group in (
        diagnostics.groupby(
            "model_name",
            sort=False,
        )
    ):
        ic = pd.to_numeric(
            group[
                "mean_daily_ic"
            ],
            errors="coerce",
        ).dropna()
        positive = pd.to_numeric(
            group[
                "positive_ic_fraction"
            ],
            errors="coerce",
        ).dropna()
        decile = pd.to_numeric(
            group[
                "mean_top_decile_excess"
            ],
            errors="coerce",
        ).dropna()

        rows.append({
            "model_name": (
                model_name
            ),
            "oos_years": int(
                group[
                    "year"
                ].nunique()
            ),
            "median_annual_mean_ic": (
                float(
                    ic.median()
                )
                if len(ic)
                else None
            ),
            "q25_annual_mean_ic": (
                float(
                    ic.quantile(
                        0.25
                    )
                )
                if len(ic)
                else None
            ),
            "median_positive_ic_fraction": (
                float(
                    positive.median()
                )
                if len(positive)
                else None
            ),
            "median_top_decile_excess": (
                float(
                    decile.median()
                )
                if len(decile)
                else None
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_consensus(
    df: pd.DataFrame,
    *,
    eval_mask: pd.Series,
    score_store: dict[
        str,
        np.ndarray,
    ],
    models: list[str],
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    available = [
        name
        for name in models
        if name in score_store
    ]

    if len(available) < 3:
        raise RuntimeError(
            "Tree consensus requires at "
            "least three available models."
        )

    view = pd.DataFrame({
        "date": (
            df.loc[
                eval_mask,
                "date",
            ]
            .to_numpy()
        ),
    })

    for name in available:
        values = score_store[
            name
        ]
        if len(values) != len(view):
            raise RuntimeError(
                f"Score length mismatch "
                f"for {name}."
            )
        view[name] = values

    rank_columns: list[
        str
    ] = []
    for name in available:
        rank_col = (
            f"{name}__rank"
        )
        view[
            rank_col
        ] = (
            view.groupby(
                "date",
                sort=False,
            )[name]
            .rank(
                method="average",
                pct=True,
            )
        )
        rank_columns.append(
            rank_col
        )

    view[
        "consensus_score"
    ] = (
        view[
            rank_columns
        ].mean(
            axis=1,
            skipna=True,
        )
    )

    corr = (
        view[
            rank_columns
        ]
        .corr(
            method="pearson"
        )
    )
    corr.index = available
    corr.columns = available

    return (
        view[
            "consensus_score"
        ].to_numpy(
            dtype=float
        ),
        corr,
    )


def self_test() -> None:
    annual = pd.DataFrame({
        "year": [
            2019,
            2020,
            2021,
            2022,
        ],
        "partial_year": [
            False,
            False,
            False,
            True,
        ],
        "cagr": [
            0.10,
            0.20,
            -0.05,
            0.50,
        ],
        "sharpe": [
            0.5,
            1.2,
            -0.2,
            2.0,
        ],
        "max_drawdown": [
            -0.1,
            -0.2,
            -0.3,
            -0.1,
        ],
    })
    summary = (
        complete_year_summary(
            annual
        )
    )
    assert (
        summary[
            "complete_years"
        ]
        == 3
    )
    assert math.isclose(
        summary[
            "positive_cagr_fraction"
        ],
        2.0 / 3.0,
    )

    fake = pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
        ]),
    })
    mask = pd.Series(
        [True] * 4
    )
    scores = {
        "a": np.array([
            0.1,
            0.9,
            0.2,
            0.8,
        ]),
        "b": np.array([
            0.2,
            0.8,
            0.1,
            0.9,
        ]),
        "c": np.array([
            0.3,
            0.7,
            0.4,
            0.6,
        ]),
    }
    values, corr = (
        build_consensus(
            fake,
            eval_mask=mask,
            score_store=scores,
            models=[
                "a",
                "b",
                "c",
            ],
        )
    )
    assert len(values) == 4
    assert corr.shape == (
        3,
        3,
    )

    print(
        "Model tournament self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Walk-forward India model "
            "tournament using one frozen "
            "portfolio/execution protocol."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=(
            DEFAULT_MODELS
        ),
        choices=[
            "ridge",
            "hist_gb",
            "hist_gb_fixed",
            "random_forest",
            "extra_trees",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
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
        "--oos-start-year",
        type=int,
        default=2019,
    )
    ap.add_argument(
        "--oos-end-year",
        type=int,
        default=2026,
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
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--no-consensus",
        action="store_true",
    )
    ap.add_argument(
        "--rebuild-predictions",
        action="store_true",
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
        int(k) < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k values must be >= 1."
        )

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "model_tournament"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_ml_panel(
        root
        / "data/processed/"
        "research_panel"
    )
    df = add_target_rank(
        df
    )

    eval_start_date = (
        first_aligned_signal_date(
            df,
            oos_start_year=(
                args.oos_start_year
            ),
            eval_start_year=(
                args.oos_start_year
            ),
        )
    )
    eval_mask = (
        df["date"].between(
            eval_start_date,
            pd.Timestamp(
                f"{args.oos_end_year}-12-31"
            ),
            inclusive="both",
        )
        & df[
            "eligible_universe"
        ]
    )
    years = list(
        range(
            args.oos_start_year,
            args.oos_end_year
            + 1,
        )
    )

    events = (
        load_mechanical_events(
            root
        )
    )
    costs = cost_profile(
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

    panel_dates = set(
        df.loc[
            df["date"].between(
                eval_start_date,
                pd.Timestamp(
                    f"{args.oos_end_year}-12-31"
                ),
                inclusive="both",
            ),
            "date",
        ].dropna()
    )
    benchmark_table, benchmark_curves = (
        benchmark_rows(
            root,
            panel_dates=(
                panel_dates
            ),
            capital=(
                args.capital
            ),
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
        )
    )
    benchmark_curve = (
        benchmark_curves[
            "NIFTY 500"
        ]
    )

    diagnostic_frames: list[
        pd.DataFrame
    ] = []
    leaderboard_frames: list[
        pd.DataFrame
    ] = []
    annual_frames: list[
        pd.DataFrame
    ] = []
    equity_frames: list[
        pd.DataFrame
    ] = []
    failures: list[
        dict
    ] = []
    score_store: dict[
        str,
        np.ndarray,
    ] = {}

    for model_name in args.models:
        print(
            "\n================================"
        )
        print(
            f"MODEL: {model_name}"
        )
        print(
            "================================"
        )

        try:
            diagnostics = (
                build_annual_oos_predictions(
                    df,
                    root=root,
                    model_name=(
                        model_name
                    ),
                    years=years,
                    random_state=(
                        args.seed
                    ),
                    rebuild=(
                        args.rebuild_predictions
                    ),
                )
            )
            diagnostics[
                "model_name"
            ] = model_name
            diagnostics[
                "seed"
            ] = int(
                args.seed
            )
            diagnostic_frames.append(
                diagnostics
            )

            score_store[
                model_name
            ] = (
                pd.to_numeric(
                    df.loc[
                        eval_mask,
                        "persistent_score",
                    ],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float32
                )
                .copy()
            )

            (
                leaderboard,
                annual,
                equity,
            ) = evaluate_scores(
                df,
                events,
                model_name=(
                    model_name
                ),
                eval_start_date=(
                    eval_start_date
                ),
                eval_end_year=(
                    args.oos_end_year
                ),
                top_k_values=[
                    int(k)
                    for k
                    in args.top_k
                ],
                initial_capital=(
                    args.capital
                ),
                costs=costs,
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
                benchmark_curve=(
                    benchmark_curve
                ),
            )
            leaderboard_frames.append(
                leaderboard
            )
            annual_frames.append(
                annual
            )
            equity_frames.append(
                equity
            )

        except Exception as exc:
            print(
                f"FAILED {model_name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            failures.append({
                "model_name": (
                    model_name
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(
                    exc
                ),
            })

    correlation = pd.DataFrame()

    if (
        not args.no_consensus
        and len(
            [
                name
                for name
                in TREE_CONSENSUS
                if name
                in score_store
            ]
        )
        >= 3
    ):
        print(
            "\n================================"
        )
        print(
            "MODEL: tree_consensus"
        )
        print(
            "================================"
        )

        consensus, correlation = (
            build_consensus(
                df,
                eval_mask=(
                    eval_mask
                ),
                score_store=(
                    score_store
                ),
                models=(
                    TREE_CONSENSUS
                ),
            )
        )

        df[
            "persistent_score"
        ] = np.nan
        df.loc[
            eval_mask,
            "persistent_score",
        ] = consensus
        add_daily_ranks(
            df
        )

        (
            leaderboard,
            annual,
            equity,
        ) = evaluate_scores(
            df,
            events,
            model_name=(
                "tree_consensus"
            ),
            eval_start_date=(
                eval_start_date
            ),
            eval_end_year=(
                args.oos_end_year
            ),
            top_k_values=[
                int(k)
                for k
                in args.top_k
            ],
            initial_capital=(
                args.capital
            ),
            costs=costs,
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
            benchmark_curve=(
                benchmark_curve
            ),
        )
        leaderboard_frames.append(
            leaderboard
        )
        annual_frames.append(
            annual
        )
        equity_frames.append(
            equity
        )

    if not leaderboard_frames:
        raise RuntimeError(
            "No tournament model completed."
        )

    leaderboard = pd.concat(
        leaderboard_frames,
        ignore_index=True,
    )
    annual = pd.concat(
        annual_frames,
        ignore_index=True,
    )
    equity = pd.concat(
        equity_frames,
        ignore_index=True,
    )

    leaderboard = (
        leaderboard.sort_values(
            [
                "economic_target_excess5_sharpe1",
                "sharpe",
                "excess_cagr",
                "cagr",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    leaderboard.to_csv(
        output_root
        / "leaderboard.csv",
        index=False,
    )
    annual.to_csv(
        output_root
        / "annual_portfolio_metrics.csv",
        index=False,
    )
    equity.to_parquet(
        output_root
        / "equity_curves.parquet",
        index=False,
        compression="zstd",
    )

    if diagnostic_frames:
        diagnostics = pd.concat(
            diagnostic_frames,
            ignore_index=True,
        )
        diagnostics.to_csv(
            output_root
            / "annual_prediction_diagnostics.csv",
            index=False,
        )
        model_diagnostic_summary(
            diagnostics
        ).to_csv(
            output_root
            / "prediction_summary.csv",
            index=False,
        )

    benchmark_table.to_csv(
        output_root
        / "benchmark_reference.csv",
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        output_root
        / "failures.csv",
        index=False,
    )

    if not correlation.empty:
        correlation.to_csv(
            output_root
            / "tree_consensus_score_correlation.csv"
        )

    summary = {
        "goal": {
            "economic": (
                "net excess CAGR vs NIFTY 500 "
                ">= 5 percentage points and "
                "Sharpe > 1"
            ),
            "factor_adjusted": (
                "CAPM-like annual alpha >= 5% "
                "and Sharpe > 1"
            ),
        },
        "protocol": {
            "oos_start_year": int(
                args.oos_start_year
            ),
            "oos_end_year": int(
                args.oos_end_year
            ),
            "capital": float(
                args.capital
            ),
            "risk_free_rate": float(
                args.risk_free_rate
            ),
            "top_k": [
                int(k)
                for k in args.top_k
            ],
            "policy": (
                "buy fresh top-K; keep while "
                "rank <= K; gap=0; next-open "
                "execution; rebalance cadence "
                "20 sessions"
            ),
            "seed": int(
                args.seed
            ),
        },
        "models_requested": list(
            args.models
        ),
        "tree_consensus_members": [
            name
            for name
            in TREE_CONSENSUS
            if name in score_store
        ],
        "failures": failures,
        "important_note": (
            "This is a fixed-configuration "
            "model tournament, not per-model "
            "hyperparameter optimization."
        ),
    }

    (
        output_root
        / "tournament_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    preview = leaderboard[
        [
            "model_name",
            "top_k",
            "cagr",
            "benchmark_cagr",
            "excess_cagr",
            "alpha_annualized",
            "sharpe",
            "sortino",
            "max_drawdown",
            "median_annual_cagr",
            "q25_annual_cagr",
            "median_annual_sharpe",
            "q25_annual_sharpe",
            "turnover_multiple",
            "total_fees",
            "economic_target_excess5_sharpe1",
            "factor_alpha5_sharpe1",
        ]
    ].copy()

    print(
        "\n=== INDIA MODEL TOURNAMENT ==="
    )
    print(
        preview.to_string(
            index=False,
            formatters={
                "cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "benchmark_cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "excess_cagr": (
                    lambda x: f"{x:+.2%}"
                ),
                "alpha_annualized": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:+.2%}"
                    )
                ),
                "sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "sortino": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "max_drawdown": (
                    lambda x: f"{x:.2%}"
                ),
                "median_annual_cagr": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.2%}"
                    )
                ),
                "q25_annual_cagr": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.2%}"
                    )
                ),
                "median_annual_sharpe": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "q25_annual_sharpe": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
                "turnover_multiple": (
                    lambda x: f"{x:.1f}x"
                ),
                "total_fees": (
                    lambda x: f"₹{x:,.0f}"
                ),
            },
        )
    )

    hits = leaderboard.loc[
        leaderboard[
            "economic_target_excess5_sharpe1"
        ]
    ]
    print(
        "\n=== TARGET STATUS ==="
    )
    if hits.empty:
        print(
            "No fixed configuration yet "
            "clears both net excess CAGR "
            ">=5pp and Sharpe>1."
        )
    else:
        print(
            hits[
                [
                    "model_name",
                    "top_k",
                    "excess_cagr",
                    "sharpe",
                    "alpha_annualized",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nOutputs: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
