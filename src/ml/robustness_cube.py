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
        sys.path.insert(0, str(path))

from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
)
from classical import (  # noqa: E402
    add_target_rank,
    load_ml_panel,
)
from nested_generalization import (  # noqa: E402
    DEFAULT_OOS_END_YEAR,
    DEFAULT_OOS_START_YEAR,
    actual_transaction_costs_by_year,
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
    captioned_metric_rows,
)


DEFAULT_TOP_K = [5, 10, 15, 20]
DEFAULT_CAPITALS = [
    25_000.0,
    50_000.0,
    100_000.0,
    250_000.0,
    500_000.0,
]
DEFAULT_COST_FACTORS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]
DEFAULT_LIQUIDITY = [
    5_000_000.0,
    10_000_000.0,
    25_000_000.0,
    50_000_000.0,
]
DEFAULT_MODELS = [
    "ridge",
    "hist_gb",
    "hist_gb_fixed",
]
DEFAULT_SEEDS = [
    7,
    19,
    42,
    123,
    2026,
]

JOINT_STRESSES = [
    {
        "name": "base",
        "capital": 50_000.0,
        "cost_factor": 1.0,
        "liquidity": 5_000_000.0,
    },
    {
        "name": "small_capital_high_cost",
        "capital": 25_000.0,
        "cost_factor": 2.0,
        "liquidity": 10_000_000.0,
    },
    {
        "name": "strict_liquidity",
        "capital": 50_000.0,
        "cost_factor": 1.5,
        "liquidity": 25_000_000.0,
    },
    {
        "name": "very_liquid_stress",
        "capital": 100_000.0,
        "cost_factor": 1.5,
        "liquidity": 50_000_000.0,
    },
    {
        "name": "larger_capital_cost_stress",
        "capital": 250_000.0,
        "cost_factor": 2.0,
        "liquidity": 25_000_000.0,
    },
]


def scaled_cost_profile(
    factor: float,
    *,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> CostProfile:
    if factor < 0:
        raise ValueError(
            "Cost factor must be >= 0."
        )

    base = CostProfile(
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

    if factor == 0:
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

    # GST itself remains the statutory percentage. Its taxable base
    # (brokerage/exchange/SEBI) is stressed by factor, which scales the
    # resulting GST contribution without double-scaling the GST rate.
    return CostProfile(
        stt_buy_rate=(
            base.stt_buy_rate
            * factor
        ),
        stt_sell_rate=(
            base.stt_sell_rate
            * factor
        ),
        stamp_buy_rate=(
            base.stamp_buy_rate
            * factor
        ),
        sebi_rate=(
            base.sebi_rate
            * factor
        ),
        exchange_rate=(
            base.exchange_rate
            * factor
        ),
        gst_rate=(
            base.gst_rate
        ),
        brokerage_per_order=(
            base.brokerage_per_order
            * factor
        ),
        dp_charge_per_sell=(
            base.dp_charge_per_sell
            * factor
        ),
        slippage_bps=(
            base.slippage_bps
            * factor
        ),
    )


def cache_scores(
    df: pd.DataFrame,
) -> np.ndarray:
    return (
        pd.to_numeric(
            df[
                "persistent_score"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
        .copy()
    )


def restore_scores(
    df: pd.DataFrame,
    values: np.ndarray,
) -> None:
    if len(values) != len(df):
        raise RuntimeError(
            "Cached score length mismatch."
        )

    df[
        "persistent_score"
    ] = values


def ensure_model_scores(
    df: pd.DataFrame,
    *,
    root: Path,
    model_name: str,
    seed: int,
    years: list[int],
    rebuild_predictions: bool,
) -> pd.DataFrame:
    diagnostics = (
        build_annual_oos_predictions(
            df,
            root=root,
            model_name=model_name,
            years=years,
            random_state=seed,
            rebuild=(
                rebuild_predictions
            ),
        )
    )

    return diagnostics


def make_execution_panel(
    df: pd.DataFrame,
    *,
    eval_start_date: pd.Timestamp,
    eval_end_year: int,
    liquidity_threshold: float,
) -> pd.DataFrame:
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

    # Robustness stress only: the predictive model remains frozen. We tighten
    # the point-in-time execution/inference universe by contemporaneous trailing
    # turnover, then re-rank scores inside that stricter universe.
    panel[
        "eligible_universe"
    ] = (
        panel[
            "eligible_universe"
        ].fillna(False)
        & panel[
            "turnover_median_20d"
        ].ge(
            liquidity_threshold
        )
    )

    panel[
        "persistent_rank"
    ] = np.nan
    panel[
        "persistent_score_pct"
    ] = np.nan
    add_daily_ranks(
        panel
    )

    return panel


def strategy_schedule(
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


def run_one(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    axis: str,
    setting: str,
    model_name: str,
    seed: int,
    capital: float,
    cost_factor: float,
    liquidity_threshold: float,
    top_k: int,
    costs: CostProfile,
    eval_years: list[int],
    risk_free_rate: float,
) -> tuple[
    dict,
    pd.DataFrame,
    dict,
]:
    result = run_persistent_strategy(
        panel,
        events,
        name=(
            f"robustness_{axis}_"
            f"{setting}_k{top_k}"
        ),
        capital=capital,
        top_k=top_k,
        schedule=(
            strategy_schedule(
                top_k=top_k,
                years=eval_years,
            )
        ),
        costs=costs,
        annual_risk_free_rate=(
            risk_free_rate
        ),
    )

    meta = {
        "axis": axis,
        "setting": setting,
        "model_name": (
            model_name
        ),
        "seed": int(seed),
        "capital": float(
            capital
        ),
        "cost_factor": float(
            cost_factor
        ),
        "liquidity_threshold": float(
            liquidity_threshold
        ),
        "top_k": int(
            top_k
        ),
        "policy": (
            "hold_while_rank_le_k_gap0"
        ),
    }

    row = {
        **meta,
        **result[
            "metrics"
        ],
    }

    annual = (
        annual_metrics_from_result(
            result,
            annual_risk_free_rate=(
                risk_free_rate
            ),
            metadata=meta,
        )
    )

    return (
        row,
        annual,
        result,
    )


def summarize_axis(
    annual: pd.DataFrame,
    terminal: pd.DataFrame,
) -> pd.DataFrame:
    complete = annual.loc[
        ~annual[
            "partial_year"
        ].fillna(False)
    ].copy()

    rows: list[
        dict
    ] = []

    keys = [
        "axis",
        "setting",
        "model_name",
        "seed",
        "capital",
        "cost_factor",
        "liquidity_threshold",
    ]

    for group_key, group in (
        complete.groupby(
            keys,
            dropna=False,
            sort=False,
        )
    ):
        values = dict(
            zip(
                keys,
                group_key,
            )
        )

        cagr = pd.to_numeric(
            group["cagr"],
            errors="coerce",
        ).dropna()
        sharpe = pd.to_numeric(
            group["sharpe"],
            errors="coerce",
        ).dropna()

        terminal_group = (
            terminal.loc[
                terminal[
                    "axis"
                ].eq(
                    values[
                        "axis"
                    ]
                )
                & terminal[
                    "setting"
                ].eq(
                    values[
                        "setting"
                    ]
                )
                & terminal[
                    "model_name"
                ].eq(
                    values[
                        "model_name"
                    ]
                )
                & terminal[
                    "seed"
                ].eq(
                    values[
                        "seed"
                    ]
                )
                & terminal[
                    "capital"
                ].eq(
                    values[
                        "capital"
                    ]
                )
                & terminal[
                    "cost_factor"
                ].eq(
                    values[
                        "cost_factor"
                    ]
                )
                & terminal[
                    "liquidity_threshold"
                ].eq(
                    values[
                        "liquidity_threshold"
                    ]
                )
            ]
        )

        rows.append({
            **values,
            "k_values": int(
                group[
                    "top_k"
                ].nunique()
            ),
            "complete_years": int(
                group[
                    "year"
                ].nunique()
            ),
            "year_k_observations": int(
                len(group)
            ),
            "median_annual_cagr": float(
                cagr.median()
            ),
            "q25_annual_cagr": float(
                cagr.quantile(
                    0.25
                )
            ),
            "worst_annual_cagr": float(
                cagr.min()
            ),
            "positive_cagr_fraction": float(
                cagr.gt(0).mean()
            ),
            "median_annual_sharpe": float(
                sharpe.median()
            ),
            "q25_annual_sharpe": float(
                sharpe.quantile(
                    0.25
                )
            ),
            "positive_sharpe_fraction": float(
                sharpe.gt(0).mean()
            ),
            "worst_year_max_drawdown": float(
                pd.to_numeric(
                    group[
                        "max_drawdown"
                    ],
                    errors="coerce",
                ).min()
            ),
            "median_annual_turnover": float(
                pd.to_numeric(
                    group[
                        "turnover_multiple"
                    ],
                    errors="coerce",
                ).median()
            ),
            "median_terminal_cagr_across_k": float(
                pd.to_numeric(
                    terminal_group[
                        "cagr"
                    ],
                    errors="coerce",
                ).median()
            ),
            "min_terminal_cagr_across_k": float(
                pd.to_numeric(
                    terminal_group[
                        "cagr"
                    ],
                    errors="coerce",
                ).min()
            ),
            "median_terminal_drawdown_across_k": float(
                pd.to_numeric(
                    terminal_group[
                        "max_drawdown"
                    ],
                    errors="coerce",
                ).median()
            ),
            "median_failed_entries_across_k": float(
                pd.to_numeric(
                    terminal_group.get(
                        "failed_entries",
                        pd.Series(
                            dtype=float
                        ),
                    ),
                    errors="coerce",
                ).median()
            ),
            "median_exposure_across_k": float(
                pd.to_numeric(
                    terminal_group[
                        "average_exposure"
                    ],
                    errors="coerce",
                ).median()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "axis",
                "median_annual_cagr",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def aligned_return_matrix(
    strategy_curves: dict[
        int,
        pd.DataFrame,
    ],
    benchmark_curve: pd.DataFrame,
) -> pd.DataFrame:
    base = (
        benchmark_curve[
            [
                "date",
                "equity",
            ]
        ]
        .rename(
            columns={
                "equity":
                "benchmark_equity",
            }
        )
        .copy()
    )
    base["date"] = pd.to_datetime(
        base["date"],
        errors="coerce",
    ).dt.normalize()

    for k, curve in (
        strategy_curves.items()
    ):
        part = (
            curve[
                [
                    "date",
                    "equity",
                ]
            ]
            .rename(
                columns={
                    "equity":
                    f"strategy_{k}_equity",
                }
            )
            .copy()
        )
        part["date"] = pd.to_datetime(
            part["date"],
            errors="coerce",
        ).dt.normalize()
        base = base.merge(
            part,
            on="date",
            how="inner",
            validate="one_to_one",
        )

    base = (
        base.sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    out = pd.DataFrame({
        "date": (
            base["date"]
        ),
        "benchmark": (
            base[
                "benchmark_equity"
            ]
            .pct_change()
        ),
    })

    for k in sorted(
        strategy_curves
    ):
        out[
            f"strategy_{k}"
        ] = (
            base[
                f"strategy_{k}_equity"
            ]
            .pct_change()
        )

    return (
        out.dropna()
        .reset_index(drop=True)
    )


def drawdown_from_returns(
    returns: np.ndarray,
) -> float:
    wealth = np.cumprod(
        1.0 + returns
    )
    peak = np.maximum.accumulate(
        wealth
    )
    dd = (
        wealth
        / peak
        - 1.0
    )
    return float(
        dd.min()
    )


def moving_block_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        raise ValueError(
            "Bootstrap length must be > 0."
        )
    block_length = max(
        1,
        min(
            int(block_length),
            n,
        ),
    )

    needed = int(
        math.ceil(
            n
            / block_length
        )
    )
    max_start = (
        n - block_length
    )
    starts = rng.integers(
        0,
        max_start + 1,
        size=needed,
    )

    chunks = [
        np.arange(
            start,
            start + block_length,
        )
        for start in starts
    ]

    return np.concatenate(
        chunks
    )[:n]


def bootstrap_family(
    returns: pd.DataFrame,
    *,
    top_k_values: list[int],
    reps: int,
    block_length: int,
    seed: int,
    annual_risk_free_rate: float,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    required = [
        "benchmark",
        *[
            f"strategy_{k}"
            for k
            in top_k_values
        ],
    ]
    matrix = (
        returns[
            required
        ]
        .astype(float)
        .to_numpy()
    )

    n = len(matrix)
    if n < 100:
        raise RuntimeError(
            "Too few aligned daily returns "
            "for robustness bootstrap."
        )

    rng = np.random.default_rng(
        seed
    )
    rf_daily = (
        (1.0 + annual_risk_free_rate)
        ** (1.0 / 252.0)
        - 1.0
    )

    benchmark = matrix[:, 0]
    strategies = matrix[:, 1:]
    active = (
        strategies
        - benchmark[:, None]
    )

    observed_active_mean = (
        active.mean(
            axis=0
        )
        * 252.0
    )
    observed_max = float(
        observed_active_mean.max()
    )

    centered_active = (
        active
        - active.mean(
            axis=0,
            keepdims=True,
        )
    )

    samples: dict[
        int,
        dict[str, list[float]],
    ] = {
        int(k): {
            "cagr": [],
            "sharpe": [],
            "max_drawdown": [],
            "excess_cagr": [],
        }
        for k in top_k_values
    }
    max_null: list[
        float
    ] = []

    for _ in range(
        int(reps)
    ):
        idx = moving_block_indices(
            n,
            block_length=(
                block_length
            ),
            rng=rng,
        )

        b = benchmark[
            idx
        ]
        annual_b = (
            np.prod(
                1.0 + b
            )
            ** (
                252.0
                / len(b)
            )
            - 1.0
        )

        for col, k in enumerate(
            top_k_values
        ):
            s = strategies[
                idx,
                col,
            ]

            annual_s = (
                np.prod(
                    1.0 + s
                )
                ** (
                    252.0
                    / len(s)
                )
                - 1.0
            )
            std = float(
                np.std(
                    s,
                    ddof=1,
                )
            )
            sharpe = (
                (
                    float(
                        np.mean(s)
                    )
                    - rf_daily
                )
                / std
                * math.sqrt(
                    252.0
                )
                if std > 0
                else np.nan
            )

            samples[
                int(k)
            ][
                "cagr"
            ].append(
                float(
                    annual_s
                )
            )
            samples[
                int(k)
            ][
                "sharpe"
            ].append(
                float(
                    sharpe
                )
            )
            samples[
                int(k)
            ][
                "max_drawdown"
            ].append(
                drawdown_from_returns(
                    s
                )
            )
            samples[
                int(k)
            ][
                "excess_cagr"
            ].append(
                float(
                    annual_s
                    - annual_b
                )
            )

        null_idx = moving_block_indices(
            n,
            block_length=(
                block_length
            ),
            rng=rng,
        )
        null_means = (
            centered_active[
                null_idx
            ].mean(
                axis=0
            )
            * 252.0
        )
        max_null.append(
            float(
                null_means.max()
            )
        )

    rows: list[
        dict
    ] = []

    for k in top_k_values:
        for metric, values in (
            samples[
                int(k)
            ].items()
        ):
            arr = np.asarray(
                values,
                dtype=float,
            )
            arr = arr[
                np.isfinite(
                    arr
                )
            ]
            rows.append({
                "top_k": int(k),
                "metric": metric,
                "bootstrap_reps": int(
                    reps
                ),
                "block_length": int(
                    block_length
                ),
                "p2_5": float(
                    np.quantile(
                        arr,
                        0.025,
                    )
                ),
                "median": float(
                    np.quantile(
                        arr,
                        0.50,
                    )
                ),
                "p97_5": float(
                    np.quantile(
                        arr,
                        0.975,
                    )
                ),
            })

    max_null_arr = np.asarray(
        max_null,
        dtype=float,
    )
    p_value = float(
        (
            1
            + np.sum(
                max_null_arr
                >= observed_max
            )
        )
        / (
            len(
                max_null_arr
            )
            + 1
        )
    )

    family_test = {
        "test": (
            "moving-block familywise max "
            "annualized mean active-return test"
        ),
        "family": [
            int(k)
            for k in top_k_values
        ],
        "benchmark": (
            "NIFTY 500 price index"
        ),
        "observed_max_annualized_mean_active_return": (
            observed_max
        ),
        "familywise_p_value": (
            p_value
        ),
        "bootstrap_reps": int(
            reps
        ),
        "block_length_sessions": int(
            block_length
        ),
        "null": (
            "each K active-return series is "
            "centered to zero mean; blocks are "
            "resampled jointly across K to preserve "
            "cross-strategy dependence"
        ),
        "scope_limitation": (
            "controls selection across K=5/10/15/20 "
            "within this frozen persistence family; "
            "it does not correct for every strategy "
            "experiment examined during the project"
        ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        family_test,
    )


def self_test() -> None:
    costs = scaled_cost_profile(
        2.0,
        brokerage_per_order=15.0,
        dp_charge_per_sell=0.0,
        slippage_bps=5.0,
    )
    assert (
        costs.brokerage_per_order
        == 30.0
    )
    assert (
        costs.gst_rate
        > 0
    )

    rng = np.random.default_rng(
        42
    )
    n = 600
    benchmark = rng.normal(
        0.0003,
        0.01,
        size=n,
    )
    returns = pd.DataFrame({
        "benchmark": benchmark,
        "strategy_5": (
            benchmark
            + 0.0002
            + rng.normal(
                0,
                0.002,
                size=n,
            )
        ),
        "strategy_10": (
            benchmark
            + 0.0001
            + rng.normal(
                0,
                0.002,
                size=n,
            )
        ),
    })

    ci, test = bootstrap_family(
        returns,
        top_k_values=[
            5,
            10,
        ],
        reps=100,
        block_length=20,
        seed=42,
        annual_risk_free_rate=0.0,
    )
    assert not ci.empty
    assert (
        0
        <= test[
            "familywise_p_value"
        ]
        <= 1
    )

    print(
        "Robustness cube self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generalization robustness battery for "
            "the frozen persistence rule across "
            "capital, costs, liquidity, models, "
            "random seeds and joint stresses."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--oos-start-year",
        type=int,
        default=(
            DEFAULT_OOS_START_YEAR
        ),
    )
    ap.add_argument(
        "--oos-end-year",
        type=int,
        default=(
            DEFAULT_OOS_END_YEAR
        ),
    )
    ap.add_argument(
        "--eval-start-year",
        type=int,
        default=2019,
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
        "--capital-values",
        nargs="+",
        type=float,
        default=(
            DEFAULT_CAPITALS
        ),
    )
    ap.add_argument(
        "--cost-factors",
        nargs="+",
        type=float,
        default=(
            DEFAULT_COST_FACTORS
        ),
    )
    ap.add_argument(
        "--liquidity-thresholds",
        nargs="+",
        type=float,
        default=(
            DEFAULT_LIQUIDITY
        ),
    )
    ap.add_argument(
        "--model-values",
        nargs="+",
        default=(
            DEFAULT_MODELS
        ),
        choices=[
            "ridge",
            "hist_gb",
            "hist_gb_fixed",
            "random_forest",
        ],
    )
    ap.add_argument(
        "--seed-values",
        nargs="+",
        type=int,
        default=(
            DEFAULT_SEEDS
        ),
    )
    ap.add_argument(
        "--base-model",
        default="hist_gb",
        choices=[
            "ridge",
            "hist_gb",
            "hist_gb_fixed",
            "random_forest",
        ],
    )
    ap.add_argument(
        "--base-seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--base-capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--base-liquidity",
        type=float,
        default=5_000_000.0,
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
        "--bootstrap-reps",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "--bootstrap-block",
        type=int,
        default=20,
        help=(
            "Single moving-block length retained for "
            "backward compatibility."
        ),
    )
    ap.add_argument(
        "--bootstrap-block-values",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional list of moving-block lengths. "
            "When provided, all are evaluated in one "
            "loaded process and override "
            "--bootstrap-block."
        ),
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=8675309,
    )
    ap.add_argument(
        "--sections",
        nargs="+",
        default=[
            "capital",
            "cost",
            "liquidity",
            "model",
            "seed",
            "joint",
            "bootstrap",
        ],
        choices=[
            "capital",
            "cost",
            "liquidity",
            "model",
            "seed",
            "joint",
            "bootstrap",
        ],
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
        k < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k must contain positive integers."
        )
    if any(
        value <= 0
        for value
        in args.capital_values
    ):
        raise SystemExit(
            "--capital-values must be > 0."
        )
    if any(
        value < 0
        for value
        in args.cost_factors
    ):
        raise SystemExit(
            "--cost-factors must be >= 0."
        )
    if any(
        value < 0
        for value
        in args.liquidity_thresholds
    ):
        raise SystemExit(
            "--liquidity-thresholds must be >= 0."
        )

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "robustness_cube"
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

    years = list(
        range(
            args.oos_start_year,
            args.oos_end_year
            + 1,
        )
    )
    eval_years = list(
        range(
            args.eval_start_year,
            args.oos_end_year
            + 1,
        )
    )

    eval_start_date = (
        first_aligned_signal_date(
            df,
            oos_start_year=(
                args.oos_start_year
            ),
            eval_start_year=(
                args.eval_start_year
            ),
        )
    )

    events = load_mechanical_events(
        root
    )

    diagnostics_frames: list[
        pd.DataFrame
    ] = []
    terminal_rows: list[
        dict
    ] = []
    annual_frames: list[
        pd.DataFrame
    ] = []

    # Base scores are reused for environment stresses.
    base_diag = ensure_model_scores(
        df,
        root=root,
        model_name=(
            args.base_model
        ),
        seed=args.base_seed,
        years=years,
        rebuild_predictions=(
            args.rebuild_predictions
        ),
    )
    base_diag[
        "model_name"
    ] = args.base_model
    base_diag[
        "seed"
    ] = args.base_seed
    diagnostics_frames.append(
        base_diag
    )
    base_scores = cache_scores(
        df
    )

    def evaluate_environment(
        *,
        axis: str,
        setting: str,
        capital: float,
        cost_factor: float,
        liquidity: float,
        model_name: str,
        seed: int,
    ) -> dict[
        int,
        dict,
    ]:
        panel = make_execution_panel(
            df,
            eval_start_date=(
                eval_start_date
            ),
            eval_end_year=(
                args.oos_end_year
            ),
            liquidity_threshold=(
                liquidity
            ),
        )
        costs = scaled_cost_profile(
            cost_factor,
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

        curves: dict[
            int,
            dict,
        ] = {}

        print(
            f"\n[{axis}] {setting}: "
            f"model={model_name}, seed={seed}, "
            f"capital=₹{capital:,.0f}, "
            f"cost={cost_factor:.2f}x, "
            f"liq>=₹{liquidity:,.0f}"
        )

        for k in args.top_k:
            row, annual, result = (
                run_one(
                    panel,
                    events,
                    axis=axis,
                    setting=setting,
                    model_name=(
                        model_name
                    ),
                    seed=seed,
                    capital=capital,
                    cost_factor=(
                        cost_factor
                    ),
                    liquidity_threshold=(
                        liquidity
                    ),
                    top_k=int(k),
                    costs=costs,
                    eval_years=(
                        eval_years
                    ),
                    risk_free_rate=(
                        args.risk_free_rate
                    ),
                )
            )
            terminal_rows.append(
                row
            )
            annual_frames.append(
                annual
            )
            curves[
                int(k)
            ] = result

            print(
                f"  k={int(k):2d} "
                f"CAGR={row['cagr']:.2%} "
                f"DD={row['max_drawdown']:.2%} "
                f"Sharpe={row['sharpe']:.3f} "
                f"turn={row['turnover_multiple']:.1f}x "
                f"failed={row.get('failed_entries', 0)}"
            )

        return curves

    base_curves: dict[
        int,
        dict,
    ] | None = None

    if "capital" in args.sections:
        restore_scores(
            df,
            base_scores,
        )
        for capital in (
            args.capital_values
        ):
            curves = evaluate_environment(
                axis="capital",
                setting=(
                    f"capital_{int(capital)}"
                ),
                capital=float(capital),
                cost_factor=1.0,
                liquidity=(
                    args.base_liquidity
                ),
                model_name=(
                    args.base_model
                ),
                seed=args.base_seed,
            )
            if (
                math.isclose(
                    float(capital),
                    float(
                        args.base_capital
                    ),
                )
            ):
                base_curves = curves

    if "cost" in args.sections:
        restore_scores(
            df,
            base_scores,
        )
        for factor in (
            args.cost_factors
        ):
            curves = evaluate_environment(
                axis="cost",
                setting=(
                    f"cost_{float(factor):.2f}x"
                ),
                capital=(
                    args.base_capital
                ),
                cost_factor=float(
                    factor
                ),
                liquidity=(
                    args.base_liquidity
                ),
                model_name=(
                    args.base_model
                ),
                seed=args.base_seed,
            )
            if math.isclose(
                float(factor),
                1.0,
            ):
                base_curves = (
                    base_curves
                    or curves
                )

    if "liquidity" in args.sections:
        restore_scores(
            df,
            base_scores,
        )
        for liquidity in (
            args.liquidity_thresholds
        ):
            curves = evaluate_environment(
                axis="liquidity",
                setting=(
                    "liq_"
                    f"{int(liquidity)}"
                ),
                capital=(
                    args.base_capital
                ),
                cost_factor=1.0,
                liquidity=float(
                    liquidity
                ),
                model_name=(
                    args.base_model
                ),
                seed=args.base_seed,
            )
            if math.isclose(
                float(liquidity),
                float(
                    args.base_liquidity
                ),
            ):
                base_curves = (
                    base_curves
                    or curves
                )

    if "joint" in args.sections:
        restore_scores(
            df,
            base_scores,
        )
        for stress in (
            JOINT_STRESSES
        ):
            curves = evaluate_environment(
                axis="joint",
                setting=stress[
                    "name"
                ],
                capital=float(
                    stress[
                        "capital"
                    ]
                ),
                cost_factor=float(
                    stress[
                        "cost_factor"
                    ]
                ),
                liquidity=float(
                    stress[
                        "liquidity"
                    ]
                ),
                model_name=(
                    args.base_model
                ),
                seed=args.base_seed,
            )
            if stress[
                "name"
            ] == "base":
                base_curves = (
                    base_curves
                    or curves
                )

    if "model" in args.sections:
        for model_name in (
            args.model_values
        ):
            if (
                model_name
                == args.base_model
            ):
                restore_scores(
                    df,
                    base_scores,
                )
                diag = base_diag.copy()
            else:
                diag = ensure_model_scores(
                    df,
                    root=root,
                    model_name=(
                        model_name
                    ),
                    seed=args.base_seed,
                    years=years,
                    rebuild_predictions=(
                        args.rebuild_predictions
                    ),
                )
                diag[
                    "model_name"
                ] = model_name
                diag[
                    "seed"
                ] = args.base_seed
                diagnostics_frames.append(
                    diag
                )

            curves = evaluate_environment(
                axis="model",
                setting=(
                    f"model_{model_name}"
                ),
                capital=(
                    args.base_capital
                ),
                cost_factor=1.0,
                liquidity=(
                    args.base_liquidity
                ),
                model_name=(
                    model_name
                ),
                seed=args.base_seed,
            )
            if (
                model_name
                == args.base_model
            ):
                base_curves = (
                    base_curves
                    or curves
                )

    if "seed" in args.sections:
        for seed in (
            args.seed_values
        ):
            if (
                int(seed)
                == int(
                    args.base_seed
                )
            ):
                restore_scores(
                    df,
                    base_scores,
                )
                diag = base_diag.copy()
            else:
                diag = ensure_model_scores(
                    df,
                    root=root,
                    model_name=(
                        args.base_model
                    ),
                    seed=int(seed),
                    years=years,
                    rebuild_predictions=(
                        args.rebuild_predictions
                    ),
                )
                diag[
                    "model_name"
                ] = args.base_model
                diag[
                    "seed"
                ] = int(
                    seed
                )
                diagnostics_frames.append(
                    diag
                )

            curves = evaluate_environment(
                axis="seed",
                setting=(
                    f"seed_{int(seed)}"
                ),
                capital=(
                    args.base_capital
                ),
                cost_factor=1.0,
                liquidity=(
                    args.base_liquidity
                ),
                model_name=(
                    args.base_model
                ),
                seed=int(seed),
            )
            if (
                int(seed)
                == int(
                    args.base_seed
                )
            ):
                base_curves = (
                    base_curves
                    or curves
                )

    # Restore canonical scores before any base-family bootstrap.
    restore_scores(
        df,
        base_scores,
    )

    if base_curves is None:
        base_curves = evaluate_environment(
            axis="bootstrap_base",
            setting="base",
            capital=(
                args.base_capital
            ),
            cost_factor=1.0,
            liquidity=(
                args.base_liquidity
            ),
            model_name=(
                args.base_model
            ),
            seed=args.base_seed,
        )

    terminal = pd.DataFrame(
        terminal_rows
    )
    annual = pd.concat(
        annual_frames,
        ignore_index=True,
    )
    axis_summary = summarize_axis(
        annual,
        terminal,
    )

    terminal.to_csv(
        output_root
        / "robustness_terminal_runs.csv",
        index=False,
    )
    annual.to_csv(
        output_root
        / "robustness_annual_runs.csv",
        index=False,
    )
    axis_summary.to_csv(
        output_root
        / "robustness_axis_summary.csv",
        index=False,
    )

    diagnostics = pd.concat(
        diagnostics_frames,
        ignore_index=True,
    ).drop_duplicates(
        [
            "year",
            "model_name",
            "seed",
        ],
        keep="last",
    )
    diagnostics.to_csv(
        output_root
        / "model_seed_oos_diagnostics.csv",
        index=False,
    )

    # NIFTY benchmark uses the same dates as the base family.
    base_panel = make_execution_panel(
        df,
        eval_start_date=(
            eval_start_date
        ),
        eval_end_year=(
            args.oos_end_year
        ),
        liquidity_threshold=(
            args.base_liquidity
        ),
    )
    benchmark, curves = benchmark_rows(
        root,
        panel_dates=set(
            base_panel[
                "date"
            ].dropna()
        ),
        capital=(
            args.base_capital
        ),
        annual_risk_free_rate=(
            args.risk_free_rate
        ),
    )
    benchmark.to_csv(
        output_root
        / "benchmark_reference.csv",
        index=False,
    )

    bootstrap_ci = pd.DataFrame()
    family_test: dict = {}
    family_tests: list[
        dict
    ] = []

    if "bootstrap" in args.sections:
        strategy_curves = {
            int(k): (
                base_curves[
                    int(k)
                ][
                    "equity"
                ]
            )
            for k in args.top_k
        }
        aligned = aligned_return_matrix(
            strategy_curves,
            curves[
                "NIFTY 500"
            ],
        )
        aligned.to_csv(
            output_root
            / "bootstrap_aligned_returns.csv",
            index=False,
            date_format="%Y-%m-%d",
        )

        block_values = (
            [
                int(x)
                for x in (
                    args.bootstrap_block_values
                )
            ]
            if args.bootstrap_block_values
            else [
                int(
                    args.bootstrap_block
                )
            ]
        )
        block_values = list(
            dict.fromkeys(
                block_values
            )
        )

        if any(
            value < 1
            for value in block_values
        ):
            raise SystemExit(
                "Bootstrap block lengths must be >= 1."
            )

        ci_frames: list[
            pd.DataFrame
        ] = []

        for block_length in (
            block_values
        ):
            print(
                "\nRunning moving-block bootstrap "
                f"with block={block_length} sessions "
                f"and reps={args.bootstrap_reps}..."
            )

            (
                ci_one,
                test_one,
            ) = bootstrap_family(
                aligned,
                top_k_values=[
                    int(k)
                    for k in args.top_k
                ],
                reps=(
                    args.bootstrap_reps
                ),
                block_length=(
                    block_length
                ),
                seed=(
                    args.bootstrap_seed
                    + int(
                        block_length
                    )
                ),
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
            )
            ci_one[
                "bootstrap_seed"
            ] = (
                args.bootstrap_seed
                + int(
                    block_length
                )
            )
            ci_frames.append(
                ci_one
            )

            test_one[
                "bootstrap_seed"
            ] = (
                args.bootstrap_seed
                + int(
                    block_length
                )
            )
            family_tests.append(
                test_one
            )

        bootstrap_ci = pd.concat(
            ci_frames,
            ignore_index=True,
        )
        bootstrap_ci.to_csv(
            output_root
            / "bootstrap_confidence_intervals.csv",
            index=False,
        )

        family_test = {
            "tests_by_block_length": (
                family_tests
            ),
            "scope_limitation": (
                "Each test controls selection across "
                "K=5/10/15/20 within this frozen "
                "persistence family only; none "
                "corrects for every strategy "
                "experiment examined during the "
                "project."
            ),
        }

        (
            output_root
            / "familywise_block_bootstrap_test.json"
        ).write_text(
            json.dumps(
                family_test,
                indent=2,
            )
            + "\n"
        )

    caption_frames: list[
        pd.DataFrame
    ] = []
    for row in terminal.to_dict(
        orient="records"
    ):
        frame = (
            captioned_metric_rows(
                row
            )
        )
        frame[
            "run_id"
        ] = (
            f"{row['axis']}|"
            f"{row['setting']}|"
            f"k{row['top_k']}"
        )
        caption_frames.append(
            frame
        )

    pd.concat(
        caption_frames,
        ignore_index=True,
    ).to_csv(
        output_root
        / "robustness_captioned_metrics.csv",
        index=False,
    )

    summary = {
        "goal": (
            "stress a frozen simple persistence "
            "rule rather than optimize a new rule"
        ),
        "frozen_policy": (
            "buy top-K, keep while rank <= K, "
            "replace after leaving K; score gap 0"
        ),
        "evaluation_start": str(
            eval_start_date.date()
        ),
        "evaluation_end_year": (
            args.oos_end_year
        ),
        "top_k_values": [
            int(k)
            for k in args.top_k
        ],
        "base": {
            "model": (
                args.base_model
            ),
            "seed": int(
                args.base_seed
            ),
            "capital": float(
                args.base_capital
            ),
            "liquidity_threshold": float(
                args.base_liquidity
            ),
            "cost_factor": 1.0,
        },
        "axes": {
            "capital": [
                float(x)
                for x in (
                    args.capital_values
                )
            ],
            "cost_factor": [
                float(x)
                for x in (
                    args.cost_factors
                )
            ],
            "liquidity_threshold": [
                float(x)
                for x in (
                    args.liquidity_thresholds
                )
            ],
            "model": list(
                args.model_values
            ),
            "seed": [
                int(x)
                for x in (
                    args.seed_values
                )
            ],
            "joint_stresses": (
                JOINT_STRESSES
            ),
        },
        "liquidity_interpretation": (
            "execution/inference universe stress "
            "only; model training remains the "
            "frozen base point-in-time universe"
        ),
        "cost_interpretation": (
            "stress multiplier applies to explicit "
            "STT/stamp/SEBI/exchange/brokerage/DP/"
            "slippage components; GST percentage "
            "itself is held fixed"
        ),
        "bootstrap": (
            family_test
            if family_test
            else None
        ),
        "limitations": [
            (
                "2022-2026 has been inspected "
                "during prior development"
            ),
            (
                "familywise bootstrap controls "
                "selection across K within this "
                "frozen persistence family only"
            ),
            (
                "price-index benchmarks remain "
                "return-definition matched because "
                "strategy dividends are not yet "
                "credited"
            ),
        ],
        "outputs": {
            "terminal": (
                "robustness_terminal_runs.csv"
            ),
            "annual": (
                "robustness_annual_runs.csv"
            ),
            "axis_summary": (
                "robustness_axis_summary.csv"
            ),
            "model_seed_diagnostics": (
                "model_seed_oos_diagnostics.csv"
            ),
            "benchmark": (
                "benchmark_reference.csv"
            ),
            "bootstrap_ci": (
                "bootstrap_confidence_intervals.csv"
            ),
            "bootstrap_test": (
                "familywise_block_bootstrap_test.json"
            ),
        },
    }

    (
        output_root
        / "robustness_cube_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(
        "\n=== ROBUSTNESS AXIS SUMMARY ==="
    )
    preview = (
        axis_summary[
            [
                "axis",
                "setting",
                "model_name",
                "seed",
                "median_annual_cagr",
                "q25_annual_cagr",
                "positive_cagr_fraction",
                "median_annual_sharpe",
                "q25_annual_sharpe",
                "worst_year_max_drawdown",
                "median_annual_turnover",
                "median_terminal_cagr_across_k",
                "min_terminal_cagr_across_k",
                "median_failed_entries_across_k",
            ]
        ]
        .copy()
    )

    for col in (
        "median_annual_cagr",
        "q25_annual_cagr",
        "worst_year_max_drawdown",
        "median_terminal_cagr_across_k",
        "min_terminal_cagr_across_k",
    ):
        preview[col] = (
            preview[col]
            * 100.0
        )

    print(
        preview.to_string(
            index=False,
            formatters={
                "median_annual_cagr": (
                    lambda x: f"{x:.2f}%"
                ),
                "q25_annual_cagr": (
                    lambda x: f"{x:.2f}%"
                ),
                "positive_cagr_fraction": (
                    lambda x: f"{x:.1%}"
                ),
                "median_annual_sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "q25_annual_sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "worst_year_max_drawdown": (
                    lambda x: f"{x:.2f}%"
                ),
                "median_annual_turnover": (
                    lambda x: f"{x:.1f}x"
                ),
                "median_terminal_cagr_across_k": (
                    lambda x: f"{x:.2f}%"
                ),
                "min_terminal_cagr_across_k": (
                    lambda x: f"{x:.2f}%"
                ),
            },
        )
    )

    if not bootstrap_ci.empty:
        print(
            "\n=== BLOCK-BOOTSTRAP "
            "CONFIDENCE INTERVALS ==="
        )
        ci = bootstrap_ci.copy()
        for col in (
            "p2_5",
            "median",
            "p97_5",
        ):
            ci.loc[
                ci[
                    "metric"
                ].isin(
                    [
                        "cagr",
                        "max_drawdown",
                        "excess_cagr",
                    ]
                ),
                col,
            ] *= 100.0
        print(
            ci.to_string(
                index=False
            )
        )

        print(
            "\n=== FAMILYWISE K-SELECTION "
            "BLOCK-BOOTSTRAP TESTS ==="
        )
        for test in family_tests:
            print(
                "block="
                f"{test['block_length_sessions']:>3d} "
                "sessions  p="
                f"{test['familywise_p_value']:.4f}"
            )
        print(
            family_test[
                "scope_limitation"
            ]
        )

    print(
        "\n=== ROBUSTNESS CUBE COMPLETE ==="
    )
    print(
        f"Outputs: {output_root}"
    )


if __name__ == "__main__":
    main()
