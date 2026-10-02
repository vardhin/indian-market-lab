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
    add_target_rank,
    build_model,
    cross_sectional_diagnostics,
    fit_model,
    load_ml_panel,
    predict_mask,
)
from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
    run_backtest,
)
from benchmark_report import (  # noqa: E402
    index_curve,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
    run_persistent_backtest,
)
from quant_metrics import (  # noqa: E402
    captioned_metric_rows,
    compute_performance_metrics,
)


DEFAULT_OOS_START_YEAR = 2016
DEFAULT_OOS_END_YEAR = 2026
DEFAULT_EVAL_START_YEAR = 2019
DEFAULT_TOP_K = [5, 10, 15, 20]
DEFAULT_HOLD_MULTIPLIERS = [1, 2, 3]
DEFAULT_SCORE_GAPS = [0.0, 0.05, 0.10]


def zero_cost_profile() -> CostProfile:
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


def current_cost_profile(
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


def prediction_path(
    root: Path,
    *,
    model_name: str,
    year: int,
) -> Path:
    return (
        root
        / "reports/ml/nested_generalization/"
        "predictions"
        / model_name
        / f"{year}.parquet"
    )


def attach_prediction_file(
    df: pd.DataFrame,
    *,
    path: Path,
    year: int,
) -> int:
    saved = pd.read_parquet(
        path,
        columns=[
            "date",
            "canonical_security_id",
            "persistent_score",
        ],
    )
    saved["date"] = pd.to_datetime(
        saved["date"],
        errors="coerce",
    ).dt.normalize()
    saved["canonical_security_id"] = (
        saved[
            "canonical_security_id"
        ]
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
            f"Duplicate prediction keys in {path}"
        )

    mask = (
        df["date"].dt.year.eq(
            year
        )
        & df["eligible_universe"]
    )

    mapping = (
        saved.set_index(
            keys
        )[
            "persistent_score"
        ]
    )
    idx = pd.MultiIndex.from_frame(
        df.loc[
            mask,
            keys,
        ]
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

    return int(
        np.isfinite(
            values
        ).sum()
    )


def build_annual_oos_predictions(
    df: pd.DataFrame,
    *,
    root: Path,
    model_name: str,
    years: list[int],
    random_state: int,
    rebuild: bool,
) -> pd.DataFrame:
    if (
        "persistent_score"
        not in df.columns
    ):
        df[
            "persistent_score"
        ] = np.nan
    else:
        df[
            "persistent_score"
        ] = np.nan

    diagnostics_rows: list[
        dict
    ] = []

    for year in years:
        path = prediction_path(
            root,
            model_name=model_name,
            year=year,
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            path.is_file()
            and not rebuild
        ):
            count = (
                attach_prediction_file(
                    df,
                    path=path,
                    year=year,
                )
            )
            print(
                f"OOS {year}: reused "
                f"{count:,} cached scores"
            )
        else:
            fold_start = pd.Timestamp(
                f"{year}-01-01"
            )
            fold_end = pd.Timestamp(
                f"{year}-12-31"
            )

            prior = df.loc[
                df["date"].lt(
                    fold_start
                ),
                "market_day_index",
            ].dropna()

            if prior.empty:
                raise RuntimeError(
                    "No prior market sessions "
                    f"for OOS year {year}."
                )

            cutoff = int(
                prior.max()
            )

            train_mask = (
                df[TRAINING_FLAG]
                & df[
                    "target_rank_20d"
                ].notna()
                & df["date"].ge(
                    TRAIN_START
                )
                & df[
                    "market_day_index"
                ].le(
                    cutoff
                    - HORIZON
                )
            )
            score_mask = (
                df["eligible_universe"]
                & df["date"].between(
                    fold_start,
                    fold_end,
                    inclusive="both",
                )
            )

            print(
                f"OOS {year}: fitting "
                f"{model_name} on "
                f"{int(train_mask.sum()):,} "
                f"rows; scoring "
                f"{int(score_mask.sum()):,}"
            )

            model = build_model(
                model_name,
                random_state=(
                    random_state
                ),
            )
            fit_model(
                model,
                df,
                train_mask,
            )

            scores = predict_mask(
                model,
                df,
                score_mask,
            )
            df.loc[
                score_mask,
                "persistent_score",
            ] = scores

            saved = df.loc[
                score_mask,
                [
                    "date",
                    "canonical_security_id",
                ],
            ].copy()
            saved[
                "persistent_score"
            ] = scores
            saved.to_parquet(
                path,
                index=False,
                compression="zstd",
            )

        label_mask = (
            df[TRAINING_FLAG]
            & df[
                "target_rank_20d"
            ].notna()
            & df["date"].dt.year.eq(
                year
            )
            & df[
                "persistent_score"
            ].notna()
        )

        if label_mask.any():
            label_scores = df.loc[
                label_mask,
                "persistent_score",
            ].to_numpy(
                dtype=float
            )
            diagnostics = (
                cross_sectional_diagnostics(
                    df,
                    label_mask,
                    label_scores,
                )
            )
        else:
            diagnostics = {
                "rows": 0,
                "dates": 0,
                "mean_daily_ic": None,
                "median_daily_ic": None,
                "std_daily_ic": None,
                "icir": None,
                "positive_ic_fraction": None,
                "mean_top_decile_return": None,
                "mean_universe_return": None,
                "mean_top_decile_excess": None,
            }

        diagnostics_rows.append({
            "year": int(
                year
            ),
            **diagnostics,
        })

    add_daily_ranks(
        df
    )

    return pd.DataFrame(
        diagnostics_rows
    )


def actual_transaction_costs_by_year(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "actual_fees",
                "actual_turnover",
                "entries",
                "exits",
            ]
        )

    rows: list[
        dict
    ] = []

    entry = trades.loc[
        trades["entry_date"].notna()
    ].copy()
    if not entry.empty:
        entry["year"] = (
            pd.to_datetime(
                entry["entry_date"]
            ).dt.year
        )
        grouped = (
            entry.groupby(
                "year"
            )
            .agg(
                entry_fees=(
                    "entry_fees",
                    "sum",
                ),
                entry_turnover=(
                    "entry_trade_value",
                    "sum",
                ),
                entries=(
                    "canonical_security_id",
                    "size",
                ),
            )
            .reset_index()
        )
        rows.append(
            grouped
        )

    exit_frame = trades.loc[
        trades["exit_date"].notna()
    ].copy()
    if not exit_frame.empty:
        exit_frame["year"] = (
            pd.to_datetime(
                exit_frame[
                    "exit_date"
                ]
            ).dt.year
        )
        grouped = (
            exit_frame.groupby(
                "year"
            )
            .agg(
                exit_fees=(
                    "exit_fees",
                    "sum",
                ),
                exit_turnover=(
                    "exit_trade_value",
                    "sum",
                ),
                exits=(
                    "canonical_security_id",
                    "size",
                ),
            )
            .reset_index()
        )
        rows.append(
            grouped
        )

    if not rows:
        return pd.DataFrame()

    merged = rows[0]
    for frame in rows[1:]:
        merged = merged.merge(
            frame,
            on="year",
            how="outer",
        )

    for col in (
        "entry_fees",
        "exit_fees",
        "entry_turnover",
        "exit_turnover",
        "entries",
        "exits",
    ):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce",
        ).fillna(0.0)

    merged[
        "actual_fees"
    ] = (
        merged[
            "entry_fees"
        ]
        + merged[
            "exit_fees"
        ]
    )
    merged[
        "actual_turnover"
    ] = (
        merged[
            "entry_turnover"
        ]
        + merged[
            "exit_turnover"
        ]
    )

    return merged[
        [
            "year",
            "actual_fees",
            "actual_turnover",
            "entries",
            "exits",
        ]
    ].copy()


def annual_metrics_from_result(
    result: dict,
    *,
    annual_risk_free_rate: float,
    metadata: dict,
) -> pd.DataFrame:
    equity = result[
        "equity"
    ].copy()
    trades = result[
        "trades"
    ].copy()

    if equity.empty:
        return pd.DataFrame()

    equity["date"] = pd.to_datetime(
        equity["date"],
        errors="coerce",
    ).dt.normalize()
    equity["year"] = (
        equity["date"].dt.year
    )

    txn = (
        actual_transaction_costs_by_year(
            trades
        )
    )
    txn_lookup = (
        txn.set_index(
            "year"
        )
        if not txn.empty
        else pd.DataFrame()
    )

    rows: list[
        dict
    ] = []

    for year, curve in equity.groupby(
        "year",
        sort=True,
    ):
        curve = (
            curve.sort_values(
                "date"
            )
            .copy()
        )

        if len(curve) < 2:
            continue

        initial = float(
            curve[
                "equity"
            ].iloc[0]
        )

        if not trades.empty:
            exits = trades.loc[
                pd.to_datetime(
                    trades[
                        "exit_date"
                    ],
                    errors="coerce",
                ).dt.year.eq(
                    int(year)
                )
            ].copy()
        else:
            exits = pd.DataFrame()

        metrics = (
            compute_performance_metrics(
                curve,
                exits,
                initial_capital=(
                    initial
                ),
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
            )
        )

        if (
            not txn.empty
            and int(year)
            in txn_lookup.index
        ):
            item = txn_lookup.loc[
                int(year)
            ]
            actual_fees = float(
                item[
                    "actual_fees"
                ]
            )
            actual_turnover = float(
                item[
                    "actual_turnover"
                ]
            )
            entries = int(
                item[
                    "entries"
                ]
            )
            exits_count = int(
                item[
                    "exits"
                ]
            )
        else:
            actual_fees = 0.0
            actual_turnover = 0.0
            entries = 0
            exits_count = 0

        metrics[
            "total_fees"
        ] = actual_fees
        metrics[
            "turnover"
        ] = actual_turnover
        metrics[
            "fees_pct_initial_capital"
        ] = (
            actual_fees
            / initial
            if initial > 0
            else 0.0
        )
        metrics[
            "turnover_multiple"
        ] = (
            actual_turnover
            / initial
            if initial > 0
            else 0.0
        )
        metrics[
            "annualized_turnover_multiple"
        ] = metrics[
            "turnover_multiple"
        ]

        rows.append({
            "year": int(
                year
            ),
            "partial_year": bool(
                (
                    curve[
                        "date"
                    ].iloc[0].month
                    != 1
                )
                or (
                    curve[
                        "date"
                    ].iloc[-1].month
                    != 12
                )
            ),
            "date_start": str(
                curve[
                    "date"
                ].iloc[0].date()
            ),
            "date_end": str(
                curve[
                    "date"
                ].iloc[-1].date()
            ),
            "entries": entries,
            "exits": exits_count,
            **metadata,
            **metrics,
        })

    return pd.DataFrame(
        rows
    )


def surface_path(
    root: Path,
    *,
    cost_name: str,
) -> Path:
    return (
        root
        / "reports/ml/nested_generalization/"
        f"annual_config_surface_{cost_name}.csv"
    )


def build_config_surface(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    root: Path,
    top_k_values: list[int],
    hold_multipliers: list[int],
    score_gaps: list[float],
    capital: float,
    costs: CostProfile,
    cost_name: str,
    annual_risk_free_rate: float,
    oos_start_year: int,
    oos_end_year: int,
    rebuild: bool,
) -> pd.DataFrame:
    path = surface_path(
        root,
        cost_name=cost_name,
    )

    if (
        path.is_file()
        and not rebuild
    ):
        print(
            f"Reusing {cost_name} "
            "configuration surface"
        )
        return pd.read_csv(
            path
        )

    panel = (
        df.loc[
            df["date"].dt.year.between(
                oos_start_year,
                oos_end_year,
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows: list[
        pd.DataFrame
    ] = []

    total = (
        len(top_k_values)
        * len(
            hold_multipliers
        )
        * len(
            score_gaps
        )
    )
    number = 0

    print(
        f"\n=== BUILDING {cost_name.upper()} "
        "GENERALIZATION SURFACE ==="
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
                number += 1
                print(
                    f"[{number}/{total}] "
                    f"k={top_k} "
                    f"hold={multiplier}x "
                    f"gap={gap:.2f}"
                )

                result = (
                    run_persistent_backtest(
                        panel,
                        events,
                        name=(
                            "surface_"
                            f"k{top_k}_"
                            f"m{multiplier}_"
                            f"g{gap:.2f}"
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

                annual = (
                    annual_metrics_from_result(
                        result,
                        annual_risk_free_rate=(
                            annual_risk_free_rate
                        ),
                        metadata={
                            "cost_profile": (
                                cost_name
                            ),
                            "top_k": int(
                                top_k
                            ),
                            "hold_multiplier": int(
                                multiplier
                            ),
                            "hold_rank": int(
                                hold_rank
                            ),
                            "min_score_gap": float(
                                gap
                            ),
                        },
                    )
                )
                rows.append(
                    annual
                )

    surface = pd.concat(
        rows,
        ignore_index=True,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    surface.to_csv(
        path,
        index=False,
    )

    return surface


def robust_policy_scores(
    history: pd.DataFrame,
    *,
    evaluation_year: int,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    for (
        multiplier,
        gap,
    ), group in history.groupby(
        [
            "hold_multiplier",
            "min_score_gap",
        ],
        sort=True,
    ):
        sharpe = pd.to_numeric(
            group[
                "sharpe"
            ],
            errors="coerce",
        ).dropna()
        cagr = pd.to_numeric(
            group[
                "cagr"
            ],
            errors="coerce",
        ).dropna()
        turnover = pd.to_numeric(
            group[
                "turnover_multiple"
            ],
            errors="coerce",
        ).dropna()

        if sharpe.empty:
            continue

        rows.append({
            "evaluation_year": int(
                evaluation_year
            ),
            "hold_multiplier": int(
                multiplier
            ),
            "min_score_gap": float(
                gap
            ),
            "history_years": int(
                group[
                    "year"
                ].nunique()
            ),
            "history_k_values": int(
                group[
                    "top_k"
                ].nunique()
            ),
            "observations": int(
                len(group)
            ),
            "q25_sharpe": float(
                sharpe.quantile(
                    0.25
                )
            ),
            "median_sharpe": float(
                sharpe.median()
            ),
            "positive_sharpe_fraction": float(
                sharpe.gt(0).mean()
            ),
            "q25_cagr": float(
                cagr.quantile(
                    0.25
                )
            ),
            "median_cagr": float(
                cagr.median()
            ),
            "positive_cagr_fraction": float(
                cagr.gt(0).mean()
            ),
            "median_turnover_multiple": float(
                turnover.median()
            ),
            "worst_max_drawdown": float(
                pd.to_numeric(
                    group[
                        "max_drawdown"
                    ],
                    errors="coerce",
                ).min()
            ),
        })

    scores = pd.DataFrame(
        rows
    )

    if scores.empty:
        return scores

    return (
        scores.sort_values(
            [
                "q25_sharpe",
                "median_sharpe",
                "positive_cagr_fraction",
                "q25_cagr",
                "median_cagr",
                "median_turnover_multiple",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_nested_schedule(
    surface_net: pd.DataFrame,
    *,
    eval_start_year: int,
    eval_end_year: int,
    min_prior_years: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    selection_rows: list[
        dict
    ] = []
    score_frames: list[
        pd.DataFrame
    ] = []

    for year in range(
        eval_start_year,
        eval_end_year + 1,
    ):
        history = surface_net.loc[
            surface_net[
                "year"
            ].lt(year)
            & ~surface_net[
                "partial_year"
            ].fillna(False)
        ].copy()

        prior_years = sorted(
            history[
                "year"
            ].unique()
        )

        if len(
            prior_years
        ) < min_prior_years:
            raise RuntimeError(
                f"Evaluation year {year} "
                "has only "
                f"{len(prior_years)} prior "
                "complete OOS years."
            )

        scores = (
            robust_policy_scores(
                history,
                evaluation_year=(
                    year
                ),
            )
        )
        if scores.empty:
            raise RuntimeError(
                f"No policy scores for {year}"
            )

        scores[
            "selected"
        ] = False
        scores.loc[
            0,
            "selected",
        ] = True
        score_frames.append(
            scores
        )

        best = scores.iloc[0]

        selection_rows.append({
            "year": int(
                year
            ),
            "hold_multiplier": int(
                best[
                    "hold_multiplier"
                ]
            ),
            "min_score_gap": float(
                best[
                    "min_score_gap"
                ]
            ),
            "prior_year_start": int(
                min(
                    prior_years
                )
            ),
            "prior_year_end": int(
                max(
                    prior_years
                )
            ),
            "prior_year_count": int(
                len(
                    prior_years
                )
            ),
            "q25_sharpe": float(
                best[
                    "q25_sharpe"
                ]
            ),
            "median_sharpe": float(
                best[
                    "median_sharpe"
                ]
            ),
            "positive_cagr_fraction": float(
                best[
                    "positive_cagr_fraction"
                ]
            ),
            "median_cagr": float(
                best[
                    "median_cagr"
                ]
            ),
            "median_turnover_multiple": float(
                best[
                    "median_turnover_multiple"
                ]
            ),
        })

    return (
        pd.DataFrame(
            selection_rows
        ),
        pd.concat(
            score_frames,
            ignore_index=True,
        ),
    )


def first_aligned_signal_date(
    df: pd.DataFrame,
    *,
    oos_start_year: int,
    eval_start_year: int,
) -> pd.Timestamp:
    oos = (
        df.loc[
            df["date"].dt.year.between(
                oos_start_year,
                eval_start_year,
            ),
            [
                "date",
                "market_day_index",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "market_day_index"
        )
    )

    indices = (
        oos[
            "market_day_index"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )
    signal_indices = set(
        indices[
            ::HORIZON
        ]
    )

    candidate = oos.loc[
        oos[
            "date"
        ].dt.year.eq(
            eval_start_year
        )
        & oos[
            "market_day_index"
        ].astype(int).isin(
            signal_indices
        )
    ]

    if candidate.empty:
        raise RuntimeError(
            "Could not align evaluation "
            "start to the global "
            "rebalance phase."
        )

    return pd.Timestamp(
        candidate[
            "date"
        ].iloc[0]
    ).normalize()


def schedule_for_k(
    selections: pd.DataFrame,
    *,
    top_k: int,
) -> dict[
    int,
    dict,
]:
    return {
        int(row.year): {
            "hold_rank": int(
                top_k
                * int(
                    row.hold_multiplier
                )
            ),
            "min_score_gap": float(
                row.min_score_gap
            ),
        }
        for row in selections.itertuples(
            index=False
        )
    }


def fixed_schedule(
    *,
    years: list[int],
    top_k: int,
    multiplier: int,
    gap: float,
) -> dict[
    int,
    dict,
]:
    return {
        int(year): {
            "hold_rank": int(
                top_k
                * multiplier
            ),
            "min_score_gap": float(
                gap
            ),
        }
        for year in years
    }


def run_persistent_strategy(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    name: str,
    capital: float,
    top_k: int,
    schedule: dict[int, dict],
    costs: CostProfile,
    annual_risk_free_rate: float,
) -> dict:
    first_policy = schedule[
        min(
            schedule
        )
    ]

    return run_persistent_backtest(
        panel,
        events,
        name=name,
        initial_capital=capital,
        top_k=top_k,
        hold_rank=int(
            first_policy[
                "hold_rank"
            ]
        ),
        min_score_gap=float(
            first_policy[
                "min_score_gap"
            ]
        ),
        rebalance_sessions=(
            HORIZON
        ),
        costs=costs,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        policy_schedule=(
            schedule
        ),
    )


def benchmark_rows(
    root: Path,
    *,
    panel_dates: set[
        pd.Timestamp
    ],
    capital: float,
    annual_risk_free_rate: float,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        pd.DataFrame,
    ],
]:
    path = (
        root
        / "data/processed/index_benchmarks/"
        "nse_price_indices.parquet"
    )
    data = pd.read_parquet(
        path
    )

    rows: list[
        dict
    ] = []
    curves: dict[
        str,
        pd.DataFrame,
    ] = {}

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
                panel_dates
            ),
        )
        curves[
            name
        ] = curve

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
            "kind": (
                "price_index"
            ),
            "cost_profile": (
                "theoretical_index"
            ),
            "top_k": None,
            **metrics,
        })

    return (
        pd.DataFrame(
            rows
        ),
        curves,
    )


def aggregate_annual_generalization(
    annual: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    group_cols = [
        "name",
        "kind",
        "cost_profile",
        "top_k",
    ]

    for keys, group in annual.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):
        (
            name,
            kind,
            cost_profile,
            top_k,
        ) = keys

        cagr = pd.to_numeric(
            group["cagr"],
            errors="coerce",
        ).dropna()
        sharpe = pd.to_numeric(
            group["sharpe"],
            errors="coerce",
        ).dropna()
        turnover = pd.to_numeric(
            group[
                "turnover_multiple"
            ],
            errors="coerce",
        ).dropna()

        rows.append({
            "name": name,
            "kind": kind,
            "cost_profile": (
                cost_profile
            ),
            "top_k": top_k,
            "years": int(
                group[
                    "year"
                ].nunique()
            ),
            "mean_cagr": float(
                cagr.mean()
            ),
            "median_cagr": float(
                cagr.median()
            ),
            "q25_cagr": float(
                cagr.quantile(
                    0.25
                )
            ),
            "positive_cagr_fraction": float(
                cagr.gt(0).mean()
            ),
            "mean_sharpe": float(
                sharpe.mean()
            ),
            "median_sharpe": float(
                sharpe.median()
            ),
            "q25_sharpe": float(
                sharpe.quantile(
                    0.25
                )
            ),
            "positive_sharpe_fraction": float(
                sharpe.gt(0).mean()
            ),
            "median_max_drawdown": float(
                pd.to_numeric(
                    group[
                        "max_drawdown"
                    ],
                    errors="coerce",
                ).median()
            ),
            "worst_max_drawdown": float(
                pd.to_numeric(
                    group[
                        "max_drawdown"
                    ],
                    errors="coerce",
                ).min()
            ),
            "median_turnover_multiple": float(
                turnover.median()
            ),
            "total_fees": float(
                pd.to_numeric(
                    group[
                        "total_fees"
                    ],
                    errors="coerce",
                ).fillna(
                    0
                ).sum()
            ),
            "median_profit_factor": float(
                pd.to_numeric(
                    group[
                        "profit_factor"
                    ],
                    errors="coerce",
                ).median()
            ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "median_cagr",
                "median_sharpe",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def self_test() -> None:
    surface = pd.DataFrame({
        "year": [
            2016,
            2016,
            2017,
            2017,
            2018,
            2018,
        ],
        "partial_year": [
            False,
        ] * 6,
        "top_k": [
            5,
            5,
            5,
            5,
            5,
            5,
        ],
        "hold_multiplier": [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
        "min_score_gap": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "sharpe": [
            0.4,
            0.1,
            0.5,
            -0.2,
            0.3,
            0.0,
        ],
        "cagr": [
            0.10,
            0.05,
            0.12,
            -0.03,
            0.08,
            0.01,
        ],
        "turnover_multiple": [
            5,
            3,
            6,
            3,
            4,
            2,
        ],
        "max_drawdown": [
            -0.2,
            -0.1,
            -0.2,
            -0.15,
            -0.18,
            -0.12,
        ],
    })

    scores = robust_policy_scores(
        surface,
        evaluation_year=2019,
    )
    assert not scores.empty
    assert int(
        scores.iloc[0][
            "hold_multiplier"
        ]
    ) == 1

    selection, _ = (
        build_nested_schedule(
            surface,
            eval_start_year=2019,
            eval_end_year=2019,
            min_prior_years=3,
        )
    )
    assert len(
        selection
    ) == 1

    schedule = schedule_for_k(
        selection,
        top_k=10,
    )
    assert schedule[
        2019
    ][
        "hold_rank"
    ] == 10

    print(
        "Nested generalization "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generalization-first nested annual "
            "walk-forward evaluation of persistent "
            "ML portfolios across years and breadths."
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
        default=(
            DEFAULT_EVAL_START_YEAR
        ),
    )
    ap.add_argument(
        "--min-prior-years",
        type=int,
        default=3,
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
        "--rebuild-predictions",
        action="store_true",
    )
    ap.add_argument(
        "--rebuild-surfaces",
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

    if (
        args.eval_start_year
        <= args.oos_start_year
    ):
        raise SystemExit(
            "--eval-start-year must be "
            "after --oos-start-year."
        )
    if (
        args.oos_end_year
        < args.eval_start_year
    ):
        raise SystemExit(
            "--oos-end-year must be >= "
            "--eval-start-year."
        )
    if any(
        k < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k values must be >= 1."
        )
    if any(
        value < 1
        for value
        in args.hold_multipliers
    ):
        raise SystemExit(
            "--hold-multipliers must be >= 1."
        )
    if any(
        value < 0
        for value
        in args.score_gaps
    ):
        raise SystemExit(
            "--score-gaps must be >= 0."
        )

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "nested_generalization"
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

    diagnostics = (
        build_annual_oos_predictions(
            df,
            root=root,
            model_name=args.model,
            years=years,
            random_state=(
                args.random_state
            ),
            rebuild=(
                args.rebuild_predictions
            ),
        )
    )
    diagnostics.to_csv(
        output_root
        / "annual_base_model_diagnostics.csv",
        index=False,
    )

    events = load_mechanical_events(
        root
    )
    current_costs = (
        current_cost_profile(
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
    zero_costs = (
        zero_cost_profile()
    )

    surface_net = (
        build_config_surface(
            df,
            events,
            root=root,
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
            costs=current_costs,
            cost_name="net",
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
            oos_start_year=(
                args.oos_start_year
            ),
            oos_end_year=(
                args.oos_end_year
            ),
            rebuild=(
                args.rebuild_surfaces
            ),
        )
    )

    surface_gross = (
        build_config_surface(
            df,
            events,
            root=root,
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
            costs=zero_costs,
            cost_name="gross",
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
            oos_start_year=(
                args.oos_start_year
            ),
            oos_end_year=(
                args.oos_end_year
            ),
            rebuild=(
                args.rebuild_surfaces
            ),
        )
    )

    (
        selections,
        selection_scores,
    ) = build_nested_schedule(
        surface_net,
        eval_start_year=(
            args.eval_start_year
        ),
        eval_end_year=(
            args.oos_end_year
        ),
        min_prior_years=(
            args.min_prior_years
        ),
    )

    selections.to_csv(
        output_root
        / "nested_policy_by_year.csv",
        index=False,
    )
    selection_scores.to_csv(
        output_root
        / "nested_policy_candidate_scores.csv",
        index=False,
    )

    stability = (
        selections.groupby(
            [
                "hold_multiplier",
                "min_score_gap",
            ],
            sort=False,
        )
        .agg(
            selected_years=(
                "year",
                "size",
            ),
            first_year=(
                "year",
                "min",
            ),
            last_year=(
                "year",
                "max",
            ),
        )
        .reset_index()
    )
    stability[
        "selection_fraction"
    ] = (
        stability[
            "selected_years"
        ]
        / len(
            selections
        )
    )
    stability.to_csv(
        output_root
        / "policy_stability.csv",
        index=False,
    )

    # Freeze one policy using only the pre-evaluation OOS years. This is a
    # strong control against the annually adaptive nested policy.
    pre_eval_history = (
        surface_net.loc[
            surface_net[
                "year"
            ].lt(
                args.eval_start_year
            )
            & ~surface_net[
                "partial_year"
            ].fillna(False)
        ]
        .copy()
    )
    frozen_scores = (
        robust_policy_scores(
            pre_eval_history,
            evaluation_year=(
                args.eval_start_year
            ),
        )
    )
    frozen = frozen_scores.iloc[0]
    frozen_multiplier = int(
        frozen[
            "hold_multiplier"
        ]
    )
    frozen_gap = float(
        frozen[
            "min_score_gap"
        ]
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

    panel = (
        df.loc[
            df["date"].between(
                eval_start_date,
                pd.Timestamp(
                    f"{args.oos_end_year}-12-31"
                ),
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    eval_years = list(
        range(
            args.eval_start_year,
            args.oos_end_year
            + 1,
        )
    )

    final_rows: list[
        dict
    ] = []
    annual_rows: list[
        pd.DataFrame
    ] = []

    print(
        "\n=== EXECUTABLE NESTED "
        "GENERALIZATION TEST ==="
    )
    print(
        "Evaluation starts on aligned "
        f"rebalance date {eval_start_date.date()}"
    )
    print(
        "Frozen pre-eval universal policy: "
        f"hold={frozen_multiplier}x, "
        f"gap={frozen_gap:.2f}"
    )

    persistent_specs: list[
        tuple[
            str,
            str,
        ]
    ] = [
        (
            "nested_dynamic",
            "dynamic",
        ),
        (
            "frozen_pre_eval",
            "frozen",
        ),
        (
            "buffer_only",
            "buffer",
        ),
    ]

    for top_k in args.top_k:
        nested_schedule = (
            schedule_for_k(
                selections,
                top_k=int(
                    top_k
                ),
            )
        )
        frozen_sched = (
            fixed_schedule(
                years=eval_years,
                top_k=int(
                    top_k
                ),
                multiplier=(
                    frozen_multiplier
                ),
                gap=frozen_gap,
            )
        )
        buffer_sched = (
            fixed_schedule(
                years=eval_years,
                top_k=int(
                    top_k
                ),
                multiplier=1,
                gap=0.0,
            )
        )
        schedules = {
            "dynamic": (
                nested_schedule
            ),
            "frozen": (
                frozen_sched
            ),
            "buffer": (
                buffer_sched
            ),
        }

        for (
            cost_name,
            profile,
        ) in (
            (
                "net",
                current_costs,
            ),
            (
                "gross",
                zero_costs,
            ),
        ):
            for (
                label,
                schedule_key,
            ) in persistent_specs:
                result = (
                    run_persistent_strategy(
                        panel,
                        events,
                        name=label,
                        capital=args.capital,
                        top_k=int(
                            top_k
                        ),
                        schedule=(
                            schedules[
                                schedule_key
                            ]
                        ),
                        costs=profile,
                        annual_risk_free_rate=(
                            args.risk_free_rate
                        ),
                    )
                )

                final_rows.append({
                    "name": (
                        f"{label}_k{top_k}"
                    ),
                    "kind": (
                        "persistent_ml"
                    ),
                    "cost_profile": (
                        cost_name
                    ),
                    "top_k": int(
                        top_k
                    ),
                    **result[
                        "metrics"
                    ],
                })

                annual = (
                    annual_metrics_from_result(
                        result,
                        annual_risk_free_rate=(
                            args.risk_free_rate
                        ),
                        metadata={
                            "name": (
                                f"{label}_k{top_k}"
                            ),
                            "kind": (
                                "persistent_ml"
                            ),
                            "cost_profile": (
                                cost_name
                            ),
                            "top_k": int(
                                top_k
                            ),
                        },
                    )
                )
                annual_rows.append(
                    annual
                )

        # Cohort ML and deterministic momentum are controls. Run both gross
        # and net so friction is visible across every active strategy family.
        for (
            cost_name,
            profile,
        ) in (
            (
                "net",
                current_costs,
            ),
            (
                "gross",
                zero_costs,
            ),
        ):
            cohort = run_backtest(
                panel,
                events,
                strategy=(
                    "cohort_ml"
                ),
                score_column=(
                    "persistent_score"
                ),
                score_direction=1.0,
                initial_capital=(
                    args.capital
                ),
                top_k=int(
                    top_k
                ),
                holding_sessions=(
                    HORIZON
                ),
                costs=profile,
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
            )
            final_rows.append({
                "name": (
                    f"cohort_ml_k{top_k}"
                ),
                "kind": (
                    "cohort_ml"
                ),
                "cost_profile": (
                    cost_name
                ),
                "top_k": int(
                    top_k
                ),
                **cohort[
                    "metrics"
                ],
            })
            annual_rows.append(
                annual_metrics_from_result(
                    cohort,
                    annual_risk_free_rate=(
                        args.risk_free_rate
                    ),
                    metadata={
                        "name": (
                            f"cohort_ml_k{top_k}"
                        ),
                        "kind": (
                            "cohort_ml"
                        ),
                        "cost_profile": (
                            cost_name
                        ),
                        "top_k": int(
                            top_k
                        ),
                    },
                )
            )

            momentum = run_backtest(
                panel,
                events,
                strategy=(
                    "momentum_60d"
                ),
                initial_capital=(
                    args.capital
                ),
                top_k=int(
                    top_k
                ),
                holding_sessions=(
                    HORIZON
                ),
                costs=profile,
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
            )
            final_rows.append({
                "name": (
                    f"momentum_60d_k{top_k}"
                ),
                "kind": (
                    "deterministic"
                ),
                "cost_profile": (
                    cost_name
                ),
                "top_k": int(
                    top_k
                ),
                **momentum[
                    "metrics"
                ],
            })
            annual_rows.append(
                annual_metrics_from_result(
                    momentum,
                    annual_risk_free_rate=(
                        args.risk_free_rate
                    ),
                    metadata={
                        "name": (
                            f"momentum_60d_k{top_k}"
                        ),
                        "kind": (
                            "deterministic"
                        ),
                        "cost_profile": (
                            cost_name
                        ),
                        "top_k": int(
                            top_k
                        ),
                    },
                )
            )

    benchmark, curves = benchmark_rows(
        root,
        panel_dates=set(
            panel[
                "date"
            ].dropna()
        ),
        capital=args.capital,
        annual_risk_free_rate=(
            args.risk_free_rate
        ),
    )
    final_rows.extend(
        benchmark.to_dict(
            orient="records"
        )
    )

    for name, curve in (
        curves.items()
    ):
        annual_rows.append(
            annual_metrics_from_result(
                {
                    "equity": curve[
                        [
                            "date",
                            "equity",
                        ]
                    ],
                    "trades": (
                        pd.DataFrame()
                    ),
                },
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
                metadata={
                    "name": name,
                    "kind": (
                        "price_index"
                    ),
                    "cost_profile": (
                        "theoretical_index"
                    ),
                    "top_k": None,
                },
            )
        )

    final = pd.DataFrame(
        final_rows
    )
    annual = pd.concat(
        annual_rows,
        ignore_index=True,
    )
    aggregate = (
        aggregate_annual_generalization(
            annual
        )
    )

    final.to_csv(
        output_root
        / "final_generalization_comparison.csv",
        index=False,
    )
    annual.to_csv(
        output_root
        / "annual_generalization_metrics.csv",
        index=False,
    )
    aggregate.to_csv(
        output_root
        / "aggregate_generalization.csv",
        index=False,
    )

    # Explicit gross-net drag for every strategy/top-k pair that has both.
    active = final.loc[
        final[
            "cost_profile"
        ].isin(
            [
                "gross",
                "net",
            ]
        )
    ].copy()
    drag = (
        active.pivot_table(
            index=[
                "name",
                "kind",
                "top_k",
            ],
            columns=(
                "cost_profile"
            ),
            values=[
                "cagr",
                "ending_equity",
                "sharpe",
                "turnover_multiple",
                "total_fees",
            ],
            aggfunc="first",
        )
    )
    drag.columns = [
        f"{metric}_{profile}"
        for metric, profile
        in drag.columns
    ]
    drag = drag.reset_index()
    if (
        "cagr_gross"
        in drag.columns
        and "cagr_net"
        in drag.columns
    ):
        drag[
            "cagr_cost_drag_pp"
        ] = (
            drag[
                "cagr_gross"
            ]
            - drag[
                "cagr_net"
            ]
        ) * 100.0
    drag.to_csv(
        output_root
        / "gross_net_decomposition.csv",
        index=False,
    )

    caption_frames: list[
        pd.DataFrame
    ] = []
    for row in final.to_dict(
        orient="records"
    ):
        name = str(
            row.get(
                "name",
                ""
            )
        )
        frame = captioned_metric_rows(
            row
        )
        frame[
            "name"
        ] = name
        caption_frames.append(
            frame
        )
    pd.concat(
        caption_frames,
        ignore_index=True,
    ).to_csv(
        output_root
        / "final_generalization_captioned.csv",
        index=False,
    )

    summary = {
        "goal": (
            "generalization-first persistence "
            "evaluation across multiple years "
            "and portfolio breadths"
        ),
        "base_model": (
            args.model
        ),
        "oos_prediction_years": (
            years
        ),
        "evaluation_start": str(
            eval_start_date.date()
        ),
        "evaluation_end_year": (
            args.oos_end_year
        ),
        "portfolio_breadths": [
            int(x)
            for x in args.top_k
        ],
        "candidate_hold_multipliers": [
            int(x)
            for x
            in args.hold_multipliers
        ],
        "candidate_score_gaps": [
            float(x)
            for x
            in args.score_gaps
        ],
        "selection_rule": (
            "for each evaluation year, use only "
            "prior complete OOS years and aggregate "
            "each persistence rule across every "
            "portfolio breadth; rank first by 25th-"
            "percentile Sharpe, then median Sharpe, "
            "positive-CAGR fraction, 25th-percentile "
            "CAGR, median CAGR, and lower turnover"
        ),
        "universal_across_k": True,
        "min_prior_oos_years": int(
            args.min_prior_years
        ),
        "frozen_pre_eval_policy": {
            "hold_multiplier": (
                frozen_multiplier
            ),
            "min_score_gap": (
                frozen_gap
            ),
        },
        "cost_profile": asdict(
            current_costs
        ),
        "risk_free_rate_annual": float(
            args.risk_free_rate
        ),
        "important_limitation": (
            "2022-2026 has already been inspected "
            "during development, so this is nested "
            "walk-forward generalization analysis, "
            "not a pristine never-seen final holdout"
        ),
        "outputs": {
            "base_model_diagnostics": (
                "annual_base_model_diagnostics.csv"
            ),
            "surface_net": (
                "annual_config_surface_net.csv"
            ),
            "surface_gross": (
                "annual_config_surface_gross.csv"
            ),
            "policy_by_year": (
                "nested_policy_by_year.csv"
            ),
            "candidate_scores": (
                "nested_policy_candidate_scores.csv"
            ),
            "policy_stability": (
                "policy_stability.csv"
            ),
            "final_comparison": (
                "final_generalization_comparison.csv"
            ),
            "annual_metrics": (
                "annual_generalization_metrics.csv"
            ),
            "aggregate": (
                "aggregate_generalization.csv"
            ),
            "gross_net": (
                "gross_net_decomposition.csv"
            ),
        },
    }

    (
        output_root
        / "nested_generalization_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(
        "\n=== NESTED POLICY BY YEAR ==="
    )
    print(
        selections[
            [
                "year",
                "hold_multiplier",
                "min_score_gap",
                "q25_sharpe",
                "median_sharpe",
                "positive_cagr_fraction",
                "median_cagr",
                "median_turnover_multiple",
            ]
        ].to_string(
            index=False,
            formatters={
                "q25_sharpe": (
                    lambda x: (
                        f"{x:.3f}"
                    )
                ),
                "median_sharpe": (
                    lambda x: (
                        f"{x:.3f}"
                    )
                ),
                "positive_cagr_fraction": (
                    lambda x: (
                        f"{x:.1%}"
                    )
                ),
                "median_cagr": (
                    lambda x: (
                        f"{x:.2%}"
                    )
                ),
                "median_turnover_multiple": (
                    lambda x: (
                        f"{x:.1f}x"
                    )
                ),
            },
        )
    )

    preview = final.loc[
        final[
            "cost_profile"
        ].isin(
            [
                "net",
                "theoretical_index",
            ]
        )
    ][
        [
            col
            for col in (
                "name",
                "kind",
                "cost_profile",
                "top_k",
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
            if col in final.columns
        ]
    ].copy()

    for col in (
        "cagr",
        "max_drawdown",
    ):
        preview[col] = (
            pd.to_numeric(
                preview[col],
                errors="coerce",
            )
            * 100.0
        )

    preview = preview.sort_values(
        [
            "cagr",
            "sharpe",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print(
        "\n=== FINAL GENERALIZATION "
        "COMPARISON (NET) ==="
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
        "\n=== AGGREGATE YEAR-BY-YEAR "
        "GENERALIZATION ==="
    )
    agg_preview = aggregate.loc[
        aggregate[
            "cost_profile"
        ].isin(
            [
                "net",
                "theoretical_index",
            ]
        )
    ][
        [
            "name",
            "top_k",
            "years",
            "median_cagr",
            "q25_cagr",
            "positive_cagr_fraction",
            "median_sharpe",
            "q25_sharpe",
            "worst_max_drawdown",
            "median_turnover_multiple",
        ]
    ].copy()

    for col in (
        "median_cagr",
        "q25_cagr",
        "worst_max_drawdown",
    ):
        agg_preview[col] = (
            agg_preview[
                col
            ]
            * 100.0
        )

    print(
        agg_preview.to_string(
            index=False,
            formatters={
                "median_cagr": (
                    lambda x: (
                        f"{x:.2f}%"
                    )
                ),
                "q25_cagr": (
                    lambda x: (
                        f"{x:.2f}%"
                    )
                ),
                "positive_cagr_fraction": (
                    lambda x: (
                        f"{x:.1%}"
                    )
                ),
                "median_sharpe": (
                    lambda x: (
                        f"{x:.3f}"
                    )
                ),
                "q25_sharpe": (
                    lambda x: (
                        f"{x:.3f}"
                    )
                ),
                "worst_max_drawdown": (
                    lambda x: (
                        f"{x:.2f}%"
                    )
                ),
                "median_turnover_multiple": (
                    lambda x: (
                        f"{x:.1f}x"
                    )
                ),
            },
        )
    )

    print(
        "\n=== NESTED GENERALIZATION "
        "EXPERIMENT COMPLETE ==="
    )
    print(
        f"Outputs: {output_root}"
    )


if __name__ == "__main__":
    main()
