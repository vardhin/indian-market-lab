from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
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
    FEATURE_COLUMNS,
    HORIZON,
    TARGET_COLUMN,
    TRAINING_FLAG,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    TEST_START,
    TEST_END,
    add_index_benchmarks,
    add_target_rank,
    build_model,
    cross_sectional_diagnostics,
    fit_model,
    load_ml_panel,
    maybe_sample_mask,
    predict_mask,
)
from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
    run_backtest,
)


DEFAULT_TOP_K = [
    5,
    10,
]
DEFAULT_ALPHAS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
]
TAIL_FRACTIONS = [
    0.20,
    0.10,
    0.05,
    0.02,
    0.01,
]
TAIL_COUNTS = [
    10,
    5,
]
WALK_FORWARD_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026,
]


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


def add_daily_score_ranks(
    df: pd.DataFrame,
    *,
    raw_score_column: str,
    output_prefix: str,
    mask: pd.Series,
) -> None:
    df[
        f"{output_prefix}_pct"
    ] = np.nan
    df[
        f"{output_prefix}_mom60_pct"
    ] = np.nan

    view = df.loc[
        mask,
        [
            "date",
            raw_score_column,
            "return_60d",
        ],
    ].copy()

    view[
        f"{output_prefix}_pct"
    ] = (
        view.groupby(
            "date",
            sort=False,
        )[raw_score_column]
        .rank(
            method="average",
            pct=True,
        )
    )

    view[
        f"{output_prefix}_mom60_pct"
    ] = (
        view.groupby(
            "date",
            sort=False,
        )["return_60d"]
        .rank(
            method="average",
            pct=True,
        )
    )

    df.loc[
        view.index,
        f"{output_prefix}_pct",
    ] = view[
        f"{output_prefix}_pct"
    ].to_numpy()

    df.loc[
        view.index,
        f"{output_prefix}_mom60_pct",
    ] = view[
        f"{output_prefix}_mom60_pct"
    ].to_numpy()


def decile_analysis(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    score_pct_column: str,
    split_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    view = df.loc[
        mask,
        [
            "date",
            TARGET_COLUMN,
            score_pct_column,
        ],
    ].dropna().copy()

    if view.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    view["decile"] = np.ceil(
        view[
            score_pct_column
        ].clip(
            lower=1e-12,
            upper=1.0,
        )
        * 10.0
    ).astype(int)

    daily = (
        view.groupby(
            [
                "date",
                "decile",
            ],
            sort=True,
        )
        .agg(
            realized_return=(
                TARGET_COLUMN,
                "mean",
            ),
            rows=(
                TARGET_COLUMN,
                "size",
            ),
        )
        .reset_index()
    )
    daily["split"] = (
        split_name
    )

    summary = (
        daily.groupby(
            "decile",
            sort=True,
        )
        .agg(
            mean_daily_return=(
                "realized_return",
                "mean",
            ),
            median_daily_return=(
                "realized_return",
                "median",
            ),
            positive_day_fraction=(
                "realized_return",
                lambda x: float(
                    pd.Series(x)
                    .gt(0)
                    .mean()
                ),
            ),
            dates=(
                "date",
                "nunique",
            ),
            mean_rows_per_date=(
                "rows",
                "mean",
            ),
        )
        .reset_index()
    )
    summary["split"] = (
        split_name
    )

    return (
        daily,
        summary,
    )


def top_tail_analysis(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    score_pct_column: str,
    raw_score_column: str,
    split_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    view = df.loc[
        mask,
        [
            "date",
            TARGET_COLUMN,
            score_pct_column,
            raw_score_column,
        ],
    ].dropna().copy()

    if view.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    daily_rows: list[
        dict
    ] = []

    for day, group in view.groupby(
        "date",
        sort=True,
    ):
        universe_return = float(
            group[
                TARGET_COLUMN
            ].mean()
        )

        for fraction in (
            TAIL_FRACTIONS
        ):
            selected = group.loc[
                group[
                    score_pct_column
                ].ge(
                    1.0 - fraction
                )
            ]

            if selected.empty:
                continue

            realized = float(
                selected[
                    TARGET_COLUMN
                ].mean()
            )

            daily_rows.append({
                "split": split_name,
                "date": day,
                "selection": (
                    f"top_{int(fraction * 100)}pct"
                ),
                "selection_type": (
                    "fraction"
                ),
                "selection_value": float(
                    fraction
                ),
                "selected_rows": int(
                    len(selected)
                ),
                "realized_return": (
                    realized
                ),
                "universe_return": (
                    universe_return
                ),
                "excess_return": (
                    realized
                    - universe_return
                ),
            })

        ordered = group.sort_values(
            raw_score_column,
            ascending=False,
            kind="stable",
        )

        for count in TAIL_COUNTS:
            selected = ordered.head(
                count
            )

            if selected.empty:
                continue

            realized = float(
                selected[
                    TARGET_COLUMN
                ].mean()
            )

            daily_rows.append({
                "split": split_name,
                "date": day,
                "selection": (
                    f"top_{count}"
                ),
                "selection_type": (
                    "count"
                ),
                "selection_value": float(
                    count
                ),
                "selected_rows": int(
                    len(selected)
                ),
                "realized_return": (
                    realized
                ),
                "universe_return": (
                    universe_return
                ),
                "excess_return": (
                    realized
                    - universe_return
                ),
            })

    daily = pd.DataFrame(
        daily_rows
    )

    if daily.empty:
        return (
            daily,
            pd.DataFrame(),
        )

    summary = (
        daily.groupby(
            [
                "split",
                "selection",
                "selection_type",
                "selection_value",
            ],
            sort=False,
        )
        .agg(
            mean_realized_return=(
                "realized_return",
                "mean",
            ),
            median_realized_return=(
                "realized_return",
                "median",
            ),
            mean_universe_return=(
                "universe_return",
                "mean",
            ),
            mean_excess_return=(
                "excess_return",
                "mean",
            ),
            positive_excess_fraction=(
                "excess_return",
                lambda x: float(
                    pd.Series(x)
                    .gt(0)
                    .mean()
                ),
            ),
            dates=(
                "date",
                "nunique",
            ),
            mean_selected_rows=(
                "selected_rows",
                "mean",
            ),
        )
        .reset_index()
    )

    return (
        daily,
        summary,
    )


def _set_overlap(
    left: set[str],
    right: set[str],
) -> tuple[
    float,
    float,
]:
    if (
        not left
        and not right
    ):
        return (
            1.0,
            1.0,
        )

    union = left | right

    jaccard = (
        len(left & right)
        / len(union)
        if union
        else 1.0
    )

    denominator = min(
        len(left),
        len(right),
    )
    overlap = (
        len(left & right)
        / denominator
        if denominator
        else 0.0
    )

    return (
        float(jaccard),
        float(overlap),
    )


def persistence_analysis(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    score_pct_column: str,
    raw_score_column: str,
    split_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    view = df.loc[
        mask,
        [
            "date",
            "canonical_security_id",
            raw_score_column,
            score_pct_column,
        ],
    ].dropna().copy()

    dates = list(
        sorted(
            view["date"].unique()
        )
    )

    if len(dates) < 2:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    by_date = {
        pd.Timestamp(day): (
            group.set_index(
                "canonical_security_id"
            )[
                [
                    raw_score_column,
                    score_pct_column,
                ]
            ]
        )
        for day, group in (
            view.groupby(
                "date",
                sort=True,
            )
        )
    }

    rows: list[
        dict
    ] = []

    for lag in (
        1,
        HORIZON,
    ):
        for position in range(
            lag,
            len(dates),
        ):
            left_date = pd.Timestamp(
                dates[
                    position - lag
                ]
            )
            right_date = pd.Timestamp(
                dates[position]
            )

            left = by_date[
                left_date
            ]
            right = by_date[
                right_date
            ]

            common = (
                left.index.intersection(
                    right.index
                )
            )

            rank_corr = np.nan

            if len(common) >= 10:
                rank_corr = (
                    left.loc[
                        common,
                        score_pct_column,
                    ]
                    .corr(
                        right.loc[
                            common,
                            score_pct_column,
                        ],
                        method="spearman",
                    )
                )

            row = {
                "split": split_name,
                "lag_sessions": int(
                    lag
                ),
                "left_date": left_date,
                "right_date": right_date,
                "common_securities": int(
                    len(common)
                ),
                "rank_spearman": (
                    float(rank_corr)
                    if not pd.isna(
                        rank_corr
                    )
                    else np.nan
                ),
            }

            left_decile = set(
                left.index[
                    left[
                        score_pct_column
                    ].ge(0.90)
                ]
            )
            right_decile = set(
                right.index[
                    right[
                        score_pct_column
                    ].ge(0.90)
                ]
            )
            (
                row[
                    "top_decile_jaccard"
                ],
                row[
                    "top_decile_overlap"
                ],
            ) = _set_overlap(
                left_decile,
                right_decile,
            )

            for top_k in (
                5,
                10,
                20,
                50,
            ):
                left_top = set(
                    left.nlargest(
                        top_k,
                        raw_score_column,
                    ).index
                )
                right_top = set(
                    right.nlargest(
                        top_k,
                        raw_score_column,
                    ).index
                )

                (
                    jaccard,
                    overlap,
                ) = _set_overlap(
                    left_top,
                    right_top,
                )

                row[
                    f"top{top_k}_jaccard"
                ] = jaccard
                row[
                    f"top{top_k}_overlap"
                ] = overlap
                row[
                    f"top{top_k}_turnover_proxy"
                ] = (
                    1.0 - overlap
                )

            rows.append(row)

    daily = pd.DataFrame(
        rows
    )

    summary = (
        daily.groupby(
            [
                "split",
                "lag_sessions",
            ],
            sort=True,
        )
        .agg(
            observations=(
                "right_date",
                "size",
            ),
            mean_rank_spearman=(
                "rank_spearman",
                "mean",
            ),
            median_rank_spearman=(
                "rank_spearman",
                "median",
            ),
            mean_top_decile_jaccard=(
                "top_decile_jaccard",
                "mean",
            ),
            mean_top_decile_overlap=(
                "top_decile_overlap",
                "mean",
            ),
            mean_top5_turnover_proxy=(
                "top5_turnover_proxy",
                "mean",
            ),
            mean_top10_turnover_proxy=(
                "top10_turnover_proxy",
                "mean",
            ),
            mean_top20_turnover_proxy=(
                "top20_turnover_proxy",
                "mean",
            ),
            mean_top50_turnover_proxy=(
                "top50_turnover_proxy",
                "mean",
            ),
        )
        .reset_index()
    )

    return (
        daily,
        summary,
    )


def validation_masks(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    train_cutoff_index = int(
        df.loc[
            df["date"].le(
                TRAIN_END
            ),
            "market_day_index",
        ]
        .dropna()
        .max()
    )

    validation_cutoff_index = int(
        df.loc[
            df["date"].le(
                VALIDATION_END
            ),
            "market_day_index",
        ]
        .dropna()
        .max()
    )

    train_mask = (
        df[TRAINING_FLAG]
        & df[
            "target_rank_20d"
        ].notna()
        & df["date"].between(
            TRAIN_START,
            TRAIN_END,
            inclusive="both",
        )
        & df[
            "market_day_index"
        ].le(
            train_cutoff_index
            - HORIZON
        )
    )

    validation_score_mask = (
        df["eligible_universe"]
        & df["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
    )

    validation_label_mask = (
        df[TRAINING_FLAG]
        & df[
            "target_rank_20d"
        ].notna()
        & df["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
        & df[
            "market_day_index"
        ].le(
            validation_cutoff_index
            - HORIZON
        )
    )

    return (
        train_mask,
        validation_score_mask,
        validation_label_mask,
    )


def fit_validation_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    max_train_rows: int,
    random_state: int,
) -> tuple[
    object,
    pd.Series,
    pd.Series,
]:
    (
        train_mask,
        validation_score_mask,
        validation_label_mask,
    ) = validation_masks(
        df
    )

    sampled_train = (
        maybe_sample_mask(
            train_mask,
            max_rows=(
                max_train_rows
            ),
            random_state=(
                random_state
            ),
        )
    )

    print(
        "\nFitting frozen model "
        f"{model_name} on "
        f"{int(sampled_train.sum()):,} "
        "pre-validation rows..."
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
        sampled_train,
    )

    df[
        "validation_ml_score"
    ] = np.nan

    scores = predict_mask(
        model,
        df,
        validation_score_mask,
    )
    df.loc[
        validation_score_mask,
        "validation_ml_score",
    ] = scores

    add_daily_score_ranks(
        df,
        raw_score_column=(
            "validation_ml_score"
        ),
        output_prefix=(
            "validation_ml"
        ),
        mask=(
            validation_score_mask
        ),
    )

    return (
        model,
        validation_score_mask,
        validation_label_mask,
    )


def add_ensemble_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    alphas: list[float],
    score_mask: pd.Series,
) -> dict[
    float,
    str,
]:
    ml_pct = (
        f"{prefix}_pct"
    )
    mom_pct = (
        f"{prefix}_mom60_pct"
    )

    columns: dict[
        float,
        str,
    ] = {}

    for alpha in alphas:
        tag = (
            f"{alpha:.2f}"
            .replace(
                ".",
                "p",
            )
        )
        column = (
            f"{prefix}_ensemble_"
            f"a{tag}"
        )

        df[column] = np.nan
        df.loc[
            score_mask,
            column,
        ] = (
            float(alpha)
            * df.loc[
                score_mask,
                ml_pct,
            ]
            + (
                1.0
                - float(alpha)
            )
            * df.loc[
                score_mask,
                mom_pct,
            ]
        )
        columns[
            float(alpha)
        ] = column

    confirmed = (
        f"{prefix}_"
        "ml_momentum_confirmed"
    )
    df[confirmed] = np.nan

    confirmation_mask = (
        score_mask
        & df[
            "return_60d"
        ].gt(0)
    )
    df.loc[
        confirmation_mask,
        confirmed,
    ] = df.loc[
        confirmation_mask,
        ml_pct,
    ]

    return columns


def backtest_score(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    name: str,
    score_column: str,
    top_k: int,
    capital: float,
    costs: CostProfile,
) -> dict:
    result = run_backtest(
        panel,
        events,
        strategy=name,
        score_column=score_column,
        score_direction=1.0,
        initial_capital=capital,
        top_k=top_k,
        holding_sessions=(
            HORIZON
        ),
        costs=costs,
    )

    return {
        "name": name,
        "top_k": int(
            top_k
        ),
        **result["metrics"],
    }


def validation_ensemble_search(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    validation_score_mask: pd.Series,
    alphas: list[float],
    top_k_values: list[int],
    capital: float,
    current_costs: CostProfile,
) -> tuple[
    pd.DataFrame,
    float,
]:
    ensemble_columns = (
        add_ensemble_columns(
            df,
            prefix="validation_ml",
            alphas=alphas,
            score_mask=(
                validation_score_mask
            ),
        )
    )

    panel = (
        df.loc[
            df["date"].between(
                VALIDATION_START,
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
        "\n=== VALIDATION-ONLY "
        "ENSEMBLE SEARCH ==="
    )

    for alpha in alphas:
        column = (
            ensemble_columns[
                float(alpha)
            ]
        )

        for top_k in (
            top_k_values
        ):
            row = backtest_score(
                panel,
                events,
                name=(
                    f"validation_ensemble_"
                    f"a{alpha:.2f}"
                ),
                score_column=(
                    column
                ),
                top_k=top_k,
                capital=capital,
                costs=current_costs,
            )
            row[
                "alpha"
            ] = float(
                alpha
            )
            rows.append(
                row
            )

            print(
                f"alpha={alpha:.2f} "
                f"k={top_k} "
                f"CAGR={row['cagr']:.2%} "
                f"Sharpe={row['sharpe']:.3f}"
            )

    results = pd.DataFrame(
        rows
    )

    aggregate = (
        results.groupby(
            "alpha",
            sort=True,
        )
        .agg(
            mean_net_sharpe=(
                "sharpe",
                "mean",
            ),
            mean_net_cagr=(
                "cagr",
                "mean",
            ),
            worst_max_drawdown=(
                "max_drawdown",
                "min",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "mean_net_sharpe",
                "mean_net_cagr",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    selected_alpha = float(
        aggregate.iloc[0][
            "alpha"
        ]
    )

    results = results.merge(
        aggregate,
        on="alpha",
        how="left",
        validate="many_to_one",
    )
    results[
        "selected_alpha"
    ] = results[
        "alpha"
    ].eq(
        selected_alpha
    )

    print(
        "\nSelected alpha "
        "(validation mean net "
        "Sharpe across k): "
        f"{selected_alpha:.2f}"
    )

    return (
        results,
        selected_alpha,
    )


def gross_net_decomposition(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    score_columns: dict[
        str,
        str,
    ],
    top_k_values: list[int],
    capital: float,
    current_costs: CostProfile,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    for label, column in (
        score_columns.items()
    ):
        for top_k in (
            top_k_values
        ):
            gross = (
                backtest_score(
                    panel,
                    events,
                    name=label,
                    score_column=column,
                    top_k=top_k,
                    capital=capital,
                    costs=(
                        zero_cost_profile()
                    ),
                )
            )
            net = backtest_score(
                panel,
                events,
                name=label,
                score_column=column,
                top_k=top_k,
                capital=capital,
                costs=(
                    current_costs
                ),
            )

            rows.append({
                "name": label,
                "top_k": int(
                    top_k
                ),
                "gross_ending_equity": (
                    gross[
                        "ending_equity"
                    ]
                ),
                "net_ending_equity": (
                    net[
                        "ending_equity"
                    ]
                ),
                "gross_cagr": gross[
                    "cagr"
                ],
                "net_cagr": net[
                    "cagr"
                ],
                "cagr_cost_drag_pp": (
                    (
                        gross["cagr"]
                        - net["cagr"]
                    )
                    * 100.0
                ),
                "gross_sharpe": gross[
                    "sharpe"
                ],
                "net_sharpe": net[
                    "sharpe"
                ],
                "gross_max_drawdown": (
                    gross[
                        "max_drawdown"
                    ]
                ),
                "net_max_drawdown": (
                    net[
                        "max_drawdown"
                    ]
                ),
                "net_total_fees": net[
                    "total_fees"
                ],
                "gross_trades": gross[
                    "trades"
                ],
                "net_trades": net[
                    "trades"
                ],
            })

    return pd.DataFrame(
        rows
    )


def walkforward_predictions(
    df: pd.DataFrame,
    *,
    model_name: str,
    max_train_rows: int,
    random_state: int,
    models_root: Path,
) -> tuple[
    pd.Series,
    pd.DataFrame,
]:
    df[
        "walkforward_ml_score"
    ] = np.nan

    fold_rows: list[
        dict
    ] = []

    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for offset, year in enumerate(
        WALK_FORWARD_YEARS
    ):
        fold_start = max(
            TEST_START,
            pd.Timestamp(
                f"{year}-01-01"
            ),
        )
        fold_end = min(
            TEST_END,
            pd.Timestamp(
                f"{year}-12-31"
            ),
        )

        if fold_start > fold_end:
            continue

        prior = df.loc[
            df["date"].lt(
                fold_start
            ),
            "market_day_index",
        ].dropna()

        if prior.empty:
            raise RuntimeError(
                "No market session exists "
                f"before fold {year}."
            )

        cutoff_index = int(
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
                cutoff_index
                - HORIZON
            )
        )

        sampled_train = (
            maybe_sample_mask(
                train_mask,
                max_rows=(
                    max_train_rows
                ),
                random_state=(
                    random_state
                    + offset
                ),
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
            "\nWalk-forward fold "
            f"{year}: train="
            f"{int(sampled_train.sum()):,}, "
            "score="
            f"{int(score_mask.sum()):,}"
        )

        model = build_model(
            model_name,
            random_state=(
                random_state
                + offset
            ),
        )
        fit_model(
            model,
            df,
            sampled_train,
        )

        scores = predict_mask(
            model,
            df,
            score_mask,
        )
        df.loc[
            score_mask,
            "walkforward_ml_score",
        ] = scores

        joblib.dump(
            model,
            models_root
            / (
                f"{model_name}_"
                f"through_{year - 1}.joblib"
            ),
        )

        label_mask = (
            df[TRAINING_FLAG]
            & df[
                "target_rank_20d"
            ].notna()
            & df["date"].between(
                fold_start,
                fold_end,
                inclusive="both",
            )
        )

        label_scores = df.loc[
            label_mask,
            "walkforward_ml_score",
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

        fold_rows.append({
            "fold_year": int(
                year
            ),
            "train_cutoff_market_index": (
                int(
                    cutoff_index
                    - HORIZON
                )
            ),
            "train_rows": int(
                sampled_train.sum()
            ),
            "score_rows": int(
                score_mask.sum()
            ),
            **diagnostics,
        })

    score_mask = (
        df["eligible_universe"]
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
        & df[
            "walkforward_ml_score"
        ].notna()
    )

    add_daily_score_ranks(
        df,
        raw_score_column=(
            "walkforward_ml_score"
        ),
        output_prefix=(
            "walkforward_ml"
        ),
        mask=score_mask,
    )

    return (
        score_mask,
        pd.DataFrame(
            fold_rows
        ),
    )


def final_walkforward_backtests(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    root: Path,
    score_mask: pd.Series,
    selected_alpha: float,
    top_k_values: list[int],
    capital: float,
    current_costs: CostProfile,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    ensemble_columns = (
        add_ensemble_columns(
            df,
            prefix="walkforward_ml",
            alphas=[
                selected_alpha
            ],
            score_mask=(
                score_mask
            ),
        )
    )
    ensemble_column = (
        ensemble_columns[
            float(
                selected_alpha
            )
        ]
    )
    confirmation_column = (
        "walkforward_ml_"
        "ml_momentum_confirmed"
    )

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

    rows: list[
        dict
    ] = []

    for top_k in top_k_values:
        for (
            name,
            score_column,
        ) in (
            (
                "walkforward_ml",
                "walkforward_ml_pct",
            ),
            (
                (
                    "walkforward_ensemble_"
                    f"a{selected_alpha:.2f}"
                ),
                ensemble_column,
            ),
            (
                "walkforward_ml_"
                "momentum_confirmed",
                confirmation_column,
            ),
        ):
            row = backtest_score(
                panel,
                events,
                name=name,
                score_column=(
                    score_column
                ),
                top_k=top_k,
                capital=capital,
                costs=current_costs,
            )
            row["kind"] = (
                "walkforward_ml"
            )
            rows.append(row)

        for strategy in (
            "momentum_60d",
            "liquidity_control",
        ):
            result = run_backtest(
                panel,
                events,
                strategy=strategy,
                initial_capital=(
                    capital
                ),
                top_k=top_k,
                holding_sessions=(
                    HORIZON
                ),
                costs=current_costs,
            )
            rows.append({
                "name": (
                    f"{strategy}_"
                    f"h20_k{top_k}"
                ),
                "kind": (
                    "deterministic_strategy"
                ),
                "top_k": int(
                    top_k
                ),
                **result["metrics"],
            })

    index_rows, _ = (
        add_index_benchmarks(
            root,
            test_panel_dates=set(
                panel[
                    "date"
                ].dropna()
            ),
            capital=capital,
        )
    )

    for row in (
        index_rows.to_dict(
            orient="records"
        )
    ):
        row["kind"] = (
            "price_index"
        )
        rows.append(row)

    comparison = pd.DataFrame(
        rows
    )

    score_columns = {
        "walkforward_ml": (
            "walkforward_ml_pct"
        ),
        (
            "walkforward_ensemble_"
            f"a{selected_alpha:.2f}"
        ): ensemble_column,
    }

    gross_net = (
        gross_net_decomposition(
            panel,
            events,
            score_columns=(
                score_columns
            ),
            top_k_values=(
                top_k_values
            ),
            capital=capital,
            current_costs=(
                current_costs
            ),
        )
    )

    return (
        comparison,
        gross_net,
    )


def self_test() -> None:
    dates = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="D",
    )

    rows: list[
        dict
    ] = []

    for day in dates:
        for i in range(20):
            rows.append({
                "date": day,
                "canonical_security_id": (
                    f"S{i:02d}"
                ),
                TARGET_COLUMN: (
                    i / 100.0
                ),
                "score": float(i),
                "score_pct": (
                    (i + 1) / 20.0
                ),
            })

    frame = pd.DataFrame(
        rows
    )
    mask = pd.Series(
        True,
        index=frame.index,
    )

    _, deciles = (
        decile_analysis(
            frame,
            mask=mask,
            score_pct_column=(
                "score_pct"
            ),
            split_name="test",
        )
    )

    assert (
        deciles.loc[
            deciles[
                "decile"
            ].eq(10),
            "mean_daily_return",
        ].iloc[0]
        > deciles.loc[
            deciles[
                "decile"
            ].eq(1),
            "mean_daily_return",
        ].iloc[0]
    )

    _, tail = (
        top_tail_analysis(
            frame,
            mask=mask,
            score_pct_column=(
                "score_pct"
            ),
            raw_score_column=(
                "score"
            ),
            split_name="test",
        )
    )

    assert (
        tail.loc[
            tail[
                "selection"
            ].eq("top_5"),
            "mean_excess_return",
        ].iloc[0]
        > 0
    )

    (
        _,
        persistence,
    ) = persistence_analysis(
        frame,
        mask=mask,
        score_pct_column=(
            "score_pct"
        ),
        raw_score_column=(
            "score"
        ),
        split_name="test",
    )

    lag1 = persistence.loc[
        persistence[
            "lag_sessions"
        ].eq(1)
    ]

    assert not lag1.empty
    assert (
        lag1[
            "mean_rank_spearman"
        ].iloc[0]
        > 0.99
    )

    print(
        "ML portfolio diagnostics "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validation-only diagnostics, "
            "ensemble construction and "
            "expanding walk-forward evaluation "
            "for the classical NSE ranking model."
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
        help=(
            "Frozen model family selected "
            "from the prior validation experiment."
        ),
    )
    ap.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=(
            DEFAULT_ALPHAS
        ),
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
        "--capital",
        type=float,
        default=50_000.0,
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--max-walkforward-train-rows",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
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

    if any(
        alpha < 0
        or alpha > 1
        for alpha in args.alphas
    ):
        raise SystemExit(
            "--alphas must be between "
            "0 and 1 inclusive."
        )

    if any(
        k < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k values must "
            "all be >= 1."
        )

    root = Path(
        args.root
    ).resolve()

    df = load_ml_panel(
        root
        / "data/processed/"
        "research_panel"
    )
    df = add_target_rank(
        df
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

    output_root = (
        root
        / "reports/ml/"
        "portfolio_diagnostics"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    models_root = (
        output_root
        / "walkforward_models"
    )

    (
        validation_model,
        validation_score_mask,
        validation_label_mask,
    ) = fit_validation_model(
        df,
        model_name=args.model,
        max_train_rows=(
            args.max_train_rows
        ),
        random_state=(
            args.random_state
        ),
    )

    joblib.dump(
        validation_model,
        output_root
        / (
            f"{args.model}_"
            "train_through_2018.joblib"
        ),
    )

    validation_scores = df.loc[
        validation_label_mask,
        "validation_ml_score",
    ].to_numpy(
        dtype=float
    )
    validation_diagnostics = (
        cross_sectional_diagnostics(
            df,
            validation_label_mask,
            validation_scores,
        )
    )

    (
        validation_decile_daily,
        validation_decile_summary,
    ) = decile_analysis(
        df,
        mask=(
            validation_label_mask
        ),
        score_pct_column=(
            "validation_ml_pct"
        ),
        split_name=(
            "validation_2019_2021"
        ),
    )

    (
        validation_tail_daily,
        validation_tail_summary,
    ) = top_tail_analysis(
        df,
        mask=(
            validation_label_mask
        ),
        score_pct_column=(
            "validation_ml_pct"
        ),
        raw_score_column=(
            "validation_ml_score"
        ),
        split_name=(
            "validation_2019_2021"
        ),
    )

    (
        validation_persistence_daily,
        validation_persistence_summary,
    ) = persistence_analysis(
        df,
        mask=(
            validation_score_mask
        ),
        score_pct_column=(
            "validation_ml_pct"
        ),
        raw_score_column=(
            "validation_ml_score"
        ),
        split_name=(
            "validation_2019_2021"
        ),
    )

    (
        ensemble_search,
        selected_alpha,
    ) = validation_ensemble_search(
        df,
        events,
        validation_score_mask=(
            validation_score_mask
        ),
        alphas=[
            float(x)
            for x in args.alphas
        ],
        top_k_values=list(
            args.top_k
        ),
        capital=args.capital,
        current_costs=(
            current_costs
        ),
    )

    validation_panel = (
        df.loc[
            df["date"].between(
                VALIDATION_START,
                VALIDATION_END,
                inclusive="both",
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    selected_validation_column = (
        f"validation_ml_ensemble_a"
        + (
            f"{selected_alpha:.2f}"
            .replace(
                ".",
                "p",
            )
        )
    )

    validation_gross_net = (
        gross_net_decomposition(
            validation_panel,
            events,
            score_columns={
                "validation_pure_ml": (
                    "validation_ml_pct"
                ),
                (
                    "validation_selected_"
                    "ensemble"
                ): (
                    selected_validation_column
                ),
            },
            top_k_values=list(
                args.top_k
            ),
            capital=args.capital,
            current_costs=(
                current_costs
            ),
        )
    )

    print(
        "\n=== WALK-FORWARD "
        "2022-2026 ==="
    )

    (
        walkforward_score_mask,
        fold_diagnostics,
    ) = walkforward_predictions(
        df,
        model_name=args.model,
        max_train_rows=(
            args.max_walkforward_train_rows
        ),
        random_state=(
            args.random_state
        ),
        models_root=(
            models_root
        ),
    )

    test_label_mask = (
        df[TRAINING_FLAG]
        & df[
            "target_rank_20d"
        ].notna()
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
        & df[
            "walkforward_ml_score"
        ].notna()
    )

    (
        test_decile_daily,
        test_decile_summary,
    ) = decile_analysis(
        df,
        mask=test_label_mask,
        score_pct_column=(
            "walkforward_ml_pct"
        ),
        split_name=(
            "walkforward_2022_2026"
        ),
    )

    (
        test_tail_daily,
        test_tail_summary,
    ) = top_tail_analysis(
        df,
        mask=test_label_mask,
        score_pct_column=(
            "walkforward_ml_pct"
        ),
        raw_score_column=(
            "walkforward_ml_score"
        ),
        split_name=(
            "walkforward_2022_2026"
        ),
    )

    (
        test_persistence_daily,
        test_persistence_summary,
    ) = persistence_analysis(
        df,
        mask=(
            walkforward_score_mask
        ),
        score_pct_column=(
            "walkforward_ml_pct"
        ),
        raw_score_column=(
            "walkforward_ml_score"
        ),
        split_name=(
            "walkforward_2022_2026"
        ),
    )

    (
        final_comparison,
        test_gross_net,
    ) = final_walkforward_backtests(
        df,
        events,
        score_mask=(
            walkforward_score_mask
        ),
        selected_alpha=(
            selected_alpha
        ),
        top_k_values=list(
            args.top_k
        ),
        capital=args.capital,
        current_costs=(
            current_costs
        ),
    )

    validation_decile_daily.to_csv(
        output_root
        / "validation_decile_daily.csv",
        index=False,
    )
    validation_decile_summary.to_csv(
        output_root
        / "validation_decile_summary.csv",
        index=False,
    )
    validation_tail_daily.to_csv(
        output_root
        / "validation_top_tail_daily.csv",
        index=False,
    )
    validation_tail_summary.to_csv(
        output_root
        / "validation_top_tail_summary.csv",
        index=False,
    )
    validation_persistence_daily.to_csv(
        output_root
        / "validation_score_persistence_daily.csv",
        index=False,
    )
    validation_persistence_summary.to_csv(
        output_root
        / "validation_score_persistence_summary.csv",
        index=False,
    )
    ensemble_search.to_csv(
        output_root
        / "validation_ensemble_search.csv",
        index=False,
    )
    validation_gross_net.to_csv(
        output_root
        / "validation_gross_vs_net.csv",
        index=False,
    )

    fold_diagnostics.to_csv(
        output_root
        / "walkforward_fold_diagnostics.csv",
        index=False,
    )
    test_decile_daily.to_csv(
        output_root
        / "walkforward_decile_daily.csv",
        index=False,
    )
    test_decile_summary.to_csv(
        output_root
        / "walkforward_decile_summary.csv",
        index=False,
    )
    test_tail_daily.to_csv(
        output_root
        / "walkforward_top_tail_daily.csv",
        index=False,
    )
    test_tail_summary.to_csv(
        output_root
        / "walkforward_top_tail_summary.csv",
        index=False,
    )
    test_persistence_daily.to_csv(
        output_root
        / "walkforward_score_persistence_daily.csv",
        index=False,
    )
    test_persistence_summary.to_csv(
        output_root
        / "walkforward_score_persistence_summary.csv",
        index=False,
    )
    final_comparison.to_csv(
        output_root
        / "walkforward_portfolio_comparison.csv",
        index=False,
    )
    test_gross_net.to_csv(
        output_root
        / "walkforward_gross_vs_net.csv",
        index=False,
    )

    predictions = df.loc[
        walkforward_score_mask,
        [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "return_60d",
            "walkforward_ml_score",
            "walkforward_ml_pct",
            "walkforward_ml_mom60_pct",
            TARGET_COLUMN,
            "target_rank_20d",
        ],
    ].copy()
    predictions.to_parquet(
        output_root
        / "walkforward_predictions.parquet",
        index=False,
        compression="zstd",
    )

    summary = {
        "model_family": (
            args.model
        ),
        "validation_model_fit": {
            "train_period": (
                f"{TRAIN_START.date()} "
                f"to {TRAIN_END.date()}"
            ),
            "validation_period": (
                f"{VALIDATION_START.date()} "
                f"to {VALIDATION_END.date()}"
            ),
            "diagnostics": (
                validation_diagnostics
            ),
        },
        "ensemble": {
            "formula": (
                "alpha * ML daily percentile "
                "+ (1-alpha) * 60d-momentum "
                "daily percentile"
            ),
            "candidate_alphas": [
                float(x)
                for x in args.alphas
            ],
            "selection_data": (
                "2019-2021 validation only"
            ),
            "selection_metric": (
                "mean net Sharpe across "
                "top-5 and top-10; mean net "
                "CAGR tie-breaker"
            ),
            "selected_alpha": (
                selected_alpha
            ),
        },
        "momentum_confirmation": (
            "ML percentile ranked only among "
            "eligible rows with return_60d > 0; "
            "others receive no score"
        ),
        "walkforward": {
            "years": (
                WALK_FORWARD_YEARS
            ),
            "training_rule": (
                "expanding history from "
                "2010-06-28 through 20 market "
                "sessions before each fold start"
            ),
            "ensemble_alpha_retuned": (
                False
            ),
        },
        "capital": float(
            args.capital
        ),
        "top_k": [
            int(x)
            for x in args.top_k
        ],
        "cost_profile": (
            asdict(
                current_costs
            )
        ),
        "outputs": {
            "validation_deciles": (
                "validation_decile_summary.csv"
            ),
            "validation_top_tail": (
                "validation_top_tail_summary.csv"
            ),
            "validation_persistence": (
                "validation_score_persistence_summary.csv"
            ),
            "validation_ensemble_search": (
                "validation_ensemble_search.csv"
            ),
            "validation_gross_net": (
                "validation_gross_vs_net.csv"
            ),
            "walkforward_folds": (
                "walkforward_fold_diagnostics.csv"
            ),
            "walkforward_deciles": (
                "walkforward_decile_summary.csv"
            ),
            "walkforward_top_tail": (
                "walkforward_top_tail_summary.csv"
            ),
            "walkforward_persistence": (
                "walkforward_score_persistence_summary.csv"
            ),
            "walkforward_portfolios": (
                "walkforward_portfolio_comparison.csv"
            ),
            "walkforward_gross_net": (
                "walkforward_gross_vs_net.csv"
            ),
        },
    }

    (
        output_root
        / "diagnostics_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(
        "\n=== VALIDATION TOP-TAIL "
        "SUMMARY ==="
    )
    print(
        validation_tail_summary[
            [
                "selection",
                "mean_realized_return",
                "mean_excess_return",
                "positive_excess_fraction",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_realized_return": (
                    lambda x: (
                        f"{x:.4%}"
                    )
                ),
                "mean_excess_return": (
                    lambda x: (
                        f"{x:.4%}"
                    )
                ),
                "positive_excess_fraction": (
                    lambda x: (
                        f"{x:.2%}"
                    )
                ),
            },
        )
    )

    print(
        "\n=== WALK-FORWARD FOLDS ==="
    )
    print(
        fold_diagnostics[
            [
                "fold_year",
                "train_rows",
                "mean_daily_ic",
                "positive_ic_fraction",
                "mean_top_decile_excess",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_daily_ic": (
                    lambda x: (
                        f"{x:.4f}"
                    )
                ),
                "positive_ic_fraction": (
                    lambda x: (
                        f"{x:.2%}"
                    )
                ),
                "mean_top_decile_excess": (
                    lambda x: (
                        f"{x:.4%}"
                    )
                ),
            },
        )
    )

    preview = (
        final_comparison[
            [
                column
                for column in (
                    "name",
                    "kind",
                    "ending_equity",
                    "cagr",
                    "max_drawdown",
                    "sharpe",
                    "trades",
                    "total_fees",
                )
                if column
                in final_comparison.columns
            ]
        ]
        .copy()
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
    )
    preview["cagr"] = (
        pd.to_numeric(
            preview["cagr"],
            errors="coerce",
        )
        * 100.0
    )
    preview[
        "max_drawdown"
    ] = (
        pd.to_numeric(
            preview[
                "max_drawdown"
            ],
            errors="coerce",
        )
        * 100.0
    )

    print(
        "\n=== WALK-FORWARD "
        "PORTFOLIO COMPARISON ==="
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
        "\n=== PORTFOLIO "
        "DIAGNOSTICS COMPLETE ==="
    )
    print(
        "Selected validation alpha: "
        f"{selected_alpha:.2f}"
    )
    print(
        f"Outputs: {output_root}"
    )


if __name__ == "__main__":
    main()
