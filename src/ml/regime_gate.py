from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    add_index_benchmarks,
    add_target_rank,
    build_model,
    fit_model,
    load_ml_panel,
    predict_mask,
)
from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
    run_backtest,
)


META_FEATURES = [
    "score_std",
    "score_iqr",
    "score_q90_q50",
    "score_q95_q50",
    "score_top10_mean_minus_median",
    "positive_momentum_fraction",
    "median_return_60d",
    "return_60d_dispersion",
    "median_volatility_20d",
    "median_volatility_60d",
    "universe_size",
    "recent_ic_20",
    "recent_ic_60",
    "recent_ic_120",
    "recent_top_decile_excess_20",
    "recent_top_decile_excess_60",
    "recent_top_decile_excess_120",
    "recent_positive_ic_fraction_60",
    "recent_positive_top_decile_fraction_60",
]

DEFAULT_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
]
DEFAULT_TOP_K = [
    5,
    10,
]


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


def fit_validation_base_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
) -> pd.Series:
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

    score_mask = (
        df["eligible_universe"]
        & df["date"].between(
            VALIDATION_START,
            VALIDATION_END,
            inclusive="both",
        )
    )

    print(
        "Fitting base model for "
        "2019-2021 OOS scores on "
        f"{int(train_mask.sum()):,} rows..."
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

    df[
        "base_oos_score"
    ] = np.nan
    df.loc[
        score_mask,
        "base_oos_score",
    ] = predict_mask(
        model,
        df,
        score_mask,
    )

    return score_mask


def load_walkforward_scores(
    df: pd.DataFrame,
    *,
    root: Path,
) -> pd.Series:
    path = (
        root
        / "reports/ml/"
        "portfolio_diagnostics/"
        "walkforward_predictions.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Walk-forward predictions are "
            "missing. Run "
            "src/ml/portfolio_diagnostics.py "
            "first."
        )

    saved = pd.read_parquet(
        path,
        columns=[
            "date",
            "canonical_security_id",
            "walkforward_ml_score",
        ],
    )

    saved["date"] = (
        pd.to_datetime(
            saved["date"],
            errors="coerce",
        ).dt.normalize()
    )
    saved[
        "canonical_security_id"
    ] = (
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
            "Saved walk-forward predictions "
            "contain duplicate keys."
        )

    mapping = (
        saved.set_index(
            keys
        )[
            "walkforward_ml_score"
        ]
    )

    key_index = pd.MultiIndex.from_frame(
        df[keys]
    )

    matched = mapping.reindex(
        key_index
    ).to_numpy(
        dtype=float
    )

    test_mask = (
        df["eligible_universe"]
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
    )

    df.loc[
        test_mask,
        "base_oos_score",
    ] = matched[
        test_mask.to_numpy()
    ]

    coverage = float(
        df.loc[
            test_mask,
            "base_oos_score",
        ]
        .notna()
        .mean()
    )

    if coverage < 0.98:
        raise RuntimeError(
            "Walk-forward prediction coverage "
            f"is only {coverage:.2%}."
        )

    return test_mask


def add_daily_percentile_score(
    df: pd.DataFrame,
) -> None:
    df[
        "base_oos_score_pct"
    ] = np.nan

    score_mask = (
        df["eligible_universe"]
        & df[
            "base_oos_score"
        ].notna()
    )

    ranked = (
        df.loc[
            score_mask
        ]
        .groupby(
            "date",
            sort=False,
        )[
            "base_oos_score"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    df.loc[
        ranked.index,
        "base_oos_score_pct",
    ] = ranked.to_numpy()


def build_daily_meta(
    df: pd.DataFrame,
) -> pd.DataFrame:
    source = df.loc[
        df["eligible_universe"]
        & df[
            "base_oos_score"
        ].notna()
        & df["date"].between(
            VALIDATION_START,
            TEST_END,
            inclusive="both",
        )
    ].copy()

    rows: list[
        dict
    ] = []

    for day, group in source.groupby(
        "date",
        sort=True,
    ):
        scores = pd.to_numeric(
            group[
                "base_oos_score"
            ],
            errors="coerce",
        ).dropna()

        if len(scores) < 20:
            continue

        q25 = float(
            scores.quantile(0.25)
        )
        q50 = float(
            scores.quantile(0.50)
        )
        q75 = float(
            scores.quantile(0.75)
        )
        q90 = float(
            scores.quantile(0.90)
        )
        q95 = float(
            scores.quantile(0.95)
        )

        top10_mask = (
            group[
                "base_oos_score_pct"
            ].ge(0.90)
        )

        label_group = group.loc[
            group[TRAINING_FLAG]
            & group[
                TARGET_COLUMN
            ].notna()
        ].copy()

        ic = np.nan
        top_decile_excess = (
            np.nan
        )
        top5_excess = np.nan

        if len(label_group) >= 20:
            score_rank = (
                label_group[
                    "base_oos_score"
                ].rank(
                    method="average",
                    pct=True,
                )
            )
            target_rank = (
                label_group[
                    TARGET_COLUMN
                ].rank(
                    method="average",
                    pct=True,
                )
            )
            ic = score_rank.corr(
                target_rank,
                method="spearman",
            )

            universe_return = float(
                label_group[
                    TARGET_COLUMN
                ].mean()
            )

            top_decile = (
                label_group.loc[
                    score_rank.ge(
                        0.90
                    )
                ]
            )
            if len(top_decile):
                top_decile_excess = (
                    float(
                        top_decile[
                            TARGET_COLUMN
                        ].mean()
                    )
                    - universe_return
                )

            top5 = (
                label_group.nlargest(
                    5,
                    "base_oos_score",
                )
            )
            if len(top5):
                top5_excess = (
                    float(
                        top5[
                            TARGET_COLUMN
                        ].mean()
                    )
                    - universe_return
                )

        market_index = int(
            group[
                "market_day_index"
            ].iloc[0]
        )

        rows.append({
            "date": pd.Timestamp(
                day
            ).normalize(),
            "market_day_index": (
                market_index
            ),
            "universe_size": float(
                len(group)
            ),
            "score_std": float(
                scores.std()
            ),
            "score_iqr": (
                q75 - q25
            ),
            "score_q90_q50": (
                q90 - q50
            ),
            "score_q95_q50": (
                q95 - q50
            ),
            "score_top10_mean_minus_median": (
                float(
                    group.loc[
                        top10_mask,
                        "base_oos_score",
                    ].mean()
                )
                - q50
            ),
            "positive_momentum_fraction": float(
                group[
                    "return_60d"
                ].gt(0).mean()
            ),
            "median_return_60d": float(
                group[
                    "return_60d"
                ].median()
            ),
            "return_60d_dispersion": float(
                group[
                    "return_60d"
                ].std()
            ),
            "median_volatility_20d": float(
                group[
                    "volatility_20d"
                ].median()
            ),
            "median_volatility_60d": float(
                group[
                    "volatility_60d"
                ].median()
            ),
            "realized_ic": (
                float(ic)
                if not pd.isna(ic)
                else np.nan
            ),
            "realized_top_decile_excess": (
                float(
                    top_decile_excess
                )
                if not pd.isna(
                    top_decile_excess
                )
                else np.nan
            ),
            "realized_top5_excess": (
                float(
                    top5_excess
                )
                if not pd.isna(
                    top5_excess
                )
                else np.nan
            ),
        })

    daily = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "market_day_index"
        )
        .reset_index(drop=True)
    )

    # A target generated from signal date t is known only at close(t+20).
    # Shift realized diagnostics by the forecast horizon before using them as
    # gate features, then roll over only information that had matured.
    matured_ic = daily[
        "realized_ic"
    ].shift(HORIZON)
    matured_excess = daily[
        "realized_top_decile_excess"
    ].shift(HORIZON)

    for window in (
        20,
        60,
        120,
    ):
        min_periods = min(
            20,
            window,
        )
        daily[
            f"recent_ic_{window}"
        ] = (
            matured_ic.rolling(
                window,
                min_periods=(
                    min_periods
                ),
            ).mean()
        )
        daily[
            (
                "recent_top_decile_"
                f"excess_{window}"
            )
        ] = (
            matured_excess.rolling(
                window,
                min_periods=(
                    min_periods
                ),
            ).mean()
        )

    daily[
        "recent_positive_ic_fraction_60"
    ] = (
        matured_ic.gt(0)
        .astype(float)
        .where(
            matured_ic.notna()
        )
        .rolling(
            60,
            min_periods=20,
        )
        .mean()
    )

    daily[
        (
            "recent_positive_"
            "top_decile_fraction_60"
        )
    ] = (
        matured_excess.gt(0)
        .astype(float)
        .where(
            matured_excess.notna()
        )
        .rolling(
            60,
            min_periods=20,
        )
        .mean()
    )

    daily[
        "gate_target"
    ] = (
        daily[
            "realized_top_decile_excess"
        ].gt(0)
        .where(
            daily[
                "realized_top_decile_excess"
            ].notna()
        )
        .astype("Float64")
    )

    return daily


def build_gate_model(
    *,
    random_state: int,
) -> Pipeline:
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                class_weight=(
                    "balanced"
                ),
                max_iter=2000,
                random_state=(
                    random_state
                ),
            ),
        ),
    ])


def fit_gate(
    daily: pd.DataFrame,
    mask: pd.Series,
    *,
    random_state: int,
) -> Pipeline:
    rows = daily.loc[
        mask
        & daily[
            "gate_target"
        ].notna()
    ]

    if len(rows) < 100:
        raise RuntimeError(
            "Insufficient daily meta rows "
            "to fit regime gate."
        )

    model = build_gate_model(
        random_state=(
            random_state
        ),
    )
    model.fit(
        rows[
            META_FEATURES
        ],
        rows[
            "gate_target"
        ].astype(int),
    )

    return model


def score_gate(
    model: Pipeline,
    daily: pd.DataFrame,
    mask: pd.Series,
) -> pd.Series:
    output = pd.Series(
        np.nan,
        index=daily.index,
        dtype=float,
    )

    rows = daily.loc[
        mask
    ]

    if rows.empty:
        return output

    output.loc[
        rows.index
    ] = model.predict_proba(
        rows[
            META_FEATURES
        ]
    )[:, 1]

    return output


def map_daily_gate_to_panel(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    probability_column: str,
) -> None:
    mapping = (
        daily.set_index(
            "date"
        )[
            probability_column
        ]
    )

    df[
        probability_column
    ] = df[
        "date"
    ].map(mapping)


def backtest_gate_candidate(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    name: str,
    gate_mask: pd.Series,
    top_k: int,
    capital: float,
    costs: CostProfile,
) -> dict:
    local = panel.copy()
    local[
        "gated_score"
    ] = np.nan
    local.loc[
        gate_mask,
        "gated_score",
    ] = local.loc[
        gate_mask,
        "base_oos_score_pct",
    ]

    result = run_backtest(
        local,
        events,
        strategy=name,
        score_column=(
            "gated_score"
        ),
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


def select_gate_on_2021(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    thresholds: list[float],
    top_k_values: list[int],
    capital: float,
    costs: CostProfile,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    str,
    float | None,
    int,
]:
    tune_start = pd.Timestamp(
        "2021-01-01"
    )
    last_pre_tune_index = int(
        daily.loc[
            daily["date"].lt(
                tune_start
            ),
            "market_day_index",
        ]
        .dropna()
        .max()
    )

    train_mask = (
        daily["date"].between(
            VALIDATION_START,
            pd.Timestamp(
                "2020-12-31"
            ),
            inclusive="both",
        )
        & daily[
            "market_day_index"
        ].le(
            last_pre_tune_index
            - HORIZON
        )
    )

    model = fit_gate(
        daily,
        train_mask,
        random_state=(
            random_state
        ),
    )

    tune_mask = (
        daily["date"].between(
            pd.Timestamp(
                "2021-01-01"
            ),
            VALIDATION_END,
            inclusive="both",
        )
    )

    daily[
        "gate_probability_2021"
    ] = score_gate(
        model,
        daily,
        tune_mask,
    )
    map_daily_gate_to_panel(
        df,
        daily,
        probability_column=(
            "gate_probability_2021"
        ),
    )

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

    candidates: list[
        tuple[
            str,
            float | None,
        ]
    ] = [
        (
            "always_on",
            None,
        ),
        (
            "recent_ic_positive",
            None,
        ),
    ]

    candidates.extend(
        (
            "logistic",
            float(threshold),
        )
        for threshold in (
            thresholds
        )
    )

    print(
        "\n=== 2021 INNER GATE "
        "SELECTION ==="
    )

    for method, threshold in (
        candidates
    ):
        for top_k in (
            top_k_values
        ):
            if method == "always_on":
                gate_mask = (
                    panel[
                        "base_oos_score"
                    ].notna()
                )
                label = (
                    "always_on"
                )

            elif (
                method
                == "recent_ic_positive"
            ):
                gate_mask = (
                    panel[
                        "recent_ic_60"
                    ].gt(0)
                    & panel[
                        "base_oos_score"
                    ].notna()
                )
                label = (
                    "recent_ic_positive"
                )

            else:
                gate_mask = (
                    panel[
                        "gate_probability_2021"
                    ].ge(
                        float(
                            threshold
                        )
                    )
                    & panel[
                        "base_oos_score"
                    ].notna()
                )
                label = (
                    "logistic_"
                    f"{threshold:.2f}"
                )

            row = (
                backtest_gate_candidate(
                    panel,
                    events,
                    name=label,
                    gate_mask=(
                        gate_mask
                    ),
                    top_k=top_k,
                    capital=capital,
                    costs=costs,
                )
            )
            row[
                "method"
            ] = method
            row[
                "threshold"
            ] = threshold
            row[
                "active_signal_fraction"
            ] = float(
                gate_mask.mean()
            )
            rows.append(row)

            print(
                f"{label:24s} "
                f"k={top_k:2d} "
                f"CAGR={row['cagr']:.2%} "
                f"Sharpe={row['sharpe']:.3f} "
                f"DD={row['max_drawdown']:.2%}"
            )

    results = pd.DataFrame(
        rows
    ).sort_values(
        [
            "sharpe",
            "cagr",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = results.iloc[0]

    method = str(
        best["method"]
    )
    threshold = (
        None
        if pd.isna(
            best["threshold"]
        )
        else float(
            best["threshold"]
        )
    )
    top_k = int(
        best["top_k"]
    )

    print(
        "\nSelected on 2021 only: "
        f"method={method}, "
        f"threshold={threshold}, "
        f"k={top_k}"
    )

    return (
        results,
        method,
        threshold,
        top_k,
    )


def walkforward_gate_probabilities(
    daily: pd.DataFrame,
    *,
    random_state: int,
) -> pd.DataFrame:
    daily = daily.copy()
    daily[
        "walkforward_gate_probability"
    ] = np.nan

    folds: list[
        dict
    ] = []

    for offset, year in enumerate(
        range(
            2022,
            2027,
        )
    ):
        fold_start = pd.Timestamp(
            f"{year}-01-01"
        )
        fold_end = min(
            TEST_END,
            pd.Timestamp(
                f"{year}-12-31"
            ),
        )

        train_rows = (
            daily["date"].lt(
                fold_start
            )
            & daily[
                "gate_target"
            ].notna()
            & daily[
                "market_day_index"
            ].le(
                int(
                    daily.loc[
                        daily[
                            "date"
                        ].lt(
                            fold_start
                        ),
                        "market_day_index",
                    ].max()
                )
                - HORIZON
            )
        )

        model = fit_gate(
            daily,
            train_rows,
            random_state=(
                random_state
                + offset
            ),
        )

        score_rows = (
            daily["date"].between(
                fold_start,
                fold_end,
                inclusive="both",
            )
        )

        daily.loc[
            score_rows,
            (
                "walkforward_"
                "gate_probability"
            ),
        ] = score_gate(
            model,
            daily,
            score_rows,
        ).loc[
            score_rows
        ]

        folds.append({
            "fold_year": int(
                year
            ),
            "gate_train_days": int(
                train_rows.sum()
            ),
            "gate_score_days": int(
                score_rows.sum()
            ),
            "mean_gate_probability": float(
                daily.loc[
                    score_rows,
                    (
                        "walkforward_"
                        "gate_probability"
                    ),
                ].mean()
            ),
        })

    return (
        daily,
        pd.DataFrame(
            folds
        ),
    )


def final_test(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    root: Path,
    method: str,
    threshold: float | None,
    top_k: int,
    capital: float,
    costs: CostProfile,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    mapping = (
        daily.set_index(
            "date"
        )
    )

    for column in (
        "walkforward_gate_probability",
        "recent_ic_60",
    ):
        df[column] = df[
            "date"
        ].map(
            mapping[
                column
            ]
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

    always_mask = (
        panel[
            "base_oos_score"
        ].notna()
    )
    recent_ic_mask = (
        always_mask
        & panel[
            "recent_ic_60"
        ].gt(0)
    )

    if method == "always_on":
        selected_mask = (
            always_mask
        )
    elif (
        method
        == "recent_ic_positive"
    ):
        selected_mask = (
            recent_ic_mask
        )
    elif method == "logistic":
        selected_mask = (
            always_mask
            & panel[
                (
                    "walkforward_"
                    "gate_probability"
                )
            ].ge(
                float(threshold)
            )
        )
    else:
        raise ValueError(
            f"Unknown gate method: {method}"
        )

    candidates = {
        "always_on_ml": (
            always_mask
        ),
        "recent_ic_positive_gate": (
            recent_ic_mask
        ),
        "selected_regime_gate": (
            selected_mask
        ),
    }

    rows: list[
        dict
    ] = []

    gross_net_rows: list[
        dict
    ] = []

    for name, gate_mask in (
        candidates.items()
    ):
        for (
            cost_name,
            profile,
        ) in (
            (
                "gross",
                zero_cost_profile(),
            ),
            (
                "net",
                costs,
            ),
        ):
            row = (
                backtest_gate_candidate(
                    panel,
                    events,
                    name=name,
                    gate_mask=(
                        gate_mask
                    ),
                    top_k=top_k,
                    capital=capital,
                    costs=profile,
                )
            )
            row[
                "cost_profile"
            ] = cost_name
            row[
                "active_row_fraction"
            ] = float(
                gate_mask.mean()
            )
            gross_net_rows.append(
                row
            )

            if cost_name == "net":
                row[
                    "kind"
                ] = (
                    "ml_regime_gate"
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
            costs=costs,
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
            **result[
                "metrics"
            ],
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

    comparison = (
        pd.DataFrame(
            rows
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

    gross_net = pd.DataFrame(
        gross_net_rows
    )

    return (
        comparison,
        gross_net,
    )


def self_test() -> None:
    dates = pd.bdate_range(
        "2020-01-01",
        periods=180,
    )

    daily = pd.DataFrame({
        "date": dates,
        "market_day_index": (
            np.arange(
                len(dates)
            )
        ),
        "realized_ic": np.linspace(
            -0.1,
            0.1,
            len(dates),
        ),
        "realized_top_decile_excess": (
            np.linspace(
                -0.02,
                0.02,
                len(dates),
            )
        ),
    })

    matured = daily[
        "realized_ic"
    ].shift(HORIZON)
    recent = matured.rolling(
        20,
        min_periods=20,
    ).mean()

    assert pd.isna(
        recent.iloc[
            HORIZON + 18
        ]
    )
    assert not pd.isna(
        recent.iloc[
            HORIZON + 19
        ]
    )

    model = build_gate_model(
        random_state=42
    )
    X = pd.DataFrame({
        column: np.linspace(
            0,
            1,
            200,
        )
        for column in (
            META_FEATURES
        )
    })
    y = (
        np.arange(200)
        % 2
    )
    model.fit(
        X,
        y,
    )
    prob = model.predict_proba(
        X
    )[:, 1]

    assert len(prob) == 200
    assert np.all(
        (prob >= 0)
        & (prob <= 1)
    )

    print(
        "ML regime-gate "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a point-in-time model-trust "
            "gate from delayed realized IC, score "
            "dispersion and market-state features."
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
        "--thresholds",
        nargs="+",
        type=float,
        default=(
            DEFAULT_THRESHOLDS
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
        threshold <= 0
        or threshold >= 1
        for threshold
        in args.thresholds
    ):
        raise SystemExit(
            "--thresholds must be "
            "strictly between 0 and 1."
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

    validation_score_mask = (
        fit_validation_base_model(
            df,
            model_name=args.model,
            random_state=(
                args.random_state
            ),
        )
    )
    test_score_mask = (
        load_walkforward_scores(
            df,
            root=root,
        )
    )
    add_daily_percentile_score(
        df
    )

    daily = build_daily_meta(
        df
    )

    # Map delayed trust diagnostics back before inner portfolio selection.
    for column in (
        "recent_ic_20",
        "recent_ic_60",
        "recent_ic_120",
        "recent_top_decile_excess_20",
        "recent_top_decile_excess_60",
        "recent_top_decile_excess_120",
        "recent_positive_ic_fraction_60",
        "recent_positive_top_decile_fraction_60",
    ):
        df[column] = df[
            "date"
        ].map(
            daily.set_index(
                "date"
            )[column]
        )

    events = load_mechanical_events(
        root
    )
    costs = current_cost_profile(
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
        inner_search,
        selected_method,
        selected_threshold,
        selected_top_k,
    ) = select_gate_on_2021(
        df,
        daily,
        events,
        thresholds=[
            float(x)
            for x in args.thresholds
        ],
        top_k_values=[
            int(x)
            for x in args.top_k
        ],
        capital=args.capital,
        costs=costs,
        random_state=(
            args.random_state
        ),
    )

    (
        daily,
        gate_folds,
    ) = (
        walkforward_gate_probabilities(
            daily,
            random_state=(
                args.random_state
            ),
        )
    )

    (
        comparison,
        gross_net,
    ) = final_test(
        df,
        daily,
        events,
        root=root,
        method=(
            selected_method
        ),
        threshold=(
            selected_threshold
        ),
        top_k=(
            selected_top_k
        ),
        capital=args.capital,
        costs=costs,
    )

    output_root = (
        root
        / "reports/ml/regime_gate"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        output_root
        / "daily_meta_features.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    inner_search.to_csv(
        output_root
        / "inner_2021_gate_search.csv",
        index=False,
    )
    gate_folds.to_csv(
        output_root
        / "walkforward_gate_folds.csv",
        index=False,
    )
    comparison.to_csv(
        output_root
        / "walkforward_regime_gate_comparison.csv",
        index=False,
    )
    gross_net.to_csv(
        output_root
        / "walkforward_regime_gate_gross_net.csv",
        index=False,
    )

    summary = {
        "base_model": (
            args.model
        ),
        "meta_target": (
            "whether the base model's "
            "top-decile 20-session return "
            "exceeds the eligible-universe "
            "mean on that signal date"
        ),
        "meta_features": (
            META_FEATURES
        ),
        "point_in_time_rule": (
            "realized IC/excess diagnostics "
            "are shifted by 20 market sessions "
            "before entering gate features"
        ),
        "inner_selection": {
            "gate_train": (
                "2019-2020"
            ),
            "gate_tune": (
                "2021"
            ),
            "candidate_methods": [
                "always_on",
                "recent_ic_60 > 0",
                (
                    "logistic trust "
                    "probability thresholds"
                ),
            ],
            "candidate_thresholds": [
                float(x)
                for x in (
                    args.thresholds
                )
            ],
            "candidate_top_k": [
                int(x)
                for x in args.top_k
            ],
            "selection_metric": (
                "2021 net Sharpe; net "
                "CAGR tie-breaker"
            ),
            "selected_method": (
                selected_method
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "selected_top_k": (
                selected_top_k
            ),
        },
        "walkforward_gate": (
            "gate model refit annually on all "
            "prior daily meta rows whose "
            "20-session regime label had matured"
        ),
        "base_predictions": (
            "2019-2021 from model fit through "
            "2018; 2022-2026 from previously "
            "generated annual walk-forward "
            "base-model predictions"
        ),
        "capital": float(
            args.capital
        ),
        "cost_profile": (
            asdict(
                costs
            )
        ),
        "outputs": {
            "daily_meta": (
                "daily_meta_features.csv"
            ),
            "inner_search": (
                "inner_2021_gate_search.csv"
            ),
            "gate_folds": (
                "walkforward_gate_folds.csv"
            ),
            "comparison": (
                "walkforward_regime_gate_comparison.csv"
            ),
            "gross_net": (
                "walkforward_regime_gate_gross_net.csv"
            ),
        },
    }

    (
        output_root
        / "regime_gate_summary.json"
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
            column
            for column in (
                "name",
                "kind",
                "top_k",
                "ending_equity",
                "cagr",
                "max_drawdown",
                "sharpe",
                "trades",
                "total_fees",
            )
            if column
            in comparison.columns
        ]
    ].copy()

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
        "REGIME-GATE COMPARISON ==="
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
        "\n=== REGIME-GATE "
        "EXPERIMENT COMPLETE ==="
    )
    print(
        "Selected on 2021: "
        f"method={selected_method}, "
        f"threshold={selected_threshold}, "
        f"k={selected_top_k}"
    )
    print(
        f"Outputs: {output_root}"
    )


if __name__ == "__main__":
    main()
