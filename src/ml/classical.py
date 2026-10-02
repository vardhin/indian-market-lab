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
import pyarrow.parquet as pq
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
BACKTEST_DIR = SCRIPT_PATH.parents[1] / "backtest"

if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKTEST_DIR),
    )

from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
    run_backtest,
)
from benchmark_report import (  # noqa: E402
    curve_metrics,
    index_curve,
)


HORIZON = 20

TRAIN_START = pd.Timestamp(
    "2010-06-28"
)
TRAIN_END = pd.Timestamp(
    "2018-12-31"
)
VALIDATION_START = pd.Timestamp(
    "2019-01-01"
)
VALIDATION_END = pd.Timestamp(
    "2021-12-31"
)
TEST_START = pd.Timestamp(
    "2022-01-01"
)
TEST_END = pd.Timestamp(
    "2026-09-30"
)

TARGET_COLUMN = (
    "target_next_open_to_close_20d"
)
TRAINING_FLAG = (
    "training_eligible_20d"
)
UNSAFE_TARGET_COLUMN = (
    "unsafe_target_window_20d"
)

BASE_FEATURES = [
    "gap_return_1d",
    "intraday_return_1d",
    "range_pct_1d",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "close_to_ma_5d",
    "close_to_ma_20d",
    "close_to_ma_60d",
    "distance_high_5d",
    "distance_high_20d",
    "distance_high_60d",
    "distance_low_5d",
    "distance_low_20d",
    "distance_low_60d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "turnover_zscore_20d",
    "volume_zscore_20d",
    "active_day_ratio_20d",
]

DERIVED_FEATURES = [
    "log_turnover_median_20d",
    "log_turnover_median_60d",
]

FEATURE_COLUMNS = (
    BASE_FEATURES
    + DERIVED_FEATURES
)

PANEL_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "close",
    "turnover_median_20d",
    "turnover_median_60d",
    "return_60d",
    UNSAFE_TARGET_COLUMN,
    TRAINING_FLAG,
    TARGET_COLUMN,
    *BASE_FEATURES,
]


def required_unique(
    values: list[str],
) -> list[str]:
    return list(
        dict.fromkeys(values)
    )


PANEL_COLUMNS = required_unique(
    PANEL_COLUMNS
)


def load_ml_panel(
    panel_root: Path,
) -> pd.DataFrame:
    panel_root = Path(panel_root)

    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No research-panel partitions "
            f"under {panel_root}"
        )

    schema = pq.read_schema(
        files[0]
    )
    available = set(
        schema.names
    )

    missing = [
        column
        for column in PANEL_COLUMNS
        if column not in available
    ]

    if missing:
        raise RuntimeError(
            "Research panel is missing "
            "ML-required columns: "
            f"{missing}"
        )

    frames: list[
        pd.DataFrame
    ] = []

    print(
        f"Loading {len(files):,} "
        "research-panel partitions "
        "for ML..."
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        frame = pd.read_parquet(
            path,
            columns=PANEL_COLUMNS,
        )
        frames.append(frame)

        if (
            number % 500 == 0
            or number == len(files)
        ):
            print(
                f"  loaded {number:,}/"
                f"{len(files):,}",
                flush=True,
            )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    df["market_day_index"] = (
        pd.to_numeric(
            df["market_day_index"],
            errors="coerce",
        ).astype("Int64")
    )

    for column in required_unique(
        [
            "open",
            "close",
            "turnover_median_20d",
            "turnover_median_60d",
            TARGET_COLUMN,
            *BASE_FEATURES,
        ]
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in (
        "eligible_universe",
        TRAINING_FLAG,
        UNSAFE_TARGET_COLUMN,
    ):
        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    df[
        "canonical_security_id"
    ] = (
        df[
            "canonical_security_id"
        ]
        .astype("string")
        .str.strip()
    )
    df["symbol"] = (
        df["symbol"]
        .astype("string")
        .str.strip()
    )

    df[
        "log_turnover_median_20d"
    ] = np.log1p(
        df[
            "turnover_median_20d"
        ].clip(lower=0)
    )
    df[
        "log_turnover_median_60d"
    ] = np.log1p(
        df[
            "turnover_median_60d"
        ].clip(lower=0)
    )

    for column in FEATURE_COLUMNS:
        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .astype("float32")
        )

    df[TARGET_COLUMN] = (
        pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .astype("float32")
    )

    duplicate = df.duplicated(
        [
            "market_day_index",
            "canonical_security_id",
        ],
        keep=False,
    )

    if duplicate.any():
        sample = df.loc[
            duplicate,
            [
                "date",
                "symbol",
                "canonical_security_id",
            ],
        ].head(20)

        raise RuntimeError(
            "ML panel contains duplicate "
            "date/security rows. Sample:\n"
            + sample.to_string(
                index=False
            )
        )

    return (
        df.sort_values(
            [
                "market_day_index",
                "canonical_security_id",
            ]
        )
        .reset_index(drop=True)
    )


def last_market_index_on_or_before(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> int:
    values = (
        df.loc[
            df["date"].le(
                pd.Timestamp(
                    cutoff
                ).normalize()
            ),
            "market_day_index",
        ]
        .dropna()
        .astype(int)
    )

    if values.empty:
        raise RuntimeError(
            "No market day exists on or "
            f"before {cutoff.date()}"
        )

    return int(
        values.max()
    )


def add_target_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    safe = (
        df[TRAINING_FLAG]
        & df[TARGET_COLUMN].notna()
    )

    df[
        "target_rank_20d"
    ] = np.nan

    df.loc[
        safe,
        "target_rank_20d",
    ] = (
        df.loc[
            safe
        ]
        .groupby(
            "date",
            sort=False,
        )[TARGET_COLUMN]
        .rank(
            method="average",
            pct=True,
        )
        .astype("float32")
    )

    return df


def split_masks(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    train_cutoff_index = (
        last_market_index_on_or_before(
            df,
            TRAIN_END,
        )
    )
    validation_cutoff_index = (
        last_market_index_on_or_before(
            df,
            VALIDATION_END,
        )
    )

    training_base = (
        df[TRAINING_FLAG]
        & df[
            "target_rank_20d"
        ].notna()
    )

    train = (
        training_base
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

    validation = (
        training_base
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

    train_validation_refit = (
        training_base
        & df["date"].between(
            TRAIN_START,
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

    test_labels = (
        training_base
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
    )

    test_scoring = (
        df["eligible_universe"]
        & df["date"].between(
            TEST_START,
            TEST_END,
            inclusive="both",
        )
    )

    return {
        "train": train,
        "validation": validation,
        "train_validation_refit": (
            train_validation_refit
        ),
        "test_labels": test_labels,
        "test_scoring": test_scoring,
    }


def maybe_sample_mask(
    mask: pd.Series,
    *,
    max_rows: int,
    random_state: int,
) -> pd.Series:
    if (
        max_rows <= 0
        or int(mask.sum())
        <= max_rows
    ):
        return mask

    positions = np.flatnonzero(
        mask.to_numpy()
    )

    rng = np.random.default_rng(
        random_state
    )
    selected = rng.choice(
        positions,
        size=max_rows,
        replace=False,
    )

    out = pd.Series(
        False,
        index=mask.index,
    )
    out.iloc[selected] = True

    return out


def build_model(
    name: str,
    *,
    random_state: int,
):
    if name == "ridge":
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
                Ridge(
                    alpha=10.0,
                ),
            ),
        ])

    if name == "hist_gb":
        return (
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=25,
                random_state=(
                    random_state
                ),
            )
        )

    if name == "random_forest":
        return Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    bootstrap=True,
                    max_samples=0.35,
                    n_jobs=-1,
                    random_state=(
                        random_state
                    ),
                ),
            ),
        ])

    raise ValueError(
        f"Unknown model: {name}"
    )


def fit_model(
    model,
    df: pd.DataFrame,
    mask: pd.Series,
):
    X = df.loc[
        mask,
        FEATURE_COLUMNS,
    ]
    y = df.loc[
        mask,
        "target_rank_20d",
    ].astype("float32")

    model.fit(
        X,
        y,
    )

    return model


def predict_mask(
    model,
    df: pd.DataFrame,
    mask: pd.Series,
) -> np.ndarray:
    if int(
        mask.sum()
    ) == 0:
        return np.array(
            [],
            dtype=float,
        )

    return np.asarray(
        model.predict(
            df.loc[
                mask,
                FEATURE_COLUMNS,
            ]
        ),
        dtype=float,
    )


def cross_sectional_diagnostics(
    df: pd.DataFrame,
    mask: pd.Series,
    scores: np.ndarray,
) -> dict:
    view = df.loc[
        mask,
        [
            "date",
            TARGET_COLUMN,
            "target_rank_20d",
        ],
    ].copy()

    if len(view) != len(
        scores
    ):
        raise RuntimeError(
            "Prediction length does not "
            "match diagnostic rows."
        )

    view["score"] = scores

    daily_rows: list[
        dict
    ] = []

    for day, group in view.groupby(
        "date",
        sort=True,
    ):
        group = group.loc[
            group["score"].notna()
            & group[
                "target_rank_20d"
            ].notna()
            & group[
                TARGET_COLUMN
            ].notna()
        ].copy()

        if len(group) < 10:
            continue

        group[
            "score_rank"
        ] = group[
            "score"
        ].rank(
            method="average",
            pct=True,
        )

        ic = group[
            [
                "score_rank",
                "target_rank_20d",
            ]
        ].corr().iloc[
            0,
            1,
        ]

        top = group.loc[
            group[
                "score_rank"
            ].ge(0.90)
        ]

        daily_rows.append({
            "date": day,
            "rows": int(
                len(group)
            ),
            "ic": float(ic)
            if not pd.isna(ic)
            else np.nan,
            "top_decile_return": float(
                top[
                    TARGET_COLUMN
                ].mean()
            )
            if len(top)
            else np.nan,
            "universe_return": float(
                group[
                    TARGET_COLUMN
                ].mean()
            ),
        })

    daily = pd.DataFrame(
        daily_rows
    )

    if daily.empty:
        return {
            "rows": int(
                len(view)
            ),
            "dates": 0,
            "mean_daily_ic": None,
            "median_daily_ic": None,
            "positive_ic_fraction": None,
            "mean_top_decile_return": None,
            "mean_universe_return": None,
            "mean_top_decile_excess": None,
        }

    daily[
        "top_decile_excess"
    ] = (
        daily[
            "top_decile_return"
        ]
        - daily[
            "universe_return"
        ]
    )

    return {
        "rows": int(
            len(view)
        ),
        "dates": int(
            len(daily)
        ),
        "mean_daily_ic": float(
            daily["ic"].mean()
        ),
        "median_daily_ic": float(
            daily["ic"].median()
        ),
        "positive_ic_fraction": float(
            daily["ic"].gt(0).mean()
        ),
        "mean_top_decile_return": float(
            daily[
                "top_decile_return"
            ].mean()
        ),
        "mean_universe_return": float(
            daily[
                "universe_return"
            ].mean()
        ),
        "mean_top_decile_excess": float(
            daily[
                "top_decile_excess"
            ].mean()
        ),
    }


def train_validation_models(
    df: pd.DataFrame,
    masks: dict[
        str,
        pd.Series,
    ],
    *,
    model_names: list[str],
    max_train_rows: int,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    str,
]:
    train_mask = (
        maybe_sample_mask(
            masks["train"],
            max_rows=(
                max_train_rows
            ),
            random_state=(
                random_state
            ),
        )
    )

    comparison: list[
        dict
    ] = []

    print(
        "\n=== MODEL SELECTION "
        "(TRAIN -> VALIDATION) ==="
    )
    print(
        "Train rows: "
        f"{int(train_mask.sum()):,}"
    )
    print(
        "Validation rows: "
        f"{int(masks['validation'].sum()):,}"
    )

    for model_name in model_names:
        print(
            f"\nTraining {model_name}..."
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
            masks[
                "validation"
            ],
        )

        diagnostics = (
            cross_sectional_diagnostics(
                df,
                masks[
                    "validation"
                ],
                scores,
            )
        )

        comparison.append({
            "model": model_name,
            "train_rows": int(
                train_mask.sum()
            ),
            "validation_rows": int(
                masks[
                    "validation"
                ].sum()
            ),
            **diagnostics,
        })

        print(
            "  validation mean IC: "
            f"{diagnostics['mean_daily_ic']:.4f}"
        )
        print(
            "  positive IC days:    "
            f"{diagnostics['positive_ic_fraction']:.2%}"
        )
        print(
            "  top-decile excess:   "
            f"{diagnostics['mean_top_decile_excess']:.4%}"
        )

    comparison_df = (
        pd.DataFrame(
            comparison
        )
        .sort_values(
            [
                "mean_daily_ic",
                "mean_top_decile_excess",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if comparison_df.empty:
        raise RuntimeError(
            "No validation model results."
        )

    selected = str(
        comparison_df.iloc[0][
            "model"
        ]
    )

    return (
        comparison_df,
        selected,
    )


def refit_selected_model(
    df: pd.DataFrame,
    masks: dict[
        str,
        pd.Series,
    ],
    *,
    selected_model: str,
    max_refit_rows: int,
    random_state: int,
):
    refit_mask = (
        maybe_sample_mask(
            masks[
                "train_validation_refit"
            ],
            max_rows=(
                max_refit_rows
            ),
            random_state=(
                random_state + 1
            ),
        )
    )

    print(
        "\nRefitting selected model "
        f"{selected_model} on "
        f"{int(refit_mask.sum()):,} "
        "pre-2022 rows..."
    )

    model = build_model(
        selected_model,
        random_state=(
            random_state
        ),
    )

    fit_model(
        model,
        df,
        refit_mask,
    )

    return (
        model,
        refit_mask,
    )


def run_test_portfolios(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    model_name: str,
    model,
    masks: dict[
        str,
        pd.Series,
    ],
    top_k_values: list[int],
    capital: float,
    costs: CostProfile,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    test_score_mask = (
        masks[
            "test_scoring"
        ]
    )
    test_label_mask = (
        masks[
            "test_labels"
        ]
    )

    scores = predict_mask(
        model,
        df,
        test_score_mask,
    )

    df = df.copy()
    df["ml_score"] = np.nan
    df.loc[
        test_score_mask,
        "ml_score",
    ] = scores

    label_scores = df.loc[
        test_label_mask,
        "ml_score",
    ].to_numpy(
        dtype=float
    )

    diagnostics = (
        cross_sectional_diagnostics(
            df,
            test_label_mask,
            label_scores,
        )
    )

    print(
        "\n=== HELD-OUT TEST "
        "PREDICTION DIAGNOSTICS ==="
    )
    print(
        "Mean daily IC:       "
        f"{diagnostics['mean_daily_ic']:.4f}"
    )
    print(
        "Positive IC days:    "
        f"{diagnostics['positive_ic_fraction']:.2%}"
    )
    print(
        "Top-decile excess:   "
        f"{diagnostics['mean_top_decile_excess']:.4%}"
    )

    test_panel = (
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
    curves: list[
        pd.DataFrame
    ] = []

    for top_k in top_k_values:
        print(
            f"\nBacktesting ML "
            f"{model_name} h20/k{top_k}..."
        )

        result = run_backtest(
            test_panel,
            events,
            strategy=(
                f"ml_{model_name}"
            ),
            score_column=(
                "ml_score"
            ),
            score_direction=1.0,
            initial_capital=(
                capital
            ),
            top_k=top_k,
            holding_sessions=(
                HORIZON
            ),
            costs=costs,
        )

        row = {
            "name": (
                f"ml_{model_name}_"
                f"h20_k{top_k}"
            ),
            "kind": "ml_strategy",
            **result["metrics"],
        }
        rows.append(row)

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
        curve["name"] = row[
            "name"
        ]
        curves.append(curve)

        for baseline in (
            "momentum_60d",
            "liquidity_control",
        ):
            baseline_result = (
                run_backtest(
                    test_panel,
                    events,
                    strategy=baseline,
                    initial_capital=(
                        capital
                    ),
                    top_k=top_k,
                    holding_sessions=(
                        HORIZON
                    ),
                    costs=costs,
                )
            )

            baseline_name = (
                f"{baseline}_"
                f"h20_k{top_k}"
            )

            rows.append({
                "name": baseline_name,
                "kind": (
                    "deterministic_strategy"
                ),
                **baseline_result[
                    "metrics"
                ],
            })

            baseline_curve = (
                baseline_result[
                    "equity"
                ]
                [
                    [
                        "date",
                        "equity",
                    ]
                ]
                .copy()
            )
            baseline_curve[
                "name"
            ] = baseline_name
            curves.append(
                baseline_curve
            )

    predictions = df.loc[
        test_score_mask,
        [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "ml_score",
            TARGET_COLUMN,
            "target_rank_20d",
        ],
    ].copy()

    return (
        pd.DataFrame(rows),
        pd.concat(
            curves,
            ignore_index=True,
        ),
        predictions,
    )


def add_index_benchmarks(
    root: Path,
    *,
    test_panel_dates: set[
        pd.Timestamp
    ],
    capital: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    path = (
        root
        / "data/processed/"
        "index_benchmarks/"
        "nse_price_indices.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Index benchmark data missing. "
            "Run src/data/index_benchmarks.py."
        )

    indices = pd.read_parquet(
        path
    )

    rows: list[
        dict
    ] = []
    curves: list[
        pd.DataFrame
    ] = []

    for index_name in (
        "NIFTY 50",
        "NIFTY 500",
    ):
        source = indices.loc[
            indices[
                "requested_index"
            ].eq(index_name)
        ].copy()

        curve = index_curve(
            source,
            initial_capital=(
                capital
            ),
            allowed_dates=(
                test_panel_dates
            ),
        )

        metric = curve_metrics(
            curve,
            name=index_name,
            kind="price_index",
        )

        rows.append({
            **metric,
            "strategy": None,
            "top_k": None,
            "holding_sessions": None,
            "trades": None,
            "win_rate": None,
            "total_fees": 0.0,
            "turnover": None,
            "average_exposure": 1.0,
        })

        x = curve[
            [
                "date",
                "equity",
            ]
        ].copy()
        x["name"] = (
            index_name
        )
        curves.append(x)

    return (
        pd.DataFrame(rows),
        pd.concat(
            curves,
            ignore_index=True,
        ),
    )


def normalize_portfolio_metrics(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    keep = [
        "name",
        "kind",
        "starting_capital",
        "ending_equity",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "annualized_volatility",
        "trades",
        "win_rate",
        "total_fees",
        "turnover",
        "average_exposure",
        "top_k",
        "holding_sessions",
    ]

    for column in keep:
        if column not in rows.columns:
            rows[column] = np.nan

    return rows[
        keep
    ].copy()


def self_test() -> None:
    dates = pd.bdate_range(
        "2018-11-01",
        "2022-03-31",
    )

    frame = pd.DataFrame({
        "date": dates,
        "market_day_index": (
            np.arange(
                len(dates)
            )
        ),
        TRAINING_FLAG: True,
        TARGET_COLUMN: np.linspace(
            -0.2,
            0.2,
            len(dates),
        ),
        "eligible_universe": True,
    })

    frame[
        "target_rank_20d"
    ] = 0.5

    masks = split_masks(
        frame
    )

    train_cutoff = (
        last_market_index_on_or_before(
            frame,
            TRAIN_END,
        )
    )
    validation_cutoff = (
        last_market_index_on_or_before(
            frame,
            VALIDATION_END,
        )
    )

    assert (
        frame.loc[
            masks["train"],
            "market_day_index",
        ].max()
        <= train_cutoff
        - HORIZON
    )
    assert (
        frame.loc[
            masks[
                "validation"
            ],
            "market_day_index",
        ].max()
        <= validation_cutoff
        - HORIZON
    )
    assert not (
        frame.loc[
            masks["test_labels"],
            "date",
        ]
        < TEST_START
    ).any()

    diag_frame = pd.DataFrame({
        "date": (
            [pd.Timestamp(
                "2026-01-01"
            )] * 20
        ),
        TARGET_COLUMN: (
            np.arange(20)
            / 100.0
        ),
        "target_rank_20d": (
            (
                np.arange(20)
                + 1
            )
            / 20.0
        ),
    })
    diag_mask = pd.Series(
        True,
        index=diag_frame.index,
    )

    diag = (
        cross_sectional_diagnostics(
            diag_frame,
            diag_mask,
            np.arange(
                20,
                dtype=float,
            ),
        )
    )

    assert (
        diag["mean_daily_ic"]
        > 0.99
    )
    assert (
        diag[
            "mean_top_decile_excess"
        ]
        > 0
    )

    print(
        "Classical ML temporal "
        "pipeline self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Leakage-safe classical ML "
            "cross-sectional ranking experiment "
            "for NSE cash equities."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=[
            "ridge",
            "hist_gb",
            "random_forest",
        ],
        default=[
            "ridge",
            "hist_gb",
        ],
    )
    ap.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[
            5,
            10,
        ],
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
        help=(
            "Optional deterministic random cap "
            "for each initial model fit; 0 uses all rows."
        ),
    )
    ap.add_argument(
        "--max-refit-rows",
        type=int,
        default=0,
        help=(
            "Optional deterministic random cap "
            "for train+validation refit; 0 uses all rows."
        ),
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

    if args.capital <= 0:
        raise SystemExit(
            "--capital must be > 0"
        )

    if any(
        k < 1
        for k in args.top_k
    ):
        raise SystemExit(
            "--top-k values must be >= 1"
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
    masks = split_masks(
        df
    )

    split_summary = {
        name: int(
            mask.sum()
        )
        for name, mask
        in masks.items()
    }

    print(
        "\n=== TEMPORAL SPLIT ==="
    )
    print(
        f"Train:      {TRAIN_START.date()} "
        f"-> {TRAIN_END.date()} "
        f"(purged {HORIZON} sessions)"
    )
    print(
        "Validation: "
        f"{VALIDATION_START.date()} "
        f"-> {VALIDATION_END.date()} "
        f"(purged {HORIZON} sessions)"
    )
    print(
        f"Test:       {TEST_START.date()} "
        f"-> {TEST_END.date()}"
    )

    for name, count in (
        split_summary.items()
    ):
        print(
            f"{name:24s} "
            f"{count:,}"
        )

    comparison, selected = (
        train_validation_models(
            df,
            masks,
            model_names=list(
                args.models
            ),
            max_train_rows=(
                args.max_train_rows
            ),
            random_state=(
                args.random_state
            ),
        )
    )

    print(
        "\nSelected by validation "
        "mean daily IC: "
        f"{selected}"
    )

    selected_model, refit_mask = (
        refit_selected_model(
            df,
            masks,
            selected_model=(
                selected
            ),
            max_refit_rows=(
                args.max_refit_rows
            ),
            random_state=(
                args.random_state
            ),
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
    events = load_mechanical_events(
        root
    )

    portfolio_rows, curves, predictions = (
        run_test_portfolios(
            df,
            events,
            model_name=(
                selected
            ),
            model=(
                selected_model
            ),
            masks=masks,
            top_k_values=list(
                args.top_k
            ),
            capital=args.capital,
            costs=costs,
        )
    )

    test_dates = set(
        pd.to_datetime(
            df.loc[
                df["date"].between(
                    TEST_START,
                    TEST_END,
                    inclusive="both",
                ),
                "date",
            ]
        ).dt.normalize()
    )

    index_rows, index_curves = (
        add_index_benchmarks(
            root,
            test_panel_dates=(
                test_dates
            ),
            capital=args.capital,
        )
    )

    portfolio_rows = (
        normalize_portfolio_metrics(
            portfolio_rows
        )
    )
    index_rows = (
        normalize_portfolio_metrics(
            index_rows
        )
    )

    final_comparison = (
        pd.concat(
            [
                portfolio_rows,
                index_rows,
            ],
            ignore_index=True,
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

    output_root = (
        root
        / "reports/ml/classical"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_root = (
        output_root
        / "models"
    )
    models_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_root
        / "validation_model_comparison.csv",
        index=False,
    )

    final_comparison.to_csv(
        output_root
        / "test_portfolio_comparison.csv",
        index=False,
    )

    predictions.to_parquet(
        output_root
        / "test_predictions.parquet",
        index=False,
        compression="zstd",
    )

    all_curves = pd.concat(
        [
            curves,
            index_curves,
        ],
        ignore_index=True,
    )

    all_curves.to_csv(
        output_root
        / "test_equity_curves_long.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    model_path = (
        models_root
        / (
            f"{selected}_"
            "selected.joblib"
        )
    )
    joblib.dump(
        selected_model,
        model_path,
    )

    selected_validation = (
        comparison.loc[
            comparison[
                "model"
            ].eq(selected)
        ]
        .iloc[0]
        .to_dict()
    )

    test_prediction_scores = (
        predict_mask(
            selected_model,
            df,
            masks[
                "test_labels"
            ],
        )
    )
    test_diagnostics = (
        cross_sectional_diagnostics(
            df,
            masks[
                "test_labels"
            ],
            test_prediction_scores,
        )
    )

    summary = {
        "objective": (
            "cross-sectional ranking of "
            "next-open to close(t+20) return"
        ),
        "horizon_sessions": (
            HORIZON
        ),
        "target_column": (
            TARGET_COLUMN
        ),
        "target_transformation": (
            "within-date percentile rank "
            "computed only for training-eligible rows"
        ),
        "features": (
            FEATURE_COLUMNS
        ),
        "temporal_split": {
            "train": {
                "start": str(
                    TRAIN_START.date()
                ),
                "end": str(
                    TRAIN_END.date()
                ),
                "boundary_purge_sessions": (
                    HORIZON
                ),
            },
            "validation": {
                "start": str(
                    VALIDATION_START.date()
                ),
                "end": str(
                    VALIDATION_END.date()
                ),
                "boundary_purge_sessions": (
                    HORIZON
                ),
            },
            "test": {
                "start": str(
                    TEST_START.date()
                ),
                "end": str(
                    TEST_END.date()
                ),
            },
        },
        "split_rows": (
            split_summary
        ),
        "candidate_models": list(
            args.models
        ),
        "selection_metric": (
            "validation mean daily "
            "cross-sectional Spearman IC"
        ),
        "selected_model": (
            selected
        ),
        "selected_validation_metrics": (
            selected_validation
        ),
        "test_prediction_metrics": (
            test_diagnostics
        ),
        "refit_rows": int(
            refit_mask.sum()
        ),
        "capital": float(
            args.capital
        ),
        "top_k_values": list(
            args.top_k
        ),
        "cost_profile": (
            "current_2026_delivery"
        ),
        "cost_parameters": (
            asdict(costs)
        ),
        "model_file": str(
            model_path
        ),
        "outputs": {
            "validation_model_comparison": str(
                output_root
                / "validation_model_comparison.csv"
            ),
            "test_portfolio_comparison": str(
                output_root
                / "test_portfolio_comparison.csv"
            ),
            "test_predictions": str(
                output_root
                / "test_predictions.parquet"
            ),
            "test_equity_curves": str(
                output_root
                / "test_equity_curves_long.csv"
            ),
        },
    }

    summary_path = (
        output_root
        / "experiment_summary.json"
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
        final_comparison[
            [
                "name",
                "kind",
                "ending_equity",
                "cagr",
                "max_drawdown",
                "sharpe",
                "trades",
                "total_fees",
            ]
        ]
        .copy()
    )

    preview["cagr"] *= 100.0
    preview[
        "max_drawdown"
    ] *= 100.0

    print(
        "\n=== HELD-OUT TEST "
        "PORTFOLIO COMPARISON ==="
    )
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
                        ""
                        if pd.isna(x)
                        else f"{x:,.2f}"
                    )
                ),
            },
        )
    )

    print(
        "\n=== CLASSICAL ML "
        "EXPERIMENT COMPLETE ==="
    )
    print(
        f"Selected: {selected}"
    )
    print(
        f"Outputs:  {output_root}"
    )


if __name__ == "__main__":
    main()
