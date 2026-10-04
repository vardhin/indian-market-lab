from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
FEATURE_DIR = SCRIPT_PATH.parent
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(FEATURE_DIR),
    )

from feature_registry import (  # noqa: E402
    ATTRIBUTION_ONLY_COLUMNS,
    FEATURE_GROUPS,
    F0_BASELINE,
    registry_records,
)


TARGET_HORIZONS = (
    1,
    5,
    20,
)
PRIMARY_MARKET_PREFERENCE = [
    "NIFTY 100",
    "NIFTY 50",
    "NIFTY 500",
]

CORE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "series",
    "eligible_universe",
    "history_observations",
    "open",
    "high",
    "low",
    "close",
    "turnover",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "turnover_median_20d",
    "turnover_median_60d",
    *[
        name
        for name in F0_BASELINE
        if not name.startswith(
            "log_turnover_"
        )
    ],
]

for horizon in TARGET_HORIZONS:
    CORE_COLUMNS.extend([
        (
            "target_next_open_to_close_"
            f"{horizon}d"
        ),
        (
            "target_close_to_close_"
            f"{horizon}d"
        ),
        (
            "training_eligible_"
            f"{horizon}d"
        ),
        (
            "unsafe_target_window_"
            f"{horizon}d"
        ),
    ])

CORE_COLUMNS = list(
    dict.fromkeys(
        CORE_COLUMNS
    )
)


def _rolling(
    df: pd.DataFrame,
    column: str,
    window: int,
    stat: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window

    rolling = (
        df.groupby(
            "canonical_security_id",
            sort=False,
        )[column]
        .rolling(
            window,
            min_periods=min_periods,
        )
    )

    if stat == "mean":
        out = rolling.mean()
    elif stat == "std":
        out = rolling.std()
    elif stat == "sum":
        out = rolling.sum()
    else:
        raise ValueError(
            f"Unsupported stat: {stat}"
        )

    return out.reset_index(
        level=0,
        drop=True,
    )


def _consecutive_true(
    df: pd.DataFrame,
    condition: pd.Series,
) -> pd.Series:
    condition = (
        condition
        .fillna(False)
        .astype(bool)
    )
    ids = df[
        "canonical_security_id"
    ]

    reset_group = (
        (~condition)
        .groupby(
            ids,
            sort=False,
        )
        .cumsum()
    )

    streak = (
        condition.astype(
            "int16"
        )
        .groupby(
            [
                ids,
                reset_group,
            ],
            sort=False,
        )
        .cumsum()
    )

    return streak.astype(
        "int16"
    )


def discover_panel_columns(
    files: list[Path],
) -> list[str]:
    if not files:
        raise FileNotFoundError(
            "No base research-panel partitions."
        )

    available = set(
        pq.read_schema(
            files[0]
        ).names
    )

    required = {
        "date",
        "market_day_index",
        "canonical_security_id",
        "symbol",
        "eligible_universe",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "turnover_median_20d",
        "turnover_median_60d",
        "training_eligible_20d",
        "target_next_open_to_close_20d",
    }

    missing = sorted(
        required - available
    )
    if missing:
        raise RuntimeError(
            "Base research panel is missing: "
            f"{missing}"
        )

    return [
        column
        for column in CORE_COLUMNS
        if column in available
    ]


def load_base_panel(
    panel_root: Path,
) -> pd.DataFrame:
    files = sorted(
        Path(
            panel_root
        ).glob(
            "date=*/data.parquet"
        )
    )
    columns = (
        discover_panel_columns(
            files
        )
    )

    print(
        f"Loading {len(files):,} base "
        "research-panel partitions..."
    )

    frames: list[
        pd.DataFrame
    ] = []

    for number, path in enumerate(
        files,
        start=1,
    ):
        frames.append(
            pd.read_parquet(
                path,
                columns=columns,
            )
        )
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

    df = (
        df.sort_values(
            [
                "canonical_security_id",
                "date",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    duplicate = df.duplicated(
        [
            "date",
            "canonical_security_id",
        ],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "Base panel has duplicate "
            "date/security rows."
        )

    return df


def add_ohlc_geometry(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    open_ = pd.to_numeric(
        df["adj_open"],
        errors="coerce",
    )
    high = pd.to_numeric(
        df["adj_high"],
        errors="coerce",
    )
    low = pd.to_numeric(
        df["adj_low"],
        errors="coerce",
    )
    close = pd.to_numeric(
        df["adj_close"],
        errors="coerce",
    )

    day_range = (
        high - low
    )
    valid_range = day_range.where(
        day_range > 0
    )
    valid_open = open_.where(
        open_ > 0
    )

    df[
        "close_location_1d"
    ] = (
        (close - low)
        / valid_range
    )
    df[
        "body_to_range_1d"
    ] = (
        (close - open_)
        / valid_range
    )

    upper_body = pd.concat(
        [
            open_,
            close,
        ],
        axis=1,
    ).max(
        axis=1
    )
    lower_body = pd.concat(
        [
            open_,
            close,
        ],
        axis=1,
    ).min(
        axis=1
    )

    df[
        "upper_wick_pct_1d"
    ] = (
        high - upper_body
    ) / valid_open
    df[
        "lower_wick_pct_1d"
    ] = (
        lower_body - low
    ) / valid_open

    previous_close = (
        df.groupby(
            "canonical_security_id",
            sort=False,
        )[
            "adj_close"
        ].shift(1)
    )
    true_high = pd.concat(
        [
            high,
            previous_close,
        ],
        axis=1,
    ).max(
        axis=1
    )
    true_low = pd.concat(
        [
            low,
            previous_close,
        ],
        axis=1,
    ).min(
        axis=1
    )

    df[
        "true_range_pct_1d"
    ] = (
        true_high - true_low
    ) / previous_close.where(
        previous_close > 0
    )

    return df


def add_streak_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    group = df.groupby(
        "canonical_security_id",
        sort=False,
    )
    previous_high = group[
        "adj_high"
    ].shift(1)
    previous_low = group[
        "adj_low"
    ].shift(1)

    return_1d = pd.to_numeric(
        df["return_1d"],
        errors="coerce",
    )
    intraday = pd.to_numeric(
        df["intraday_return_1d"],
        errors="coerce",
    )

    df[
        "up_close_streak"
    ] = _consecutive_true(
        df,
        return_1d > 0,
    )
    df[
        "down_close_streak"
    ] = _consecutive_true(
        df,
        return_1d < 0,
    )
    df[
        "bullish_candle_streak"
    ] = _consecutive_true(
        df,
        intraday > 0,
    )
    df[
        "bearish_candle_streak"
    ] = _consecutive_true(
        df,
        intraday < 0,
    )
    df[
        "higher_high_streak"
    ] = _consecutive_true(
        df,
        pd.to_numeric(
            df["adj_high"],
            errors="coerce",
        )
        > previous_high,
    )
    df[
        "lower_low_streak"
    ] = _consecutive_true(
        df,
        pd.to_numeric(
            df["adj_low"],
            errors="coerce",
        )
        < previous_low,
    )

    return df


def _conditional_streak_history(
    df: pd.DataFrame,
    *,
    state_column: str,
    prefix: str,
    minimum_history: int = 5,
) -> pd.DataFrame:
    ids = df[
        "canonical_security_id"
    ]
    group = df.groupby(
        "canonical_security_id",
        sort=False,
    )

    prior_state = (
        group[
            state_column
        ]
        .shift(1)
        .fillna(0)
        .clip(
            lower=0,
            upper=5,
        )
        .astype(
            "int8"
        )
    )

    realized = pd.to_numeric(
        df["return_1d"],
        errors="coerce",
    )
    current_state = (
        pd.to_numeric(
            df[state_column],
            errors="coerce",
        )
        .fillna(0)
        .clip(
            lower=0,
            upper=5,
        )
        .astype(
            "int8"
        )
    )

    means: dict[
        int,
        pd.Series
    ] = {}
    probs: dict[
        int,
        pd.Series
    ] = {}
    counts: dict[
        int,
        pd.Series
    ] = {}

    for bucket in range(
        1,
        6,
    ):
        matched = (
            prior_state.eq(
                bucket
            )
            & realized.notna()
        )

        count = (
            matched.astype(
                "int32"
            )
            .groupby(
                ids,
                sort=False,
            )
            .cumsum()
        )
        total = (
            realized.where(
                matched,
                0.0,
            )
            .groupby(
                ids,
                sort=False,
            )
            .cumsum()
        )
        positive = (
            (
                realized.gt(0)
                & matched
            )
            .astype(
                "int32"
            )
            .groupby(
                ids,
                sort=False,
            )
            .cumsum()
        )

        means[
            bucket
        ] = (
            total
            / count.where(
                count > 0
            )
        )
        probs[
            bucket
        ] = (
            positive
            / count.where(
                count > 0
            )
        )
        counts[
            bucket
        ] = count

    mean_out = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )
    prob_out = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )
    count_out = pd.Series(
        0,
        index=df.index,
        dtype="int32",
    )

    for bucket in range(
        1,
        6,
    ):
        use = current_state.eq(
            bucket
        )
        count_out.loc[
            use
        ] = counts[
            bucket
        ].loc[
            use
        ]
        mature = (
            use
            & counts[
                bucket
            ].ge(
                minimum_history
            )
        )
        mean_out.loc[
            mature
        ] = means[
            bucket
        ].loc[
            mature
        ]
        prob_out.loc[
            mature
        ] = probs[
            bucket
        ].loc[
            mature
        ]

    df[
        f"{prefix}_mean"
    ] = mean_out
    df[
        f"{prefix}_up_prob"
    ] = prob_out
    df[
        f"{prefix}_count"
    ] = count_out

    return df


def add_stock_tendencies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df[
        "_is_up_close"
    ] = (
        pd.to_numeric(
            df["return_1d"],
            errors="coerce",
        )
        .gt(0)
        .astype(float)
    )
    df[
        "_is_gap_up"
    ] = (
        pd.to_numeric(
            df["gap_return_1d"],
            errors="coerce",
        )
        .gt(0)
        .astype(float)
    )

    for window in (
        20,
        60,
        252,
    ):
        min_periods = max(
            10,
            window // 2,
        )
        df[
            f"intraday_mean_{window}d"
        ] = _rolling(
            df,
            "intraday_return_1d",
            window,
            "mean",
            min_periods=min_periods,
        )
        df[
            f"gap_mean_{window}d"
        ] = _rolling(
            df,
            "gap_return_1d",
            window,
            "mean",
            min_periods=min_periods,
        )
        df[
            f"up_fraction_{window}d"
        ] = _rolling(
            df,
            "_is_up_close",
            window,
            "mean",
            min_periods=min_periods,
        )
        df[
            f"gap_up_fraction_{window}d"
        ] = _rolling(
            df,
            "_is_gap_up",
            window,
            "mean",
            min_periods=min_periods,
        )

    for window in (
        20,
        60,
    ):
        df[
            f"close_location_mean_{window}d"
        ] = _rolling(
            df,
            "close_location_1d",
            window,
            "mean",
            min_periods=max(
                10,
                window // 2,
            ),
        )

    df = (
        _conditional_streak_history(
            df,
            state_column=(
                "down_close_streak"
            ),
            prefix=(
                "hist_next1_after_"
                "down_streak"
            ),
        )
    )
    df = (
        _conditional_streak_history(
            df,
            state_column=(
                "up_close_streak"
            ),
            prefix=(
                "hist_next1_after_"
                "up_streak"
            ),
        )
    )

    df = df.drop(
        columns=[
            "_is_up_close",
            "_is_gap_up",
        ]
    )

    return df


def load_primary_market(
    root: Path,
) -> tuple[
    pd.DataFrame,
    str,
]:
    path = (
        Path(root)
        / "data/processed/"
        "index_benchmarks/"
        "nse_price_indices.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Historical index panel missing: "
            f"{path}. Run "
            "src/data/index_benchmarks.py."
        )

    indices = pd.read_parquet(
        path
    )
    indices["date"] = pd.to_datetime(
        indices["date"],
        errors="coerce",
    ).dt.normalize()
    indices[
        "index_name"
    ] = (
        indices[
            "index_name"
        ]
        .astype("string")
        .str.strip()
    )

    names = set(
        indices[
            "index_name"
        ].dropna()
    )
    selected = next(
        (
            name
            for name
            in PRIMARY_MARKET_PREFERENCE
            if name in names
        ),
        None,
    )

    if selected is None:
        raise RuntimeError(
            "None of the supported broad "
            "market indices are available: "
            f"{PRIMARY_MARKET_PREFERENCE}"
        )

    market = (
        indices.loc[
            indices[
                "index_name"
            ].eq(
                selected
            ),
            [
                "date",
                "open",
                "high",
                "low",
                "close",
            ],
        ]
        .dropna(
            subset=[
                "date",
                "close",
            ]
        )
        .sort_values(
            "date"
        )
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    previous_close = (
        market[
            "close"
        ].shift(1)
    )
    market[
        "market_return_1d"
    ] = (
        market[
            "close"
        ]
        / previous_close.where(
            previous_close > 0
        )
        - 1.0
    )

    for horizon in (
        5,
        20,
        60,
    ):
        market[
            f"market_return_{horizon}d"
        ] = (
            market[
                "close"
            ]
            / market[
                "close"
            ].shift(
                horizon
            ).where(
                lambda s: s > 0
            )
            - 1.0
        )

    market[
        "market_gap_return_1d"
    ] = (
        market[
            "open"
        ]
        / previous_close.where(
            previous_close > 0
        )
        - 1.0
    )
    market[
        "market_intraday_return_1d"
    ] = (
        market[
            "close"
        ]
        / market[
            "open"
        ].where(
            market[
                "open"
            ] > 0
        )
        - 1.0
    )

    market[
        "market_volatility_20d"
    ] = (
        market[
            "market_return_1d"
        ].rolling(
            20,
            min_periods=10,
        ).std()
    )
    market[
        "market_volatility_60d"
    ] = (
        market[
            "market_return_1d"
        ].rolling(
            60,
            min_periods=30,
        ).std()
    )

    market[
        "market_vol_regime_pct_252d"
    ] = (
        market[
            "market_volatility_20d"
        ]
        .rolling(
            252,
            min_periods=126,
        )
        .apply(
            lambda values: (
                float(
                    pd.Series(
                        values
                    )
                    .rank(
                        pct=True
                    )
                    .iloc[-1]
                )
            ),
            raw=False,
        )
    )

    for lag in (
        0,
        1,
        2,
        3,
        5,
    ):
        market[
            f"_market_return_lag{lag}"
        ] = (
            market[
                "market_return_1d"
            ].shift(
                lag
            )
        )

    return (
        market,
        selected,
    )


def _rolling_pair_stats(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    window: int,
    min_periods: int,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    token = (
        f"_pair_{abs(hash((x_col, y_col, window))) % 10**8}"
    )
    x2 = f"{token}_x2"
    y2 = f"{token}_y2"
    xy = f"{token}_xy"

    x = pd.to_numeric(
        df[x_col],
        errors="coerce",
    )
    y = pd.to_numeric(
        df[y_col],
        errors="coerce",
    )

    df[x2] = x * x
    df[y2] = y * y
    df[xy] = x * y

    mx = _rolling(
        df,
        x_col,
        window,
        "mean",
        min_periods=min_periods,
    )
    my = _rolling(
        df,
        y_col,
        window,
        "mean",
        min_periods=min_periods,
    )
    mx2 = _rolling(
        df,
        x2,
        window,
        "mean",
        min_periods=min_periods,
    )
    my2 = _rolling(
        df,
        y2,
        window,
        "mean",
        min_periods=min_periods,
    )
    mxy = _rolling(
        df,
        xy,
        window,
        "mean",
        min_periods=min_periods,
    )

    cov = (
        mxy - mx * my
    )
    var_x = (
        mx2 - mx * mx
    ).clip(
        lower=0
    )
    var_y = (
        my2 - my * my
    ).clip(
        lower=0
    )

    corr = (
        cov
        / np.sqrt(
            var_x * var_y
        ).where(
            (var_x > 0)
            & (var_y > 0)
        )
    )
    beta = (
        cov
        / var_y.where(
            var_y > 0
        )
    )

    df.drop(
        columns=[
            x2,
            y2,
            xy,
        ],
        inplace=True,
    )

    return (
        corr,
        beta,
    )


def add_market_context(
    df: pd.DataFrame,
    root: Path,
) -> tuple[
    pd.DataFrame,
    str,
]:
    df = df.copy()
    market, selected = (
        load_primary_market(
            root
        )
    )

    market_columns = [
        column
        for column in market.columns
        if column not in {
            "open",
            "high",
            "low",
            "close",
        }
    ]

    df = df.merge(
        market[
            market_columns
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )

    for horizon in (
        1,
        5,
        20,
        60,
    ):
        df[
            f"stock_minus_market_{horizon}d"
        ] = (
            pd.to_numeric(
                df[
                    f"return_{horizon}d"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                df[
                    f"market_return_{horizon}d"
                ],
                errors="coerce",
            )
        )

    raw_beta: dict[
        int,
        pd.Series
    ] = {}

    for window in (
        60,
        252,
    ):
        _, beta = (
            _rolling_pair_stats(
                df,
                x_col="return_1d",
                y_col=(
                    "_market_return_lag0"
                ),
                window=window,
                min_periods=max(
                    30,
                    window // 2,
                ),
            )
        )
        raw_beta[
            window
        ] = beta
        df[
            f"market_beta_{window}d"
        ] = (
            beta.groupby(
                df[
                    "canonical_security_id"
                ],
                sort=False,
            )
            .shift(1)
        )

    for lag in (
        0,
        1,
        2,
        3,
        5,
    ):
        corr, _ = (
            _rolling_pair_stats(
                df,
                x_col="return_1d",
                y_col=(
                    f"_market_return_lag{lag}"
                ),
                window=60,
                min_periods=30,
            )
        )
        df[
            f"market_corr_lag{lag}_60d"
        ] = corr

    df[
        "market_response_gap_1d"
    ] = (
        pd.to_numeric(
            df["return_1d"],
            errors="coerce",
        )
        - df[
            "market_beta_60d"
        ]
        * pd.to_numeric(
            df[
                "market_return_1d"
            ],
            errors="coerce",
        )
    )
    df[
        "market_response_gap_20d"
    ] = (
        pd.to_numeric(
            df["return_20d"],
            errors="coerce",
        )
        - df[
            "market_beta_60d"
        ]
        * pd.to_numeric(
            df[
                "market_return_20d"
            ],
            errors="coerce",
        )
    )

    drop_lags = [
        f"_market_return_lag{lag}"
        for lag in (
            0,
            1,
            2,
            3,
            5,
        )
    ]
    df = df.drop(
        columns=drop_lags,
    )

    df[
        "market_context_index"
    ] = selected

    return (
        df,
        selected,
    )


def add_liquidity_risk(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df[
        "turnover_ratio_20d_60d"
    ] = (
        pd.to_numeric(
            df[
                "turnover_median_20d"
            ],
            errors="coerce",
        )
        / pd.to_numeric(
            df[
                "turnover_median_60d"
            ],
            errors="coerce",
        ).where(
            pd.to_numeric(
                df[
                    "turnover_median_60d"
                ],
                errors="coerce",
            ) > 0
        )
    )

    df[
        "volatility_ratio_5d_20d"
    ] = (
        pd.to_numeric(
            df[
                "volatility_5d"
            ],
            errors="coerce",
        )
        / pd.to_numeric(
            df[
                "volatility_20d"
            ],
            errors="coerce",
        ).where(
            pd.to_numeric(
                df[
                    "volatility_20d"
                ],
                errors="coerce",
            ) > 0
        )
    )

    df[
        "volatility_ratio_20d_60d"
    ] = (
        pd.to_numeric(
            df[
                "volatility_20d"
            ],
            errors="coerce",
        )
        / pd.to_numeric(
            df[
                "volatility_60d"
            ],
            errors="coerce",
        ).where(
            pd.to_numeric(
                df[
                    "volatility_60d"
                ],
                errors="coerce",
            ) > 0
        )
    )

    df[
        "range_mean_20d"
    ] = _rolling(
        df,
        "range_pct_1d",
        20,
        "mean",
        min_periods=10,
    )
    range_std = _rolling(
        df,
        "range_pct_1d",
        20,
        "std",
        min_periods=10,
    )
    df[
        "range_zscore_20d"
    ] = (
        pd.to_numeric(
            df[
                "range_pct_1d"
            ],
            errors="coerce",
        )
        - df[
            "range_mean_20d"
        ]
    ) / range_std.where(
        range_std > 0
    )

    return df


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df[
        "day_of_week"
    ] = dates.dt.dayofweek.astype(
        "int8"
    )
    df[
        "month_of_year"
    ] = dates.dt.month.astype(
        "int8"
    )
    df[
        "quarter_of_year"
    ] = dates.dt.quarter.astype(
        "int8"
    )
    df[
        "is_month_end"
    ] = dates.dt.is_month_end.astype(
        "int8"
    )
    df[
        "is_quarter_end"
    ] = dates.dt.is_quarter_end.astype(
        "int8"
    )
    df[
        "is_financial_year_end"
    ] = (
        dates.dt.month.eq(3)
        & dates.dt.is_month_end
    ).astype(
        "int8"
    )

    # Deliberately attribution-only: absolute year is
    # not a default model feature.
    df[
        "calendar_year"
    ] = dates.dt.year.astype(
        "int16"
    )

    return df


def add_cross_sectional_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    mapping = {
        "return_5d": (
            "xs_rank_return_5d"
        ),
        "return_20d": (
            "xs_rank_return_20d"
        ),
        "return_60d": (
            "xs_rank_return_60d"
        ),
        "volatility_20d": (
            "xs_rank_volatility_20d"
        ),
        "turnover_zscore_20d": (
            "xs_rank_turnover_zscore_20d"
        ),
        "stock_minus_market_20d": (
            "xs_rank_stock_minus_market_20d"
        ),
        "market_response_gap_1d": (
            "xs_rank_market_response_gap_1d"
        ),
    }

    for source, target in (
        mapping.items()
    ):
        df[target] = (
            df.groupby(
                "date",
                sort=False,
            )[source]
            .rank(
                method="average",
                pct=True,
            )
        )

    return df


def attach_point_in_time_context(
    df: pd.DataFrame,
    context_path: Path,
) -> tuple[
    pd.DataFrame,
    bool,
]:
    if not context_path.is_file():
        df = df.copy()
        df[
            "largecap_flag_numeric"
        ] = np.nan
        df[
            "market_cap_log"
        ] = np.nan
        df[
            "market_cap_percentile"
        ] = np.nan
        for column in (
            "sector",
            "industry",
            "sector_index_name",
        ):
            df[column] = pd.NA
        return (
            df,
            False,
        )

    context = pd.read_parquet(
        context_path
    )

    required = {
        "asof_date",
        "canonical_security_id",
    }
    missing = (
        required
        - set(
            context.columns
        )
    )
    if missing:
        raise RuntimeError(
            "Point-in-time security context "
            f"is missing {sorted(missing)}"
        )

    context = context.copy()
    context[
        "asof_date"
    ] = pd.to_datetime(
        context[
            "asof_date"
        ],
        errors="coerce",
    ).dt.normalize()
    context[
        "canonical_security_id"
    ] = (
        context[
            "canonical_security_id"
        ]
        .astype("string")
        .str.strip()
    )

    duplicate = context.duplicated(
        [
            "asof_date",
            "canonical_security_id",
        ],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "Security context has duplicate "
            "as-of/security rows."
        )

    keep = [
        "asof_date",
        "canonical_security_id",
        *[
            column
            for column in (
                "sector",
                "industry",
                "sector_index_name",
                "market_cap",
                "largecap_flag",
            )
            if column
            in context.columns
        ],
    ]
    context = context[
        keep
    ]

    left = (
        df.sort_values(
            [
                "date",
                "canonical_security_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )
    right = (
        context.sort_values(
            [
                "asof_date",
                "canonical_security_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="asof_date",
        by="canonical_security_id",
        direction="backward",
        allow_exact_matches=True,
    )

    leaked = (
        merged[
            "asof_date"
        ].notna()
        & merged[
            "asof_date"
        ].gt(
            merged[
                "date"
            ]
        )
    )
    if leaked.any():
        raise RuntimeError(
            "Security context contains "
            "future-dated joins."
        )

    if "largecap_flag" in (
        merged.columns
    ):
        merged[
            "largecap_flag_numeric"
        ] = (
            merged[
                "largecap_flag"
            ]
            .astype(
                "boolean"
            )
            .astype(
                "Float32"
            )
        )
    else:
        merged[
            "largecap_flag_numeric"
        ] = np.nan

    if "market_cap" in (
        merged.columns
    ):
        cap = pd.to_numeric(
            merged[
                "market_cap"
            ],
            errors="coerce",
        )
        merged[
            "market_cap_log"
        ] = np.log1p(
            cap.clip(
                lower=0
            )
        )
        merged[
            "market_cap_percentile"
        ] = (
            cap.groupby(
                merged[
                    "date"
                ],
                sort=False,
            )
            .rank(
                method="average",
                pct=True,
            )
        )
    else:
        merged[
            "market_cap_log"
        ] = np.nan
        merged[
            "market_cap_percentile"
        ] = np.nan

    for column in (
        "sector",
        "industry",
        "sector_index_name",
    ):
        if column not in (
            merged.columns
        ):
            merged[
                column
            ] = pd.NA

    merged = merged.drop(
        columns=[
            "asof_date"
        ],
        errors="ignore",
    )

    return (
        merged,
        True,
    )


def finalize_numeric_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    derived_logs = {
        "log_turnover_median_20d": (
            "turnover_median_20d"
        ),
        "log_turnover_median_60d": (
            "turnover_median_60d"
        ),
    }
    for target, source in (
        derived_logs.items()
    ):
        values = pd.to_numeric(
            df[source],
            errors="coerce",
        )
        df[target] = np.log1p(
            values.clip(
                lower=0
            )
        )

    feature_names = list(
        dict.fromkeys(
            name
            for names
            in FEATURE_GROUPS.values()
            for name in names
        )
    )

    for column in feature_names:
        if column not in df.columns:
            df[column] = np.nan
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
            .astype(
                "float32"
            )
        )

    return df


def build_enriched_features(
    df: pd.DataFrame,
    *,
    root: Path,
    context_path: Path,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    df = add_ohlc_geometry(
        df
    )
    df = add_streak_features(
        df
    )
    df = add_stock_tendencies(
        df
    )
    df, market_index = (
        add_market_context(
            df,
            root,
        )
    )
    df = add_liquidity_risk(
        df
    )
    df = add_calendar_features(
        df
    )
    df = add_cross_sectional_features(
        df
    )
    df, has_context = (
        attach_point_in_time_context(
            df,
            context_path,
        )
    )
    df = finalize_numeric_types(
        df
    )

    metadata = {
        "signal_time": (
            "after_close_t"
        ),
        "earliest_execution": (
            "open_t_plus_1"
        ),
        "primary_market_context": (
            market_index
        ),
        "point_in_time_security_context": (
            bool(
                has_context
            )
        ),
        "absolute_year_model_feature": (
            False
        ),
        "current_snapshot_backfill_allowed": (
            False
        ),
    }

    return (
        df,
        metadata,
    )


def write_partitioned(
    df: pd.DataFrame,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    output_root = Path(
        output_root
    )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. "
                "Use --overwrite."
            )
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    dates = sorted(
        df[
            "date"
        ].dropna().unique()
    )
    print(
        f"Writing {len(dates):,} enriched "
        "daily partitions..."
    )

    for number, date in enumerate(
        dates,
        start=1,
    ):
        day = pd.Timestamp(
            date
        )
        part = df.loc[
            df[
                "date"
            ].eq(
                day
            )
        ]
        path = (
            output_root
            / (
                "date="
                f"{day:%Y-%m-%d}"
            )
            / "data.parquet"
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        part.to_parquet(
            path,
            index=False,
            compression="zstd",
        )

        if (
            number % 500 == 0
            or number == len(
                dates
            )
        ):
            print(
                f"  wrote {number:,}/"
                f"{len(dates):,}",
                flush=True,
            )


def self_test() -> None:
    dates = pd.bdate_range(
        "2020-01-01",
        periods=320,
    )
    rows: list[
        dict
    ] = []

    rng = np.random.default_rng(
        7
    )

    for security_number, cid in (
        enumerate(
            [
                "A",
                "B",
            ]
        )
    ):
        close = (
            100.0
            + security_number
            * 20.0
        )
        for market_index, date in (
            enumerate(
                dates
            )
        ):
            overnight = (
                0.001
                * math.sin(
                    market_index
                    / 9.0
                )
            )
            intraday = (
                float(
                    rng.normal(
                        0.0003,
                        0.01,
                    )
                )
            )
            open_ = close * (
                1.0 + overnight
            )
            new_close = open_ * (
                1.0 + intraday
            )
            high = max(
                open_,
                new_close,
            ) * 1.005
            low = min(
                open_,
                new_close,
            ) * 0.995
            ret = (
                new_close
                / close
                - 1.0
            )

            rows.append({
                "date": date,
                "market_day_index": (
                    market_index
                ),
                "canonical_security_id": (
                    cid
                ),
                "symbol": cid,
                "eligible_universe": True,
                "adj_open": open_,
                "adj_high": high,
                "adj_low": low,
                "adj_close": new_close,
                "adj_volume": 1000.0,
                "turnover_median_20d": (
                    1e8
                ),
                "turnover_median_60d": (
                    1e8
                ),
                "gap_return_1d": (
                    overnight
                ),
                "intraday_return_1d": (
                    intraday
                ),
                "range_pct_1d": (
                    (
                        high - low
                    )
                    / open_
                ),
                "return_1d": ret,
                "return_3d": np.nan,
                "return_5d": np.nan,
                "return_10d": np.nan,
                "return_20d": np.nan,
                "return_60d": np.nan,
                "volatility_5d": 0.01,
                "volatility_20d": 0.01,
                "volatility_60d": 0.01,
                "turnover_zscore_20d": 0.0,
                "volume_zscore_20d": 0.0,
                "active_day_ratio_20d": 1.0,
            })
            close = new_close

    base = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "canonical_security_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    def stock_only(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        out = add_ohlc_geometry(
            frame
        )
        out = add_streak_features(
            out
        )
        out = add_stock_tendencies(
            out
        )
        out = add_liquidity_risk(
            out
        )
        out = add_calendar_features(
            out
        )
        return out

    first = stock_only(
        base.copy()
    )

    cutoff = dates[
        250
    ]
    changed = base.copy()
    future = changed[
        "date"
    ].gt(
        cutoff
    )
    changed.loc[
        future,
        [
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "return_1d",
            "gap_return_1d",
            "intraday_return_1d",
        ],
    ] = 999999.0

    second = stock_only(
        changed
    )

    audit_columns = (
        FEATURE_GROUPS[
            "F1"
        ]
        + FEATURE_GROUPS[
            "F2"
        ]
        + FEATURE_GROUPS[
            "F3"
        ]
        + FEATURE_GROUPS[
            "F9"
        ]
        + [
            name
            for name
            in FEATURE_GROUPS[
                "F10"
            ]
            if name
            != (
                "market_vol_regime_pct_252d"
            )
        ]
    )

    past = first[
        "date"
    ].le(
        cutoff
    )

    for column in audit_columns:
        left = pd.to_numeric(
            first.loc[
                past,
                column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
        right = pd.to_numeric(
            second.loc[
                past,
                column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
        if not np.allclose(
            left,
            right,
            equal_nan=True,
        ):
            raise AssertionError(
                "Future perturbation changed "
                f"past feature: {column}"
            )

    assert (
        "calendar_year"
        not in FEATURE_GROUPS[
            "F9"
        ]
    )

    print(
        "Enriched feature self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a leakage-audited enriched "
            "daily feature panel for swing "
            "research."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--base-panel",
        default=(
            "data/processed/"
            "research_panel"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "data/processed/"
            "research_panel_enriched"
        ),
    )
    ap.add_argument(
        "--security-context",
        default=(
            "data/processed/context/"
            "security_context.parquet"
        ),
    )
    ap.add_argument(
        "--overwrite",
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

    root = Path(
        args.root
    ).resolve()
    base_root = (
        root
        / args.base_panel
    )
    output_root = (
        root
        / args.output
    )
    context_path = (
        root
        / args.security_context
    )

    df = load_base_panel(
        base_root
    )
    df, metadata = (
        build_enriched_features(
            df,
            root=root,
            context_path=(
                context_path
            ),
        )
    )

    write_partitioned(
        df,
        output_root,
        overwrite=(
            args.overwrite
        ),
    )

    report_root = (
        root
        / "reports/features"
    )
    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        report_root
        / "feature_registry.json"
    ).write_text(
        json.dumps(
            registry_records(),
            indent=2,
        )
        + "\n"
    )
    (
        report_root
        / "enriched_panel_metadata.json"
    ).write_text(
        json.dumps(
            {
                **metadata,
                "rows": int(
                    len(df)
                ),
                "dates": int(
                    df[
                        "date"
                    ].nunique()
                ),
                "securities": int(
                    df[
                        "canonical_security_id"
                    ].nunique()
                ),
                "feature_groups": {
                    key: value
                    for key, value
                    in FEATURE_GROUPS.items()
                },
                "attribution_only": (
                    ATTRIBUTION_ONLY_COLUMNS
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(
        "\nEnriched feature panel built."
    )
    print(
        "Security context: "
        + (
            "POINT-IN-TIME attached"
            if metadata[
                "point_in_time_security_context"
            ]
            else (
                "NOT AVAILABLE; F5 remains "
                "missing by design"
            )
        )
    )
    print(
        "Primary market context: "
        f"{metadata['primary_market_context']}"
    )
    print(
        f"Output: {output_root}"
    )


if __name__ == "__main__":
    main()
