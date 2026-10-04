from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LOOKBACKS = (5, 20, 60, 120, 252)
MARKET_LOOKBACKS = (1, 5, 20, 60)
LEAD_LAGS = (0, 1, 2, 3, 5)
TENDENCY_STREAK_CAP = 5

SOURCE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "turnover",
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
]

REGISTRY_COLUMNS = [
    "feature",
    "family",
    "source",
    "availability_time",
    "lookback_sessions",
    "point_in_time",
    "uses_cross_section",
    "status",
    "notes",
]


def _safe_div(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denom = pd.to_numeric(
        denominator,
        errors="coerce",
    )
    return (
        pd.to_numeric(
            numerator,
            errors="coerce",
        )
        / denom.where(
            denom.abs() > 1e-12
        )
    )


def _rolling(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    stat: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(
            3,
            window // 2,
        )

    grouped = (
        df.groupby(
            "canonical_security_id",
            sort=False,
        )[value_col]
        .rolling(
            window,
            min_periods=min_periods,
        )
    )

    if stat == "mean":
        out = grouped.mean()
    elif stat == "std":
        out = grouped.std()
    elif stat == "sum":
        out = grouped.sum()
    elif stat == "max":
        out = grouped.max()
    elif stat == "min":
        out = grouped.min()
    else:
        raise ValueError(
            f"Unsupported rolling stat: {stat}"
        )

    return out.reset_index(
        level=0,
        drop=True,
    )


def _register(
    rows: list[dict],
    feature: str,
    family: str,
    source: str,
    *,
    lookback: int | str,
    cross_section: bool = False,
    status: str = "implemented",
    notes: str = "",
) -> None:
    rows.append({
        "feature": feature,
        "family": family,
        "source": source,
        "availability_time": "after_close_t",
        "lookback_sessions": lookback,
        "point_in_time": True,
        "uses_cross_section": bool(
            cross_section
        ),
        "status": status,
        "notes": notes,
    })


def discover_panel(
    panel_root: Path,
) -> tuple[list[Path], list[str]]:
    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            "No frozen research-panel "
            f"partitions under {panel_root}"
        )

    schema = pq.read_schema(
        files[0]
    )
    available = set(
        schema.names
    )
    missing = [
        col
        for col in SOURCE_COLUMNS
        if col not in available
    ]
    if missing:
        raise RuntimeError(
            "Frozen research panel lacks "
            f"required v2 source columns: {missing}"
        )

    return files, SOURCE_COLUMNS


def load_panel(
    panel_root: Path,
) -> pd.DataFrame:
    files, columns = (
        discover_panel(
            panel_root
        )
    )

    print(
        f"Loading {len(files):,} frozen "
        "research-panel partitions..."
    )

    frames: list[
        pd.DataFrame
    ] = []
    for i, path in enumerate(
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
            i % 500 == 0
            or i == len(files)
        ):
            print(
                f"  loaded {i:,}/"
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
    df["canonical_security_id"] = (
        df[
            "canonical_security_id"
        ]
        .astype("string")
        .str.strip()
    )

    numeric = [
        col
        for col in SOURCE_COLUMNS
        if col not in {
            "date",
            "canonical_security_id",
            "symbol",
            "eligible_universe",
        }
    ]
    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
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
            "Duplicate date/security keys "
            "in frozen research panel."
        )

    return (
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


def load_market_context(
    root: Path,
) -> pd.DataFrame:
    path = (
        root
        / "data/processed/"
        "index_benchmarks/"
        "nse_price_indices.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Missing NSE benchmark dataset: "
            f"{path}"
        )

    raw = pd.read_parquet(
        path
    )
    required = {
        "date",
        "requested_index",
        "close",
    }
    missing = (
        required
        - set(raw.columns)
    )
    if missing:
        raise RuntimeError(
            "Benchmark dataset lacks: "
            f"{sorted(missing)}"
        )

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce",
    ).dt.normalize()
    raw["close"] = pd.to_numeric(
        raw["close"],
        errors="coerce",
    )

    frames: list[
        pd.DataFrame
    ] = []
    for index_name, prefix in (
        ("NIFTY 50", "nifty50"),
        ("NIFTY 500", "nifty500"),
    ):
        d = (
            raw.loc[
                raw[
                    "requested_index"
                ].eq(
                    index_name
                ),
                [
                    "date",
                    "close",
                ],
            ]
            .dropna()
            .sort_values(
                "date"
            )
            .drop_duplicates(
                "date",
                keep="last",
            )
            .rename(
                columns={
                    "close": (
                        f"{prefix}_close"
                    )
                }
            )
            .reset_index(
                drop=True
            )
        )

        close_col = (
            f"{prefix}_close"
        )
        for lookback in (
            1,
            2,
            3,
            5,
            20,
            60,
        ):
            d[
                f"{prefix}_return_{lookback}d"
            ] = (
                d[close_col]
                / d[close_col].shift(
                    lookback
                )
                - 1.0
            )
        frames.append(d)

    market = frames[0].merge(
        frames[1],
        on="date",
        how="outer",
        validate="one_to_one",
    )
    return market.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


def _signed_streak(
    values: pd.Series,
) -> pd.Series:
    arr = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )
    out = np.zeros(
        len(arr),
        dtype=np.int16,
    )
    current = 0
    previous_sign = 0

    for i, value in enumerate(
        arr
    ):
        if (
            not np.isfinite(
                value
            )
            or abs(value) <= 1e-15
        ):
            current = 0
            previous_sign = 0
            out[i] = 0
            continue

        sign = (
            1
            if value > 0
            else -1
        )
        if sign == previous_sign:
            current += sign
        else:
            current = sign

        previous_sign = sign
        out[i] = current

    return pd.Series(
        out,
        index=values.index,
        dtype="int16",
    )


def add_ohlc_features(
    df: pd.DataFrame,
    registry: list[dict],
) -> None:
    open_ = df[
        "adj_open"
    ]
    high = df[
        "adj_high"
    ]
    low = df[
        "adj_low"
    ]
    close = df[
        "adj_close"
    ]

    candle_range = (
        high - low
    )
    body = (
        close - open_
    )

    features = {
        "body_pct_1d": (
            _safe_div(
                body,
                open_,
            )
        ),
        "upper_wick_pct_1d": (
            _safe_div(
                high
                - pd.concat(
                    [
                        open_,
                        close,
                    ],
                    axis=1,
                ).max(axis=1),
                open_,
            )
        ),
        "lower_wick_pct_1d": (
            _safe_div(
                pd.concat(
                    [
                        open_,
                        close,
                    ],
                    axis=1,
                ).min(axis=1)
                - low,
                open_,
            )
        ),
        "close_location_1d": (
            _safe_div(
                close - low,
                candle_range,
            )
        ),
        "body_to_range_1d": (
            _safe_div(
                body.abs(),
                candle_range,
            )
        ),
    }

    for name, values in (
        features.items()
    ):
        df[name] = (
            values.astype(
                "float32"
            )
        )
        _register(
            registry,
            name,
            "F1_ohlc_geometry",
            "adjusted daily OHLC",
            lookback=1,
        )

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
    ).max(axis=1)
    true_low = pd.concat(
        [
            low,
            previous_close,
        ],
        axis=1,
    ).min(axis=1)

    df[
        "true_range_pct_1d"
    ] = _safe_div(
        true_high - true_low,
        previous_close,
    ).astype(
        "float32"
    )
    _register(
        registry,
        "true_range_pct_1d",
        "F1_ohlc_geometry",
        "adjusted daily OHLC",
        lookback=2,
    )

    for window in (
        20,
        60,
        252,
    ):
        high_roll = _rolling(
            df,
            "adj_high",
            window,
            "max",
        )
        low_roll = _rolling(
            df,
            "adj_low",
            window,
            "min",
        )
        df[
            f"position_in_range_{window}d"
        ] = _safe_div(
            close - low_roll,
            high_roll - low_roll,
        ).astype(
            "float32"
        )
        _register(
            registry,
            f"position_in_range_{window}d",
            "F1_ohlc_geometry",
            "adjusted daily OHLC",
            lookback=window,
        )


def add_streak_features(
    df: pd.DataFrame,
    registry: list[dict],
) -> None:
    group = df.groupby(
        "canonical_security_id",
        sort=False,
        group_keys=False,
    )

    close_change = group[
        "adj_close"
    ].pct_change(
        fill_method=None
    )
    intraday = (
        df["adj_close"]
        / df["adj_open"].where(
            df["adj_open"] > 0
        )
        - 1.0
    )
    higher_high = (
        df["adj_high"]
        - group[
            "adj_high"
        ].shift(1)
    )
    higher_low = (
        df["adj_low"]
        - group[
            "adj_low"
        ].shift(1)
    )

    raw = {
        "close_direction_streak": (
            close_change
        ),
        "intraday_direction_streak": (
            intraday
        ),
        "high_direction_streak": (
            higher_high
        ),
        "low_direction_streak": (
            higher_low
        ),
    }

    for name, source in (
        raw.items()
    ):
        df[name] = (
            source.groupby(
                df[
                    "canonical_security_id"
                ],
                sort=False,
                group_keys=False,
            )
            .apply(
                _signed_streak
            )
            .reset_index(
                level=0,
                drop=True,
            )
            .reindex(
                df.index
            )
            .astype(
                "int16"
            )
        )
        _register(
            registry,
            name,
            "F2_streak_state",
            "adjusted daily OHLC",
            lookback="variable",
        )

    gap = pd.to_numeric(
        df[
            "gap_return_1d"
        ]
        if "gap_return_1d"
        in df.columns
        else (
            df["adj_open"]
            / group[
                "adj_close"
            ].shift(1)
            - 1.0
        ),
        errors="coerce",
    )
    df[
        "gap_direction_streak"
    ] = (
        gap.groupby(
            df[
                "canonical_security_id"
            ],
            sort=False,
            group_keys=False,
        )
        .apply(
            _signed_streak
        )
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(
            df.index
        )
        .astype(
            "int16"
        )
    )
    _register(
        registry,
        "gap_direction_streak",
        "F2_streak_state",
        "adjusted open and prior adjusted close",
        lookback="variable",
    )


def add_calendar_features(
    df: pd.DataFrame,
    registry: list[dict],
) -> None:
    date = df[
        "date"
    ]
    values = {
        "day_of_week": (
            date.dt.dayofweek
        ),
        "month_of_year": (
            date.dt.month
        ),
        "quarter_of_year": (
            date.dt.quarter
        ),
        "is_month_end": (
            date.dt.is_month_end.astype(
                "int8"
            )
        ),
        "is_quarter_end": (
            date.dt.is_quarter_end.astype(
                "int8"
            )
        ),
        "is_fiscal_year_end_month": (
            date.dt.month.eq(
                3
            ).astype(
                "int8"
            )
        ),
    }

    for name, series in (
        values.items()
    ):
        df[name] = series
        _register(
            registry,
            name,
            "F9_calendar",
            "calendar known before session",
            lookback=0,
            notes=(
                "Absolute year deliberately excluded "
                "to reduce regime memorization."
            )
            if name
            == "day_of_week"
            else "",
        )


def add_stock_tendencies(
    df: pd.DataFrame,
    registry: list[dict],
) -> None:
    group = df.groupby(
        "canonical_security_id",
        sort=False,
    )

    forward_1d = (
        group[
            "adj_close"
        ].shift(-1)
        / df[
            "adj_close"
        ].where(
            df[
                "adj_close"
            ] > 0
        )
        - 1.0
    )

    state = (
        pd.to_numeric(
            df[
                "close_direction_streak"
            ],
            errors="coerce",
        )
        .clip(
            -TENDENCY_STREAK_CAP,
            TENDENCY_STREAK_CAP,
        )
        .fillna(0)
        .astype(
            "int8"
        )
    )
    df[
        "_streak_state_capped"
    ] = state
    df[
        "_state_forward_1d"
    ] = forward_1d
    df[
        "_state_forward_up"
    ] = (
        forward_1d.gt(0)
        .where(
            forward_1d.notna()
        )
        .astype(float)
    )

    keys = [
        df[
            "canonical_security_id"
        ],
        df[
            "_streak_state_capped"
        ],
    ]

    for source_col, suffix in (
        (
            "_state_forward_1d",
            "mean_next_1d",
        ),
        (
            "_state_forward_up",
            "prob_up_next_1d",
        ),
    ):
        expanding = (
            df[source_col]
            .groupby(
                keys,
                sort=False,
            )
            .expanding()
            .mean()
            .reset_index(
                level=[
                    0,
                    1,
                ],
                drop=True,
            )
        )

        # Exclude the current event outcome. A previous event's t+1
        # outcome is already known by the current close, so one-event
        # shifting within each security/state is point-in-time safe.
        safe = (
            expanding.groupby(
                keys,
                sort=False,
            )
            .shift(1)
        )
        name = (
            "hist_streak_"
            + suffix
        )
        df[name] = (
            safe.reindex(
                df.index
            ).astype(
                "float32"
            )
        )
        _register(
            registry,
            name,
            "F3_stock_tendency",
            (
                "expanding prior same-stock "
                "streak-state outcomes"
            ),
            lookback="expanding",
            notes=(
                "Current state's future outcome "
                "is excluded."
            ),
        )

    counts = (
        df[
            "_state_forward_1d"
        ]
        .notna()
        .astype(
            "int32"
        )
        .groupby(
            keys,
            sort=False,
        )
        .cumsum()
        .groupby(
            keys,
            sort=False,
        )
        .shift(1)
    )
    df[
        "hist_streak_prior_count"
    ] = counts.fillna(
        0
    ).astype(
        "int32"
    )
    _register(
        registry,
        "hist_streak_prior_count",
        "F3_stock_tendency",
        "prior same-stock streak-state occurrences",
        lookback="expanding",
    )

    df.drop(
        columns=[
            "_streak_state_capped",
            "_state_forward_1d",
            "_state_forward_up",
        ],
        inplace=True,
    )


def _rolling_pair_feature(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window: int,
    *,
    prefix: str,
    registry: list[dict],
    family: str,
) -> None:
    x = pd.to_numeric(
        df[x_col],
        errors="coerce",
    )
    y = pd.to_numeric(
        df[y_col],
        errors="coerce",
    )

    temp_x = (
        f"_pair_x_{prefix}_{window}"
    )
    temp_y = (
        f"_pair_y_{prefix}_{window}"
    )
    temp_xy = (
        f"_pair_xy_{prefix}_{window}"
    )
    temp_yy = (
        f"_pair_yy_{prefix}_{window}"
    )

    df[temp_x] = x
    df[temp_y] = y
    df[temp_xy] = x * y
    df[temp_yy] = y * y

    mean_x = _rolling(
        df,
        temp_x,
        window,
        "mean",
    )
    mean_y = _rolling(
        df,
        temp_y,
        window,
        "mean",
    )
    mean_xy = _rolling(
        df,
        temp_xy,
        window,
        "mean",
    )
    mean_yy = _rolling(
        df,
        temp_yy,
        window,
        "mean",
    )
    std_x = _rolling(
        df,
        temp_x,
        window,
        "std",
    )
    std_y = _rolling(
        df,
        temp_y,
        window,
        "std",
    )

    covariance = (
        mean_xy
        - mean_x
        * mean_y
    )
    variance_y = (
        mean_yy
        - mean_y.pow(2)
    )

    corr_name = (
        f"{prefix}_corr_{window}d"
    )
    beta_name = (
        f"{prefix}_beta_{window}d"
    )

    df[corr_name] = (
        _safe_div(
            covariance,
            std_x * std_y,
        )
        .clip(
            -1.0,
            1.0,
        )
        .astype(
            "float32"
        )
    )
    df[beta_name] = (
        _safe_div(
            covariance,
            variance_y,
        )
        .astype(
            "float32"
        )
    )

    _register(
        registry,
        corr_name,
        family,
        f"rolling {x_col} vs {y_col}",
        lookback=window,
    )
    _register(
        registry,
        beta_name,
        family,
        f"rolling {x_col} vs {y_col}",
        lookback=window,
    )

    df.drop(
        columns=[
            temp_x,
            temp_y,
            temp_xy,
            temp_yy,
        ],
        inplace=True,
    )


def add_market_features(
    df: pd.DataFrame,
    market: pd.DataFrame,
    registry: list[dict],
) -> None:
    df_merge = df.merge(
        market,
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Preserve original ordering/index.
    for col in market.columns:
        if col == "date":
            continue
        df[col] = (
            df_merge[
                col
            ].to_numpy()
        )

    for prefix in (
        "nifty50",
        "nifty500",
    ):
        for lookback in (
            1,
            5,
            20,
            60,
        ):
            market_col = (
                f"{prefix}_return_"
                f"{lookback}d"
            )
            relative_name = (
                f"stock_minus_{prefix}_"
                f"{lookback}d"
            )
            stock_col = (
                f"return_{lookback}d"
            )

            if (
                stock_col
                not in df.columns
            ):
                continue

            df[
                relative_name
            ] = (
                pd.to_numeric(
                    df[
                        stock_col
                    ],
                    errors="coerce",
                )
                - pd.to_numeric(
                    df[
                        market_col
                    ],
                    errors="coerce",
                )
            ).astype(
                "float32"
            )
            _register(
                registry,
                relative_name,
                "F4_market_context",
                (
                    f"stock return minus "
                    f"{prefix} return"
                ),
                lookback=lookback,
            )

        for window in (
            60,
            252,
        ):
            _rolling_pair_feature(
                df,
                "return_1d",
                f"{prefix}_return_1d",
                window,
                prefix=(
                    f"stock_vs_{prefix}"
                ),
                registry=registry,
                family=(
                    "F4_market_context"
                ),
            )

            beta_col = (
                f"stock_vs_{prefix}_"
                f"beta_{window}d"
            )
            response_name = (
                f"{prefix}_response_gap_"
                f"{window}d"
            )
            df[
                response_name
            ] = (
                pd.to_numeric(
                    df[
                        "return_1d"
                    ],
                    errors="coerce",
                )
                - pd.to_numeric(
                    df[
                        beta_col
                    ],
                    errors="coerce",
                )
                * pd.to_numeric(
                    df[
                        f"{prefix}_return_1d"
                    ],
                    errors="coerce",
                )
            ).astype(
                "float32"
            )
            _register(
                registry,
                response_name,
                "F6_lead_lag",
                (
                    "current stock return minus "
                    "rolling-beta expected market "
                    "response"
                ),
                lookback=window,
            )

        # Market -> stock: can today's stock return be explained by
        # market returns observed k sessions earlier?
        for lag in LEAD_LAGS:
            y_col = (
                f"_{prefix}_market_lag_{lag}"
            )
            df[y_col] = (
                pd.to_numeric(
                    df[
                        f"{prefix}_return_1d"
                    ],
                    errors="coerce",
                )
                .groupby(
                    df[
                        "canonical_security_id"
                    ],
                    sort=False,
                )
                .shift(
                    lag
                )
            )

            _rolling_pair_feature(
                df,
                "return_1d",
                y_col,
                120,
                prefix=(
                    f"{prefix}_leads_stock_"
                    f"lag{lag}"
                ),
                registry=registry,
                family="F6_lead_lag",
            )
            df.drop(
                columns=[
                    y_col
                ],
                inplace=True,
            )

        # Stock -> market diagnostic relationship using past stock
        # returns against the current market return. This is still
        # knowable at close(t); it does not use market(t+1).
        for lag in (
            1,
            2,
            3,
            5,
        ):
            x_col = (
                f"_stock_lag_{prefix}_{lag}"
            )
            df[x_col] = (
                pd.to_numeric(
                    df[
                        "return_1d"
                    ],
                    errors="coerce",
                )
                .groupby(
                    df[
                        "canonical_security_id"
                    ],
                    sort=False,
                )
                .shift(
                    lag
                )
            )
            _rolling_pair_feature(
                df,
                x_col,
                f"{prefix}_return_1d",
                120,
                prefix=(
                    f"stock_leads_{prefix}_"
                    f"lag{lag}"
                ),
                registry=registry,
                family="F6_lead_lag",
            )
            df.drop(
                columns=[
                    x_col
                ],
                inplace=True,
            )


def add_cross_section_features(
    df: pd.DataFrame,
    registry: list[dict],
) -> None:
    for source in (
        "return_5d",
        "return_20d",
        "return_60d",
        "turnover",
    ):
        name = (
            f"cross_section_rank_"
            f"{source}"
        )
        df[name] = (
            df.groupby(
                "date",
                sort=False,
            )[
                source
            ]
            .rank(
                method="average",
                pct=True,
            )
            .astype(
                "float32"
            )
        )
        _register(
            registry,
            name,
            "F8_cross_section",
            source,
            lookback=0,
            cross_section=True,
            notes=(
                "Computed using only same-date "
                "close-known observations."
            ),
        )


def add_pending_registry_rows(
    registry: list[dict],
) -> None:
    pending = [
        (
            "pit_largecap_membership",
            "F7_size_largecap",
            "historical market-cap/index membership",
            "Need trustworthy point-in-time large-cap source; current membership must not be backfilled.",
        ),
        (
            "pit_market_cap_rank",
            "F7_size_largecap",
            "historical market capitalization",
            "Need point-in-time shares outstanding and price or historical official classification.",
        ),
        (
            "pit_sector_id",
            "F5_sector_industry",
            "historical sector classification",
            "Do not backfill current sector taxonomy into the past without an effective-date policy.",
        ),
        (
            "pit_industry_id",
            "F5_sector_industry",
            "historical industry classification",
            "Pending point-in-time classification data.",
        ),
        (
            "sector_relative_returns",
            "F5_sector_industry",
            "historical sector index/member returns",
            "Pending point-in-time sector mapping.",
        ),
    ]

    for (
        feature,
        family,
        source,
        notes,
    ) in pending:
        _register(
            registry,
            feature,
            family,
            source,
            lookback="pending",
            status=(
                "pending_external_pit_data"
            ),
            notes=notes,
        )


def finite_clean(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    for col in feature_cols:
        if (
            pd.api.types
            .is_numeric_dtype(
                df[col]
            )
        ):
            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
            )


def write_partitions(
    frame: pd.DataFrame,
    output_root: Path,
) -> int:
    if output_root.exists():
        shutil.rmtree(
            output_root
        )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0
    for date, group in frame.groupby(
        "date",
        sort=True,
    ):
        day = pd.Timestamp(
            date
        ).strftime(
            "%Y-%m-%d"
        )
        path = (
            output_root
            / f"date={day}"
        )
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        group.to_parquet(
            path
            / "data.parquet",
            index=False,
            compression="zstd",
        )
        count += 1

    return count


def build_v2(
    root: Path,
) -> dict:
    root = Path(
        root
    ).resolve()
    panel_root = (
        root
        / "data/processed/"
        "research_panel"
    )
    output_root = (
        root
        / "data/processed/"
        "feature_panel_v2"
    )
    reports_root = (
        root
        / "reports/features_v2"
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_panel(
        panel_root
    )
    market = load_market_context(
        root
    )
    registry: list[
        dict
    ] = []

    print(
        "Building F1 OHLC geometry..."
    )
    add_ohlc_features(
        df,
        registry,
    )

    print(
        "Building F2 streak/state..."
    )
    add_streak_features(
        df,
        registry,
    )

    print(
        "Building F3 stock-specific "
        "historical tendencies..."
    )
    add_stock_tendencies(
        df,
        registry,
    )

    print(
        "Building F4/F6 market context "
        "and lead-lag..."
    )
    add_market_features(
        df,
        market,
        registry,
    )

    print(
        "Building F8 cross-sectional "
        "state..."
    )
    add_cross_section_features(
        df,
        registry,
    )

    print(
        "Building F9 calendar..."
    )
    add_calendar_features(
        df,
        registry,
    )

    add_pending_registry_rows(
        registry
    )

    registry_frame = (
        pd.DataFrame(
            registry,
            columns=REGISTRY_COLUMNS,
        )
        .drop_duplicates(
            "feature",
            keep="last",
        )
        .sort_values(
            [
                "family",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    implemented = (
        registry_frame.loc[
            registry_frame[
                "status"
            ].eq(
                "implemented"
            ),
            "feature",
        ]
        .tolist()
    )

    finite_clean(
        df,
        implemented,
    )

    feature_frame = df[
        [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "eligible_universe",
            *implemented,
        ]
    ].copy()

    duplicate = (
        feature_frame.duplicated(
            [
                "date",
                "canonical_security_id",
            ],
            keep=False,
        )
    )
    if duplicate.any():
        raise RuntimeError(
            "Feature sidecar contains "
            "duplicate date/security keys."
        )

    print(
        "Writing v2 feature sidecar..."
    )
    partitions = (
        write_partitions(
            feature_frame,
            output_root,
        )
    )

    registry_path = (
        reports_root
        / "feature_registry.csv"
    )
    registry_frame.to_csv(
        registry_path,
        index=False,
    )

    coverage_rows = []
    for feature in implemented:
        values = (
            feature_frame[
                feature
            ]
        )
        coverage_rows.append({
            "feature": feature,
            "non_null_fraction": float(
                values.notna().mean()
            ),
            "eligible_non_null_fraction": float(
                values.loc[
                    feature_frame[
                        "eligible_universe"
                    ].fillna(
                        False
                    )
                ].notna().mean()
            ),
        })

    coverage = pd.DataFrame(
        coverage_rows
    )
    coverage.to_csv(
        reports_root
        / "feature_coverage.csv",
        index=False,
    )

    summary = {
        "rows": int(
            len(
                feature_frame
            )
        ),
        "dates": int(
            feature_frame[
                "date"
            ].nunique()
        ),
        "securities": int(
            feature_frame[
                "canonical_security_id"
            ].nunique()
        ),
        "implemented_features": int(
            len(
                implemented
            )
        ),
        "feature_families": sorted(
            registry_frame.loc[
                registry_frame[
                    "status"
                ].eq(
                    "implemented"
                ),
                "family",
            ].unique().tolist()
        ),
        "pending_point_in_time_families": (
            registry_frame.loc[
                registry_frame[
                    "status"
                ].eq(
                    "pending_external_pit_data"
                ),
                [
                    "feature",
                    "family",
                    "notes",
                ],
            ].to_dict(
                orient="records"
            )
        ),
        "decision_timestamp": (
            "after_close_t"
        ),
        "execution_timestamp": (
            "next_session_open"
        ),
        "absolute_year_feature": False,
        "output_root": str(
            output_root
        ),
    }

    (
        reports_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    dates = pd.date_range(
        "2025-01-01",
        periods=12,
        freq="D",
    )
    close = np.array([
        100,
        99,
        98,
        97,
        99,
        101,
        100,
        102,
        101,
        100,
        103,
        104,
    ], dtype=float)

    frame = pd.DataFrame({
        "date": dates,
        "market_day_index": np.arange(
            len(dates)
        ),
        "canonical_security_id": [
            "TEST"
        ] * len(
            dates
        ),
        "symbol": [
            "TEST"
        ] * len(
            dates
        ),
        "eligible_universe": [
            True
        ] * len(
            dates
        ),
        "adj_open": close - 0.5,
        "adj_high": close + 1.0,
        "adj_low": close - 1.0,
        "adj_close": close,
        "adj_volume": [
            1000
        ] * len(
            dates
        ),
        "turnover": [
            1_000_000
        ] * len(
            dates
        ),
        "return_1d": pd.Series(
            close
        ).pct_change(),
        "return_5d": pd.Series(
            close
        ).pct_change(
            5
        ),
        "return_20d": np.nan,
        "return_60d": np.nan,
    })

    registry: list[
        dict
    ] = []
    add_ohlc_features(
        frame,
        registry,
    )
    add_streak_features(
        frame,
        registry,
    )
    add_stock_tendencies(
        frame,
        registry,
    )
    add_calendar_features(
        frame,
        registry,
    )

    assert int(
        frame.loc[
            3,
            "close_direction_streak",
        ]
    ) == -3
    assert int(
        frame.loc[
            5,
            "close_direction_streak",
        ]
    ) == 2

    # Mutating the future must not alter a feature in the past.
    baseline = float(
        frame.loc[
            5,
            "body_pct_1d",
        ]
    )

    mutated = frame.copy()
    mutated.loc[
        11,
        "adj_close",
    ] = 99999.0

    registry_2: list[
        dict
    ] = []
    add_ohlc_features(
        mutated,
        registry_2,
    )

    assert math.isclose(
        baseline,
        float(
            mutated.loc[
                5,
                "body_pct_1d",
            ]
        ),
        rel_tol=1e-12,
    )

    print(
        "Feature-engineering-v2 "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build an auditable point-in-time "
            "feature sidecar for the frozen "
            "Indian equity research panel."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    summary = build_v2(
        Path(
            args.root
        )
    )

    print(
        "\n=== FEATURE ENGINEERING V2 COMPLETE ==="
    )
    print(
        f"Rows:                 "
        f"{summary['rows']:,}"
    )
    print(
        f"Dates:                "
        f"{summary['dates']:,}"
    )
    print(
        f"Securities:           "
        f"{summary['securities']:,}"
    )
    print(
        f"Implemented features: "
        f"{summary['implemented_features']:,}"
    )
    print(
        "Families:             "
        + ", ".join(
            summary[
                "feature_families"
            ]
        )
    )
    print(
        f"Output:               "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
