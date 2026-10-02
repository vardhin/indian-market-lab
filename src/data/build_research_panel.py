from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


HORIZONS = (1, 3, 5, 10, 20)
RETURN_LOOKBACKS = (1, 3, 5, 10, 20, 60)
ROLLING_WINDOWS = (5, 20, 60)

REQUIRED_COLUMNS = {
    "date",
    "canonical_security_id",
    "symbol",
    "series",
    "close",
    "volume",
    "turnover",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
}

PREFERRED_COLUMNS = [
    "date",
    "canonical_security_id",
    "symbol",
    "series",
    "isin",
    "isin_resolved",
    "resolution_method",
    "resolution_confidence",
    "open",
    "high",
    "low",
    "close",
    "last",
    "prev_close",
    "volume",
    "turnover",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_last",
    "adj_volume",
    "price_adjustment_factor",
    "volume_adjustment_factor",
    "mechanically_adjusted",
]


def _rolling(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    stat: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window

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
    elif stat == "median":
        out = grouped.median()
    elif stat == "std":
        out = grouped.std()
    elif stat == "max":
        out = grouped.max()
    elif stat == "min":
        out = grouped.min()
    elif stat == "sum":
        out = grouped.sum()
    else:
        raise ValueError(
            f"Unsupported rolling stat: {stat}"
        )

    return out.reset_index(
        level=0,
        drop=True,
    )


def discover_columns(
    files: list[Path],
) -> list[str]:
    if not files:
        raise FileNotFoundError(
            "No adjusted partitions found."
        )

    schema = pq.read_schema(files[0])
    available = set(schema.names)

    missing = sorted(
        REQUIRED_COLUMNS - available
    )

    if missing:
        raise RuntimeError(
            f"{files[0]} is missing required columns: "
            f"{missing}"
        )

    return [
        c
        for c in PREFERRED_COLUMNS
        if c in available
    ]


def load_adjusted_panel(
    adjusted_root: Path,
) -> pd.DataFrame:
    adjusted_root = Path(adjusted_root)

    files = sorted(
        adjusted_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No adjusted daily partitions under "
            f"{adjusted_root}"
        )

    columns = discover_columns(files)

    print(
        f"Loading {len(files):,} adjusted "
        "daily partitions..."
    )

    frames: list[pd.DataFrame] = []

    for index, path in enumerate(
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
            index % 500 == 0
            or index == len(files)
        ):
            print(
                f"  loaded {index:,}/"
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

    for col in (
        "canonical_security_id",
        "symbol",
        "series",
    ):
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "last",
        "prev_close",
        "volume",
        "turnover",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_last",
        "adj_volume",
        "price_adjustment_factor",
        "volume_adjustment_factor",
        "resolution_confidence",
    ]

    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    bad_keys = (
        df[
            [
                "date",
                "canonical_security_id",
                "symbol",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    if bad_keys.any():
        raise RuntimeError(
            "Adjusted panel contains "
            f"{int(bad_keys.sum()):,} rows "
            "with unusable keys."
        )

    duplicates = df.duplicated(
        [
            "date",
            "canonical_security_id",
        ],
        keep=False,
    )

    if duplicates.any():
        sample = df.loc[
            duplicates,
            [
                "date",
                "symbol",
                "canonical_security_id",
            ],
        ].head(30)

        raise RuntimeError(
            "Adjusted panel has duplicate "
            "date/security rows. Sample:\n"
            + sample.to_string(
                index=False
            )
        )

    return (
        df.sort_values(
            [
                "canonical_security_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def add_point_in_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features using only information available through each row's close.

    These features are intended for a signal generated after market close on
    date t and a possible trade at the next trading session's open.
    """
    df = df.copy()

    group = df.groupby(
        "canonical_security_id",
        sort=False,
    )

    df["history_observations"] = (
        group.cumcount() + 1
    )

    previous_adj_close = group[
        "adj_close"
    ].shift(1)

    df["gap_return_1d"] = (
        df["adj_open"]
        / previous_adj_close.where(
            previous_adj_close > 0
        )
        - 1.0
    )

    df["intraday_return_1d"] = (
        df["adj_close"]
        / df["adj_open"].where(
            df["adj_open"] > 0
        )
        - 1.0
    )

    df["range_pct_1d"] = (
        (
            df["adj_high"]
            - df["adj_low"]
        )
        / df["adj_open"].where(
            df["adj_open"] > 0
        )
    )

    for lookback in RETURN_LOOKBACKS:
        lagged = group[
            "adj_close"
        ].shift(lookback)

        df[f"return_{lookback}d"] = (
            df["adj_close"]
            / lagged.where(
                lagged > 0
            )
            - 1.0
        )

    for window in ROLLING_WINDOWS:
        min_periods = max(
            2,
            window // 2,
        )

        rolling_mean = _rolling(
            df,
            "adj_close",
            window,
            "mean",
            min_periods=min_periods,
        )

        df[f"close_to_ma_{window}d"] = (
            df["adj_close"]
            / rolling_mean.where(
                rolling_mean > 0
            )
            - 1.0
        )

        rolling_high = _rolling(
            df,
            "adj_high",
            window,
            "max",
            min_periods=min_periods,
        )
        rolling_low = _rolling(
            df,
            "adj_low",
            window,
            "min",
            min_periods=min_periods,
        )

        df[f"distance_high_{window}d"] = (
            df["adj_close"]
            / rolling_high.where(
                rolling_high > 0
            )
            - 1.0
        )

        df[f"distance_low_{window}d"] = (
            df["adj_close"]
            / rolling_low.where(
                rolling_low > 0
            )
            - 1.0
        )

    for window in (
        5,
        10,
        20,
        60,
    ):
        df[f"volatility_{window}d"] = (
            _rolling(
                df,
                "return_1d",
                window,
                "std",
                min_periods=max(
                    3,
                    window // 2,
                ),
            )
        )

    df["turnover_median_20d"] = (
        _rolling(
            df,
            "turnover",
            20,
            "median",
            min_periods=10,
        )
    )
    df["turnover_median_60d"] = (
        _rolling(
            df,
            "turnover",
            60,
            "median",
            min_periods=30,
        )
    )

    turnover_mean_20 = _rolling(
        df,
        "turnover",
        20,
        "mean",
        min_periods=10,
    )
    turnover_std_20 = _rolling(
        df,
        "turnover",
        20,
        "std",
        min_periods=10,
    )

    df["turnover_zscore_20d"] = (
        (
            df["turnover"]
            - turnover_mean_20
        )
        / turnover_std_20.where(
            turnover_std_20 > 0
        )
    )

    volume_mean_20 = _rolling(
        df,
        "adj_volume",
        20,
        "mean",
        min_periods=10,
    )
    volume_std_20 = _rolling(
        df,
        "adj_volume",
        20,
        "std",
        min_periods=10,
    )

    df["volume_zscore_20d"] = (
        (
            df["adj_volume"]
            - volume_mean_20
        )
        / volume_std_20.where(
            volume_std_20 > 0
        )
    )

    df["_active_day"] = (
        pd.to_numeric(
            df["volume"],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype(float)
    )

    df["active_day_ratio_20d"] = (
        _rolling(
            df,
            "_active_day",
            20,
            "mean",
            min_periods=10,
        )
    )

    df = df.drop(
        columns=["_active_day"]
    )

    return df


def add_forward_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe labels.

    Signal timestamp:
        after close on trading day t

    Realistically tradable target:
        enter at adjusted open on t+1
        exit at adjusted close on t+h

    A close-to-close target is retained separately for forecasting diagnostics.
    Neither target is used in universe construction or feature calculation.
    """
    df = df.copy()

    group = df.groupby(
        "canonical_security_id",
        sort=False,
    )

    next_open = group[
        "adj_open"
    ].shift(-1)

    for horizon in HORIZONS:
        future_close = group[
            "adj_close"
        ].shift(-horizon)

        df[
            f"target_next_open_to_close_{horizon}d"
        ] = (
            future_close
            / next_open.where(
                next_open > 0
            )
            - 1.0
        )

        df[
            f"target_close_to_close_{horizon}d"
        ] = (
            future_close
            / df["adj_close"].where(
                df["adj_close"] > 0
            )
            - 1.0
        )

    return df


def add_universe_flags(
    df: pd.DataFrame,
    *,
    min_history: int,
    min_price: float,
    min_median_turnover: float,
    min_active_ratio: float,
) -> pd.DataFrame:
    """
    Define a configurable point-in-time tradable universe.

    Raw/as-traded close and turnover are used for execution-oriented filters;
    adjusted prices remain dedicated to return/features continuity.
    """
    df = df.copy()

    df["feature_history_ready"] = (
        df["history_observations"]
        .ge(min_history)
    )

    df["price_eligible"] = (
        df["close"]
        .ge(min_price)
    )

    df["liquidity_eligible"] = (
        df["turnover_median_20d"]
        .ge(min_median_turnover)
    )

    df["activity_eligible"] = (
        df["active_day_ratio_20d"]
        .ge(min_active_ratio)
    )

    df["has_next_session_open"] = (
        df.groupby(
            "canonical_security_id",
            sort=False,
        )["adj_open"]
        .shift(-1)
        .notna()
    )

    df["eligible_universe"] = (
        df["feature_history_ready"]
        & df["price_eligible"]
        & df["liquidity_eligible"]
        & df["activity_eligible"]
        & df["has_next_session_open"]
    )

    return df


def validate_panel(
    df: pd.DataFrame,
) -> dict:
    feature_cols = [
        c
        for c in df.columns
        if (
            c.startswith("return_")
            or c.startswith(
                "volatility_"
            )
            or c.startswith(
                "close_to_ma_"
            )
            or c.startswith(
                "distance_"
            )
            or c.endswith(
                "_zscore_20d"
            )
        )
    ]

    target_cols = [
        c
        for c in df.columns
        if c.startswith("target_")
    ]

    nonfinite_features = 0

    for col in feature_cols:
        s = pd.to_numeric(
            df[col],
            errors="coerce",
        )
        nonfinite_features += int(
            s.map(
                lambda x: (
                    pd.notna(x)
                    and not math.isfinite(
                        float(x)
                    )
                )
            ).sum()
        )

    return {
        "rows": int(len(df)),
        "dates": int(
            df["date"].nunique()
        ),
        "securities": int(
            df[
                "canonical_security_id"
            ].nunique()
        ),
        "eligible_rows": int(
            df["eligible_universe"].sum()
        ),
        "eligible_dates": int(
            df.loc[
                df["eligible_universe"],
                "date",
            ].nunique()
        ),
        "eligible_securities": int(
            df.loc[
                df["eligible_universe"],
                "canonical_security_id",
            ].nunique()
        ),
        "nonfinite_feature_values": int(
            nonfinite_features
        ),
        "target_nonnull_counts": {
            c: int(
                df[c].notna().sum()
            )
            for c in target_cols
        },
    }


def write_partitions(
    df: pd.DataFrame,
    output_root: Path,
) -> int:
    output_root = Path(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in output_root.glob(
        "date=*"
    ):
        if child.is_dir():
            shutil.rmtree(child)

    count = 0

    for day, g in df.groupby(
        "date",
        sort=True,
    ):
        out_dir = (
            output_root
            / f"date={pd.Timestamp(day).date()}"
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        g.sort_values(
            [
                "eligible_universe",
                "turnover_median_20d",
                "canonical_security_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).to_parquet(
            out_dir / "data.parquet",
            index=False,
            compression="zstd",
        )

        count += 1

        if (
            count % 250 == 0
            or count == df["date"].nunique()
        ):
            print(
                f"  wrote {count:,}/"
                f"{df['date'].nunique():,} "
                "research partitions",
                flush=True,
            )

    return count


def build_research_panel(
    root: Path,
    *,
    min_history: int = 120,
    min_price: float = 10.0,
    min_median_turnover: float = 5_000_000.0,
    min_active_ratio: float = 0.90,
) -> dict:
    root = Path(root).resolve()

    adjusted_root = (
        root
        / "data/processed/equities_adjusted"
    )
    output_root = (
        root
        / "data/processed/research_panel"
    )
    reports_root = (
        root
        / "reports"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel = load_adjusted_panel(
        adjusted_root
    )

    print(
        f"Loaded rows={len(panel):,}, "
        f"dates={panel['date'].nunique():,}, "
        "securities="
        f"{panel['canonical_security_id'].nunique():,}"
    )

    print(
        "Building point-in-time features..."
    )
    panel = add_point_in_time_features(
        panel
    )

    print(
        "Building forward targets..."
    )
    panel = add_forward_targets(
        panel
    )

    print(
        "Applying configurable point-in-time "
        "universe rules..."
    )
    panel = add_universe_flags(
        panel,
        min_history=min_history,
        min_price=min_price,
        min_median_turnover=(
            min_median_turnover
        ),
        min_active_ratio=min_active_ratio,
    )

    report = validate_panel(panel)

    if report[
        "nonfinite_feature_values"
    ]:
        raise RuntimeError(
            "Feature validation failed: "
            f"{report['nonfinite_feature_values']:,} "
            "non-finite values."
        )

    print(
        "Writing partitioned research panel..."
    )
    partitions = write_partitions(
        panel,
        output_root,
    )

    eligible = panel.loc[
        panel["eligible_universe"]
    ]

    year_summary = (
        eligible.assign(
            year=eligible[
                "date"
            ].dt.year
        )
        .groupby(
            "year",
            sort=True,
        )
        .agg(
            eligible_rows=(
                "canonical_security_id",
                "size",
            ),
            eligible_securities=(
                "canonical_security_id",
                "nunique",
            ),
            eligible_dates=(
                "date",
                "nunique",
            ),
            median_daily_turnover=(
                "turnover",
                "median",
            ),
        )
        .reset_index()
    )

    year_summary_path = (
        reports_root
        / "research_universe_by_year.csv"
    )
    year_summary.to_csv(
        year_summary_path,
        index=False,
    )

    summary = {
        **report,
        "partitions": int(partitions),
        "feature_timestamp": (
            "after_close_t"
        ),
        "trade_entry_convention": (
            "next_trading_session_open"
        ),
        "universe_parameters": {
            "min_history_observations": int(
                min_history
            ),
            "min_raw_close": float(
                min_price
            ),
            "min_trailing_20d_median_turnover": float(
                min_median_turnover
            ),
            "min_trailing_20d_active_day_ratio": float(
                min_active_ratio
            ),
        },
        "horizons": list(
            HORIZONS
        ),
        "output_root": str(
            output_root
        ),
        "universe_by_year": str(
            year_summary_path
        ),
    }

    summary_path = (
        reports_root
        / "research_panel_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    return {
        "summary": summary,
        "summary_path": (
            summary_path
        ),
    }


def self_test() -> None:
    dates = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="D",
    )

    df = pd.DataFrame({
        "date": dates,
        "canonical_security_id": (
            ["ISIN:TEST"] * 8
        ),
        "symbol": ["TEST"] * 8,
        "series": ["EQ"] * 8,
        "open": [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
        ],
        "high": [
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ],
        "low": [
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
        ],
        "close": [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
        ],
        "volume": [1000] * 8,
        "turnover": [1_000_000] * 8,
        "adj_open": [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
        ],
        "adj_high": [
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
        ],
        "adj_low": [
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
        ],
        "adj_close": [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
        ],
        "adj_volume": [1000] * 8,
    })

    out = add_point_in_time_features(
        df
    )
    out = add_forward_targets(out)

    # On t, the 3-day realistic target is:
    # buy at open(t+1), sell at close(t+3).
    expected = (
        out.loc[3, "adj_close"]
        / out.loc[1, "adj_open"]
        - 1.0
    )

    actual = out.loc[
        0,
        "target_next_open_to_close_3d",
    ]

    assert math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
    )

    # Future information must not affect a past feature.
    baseline_return = float(
        out.loc[
            4,
            "return_3d",
        ]
    )

    mutated = df.copy()
    mutated.loc[
        7,
        "adj_close",
    ] = 999999.0

    mutated_out = (
        add_point_in_time_features(
            mutated
        )
    )

    assert math.isclose(
        baseline_return,
        float(
            mutated_out.loc[
                4,
                "return_3d",
            ]
        ),
        rel_tol=1e-12,
    )

    print(
        "Research-panel leakage self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a point-in-time tradable universe, leakage-safe features "
            "and next-session-open forward targets from adjusted NSE equities."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--min-history",
        type=int,
        default=120,
        help=(
            "Minimum prior trading observations "
            "(default: 120)"
        ),
    )
    ap.add_argument(
        "--min-price",
        type=float,
        default=10.0,
        help=(
            "Minimum raw/as-traded close price "
            "(default: 10)"
        ),
    )
    ap.add_argument(
        "--min-median-turnover",
        type=float,
        default=5_000_000.0,
        help=(
            "Minimum trailing 20-day median daily turnover in INR "
            "(default: 5,000,000)"
        ),
    )
    ap.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.90,
        help=(
            "Minimum positive-volume fraction over trailing 20 observations "
            "(default: 0.90)"
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.min_history < 2:
        raise SystemExit(
            "--min-history must be >= 2"
        )
    if args.min_price <= 0:
        raise SystemExit(
            "--min-price must be > 0"
        )
    if args.min_median_turnover < 0:
        raise SystemExit(
            "--min-median-turnover must be >= 0"
        )
    if not (
        0
        <= args.min_active_ratio
        <= 1
    ):
        raise SystemExit(
            "--min-active-ratio must be in [0, 1]"
        )

    result = build_research_panel(
        Path(args.root),
        min_history=args.min_history,
        min_price=args.min_price,
        min_median_turnover=(
            args.min_median_turnover
        ),
        min_active_ratio=(
            args.min_active_ratio
        ),
    )

    summary = result["summary"]

    print(
        "\n=== RESEARCH PANEL BUILD COMPLETE ==="
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
        f"Eligible rows:        "
        f"{summary['eligible_rows']:,}"
    )
    print(
        f"Eligible dates:       "
        f"{summary['eligible_dates']:,}"
    )
    print(
        f"Eligible securities:  "
        f"{summary['eligible_securities']:,}"
    )
    print(
        f"Output:               "
        f"{summary['output_root']}"
    )
    print(
        f"Universe by year:     "
        f"{summary['universe_by_year']}"
    )
    print(
        f"Summary:              "
        f"{result['summary_path']}"
    )


if __name__ == "__main__":
    main()
