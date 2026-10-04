from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
from nested_generalization import (  # noqa: E402
    fixed_schedule,
    run_persistent_strategy,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
)


BASE_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "turnover_median_20d",
    "unsafe_target_window_20d",
    "target_next_open_to_close_20d",
    "largecap_flag_numeric",
    "sector",
    "industry",
    "sector_index_name",
    "market_context_index",
    "calendar_year",
    "day_of_week",
    "month_of_year",
    "quarter_of_year",
    "close_location_1d",
    "up_close_streak",
    "down_close_streak",
    "stock_minus_market_20d",
    "market_response_gap_1d",
    "market_beta_60d",
]


def load_panel(
    panel_root: Path,
) -> pd.DataFrame:
    files = sorted(
        Path(
            panel_root
        ).glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No panel under {panel_root}"
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
        "open",
        "close",
        "turnover_median_20d",
        "unsafe_target_window_20d",
    }
    missing = sorted(
        required - available
    )
    if missing:
        raise RuntimeError(
            f"Panel missing {missing}"
        )

    columns = [
        column
        for column in BASE_COLUMNS
        if column in available
    ]

    frames: list[
        pd.DataFrame
    ] = []

    print(
        f"Loading {len(files):,} attribution "
        "panel partitions..."
    )

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
    df[
        "eligible_universe"
    ] = (
        df[
            "eligible_universe"
        ]
        .fillna(False)
        .astype(bool)
    )
    df[
        "unsafe_target_window_20d"
    ] = (
        df[
            "unsafe_target_window_20d"
        ]
        .fillna(False)
        .astype(bool)
    )

    return (
        df.sort_values(
            [
                "market_day_index",
                "canonical_security_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def restrict_universe(
    df: pd.DataFrame,
    *,
    universe: str,
) -> pd.Series:
    base = df[
        "eligible_universe"
    ]

    if universe == "existing":
        return base

    if (
        "largecap_flag_numeric"
        not in df.columns
    ):
        raise RuntimeError(
            "Large-cap attribution requires "
            "point-in-time context."
        )

    flag = pd.to_numeric(
        df[
            "largecap_flag_numeric"
        ],
        errors="coerce",
    )
    if flag.notna().mean() < 0.01:
        raise RuntimeError(
            "Large-cap context is missing."
        )

    return (
        base
        & flag.eq(
            1.0
        )
    )


def attach_predictions(
    df: pd.DataFrame,
    *,
    prediction_dir: Path,
    start_year: int,
    end_year: int,
) -> None:
    df[
        "persistent_score"
    ] = np.nan

    keys = [
        "date",
        "canonical_security_id",
    ]

    for year in range(
        start_year,
        end_year + 1,
    ):
        path = (
            prediction_dir
            / f"{year}.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing prediction cache: {path}"
            )

        saved = pd.read_parquet(
            path
        )
        saved["date"] = pd.to_datetime(
            saved["date"],
            errors="coerce",
        ).dt.normalize()
        saved[
            "canonical_security_id"
        ] = (
            saved[
                "canonical_security_id"
            ]
            .astype("string")
            .str.strip()
        )

        mapping = saved.set_index(
            keys
        )[
            "persistent_score"
        ]

        mask = df[
            "date"
        ].dt.year.eq(
            year
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

        print(
            f"{year}: attached "
            f"{int(np.isfinite(values).sum()):,} "
            "scores"
        )


def summarize_trades(
    trades: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if (
        trades.empty
        or column not in trades.columns
    ):
        return pd.DataFrame()

    frame = trades.loc[
        trades[
            column
        ].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    return (
        frame.groupby(
            column,
            dropna=False,
        )
        .agg(
            trades=(
                "canonical_security_id",
                "size",
            ),
            net_pnl=(
                "net_pnl",
                "sum",
            ),
            mean_net_return=(
                "net_return",
                "mean",
            ),
            median_net_return=(
                "net_return",
                "median",
            ),
            hit_rate=(
                "net_pnl",
                lambda x: float(
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    ).gt(0).mean()
                ),
            ),
            total_fees=(
                "total_fees",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "net_pnl",
            ascending=False,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create signal, basket and actual "
            "trade attribution ledgers."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--panel",
        default=(
            "data/processed/"
            "research_panel_enriched"
        ),
    )
    ap.add_argument(
        "--predictions",
        required=True,
        help=(
            "Directory containing "
            "YYYY.parquet prediction caches."
        ),
    )
    ap.add_argument(
        "--name",
        default="strategy",
    )
    ap.add_argument(
        "--universe",
        choices=[
            "largecap",
            "existing",
        ],
        default="largecap",
    )
    ap.add_argument(
        "--start-year",
        type=int,
        default=2019,
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=2026,
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
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
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    panel = load_panel(
        root
        / args.panel
    )

    universe_mask = (
        restrict_universe(
            panel,
            universe=(
                args.universe
            ),
        )
    )
    panel[
        "eligible_universe"
    ] = universe_mask

    attach_predictions(
        panel,
        prediction_dir=(
            root
            / args.predictions
        ),
        start_year=int(
            args.start_year
        ),
        end_year=int(
            args.end_year
        ),
    )
    add_daily_ranks(
        panel
    )

    period = panel[
        "date"
    ].between(
        pd.Timestamp(
            f"{args.start_year}-01-01"
        ),
        pd.Timestamp(
            f"{args.end_year}-12-31"
        ),
        inclusive="both",
    )
    panel = (
        panel.loc[
            period
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    signal_ledger = (
        panel.loc[
            panel[
                "eligible_universe"
            ]
            & pd.to_numeric(
                panel[
                    "persistent_rank"
                ],
                errors="coerce",
            ).le(
                int(
                    args.top_k
                )
            )
        ]
        .copy()
    )

    signal_ledger[
        "signal_year"
    ] = signal_ledger[
        "date"
    ].dt.year
    signal_ledger[
        "signal_month"
    ] = signal_ledger[
        "date"
    ].dt.month
    signal_ledger[
        "signal_weekday"
    ] = signal_ledger[
        "date"
    ].dt.day_name()

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
    events = (
        load_mechanical_events(
            root
        )
    )
    years = list(
        range(
            int(
                args.start_year
            ),
            int(
                args.end_year
            )
            + 1,
        )
    )
    schedule = fixed_schedule(
        years=years,
        top_k=int(
            args.top_k
        ),
        multiplier=1,
        gap=0.0,
    )

    result = run_persistent_strategy(
        panel,
        events,
        name=args.name,
        capital=float(
            args.capital
        ),
        top_k=int(
            args.top_k
        ),
        schedule=schedule,
        costs=costs,
        annual_risk_free_rate=float(
            args.risk_free_rate
        ),
    )

    trades = result[
        "trades"
    ].copy()

    context_columns = [
        column
        for column in (
            "date",
            "canonical_security_id",
            "symbol",
            "sector",
            "industry",
            "sector_index_name",
            "largecap_flag_numeric",
            "calendar_year",
            "day_of_week",
            "month_of_year",
            "quarter_of_year",
            "open",
            "high",
            "low",
            "close",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "close_location_1d",
            "up_close_streak",
            "down_close_streak",
            "stock_minus_market_20d",
            "market_response_gap_1d",
            "market_beta_60d",
            "persistent_score",
            "persistent_rank",
        )
        if column in panel.columns
    ]

    signal_context = (
        panel[
            context_columns
        ]
        .rename(
            columns={
                "date": "signal_date",
                **{
                    column: (
                        "signal_"
                        + column
                    )
                    for column
                    in context_columns
                    if column
                    not in {
                        "date",
                        "canonical_security_id",
                    }
                },
            }
        )
    )

    trades = trades.merge(
        signal_context,
        on=[
            "signal_date",
            "canonical_security_id",
        ],
        how="left",
        validate="many_to_one",
    )
    trades[
        "entry_year"
    ] = pd.to_datetime(
        trades[
            "entry_date"
        ],
        errors="coerce",
    ).dt.year
    trades[
        "entry_month"
    ] = pd.to_datetime(
        trades[
            "entry_date"
        ],
        errors="coerce",
    ).dt.month
    trades[
        "entry_weekday"
    ] = pd.to_datetime(
        trades[
            "entry_date"
        ],
        errors="coerce",
    ).dt.day_name()

    output = (
        root
        / "reports/ml/"
        "strategy_attribution"
        / args.name
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal_ledger.to_parquet(
        output
        / "topk_signal_ledger.parquet",
        index=False,
        compression="zstd",
    )
    trades.to_parquet(
        output
        / "trade_ledger.parquet",
        index=False,
        compression="zstd",
    )
    result[
        "rebalances"
    ].to_parquet(
        output
        / "rebalance_ledger.parquet",
        index=False,
        compression="zstd",
    )
    result[
        "equity"
    ].to_parquet(
        output
        / "equity_curve.parquet",
        index=False,
        compression="zstd",
    )

    summary_dims = [
        "signal_symbol",
        "signal_sector",
        "signal_industry",
        "entry_year",
        "entry_month",
        "entry_weekday",
    ]

    for dimension in summary_dims:
        summary = summarize_trades(
            trades,
            dimension,
        )
        if not summary.empty:
            summary.to_csv(
                output
                / (
                    "pnl_by_"
                    f"{dimension}.csv"
                ),
                index=False,
            )

    (
        output
        / "metrics.json"
    ).write_text(
        json.dumps(
            {
                key: (
                    float(value)
                    if isinstance(
                        value,
                        (
                            np.floating,
                            float,
                        ),
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (
                            np.integer,
                            int,
                        ),
                    )
                    else value
                )
                for key, value
                in result[
                    "metrics"
                ].items()
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(
        "\n=== STRATEGY ATTRIBUTION ==="
    )
    print(
        json.dumps(
            result[
                "metrics"
            ],
            indent=2,
            default=str,
        )
    )
    print(
        f"\nSignal rows: "
        f"{len(signal_ledger):,}"
    )
    print(
        f"Trades: {len(trades):,}"
    )
    print(
        f"Outputs: {output}"
    )


if __name__ == "__main__":
    main()
