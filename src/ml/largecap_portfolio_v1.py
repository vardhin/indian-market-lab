from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(
    __file__
).resolve()
ML_DIR = SCRIPT_PATH.parent
BACKTEST_DIR = (
    SCRIPT_PATH.parents[
        1
    ]
    / "backtest"
)

for path in (
    ML_DIR,
    BACKTEST_DIR,
):
    if str(
        path
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                path
            ),
        )

from baselines import (  # noqa: E402
    CostProfile,
    load_mechanical_events,
)
from nested_generalization import (  # noqa: E402
    annual_metrics_from_result,
    benchmark_rows,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
    run_persistent_backtest,
)
from quant_metrics import (  # noqa: E402
    benchmark_relative_metrics,
)


FEATURE_SET = (
    "B4_core_plus_F8"
)

DEVELOPMENT_YEARS = [
    2021,
    2022,
    2023,
]

CONFIRMATION_YEARS = [
    2024,
    2025,
    2026,
]

PANEL_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "eligible_universe",
    "open",
    "close",
    "turnover_median_20d",
    "unsafe_target_window_20d",
]

POLICIES = [
    {
        "name": (
            f"k{k}_r{rebalance}_"
            f"h{int(k * multiplier)}"
        ),
        "top_k": int(
            k
        ),
        "rebalance_sessions": int(
            rebalance
        ),
        "hold_rank": int(
            k
            * multiplier
        ),
        "hold_multiplier": int(
            multiplier
        ),
        "min_score_gap": 0.0,
    }
    for k in (
        5,
        10,
        15,
    )
    for rebalance in (
        10,
        20,
    )
    for multiplier in (
        1,
        2,
    )
]


def load_panel(
    root: Path,
    *,
    years: list[int],
) -> pd.DataFrame:
    panel_root = (
        root
        / "data/processed/"
        "largecap_model_panel_v2"
    )

    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No large-cap model-panel "
            f"partitions under {panel_root}"
        )

    schema = pq.read_schema(
        files[
            0
        ]
    )
    missing = [
        column
        for column in PANEL_COLUMNS
        if column not in set(
            schema.names
        )
    ]

    if missing:
        raise RuntimeError(
            "Large-cap model panel is "
            "missing portfolio columns: "
            f"{missing}"
        )

    selected = []

    print(
        "Loading large-cap execution panel "
        f"for {min(years)}-{max(years)}..."
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        name = (
            path.parent.name
        )

        if not name.startswith(
            "date="
        ):
            continue

        date = pd.Timestamp(
            name.split(
                "=",
                1,
            )[
                1
            ]
        ).normalize()

        if int(
            date.year
        ) not in years:
            continue

        selected.append(
            pd.read_parquet(
                path,
                columns=(
                    PANEL_COLUMNS
                ),
            )
        )

        if (
            len(
                selected
            )
            % 250
            == 0
        ):
            print(
                "  loaded "
                f"{len(selected):,} "
                "stage dates",
                flush=True,
            )

    if not selected:
        raise RuntimeError(
            "No model-panel rows for "
            f"years {years}"
        )

    df = pd.concat(
        selected,
        ignore_index=True,
    )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    df[
        "market_day_index"
    ] = pd.to_numeric(
        df[
            "market_day_index"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    df[
        "canonical_security_id"
    ] = (
        df[
            "canonical_security_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    for column in (
        "eligible_universe",
        "unsafe_target_window_20d",
    ):
        df[
            column
        ] = (
            df[
                column
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    for column in (
        "open",
        "close",
        "turnover_median_20d",
    ):
        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    duplicate = (
        df.duplicated(
            [
                "market_day_index",
                "canonical_security_id",
            ],
            keep=False,
        )
    )

    if duplicate.any():
        raise RuntimeError(
            "Execution panel contains "
            "duplicate date/security rows."
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


def attach_scores(
    df: pd.DataFrame,
    *,
    root: Path,
    stage: str,
    years: list[int],
) -> dict:
    prediction_stage = (
        "branch"
        if stage
        == "development"
        else "confirmation"
    )

    base = (
        root
        / "reports/ml/"
        "largecap_feature_ablation"
        / prediction_stage
        / "predictions"
        / FEATURE_SET
    )

    df[
        "persistent_score"
    ] = np.nan

    keys = [
        "date",
        "canonical_security_id",
    ]

    expected_rows = 0
    attached_rows = 0

    for year in years:
        path = (
            base
            / f"{year}.parquet"
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Missing frozen B4 prediction "
                f"cache from {prediction_stage} "
                f"stage: {path}"
            )

        saved = pd.read_parquet(
            path,
            columns=[
                "date",
                "canonical_security_id",
                "score",
            ],
        )

        saved[
            "date"
        ] = pd.to_datetime(
            saved[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()

        saved[
            "canonical_security_id"
        ] = (
            saved[
                "canonical_security_id"
            ]
            .astype(
                "string"
            )
            .str.strip()
        )

        if saved.duplicated(
            keys,
            keep=False,
        ).any():
            raise RuntimeError(
                "Duplicate B4 prediction "
                f"keys in {path}"
            )

        mask = (
            df[
                "date"
            ].dt.year.eq(
                year
            )
            & df[
                "eligible_universe"
            ]
        )

        expected_rows += int(
            mask.sum()
        )

        mapping = (
            saved.set_index(
                keys
            )[
                "score"
            ]
        )

        idx = (
            pd.MultiIndex.from_frame(
                df.loc[
                    mask,
                    keys,
                ]
            )
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

        count = int(
            np.isfinite(
                values
            ).sum()
        )
        attached_rows += count

        print(
            f"  {year}: attached "
            f"{count:,} frozen "
            "B4 scores"
        )

    coverage = (
        attached_rows
        / expected_rows
        if expected_rows
        else 0.0
    )

    if coverage < 0.995:
        raise RuntimeError(
            "Frozen B4 score coverage "
            f"is only {coverage:.2%}."
        )

    add_daily_ranks(
        df
    )

    return {
        "prediction_stage": (
            prediction_stage
        ),
        "expected_rows": int(
            expected_rows
        ),
        "attached_rows": int(
            attached_rows
        ),
        "coverage": float(
            coverage
        ),
    }


def policy_by_name(
    name: str,
) -> dict:
    for policy in POLICIES:
        if policy[
            "name"
        ] == name:
            return dict(
                policy
            )

    raise ValueError(
        "Unknown policy. Choose one of: "
        + ", ".join(
            item[
                "name"
            ]
            for item in POLICIES
        )
    )


def stage_policies(
    *,
    stage: str,
    selected: str | None,
) -> list[dict]:
    if stage == "development":
        return [
            dict(
                item
            )
            for item in POLICIES
        ]

    if selected is None:
        raise ValueError(
            "--policy is required "
            "for confirmation stage."
        )

    return [
        policy_by_name(
            selected
        )
    ]


def annual_summary(
    annual: pd.DataFrame,
) -> dict:
    if annual.empty:
        return {
            "median_annual_sharpe": None,
            "worst_annual_sharpe": None,
            "median_annual_cagr": None,
            "worst_annual_cagr": None,
            "median_annual_drawdown": None,
        }

    sharpe = pd.to_numeric(
        annual[
            "sharpe"
        ],
        errors="coerce",
    ).dropna()

    cagr = pd.to_numeric(
        annual[
            "cagr"
        ],
        errors="coerce",
    ).dropna()

    drawdown = pd.to_numeric(
        annual[
            "max_drawdown"
        ],
        errors="coerce",
    ).dropna()

    return {
        "median_annual_sharpe": (
            float(
                sharpe.median()
            )
            if len(
                sharpe
            )
            else None
        ),
        "worst_annual_sharpe": (
            float(
                sharpe.min()
            )
            if len(
                sharpe
            )
            else None
        ),
        "median_annual_cagr": (
            float(
                cagr.median()
            )
            if len(
                cagr
            )
            else None
        ),
        "worst_annual_cagr": (
            float(
                cagr.min()
            )
            if len(
                cagr
            )
            else None
        ),
        "median_annual_drawdown": (
            float(
                drawdown.median()
            )
            if len(
                drawdown
            )
            else None
        ),
    }


def run_experiment(
    root: Path,
    *,
    stage: str,
    selected_policy: str | None,
    capital: float,
    risk_free_rate: float,
    brokerage_per_order: float,
    dp_charge_per_sell: float,
    slippage_bps: float,
) -> dict:
    root = Path(
        root
    ).resolve()

    years = (
        DEVELOPMENT_YEARS
        if stage
        == "development"
        else CONFIRMATION_YEARS
    )

    policies = (
        stage_policies(
            stage=stage,
            selected=(
                selected_policy
            ),
        )
    )

    df = load_panel(
        root,
        years=years,
    )

    coverage = attach_scores(
        df,
        root=root,
        stage=stage,
        years=years,
    )

    events = (
        load_mechanical_events(
            root
        )
    )

    costs = CostProfile(
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

    output_root = (
        root
        / "reports/ml/"
        "largecap_portfolio_v1"
        / stage
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_table, curves = (
        benchmark_rows(
            root,
            panel_dates=set(
                df[
                    "date"
                ].dropna()
            ),
            capital=capital,
            annual_risk_free_rate=(
                risk_free_rate
            ),
        )
    )

    if (
        "NIFTY 100"
        not in curves
    ):
        raise RuntimeError(
            "NIFTY 100 benchmark curve "
            "is required for large-cap "
            "portfolio evaluation."
        )

    nifty100 = curves[
        "NIFTY 100"
    ]

    leaderboard_rows = []
    annual_frames = []
    equity_frames = []
    trade_frames = []
    rebalance_frames = []

    for number, policy in enumerate(
        policies,
        start=1,
    ):
        print(
            "\n["
            f"{number}/{len(policies)}"
            "] "
            f"{policy['name']} "
            f"K={policy['top_k']} "
            f"reb={policy['rebalance_sessions']} "
            f"hold={policy['hold_rank']}"
        )

        result = (
            run_persistent_backtest(
                df,
                events,
                name=(
                    policy[
                        "name"
                    ]
                ),
                initial_capital=(
                    capital
                ),
                top_k=int(
                    policy[
                        "top_k"
                    ]
                ),
                hold_rank=int(
                    policy[
                        "hold_rank"
                    ]
                ),
                min_score_gap=float(
                    policy[
                        "min_score_gap"
                    ]
                ),
                rebalance_sessions=int(
                    policy[
                        "rebalance_sessions"
                    ]
                ),
                costs=costs,
                annual_risk_free_rate=(
                    risk_free_rate
                ),
            )
        )

        relative = (
            benchmark_relative_metrics(
                result[
                    "equity"
                ],
                nifty100,
                annual_risk_free_rate=(
                    risk_free_rate
                ),
            )
        )

        annual = (
            annual_metrics_from_result(
                result,
                annual_risk_free_rate=(
                    risk_free_rate
                ),
                metadata={
                    "policy": (
                        policy[
                            "name"
                        ]
                    ),
                    "top_k": int(
                        policy[
                            "top_k"
                        ]
                    ),
                    "rebalance_sessions": int(
                        policy[
                            "rebalance_sessions"
                        ]
                    ),
                    "hold_rank": int(
                        policy[
                            "hold_rank"
                        ]
                    ),
                    "hold_multiplier": int(
                        policy[
                            "hold_multiplier"
                        ]
                    ),
                    "min_score_gap": float(
                        policy[
                            "min_score_gap"
                        ]
                    ),
                },
            )
        )

        annual_frames.append(
            annual
        )

        metrics = dict(
            result[
                "metrics"
            ]
        )

        row = {
            "policy": (
                policy[
                    "name"
                ]
            ),
            "top_k": int(
                policy[
                    "top_k"
                ]
            ),
            "rebalance_sessions": int(
                policy[
                    "rebalance_sessions"
                ]
            ),
            "hold_rank": int(
                policy[
                    "hold_rank"
                ]
            ),
            "hold_multiplier": int(
                policy[
                    "hold_multiplier"
                ]
            ),
            "min_score_gap": float(
                policy[
                    "min_score_gap"
                ]
            ),
            **metrics,
            **relative,
            **annual_summary(
                annual
            ),
        }

        leaderboard_rows.append(
            row
        )

        equity = (
            result[
                "equity"
            ].copy()
        )
        equity[
            "policy"
        ] = (
            policy[
                "name"
            ]
        )
        equity_frames.append(
            equity
        )

        trades = (
            result[
                "trades"
            ].copy()
        )
        if not trades.empty:
            trades[
                "policy"
            ] = (
                policy[
                    "name"
                ]
            )
            trade_frames.append(
                trades
            )

        rebalances = (
            result[
                "rebalances"
            ].copy()
        )
        if not rebalances.empty:
            rebalances[
                "policy"
            ] = (
                policy[
                    "name"
                ]
            )
            rebalance_frames.append(
                rebalances
            )

        print(
            "  CAGR="
            f"{metrics['cagr']:.2%} "
            "Sharpe="
            f"{metrics['sharpe']:.3f} "
            "MDD="
            f"{metrics['max_drawdown']:.2%} "
            "excess="
            f"{float(relative['excess_cagr']):+.2%} "
            "turnover="
            f"{metrics['turnover_multiple']:.1f}x"
        )

    leaderboard = (
        pd.DataFrame(
            leaderboard_rows
        )
    )

    # Predeclared development ranking: net Sharpe first, then the
    # weakest annual Sharpe, then shallower drawdown and lower turnover.
    leaderboard = (
        leaderboard.sort_values(
            [
                "sharpe",
                "worst_annual_sharpe",
                "max_drawdown",
                "turnover_multiple",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    leaderboard[
        "development_rank"
    ] = np.arange(
        1,
        len(
            leaderboard
        )
        + 1,
    )

    leaderboard.to_csv(
        output_root
        / "leaderboard.csv",
        index=False,
    )

    if annual_frames:
        pd.concat(
            annual_frames,
            ignore_index=True,
        ).to_csv(
            output_root
            / "annual_metrics.csv",
            index=False,
        )

    if equity_frames:
        pd.concat(
            equity_frames,
            ignore_index=True,
        ).to_parquet(
            output_root
            / "equity_curves.parquet",
            index=False,
            compression="zstd",
        )

    if trade_frames:
        pd.concat(
            trade_frames,
            ignore_index=True,
        ).to_parquet(
            output_root
            / "trades.parquet",
            index=False,
            compression="zstd",
        )

    if rebalance_frames:
        pd.concat(
            rebalance_frames,
            ignore_index=True,
        ).to_csv(
            output_root
            / "rebalance_log.csv",
            index=False,
        )

    benchmark_table.to_csv(
        output_root
        / "benchmark_metrics.csv",
        index=False,
    )

    manifest = {
        "stage": stage,
        "signal": {
            "model": "xgboost",
            "feature_set": (
                FEATURE_SET
            ),
            "feature_count": 86,
            "decision_time": (
                "close(t)"
            ),
            "execution": (
                "next market open"
            ),
            "target_horizon_sessions": 20,
        },
        "years": years,
        "prediction_coverage": (
            coverage
        ),
        "portfolio_search": {
            "principle": (
                "small predeclared first-pass "
                "policy grid; no score-gap, "
                "volatility weighting, sector "
                "caps or regime filters yet"
            ),
            "policies": policies,
            "development_selection_rule": (
                "highest net Sharpe; tie-break "
                "higher worst annual Sharpe, "
                "shallower max drawdown, then "
                "lower turnover"
            ),
        },
        "capital": float(
            capital
        ),
        "risk_free_rate_annual": float(
            risk_free_rate
        ),
        "cost_profile": asdict(
            costs
        ),
        "benchmark": (
            "NIFTY 100 price index"
        ),
        "known_limitations": [
            (
                "benchmark is a price index "
                "rather than total-return index"
            ),
            (
                "2026 is partial through "
                "2026-09-30"
            ),
            (
                "current 2026 cost schedule is "
                "stressed across historical years"
            ),
            (
                "dividends are not credited"
            ),
            (
                "market impact is not modeled"
            ),
        ],
    }

    (
        output_root
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    summary = {
        "stage": stage,
        "years": years,
        "policies": int(
            len(
                policies
            )
        ),
        "best_policy_by_declared_rule": (
            str(
                leaderboard.iloc[
                    0
                ][
                    "policy"
                ]
            )
            if not leaderboard.empty
            else None
        ),
        "best_sharpe": (
            float(
                leaderboard.iloc[
                    0
                ][
                    "sharpe"
                ]
            )
            if not leaderboard.empty
            else None
        ),
        "target_sharpe": 1.4,
        "output_root": str(
            output_root
        ),
    }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    names = [
        policy[
            "name"
        ]
        for policy in POLICIES
    ]

    assert len(
        names
    ) == len(
        set(
            names
        )
    )

    assert len(
        POLICIES
    ) == 12

    for policy in POLICIES:
        assert policy[
            "top_k"
        ] in {
            5,
            10,
            15,
        }
        assert policy[
            "rebalance_sessions"
        ] in {
            10,
            20,
        }
        assert policy[
            "hold_rank"
        ] >= policy[
            "top_k"
        ]
        assert policy[
            "min_score_gap"
        ] == 0.0

    print(
        "Large-cap portfolio-v1 "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "First portfolio-construction "
            "experiment for the frozen B4 "
            "large-cap XGBoost signal."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--stage",
        choices=[
            "development",
            "confirmation",
        ],
        default="development",
    )
    ap.add_argument(
        "--policy",
        default=None,
        help=(
            "Required for confirmation. "
            "Example: k10_r20_h20"
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
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if (
        args.stage
        == "confirmation"
        and args.policy
        is None
    ):
        raise SystemExit(
            "--policy is required "
            "for confirmation stage."
        )

    summary = run_experiment(
        Path(
            args.root
        ),
        stage=args.stage,
        selected_policy=(
            args.policy
        ),
        capital=float(
            args.capital
        ),
        risk_free_rate=float(
            args.risk_free_rate
        ),
        brokerage_per_order=float(
            args.brokerage_per_order
        ),
        dp_charge_per_sell=float(
            args.dp_charge_per_sell
        ),
        slippage_bps=float(
            args.slippage_bps
        ),
    )

    print(
        "\n=== LARGE-CAP PORTFOLIO V1 COMPLETE ==="
    )
    print(
        f"Stage:       "
        f"{summary['stage']}"
    )
    print(
        f"Years:       "
        f"{summary['years']}"
    )
    print(
        f"Policies:    "
        f"{summary['policies']}"
    )
    print(
        f"Best policy: "
        f"{summary['best_policy_by_declared_rule']}"
    )
    print(
        f"Best Sharpe: "
        f"{summary['best_sharpe']:.3f}"
    )
    print(
        f"Target:      "
        f"{summary['target_sharpe']:.1f}"
    )
    print(
        f"Output:      "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
