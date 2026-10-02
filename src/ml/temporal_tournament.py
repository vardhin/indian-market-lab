from __future__ import annotations

import argparse
import json
import sys
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

from baselines import (  # noqa: E402
    load_mechanical_events,
)
from classical import (  # noqa: E402
    add_target_rank,
    load_ml_panel,
)
from model_tournament import (  # noqa: E402
    cost_profile,
    evaluate_scores,
)
from nested_generalization import (  # noqa: E402
    benchmark_rows,
    first_aligned_signal_date,
)
from sequence_rankers import (  # noqa: E402
    SequenceConfig,
    annual_sequence_predictions,
)


DEFAULT_MODELS = [
    "seq_mlp",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "patch_transformer",
    "itransformer",
    "nhits_style",
]
DEFAULT_TOP_K = [
    5,
    10,
    15,
    20,
]


def self_test() -> None:
    config = SequenceConfig()
    assert config.lookback > 0
    assert config.epochs > 0
    assert config.batch_size > 0
    print(
        "Temporal tournament self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Annual walk-forward tournament "
            "for sequence models using the "
            "same frozen portfolio protocol."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=(
            DEFAULT_MODELS
        ),
        choices=[
            "seq_mlp",
            "lstm",
            "gru",
            "tcn",
            "transformer",
            "patch_transformer",
            "itransformer",
            "nhits_style",
            "mamba",
        ],
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
        "--oos-start-year",
        type=int,
        default=2019,
    )
    ap.add_argument(
        "--oos-end-year",
        type=int,
        default=2026,
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
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--lookback",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=6,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--max-train-sequences",
        type=int,
        default=300_000,
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

    config = SequenceConfig(
        lookback=int(
            args.lookback
        ),
        hidden_size=int(
            args.hidden_size
        ),
        layers=int(
            args.layers
        ),
        dropout=float(
            args.dropout
        ),
        epochs=int(
            args.epochs
        ),
        batch_size=int(
            args.batch_size
        ),
        learning_rate=float(
            args.learning_rate
        ),
        weight_decay=float(
            args.weight_decay
        ),
        max_train_sequences=int(
            args.max_train_sequences
        ),
    )

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "model_tournament/"
        "temporal"
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

    eval_start_date = (
        first_aligned_signal_date(
            df,
            oos_start_year=(
                args.oos_start_year
            ),
            eval_start_year=(
                args.oos_start_year
            ),
        )
    )
    years = list(
        range(
            args.oos_start_year,
            args.oos_end_year
            + 1,
        )
    )

    events = (
        load_mechanical_events(
            root
        )
    )
    costs = cost_profile(
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

    panel_dates = set(
        df.loc[
            df["date"].between(
                eval_start_date,
                pd.Timestamp(
                    f"{args.oos_end_year}-12-31"
                ),
                inclusive="both",
            ),
            "date",
        ].dropna()
    )
    benchmark_table, benchmark_curves = (
        benchmark_rows(
            root,
            panel_dates=(
                panel_dates
            ),
            capital=(
                args.capital
            ),
            annual_risk_free_rate=(
                args.risk_free_rate
            ),
        )
    )
    benchmark_curve = (
        benchmark_curves[
            "NIFTY 500"
        ]
    )

    leaderboard_frames: list[
        pd.DataFrame
    ] = []
    annual_frames: list[
        pd.DataFrame
    ] = []
    equity_frames: list[
        pd.DataFrame
    ] = []
    diagnostics_frames: list[
        pd.DataFrame
    ] = []
    failures: list[
        dict
    ] = []

    for model_name in args.models:
        print(
            "\n================================"
        )
        print(
            f"TEMPORAL MODEL: {model_name}"
        )
        print(
            "================================"
        )

        try:
            diagnostics = (
                annual_sequence_predictions(
                    df,
                    root=root,
                    model_name=(
                        model_name
                    ),
                    years=years,
                    seed=int(
                        args.seed
                    ),
                    config=config,
                    rebuild=(
                        args.rebuild_predictions
                    ),
                )
            )
            diagnostics[
                "seed"
            ] = int(
                args.seed
            )
            diagnostics_frames.append(
                diagnostics
            )

            (
                leaderboard,
                annual,
                equity,
            ) = evaluate_scores(
                df,
                events,
                model_name=(
                    model_name
                ),
                eval_start_date=(
                    eval_start_date
                ),
                eval_end_year=(
                    args.oos_end_year
                ),
                top_k_values=[
                    int(k)
                    for k in args.top_k
                ],
                initial_capital=(
                    args.capital
                ),
                costs=costs,
                annual_risk_free_rate=(
                    args.risk_free_rate
                ),
                benchmark_curve=(
                    benchmark_curve
                ),
            )
            leaderboard_frames.append(
                leaderboard
            )
            annual_frames.append(
                annual
            )
            equity_frames.append(
                equity
            )

        except Exception as exc:
            print(
                f"FAILED {model_name}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            failures.append({
                "model_name": (
                    model_name
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(
                    exc
                ),
            })

    if leaderboard_frames:
        leaderboard = pd.concat(
            leaderboard_frames,
            ignore_index=True,
        )
        leaderboard = (
            leaderboard.sort_values(
                [
                    "economic_target_excess5_sharpe1",
                    "sharpe",
                    "excess_cagr",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )
        leaderboard.to_csv(
            output_root
            / "leaderboard.csv",
            index=False,
        )

        pd.concat(
            annual_frames,
            ignore_index=True,
        ).to_csv(
            output_root
            / "annual_portfolio_metrics.csv",
            index=False,
        )

        pd.concat(
            equity_frames,
            ignore_index=True,
        ).to_parquet(
            output_root
            / "equity_curves.parquet",
            index=False,
            compression="zstd",
        )

        print(
            "\n=== TEMPORAL TOURNAMENT ==="
        )
        preview = leaderboard[
            [
                "model_name",
                "top_k",
                "cagr",
                "excess_cagr",
                "alpha_annualized",
                "sharpe",
                "max_drawdown",
                "median_annual_cagr",
                "q25_annual_cagr",
                "economic_target_excess5_sharpe1",
            ]
        ].copy()
        print(
            preview.to_string(
                index=False,
                formatters={
                    "cagr": (
                        lambda x: f"{x:.2%}"
                    ),
                    "excess_cagr": (
                        lambda x: f"{x:+.2%}"
                    ),
                    "alpha_annualized": (
                        lambda x: (
                            ""
                            if pd.isna(x)
                            else f"{x:+.2%}"
                        )
                    ),
                    "sharpe": (
                        lambda x: f"{x:.3f}"
                    ),
                    "max_drawdown": (
                        lambda x: f"{x:.2%}"
                    ),
                    "median_annual_cagr": (
                        lambda x: (
                            ""
                            if pd.isna(x)
                            else f"{x:.2%}"
                        )
                    ),
                    "q25_annual_cagr": (
                        lambda x: (
                            ""
                            if pd.isna(x)
                            else f"{x:.2%}"
                        )
                    ),
                },
            )
        )

    if diagnostics_frames:
        pd.concat(
            diagnostics_frames,
            ignore_index=True,
        ).to_csv(
            output_root
            / "annual_prediction_diagnostics.csv",
            index=False,
        )

    benchmark_table.to_csv(
        output_root
        / "benchmark_reference.csv",
        index=False,
    )
    pd.DataFrame(
        failures
    ).to_csv(
        output_root
        / "failures.csv",
        index=False,
    )

    summary = {
        "models_requested": list(
            args.models
        ),
        "sequence_config": {
            "lookback": (
                config.lookback
            ),
            "hidden_size": (
                config.hidden_size
            ),
            "layers": (
                config.layers
            ),
            "dropout": (
                config.dropout
            ),
            "epochs": (
                config.epochs
            ),
            "batch_size": (
                config.batch_size
            ),
            "learning_rate": (
                config.learning_rate
            ),
            "weight_decay": (
                config.weight_decay
            ),
            "max_train_sequences": (
                config.max_train_sequences
            ),
        },
        "protocol": (
            "same annual OOS target, "
            "same 20-session purge, same "
            "persistent top-K portfolio and "
            "same execution/cost model as "
            "tabular tournament"
        ),
        "model_notes": {
            "nhits_style": (
                "N-HiTS-inspired hierarchical "
                "multi-resolution supervised "
                "ranker; not the canonical "
                "Nixtla multi-step forecaster"
            ),
            "patch_transformer": (
                "Patch-based transformer "
                "adapted to supervised "
                "cross-sectional rank regression"
            ),
            "itransformer": (
                "Variate-token transformer "
                "adapted to supervised "
                "cross-sectional rank regression"
            ),
            "mamba": (
                "Optional mamba_ssm sequence "
                "ranker; requires a compatible "
                "CUDA/compiler environment"
            ),
        },
        "failures": failures,
    }

    (
        output_root
        / "temporal_tournament_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print(
        "\nOutputs: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
