from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
ML_DIR = SCRIPT_PATH.parent
SRC_DIR = SCRIPT_PATH.parents[1]
FEATURE_DIR = SRC_DIR / "features"
BACKTEST_DIR = SRC_DIR / "backtest"

for path in (
    ML_DIR,
    FEATURE_DIR,
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
    TRAIN_START,
    build_model,
    cross_sectional_diagnostics,
    fit_model,
    predict_mask,
)
from feature_registry import (  # noqa: E402
    SAFE_DEFAULT_GROUPS,
    cumulative_feature_sets,
    feature_columns,
)
from model_tournament import (  # noqa: E402
    cost_profile,
    evaluate_scores,
)
from nested_generalization import (  # noqa: E402
    benchmark_rows,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
)


HORIZON = 20
TARGET_COLUMN = (
    "target_next_open_to_close_20d"
)
TRAINING_FLAG = (
    "training_eligible_20d"
)
UNSAFE_COLUMN = (
    "unsafe_target_window_20d"
)

DEFAULT_MODELS = [
    "xgboost",
]
DEFAULT_TOP_K = [
    5,
    10,
    20,
]
DEVELOPMENT_YEARS = (
    2019,
    2023,
)
EVALUATION_YEARS = (
    2024,
    2026,
)


def cache_tag(
    feature_set: str,
    columns: list[str],
) -> str:
    digest = hashlib.sha1(
        "\n".join(
            columns
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    clean = (
        feature_set
        .replace("+", "_")
        .replace("/", "_")
    )
    return (
        f"{clean}__{digest}"
    )


def load_enriched_panel(
    panel_root: Path,
    *,
    feature_columns_union: list[str],
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
            "No enriched-panel partitions "
            f"under {panel_root}"
        )

    available = set(
        pq.read_schema(
            files[0]
        ).names
    )

    base = [
        "date",
        "market_day_index",
        "canonical_security_id",
        "symbol",
        "eligible_universe",
        "open",
        "close",
        "turnover_median_20d",
        UNSAFE_COLUMN,
        TRAINING_FLAG,
        TARGET_COLUMN,
        "largecap_flag_numeric",
        "sector",
        "industry",
        "market_context_index",
        "calendar_year",
    ]

    missing_features = [
        column
        for column in feature_columns_union
        if column not in available
    ]
    if missing_features:
        raise RuntimeError(
            "Enriched panel is missing "
            "requested feature columns: "
            f"{missing_features}"
        )

    required = [
        column
        for column in base
        if column
        not in {
            "largecap_flag_numeric",
            "sector",
            "industry",
            "market_context_index",
            "calendar_year",
        }
    ]
    missing_required = [
        column
        for column in required
        if column not in available
    ]
    if missing_required:
        raise RuntimeError(
            "Enriched panel is missing "
            "required columns: "
            f"{missing_required}"
        )

    columns = list(
        dict.fromkeys(
            [
                column
                for column in base
                if column in available
            ]
            + feature_columns_union
        )
    )

    print(
        f"Loading {len(files):,} enriched "
        "partitions for feature ablation..."
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
        .astype("string")
        .str.strip()
    )

    for column in feature_columns_union:
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

    df[
        TARGET_COLUMN
    ] = pd.to_numeric(
        df[
            TARGET_COLUMN
        ],
        errors="coerce",
    ).astype(
        "float32"
    )

    for column in (
        "eligible_universe",
        TRAINING_FLAG,
        UNSAFE_COLUMN,
    ):
        df[column] = (
            df[column]
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


def apply_universe(
    df: pd.DataFrame,
    *,
    universe: str,
) -> pd.Series:
    base = (
        df[
            "eligible_universe"
        ].fillna(False)
    )

    if universe == "existing":
        return base

    if universe != "largecap":
        raise ValueError(
            f"Unknown universe: {universe}"
        )

    if (
        "largecap_flag_numeric"
        not in df.columns
    ):
        raise RuntimeError(
            "Large-cap research requires "
            "point-in-time security context."
        )

    flag = pd.to_numeric(
        df[
            "largecap_flag_numeric"
        ],
        errors="coerce",
    )

    coverage = float(
        flag.notna().mean()
    )
    if coverage < 0.01:
        raise RuntimeError(
            "Large-cap context is effectively "
            "missing. Do not substitute the "
            "current 2026 security master."
        )

    return (
        base
        & flag.eq(
            1.0
        )
    )


def add_target_rank(
    df: pd.DataFrame,
    *,
    universe_mask: pd.Series,
) -> None:
    safe = (
        universe_mask
        & df[
            TRAINING_FLAG
        ]
        & df[
            TARGET_COLUMN
        ].notna()
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
        )[
            TARGET_COLUMN
        ]
        .rank(
            method="average",
            pct=True,
        )
        .astype(
            "float32"
        )
    )


def prediction_path(
    root: Path,
    *,
    stage: str,
    universe: str,
    feature_set: str,
    columns: list[str],
    model: str,
    seed: int,
    year: int,
) -> Path:
    return (
        root
        / "reports/ml/feature_ablation/"
        "predictions"
        / stage
        / universe
        / cache_tag(
            feature_set,
            columns,
        )
        / f"{model}_rs{seed}"
        / f"{year}.parquet"
    )


def attach_cache(
    df: pd.DataFrame,
    *,
    path: Path,
    year: int,
    score_mask: pd.Series,
) -> int:
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

    keys = [
        "date",
        "canonical_security_id",
    ]
    if saved.duplicated(
        keys,
        keep=False,
    ).any():
        raise RuntimeError(
            f"Duplicate prediction keys: {path}"
        )

    mapping = saved.set_index(
        keys
    )[
        "persistent_score"
    ]
    idx = pd.MultiIndex.from_frame(
        df.loc[
            score_mask,
            keys,
        ]
    )
    values = mapping.reindex(
        idx
    ).to_numpy(
        dtype=float
    )
    df.loc[
        score_mask,
        "persistent_score",
    ] = values

    return int(
        np.isfinite(
            values
        ).sum()
    )


def annual_predictions(
    df: pd.DataFrame,
    *,
    root: Path,
    stage: str,
    universe: str,
    universe_mask: pd.Series,
    feature_set: str,
    columns: list[str],
    model_name: str,
    seed: int,
    years: list[int],
    rebuild: bool,
) -> pd.DataFrame:
    df[
        "persistent_score"
    ] = np.nan

    rows: list[
        dict
    ] = []

    for year in years:
        fold_start = pd.Timestamp(
            f"{year}-01-01"
        )
        fold_end = pd.Timestamp(
            f"{year}-12-31"
        )

        score_mask = (
            universe_mask
            & df[
                "date"
            ].between(
                fold_start,
                fold_end,
                inclusive="both",
            )
        )

        path = prediction_path(
            root,
            stage=stage,
            universe=universe,
            feature_set=(
                feature_set
            ),
            columns=columns,
            model=model_name,
            seed=seed,
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
            count = attach_cache(
                df,
                path=path,
                year=year,
                score_mask=(
                    score_mask
                ),
            )
            print(
                f"{feature_set} / "
                f"{model_name} / {year}: "
                f"reused {count:,} scores"
            )
        else:
            prior = df.loc[
                df[
                    "date"
                ].lt(
                    fold_start
                ),
                "market_day_index",
            ].dropna()
            if prior.empty:
                raise RuntimeError(
                    f"No prior sessions for {year}"
                )

            cutoff = int(
                prior.max()
            )

            train_mask = (
                universe_mask
                & df[
                    TRAINING_FLAG
                ]
                & df[
                    "target_rank_20d"
                ].notna()
                & df[
                    "date"
                ].ge(
                    TRAIN_START
                )
                & df[
                    "market_day_index"
                ].le(
                    cutoff
                    - HORIZON
                )
            )

            print(
                f"{feature_set} / "
                f"{model_name} / {year}: "
                f"fit {int(train_mask.sum()):,}, "
                f"score {int(score_mask.sum()):,}",
                flush=True,
            )

            model = build_model(
                model_name,
                random_state=seed,
            )
            fit_model(
                model,
                df,
                train_mask,
                feature_columns=(
                    columns
                ),
                target_column=(
                    "target_rank_20d"
                ),
            )
            scores = predict_mask(
                model,
                df,
                score_mask,
                feature_columns=(
                    columns
                ),
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
            score_mask
            & df[
                TRAINING_FLAG
            ]
            & df[
                "target_rank_20d"
            ].notna()
            & df[
                "persistent_score"
            ].notna()
        )

        if label_mask.any():
            diag = (
                cross_sectional_diagnostics(
                    df,
                    label_mask,
                    df.loc[
                        label_mask,
                        "persistent_score",
                    ].to_numpy(
                        dtype=float
                    ),
                )
            )
        else:
            diag = {
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

        rows.append({
            "feature_set": feature_set,
            "model_name": model_name,
            "year": int(year),
            **diag,
        })

    add_daily_ranks(
        df
    )
    return pd.DataFrame(
        rows
    )


def stage_years(
    stage: str,
) -> list[int]:
    if stage == "development":
        start, end = (
            DEVELOPMENT_YEARS
        )
    elif stage == "evaluation":
        start, end = (
            EVALUATION_YEARS
        )
    else:
        raise ValueError(
            f"Unknown stage: {stage}"
        )

    return list(
        range(
            start,
            end + 1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Predeclared feature-family ablation "
            "with a development/evaluation split."
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
        "--stage",
        choices=[
            "development",
            "evaluation",
        ],
        default="development",
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
        "--models",
        nargs="+",
        default=(
            DEFAULT_MODELS
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
        "--feature-set",
        default=None,
        help=(
            "Required for evaluation. Example: "
            "F0+F1+F2+F3+F4+F6"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--rebuild",
        action="store_true",
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()

    feature_sets = (
        cumulative_feature_sets(
            SAFE_DEFAULT_GROUPS
        )
    )

    if args.stage == "evaluation":
        if not args.feature_set:
            raise RuntimeError(
                "Evaluation is sealed to one "
                "preselected feature set. Pass "
                "--feature-set explicitly."
            )
        if (
            args.feature_set
            not in feature_sets
        ):
            raise RuntimeError(
                "Evaluation feature set must "
                "be one of the preregistered "
                "cumulative sets."
            )
        selected_sets = {
            args.feature_set: (
                feature_sets[
                    args.feature_set
                ]
            )
        }
    else:
        if args.feature_set:
            if (
                args.feature_set
                not in feature_sets
            ):
                raise RuntimeError(
                    "Unknown preregistered "
                    "feature set."
                )
            selected_sets = {
                args.feature_set: (
                    feature_sets[
                        args.feature_set
                    ]
                )
            }
        else:
            selected_sets = (
                feature_sets
            )

    union = list(
        dict.fromkeys(
            column
            for columns
            in selected_sets.values()
            for column in columns
        )
    )

    df = load_enriched_panel(
        root
        / args.panel,
        feature_columns_union=(
            union
        ),
    )

    universe_mask = (
        apply_universe(
            df,
            universe=(
                args.universe
            ),
        )
    )

    # Backtester consumes eligible_universe;
    # preserve the original and then restrict it
    # to the research universe.
    df[
        "eligible_universe_original"
    ] = df[
        "eligible_universe"
    ]
    df[
        "eligible_universe"
    ] = universe_mask

    add_target_rank(
        df,
        universe_mask=(
            universe_mask
        ),
    )

    years = stage_years(
        args.stage
    )
    eval_start = pd.Timestamp(
        f"{years[0]}-01-01"
    )
    eval_end = pd.Timestamp(
        f"{years[-1]}-12-31"
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
            df[
                "date"
            ].between(
                eval_start,
                eval_end,
                inclusive="both",
            ),
            "date",
        ].dropna()
    )
    benchmark_table, curves = (
        benchmark_rows(
            root,
            panel_dates=(
                panel_dates
            ),
            capital=float(
                args.capital
            ),
            annual_risk_free_rate=float(
                args.risk_free_rate
            ),
        )
    )

    benchmark_name = (
        "NIFTY 100"
        if args.universe
        == "largecap"
        else "NIFTY 500"
    )
    if benchmark_name not in curves:
        raise RuntimeError(
            f"{benchmark_name} benchmark is "
            "required for universe="
            f"{args.universe}. Rebuild index "
            "history first."
        )
    benchmark_curve = curves[
        benchmark_name
    ]

    leaderboard_frames: list[
        pd.DataFrame
    ] = []
    annual_frames: list[
        pd.DataFrame
    ] = []
    diagnostics_frames: list[
        pd.DataFrame
    ] = []
    equity_frames: list[
        pd.DataFrame
    ] = []

    for (
        feature_set,
        columns,
    ) in selected_sets.items():
        print(
            "\n================================"
        )
        print(
            f"FEATURE SET: {feature_set}"
        )
        print(
            f"Features: {len(columns)}"
        )
        print(
            "================================"
        )

        for model_name in args.models:
            diagnostics = (
                annual_predictions(
                    df,
                    root=root,
                    stage=args.stage,
                    universe=(
                        args.universe
                    ),
                    universe_mask=(
                        universe_mask
                    ),
                    feature_set=(
                        feature_set
                    ),
                    columns=columns,
                    model_name=(
                        model_name
                    ),
                    seed=int(
                        args.seed
                    ),
                    years=years,
                    rebuild=(
                        args.rebuild
                    ),
                )
            )
            diagnostics[
                "feature_count"
            ] = len(
                columns
            )
            diagnostics_frames.append(
                diagnostics
            )

            display_name = (
                f"{model_name}__"
                f"{feature_set}"
            )

            leaderboard, annual, equity = (
                evaluate_scores(
                    df,
                    events,
                    model_name=(
                        display_name
                    ),
                    eval_start_date=(
                        eval_start
                    ),
                    eval_end_year=(
                        years[-1]
                    ),
                    top_k_values=[
                        int(k)
                        for k
                        in args.top_k
                    ],
                    initial_capital=float(
                        args.capital
                    ),
                    costs=costs,
                    annual_risk_free_rate=float(
                        args.risk_free_rate
                    ),
                    benchmark_curve=(
                        benchmark_curve
                    ),
                )
            )

            for frame in (
                leaderboard,
                annual,
            ):
                frame[
                    "feature_set"
                ] = feature_set
                frame[
                    "feature_count"
                ] = len(
                    columns
                )
                frame[
                    "base_model"
                ] = model_name
                frame[
                    "universe"
                ] = args.universe
                frame[
                    "benchmark_name"
                ] = benchmark_name
                frame[
                    "stage"
                ] = args.stage

            equity[
                "feature_set"
            ] = feature_set
            equity[
                "base_model"
            ] = model_name
            equity[
                "universe"
            ] = args.universe
            equity[
                "stage"
            ] = args.stage

            leaderboard_frames.append(
                leaderboard
            )
            annual_frames.append(
                annual
            )
            equity_frames.append(
                equity
            )

    leaderboard = pd.concat(
        leaderboard_frames,
        ignore_index=True,
    )
    annual = pd.concat(
        annual_frames,
        ignore_index=True,
    )
    diagnostics = pd.concat(
        diagnostics_frames,
        ignore_index=True,
    )
    equity = pd.concat(
        equity_frames,
        ignore_index=True,
    )

    leaderboard = (
        leaderboard.sort_values(
            [
                "sharpe",
                "excess_cagr",
                "cagr",
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

    output = (
        root
        / "reports/ml/feature_ablation"
        / args.stage
        / args.universe
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard.to_csv(
        output
        / "leaderboard.csv",
        index=False,
    )
    annual.to_csv(
        output
        / "annual_metrics.csv",
        index=False,
    )
    diagnostics.to_csv(
        output
        / "prediction_diagnostics.csv",
        index=False,
    )
    equity.to_parquet(
        output
        / "equity_curves.parquet",
        index=False,
        compression="zstd",
    )
    benchmark_table.to_csv(
        output
        / "benchmark_reference.csv",
        index=False,
    )

    protocol = {
        "stage": args.stage,
        "development_years": (
            DEVELOPMENT_YEARS
        ),
        "evaluation_years": (
            EVALUATION_YEARS
        ),
        "universe": args.universe,
        "benchmark": benchmark_name,
        "horizon": HORIZON,
        "signal_time": (
            "after_close_t"
        ),
        "execution": (
            "next_session_open"
        ),
        "selected_feature_sets": {
            name: columns
            for name, columns
            in selected_sets.items()
        },
        "rule": (
            "Do not change feature families "
            "after inspecting sealed evaluation "
            "results without declaring a new "
            "research cycle."
        ),
    }
    (
        output
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n=== FEATURE ABLATION LEADERBOARD ==="
    )
    cols = [
        "feature_set",
        "base_model",
        "top_k",
        "feature_count",
        "cagr",
        "benchmark_cagr",
        "excess_cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "median_annual_cagr",
        "q25_annual_cagr",
        "median_annual_sharpe",
        "q25_annual_sharpe",
        "turnover_multiple",
        "total_fees",
    ]
    print(
        leaderboard[
            cols
        ].to_string(
            index=False,
            formatters={
                "cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "benchmark_cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "excess_cagr": (
                    lambda x: f"{x:+.2%}"
                ),
                "sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "sortino": (
                    lambda x: f"{x:.3f}"
                ),
                "max_drawdown": (
                    lambda x: f"{x:.2%}"
                ),
                "median_annual_cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "q25_annual_cagr": (
                    lambda x: f"{x:.2%}"
                ),
                "median_annual_sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "q25_annual_sharpe": (
                    lambda x: f"{x:.3f}"
                ),
                "turnover_multiple": (
                    lambda x: f"{x:.1f}x"
                ),
                "total_fees": (
                    lambda x: f"₹{x:,.0f}"
                ),
            },
        )
    )
    print(
        f"\nOutputs: {output}"
    )


if __name__ == "__main__":
    main()
