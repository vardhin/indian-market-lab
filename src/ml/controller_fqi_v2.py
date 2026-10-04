from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from controller_v1 import (
    CONTROLLER_FEATURES,
    TRAIN_YEARS,
    VALIDATION_YEARS,
    evaluate_sequential_policy,
)


DEFAULT_THRESHOLDS = [
    -0.005,
    0.0,
    0.0025,
    0.005,
    0.01,
    0.02,
]


def model_spec(
    name: str,
    *,
    random_state: int,
):
    if name == "histgb":
        model = (
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=0.5,
                random_state=(
                    random_state
                ),
            )
        )
    elif name == "random_forest":
        model = (
            RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=20,
                max_features=0.6,
                n_jobs=1,
                random_state=(
                    random_state
                ),
            )
        )
    elif name == "extra_trees":
        model = (
            ExtraTreesRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=18,
                max_features=0.7,
                n_jobs=-1,
                random_state=(
                    random_state
                ),
            )
        )
    else:
        raise ValueError(
            f"Unknown model: {name}"
        )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "model",
            model,
        ),
    ])


def prepare_transitions(
    teacher: pd.DataFrame,
) -> pd.DataFrame:
    work = teacher.copy()

    work[
        "state_date"
    ] = pd.to_datetime(
        work[
            "state_date"
        ],
        errors="coerce",
    ).dt.normalize()

    work = (
        work.sort_values(
            [
                "episode_id",
                "state_market_index",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    work[
        "next_episode_id"
    ] = work[
        "episode_id"
    ].shift(
        -1
    )
    work[
        "next_state_market_index"
    ] = work[
        "state_market_index"
    ].shift(
        -1
    )

    same_episode = (
        work[
            "episode_id"
        ].eq(
            work[
                "next_episode_id"
            ]
        )
    )

    work[
        "has_next_state"
    ] = (
        same_episode
    )

    return work


def build_next_matrix(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    next_rows = (
        transitions.groupby(
            "episode_id",
            sort=False,
        )[
            CONTROLLER_FEATURES
        ]
        .shift(
            -1
        )
    )
    return next_rows


def fit_exit_model(
    train: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
):
    model = model_spec(
        model_name,
        random_state=(
            random_state
        ),
    )
    model.fit(
        train[
            CONTROLLER_FEATURES
        ],
        train[
            "exit_next_open_net_return"
        ].astype(
            float
        ),
    )
    return model


def fitted_q_iteration(
    train: pd.DataFrame,
    *,
    model_name: str,
    iterations: int,
    random_state: int,
) -> tuple[
    object,
    object,
    pd.DataFrame,
]:
    train = (
        prepare_transitions(
            train
        )
    )

    next_X = build_next_matrix(
        train
    )

    exit_model = fit_exit_model(
        train,
        model_name=model_name,
        random_state=(
            random_state
        ),
    )

    q_exit = exit_model.predict(
        train[
            CONTROLLER_FEATURES
        ]
    )

    # Conservative initialization: before bootstrapping, assume HOLD is worth
    # no more than the expected exit value. The finite horizon encoded in the
    # state features and terminal-mask then shapes continuation values.
    hold_model = model_spec(
        model_name,
        random_state=(
            random_state
            + 10_000
        ),
    )
    hold_model.fit(
        train[
            CONTROLLER_FEATURES
        ],
        q_exit,
    )

    diagnostics = []

    for iteration in range(
        1,
        iterations + 1,
    ):
        q_hold_next = (
            hold_model.predict(
                next_X
            )
        )
        q_exit_next = (
            exit_model.predict(
                next_X
            )
        )

        next_value = np.maximum(
            q_exit_next,
            q_hold_next,
        )

        terminal = ~train[
            "has_next_state"
        ].to_numpy(
            dtype=bool
        )

        # HOLD is unavailable at the terminal controller state. Its target is
        # anchored to the expected EXIT value so it cannot manufacture value
        # beyond the finite horizon.
        target_hold = next_value.copy()
        target_hold[
            terminal
        ] = q_exit[
            terminal
        ]

        # Bound bootstrap targets to the observed training-return envelope.
        # This prevents tree extrapolation from producing runaway recursive
        # values while retaining the actual empirical support.
        realized_exit = train[
            "exit_next_open_net_return"
        ].to_numpy(
            dtype=float
        )
        lower = float(
            np.quantile(
                realized_exit,
                0.005,
            )
        )
        upper = float(
            np.quantile(
                realized_exit,
                0.995,
            )
        )
        target_hold = np.clip(
            target_hold,
            lower,
            upper,
        )

        new_hold_model = model_spec(
            model_name,
            random_state=(
                random_state
                + 10_000
                + iteration
            ),
        )
        new_hold_model.fit(
            train[
                CONTROLLER_FEATURES
            ],
            target_hold,
        )

        old_pred = hold_model.predict(
            train[
                CONTROLLER_FEATURES
            ]
        )
        new_pred = new_hold_model.predict(
            train[
                CONTROLLER_FEATURES
            ]
        )

        mean_abs_change = float(
            np.mean(
                np.abs(
                    new_pred
                    - old_pred
                )
            )
        )
        max_abs_change = float(
            np.max(
                np.abs(
                    new_pred
                    - old_pred
                )
            )
        )

        diagnostics.append({
            "iteration": int(
                iteration
            ),
            "mean_abs_q_hold_change": (
                mean_abs_change
            ),
            "max_abs_q_hold_change": (
                max_abs_change
            ),
            "mean_q_exit": float(
                np.mean(
                    q_exit
                )
            ),
            "mean_q_hold": float(
                np.mean(
                    new_pred
                )
            ),
        })

        hold_model = new_hold_model

    return (
        exit_model,
        hold_model,
        pd.DataFrame(
            diagnostics
        ),
    )


def predict_advantage(
    frame: pd.DataFrame,
    *,
    exit_model,
    hold_model,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X = frame[
        CONTROLLER_FEATURES
    ]
    q_exit = exit_model.predict(
        X
    )
    q_hold = hold_model.predict(
        X
    )
    advantage = (
        q_hold
        - q_exit
    )

    terminal = (
        pd.to_numeric(
            frame[
                "remaining_sessions"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .le(
            0
        )
        .to_numpy(
            dtype=bool
        )
    )

    # Force EXIT at the finite horizon.
    advantage[
        terminal
    ] = -np.inf

    return (
        q_exit,
        q_hold,
        advantage,
    )


def evaluate_fqi(
    root: Path,
    *,
    teacher_path: Path | None,
    models: list[str],
    iterations: int,
    random_state: int,
) -> dict:
    if teacher_path is None:
        teacher_path = (
            root
            / "data/processed/"
            "controller_v1/"
            "development_teacher.parquet"
        )

    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing teacher: {teacher_path}"
        )

    teacher = pd.read_parquet(
        teacher_path
    )
    teacher[
        "state_date"
    ] = pd.to_datetime(
        teacher[
            "state_date"
        ],
        errors="coerce",
    ).dt.normalize()

    train = teacher.loc[
        teacher[
            "state_date"
        ].dt.year.isin(
            TRAIN_YEARS
        )
    ].copy()

    validation = teacher.loc[
        teacher[
            "state_date"
        ].dt.year.isin(
            VALIDATION_YEARS
        )
    ].copy()

    if (
        train.empty
        or validation.empty
    ):
        raise RuntimeError(
            "FQI train/validation split "
            "is empty."
        )

    report_root = (
        root
        / "reports/ml/"
        "controller_fqi_v2/"
        "development"
    )
    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard_rows = []
    state_frames = []
    episode_frames = []
    iteration_frames = []

    print(
        "\n=== CONTROLLER FQI V2 ==="
    )
    print(
        f"Train states:      {len(train):,} "
        f"({TRAIN_YEARS})"
    )
    print(
        f"Validation states: {len(validation):,} "
        f"({VALIDATION_YEARS})"
    )
    print(
        f"Features:          "
        f"{len(CONTROLLER_FEATURES)}"
    )
    print(
        f"Bellman iterations: {iterations}"
    )

    for model_number, name in enumerate(
        models,
        start=1,
    ):
        print(
            f"\n=== {name} ==="
        )

        (
            exit_model,
            hold_model,
            iteration_diag,
        ) = fitted_q_iteration(
            train,
            model_name=name,
            iterations=(
                iterations
            ),
            random_state=(
                random_state
                + model_number
                * 100
            ),
        )

        (
            q_exit,
            q_hold,
            advantage,
        ) = predict_advantage(
            validation,
            exit_model=(
                exit_model
            ),
            hold_model=(
                hold_model
            ),
        )

        state = validation[
            [
                "episode_id",
                "state_date",
                "state_market_index",
                "canonical_security_id",
                "symbol",
                "holding_age_sessions",
                "remaining_sessions",
                "exit_next_open_net_return",
                "oracle_best_net_return",
            ]
        ].copy()
        state[
            "model"
        ] = name
        state[
            "q_exit"
        ] = q_exit
        state[
            "q_hold"
        ] = q_hold
        state[
            "q_advantage"
        ] = advantage
        state_frames.append(
            state
        )

        iteration_diag = (
            iteration_diag.copy()
        )
        iteration_diag[
            "model"
        ] = name
        iteration_frames.append(
            iteration_diag
        )

        for threshold in (
            DEFAULT_THRESHOLDS
        ):
            (
                metrics,
                episodes,
            ) = evaluate_sequential_policy(
                validation,
                advantage,
                threshold=float(
                    threshold
                ),
            )

            leaderboard_rows.append({
                "model": name,
                **metrics,
            })

            episodes = (
                episodes.copy()
            )
            episodes[
                "model"
            ] = name
            episodes[
                "threshold"
            ] = float(
                threshold
            )
            episode_frames.append(
                episodes
            )

        model_rows = pd.DataFrame([
            row
            for row in leaderboard_rows
            if row[
                "model"
            ] == name
        ])
        best = (
            model_rows.sort_values(
                [
                    "mean_gain_vs_forced",
                    "mean_regret_vs_oracle",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[
                0
            ]
        )

        print(
            "  best threshold="
            f"{float(best['threshold']):+.4f} "
            "gain-vs-20d="
            f"{float(best['mean_gain_vs_forced']):+.2%} "
            "oracle-captured="
            f"{float(best['oracle_headroom_captured']):.1%} "
            "exit-age="
            f"{float(best['mean_exit_age_sessions']):.1f}"
        )

    leaderboard = (
        pd.DataFrame(
            leaderboard_rows
        )
        .sort_values(
            [
                "mean_gain_vs_forced",
                "mean_regret_vs_oracle",
                "positive_gain_fraction",
                "forced_terminal_fraction",
            ],
            ascending=[
                False,
                True,
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
        report_root
        / "fqi_sequential_leaderboard.csv",
        index=False,
    )

    pd.concat(
        state_frames,
        ignore_index=True,
    ).to_parquet(
        report_root
        / "fqi_state_values.parquet",
        index=False,
        compression="zstd",
    )

    pd.concat(
        episode_frames,
        ignore_index=True,
    ).to_parquet(
        report_root
        / "fqi_episode_results.parquet",
        index=False,
        compression="zstd",
    )

    pd.concat(
        iteration_frames,
        ignore_index=True,
    ).to_csv(
        report_root
        / "fqi_iteration_diagnostics.csv",
        index=False,
    )

    best = leaderboard.iloc[
        0
    ]

    summary = {
        "algorithm": (
            "finite-horizon fitted Q iteration"
        ),
        "teacher_source": str(
            teacher_path
        ),
        "important_difference_from_v1": (
            "V1 imitates a pathwise hindsight maximum. V2 instead learns "
            "expected action values Q_EXIT(state) and Q_HOLD(state), then "
            "acts on their predicted difference."
        ),
        "train_years": (
            TRAIN_YEARS
        ),
        "validation_years": (
            VALIDATION_YEARS
        ),
        "features": int(
            len(
                CONTROLLER_FEATURES
            )
        ),
        "models": models,
        "iterations": int(
            iterations
        ),
        "thresholds": (
            DEFAULT_THRESHOLDS
        ),
        "selection_rule": (
            "highest sequential 2023 mean episode gain versus forced "
            "20-session exit; then lower regret versus the hindsight "
            "upper-bound oracle, higher positive-gain fraction, then "
            "lower terminal-hold fraction"
        ),
        "best_model": str(
            best[
                "model"
            ]
        ),
        "best_threshold": float(
            best[
                "threshold"
            ]
        ),
        "best_metrics": (
            best.to_dict()
        ),
        "next_required_test": (
            "If FQI improves materially over Controller V1, freeze the "
            "development policy and integrate it into the full portfolio "
            "simulator before opening 2024-2026."
        ),
    }

    (
        report_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=float,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    base = pd.DataFrame({
        "episode_id": (
            ["a"] * 3
            + ["b"] * 3
        ),
        "state_market_index": [
            1,
            2,
            3,
            1,
            2,
            3,
        ],
        "state_date": pd.to_datetime([
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-02-01",
            "2023-02-02",
            "2023-02-03",
        ]),
    })

    feature_values = pd.DataFrame(
        {
            feature: np.arange(
                len(
                    base
                ),
                dtype=float,
            )
            for feature
            in CONTROLLER_FEATURES
        }
    )

    sample = pd.concat(
        [
            base,
            feature_values,
        ],
        axis=1,
    )

    transitions = (
        prepare_transitions(
            sample
        )
    )

    assert transitions[
        "has_next_state"
    ].tolist() == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]

    next_X = build_next_matrix(
        transitions
    )
    first_feature = (
        CONTROLLER_FEATURES[
            0
        ]
    )
    assert next_X[
        first_feature
    ].iloc[
        0
    ] == 1.0
    assert pd.isna(
        next_X[
            first_feature
        ].iloc[
            2
        ]
    )

    print(
        "Controller FQI-v2 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Finite-horizon fitted-Q controller for "
            "HOLD versus EXIT on the frozen B4 "
            "large-cap selector."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--teacher-path",
        default=None,
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "histgb",
            "random_forest",
            "extra_trees",
        ],
        choices=[
            "histgb",
            "random_forest",
            "extra_trees",
        ],
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
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

    teacher_path = (
        Path(
            args.teacher_path
        ).resolve()
        if args.teacher_path
        is not None
        else None
    )

    summary = evaluate_fqi(
        root,
        teacher_path=(
            teacher_path
        ),
        models=list(
            args.models
        ),
        iterations=int(
            args.iterations
        ),
        random_state=int(
            args.random_state
        ),
    )

    print(
        "\n=== CONTROLLER FQI V2 COMPLETE ==="
    )
    print(
        f"Best model:     "
        f"{summary['best_model']}"
    )
    print(
        "Best threshold: "
        f"{summary['best_threshold']:+.4f}"
    )
    metrics = summary[
        "best_metrics"
    ]
    print(
        "Sequential gain vs 20d: "
        f"{float(metrics['mean_gain_vs_forced']):+.2%}"
    )
    print(
        "Oracle headroom captured: "
        f"{float(metrics['oracle_headroom_captured']):.1%}"
    )
    print(
        "Mean exit age:  "
        f"{float(metrics['mean_exit_age_sessions']):.1f} sessions"
    )


if __name__ == "__main__":
    main()
