from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


TRAIN_YEARS = [2021, 2022]
VALIDATION_YEARS = [2023]
RANDOM_STATE = 242


def model_spec(
    name: str,
    *,
    random_state: int,
):
    if name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=8,
            max_features=0.6,
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=6,
            max_features=0.7,
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "histgb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=350,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown model: {name}"
        )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "model",
            model,
        ),
    ])


def feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        str(column)
        for column
        in frame.columns
        if str(column).startswith(
            "f__"
        )
    ]


def ranking_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    work = frame[
        [
            "state_key",
            "state_date",
            "oracle_q",
            "action_rank",
            "is_oracle_action",
            "meta_target_signature",
        ]
    ].copy()
    work[
        "predicted_q"
    ] = np.asarray(
        predictions,
        dtype=float,
    )

    rows = []

    for state_key, group in work.groupby(
        "state_key",
        sort=False,
    ):
        ordered_pred = (
            group.sort_values(
                [
                    "predicted_q",
                    "oracle_q",
                    "meta_target_signature",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
        )
        selected = ordered_pred.iloc[
            0
        ]
        oracle_best = float(
            group[
                "oracle_q"
            ].max()
        )
        selected_value = float(
            selected[
                "oracle_q"
            ]
        )
        regret = (
            oracle_best
            - selected_value
        )

        oracle_signatures = set(
            group.loc[
                group[
                    "oracle_q"
                ].eq(
                    oracle_best
                ),
                "meta_target_signature",
            ].astype(
                str
            )
        )

        rows.append({
            "state_key": (
                state_key
            ),
            "state_date": (
                selected[
                    "state_date"
                ]
            ),
            "selected_signature": str(
                selected[
                    "meta_target_signature"
                ]
            ),
            "selected_oracle_q": (
                selected_value
            ),
            "best_oracle_q": (
                oracle_best
            ),
            "regret": float(
                regret
            ),
            "exact_best_action": bool(
                str(
                    selected[
                        "meta_target_signature"
                    ]
                )
                in oracle_signatures
            ),
            "selected_oracle_rank": int(
                selected[
                    "action_rank"
                ]
            ),
        })

    result = pd.DataFrame(
        rows
    )

    metrics = {
        "states": int(
            len(
                result
            )
        ),
        "exact_best_action_fraction": float(
            result[
                "exact_best_action"
            ].mean()
        ),
        "mean_oracle_rank_selected": float(
            result[
                "selected_oracle_rank"
            ].mean()
        ),
        "median_oracle_rank_selected": float(
            result[
                "selected_oracle_rank"
            ].median()
        ),
        "mean_action_regret": float(
            result[
                "regret"
            ].mean()
        ),
        "median_action_regret": float(
            result[
                "regret"
            ].median()
        ),
        "p90_action_regret": float(
            result[
                "regret"
            ].quantile(
                0.90
            )
        ),
        "nonpositive_regret_fraction": float(
            result[
                "regret"
            ].le(
                1e-12
            ).mean()
        ),
    }

    return metrics, result


def evaluate(
    teacher: pd.DataFrame,
    *,
    models: list[str],
    root: Path,
) -> dict:
    teacher = teacher.copy()
    teacher[
        "state_date"
    ] = pd.to_datetime(
        teacher[
            "state_date"
        ],
        errors="coerce",
    ).dt.normalize()

    features = feature_columns(
        teacher
    )
    if not features:
        raise RuntimeError(
            "Teacher has no f__ predictor columns."
        )

    identity_leaks = [
        feature
        for feature in features
        if (
            "symbol"
            in feature.lower()
            or "security_id"
            in feature.lower()
            or "canonical"
            in feature.lower()
            or "ticker"
            in feature.lower()
        )
    ]
    if identity_leaks:
        raise RuntimeError(
            "Identity leaked into predictor columns: "
            f"{identity_leaks[:20]}"
        )

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

    if train.empty:
        raise RuntimeError(
            "Training teacher is empty."
        )
    if validation.empty:
        raise RuntimeError(
            "2023 validation teacher is empty. "
            "Build the oracle on 2021-2023 before model selection."
        )

    report_root = (
        root
        / "reports/ml/"
        "portfolio_student_v4/"
        "development"
    )
    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_root = (
        root
        / "data/processed/"
        "portfolio_student_v4"
    )
    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard_rows = []
    ranking_frames = []
    prediction_frames = []

    print(
        "\n=== PORTFOLIO STUDENT V4 ==="
    )
    print(
        "Train states:      "
        f"{train['state_key'].nunique():,} "
        f"({TRAIN_YEARS})"
    )
    print(
        "Train actions:     "
        f"{len(train):,}"
    )
    print(
        "Validation states: "
        f"{validation['state_key'].nunique():,} "
        f"({VALIDATION_YEARS})"
    )
    print(
        "Validation actions:"
        f" {len(validation):,}"
    )
    print(
        "Predictors:        "
        f"{len(features):,}"
    )

    fitted = {}

    for number, name in enumerate(
        models,
        start=1,
    ):
        print(
            f"\n=== {name} ==="
        )

        model = model_spec(
            name,
            random_state=(
                RANDOM_STATE
                + number
                * 100
            ),
        )
        model.fit(
            train[
                features
            ],
            train[
                "oracle_q"
            ].astype(
                float
            ),
        )

        pred = model.predict(
            validation[
                features
            ]
        )
        actual = validation[
            "oracle_q"
        ].to_numpy(
            dtype=float
        )

        mae = float(
            mean_absolute_error(
                actual,
                pred,
            )
        )
        rmse = float(
            math.sqrt(
                mean_squared_error(
                    actual,
                    pred,
                )
            )
        )
        r2 = float(
            r2_score(
                actual,
                pred,
            )
        )
        spearman = float(
            pd.Series(
                actual
            ).corr(
                pd.Series(
                    pred
                ),
                method="spearman",
            )
        )

        rank_metrics, rank_rows = (
            ranking_metrics(
                validation,
                pred,
            )
        )

        leaderboard_rows.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman": spearman,
            **rank_metrics,
        })

        rank_rows[
            "model"
        ] = name
        ranking_frames.append(
            rank_rows
        )

        prediction_frame = validation[
            [
                "state_key",
                "state_date",
                "action_rank",
                "is_oracle_action",
                "oracle_q",
                "oracle_terminal_return",
                "oracle_max_drawdown",
                "oracle_turnover",
                "meta_target_signature",
            ]
        ].copy()
        prediction_frame[
            "predicted_q"
        ] = pred
        prediction_frame[
            "model"
        ] = name
        prediction_frames.append(
            prediction_frame
        )

        fitted[
            name
        ] = model

        print(
            "  mean-regret="
            f"{rank_metrics['mean_action_regret']:.6f} "
            "exact-best="
            f"{rank_metrics['exact_best_action_fraction']:.1%} "
            "rank="
            f"{rank_metrics['mean_oracle_rank_selected']:.2f} "
            "Spearman="
            f"{spearman:+.3f}"
        )

    leaderboard = (
        pd.DataFrame(
            leaderboard_rows
        )
        .sort_values(
            [
                "mean_action_regret",
                "mean_oracle_rank_selected",
                "mae",
                "spearman",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
            kind="stable",
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

    best_name = str(
        leaderboard.iloc[
            0
        ][
            "model"
        ]
    )
    best_model = fitted[
        best_name
    ]

    leaderboard.to_csv(
        report_root
        / "leaderboard.csv",
        index=False,
    )
    pd.concat(
        ranking_frames,
        ignore_index=True,
    ).to_parquet(
        report_root
        / "state_ranking_results.parquet",
        index=False,
        compression="zstd",
    )
    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_parquet(
        report_root
        / "action_value_predictions.parquet",
        index=False,
        compression="zstd",
    )

    model_path = (
        model_root
        / "development_best.joblib"
    )
    joblib.dump(
        {
            "model": (
                best_model
            ),
            "features": (
                features
            ),
            "model_name": (
                best_name
            ),
            "train_years": (
                TRAIN_YEARS
            ),
            "validation_years": (
                VALIDATION_YEARS
            ),
        },
        model_path,
    )

    underlying = (
        best_model.named_steps[
            "model"
        ]
    )
    if hasattr(
        underlying,
        "feature_importances_",
    ):
        importance = pd.DataFrame({
            "feature": features,
            "importance": np.asarray(
                underlying.feature_importances_,
                dtype=float,
            ),
        }).sort_values(
            "importance",
            ascending=False,
            kind="stable",
        )
        importance.to_csv(
            report_root
            / "feature_importance.csv",
            index=False,
        )

    best = leaderboard.iloc[
        0
    ]
    summary = {
        "teacher": (
            "portfolio_oracle_v4"
        ),
        "target": (
            "oracle_q for anonymized whole-portfolio actions"
        ),
        "train_years": (
            TRAIN_YEARS
        ),
        "validation_years": (
            VALIDATION_YEARS
        ),
        "feature_count": int(
            len(
                features
            )
        ),
        "identity_features": [],
        "models": models,
        "selection_rule": (
            "lowest 2023 mean oracle action regret; then lower mean "
            "oracle rank selected; lower MAE; higher Spearman"
        ),
        "best_model": (
            best_name
        ),
        "best_metrics": (
            best.to_dict()
        ),
        "model_path": str(
            model_path
        ),
        "confirmation_rule": (
            "Do not build oracle labels or retune on 2024-2026 until the "
            "development policy and action-search procedure are frozen."
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

    print(
        "\n=== PORTFOLIO STUDENT V4 COMPLETE ==="
    )
    print(
        "Best model: "
        f"{best_name}"
    )
    print(
        "Mean regret: "
        f"{float(best['mean_action_regret']):.6f}"
    )
    print(
        "Exact best:  "
        f"{float(best['exact_best_action_fraction']):.1%}"
    )
    print(
        "Model:       "
        f"{model_path}"
    )

    return summary


def self_test() -> None:
    toy = pd.DataFrame({
        "state_key": [
            "a",
            "a",
            "b",
            "b",
        ],
        "state_date": pd.to_datetime([
            "2023-01-01",
            "2023-01-01",
            "2023-01-02",
            "2023-01-02",
        ]),
        "oracle_q": [
            0.1,
            0.2,
            -0.1,
            0.0,
        ],
        "action_rank": [
            2,
            1,
            2,
            1,
        ],
        "is_oracle_action": [
            False,
            True,
            False,
            True,
        ],
        "meta_target_signature": [
            "x",
            "y",
            "u",
            "v",
        ],
    })
    metrics, rows = (
        ranking_metrics(
            toy,
            np.asarray([
                0.0,
                1.0,
                0.0,
                1.0,
            ]),
        )
    )
    assert len(rows) == 2
    assert (
        metrics[
            "exact_best_action_fraction"
        ]
        == 1.0
    )
    assert (
        metrics[
            "mean_action_regret"
        ]
        == 0.0
    )

    print(
        "Portfolio-student-v4 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train interpretable tree ensembles to rank anonymized "
            "whole-portfolio actions produced by portfolio_oracle_v4."
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
            "random_forest",
            "extra_trees",
            "histgb",
        ],
        choices=[
            "random_forest",
            "extra_trees",
            "histgb",
        ],
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
        else (
            root
            / "data/processed/"
            "portfolio_oracle_v4/"
            "development_teacher.parquet"
        )
    )

    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing oracle teacher: {teacher_path}"
        )

    teacher = pd.read_parquet(
        teacher_path
    )

    evaluate(
        teacher,
        models=list(
            args.models
        ),
        root=root,
    )


if __name__ == "__main__":
    main()
