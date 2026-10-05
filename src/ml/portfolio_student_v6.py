from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from portfolio_student_v5 import (
    RANDOM_STATE,
    EPS,
    add_relative_targets,
    feature_columns,
    ranking_metrics,
    random_expected_metrics,
    teacher_diagnostics,
    find_pairwise_feature_partition,
)


TRAIN_YEARS = [2021, 2022]
VALIDATION_YEARS = [2023]

CANDIDATES = [
    "pairwise_sgd_raw",
    "pairwise_sgd_scaled",
    "pairwise_sgd_topfocus",
    "lightgbm_lambdarank",
    "xgb_rank_pairwise",
    "xgb_rank_ndcg",
]


def make_sgd() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        max_iter=6,
        tol=None,
        random_state=RANDOM_STATE,
        average=True,
    )


def top_focus_factor(
    rank_a: int,
    rank_b: int,
) -> float:
    best = min(
        int(rank_a),
        int(rank_b),
    )
    worst = max(
        int(rank_a),
        int(rank_b),
    )

    if best == 1:
        return 5.0
    if best <= 5 and worst > 5:
        return 4.0
    if best <= 5:
        return 2.5
    if best <= 10 and worst > 10:
        return 2.0
    if best <= 10:
        return 1.2
    return 0.35


def build_pairwise_training(
    train: pd.DataFrame,
    x_train: np.ndarray,
    *,
    context_indices: np.ndarray,
    varying_indices: np.ndarray,
    max_pairs_per_state: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(
        RANDOM_STATE
    )
    groups = train.groupby(
        "state_key",
        sort=False,
    ).indices

    q_all = train[
        "oracle_q"
    ].to_numpy(
        dtype=float,
    )
    range_all = train[
        "state_q_range"
    ].to_numpy(
        dtype=float,
    )
    rank_all = train[
        "action_rank"
    ].to_numpy(
        dtype=int,
    )

    pair_specs: list[
        tuple[int, int]
    ] = []

    for _state_key, indices in tqdm(
        groups.items(),
        total=len(groups),
        desc="Enumerate V6 action pairs",
        unit="state",
        dynamic_ncols=True,
        leave=False,
    ):
        idx = np.asarray(
            indices,
            dtype=np.int64,
        )
        combinations = np.asarray(
            list(
                itertools.combinations(
                    idx.tolist(),
                    2,
                )
            ),
            dtype=np.int64,
        )

        if len(combinations) == 0:
            continue

        if (
            max_pairs_per_state > 0
            and len(combinations)
            > max_pairs_per_state
        ):
            keep = rng.choice(
                len(combinations),
                size=max_pairs_per_state,
                replace=False,
            )
            combinations = (
                combinations[
                    keep
                ]
            )

        for left, right in combinations:
            if abs(
                float(
                    q_all[left]
                    - q_all[right]
                )
            ) <= EPS:
                continue

            if rng.random() < 0.5:
                pair_specs.append(
                    (
                        int(left),
                        int(right),
                    )
                )
            else:
                pair_specs.append(
                    (
                        int(right),
                        int(left),
                    )
                )

    width = (
        len(context_indices)
        + len(varying_indices)
    )

    pair_x = np.empty(
        (
            len(pair_specs),
            width,
        ),
        dtype=np.float32,
    )
    pair_y = np.empty(
        len(pair_specs),
        dtype=np.int8,
    )
    base_weight = np.empty(
        len(pair_specs),
        dtype=np.float32,
    )
    top_weight = np.empty(
        len(pair_specs),
        dtype=np.float32,
    )

    context_width = len(
        context_indices
    )

    for row_number, (
        left,
        right,
    ) in tqdm(
        enumerate(pair_specs),
        total=len(pair_specs),
        desc="Materialize V6 pair matrix",
        unit="pair",
        dynamic_ncols=True,
        leave=False,
    ):
        pair_x[
            row_number,
            :context_width,
        ] = x_train[
            left,
            context_indices,
        ]
        pair_x[
            row_number,
            context_width:,
        ] = (
            x_train[
                left,
                varying_indices,
            ]
            - x_train[
                right,
                varying_indices,
            ]
        )

        left_q = float(
            q_all[
                left
            ]
        )
        right_q = float(
            q_all[
                right
            ]
        )

        pair_y[
            row_number
        ] = int(
            left_q
            > right_q
        )

        state_range = max(
            float(
                range_all[
                    left
                ]
            ),
            EPS,
        )
        normalized_gap = min(
            1.0,
            abs(
                left_q
                - right_q
            )
            / state_range,
        )

        base = (
            0.20
            + 0.80
            * normalized_gap
        )
        base_weight[
            row_number
        ] = base

        top_weight[
            row_number
        ] = (
            base
            * top_focus_factor(
                int(
                    rank_all[
                        left
                    ]
                ),
                int(
                    rank_all[
                        right
                    ]
                ),
            )
        )

    mean_top = float(
        np.mean(
            top_weight
        )
    )
    if mean_top > EPS:
        top_weight /= mean_top

    return (
        pair_x,
        pair_y,
        base_weight,
        top_weight,
    )


def fit_sgd_with_progress(
    model: SGDClassifier,
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray,
    desc: str,
    batch_size: int = 8192,
):
    epochs = max(
        1,
        int(
            model.max_iter
        ),
    )
    n_rows = int(
        len(
            y
        )
    )
    batches_per_epoch = int(
        math.ceil(
            n_rows
            / batch_size
        )
    )
    total_batches = (
        epochs
        * batches_per_epoch
    )

    classes = np.asarray(
        [
            0,
            1,
        ],
        dtype=np.int8,
    )
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    first = True

    with tqdm(
        total=total_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ) as bar:
        for epoch in range(
            epochs
        ):
            order = rng.permutation(
                n_rows
            )

            for start in range(
                0,
                n_rows,
                batch_size,
            ):
                batch_idx = order[
                    start:
                    start
                    + batch_size
                ]

                kwargs = {
                    "sample_weight": (
                        sample_weight[
                            batch_idx
                        ]
                    ),
                }

                if first:
                    model.partial_fit(
                        x[
                            batch_idx
                        ],
                        y[
                            batch_idx
                        ],
                        classes=classes,
                        **kwargs,
                    )
                    first = False
                else:
                    model.partial_fit(
                        x[
                            batch_idx
                        ],
                        y[
                            batch_idx
                        ],
                        **kwargs,
                    )

                bar.update(
                    1
                )
                bar.set_postfix(
                    epoch=(
                        f"{epoch + 1}/"
                        f"{epochs}"
                    ),
                    rows=(
                        f"{min(start + batch_size, n_rows):,}/"
                        f"{n_rows:,}"
                    ),
                    refresh=False,
                )

    return model


def pairwise_scores(
    frame: pd.DataFrame,
    x: np.ndarray,
    *,
    model,
    context_indices: np.ndarray,
    varying_indices: np.ndarray,
    scaler: StandardScaler | None,
    desc: str,
) -> np.ndarray:
    result = np.zeros(
        len(
            frame
        ),
        dtype=float,
    )
    context_width = len(
        context_indices
    )

    groups = frame.groupby(
        "state_key",
        sort=False,
    ).indices

    for _state_key, indices in tqdm(
        groups.items(),
        total=len(groups),
        desc=desc,
        unit="state",
        dynamic_ncols=True,
        leave=False,
    ):
        idx = np.asarray(
            indices,
            dtype=int,
        )

        if len(idx) == 1:
            result[
                idx[0]
            ] = 1.0
            continue

        local_x = x[
            idx
        ]

        pair_rows: list[
            np.ndarray
        ] = []
        owners: list[
            int
        ] = []

        context = local_x[
            0,
            context_indices,
        ]

        for local_i in range(
            len(
                idx
            )
        ):
            for local_j in range(
                len(
                    idx
                )
            ):
                if local_i == local_j:
                    continue

                row = np.empty(
                    context_width
                    + len(
                        varying_indices
                    ),
                    dtype=np.float32,
                )

                row[
                    :context_width
                ] = context

                row[
                    context_width:
                ] = (
                    local_x[
                        local_i,
                        varying_indices,
                    ]
                    - local_x[
                        local_j,
                        varying_indices,
                    ]
                )

                pair_rows.append(
                    row
                )
                owners.append(
                    local_i
                )

        matrix = np.vstack(
            pair_rows
        )

        if scaler is not None:
            matrix = scaler.transform(
                matrix
            )

        probabilities = (
            model.predict_proba(
                matrix
            )[
                :,
                1
            ]
        )

        local_scores = np.zeros(
            len(
                idx
            ),
            dtype=float,
        )
        local_counts = np.zeros(
            len(
                idx
            ),
            dtype=float,
        )

        for owner, probability in zip(
            owners,
            probabilities,
        ):
            local_scores[
                owner
            ] += float(
                probability
            )
            local_counts[
                owner
            ] += 1.0

        local_scores /= np.maximum(
            local_counts,
            1.0,
        )
        result[
            idx
        ] = local_scores

    return result


def prepare_ranker_data(
    frame: pd.DataFrame,
    x: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    order = (
        frame.sort_values(
            [
                "state_date",
                "state_key",
                "action_rank",
            ],
            kind="stable",
        )
        .index
        .to_numpy(
            dtype=int,
        )
    )

    ordered_frame = frame.loc[
        order
    ]

    qid, _uniques = (
        pd.factorize(
            ordered_frame[
                "state_key"
            ],
            sort=False,
        )
    )
    qid = np.asarray(
        qid,
        dtype=np.int32,
    )

    group_sizes = (
        ordered_frame.groupby(
            "state_key",
            sort=False,
        )
        .size()
        .astype(
            int
        )
        .tolist()
    )

    max_rank_by_state = (
        ordered_frame.groupby(
            "state_key",
            sort=False,
        )[
            "action_rank"
        ]
        .transform(
            "max"
        )
        .to_numpy(
            dtype=int,
        )
    )

    rank = ordered_frame[
        "action_rank"
    ].to_numpy(
        dtype=int,
    )

    raw_relevance = (
        max_rank_by_state
        - rank
    )
    relevance = (
        (
            raw_relevance
            + 1
        )
        // 2
    ).astype(
        np.int32,
        copy=False,
    )

    return (
        x[
            order
        ],
        relevance,
        qid,
        group_sizes,
    )


def fit_lightgbm_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    group_sizes: list[int],
    *,
    n_jobs: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is missing. Run: uv sync --extra tournament"
        ) from exc

    total = 500
    bar = tqdm(
        total=total,
        desc="Fit lightgbm_lambdarank",
        unit="tree",
        dynamic_ncols=True,
        leave=False,
    )

    def progress_callback(
        env,
    ):
        target = (
            int(
                env.iteration
            )
            + 1
        )
        delta = (
            target
            - bar.n
        )
        if delta > 0:
            bar.update(
                delta
            )

    progress_callback.order = 50

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=total,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_lambda=1.0,
        lambdarank_truncation_level=10,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbosity=-1,
    )

    try:
        model.fit(
            x_train,
            y_train,
            group=group_sizes,
            eval_at=(
                1,
                3,
                5,
                10,
            ),
            callbacks=[
                progress_callback,
                lgb.log_evaluation(
                    period=0
                ),
            ],
        )
    finally:
        bar.close()

    return model


def fit_xgb_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    qid_train: np.ndarray,
    *,
    objective: str,
    candidate_name: str,
    n_jobs: int,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is missing. Run: uv sync --extra tournament"
        ) from exc

    total = 500
    bar = tqdm(
        total=total,
        desc=f"Fit {candidate_name}",
        unit="tree",
        dynamic_ncols=True,
        leave=False,
    )

    class ProgressCallback(
        xgb.callback.TrainingCallback
    ):
        def after_iteration(
            self,
            model,
            epoch,
            evals_log,
        ):
            target = (
                int(
                    epoch
                )
                + 1
            )
            delta = (
                target
                - bar.n
            )
            if delta > 0:
                bar.update(
                    delta
                )
            return False

    model = xgb.XGBRanker(
        objective=objective,
        n_estimators=total,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        ndcg_exp_gain=True,
        callbacks=[
            ProgressCallback()
        ],
        verbosity=0,
    )

    try:
        model.fit(
            x_train,
            y_train,
            qid=qid_train,
            verbose=False,
        )
    finally:
        bar.close()

    # The progress callback closes over a tqdm object and is intentionally
    # local to this training call. Strip it after fitting so the trained
    # XGBRanker is safely serializable with joblib.
    model.set_params(
        callbacks=None,
    )
    model.callbacks = None

    return model


def evaluate(
    teacher: pd.DataFrame,
    *,
    root: Path,
    n_jobs: int,
    max_pairs_per_state: int,
    only: str | None,
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
        if any(
            token
            in feature.lower()
            for token in (
                "symbol",
                "security_id",
                "canonical",
                "ticker",
            )
        )
    ]
    if identity_leaks:
        raise RuntimeError(
            "Identity leaked into predictor columns: "
            f"{identity_leaks[:20]}"
        )

    teacher = add_relative_targets(
        teacher
    )

    train = (
        teacher.loc[
            teacher[
                "state_date"
            ].dt.year.isin(
                TRAIN_YEARS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation = (
        teacher.loc[
            teacher[
                "state_date"
            ].dt.year.isin(
                VALIDATION_YEARS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if train.empty or validation.empty:
        raise RuntimeError(
            "Expected non-empty 2021-2022 train and 2023 validation sets."
        )

    report_root = (
        root
        / "reports/ml/portfolio_student_v6/development"
    )
    model_root = (
        root
        / "data/processed/portfolio_student_v6"
    )
    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics = teacher_diagnostics(
        teacher
    )
    diagnostics.to_csv(
        report_root
        / "teacher_state_diagnostics.csv",
        index=False,
    )

    print(
        "\n=== PORTFOLIO STUDENT V6 ==="
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

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    with tqdm(
        total=2,
        desc="Impute feature matrices",
        unit="matrix",
        dynamic_ncols=True,
        leave=False,
    ) as bar:
        x_train = (
            imputer.fit_transform(
                train[
                    features
                ]
            )
            .astype(
                np.float32,
                copy=False,
            )
        )
        bar.update(
            1
        )

        x_validation = (
            imputer.transform(
                validation[
                    features
                ]
            )
            .astype(
                np.float32,
                copy=False,
            )
        )
        bar.update(
            1
        )

    selected_candidates = (
        [
            only
        ]
        if only
        is not None
        else list(
            CANDIDATES
        )
    )

    leaderboard_rows: list[
        dict
    ] = []
    state_results: list[
        pd.DataFrame
    ] = []
    model_bundles: dict[
        str,
        dict
    ] = {}

    random_metrics = (
        random_expected_metrics(
            validation
        )
    )

    print(
        "\nRandom expected: "
        f"rank="
        f"{random_metrics['mean_oracle_rank_selected']:.2f} "
        f"top5="
        f"{random_metrics['top5_hit_fraction']:.1%} "
        f"regret="
        f"{random_metrics['mean_action_regret']:.5f} "
        f"norm="
        f"{random_metrics['mean_normalized_regret']:.3f}"
    )

    leaderboard_rows.append(
        {
            "candidate": (
                "random_expected"
            ),
            "family": "random",
            **random_metrics,
        }
    )

    partial_leaderboard = (
        report_root
        / "leaderboard_partial.csv"
    )
    partial_states = (
        report_root
        / "state_ranking_results_partial.parquet"
    )

    if (
        only is not None
        and partial_leaderboard.is_file()
    ):
        previous = pd.read_csv(
            partial_leaderboard
        )
        previous = previous.loc[
            previous[
                "candidate"
            ].ne(
                only
            )
        ]
        leaderboard_rows = (
            previous.to_dict(
                "records"
            )
        )
        if not any(
            str(
                row.get(
                    "candidate"
                )
            )
            == "random_expected"
            for row
            in leaderboard_rows
        ):
            leaderboard_rows.append(
                {
                    "candidate": (
                        "random_expected"
                    ),
                    "family": "random",
                    **random_metrics,
                }
            )
        print(
            "Loaded "
            f"{len(previous):,} checkpointed leaderboard rows."
        )

    if (
        only is not None
        and partial_states.is_file()
    ):
        previous_states = pd.read_parquet(
            partial_states
        )
        previous_states = (
            previous_states.loc[
                previous_states[
                    "candidate"
                ].ne(
                    only
                )
            ]
        )
        if not previous_states.empty:
            state_results.append(
                previous_states
            )
        print(
            "Loaded "
            f"{len(previous_states):,} checkpointed state rows."
        )

    def record(
        *,
        candidate: str,
        family: str,
        scores: np.ndarray,
        bundle: dict,
    ) -> None:
        metrics, rows = (
            ranking_metrics(
                validation,
                scores,
                progress_desc=(
                    f"Evaluate "
                    f"{candidate}"
                ),
            )
        )

        leaderboard_rows.append(
            {
                "candidate": (
                    candidate
                ),
                "family": (
                    family
                ),
                **metrics,
            }
        )

        rows[
            "candidate"
        ] = candidate
        state_results.append(
            rows
        )
        model_bundles[
            candidate
        ] = bundle

        # Metrics/state evidence are more important than model persistence.
        # Checkpoint them first so a serialization problem can never erase
        # a completed experiment.
        pd.DataFrame(
            leaderboard_rows
        ).drop_duplicates(
            subset=[
                "candidate"
            ],
            keep="last",
        ).to_csv(
            partial_leaderboard,
            index=False,
        )

        pd.concat(
            state_results,
            ignore_index=True,
        ).drop_duplicates(
            subset=[
                "candidate",
                "state_key",
            ],
            keep="last",
        ).to_parquet(
            partial_states,
            index=False,
            compression="zstd",
        )

        candidate_model_path = (
            model_root
            / f"{candidate}.joblib"
        )
        try:
            joblib.dump(
                bundle,
                candidate_model_path,
                compress=3,
            )
        except Exception as exc:
            print(
                "\nWARNING: metrics were checkpointed, but model "
                f"serialization failed for {candidate}: {exc}"
            )

        tournament_bar.update(
            1
        )

        print(
            f"{candidate:26s} "
            f"rank="
            f"{metrics['mean_oracle_rank_selected']:.2f} "
            f"top5="
            f"{metrics['top5_hit_fraction']:.1%} "
            f"regret="
            f"{metrics['mean_action_regret']:.5f} "
            f"norm="
            f"{metrics['mean_normalized_regret']:.3f} "
            f"rho="
            f"{metrics['mean_within_state_spearman']:+.3f} "
            f"ndcg5="
            f"{metrics['mean_ndcg_at_5']:.3f}"
        )

    pairwise_names = {
        "pairwise_sgd_raw",
        "pairwise_sgd_scaled",
        "pairwise_sgd_topfocus",
    }
    need_pairwise = any(
        candidate
        in pairwise_names
        for candidate
        in selected_candidates
    )

    pair_x = None
    pair_y = None
    base_pair_weight = None
    top_pair_weight = None
    pair_scaler = None
    context_indices = None
    varying_indices = None

    if need_pairwise:
        (
            context_indices,
            varying_indices,
        ) = (
            find_pairwise_feature_partition(
                train,
                x_train,
                features,
            )
        )

        print(
            "\nPairwise feature partition: "
            f"{len(context_indices)} context + "
            f"{len(varying_indices)} action-varying"
        )

        (
            pair_x,
            pair_y,
            base_pair_weight,
            top_pair_weight,
        ) = build_pairwise_training(
            train,
            x_train,
            context_indices=(
                context_indices
            ),
            varying_indices=(
                varying_indices
            ),
            max_pairs_per_state=(
                max_pairs_per_state
            ),
        )

        print(
            "Pairwise training rows: "
            f"{len(pair_x):,} × "
            f"{pair_x.shape[1]:,} features"
        )
        print(
            "Pairwise class balance: "
            f"{float(pair_y.mean()):.1%} positive"
        )

    tournament_bar = tqdm(
        total=len(
            selected_candidates
        ),
        desc="V6 tournament",
        unit="model",
        dynamic_ncols=True,
    )

    if (
        "pairwise_sgd_raw"
        in selected_candidates
    ):
        model = make_sgd()
        fit_sgd_with_progress(
            model,
            pair_x,
            pair_y,
            sample_weight=(
                base_pair_weight
            ),
            desc=(
                "Fit pairwise_sgd_raw"
            ),
        )
        scores = pairwise_scores(
            validation,
            x_validation,
            model=model,
            context_indices=(
                context_indices
            ),
            varying_indices=(
                varying_indices
            ),
            scaler=None,
            desc=(
                "Score pairwise_sgd_raw"
            ),
        )
        record(
            candidate=(
                "pairwise_sgd_raw"
            ),
            family="pairwise_sgd",
            scores=scores,
            bundle={
                "candidate": (
                    "pairwise_sgd_raw"
                ),
                "model": model,
                "imputer": imputer,
                "scaler": None,
                "features": features,
                "context_indices": (
                    context_indices
                ),
                "varying_indices": (
                    varying_indices
                ),
            },
        )

    if (
        "pairwise_sgd_scaled"
        in selected_candidates
        or
        "pairwise_sgd_topfocus"
        in selected_candidates
    ):
        pair_scaler = StandardScaler(
            copy=False,
        )

        with tqdm(
            total=2,
            desc="Scale pairwise matrix",
            unit="pass",
            dynamic_ncols=True,
            leave=False,
        ) as bar:
            pair_scaler.fit(
                pair_x
            )
            bar.update(
                1
            )
            pair_scaler.transform(
                pair_x,
                copy=False,
            )
            bar.update(
                1
            )

    if (
        "pairwise_sgd_scaled"
        in selected_candidates
    ):
        model = make_sgd()
        fit_sgd_with_progress(
            model,
            pair_x,
            pair_y,
            sample_weight=(
                base_pair_weight
            ),
            desc=(
                "Fit pairwise_sgd_scaled"
            ),
        )
        scores = pairwise_scores(
            validation,
            x_validation,
            model=model,
            context_indices=(
                context_indices
            ),
            varying_indices=(
                varying_indices
            ),
            scaler=(
                pair_scaler
            ),
            desc=(
                "Score pairwise_sgd_scaled"
            ),
        )
        record(
            candidate=(
                "pairwise_sgd_scaled"
            ),
            family="pairwise_sgd",
            scores=scores,
            bundle={
                "candidate": (
                    "pairwise_sgd_scaled"
                ),
                "model": model,
                "imputer": imputer,
                "scaler": (
                    pair_scaler
                ),
                "features": features,
                "context_indices": (
                    context_indices
                ),
                "varying_indices": (
                    varying_indices
                ),
            },
        )

    if (
        "pairwise_sgd_topfocus"
        in selected_candidates
    ):
        model = make_sgd()
        fit_sgd_with_progress(
            model,
            pair_x,
            pair_y,
            sample_weight=(
                top_pair_weight
            ),
            desc=(
                "Fit pairwise_sgd_topfocus"
            ),
        )
        scores = pairwise_scores(
            validation,
            x_validation,
            model=model,
            context_indices=(
                context_indices
            ),
            varying_indices=(
                varying_indices
            ),
            scaler=(
                pair_scaler
            ),
            desc=(
                "Score pairwise_sgd_topfocus"
            ),
        )
        record(
            candidate=(
                "pairwise_sgd_topfocus"
            ),
            family="pairwise_sgd",
            scores=scores,
            bundle={
                "candidate": (
                    "pairwise_sgd_topfocus"
                ),
                "model": model,
                "imputer": imputer,
                "scaler": (
                    pair_scaler
                ),
                "features": features,
                "context_indices": (
                    context_indices
                ),
                "varying_indices": (
                    varying_indices
                ),
                "top_focus_weighting": (
                    "rank1=5x; top5-cross=4x; "
                    "top5-internal=2.5x; "
                    "top10-cross=2x; "
                    "top10-internal=1.2x; "
                    "middle=0.35x; multiplied by normalized Q-gap weight"
                ),
            },
        )

    ranker_names = {
        "lightgbm_lambdarank",
        "xgb_rank_pairwise",
        "xgb_rank_ndcg",
    }
    need_ranker = any(
        candidate
        in ranker_names
        for candidate
        in selected_candidates
    )

    if need_ranker:
        (
            rank_x_train,
            rank_y_train,
            rank_qid_train,
            rank_group_train,
        ) = prepare_ranker_data(
            train,
            x_train,
        )

    if (
        "lightgbm_lambdarank"
        in selected_candidates
    ):
        model = fit_lightgbm_ranker(
            rank_x_train,
            rank_y_train,
            rank_group_train,
            n_jobs=n_jobs,
        )
        scores = model.predict(
            x_validation
        )
        record(
            candidate=(
                "lightgbm_lambdarank"
            ),
            family="lambdarank",
            scores=scores,
            bundle={
                "candidate": (
                    "lightgbm_lambdarank"
                ),
                "model": model,
                "imputer": imputer,
                "features": features,
                "train_grouping": (
                    "state_key query groups"
                ),
                "relevance": (
                    "ceil((max_action_rank - action_rank) / 2), "
                    "graded 0..16"
                ),
            },
        )

    if (
        "xgb_rank_pairwise"
        in selected_candidates
    ):
        model = fit_xgb_ranker(
            rank_x_train,
            rank_y_train,
            rank_qid_train,
            objective=(
                "rank:pairwise"
            ),
            candidate_name=(
                "xgb_rank_pairwise"
            ),
            n_jobs=n_jobs,
        )
        scores = model.predict(
            x_validation
        )
        record(
            candidate=(
                "xgb_rank_pairwise"
            ),
            family="xgboost_ranker",
            scores=scores,
            bundle={
                "candidate": (
                    "xgb_rank_pairwise"
                ),
                "model": model,
                "imputer": imputer,
                "features": features,
                "train_grouping": (
                    "state_key qid"
                ),
                "relevance": (
                    "ceil((max_action_rank - action_rank) / 2), "
                    "graded 0..16"
                ),
                "objective": (
                    "rank:pairwise"
                ),
                "topk_pairs": 8,
            },
        )

    if (
        "xgb_rank_ndcg"
        in selected_candidates
    ):
        model = fit_xgb_ranker(
            rank_x_train,
            rank_y_train,
            rank_qid_train,
            objective=(
                "rank:ndcg"
            ),
            candidate_name=(
                "xgb_rank_ndcg"
            ),
            n_jobs=n_jobs,
        )
        scores = model.predict(
            x_validation
        )
        record(
            candidate=(
                "xgb_rank_ndcg"
            ),
            family="xgboost_ranker",
            scores=scores,
            bundle={
                "candidate": (
                    "xgb_rank_ndcg"
                ),
                "model": model,
                "imputer": imputer,
                "features": features,
                "train_grouping": (
                    "state_key qid"
                ),
                "relevance": (
                    "ceil((max_action_rank - action_rank) / 2), "
                    "graded 0..16"
                ),
                "objective": (
                    "rank:ndcg"
                ),
                "topk_pairs": 8,
            },
        )

    tournament_bar.close()

    leaderboard = pd.DataFrame(
        leaderboard_rows
    ).drop_duplicates(
        subset=[
            "candidate"
        ],
        keep="last",
    )

    learned = (
        leaderboard.loc[
            leaderboard[
                "family"
            ].ne(
                "random"
            )
        ]
        .sort_values(
            [
                "mean_normalized_regret",
                "mean_action_regret",
                "mean_oracle_rank_selected",
                "top5_hit_fraction",
                "mean_ndcg_at_5",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    learned[
        "development_rank"
    ] = np.arange(
        1,
        len(
            learned
        )
        + 1,
    )

    leaderboard = (
        leaderboard.merge(
            learned[
                [
                    "candidate",
                    "development_rank",
                ]
            ],
            on="candidate",
            how="left",
        )
        .sort_values(
            [
                "development_rank",
                "candidate",
            ],
            na_position="last",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    leaderboard.to_csv(
        report_root
        / "leaderboard.csv",
        index=False,
    )

    pd.concat(
        state_results,
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "candidate",
            "state_key",
        ],
        keep="last",
    ).to_parquet(
        report_root
        / "state_ranking_results.parquet",
        index=False,
        compression="zstd",
    )

    best_candidate = str(
        learned.iloc[
            0
        ][
            "candidate"
        ]
    )
    best_bundle = model_bundles.get(
        best_candidate
    )
    best_candidate_path = (
        model_root
        / f"{best_candidate}.joblib"
    )
    if (
        best_bundle is None
        and best_candidate_path.is_file()
    ):
        best_bundle = joblib.load(
            best_candidate_path
        )

    if best_bundle is None:
        raise RuntimeError(
            "Best candidate model bundle is unavailable: "
            f"{best_candidate}"
        )

    selection_rule = (
        "lowest 2023 mean normalized regret; then lower mean action "
        "regret; lower mean selected oracle rank; higher top-5 hit; "
        "higher NDCG@5"
    )

    best_bundle.update(
        {
            "train_years": (
                TRAIN_YEARS
            ),
            "validation_years": (
                VALIDATION_YEARS
            ),
            "selection_rule": (
                selection_rule
            ),
        }
    )

    best_model_path = (
        model_root
        / "development_best.joblib"
    )
    try:
        joblib.dump(
            best_bundle,
            best_model_path,
            compress=3,
        )
    except Exception as exc:
        print(
            "\nWARNING: final best-model serialization failed, "
            "but all leaderboard/state results are preserved: "
            f"{exc}"
        )

    summary = {
        "teacher": (
            "portfolio_oracle_v4"
        ),
        "student_version": "v6",
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
        "candidates": (
            selected_candidates
        ),
        "pairwise_training_rows": (
            int(
                len(
                    pair_y
                )
            )
            if need_pairwise
            else None
        ),
        "pairwise_context_feature_count": (
            int(
                len(
                    context_indices
                )
            )
            if need_pairwise
            else None
        ),
        "pairwise_varying_feature_count": (
            int(
                len(
                    varying_indices
                )
            )
            if need_pairwise
            else None
        ),
        "random_expected": (
            random_metrics
        ),
        "selection_rule": (
            selection_rule
        ),
        "best_candidate": (
            best_candidate
        ),
        "best_metrics": (
            learned.iloc[
                0
            ].to_dict()
        ),
        "model_path": str(
            best_model_path
        ),
        "confirmation_rule": (
            "Do not use 2024-2026 oracle labels or retune on 2024-2026 "
            "until the development policy/action representation is frozen."
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
        "\n=== PORTFOLIO STUDENT V6 COMPLETE ==="
    )
    print(
        "\nLeaderboard:"
    )
    print(
        leaderboard[
            [
                "candidate",
                "mean_oracle_rank_selected",
                "top5_hit_fraction",
                "mean_action_regret",
                "mean_normalized_regret",
                "mean_within_state_spearman",
                "mean_ndcg_at_5",
                "development_rank",
            ]
        ].to_string(
            index=False
        )
    )
    print(
        "\nBest candidate: "
        f"{best_candidate}"
    )
    print(
        "Model: "
        f"{best_model_path}"
    )
    print(
        "Reports: "
        f"{report_root}"
    )

    return summary


def self_test() -> None:
    assert (
        top_focus_factor(
            1,
            32,
        )
        > top_focus_factor(
            6,
            30,
        )
        > top_focus_factor(
            20,
            30,
        )
    )

    toy = pd.DataFrame(
        {
            "state_key": (
                [
                    "a",
                ]
                * 4
                + [
                    "b",
                ]
                * 4
            ),
            "state_date": (
                pd.to_datetime(
                    [
                        "2022-01-01",
                    ]
                    * 4
                    + [
                        "2022-01-02",
                    ]
                    * 4
                )
            ),
            "oracle_q": [
                0.4,
                0.3,
                0.2,
                0.1,
                0.8,
                0.4,
                0.3,
                0.0,
            ],
            "action_rank": [
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                4,
            ],
        }
    )

    x = np.arange(
        8
        * 3,
        dtype=np.float32,
    ).reshape(
        8,
        3,
    )

    rank_x, relevance, qid, groups = (
        prepare_ranker_data(
            toy,
            x,
        )
    )

    assert rank_x.shape == (
        8,
        3,
    )
    assert relevance.tolist() == [
        2,
        1,
        1,
        0,
        2,
        1,
        1,
        0,
    ]
    assert qid.tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert groups == [
        4,
        4,
    ]

    print(
        "Portfolio-student-v6 self-test: PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V6 ranking tournament on the frozen portfolio-oracle teacher: "
            "raw/scaled/top-focused pairwise SGD plus LambdaRank/XGBoost."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
    )
    parser.add_argument(
        "--teacher-path",
        default=None,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max-pairs-per-state",
        type=int,
        default=496,
    )
    parser.add_argument(
        "--only",
        default=None,
        choices=CANDIDATES,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    args = parser.parse_args()

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
            "Missing oracle teacher: "
            f"{teacher_path}"
        )

    print(
        "CPU threads visible: "
        f"{os.cpu_count()}"
    )
    print(
        "Teacher: "
        f"{teacher_path}"
    )

    teacher = pd.read_parquet(
        teacher_path
    )

    evaluate(
        teacher,
        root=root,
        n_jobs=args.n_jobs,
        max_pairs_per_state=(
            args.max_pairs_per_state
        ),
        only=args.only,
    )


if __name__ == "__main__":
    main()
