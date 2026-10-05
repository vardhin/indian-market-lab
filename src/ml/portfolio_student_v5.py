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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer


TRAIN_YEARS = [2021, 2022]
VALIDATION_YEARS = [2023]
RANDOM_STATE = 242
EPS = 1e-12


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column).startswith("f__")
    ]


def add_relative_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby("state_key", sort=False)["oracle_q"]
    frame["state_best_q"] = grouped.transform("max").astype(float)
    frame["state_worst_q"] = grouped.transform("min").astype(float)
    frame["state_q_range"] = (
        frame["state_best_q"] - frame["state_worst_q"]
    )
    frame["oracle_regret"] = (
        frame["state_best_q"] - frame["oracle_q"].astype(float)
    )
    frame["oracle_normalized_regret"] = np.where(
        frame["state_q_range"].to_numpy(dtype=float) > EPS,
        frame["oracle_regret"].to_numpy(dtype=float)
        / frame["state_q_range"].to_numpy(dtype=float),
        0.0,
    )
    return frame


def teacher_diagnostics(teacher: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for state_key, group in teacher.groupby("state_key", sort=False):
        q = np.sort(
            group["oracle_q"].to_numpy(dtype=float)
        )[::-1]
        q_best = float(q[0])
        q_second = float(q[1]) if len(q) >= 2 else q_best
        q_fifth = float(q[min(4, len(q) - 1)])
        q_median = float(np.median(q))
        q_worst = float(q[-1])
        q_range = float(q_best - q_worst)

        if q_range > EPS:
            near_1pct = int(
                np.sum((q_best - q) <= 0.01 * q_range + EPS)
            )
            near_5pct = int(
                np.sum((q_best - q) <= 0.05 * q_range + EPS)
            )
            near_10pct = int(
                np.sum((q_best - q) <= 0.10 * q_range + EPS)
            )
        else:
            near_1pct = len(q)
            near_5pct = len(q)
            near_10pct = len(q)

        rows.append(
            {
                "state_key": state_key,
                "state_date": pd.to_datetime(
                    group["state_date"].iloc[0]
                ).normalize(),
                "actions": int(len(group)),
                "q_best": q_best,
                "q_second": q_second,
                "q_fifth": q_fifth,
                "q_median": q_median,
                "q_worst": q_worst,
                "q_range": q_range,
                "q_std": float(np.std(q)),
                "gap_best_second": float(q_best - q_second),
                "gap_best_fifth": float(q_best - q_fifth),
                "gap_best_median": float(q_best - q_median),
                "near_optimal_1pct_range_count": near_1pct,
                "near_optimal_5pct_range_count": near_5pct,
                "near_optimal_10pct_range_count": near_10pct,
            }
        )

    return pd.DataFrame(rows)


def ndcg_at_k(
    actual_q: np.ndarray,
    predicted_score: np.ndarray,
    k: int,
) -> float:
    q = np.asarray(actual_q, dtype=float)
    scores = np.asarray(predicted_score, dtype=float)
    if len(q) == 0:
        return float("nan")

    q_min = float(np.min(q))
    q_max = float(np.max(q))
    if q_max - q_min <= EPS:
        return 1.0

    relevance = (q - q_min) / (q_max - q_min)
    k = min(int(k), len(q))
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))

    predicted_order = np.argsort(-scores, kind="stable")[:k]
    ideal_order = np.argsort(-relevance, kind="stable")[:k]

    dcg = float(
        np.sum(relevance[predicted_order] * discounts)
    )
    idcg = float(
        np.sum(relevance[ideal_order] * discounts)
    )
    return dcg / idcg if idcg > EPS else 1.0


def ranking_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    work = frame[
        [
            "state_key",
            "state_date",
            "oracle_q",
            "action_rank",
            "meta_target_signature",
            "state_best_q",
            "state_worst_q",
            "state_q_range",
            "oracle_regret",
            "oracle_normalized_regret",
        ]
    ].copy()
    work["predicted_score"] = np.asarray(scores, dtype=float)

    rows: list[dict] = []

    for state_key, group in work.groupby("state_key", sort=False):
        group = group.copy()
        score_values = group["predicted_score"].to_numpy(dtype=float)
        q_values = group["oracle_q"].to_numpy(dtype=float)

        order = np.argsort(-score_values, kind="stable")
        selected = group.iloc[int(order[0])]

        best_q = float(np.max(q_values))
        selected_q = float(selected["oracle_q"])
        regret = float(best_q - selected_q)
        q_range = float(
            selected["state_q_range"]
        )
        normalized_regret = (
            regret / q_range
            if q_range > EPS
            else 0.0
        )

        if (
            np.std(q_values) > EPS
            and np.std(score_values) > EPS
        ):
            within_spearman = float(
                pd.Series(q_values).corr(
                    pd.Series(score_values),
                    method="spearman",
                )
            )
        else:
            within_spearman = 0.0

        selected_rank = int(selected["action_rank"])

        rows.append(
            {
                "state_key": state_key,
                "state_date": selected["state_date"],
                "selected_signature": str(
                    selected["meta_target_signature"]
                ),
                "selected_oracle_q": selected_q,
                "best_oracle_q": best_q,
                "regret": regret,
                "normalized_regret": normalized_regret,
                "exact_best_action": bool(
                    regret <= EPS
                ),
                "top3_hit": bool(selected_rank <= 3),
                "top5_hit": bool(selected_rank <= 5),
                "selected_oracle_rank": selected_rank,
                "within_state_spearman": within_spearman,
                "ndcg_at_5": ndcg_at_k(
                    q_values,
                    score_values,
                    5,
                ),
                "ndcg_at_10": ndcg_at_k(
                    q_values,
                    score_values,
                    10,
                ),
            }
        )

    result = pd.DataFrame(rows)

    metrics = {
        "states": int(len(result)),
        "exact_best_action_fraction": float(
            result["exact_best_action"].mean()
        ),
        "top3_hit_fraction": float(
            result["top3_hit"].mean()
        ),
        "top5_hit_fraction": float(
            result["top5_hit"].mean()
        ),
        "mean_oracle_rank_selected": float(
            result["selected_oracle_rank"].mean()
        ),
        "median_oracle_rank_selected": float(
            result["selected_oracle_rank"].median()
        ),
        "mean_action_regret": float(
            result["regret"].mean()
        ),
        "median_action_regret": float(
            result["regret"].median()
        ),
        "p90_action_regret": float(
            result["regret"].quantile(0.90)
        ),
        "mean_normalized_regret": float(
            result["normalized_regret"].mean()
        ),
        "median_normalized_regret": float(
            result["normalized_regret"].median()
        ),
        "p90_normalized_regret": float(
            result["normalized_regret"].quantile(0.90)
        ),
        "mean_within_state_spearman": float(
            result["within_state_spearman"].mean()
        ),
        "mean_ndcg_at_5": float(
            result["ndcg_at_5"].mean()
        ),
        "mean_ndcg_at_10": float(
            result["ndcg_at_10"].mean()
        ),
    }
    return metrics, result


def random_expected_metrics(frame: pd.DataFrame) -> dict:
    state_rows = []

    for _state_key, group in frame.groupby("state_key", sort=False):
        n = len(group)
        q = group["oracle_q"].to_numpy(dtype=float)
        best = float(np.max(q))
        regret = best - q
        q_range = float(np.max(q) - np.min(q))
        normalized = (
            regret / q_range
            if q_range > EPS
            else np.zeros_like(regret)
        )
        ranks = group["action_rank"].to_numpy(dtype=float)

        state_rows.append(
            {
                "exact": float(np.mean(regret <= EPS)),
                "top3": float(np.mean(ranks <= 3)),
                "top5": float(np.mean(ranks <= 5)),
                "rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "p90_regret": float(np.quantile(regret, 0.90)),
                "normalized": float(np.mean(normalized)),
                "median_normalized": float(np.median(normalized)),
                "p90_normalized": float(
                    np.quantile(normalized, 0.90)
                ),
                "actions": n,
            }
        )

    table = pd.DataFrame(state_rows)

    return {
        "states": int(len(table)),
        "exact_best_action_fraction": float(table["exact"].mean()),
        "top3_hit_fraction": float(table["top3"].mean()),
        "top5_hit_fraction": float(table["top5"].mean()),
        "mean_oracle_rank_selected": float(table["rank"].mean()),
        "median_oracle_rank_selected": float(
            table["median_rank"].mean()
        ),
        "mean_action_regret": float(table["regret"].mean()),
        "median_action_regret": float(
            table["median_regret"].mean()
        ),
        "p90_action_regret": float(table["p90_regret"].mean()),
        "mean_normalized_regret": float(
            table["normalized"].mean()
        ),
        "median_normalized_regret": float(
            table["median_normalized"].mean()
        ),
        "p90_normalized_regret": float(
            table["p90_normalized"].mean()
        ),
        "mean_within_state_spearman": 0.0,
        "mean_ndcg_at_5": float("nan"),
        "mean_ndcg_at_10": float("nan"),
    }


def histgb_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=RANDOM_STATE,
    )


def extra_trees_regressor(
    *,
    n_jobs: int,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=6,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def histgb_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def extra_trees_classifier(
    *,
    n_jobs: int,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=8,
        max_features=0.6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def find_pairwise_feature_partition(
    train: pd.DataFrame,
    x_train: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    varying = np.zeros(len(features), dtype=bool)

    for indices in train.groupby(
        "state_key",
        sort=False,
    ).indices.values():
        idx = np.asarray(indices, dtype=int)
        values = x_train[idx]
        span = np.max(values, axis=0) - np.min(values, axis=0)
        varying |= span > 1e-10

    varying_indices = np.flatnonzero(varying)
    context_indices = np.flatnonzero(~varying)

    if len(varying_indices) == 0:
        raise RuntimeError(
            "No action-varying features found for pairwise ranking."
        )

    return context_indices, varying_indices


def build_pairwise_training(
    train: pd.DataFrame,
    x_train: np.ndarray,
    *,
    context_indices: np.ndarray,
    varying_indices: np.ndarray,
    max_pairs_per_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    groups = train.groupby("state_key", sort=False).indices

    pair_specs: list[tuple[int, int, int]] = []
    q_all = train["oracle_q"].to_numpy(dtype=float)
    range_all = train["state_q_range"].to_numpy(dtype=float)

    for _state_key, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        combinations = np.asarray(
            list(itertools.combinations(idx.tolist(), 2)),
            dtype=np.int64,
        )
        if len(combinations) == 0:
            continue

        if (
            max_pairs_per_state > 0
            and len(combinations) > max_pairs_per_state
        ):
            keep = rng.choice(
                len(combinations),
                size=max_pairs_per_state,
                replace=False,
            )
            combinations = combinations[keep]

        for i, j in combinations:
            if abs(q_all[i] - q_all[j]) <= EPS:
                continue
            if rng.random() < 0.5:
                pair_specs.append((int(i), int(j), 1))
            else:
                pair_specs.append((int(j), int(i), 1))

    feature_count = (
        len(context_indices)
        + len(varying_indices)
    )
    pair_x = np.empty(
        (len(pair_specs), feature_count),
        dtype=np.float32,
    )
    pair_y = np.empty(
        len(pair_specs),
        dtype=np.int8,
    )
    pair_weight = np.empty(
        len(pair_specs),
        dtype=np.float32,
    )

    context_width = len(context_indices)

    for row_number, (left, right, _dummy) in enumerate(pair_specs):
        pair_x[
            row_number,
            :context_width,
        ] = x_train[left, context_indices]

        pair_x[
            row_number,
            context_width:,
        ] = (
            x_train[left, varying_indices]
            - x_train[right, varying_indices]
        )

        left_q = q_all[left]
        right_q = q_all[right]
        pair_y[row_number] = int(left_q > right_q)

        state_range = max(
            float(range_all[left]),
            EPS,
        )
        normalized_gap = min(
            1.0,
            abs(float(left_q - right_q)) / state_range,
        )
        pair_weight[row_number] = (
            0.20 + 0.80 * normalized_gap
        )

    return pair_x, pair_y, pair_weight


def pairwise_scores(
    frame: pd.DataFrame,
    x: np.ndarray,
    *,
    model,
    context_indices: np.ndarray,
    varying_indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    context_width = len(context_indices)

    for _state_key, indices in frame.groupby(
        "state_key",
        sort=False,
    ).indices.items():
        idx = np.asarray(indices, dtype=int)
        if len(idx) == 1:
            result[idx[0]] = 1.0
            continue

        local_x = x[idx]
        rows = []
        owners = []

        context = local_x[0, context_indices]

        for local_i in range(len(idx)):
            for local_j in range(len(idx)):
                if local_i == local_j:
                    continue
                pair = np.empty(
                    context_width + len(varying_indices),
                    dtype=np.float32,
                )
                pair[:context_width] = context
                pair[context_width:] = (
                    local_x[local_i, varying_indices]
                    - local_x[local_j, varying_indices]
                )
                rows.append(pair)
                owners.append(local_i)

        pair_matrix = np.vstack(rows)
        probabilities = model.predict_proba(pair_matrix)[:, 1]

        local_scores = np.zeros(len(idx), dtype=float)
        local_counts = np.zeros(len(idx), dtype=float)

        for owner, probability in zip(owners, probabilities):
            local_scores[owner] += float(probability)
            local_counts[owner] += 1.0

        local_scores /= np.maximum(local_counts, 1.0)
        result[idx] = local_scores

    return result


def evaluate_teacher(
    teacher: pd.DataFrame,
    *,
    root: Path,
    n_jobs: int,
    max_pairs_per_state: int,
) -> dict:
    teacher = teacher.copy()
    teacher["state_date"] = pd.to_datetime(
        teacher["state_date"],
        errors="coerce",
    ).dt.normalize()

    features = feature_columns(teacher)
    if not features:
        raise RuntimeError("Teacher has no f__ predictor columns.")

    identity_leaks = [
        feature
        for feature in features
        if any(
            token in feature.lower()
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

    teacher = add_relative_targets(teacher)

    train = teacher.loc[
        teacher["state_date"].dt.year.isin(TRAIN_YEARS)
    ].copy().reset_index(drop=True)

    validation = teacher.loc[
        teacher["state_date"].dt.year.isin(VALIDATION_YEARS)
    ].copy().reset_index(drop=True)

    if train.empty or validation.empty:
        raise RuntimeError(
            "Expected non-empty 2021-2022 train and 2023 validation sets."
        )

    report_root = (
        root
        / "reports/ml/portfolio_student_v5/development"
    )
    model_root = (
        root
        / "data/processed/portfolio_student_v5"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    diagnostics = teacher_diagnostics(teacher)
    diagnostics.to_csv(
        report_root / "teacher_state_diagnostics.csv",
        index=False,
    )

    print("\n=== PORTFOLIO STUDENT V5 ===")
    print(
        f"Train states:      {train['state_key'].nunique():,} "
        f"({TRAIN_YEARS})"
    )
    print(f"Train actions:     {len(train):,}")
    print(
        f"Validation states: {validation['state_key'].nunique():,} "
        f"({VALIDATION_YEARS})"
    )
    print(f"Validation actions:{len(validation):,}")
    print(f"Predictors:        {len(features):,}")

    imputer = SimpleImputer(\n        strategy="median",\n        keep_empty_features=True,\n    )
    x_train = imputer.fit_transform(
        train[features]
    ).astype(np.float32, copy=False)
    x_validation = imputer.transform(
        validation[features]
    ).astype(np.float32, copy=False)

    leaderboard_rows: list[dict] = []
    state_results: list[pd.DataFrame] = []
    model_bundles: dict[str, dict] = {}

    random_metrics = random_expected_metrics(validation)
    leaderboard_rows.append(
        {
            "candidate": "random_expected",
            "formulation": "random",
            "model": "uniform_random",
            **random_metrics,
        }
    )

    def record(
        *,
        candidate: str,
        formulation: str,
        model_name: str,
        scores: np.ndarray,
        bundle: dict,
    ) -> None:
        metrics, state_frame = ranking_metrics(
            validation,
            scores,
        )
        leaderboard_rows.append(
            {
                "candidate": candidate,
                "formulation": formulation,
                "model": model_name,
                **metrics,
            }
        )
        state_frame["candidate"] = candidate
        state_results.append(state_frame)
        model_bundles[candidate] = bundle

        print(
            f"{candidate:30s} "
            f"rank={metrics['mean_oracle_rank_selected']:.2f} "
            f"top5={metrics['top5_hit_fraction']:.1%} "
            f"regret={metrics['mean_action_regret']:.5f} "
            f"norm={metrics['mean_normalized_regret']:.3f} "
            f"rho={metrics['mean_within_state_spearman']:+.3f}"
        )

    print("\n--- Absolute-Q baseline ---")
    absolute_model = histgb_regressor()
    absolute_model.fit(
        x_train,
        train["oracle_q"].to_numpy(dtype=float),
    )
    absolute_scores = absolute_model.predict(x_validation)
    record(
        candidate="absolute_q_histgb",
        formulation="absolute_q",
        model_name="histgb",
        scores=absolute_scores,
        bundle={
            "formulation": "absolute_q",
            "model": absolute_model,
            "imputer": imputer,
            "features": features,
        },
    )

    regression_specs = [
        (
            "histgb",
            histgb_regressor,
        ),
        (
            "extra_trees",
            lambda: extra_trees_regressor(
                n_jobs=n_jobs,
            ),
        ),
    ]

    print("\n--- Within-state regret regression ---")
    for name, factory in regression_specs:
        model = factory()
        model.fit(
            x_train,
            train["oracle_regret"].to_numpy(dtype=float),
        )
        scores = -model.predict(x_validation)
        record(
            candidate=f"regret_{name}",
            formulation="regret",
            model_name=name,
            scores=scores,
            bundle={
                "formulation": "regret",
                "model": model,
                "imputer": imputer,
                "features": features,
            },
        )

    print("\n--- Normalized-regret regression ---")
    for name, factory in regression_specs:
        model = factory()
        model.fit(
            x_train,
            train[
                "oracle_normalized_regret"
            ].to_numpy(dtype=float),
        )
        scores = -model.predict(x_validation)
        record(
            candidate=f"normalized_regret_{name}",
            formulation="normalized_regret",
            model_name=name,
            scores=scores,
            bundle={
                "formulation": "normalized_regret",
                "model": model,
                "imputer": imputer,
                "features": features,
            },
        )

    print("\n--- Pairwise ranking ---")
    context_indices, varying_indices = (
        find_pairwise_feature_partition(
            train,
            x_train,
            features,
        )
    )
    print(
        "Pairwise feature partition: "
        f"{len(context_indices)} context + "
        f"{len(varying_indices)} action-varying"
    )

    pair_x, pair_y, pair_weight = build_pairwise_training(
        train,
        x_train,
        context_indices=context_indices,
        varying_indices=varying_indices,
        max_pairs_per_state=max_pairs_per_state,
    )
    print(
        f"Pairwise training rows: {len(pair_x):,} "
        f"× {pair_x.shape[1]:,} features"
    )
    print(
        f"Pairwise class balance: {float(pair_y.mean()):.1%} positive"
    )

    pair_specs = [
        (
            "histgb",
            histgb_classifier,
        ),
        (
            "extra_trees",
            lambda: extra_trees_classifier(
                n_jobs=n_jobs,
            ),
        ),
    ]

    for name, factory in pair_specs:
        model = factory()
        model.fit(
            pair_x,
            pair_y,
            sample_weight=pair_weight,
        )
        scores = pairwise_scores(
            validation,
            x_validation,
            model=model,
            context_indices=context_indices,
            varying_indices=varying_indices,
        )
        record(
            candidate=f"pairwise_{name}",
            formulation="pairwise",
            model_name=name,
            scores=scores,
            bundle={
                "formulation": "pairwise",
                "model": model,
                "imputer": imputer,
                "features": features,
                "context_indices": context_indices,
                "varying_indices": varying_indices,
                "context_features": [
                    features[i]
                    for i in context_indices
                ],
                "varying_features": [
                    features[i]
                    for i in varying_indices
                ],
            },
        )

    leaderboard = pd.DataFrame(leaderboard_rows)

    learned_mask = leaderboard["formulation"].ne("random")
    learned = (
        leaderboard.loc[learned_mask]
        .sort_values(
            [
                "mean_normalized_regret",
                "mean_action_regret",
                "mean_oracle_rank_selected",
                "top5_hit_fraction",
                "mean_within_state_spearman",
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
        .reset_index(drop=True)
    )
    learned["development_rank"] = np.arange(
        1,
        len(learned) + 1,
    )

    leaderboard = leaderboard.merge(
        learned[["candidate", "development_rank"]],
        on="candidate",
        how="left",
    ).sort_values(
        [
            "development_rank",
            "candidate",
        ],
        na_position="last",
        kind="stable",
    )

    leaderboard.to_csv(
        report_root / "leaderboard.csv",
        index=False,
    )

    if state_results:
        pd.concat(
            state_results,
            ignore_index=True,
        ).to_parquet(
            report_root / "state_ranking_results.parquet",
            index=False,
            compression="zstd",
        )

    best_candidate = str(
        learned.iloc[0]["candidate"]
    )
    best_bundle = model_bundles[best_candidate]
    best_bundle.update(
        {
            "candidate": best_candidate,
            "train_years": TRAIN_YEARS,
            "validation_years": VALIDATION_YEARS,
            "selection_rule": (
                "lowest 2023 mean normalized regret; then "
                "mean action regret; mean selected oracle rank; "
                "higher top-5 hit rate; higher within-state Spearman"
            ),
        }
    )

    model_path = (
        model_root / "development_best.joblib"
    )
    joblib.dump(
        best_bundle,
        model_path,
        compress=3,
    )

    diagnostics_summary = {
        "states": int(len(diagnostics)),
        "actions_min": int(diagnostics["actions"].min()),
        "actions_median": float(diagnostics["actions"].median()),
        "actions_max": int(diagnostics["actions"].max()),
        "median_q_range": float(
            diagnostics["q_range"].median()
        ),
        "median_gap_best_second": float(
            diagnostics["gap_best_second"].median()
        ),
        "median_gap_best_fifth": float(
            diagnostics["gap_best_fifth"].median()
        ),
        "median_gap_best_median": float(
            diagnostics["gap_best_median"].median()
        ),
        "median_near_optimal_5pct_range_count": float(
            diagnostics[
                "near_optimal_5pct_range_count"
            ].median()
        ),
    }

    summary = {
        "teacher": "portfolio_oracle_v4",
        "student_version": "v5",
        "train_years": TRAIN_YEARS,
        "validation_years": VALIDATION_YEARS,
        "feature_count": len(features),
        "identity_features": [],
        "pairwise_context_feature_count": int(
            len(context_indices)
        ),
        "pairwise_varying_feature_count": int(
            len(varying_indices)
        ),
        "pairwise_training_rows": int(
            len(pair_x)
        ),
        "max_pairs_per_state": int(
            max_pairs_per_state
        ),
        "teacher_diagnostics": diagnostics_summary,
        "random_expected": random_metrics,
        "selection_rule": best_bundle["selection_rule"],
        "best_candidate": best_candidate,
        "best_metrics": learned.iloc[0].to_dict(),
        "model_path": str(model_path),
        "confirmation_rule": (
            "Do not use 2024-2026 oracle labels or retune on "
            "2024-2026 until the development policy and action-search "
            "procedure are frozen."
        ),
    }

    (
        report_root / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=float,
        )
        + "\n"
    )

    print("\n=== PORTFOLIO STUDENT V5 COMPLETE ===")
    print("\nLeaderboard:")
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
        ].to_string(index=False)
    )
    print(f"\nBest candidate: {best_candidate}")
    print(f"Model: {model_path}")
    print(f"Reports: {report_root}")

    return summary


def self_test() -> None:
    toy = pd.DataFrame(
        {
            "state_key": ["a"] * 4 + ["b"] * 4,
            "state_date": pd.to_datetime(
                ["2023-01-01"] * 4
                + ["2023-01-02"] * 4
            ),
            "oracle_q": [
                0.1,
                0.2,
                0.0,
                0.15,
                -0.1,
                0.0,
                -0.2,
                -0.05,
            ],
            "action_rank": [
                3,
                1,
                4,
                2,
                3,
                1,
                4,
                2,
            ],
            "meta_target_signature": [
                "a0",
                "a1",
                "a2",
                "a3",
                "b0",
                "b1",
                "b2",
                "b3",
            ],
        }
    )
    toy = add_relative_targets(toy)
    scores = toy["oracle_q"].to_numpy(dtype=float)
    metrics, rows = ranking_metrics(toy, scores)
    assert len(rows) == 2
    assert metrics["exact_best_action_fraction"] == 1.0
    assert metrics["mean_action_regret"] == 0.0
    assert metrics["top5_hit_fraction"] == 1.0

    diagnostics = teacher_diagnostics(toy)
    assert len(diagnostics) == 2

    print("Portfolio-student-v5 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V5 whole-portfolio student tournament: absolute Q, "
            "within-state regret, normalized regret, and pairwise ranking."
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
        help=(
            "Maximum unordered action pairs per state. "
            "496 is all pairs for 32 actions."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    root = Path(args.root).resolve()
    teacher_path = (
        Path(args.teacher_path).resolve()
        if args.teacher_path is not None
        else (
            root
            / "data/processed/portfolio_oracle_v4/"
            "development_teacher.parquet"
        )
    )

    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Missing oracle teacher: {teacher_path}"
        )

    print(f"CPU threads visible: {os.cpu_count()}")
    print(f"Teacher: {teacher_path}")

    teacher = pd.read_parquet(teacher_path)

    evaluate_teacher(
        teacher,
        root=root,
        n_jobs=args.n_jobs,
        max_pairs_per_state=args.max_pairs_per_state,
    )


if __name__ == "__main__":
    main()
