from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ML_DIR = SCRIPT_PATH.parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ML_DIR),
    )

from classical import (  # noqa: E402
    FEATURE_COLUMNS,
    HORIZON,
    TRAINING_FLAG,
    TRAIN_START,
    add_target_rank,
    build_model,
    fit_model,
    load_ml_panel,
)


TREE_MODELS = [
    "hist_gb",
    "hist_gb_fixed",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
]


def require_shap():
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is optional. Install "
            "the tournament extras with: "
            "uv sync --extra tournament"
        ) from exc
    return shap


def unwrap_model(
    model,
    X: pd.DataFrame,
) -> tuple[
    object,
    np.ndarray,
]:
    if hasattr(
        model,
        "steps",
    ):
        if len(
            model.steps
        ) < 1:
            raise RuntimeError(
                "Empty sklearn pipeline."
            )
        base = model.steps[
            -1
        ][1]
        if len(
            model.steps
        ) > 1:
            transformed = (
                model[
                    :-1
                ].transform(
                    X
                )
            )
        else:
            transformed = (
                X.to_numpy()
            )
        return (
            base,
            np.asarray(
                transformed,
                dtype=float,
            ),
        )

    return (
        model,
        X.to_numpy(
            dtype=float,
        ),
    )


def explain_values(
    model,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame,
) -> np.ndarray:
    shap = require_shap()
    base, transformed = (
        unwrap_model(
            model,
            X,
        )
    )
    _, background_values = (
        unwrap_model(
            model,
            background,
        )
    )

    try:
        explainer = (
            shap.TreeExplainer(
                base
            )
        )
        values = (
            explainer.shap_values(
                transformed
            )
        )
        if isinstance(
            values,
            list,
        ):
            values = values[
                0
            ]
        return np.asarray(
            values,
            dtype=float,
        )
    except Exception:
        # Generic fallback is slower but keeps the
        # explanation script useful for estimators
        # TreeExplainer does not recognize.
        explainer = shap.Explainer(
            base.predict,
            background_values,
        )
        explanation = explainer(
            transformed
        )
        return np.asarray(
            explanation.values,
            dtype=float,
        )


def interaction_values(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    shap = require_shap()
    base, transformed = (
        unwrap_model(
            model,
            X,
        )
    )
    explainer = (
        shap.TreeExplainer(
            base
        )
    )
    values = (
        explainer
        .shap_interaction_values(
            transformed
        )
    )
    if isinstance(
        values,
        list,
    ):
        values = values[
            0
        ]
    return np.asarray(
        values,
        dtype=float,
    )


def sample_frame(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rows = np.flatnonzero(
        mask.to_numpy(
            dtype=bool
        )
    )
    if not len(rows):
        return df.iloc[
            []
        ][
            FEATURE_COLUMNS
        ]

    if (
        n > 0
        and len(rows) > n
    ):
        rng = np.random.default_rng(
            seed
        )
        rows = rng.choice(
            rows,
            size=n,
            replace=False,
        )
        rows.sort()

    return df.iloc[
        rows
    ][
        FEATURE_COLUMNS
    ].copy()


def feature_stability(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    years = int(
        importance[
            "year"
        ].nunique()
    )

    for feature, group in (
        importance.groupby(
            "feature",
            sort=False,
        )
    ):
        ranks = pd.to_numeric(
            group[
                "importance_rank"
            ],
            errors="coerce",
        )
        signed = pd.to_numeric(
            group[
                "mean_signed_shap"
            ],
            errors="coerce",
        )
        abs_shap = pd.to_numeric(
            group[
                "mean_abs_shap"
            ],
            errors="coerce",
        )

        rows.append({
            "feature": feature,
            "years": years,
            "median_importance_rank": float(
                ranks.median()
            ),
            "q25_importance_rank": float(
                ranks.quantile(
                    0.25
                )
            ),
            "q75_importance_rank": float(
                ranks.quantile(
                    0.75
                )
            ),
            "top5_fraction": float(
                ranks.le(5).mean()
            ),
            "top10_fraction": float(
                ranks.le(10).mean()
            ),
            "median_mean_abs_shap": float(
                abs_shap.median()
            ),
            "positive_signed_fraction": float(
                signed.gt(0).mean()
            ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "median_importance_rank",
                "top5_fraction",
                "median_mean_abs_shap",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Year-by-year SHAP stability "
            "analysis for one fixed tree model."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--model",
        choices=(
            TREE_MODELS
        ),
        default="hist_gb_fixed",
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
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--sample-per-year",
        type=int,
        default=3000,
    )
    ap.add_argument(
        "--background-size",
        type=int,
        default=500,
    )
    ap.add_argument(
        "--interaction-sample",
        type=int,
        default=300,
    )
    ap.add_argument(
        "--no-interactions",
        action="store_true",
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "model_tournament/"
        "explanations"
        / args.model
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

    feature_rows: list[
        dict
    ] = []
    interaction_rows: list[
        dict
    ] = []

    for year in range(
        args.start_year,
        args.end_year
        + 1,
    ):
        fold_start = pd.Timestamp(
            f"{year}-01-01"
        )
        prior = df.loc[
            df[
                "date"
            ].lt(
                fold_start
            ),
            "market_day_index",
        ].dropna()
        if prior.empty:
            continue
        cutoff = int(
            prior.max()
        )

        train_mask = (
            df[
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
        score_mask = (
            df[
                "eligible_universe"
            ]
            & df[
                "date"
            ].dt.year.eq(
                year
            )
        )

        print(
            f"\n{args.model} SHAP "
            f"year={year} "
            f"train={int(train_mask.sum()):,} "
            f"score={int(score_mask.sum()):,}"
        )

        model = build_model(
            args.model,
            random_state=(
                args.seed
            ),
        )
        fit_model(
            model,
            df,
            train_mask,
        )

        X = sample_frame(
            df,
            score_mask,
            n=(
                args.sample_per_year
            ),
            seed=(
                args.seed
                + year
            ),
        )
        background = sample_frame(
            df,
            train_mask,
            n=(
                args.background_size
            ),
            seed=(
                args.seed
                + year
                + 100_000
            ),
        )

        values = explain_values(
            model,
            X,
            background=(
                background
            ),
        )
        if values.ndim != 2:
            values = values.reshape(
                len(X),
                -1,
            )

        mean_abs = np.nanmean(
            np.abs(values),
            axis=0,
        )
        mean_signed = (
            np.nanmean(
                values,
                axis=0,
            )
        )
        order = np.argsort(
            -mean_abs
        )
        ranks = np.empty(
            len(order),
            dtype=int,
        )
        ranks[
            order
        ] = (
            np.arange(
                len(order)
            )
            + 1
        )

        for index, feature in enumerate(
            FEATURE_COLUMNS
        ):
            feature_rows.append({
                "year": int(
                    year
                ),
                "model_name": (
                    args.model
                ),
                "feature": feature,
                "mean_abs_shap": float(
                    mean_abs[
                        index
                    ]
                ),
                "mean_signed_shap": float(
                    mean_signed[
                        index
                    ]
                ),
                "importance_rank": int(
                    ranks[
                        index
                    ]
                ),
            })

        if (
            not args.no_interactions
            and args.interaction_sample
            > 0
        ):
            interaction_X = (
                X.head(
                    args.interaction_sample
                )
            )
            try:
                iv = interaction_values(
                    model,
                    interaction_X,
                )
                if iv.ndim == 3:
                    strength = (
                        np.nanmean(
                            np.abs(iv),
                            axis=0,
                        )
                    )
                    for i in range(
                        len(
                            FEATURE_COLUMNS
                        )
                    ):
                        for j in range(
                            i + 1,
                            len(
                                FEATURE_COLUMNS
                            ),
                        ):
                            interaction_rows.append({
                                "year": int(
                                    year
                                ),
                                "model_name": (
                                    args.model
                                ),
                                "feature_a": (
                                    FEATURE_COLUMNS[
                                        i
                                    ]
                                ),
                                "feature_b": (
                                    FEATURE_COLUMNS[
                                        j
                                    ]
                                ),
                                "mean_abs_interaction": float(
                                    strength[
                                        i,
                                        j,
                                    ]
                                ),
                            })
            except Exception as exc:
                print(
                    "  interaction SHAP "
                    "unavailable for this "
                    f"model/year: {exc}"
                )

    importance = pd.DataFrame(
        feature_rows
    )
    if importance.empty:
        raise RuntimeError(
            "No SHAP explanations produced."
        )

    importance.to_csv(
        output_root
        / "annual_feature_shap.csv",
        index=False,
    )
    feature_stability(
        importance
    ).to_csv(
        output_root
        / "feature_stability.csv",
        index=False,
    )

    if interaction_rows:
        interactions = pd.DataFrame(
            interaction_rows
        )
        interactions.to_csv(
            output_root
            / "annual_interactions.csv",
            index=False,
        )
        (
            interactions.groupby(
                [
                    "feature_a",
                    "feature_b",
                ],
                as_index=False,
            )
            .agg(
                median_interaction=(
                    "mean_abs_interaction",
                    "median",
                ),
                mean_interaction=(
                    "mean_abs_interaction",
                    "mean",
                ),
                years_present=(
                    "year",
                    "nunique",
                ),
            )
            .sort_values(
                [
                    "median_interaction",
                    "years_present",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .to_csv(
                output_root
                / "interaction_stability.csv",
                index=False,
            )
        )

    summary = {
        "model": args.model,
        "years": [
            args.start_year,
            args.end_year,
        ],
        "sample_per_year": (
            args.sample_per_year
        ),
        "background_size": (
            args.background_size
        ),
        "interaction_sample": (
            0
            if args.no_interactions
            else args.interaction_sample
        ),
        "purpose": (
            "measure whether nonlinear feature "
            "importance and pair interactions "
            "remain stable across annual OOS folds"
        ),
    }

    (
        output_root
        / "explanation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "\nTop stable features:"
    )
    stable = feature_stability(
        importance
    )
    print(
        stable.head(
            15
        ).to_string(
            index=False
        )
    )

    print(
        "\nOutputs: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
