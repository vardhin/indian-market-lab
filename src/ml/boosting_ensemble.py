from __future__ import annotations

import argparse
import itertools
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
    build_consensus,
    cost_profile,
    evaluate_scores,
)
from nested_generalization import (  # noqa: E402
    benchmark_rows,
    build_annual_oos_predictions,
    first_aligned_signal_date,
    prediction_path,
)
from persistent_portfolio import (  # noqa: E402
    add_daily_ranks,
)


BASE_MODELS = [
    "xgboost",
    "lightgbm",
    "catboost",
]
DEFAULT_TOP_K = [
    5,
    10,
    15,
    20,
]
ENSEMBLES = {
    "ensemble_xgb_lgbm": [
        "xgboost",
        "lightgbm",
    ],
    "ensemble_xgb_cat": [
        "xgboost",
        "catboost",
    ],
    "ensemble_lgbm_cat": [
        "lightgbm",
        "catboost",
    ],
    "ensemble_xgb_lgbm_cat": [
        "xgboost",
        "lightgbm",
        "catboost",
    ],
}


def require_prediction_cache(
    root: Path,
    *,
    models: list[str],
    years: list[int],
    seed: int,
) -> None:
    missing: list[str] = []

    for model in models:
        for year in years:
            path = prediction_path(
                root,
                model_name=model,
                random_state=seed,
                year=year,
            )
            if not path.is_file():
                missing.append(
                    str(
                        path.relative_to(
                            root
                        )
                    )
                )

    if missing:
        preview = "\n".join(
            f"  {item}"
            for item in missing[:20]
        )
        suffix = (
            ""
            if len(missing) <= 20
            else (
                f"\n  ... and "
                f"{len(missing) - 20} more"
            )
        )
        raise RuntimeError(
            "Cached annual predictions are "
            "missing. The complementarity "
            "experiment is cache-first so it "
            "does not unexpectedly retrain. "
            "Missing:\n"
            f"{preview}{suffix}\n"
            "Re-run with --allow-fit only if "
            "you intentionally want CPU fitting."
        )


def daily_rank_frame(
    df: pd.DataFrame,
    *,
    eval_mask: pd.Series,
    score_store: dict[
        str,
        np.ndarray,
    ],
) -> pd.DataFrame:
    view = df.loc[
        eval_mask,
        [
            "date",
            "canonical_security_id",
            "turnover_median_20d",
        ],
    ].copy()

    for model, values in (
        score_store.items()
    ):
        if len(values) != len(view):
            raise RuntimeError(
                f"Score length mismatch "
                f"for {model}: "
                f"{len(values)} vs {len(view)}"
            )
        view[
            f"score__{model}"
        ] = values

        view[
            f"pct_rank__{model}"
        ] = (
            view.groupby(
                "date",
                sort=False,
            )[
                f"score__{model}"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        ordered = (
            view[
                [
                    "date",
                    "canonical_security_id",
                    "turnover_median_20d",
                    f"score__{model}",
                ]
            ]
            .sort_values(
                [
                    "date",
                    f"score__{model}",
                    "turnover_median_20d",
                    "canonical_security_id",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
        )
        absolute_rank = (
            ordered.groupby(
                "date",
                sort=False,
            ).cumcount()
            + 1
        )
        view.loc[
            ordered.index,
            f"abs_rank__{model}",
        ] = absolute_rank.to_numpy(
            dtype=np.int32
        )

    return view


def score_correlation_summary(
    ranks: pd.DataFrame,
    *,
    models: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    daily_rows: list[
        dict
    ] = []

    pairs = list(
        itertools.combinations(
            models,
            2,
        )
    )

    for date, group in (
        ranks.groupby(
            "date",
            sort=True,
        )
    ):
        for left, right in pairs:
            cols = [
                f"pct_rank__{left}",
                f"pct_rank__{right}",
            ]
            usable = (
                group[
                    cols
                ].dropna()
            )
            if len(usable) < 10:
                continue
            corr = (
                usable.corr()
                .iloc[
                    0,
                    1,
                ]
            )
            daily_rows.append({
                "date": date,
                "model_a": left,
                "model_b": right,
                "spearman_rank_corr": (
                    float(corr)
                ),
                "rows": int(
                    len(usable)
                ),
            })

    daily = pd.DataFrame(
        daily_rows,
        columns=[
            "date",
            "model_a",
            "model_b",
            "spearman_rank_corr",
            "rows",
        ],
    )

    summary_rows: list[
        dict
    ] = []
    for (
        model_a,
        model_b,
    ), group in daily.groupby(
        [
            "model_a",
            "model_b",
        ],
        sort=False,
    ):
        values = (
            pd.to_numeric(
                group[
                    "spearman_rank_corr"
                ],
                errors="coerce",
            )
            .dropna()
        )
        summary_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "dates": int(
                len(values)
            ),
            "mean_spearman": float(
                values.mean()
            ),
            "median_spearman": float(
                values.median()
            ),
            "q25_spearman": float(
                values.quantile(
                    0.25
                )
            ),
            "q75_spearman": float(
                values.quantile(
                    0.75
                )
            ),
        })

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "model_a",
            "model_b",
            "dates",
            "mean_spearman",
            "median_spearman",
            "q25_spearman",
            "q75_spearman",
        ],
    )

    return (
        daily,
        summary,
    )


def topk_overlap_summary(
    ranks: pd.DataFrame,
    *,
    models: list[str],
    top_k_values: list[int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pairs = list(
        itertools.combinations(
            models,
            2,
        )
    )
    daily_rows: list[
        dict
    ] = []

    for date, group in ranks.groupby(
        "date",
        sort=True,
    ):
        for top_k in top_k_values:
            sets = {
                model: set(
                    group.loc[
                        pd.to_numeric(
                            group[
                                f"abs_rank__{model}"
                            ],
                            errors="coerce",
                        ).le(
                            int(top_k)
                        ),
                        "canonical_security_id",
                    ]
                    .astype(str)
                    .tolist()
                )
                for model in models
            }

            for left, right in pairs:
                a = sets[
                    left
                ]
                b = sets[
                    right
                ]
                intersection = len(
                    a & b
                )
                union = len(
                    a | b
                )
                denom = max(
                    min(
                        len(a),
                        len(b),
                    ),
                    1,
                )
                daily_rows.append({
                    "date": date,
                    "top_k": int(
                        top_k
                    ),
                    "model_a": left,
                    "model_b": right,
                    "intersection": int(
                        intersection
                    ),
                    "overlap_fraction": float(
                        intersection
                        / denom
                    ),
                    "jaccard": float(
                        intersection
                        / union
                    )
                    if union
                    else np.nan,
                })

            all_intersection = (
                set.intersection(
                    *[
                        sets[
                            model
                        ]
                        for model
                        in models
                    ]
                )
            )
            denom = max(
                min(
                    len(
                        sets[
                            model
                        ]
                    )
                    for model in models
                ),
                1,
            )
            daily_rows.append({
                "date": date,
                "top_k": int(
                    top_k
                ),
                "model_a": (
                    "ALL_THREE"
                ),
                "model_b": (
                    "ALL_THREE"
                ),
                "intersection": int(
                    len(
                        all_intersection
                    )
                ),
                "overlap_fraction": float(
                    len(
                        all_intersection
                    )
                    / denom
                ),
                "jaccard": np.nan,
            })

    daily = pd.DataFrame(
        daily_rows
    )

    summary_rows: list[
        dict
    ] = []
    for keys, group in daily.groupby(
        [
            "top_k",
            "model_a",
            "model_b",
        ],
        sort=False,
    ):
        top_k, model_a, model_b = (
            keys
        )
        overlap = pd.to_numeric(
            group[
                "overlap_fraction"
            ],
            errors="coerce",
        ).dropna()
        jaccard = pd.to_numeric(
            group[
                "jaccard"
            ],
            errors="coerce",
        ).dropna()

        summary_rows.append({
            "top_k": int(
                top_k
            ),
            "model_a": model_a,
            "model_b": model_b,
            "dates": int(
                len(group)
            ),
            "mean_overlap_fraction": float(
                overlap.mean()
            ),
            "median_overlap_fraction": float(
                overlap.median()
            ),
            "q25_overlap_fraction": float(
                overlap.quantile(
                    0.25
                )
            ),
            "mean_jaccard": (
                float(
                    jaccard.mean()
                )
                if len(jaccard)
                else None
            ),
        })

    return (
        daily,
        pd.DataFrame(
            summary_rows
        ),
    )


def portfolio_pair_diagnostics(
    equity: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    curves: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        model_name,
        top_k,
    ), group in equity.groupby(
        [
            "model_name",
            "top_k",
        ],
        sort=False,
    ):
        key = (
            f"{model_name}__k"
            f"{int(top_k)}"
        )
        curve = (
            group[
                [
                    "date",
                    "equity",
                ]
            ]
            .dropna()
            .sort_values(
                "date"
            )
            .drop_duplicates(
                "date",
                keep="last",
            )
            .copy()
        )
        curve[
            "return"
        ] = (
            curve[
                "equity"
            ]
            .pct_change()
        )
        curve[
            "drawdown"
        ] = (
            curve[
                "equity"
            ]
            / curve[
                "equity"
            ].cummax()
            - 1.0
        )
        curves[
            key
        ] = curve

    keys = sorted(
        curves
    )
    pair_rows: list[
        dict
    ] = []

    for left, right in (
        itertools.combinations(
            keys,
            2,
        )
    ):
        merged = (
            curves[
                left
            ]
            .rename(
                columns={
                    "return": "return_a",
                    "drawdown": (
                        "drawdown_a"
                    ),
                }
            )[
                [
                    "date",
                    "return_a",
                    "drawdown_a",
                ]
            ]
            .merge(
                curves[
                    right
                ]
                .rename(
                    columns={
                        "return": (
                            "return_b"
                        ),
                        "drawdown": (
                            "drawdown_b"
                        ),
                    }
                )[
                    [
                        "date",
                        "return_b",
                        "drawdown_b",
                    ]
                ],
                on="date",
                how="inner",
                validate=(
                    "one_to_one"
                ),
            )
            .dropna(
                subset=[
                    "return_a",
                    "return_b",
                ]
            )
        )

        return_corr = (
            merged[
                [
                    "return_a",
                    "return_b",
                ]
            ]
            .corr()
            .iloc[
                0,
                1,
            ]
            if len(
                merged
            ) >= 3
            else np.nan
        )

        under10_a = (
            merged[
                "drawdown_a"
            ].le(
                -0.10
            )
        )
        under10_b = (
            merged[
                "drawdown_b"
            ].le(
                -0.10
            )
        )
        under20_a = (
            merged[
                "drawdown_a"
            ].le(
                -0.20
            )
        )
        under20_b = (
            merged[
                "drawdown_b"
            ].le(
                -0.20
            )
        )

        pair_rows.append({
            "strategy_a": left,
            "strategy_b": right,
            "aligned_days": int(
                len(
                    merged
                )
            ),
            "daily_return_corr": float(
                return_corr
            )
            if pd.notna(
                return_corr
            )
            else None,
            "both_under_10pct_fraction": float(
                (
                    under10_a
                    & under10_b
                ).mean()
            ),
            "either_under_10pct_fraction": float(
                (
                    under10_a
                    | under10_b
                ).mean()
            ),
            "both_under_20pct_fraction": float(
                (
                    under20_a
                    & under20_b
                ).mean()
            ),
            "either_under_20pct_fraction": float(
                (
                    under20_a
                    | under20_b
                ).mean()
            ),
        })

    matrix = (
        pd.DataFrame({
            key: (
                curves[
                    key
                ]
                .set_index(
                    "date"
                )[
                    "return"
                ]
            )
            for key in keys
        })
        .corr()
    )

    return (
        pd.DataFrame(
            pair_rows
        ),
        matrix,
    )


def annual_winners(
    annual: pd.DataFrame,
) -> pd.DataFrame:
    complete = annual.loc[
        ~annual[
            "partial_year"
        ].fillna(False)
    ].copy()

    rows: list[
        dict
    ] = []

    for (
        year,
        top_k,
    ), group in complete.groupby(
        [
            "year",
            "top_k",
        ],
        sort=True,
    ):
        group = group.copy()

        cagr = pd.to_numeric(
            group[
                "cagr"
            ],
            errors="coerce",
        )
        sharpe = pd.to_numeric(
            group[
                "sharpe"
            ],
            errors="coerce",
        )

        if cagr.notna().any():
            idx = cagr.idxmax()
            item = group.loc[
                idx
            ]
            rows.append({
                "year": int(
                    year
                ),
                "top_k": int(
                    top_k
                ),
                "metric": (
                    "cagr"
                ),
                "winner": item[
                    "model_name"
                ],
                "value": float(
                    cagr.loc[
                        idx
                    ]
                ),
            })

        if sharpe.notna().any():
            idx = sharpe.idxmax()
            item = group.loc[
                idx
            ]
            rows.append({
                "year": int(
                    year
                ),
                "top_k": int(
                    top_k
                ),
                "metric": (
                    "sharpe"
                ),
                "winner": item[
                    "model_name"
                ],
                "value": float(
                    sharpe.loc[
                        idx
                    ]
                ),
            })

    return pd.DataFrame(
        rows
    )


def self_test() -> None:
    securities = [
        f"s{i:02d}"
        for i in range(12)
    ]
    dates = pd.to_datetime([
        "2020-01-01",
        "2020-01-02",
    ])

    rows = []
    for date in dates:
        for i, security in enumerate(
            securities
        ):
            rows.append({
                "date": date,
                "canonical_security_id": (
                    security
                ),
                "turnover_median_20d": float(
                    i + 1
                ),
            })

    fake = pd.DataFrame(
        rows
    )
    mask = pd.Series(
        [True] * len(
            fake
        )
    )

    base = np.tile(
        np.arange(
            12,
            dtype=float,
        ),
        2,
    )
    store = {
        "xgboost": (
            base.copy()
        ),
        "lightgbm": (
            base[::-1].copy()
        ),
        "catboost": (
            np.roll(
                base,
                2,
            )
        ),
    }

    ranks = daily_rank_frame(
        fake,
        eval_mask=mask,
        score_store=store,
    )
    daily_corr, corr = (
        score_correlation_summary(
            ranks,
            models=BASE_MODELS,
        )
    )
    assert len(
        daily_corr
    ) == 6
    assert len(
        corr
    ) == 3

    _, overlap = (
        topk_overlap_summary(
            ranks,
            models=BASE_MODELS,
            top_k_values=[
                5,
            ],
        )
    )
    assert not overlap.empty

    tiny = ranks.groupby(
        "date",
        sort=False,
    ).head(2)
    empty_daily, empty_summary = (
        score_correlation_summary(
            tiny,
            models=BASE_MODELS,
        )
    )
    assert empty_daily.empty
    assert empty_summary.empty
    assert list(
        empty_summary.columns
    ) == [
        "model_a",
        "model_b",
        "dates",
        "mean_spearman",
        "median_spearman",
        "q25_spearman",
        "q75_spearman",
    ]

    print(
        "Boosting ensemble self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Cache-first complementarity and "
            "fixed equal-rank ensemble study "
            "for XGBoost, LightGBM and CatBoost."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
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
        "--allow-fit",
        action="store_true",
        help=(
            "Permit missing annual prediction "
            "caches to be fitted on CPU."
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

    root = Path(
        args.root
    ).resolve()
    output_root = (
        root
        / "reports/ml/"
        "boosting_ensemble"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    years = list(
        range(
            int(
                args.oos_start_year
            ),
            int(
                args.oos_end_year
            )
            + 1,
        )
    )

    if not args.allow_fit:
        require_prediction_cache(
            root,
            models=BASE_MODELS,
            years=years,
            seed=int(
                args.seed
            ),
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
    eval_end_date = pd.Timestamp(
        f"{args.oos_end_year}-12-31"
    )
    eval_mask = (
        df[
            "eligible_universe"
        ]
        & df[
            "date"
        ].between(
            eval_start_date,
            eval_end_date,
            inclusive="both",
        )
    )

    score_store: dict[
        str,
        np.ndarray,
    ] = {}
    diagnostics_frames: list[
        pd.DataFrame
    ] = []

    for model_name in BASE_MODELS:
        print(
            "\n================================"
        )
        print(
            f"LOAD SCORES: {model_name}"
        )
        print(
            "================================"
        )

        diagnostics = (
            build_annual_oos_predictions(
                df,
                root=root,
                model_name=(
                    model_name
                ),
                years=years,
                random_state=int(
                    args.seed
                ),
                rebuild=False,
            )
        )
        diagnostics[
            "model_name"
        ] = model_name
        diagnostics_frames.append(
            diagnostics
        )

        score_store[
            model_name
        ] = (
            pd.to_numeric(
                df.loc[
                    eval_mask,
                    "persistent_score",
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float32
            )
            .copy()
        )

    ranks = daily_rank_frame(
        df,
        eval_mask=eval_mask,
        score_store=score_store,
    )

    (
        daily_corr,
        corr_summary,
    ) = score_correlation_summary(
        ranks,
        models=BASE_MODELS,
    )
    (
        daily_overlap,
        overlap_summary,
    ) = topk_overlap_summary(
        ranks,
        models=BASE_MODELS,
        top_k_values=[
            int(k)
            for k in args.top_k
        ],
    )

    daily_corr.to_csv(
        output_root
        / "daily_score_correlations.csv",
        index=False,
    )
    corr_summary.to_csv(
        output_root
        / "score_correlation_summary.csv",
        index=False,
    )
    daily_overlap.to_csv(
        output_root
        / "daily_topk_overlap.csv",
        index=False,
    )
    overlap_summary.to_csv(
        output_root
        / "topk_overlap_summary.csv",
        index=False,
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
                eval_start_date,
                eval_end_date,
                inclusive="both",
            ),
            "date",
        ].dropna()
    )
    (
        benchmark_table,
        benchmark_curves,
    ) = benchmark_rows(
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

    signal_definitions: dict[
        str,
        list[str],
    ] = {
        model: [
            model
        ]
        for model in BASE_MODELS
    }
    signal_definitions.update(
        ENSEMBLES
    )

    for signal_name, members in (
        signal_definitions.items()
    ):
        print(
            "\n================================"
        )
        print(
            f"SIGNAL: {signal_name}"
        )
        print(
            "Members: "
            + ", ".join(
                members
            )
        )
        print(
            "================================"
        )

        if len(members) == 1:
            scores = score_store[
                members[
                    0
                ]
            ]
        else:
            (
                scores,
                _,
            ) = build_consensus(
                df,
                eval_mask=eval_mask,
                score_store=score_store,
                models=members,
            )

        df[
            "persistent_score"
        ] = np.nan
        df.loc[
            eval_mask,
            "persistent_score",
        ] = scores
        add_daily_ranks(
            df
        )

        (
            leaderboard,
            annual,
            equity,
        ) = evaluate_scores(
            df,
            events,
            model_name=(
                signal_name
            ),
            eval_start_date=(
                eval_start_date
            ),
            eval_end_year=int(
                args.oos_end_year
            ),
            top_k_values=[
                int(k)
                for k in args.top_k
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
        leaderboard[
            "members"
        ] = ",".join(
            members
        )
        annual[
            "members"
        ] = ",".join(
            members
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

    leaderboard = pd.concat(
        leaderboard_frames,
        ignore_index=True,
    )
    annual = pd.concat(
        annual_frames,
        ignore_index=True,
    )
    equity = pd.concat(
        equity_frames,
        ignore_index=True,
    )

    leaderboard = (
        leaderboard.sort_values(
            [
                "economic_target_excess5_sharpe1",
                "sharpe",
                "excess_cagr",
                "cagr",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    (
        pair_diagnostics,
        return_corr_matrix,
    ) = portfolio_pair_diagnostics(
        equity
    )
    winners = annual_winners(
        annual
    )

    leaderboard.to_csv(
        output_root
        / "leaderboard.csv",
        index=False,
    )
    annual.to_csv(
        output_root
        / "annual_metrics.csv",
        index=False,
    )
    equity.to_parquet(
        output_root
        / "equity_curves.parquet",
        index=False,
        compression="zstd",
    )
    pair_diagnostics.to_csv(
        output_root
        / "portfolio_pair_diagnostics.csv",
        index=False,
    )
    return_corr_matrix.to_csv(
        output_root
        / "portfolio_return_correlation.csv"
    )
    winners.to_csv(
        output_root
        / "annual_winners_by_k.csv",
        index=False,
    )
    benchmark_table.to_csv(
        output_root
        / "benchmark_reference.csv",
        index=False,
    )
    pd.concat(
        diagnostics_frames,
        ignore_index=True,
    ).to_csv(
        output_root
        / "annual_prediction_diagnostics.csv",
        index=False,
    )

    summary = {
        "scope": (
            "exploratory complementarity "
            "analysis after the architecture "
            "tournament"
        ),
        "base_models": (
            BASE_MODELS
        ),
        "ensembles": (
            ENSEMBLES
        ),
        "ensemble_rule": (
            "equal-weight mean of within-date "
            "percentile ranks; no learned "
            "weights"
        ),
        "top_k": [
            int(k)
            for k in args.top_k
        ],
        "seed": int(
            args.seed
        ),
        "important_caveat": (
            "The choice to focus on these "
            "three boosting models is informed "
            "by already observed OOS results. "
            "Treat ensemble comparisons as "
            "exploratory until independently "
            "validated or nested."
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

    preview_cols = [
        "model_name",
        "members",
        "top_k",
        "cagr",
        "benchmark_cagr",
        "excess_cagr",
        "alpha_annualized",
        "sharpe",
        "sortino",
        "max_drawdown",
        "median_annual_cagr",
        "q25_annual_cagr",
        "median_annual_sharpe",
        "q25_annual_sharpe",
        "turnover_multiple",
        "total_fees",
        "economic_target_excess5_sharpe1",
        "factor_alpha5_sharpe1",
    ]

    print(
        "\n=== SCORE COMPLEMENTARITY ==="
    )
    print(
        corr_summary.to_string(
            index=False,
            formatters={
                "mean_spearman": (
                    lambda x: f"{x:.3f}"
                ),
                "median_spearman": (
                    lambda x: f"{x:.3f}"
                ),
                "q25_spearman": (
                    lambda x: f"{x:.3f}"
                ),
                "q75_spearman": (
                    lambda x: f"{x:.3f}"
                ),
            },
        )
    )

    print(
        "\n=== TOP-K OVERLAP ==="
    )
    print(
        overlap_summary.to_string(
            index=False,
            formatters={
                "mean_overlap_fraction": (
                    lambda x: f"{x:.3f}"
                ),
                "median_overlap_fraction": (
                    lambda x: f"{x:.3f}"
                ),
                "q25_overlap_fraction": (
                    lambda x: f"{x:.3f}"
                ),
                "mean_jaccard": (
                    lambda x: (
                        ""
                        if pd.isna(x)
                        else f"{x:.3f}"
                    )
                ),
            },
        )
    )

    print(
        "\n=== BOOSTING / ENSEMBLE LEADERBOARD ==="
    )
    preview = leaderboard[
        preview_cols
    ].copy()
    print(
        preview.to_string(
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
                "alpha_annualized": (
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

    hits = leaderboard.loc[
        leaderboard[
            "economic_target_excess5_sharpe1"
        ]
    ]
    print(
        "\n=== TARGET HITS ==="
    )
    if hits.empty:
        print(
            "No fixed equal-rank signal "
            "currently clears both +5pp "
            "NIFTY500 excess CAGR and "
            "Sharpe > 1."
        )
    else:
        print(
            hits[
                [
                    "model_name",
                    "members",
                    "top_k",
                    "cagr",
                    "excess_cagr",
                    "alpha_annualized",
                    "sharpe",
                    "max_drawdown",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nOutputs: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
