from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(
    __file__
).resolve()
ML_DIR = (
    SCRIPT_PATH.parent
)

if str(
    ML_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            ML_DIR
        ),
    )

from classical import (  # noqa: E402
    build_model,
    fit_model,
    predict_mask,
)


HORIZON = 20
TRAIN_START = pd.Timestamp(
    "2018-02-01"
)
TARGET_RANK = (
    "target_rank_20d_largecap"
)
TARGET_RETURN = (
    "target_next_open_to_close_20d"
)
TRAINING_FLAG = (
    "training_eligible_20d"
)

F0 = [
    "gap_return_1d",
    "intraday_return_1d",
    "range_pct_1d",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "close_to_ma_5d",
    "close_to_ma_20d",
    "close_to_ma_60d",
    "distance_high_5d",
    "distance_high_20d",
    "distance_high_60d",
    "distance_low_5d",
    "distance_low_20d",
    "distance_low_60d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "turnover_zscore_20d",
    "volume_zscore_20d",
    "active_day_ratio_20d",
    "log_turnover_median_20d",
    "log_turnover_median_60d",
]

F1 = [
    "body_pct_1d",
    "upper_wick_pct_1d",
    "lower_wick_pct_1d",
    "close_location_1d",
    "body_to_range_1d",
    "true_range_pct_1d",
    "position_in_range_20d",
    "position_in_range_60d",
    "position_in_range_252d",
]

F2 = [
    "close_direction_streak",
    "intraday_direction_streak",
    "high_direction_streak",
    "low_direction_streak",
    "gap_direction_streak",
]

F3 = [
    "hist_streak_mean_next_1d",
    "hist_streak_prob_up_next_1d",
    "hist_streak_prior_count",
]

# Curated broad-market context.  NIFTY 50 and NIFTY 500 relative
# returns were ~0.99 correlated in the audit, so retain one stock-
# relative representation and explicitly retain the 50-vs-500 spread.
F4 = [
    "stock_minus_nifty50_1d",
    "stock_minus_nifty50_5d",
    "stock_minus_nifty50_20d",
    "stock_minus_nifty50_60d",
    "stock_vs_nifty50_corr_60d",
    "stock_vs_nifty50_beta_60d",
    "stock_vs_nifty50_corr_252d",
    "stock_vs_nifty50_beta_252d",
    "nifty50_minus_nifty500_1d",
    "nifty50_minus_nifty500_5d",
    "nifty50_minus_nifty500_20d",
    "nifty50_minus_nifty500_60d",
]

# Sector behavior, not sector identity.  lag0 lead-corr is omitted
# because it is exactly the 120d stock/sector correlation.  The 252d
# response gap is omitted because it was ~0.992 correlated with 120d.
F5A = [
    "sector_proxy_corr_120d",
    "sector_proxy_corr_252d",
    "sector_proxy_beta_120d",
    "sector_proxy_beta_252d",
    "sector_proxy_corr_margin_252d",
    "sector_proxy_return_1d",
    "sector_proxy_return_5d",
    "sector_proxy_return_20d",
    "sector_proxy_return_60d",
    "stock_minus_sector_1d",
    "stock_minus_sector_5d",
    "stock_minus_sector_20d",
    "stock_minus_sector_60d",
    "sector_minus_nifty50_1d",
    "sector_minus_nifty50_5d",
    "sector_minus_nifty50_20d",
    "sector_minus_nifty50_60d",
    "sector_strength_rank_1d",
    "sector_strength_rank_5d",
    "sector_strength_rank_20d",
    "sector_strength_rank_60d",
    "sector_response_gap_120d",
    "sector_leads_stock_corr_lag1_120d",
    "sector_leads_stock_corr_lag2_120d",
    "sector_leads_stock_corr_lag3_120d",
    "sector_leads_stock_corr_lag5_120d",
]

F5B = [
    "sector_proxy_is_nifty_auto",
    "sector_proxy_is_nifty_bank",
    "sector_proxy_is_nifty_financial_services",
    "sector_proxy_is_nifty_fmcg",
    "sector_proxy_is_nifty_it",
    "sector_proxy_is_nifty_media",
    "sector_proxy_is_nifty_metal",
    "sector_proxy_is_nifty_oil_and_gas",
    "sector_proxy_is_nifty_pharma",
    "sector_proxy_is_nifty_private_bank",
    "sector_proxy_is_nifty_psu_bank",
    "sector_proxy_is_nifty_realty",
]

# Broad-market daily lead/lag.  Keep one index family to avoid giving
# two almost identical versions of every relationship to the trees.
F6 = [
    "nifty50_response_gap_60d",
    "nifty50_response_gap_252d",
    *[
        (
            "nifty50_leads_stock_"
            f"lag{lag}_{kind}_120d"
        )
        for lag in (
            0,
            1,
            2,
            3,
            5,
        )
        for kind in (
            "corr",
            "beta",
        )
    ],
    *[
        (
            "stock_leads_nifty50_"
            f"lag{lag}_{kind}_120d"
        )
        for lag in (
            1,
            2,
            3,
            5,
        )
        for kind in (
            "corr",
            "beta",
        )
    ],
]

# Use one stationary relative-size representation.  Raw rank and raw
# market cap remain attribution metadata, not predictor duplicates.
F7 = [
    "amfi_mcap_vs_largecap_median",
]

F8 = [
    "cross_section_rank_return_5d",
    "cross_section_rank_return_20d",
    "cross_section_rank_return_60d",
    "cross_section_rank_turnover",
]

F8_LARGECAP = [
    "largecap_rank_return_5d",
    "largecap_rank_return_20d",
    "largecap_rank_return_60d",
    "largecap_rank_turnover",
]

F9 = [
    "day_of_week",
    "month_of_year",
    "quarter_of_year",
    "is_month_end",
    "is_quarter_end",
    "is_fiscal_year_end_month",
]

FAMILIES = {
    "F0_baseline": F0,
    "F1_ohlc": F1,
    "F2_streak": F2,
    "F3_tendency": F3,
    "F4_market": F4,
    "F5a_sector_dynamics": F5A,
    "F5b_sector_identity": F5B,
    "F6_lead_lag": F6,
    "F7_size": F7,
    "F8_cross_section": F8,
    "F8_largecap_cross_section": F8_LARGECAP,
    "F9_calendar": F9,
}

ABLATION_ORDER = [
    (
        "A0_F0",
        [
            "F0_baseline",
        ],
    ),
    (
        "A1_plus_F1",
        [
            "F0_baseline",
            "F1_ohlc",
        ],
    ),
    (
        "A2_plus_F2",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
        ],
    ),
    (
        "A3_plus_F3",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
        ],
    ),
    (
        "A4_plus_F4",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
        ],
    ),
    (
        "A5_plus_F5a",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
        ],
    ),
    (
        "A6_plus_F5b",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
            "F5b_sector_identity",
        ],
    ),
    (
        "A7_plus_F6",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
            "F5b_sector_identity",
            "F6_lead_lag",
        ],
    ),
    (
        "A8_plus_F7",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
            "F5b_sector_identity",
            "F6_lead_lag",
            "F7_size",
        ],
    ),
    (
        "A9_plus_F8",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
            "F5b_sector_identity",
            "F6_lead_lag",
            "F7_size",
            "F8_cross_section",
        ],
    ),
    (
        "A10_plus_F9",
        [
            "F0_baseline",
            "F1_ohlc",
            "F2_streak",
            "F3_tendency",
            "F4_market",
            "F5a_sector_dynamics",
            "F5b_sector_identity",
            "F6_lead_lag",
            "F7_size",
            "F8_cross_section",
            "F9_calendar",
        ],
    ),
]

BRANCH_BASE_FAMILIES = [
    "F0_baseline",
    "F1_ohlc",
    "F2_streak",
    "F3_tendency",
    "F4_market",
    "F5a_sector_dynamics",
]

BRANCH_ORDER = [
    (
        "B0_A5_core",
        list(
            BRANCH_BASE_FAMILIES
        ),
    ),
    (
        "B1_core_plus_F5b",
        [
            *BRANCH_BASE_FAMILIES,
            "F5b_sector_identity",
        ],
    ),
    (
        "B2_core_plus_F6",
        [
            *BRANCH_BASE_FAMILIES,
            "F6_lead_lag",
        ],
    ),
    (
        "B3_core_plus_F7",
        [
            *BRANCH_BASE_FAMILIES,
            "F7_size",
        ],
    ),
    (
        "B4_core_plus_F8",
        [
            *BRANCH_BASE_FAMILIES,
            "F8_cross_section",
        ],
    ),
    (
        "B5_core_plus_F9",
        [
            *BRANCH_BASE_FAMILIES,
            "F9_calendar",
        ],
    ),
    (
        "B6_core_plus_F8_largecap",
        [
            *BRANCH_BASE_FAMILIES,
            "F8_largecap_cross_section",
        ],
    ),
    (
        "B7_core_plus_F8_both",
        [
            *BRANCH_BASE_FAMILIES,
            "F8_cross_section",
            "F8_largecap_cross_section",
        ],
    ),
]

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


def unique(
    values: list[str],
) -> list[str]:
    return list(
        dict.fromkeys(
            values
        )
    )


def feature_columns(
    families: list[str],
) -> list[str]:
    return unique([
        feature
        for family in families
        for feature in FAMILIES[
            family
        ]
    ])


def load_panel(
    panel_root: Path,
    *,
    required_features: list[str],
) -> pd.DataFrame:
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

    base = [
        "date",
        "market_day_index",
        "canonical_security_id",
        "symbol",
        "eligible_universe",
        TRAINING_FLAG,
        TARGET_RANK,
        TARGET_RETURN,
    ]

    requested = unique(
        base
        + required_features
    )

    schema = pq.read_schema(
        files[
            0
        ]
    )
    available = set(
        schema.names
    )

    missing = [
        column
        for column in requested
        if column not in available
    ]

    if missing:
        raise RuntimeError(
            "Large-cap model panel is missing "
            "predeclared ablation columns: "
            f"{missing}"
        )

    frames = []

    print(
        f"Loading {len(files):,} "
        "large-cap model-panel dates..."
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        frames.append(
            pd.read_parquet(
                path,
                columns=requested,
            )
        )

        if (
            number % 250 == 0
            or number
            == len(
                files
            )
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
        TRAINING_FLAG
    ] = (
        df[
            TRAINING_FLAG
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    df[
        "eligible_universe"
    ] = (
        df[
            "eligible_universe"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    for column in unique(
        [
            TARGET_RANK,
            TARGET_RETURN,
            *required_features,
        ]
    ):
        df[
            column
        ] = (
            pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
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


def daily_diagnostics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for date, group in frame.groupby(
        "date",
        sort=True,
    ):
        group = group.loc[
            group[
                "score"
            ].notna()
            & group[
                TARGET_RANK
            ].notna()
            & group[
                TARGET_RETURN
            ].notna()
        ].copy()

        if len(
            group
        ) < 20:
            continue

        score_rank = group[
            "score"
        ].rank(
            method="average",
            pct=True,
        )

        ic = pd.DataFrame({
            "score": score_rank,
            "target": group[
                TARGET_RANK
            ],
        }).corr(
            method="spearman"
        ).iloc[
            0,
            1,
        ]

        top = group.loc[
            score_rank.ge(
                0.90
            )
        ]
        bottom = group.loc[
            score_rank.le(
                0.10
            )
        ]

        universe_return = float(
            group[
                TARGET_RETURN
            ].mean()
        )

        rows.append({
            "date": date,
            "rows": int(
                len(
                    group
                )
            ),
            "ic": (
                float(
                    ic
                )
                if pd.notna(
                    ic
                )
                else np.nan
            ),
            "top_decile_return": float(
                top[
                    TARGET_RETURN
                ].mean()
            ),
            "bottom_decile_return": float(
                bottom[
                    TARGET_RETURN
                ].mean()
            ),
            "universe_return": (
                universe_return
            ),
            "top_decile_excess": float(
                top[
                    TARGET_RETURN
                ].mean()
                - universe_return
            ),
            "top_minus_bottom": float(
                top[
                    TARGET_RETURN
                ].mean()
                - bottom[
                    TARGET_RETURN
                ].mean()
            ),
        })

    return pd.DataFrame(
        rows
    )


def aggregate_diagnostics(
    daily: pd.DataFrame,
) -> dict:
    if daily.empty:
        return {
            "dates": 0,
            "mean_daily_ic": None,
            "median_daily_ic": None,
            "ic_std": None,
            "icir": None,
            "positive_ic_fraction": None,
            "mean_top_decile_excess": None,
            "mean_top_minus_bottom": None,
        }

    ic = pd.to_numeric(
        daily[
            "ic"
        ],
        errors="coerce",
    ).dropna()

    mean_ic = float(
        ic.mean()
    ) if len(
        ic
    ) else None

    std_ic = float(
        ic.std(
            ddof=1
        )
    ) if len(
        ic
    ) > 1 else None

    return {
        "dates": int(
            len(
                daily
            )
        ),
        "mean_daily_ic": (
            mean_ic
        ),
        "median_daily_ic": (
            float(
                ic.median()
            )
            if len(
                ic
            )
            else None
        ),
        "ic_std": std_ic,
        "icir": (
            float(
                mean_ic
                / std_ic
            )
            if (
                mean_ic
                is not None
                and std_ic
                is not None
                and std_ic > 0
            )
            else None
        ),
        "positive_ic_fraction": (
            float(
                ic.gt(
                    0
                ).mean()
            )
            if len(
                ic
            )
            else None
        ),
        "mean_top_decile_excess": float(
            daily[
                "top_decile_excess"
            ].mean()
        ),
        "mean_top_minus_bottom": float(
            daily[
                "top_minus_bottom"
            ].mean()
        ),
    }


def fit_oos_year(
    df: pd.DataFrame,
    *,
    year: int,
    features: list[str],
    model_name: str,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    fold_start = pd.Timestamp(
        f"{year}-01-01"
    )
    fold_end = pd.Timestamp(
        f"{year}-12-31"
    )

    prior = (
        df.loc[
            df[
                "date"
            ].lt(
                fold_start
            ),
            "market_day_index",
        ]
        .dropna()
        .astype(
            int
        )
    )

    if prior.empty:
        raise RuntimeError(
            f"No prior sessions for {year}"
        )

    cutoff = int(
        prior.max()
    )

    train_mask = (
        df[
            TRAINING_FLAG
        ]
        & df[
            TARGET_RANK
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
        ].between(
            fold_start,
            fold_end,
            inclusive="both",
        )
    )

    print(
        f"    {year}: train="
        f"{int(train_mask.sum()):,} "
        f"score={int(score_mask.sum()):,}"
    )

    model = build_model(
        model_name,
        random_state=(
            random_state
        ),
    )

    fit_model(
        model,
        df,
        train_mask,
        feature_columns=features,
        target_column=TARGET_RANK,
    )

    scores = predict_mask(
        model,
        df,
        score_mask,
        feature_columns=features,
    )

    prediction = df.loc[
        score_mask,
        [
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            TARGET_RANK,
            TARGET_RETURN,
            TRAINING_FLAG,
        ],
    ].copy()

    prediction[
        "score"
    ] = scores

    label_frame = (
        prediction.loc[
            prediction[
                TRAINING_FLAG
            ]
            & prediction[
                TARGET_RANK
            ].notna()
        ]
        .copy()
    )

    daily = daily_diagnostics(
        label_frame
    )

    metrics = (
        aggregate_diagnostics(
            daily
        )
    )

    metrics.update({
        "year": int(
            year
        ),
        "train_rows": int(
            train_mask.sum()
        ),
        "score_rows": int(
            score_mask.sum()
        ),
        "labeled_score_rows": int(
            len(
                label_frame
            )
        ),
    })

    return (
        prediction,
        metrics,
    )


def experiment_sets(
    *,
    stage: str,
    selected: str | None,
) -> list[
    tuple[
        str,
        list[str],
    ]
]:
    if stage == "development":
        return ABLATION_ORDER

    if stage == "branch":
        return BRANCH_ORDER

    if selected is None:
        raise ValueError(
            "--feature-set is required "
            "for confirmation stage."
        )

    mapping = {
        **dict(
            ABLATION_ORDER
        ),
        **dict(
            BRANCH_ORDER
        ),
    }

    if selected not in mapping:
        raise ValueError(
            "Unknown feature set. Choose one of: "
            + ", ".join(
                mapping
            )
        )

    return [
        (
            selected,
            mapping[
                selected
            ],
        )
    ]


def run(
    root: Path,
    *,
    stage: str,
    selected: str | None,
    model_name: str,
    random_state: int,
    rebuild: bool,
) -> dict:
    root = Path(
        root
    ).resolve()

    experiments = (
        experiment_sets(
            stage=stage,
            selected=selected,
        )
    )

    years = (
        DEVELOPMENT_YEARS
        if stage
        in {
            "development",
            "branch",
        }
        else CONFIRMATION_YEARS
    )

    all_required = unique([
        feature
        for _,
        families
        in experiments
        for feature in feature_columns(
            families
        )
    ])

    panel = load_panel(
        root
        / "data/processed/"
        "largecap_model_panel_v2",
        required_features=(
            all_required
        ),
    )

    reports = (
        root
        / "reports/ml/"
        "largecap_feature_ablation"
        / stage
    )
    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "stage": stage,
        "years": years,
        "model": model_name,
        "random_state": (
            random_state
        ),
        "train_start": str(
            TRAIN_START.date()
        ),
        "purge_sessions": (
            HORIZON
        ),
        "target": TARGET_RANK,
        "target_return": (
            TARGET_RETURN
        ),
        "experiments": {
            name: {
                "families": (
                    families
                ),
                "features": (
                    feature_columns(
                        families
                    )
                ),
            }
            for name, families
            in experiments
        },
        "development_rule": (
            "Cumulative and branch feature-family "
            "comparisons are restricted to 2021-2023 "
            "annual walk-forward OOS. Branch mode "
            "holds the A5 sector-dynamics core fixed "
            "and adds exactly one remaining family. "
            "Confirmation mode accepts one frozen "
            "feature set for 2024-2026."
        ),
        "portfolio_search": False,
    }

    (
        reports
        / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    leaderboard = []
    annual_rows = []
    daily_frames = []

    for (
        experiment_name,
        families,
    ) in experiments:
        features = feature_columns(
            families
        )

        print(
            "\n=== "
            f"{experiment_name} "
            f"({len(features)} features) ==="
        )

        prediction_dir = (
            reports
            / "predictions"
            / experiment_name
        )
        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        experiment_predictions = []
        experiment_annual = []

        for year in years:
            path = (
                prediction_dir
                / f"{year}.parquet"
            )

            if (
                path.is_file()
                and not rebuild
            ):
                prediction = (
                    pd.read_parquet(
                        path
                    )
                )

                prediction[
                    "date"
                ] = pd.to_datetime(
                    prediction[
                        "date"
                    ],
                    errors="coerce",
                ).dt.normalize()

                label_frame = (
                    prediction.loc[
                        prediction[
                            TRAINING_FLAG
                        ].fillna(
                            False
                        )
                        & prediction[
                            TARGET_RANK
                        ].notna()
                    ]
                    .copy()
                )

                daily = (
                    daily_diagnostics(
                        label_frame
                    )
                )

                metrics = (
                    aggregate_diagnostics(
                        daily
                    )
                )

                metrics.update({
                    "year": int(
                        year
                    ),
                    "train_rows": None,
                    "score_rows": int(
                        len(
                            prediction
                        )
                    ),
                    "labeled_score_rows": int(
                        len(
                            label_frame
                        )
                    ),
                })

                print(
                    f"    {year}: reused "
                    f"{len(prediction):,} "
                    "cached scores"
                )

            else:
                (
                    prediction,
                    metrics,
                ) = fit_oos_year(
                    panel,
                    year=year,
                    features=features,
                    model_name=(
                        model_name
                    ),
                    random_state=(
                        random_state
                    ),
                )

                prediction.to_parquet(
                    path,
                    index=False,
                    compression="zstd",
                )

                daily = (
                    daily_diagnostics(
                        prediction.loc[
                            prediction[
                                TRAINING_FLAG
                            ]
                            & prediction[
                                TARGET_RANK
                            ].notna()
                        ]
                    )
                )

            prediction[
                "ablation"
            ] = experiment_name

            experiment_predictions.append(
                prediction
            )

            metrics.update({
                "ablation": (
                    experiment_name
                ),
                "feature_count": int(
                    len(
                        features
                    )
                ),
                "families": (
                    "+".join(
                        families
                    )
                ),
            })

            experiment_annual.append(
                metrics
            )
            annual_rows.append(
                metrics
            )

            daily[
                "year"
            ] = int(
                year
            )
            daily[
                "ablation"
            ] = (
                experiment_name
            )
            daily_frames.append(
                daily
            )

        combined_prediction = (
            pd.concat(
                experiment_predictions,
                ignore_index=True,
            )
        )

        combined_daily = (
            daily_diagnostics(
                combined_prediction.loc[
                    combined_prediction[
                        TRAINING_FLAG
                    ].fillna(
                        False
                    )
                    & combined_prediction[
                        TARGET_RANK
                    ].notna()
                ]
            )
        )

        aggregate = (
            aggregate_diagnostics(
                combined_daily
            )
        )

        annual_frame = pd.DataFrame(
            experiment_annual
        )

        aggregate.update({
            "ablation": (
                experiment_name
            ),
            "feature_count": int(
                len(
                    features
                )
            ),
            "families": (
                "+".join(
                    families
                )
            ),
            "oos_year_start": int(
                min(
                    years
                )
            ),
            "oos_year_end": int(
                max(
                    years
                )
            ),
            "median_annual_ic": float(
                pd.to_numeric(
                    annual_frame[
                        "mean_daily_ic"
                    ],
                    errors="coerce",
                ).median()
            ),
            "worst_annual_ic": float(
                pd.to_numeric(
                    annual_frame[
                        "mean_daily_ic"
                    ],
                    errors="coerce",
                ).min()
            ),
            "positive_annual_ic_fraction": float(
                pd.to_numeric(
                    annual_frame[
                        "mean_daily_ic"
                    ],
                    errors="coerce",
                )
                .gt(
                    0
                )
                .mean()
            ),
        })

        leaderboard.append(
            aggregate
        )

        print(
            "  OOS "
            f"IC={aggregate['mean_daily_ic']:+.4f} "
            "top-excess="
            f"{aggregate['mean_top_decile_excess']:+.3%} "
            "top-bottom="
            f"{aggregate['mean_top_minus_bottom']:+.3%}"
        )

    leaderboard_frame = (
        pd.DataFrame(
            leaderboard
        )
    )

    annual_frame = pd.DataFrame(
        annual_rows
    )

    daily_frame = (
        pd.concat(
            daily_frames,
            ignore_index=True,
        )
        if daily_frames
        else pd.DataFrame()
    )

    # Preserve the declared ablation order rather than sorting by the
    # observed metric: this is a feature-family experiment, not a
    # leaderboard-driven feature search.
    order_source = (
        BRANCH_ORDER
        if stage
        == "branch"
        else ABLATION_ORDER
    )

    declared_order = {
        name: i
        for i, (
            name,
            _,
        )
        in enumerate(
            order_source
        )
    }

    leaderboard_frame[
        "_order"
    ] = leaderboard_frame[
        "ablation"
    ].map(
        declared_order
    )

    leaderboard_frame = (
        leaderboard_frame.sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    leaderboard_frame.to_csv(
        reports
        / "leaderboard.csv",
        index=False,
    )
    annual_frame.to_csv(
        reports
        / "annual_diagnostics.csv",
        index=False,
    )
    daily_frame.to_csv(
        reports
        / "daily_diagnostics.csv",
        index=False,
    )

    if len(
        leaderboard_frame
    ) > 1:
        delta = (
            leaderboard_frame[
                [
                    "ablation",
                    "feature_count",
                    "mean_daily_ic",
                    "mean_top_decile_excess",
                    "mean_top_minus_bottom",
                ]
            ]
            .copy()
        )

        for column in (
            "mean_daily_ic",
            "mean_top_decile_excess",
            "mean_top_minus_bottom",
        ):
            delta[
                f"delta_{column}"
            ] = (
                delta[
                    column
                ].diff()
            )

        delta.to_csv(
            reports
            / "incremental_family_deltas.csv",
            index=False,
        )

    summary = {
        "stage": stage,
        "model": model_name,
        "years": years,
        "experiments": int(
            len(
                experiments
            )
        ),
        "panel_rows": int(
            len(
                panel
            )
        ),
        "panel_dates": int(
            panel[
                "date"
            ].nunique()
        ),
        "output_root": str(
            reports
        ),
        "confirmation_isolation": (
            "2024-2026 are not evaluated by "
            "development mode; confirmation "
            "requires one explicitly selected "
            "frozen ablation."
        ),
    }

    (
        reports
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
        name
        for name, _
        in ABLATION_ORDER
    ]

    assert len(
        names
    ) == len(
        set(
            names
        )
    )

    prior = set()

    for _name, families in (
        ABLATION_ORDER
    ):
        current = set(
            feature_columns(
                families
            )
        )

        assert prior.issubset(
            current
        )

        prior = current

    branch_names = [
        name
        for name, _
        in BRANCH_ORDER
    ]
    assert len(
        branch_names
    ) == len(
        set(
            branch_names
        )
    )

    core = set(
        feature_columns(
            BRANCH_BASE_FAMILIES
        )
    )

    assert set(
        feature_columns(
            BRANCH_ORDER[
                0
            ][
                1
            ]
        )
    ) == core

    for _name, families in (
        BRANCH_ORDER[
            1:
        ]
    ):
        current = set(
            feature_columns(
                families
            )
        )
        assert core.issubset(
            current
        )
        assert len(
            current
            - core
        ) > 0

    # Exact duplicates identified by the F5 audit must not be in the
    # predictive feature manifest.
    full = set(
        feature_columns(
            list(
                FAMILIES
            )
        )
    )

    assert (
        "sector_leads_stock_corr_lag0_120d"
        not in full
    )

    assert (
        "sector_response_gap_252d"
        not in full
    )

    assert (
        "amfi_market_cap_rank"
        not in full
    )

    print(
        "Large-cap feature-ablation "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Predeclared feature-family ablation "
            "for point-in-time Indian large-cap "
            "20-session cross-sectional ranking."
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
            "branch",
            "confirmation",
        ],
        default="development",
    )
    ap.add_argument(
        "--feature-set",
        default=None,
        help=(
            "Required in confirmation mode. "
            "Example: B4_core_plus_F8"
        ),
    )
    ap.add_argument(
        "--model",
        default="xgboost",
        choices=[
            "xgboost",
            "hist_gb_fixed",
            "lightgbm",
            "catboost",
        ],
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--rebuild",
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

    summary = run(
        Path(
            args.root
        ),
        stage=args.stage,
        selected=args.feature_set,
        model_name=args.model,
        random_state=(
            args.random_state
        ),
        rebuild=args.rebuild,
    )

    print(
        "\n=== LARGE-CAP FEATURE ABLATION COMPLETE ==="
    )
    print(
        f"Stage:       "
        f"{summary['stage']}"
    )
    print(
        f"Model:       "
        f"{summary['model']}"
    )
    print(
        f"Years:       "
        f"{summary['years']}"
    )
    print(
        f"Experiments: "
        f"{summary['experiments']}"
    )
    print(
        f"Output:      "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
