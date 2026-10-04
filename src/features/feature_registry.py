from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    source: str
    availability: str
    lookback: str
    point_in_time: bool = True
    model_enabled: bool = True
    requires_context: bool = False
    notes: str = ""


F0_BASELINE = [
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

F1_OHLC_GEOMETRY = [
    "close_location_1d",
    "body_to_range_1d",
    "upper_wick_pct_1d",
    "lower_wick_pct_1d",
    "true_range_pct_1d",
]

F2_STREAK_STATE = [
    "up_close_streak",
    "down_close_streak",
    "bullish_candle_streak",
    "bearish_candle_streak",
    "higher_high_streak",
    "lower_low_streak",
]

F3_STOCK_TENDENCIES = [
    "intraday_mean_20d",
    "intraday_mean_60d",
    "intraday_mean_252d",
    "gap_mean_20d",
    "gap_mean_60d",
    "gap_mean_252d",
    "up_fraction_20d",
    "up_fraction_60d",
    "up_fraction_252d",
    "gap_up_fraction_20d",
    "gap_up_fraction_60d",
    "gap_up_fraction_252d",
    "close_location_mean_20d",
    "close_location_mean_60d",
    "hist_next1_after_down_streak_mean",
    "hist_next1_after_down_streak_up_prob",
    "hist_next1_after_down_streak_count",
    "hist_next1_after_up_streak_mean",
    "hist_next1_after_up_streak_up_prob",
    "hist_next1_after_up_streak_count",
]

F4_MARKET_CONTEXT = [
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_return_60d",
    "market_gap_return_1d",
    "market_intraday_return_1d",
    "market_volatility_20d",
    "market_volatility_60d",
    "stock_minus_market_1d",
    "stock_minus_market_5d",
    "stock_minus_market_20d",
    "stock_minus_market_60d",
]

F5_SECURITY_CONTEXT = [
    "largecap_flag_numeric",
    "market_cap_log",
    "market_cap_percentile",
]
F6_LEAD_LAG = [
    "market_beta_60d",
    "market_beta_252d",
    "market_corr_lag0_60d",
    "market_corr_lag1_60d",
    "market_corr_lag2_60d",
    "market_corr_lag3_60d",
    "market_corr_lag5_60d",
    "market_response_gap_1d",
    "market_response_gap_20d",
]

F8_CROSS_SECTIONAL = [
    "xs_rank_return_5d",
    "xs_rank_return_20d",
    "xs_rank_return_60d",
    "xs_rank_volatility_20d",
    "xs_rank_turnover_zscore_20d",
    "xs_rank_stock_minus_market_20d",
    "xs_rank_market_response_gap_1d",
]

F9_CALENDAR = [
    "day_of_week",
    "month_of_year",
    "quarter_of_year",
    "is_month_end",
    "is_quarter_end",
    "is_financial_year_end",
]

F10_LIQUIDITY_RISK = [
    "turnover_ratio_20d_60d",
    "volatility_ratio_5d_20d",
    "volatility_ratio_20d_60d",
    "range_mean_20d",
    "range_zscore_20d",
    "market_vol_regime_pct_252d",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "F0": F0_BASELINE,
    "F1": F1_OHLC_GEOMETRY,
    "F2": F2_STREAK_STATE,
    "F3": F3_STOCK_TENDENCIES,
    "F4": F4_MARKET_CONTEXT,
    "F5": F5_SECURITY_CONTEXT,
    "F6": F6_LEAD_LAG,
    "F8": F8_CROSS_SECTIONAL,
    "F9": F9_CALENDAR,
    "F10": F10_LIQUIDITY_RISK,
}

SAFE_DEFAULT_GROUPS = [
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "F6",
    "F8",
    "F9",
    "F10",
]

CONTEXT_GROUPS = [
    "F5",
]

ATTRIBUTION_ONLY_COLUMNS = [
    "calendar_year",
    "sector",
    "industry",
    "sector_index_name",
    "market_context_index",
]


def feature_columns(
    groups: Iterable[str],
) -> list[str]:
    out: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(
                f"Unknown feature group: {group}"
            )
        out.extend(
            FEATURE_GROUPS[group]
        )
    return list(
        dict.fromkeys(out)
    )


def cumulative_feature_sets(
    groups: Iterable[str] = SAFE_DEFAULT_GROUPS,
) -> dict[str, list[str]]:
    named: dict[str, list[str]] = {}
    running: list[str] = []
    labels: list[str] = []

    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(
                f"Unknown feature group: {group}"
            )
        labels.append(group)
        running.extend(
            FEATURE_GROUPS[group]
        )
        named[
            "+".join(labels)
        ] = list(
            dict.fromkeys(running)
        )

    return named


def registry() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    family_source = {
        "F0": "existing_research_panel",
        "F1": "daily_adjusted_ohlc",
        "F2": "daily_adjusted_ohlc",
        "F3": "stock_history_through_close_t",
        "F4": "nse_historical_index_data",
        "F5": "point_in_time_security_context",
        "F6": "stock_and_market_history_through_close_t",
        "F8": "same_day_cross_section_at_close_t",
        "F9": "calendar_known_ex_ante",
        "F10": "stock_and_market_history_through_close_t",
    }

    for family, names in FEATURE_GROUPS.items():
        for name in names:
            specs.append(
                FeatureSpec(
                    name=name,
                    family=family,
                    source=family_source[
                        family
                    ],
                    availability="close_t",
                    lookback=(
                        "mixed_trailing"
                    ),
                    point_in_time=True,
                    model_enabled=True,
                    requires_context=(
                        family
                        in CONTEXT_GROUPS
                    ),
                )
            )

    for name in ATTRIBUTION_ONLY_COLUMNS:
        specs.append(
            FeatureSpec(
                name=name,
                family="ATTRIBUTION",
                source=(
                    "calendar_or_security_context"
                ),
                availability="close_t",
                lookback="none",
                point_in_time=True,
                model_enabled=False,
                requires_context=(
                    name
                    in {
                        "sector",
                        "industry",
                        "sector_index_name",
                    }
                ),
                notes=(
                    "Never silently promoted to a "
                    "model feature."
                ),
            )
        )

    return specs


def registry_records() -> list[dict]:
    return [
        asdict(spec)
        for spec in registry()
    ]
