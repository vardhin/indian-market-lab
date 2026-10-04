from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RETURN_HORIZONS = (
    1,
    5,
    20,
    60,
)
AFFINITY_WINDOWS = (
    120,
    252,
)
LEAD_LAGS = (
    0,
    1,
    2,
    3,
    5,
)

RESEARCH_COLUMNS = [
    "date",
    "market_day_index",
    "canonical_security_id",
    "symbol",
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
]

PIT_COLUMNS = [
    "date",
    "canonical_security_id",
    "pit_largecap_universe",
    "amfi_market_cap_rank",
    "amfi_average_market_cap_cr",
]


def load_largecap_keys(
    metadata_root: Path,
) -> pd.DataFrame:
    files = sorted(
        metadata_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No PIT metadata partitions under "
            f"{metadata_root}"
        )

    rows = []

    print(
        f"Scanning {len(files):,} PIT metadata "
        "partitions for large-cap rows..."
    )

    for i, path in enumerate(
        files,
        start=1,
    ):
        schema = pq.read_schema(
            path
        )
        missing = [
            column
            for column in PIT_COLUMNS
            if column
            not in schema.names
        ]
        if missing:
            raise RuntimeError(
                f"{path}: missing PIT columns "
                f"{missing}"
            )

        frame = pd.read_parquet(
            path,
            columns=PIT_COLUMNS,
        )

        frame = frame.loc[
            frame[
                "pit_largecap_universe"
            ].fillna(
                False
            )
        ].copy()

        if not frame.empty:
            rows.append(
                frame
            )

        if (
            i % 500 == 0
            or i == len(
                files
            )
        ):
            print(
                f"  scanned {i:,}/"
                f"{len(files):,}",
                flush=True,
            )

    if not rows:
        raise RuntimeError(
            "PIT metadata contains no "
            "large-cap universe rows."
        )

    keys = pd.concat(
        rows,
        ignore_index=True,
    )

    keys[
        "date"
    ] = pd.to_datetime(
        keys[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    duplicate = (
        keys.duplicated(
            [
                "date",
                "canonical_security_id",
            ],
            keep=False,
        )
    )

    if duplicate.any():
        raise RuntimeError(
            "Duplicate PIT large-cap "
            "date/security keys."
        )

    return keys.sort_values(
        [
            "canonical_security_id",
            "date",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def load_research_history(
    panel_root: Path,
    *,
    securities: set[str],
    start: pd.Timestamp,
) -> pd.DataFrame:
    files = sorted(
        panel_root.glob(
            "date=*/data.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No frozen research-panel "
            f"partitions under {panel_root}"
        )

    frames = []

    print(
        "Loading research history for "
        f"{len(securities):,} securities..."
    )

    for i, path in enumerate(
        files,
        start=1,
    ):
        partition = (
            path.parent.name
        )

        if partition.startswith(
            "date="
        ):
            date = pd.Timestamp(
                partition.split(
                    "=",
                    1,
                )[
                    1
                ]
            ).normalize()

            if date < start:
                continue

        schema = pq.read_schema(
            path
        )
        missing = [
            column
            for column in RESEARCH_COLUMNS
            if column
            not in schema.names
        ]
        if missing:
            raise RuntimeError(
                f"{path}: missing research "
                f"columns {missing}"
            )

        frame = pd.read_parquet(
            path,
            columns=RESEARCH_COLUMNS,
        )

        frame = frame.loc[
            frame[
                "canonical_security_id"
            ].isin(
                securities
            )
        ].copy()

        if not frame.empty:
            frames.append(
                frame
            )

        if (
            i % 500 == 0
            or i == len(
                files
            )
        ):
            print(
                f"  inspected {i:,}/"
                f"{len(files):,}",
                flush=True,
            )

    if not frames:
        raise RuntimeError(
            "No research history found for "
            "ever-large-cap securities."
        )

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history[
        "date"
    ] = pd.to_datetime(
        history[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    return history.sort_values(
        [
            "canonical_security_id",
            "date",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def load_sector_returns(
    path: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    if not path.is_file():
        raise FileNotFoundError(
            "Sector benchmark dataset not "
            f"found: {path}"
        )

    frame = pd.read_parquet(
        path
    )

    required = {
        "date",
        "sector_key",
        "requested_index",
        *[
            f"return_{horizon}d"
            for horizon in (
                1,
                2,
                3,
                5,
                20,
                60,
            )
        ],
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )
    if missing:
        raise RuntimeError(
            "Sector benchmark dataset lacks "
            f"{sorted(missing)}"
        )

    frame[
        "date"
    ] = pd.to_datetime(
        frame[
            "date"
        ],
        errors="coerce",
    ).dt.normalize()

    sector_keys = sorted(
        frame[
            "sector_key"
        ].dropna().unique().tolist()
    )

    wide_parts = []

    for horizon in (
        1,
        2,
        3,
        5,
        20,
        60,
    ):
        pivot = frame.pivot(
            index="date",
            columns="sector_key",
            values=(
                f"return_{horizon}d"
            ),
        )
        pivot.columns = [
            (
                f"sector__{column}"
                f"__return_{horizon}d"
            )
            for column
            in pivot.columns
        ]
        wide_parts.append(
            pivot
        )

    wide = pd.concat(
        wide_parts,
        axis=1,
    ).reset_index()

    return (
        wide,
        sector_keys,
    )


def load_broad_market_returns(
    root: Path,
) -> pd.DataFrame:
    path = (
        root
        / "data/processed/"
        "index_benchmarks/"
        "nse_price_indices.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Missing broad index dataset: "
            f"{path}"
        )

    frame = pd.read_parquet(
        path
    )

    parts = []

    for index_name, prefix in (
        ("NIFTY 50", "nifty50"),
        ("NIFTY 500", "nifty500"),
    ):
        part = frame.loc[
            frame[
                "requested_index"
            ].eq(
                index_name
            ),
            [
                "date",
                "close",
            ],
        ].copy()

        if part.empty:
            raise RuntimeError(
                f"Broad index dataset lacks "
                f"{index_name}."
            )

        part[
            "date"
        ] = pd.to_datetime(
            part[
                "date"
            ],
            errors="coerce",
        ).dt.normalize()

        part = (
            part.dropna()
            .sort_values(
                "date"
            )
            .drop_duplicates(
                "date",
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        for horizon in (
            RETURN_HORIZONS
        ):
            part[
                f"{prefix}_return_{horizon}d"
            ] = (
                part[
                    "close"
                ]
                / part[
                    "close"
                ].shift(
                    horizon
                )
                - 1.0
            )

        part = part.drop(
            columns=[
                "close"
            ]
        )

        parts.append(
            part
        )

    broad = parts[
        0
    ].merge(
        parts[
            1
        ],
        on="date",
        how="outer",
        validate="one_to_one",
    )

    for horizon in (
        RETURN_HORIZONS
    ):
        broad[
            (
                "nifty50_minus_nifty500_"
                f"{horizon}d"
            )
        ] = (
            broad[
                f"nifty50_return_{horizon}d"
            ]
            - broad[
                f"nifty500_return_{horizon}d"
            ]
        )

    return broad


def rolling_corr_beta(
    stock: pd.Series,
    basket: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    stock = pd.to_numeric(
        stock,
        errors="coerce",
    )
    basket = pd.to_numeric(
        basket,
        errors="coerce",
    )

    corr = stock.rolling(
        window,
        min_periods=min_periods,
    ).corr(
        basket
    )

    covariance = stock.rolling(
        window,
        min_periods=min_periods,
    ).cov(
        basket
    )

    variance = basket.rolling(
        window,
        min_periods=min_periods,
    ).var()

    beta = (
        covariance
        / variance.where(
            variance.abs()
            > 1e-15
        )
    )

    return (
        corr,
        beta,
    )


def compute_security_context(
    group: pd.DataFrame,
    *,
    sector_keys: list[str],
) -> pd.DataFrame:
    group = group.sort_values(
        "date",
        kind="stable",
    ).copy()

    n = len(
        group
    )
    m = len(
        sector_keys
    )

    corr_120 = np.full(
        (
            n,
            m,
        ),
        np.nan,
        dtype=float,
    )
    corr_252 = np.full(
        (
            n,
            m,
        ),
        np.nan,
        dtype=float,
    )
    beta_120 = np.full(
        (
            n,
            m,
        ),
        np.nan,
        dtype=float,
    )
    beta_252 = np.full(
        (
            n,
            m,
        ),
        np.nan,
        dtype=float,
    )

    lead_corr: dict[
        int,
        np.ndarray,
    ] = {
        lag: np.full(
            (
                n,
                m,
            ),
            np.nan,
            dtype=float,
        )
        for lag in LEAD_LAGS
    }

    stock_return = pd.to_numeric(
        group[
            "return_1d"
        ],
        errors="coerce",
    )

    for j, sector in enumerate(
        sector_keys
    ):
        basket_col = (
            f"sector__{sector}"
            "__return_1d"
        )

        basket = pd.to_numeric(
            group[
                basket_col
            ],
            errors="coerce",
        )

        c120, b120 = (
            rolling_corr_beta(
                stock_return,
                basket,
                window=120,
                min_periods=60,
            )
        )

        c252, b252 = (
            rolling_corr_beta(
                stock_return,
                basket,
                window=252,
                min_periods=120,
            )
        )

        corr_120[
            :,
            j,
        ] = c120.to_numpy(
            dtype=float
        )
        corr_252[
            :,
            j,
        ] = c252.to_numpy(
            dtype=float
        )
        beta_120[
            :,
            j,
        ] = b120.to_numpy(
            dtype=float
        )
        beta_252[
            :,
            j,
        ] = b252.to_numpy(
            dtype=float
        )

        for lag in LEAD_LAGS:
            lagged_basket = (
                basket.shift(
                    lag
                )
            )

            lead = stock_return.rolling(
                120,
                min_periods=60,
            ).corr(
                lagged_basket
            )

            lead_corr[
                lag
            ][
                :,
                j,
            ] = lead.to_numpy(
                dtype=float
            )

    # The basket used to contextualize session t is selected only from
    # relationships estimated through t-1.  This is stricter than the
    # after-close information boundary and avoids same-day adaptive
    # basket selection: today's stock move cannot decide which sector's
    # today's return is then used as its context.
    def lag_matrix(
        matrix: np.ndarray,
    ) -> np.ndarray:
        lagged = np.full_like(
            matrix,
            np.nan,
            dtype=float,
        )
        if len(
            matrix
        ) > 1:
            lagged[
                1:,
                :,
            ] = matrix[
                :-1,
                :,
            ]
        return lagged

    corr_120_pit = lag_matrix(
        corr_120
    )
    corr_252_pit = lag_matrix(
        corr_252
    )
    beta_120_pit = lag_matrix(
        beta_120
    )
    beta_252_pit = lag_matrix(
        beta_252
    )
    lead_corr_pit = {
        lag: lag_matrix(
            matrix
        )
        for lag, matrix
        in lead_corr.items()
    }

    # Sector affiliation is based on the highest positive trailing
    # 252-session correlation known through t-1. This is a market-
    # implied basket affinity, not an official taxonomy label.
    selection_matrix = (
        corr_252_pit.copy()
    )
    selection_matrix[
        ~np.isfinite(
            selection_matrix
        )
    ] = -np.inf

    selected = np.argmax(
        selection_matrix,
        axis=1,
    )

    best_value = (
        selection_matrix[
            np.arange(
                n
            ),
            selected,
        ]
    )

    sorted_affinity = np.sort(
        selection_matrix,
        axis=1,
    )
    second_value = (
        sorted_affinity[
            :,
            -2,
        ]
        if m >= 2
        else np.full(
            n,
            -np.inf,
            dtype=float,
        )
    )

    no_selection = (
        ~np.isfinite(
            best_value
        )
        | (
            best_value
            == -np.inf
        )
        | (
            best_value
            <= 0
        )
    )

    sector_array = np.array(
        sector_keys,
        dtype=object,
    )

    selected_name = (
        sector_array[
            selected
        ].astype(
            object
        )
    )

    selected_name[
        no_selection
    ] = None

    group[
        "sector_proxy_name"
    ] = selected_name

    row_index = np.arange(
        n
    )

    def choose(
        matrix: np.ndarray,
    ) -> np.ndarray:
        values = matrix[
            row_index,
            selected,
        ].copy()
        values[
            no_selection
        ] = np.nan
        return values

    group[
        "sector_proxy_corr_120d"
    ] = choose(
        corr_120_pit
    )
    group[
        "sector_proxy_corr_252d"
    ] = choose(
        corr_252_pit
    )
    group[
        "sector_proxy_beta_120d"
    ] = choose(
        beta_120_pit
    )
    group[
        "sector_proxy_beta_252d"
    ] = choose(
        beta_252_pit
    )

    margin = (
        best_value
        - second_value
    )
    margin[
        no_selection
        | ~np.isfinite(
            second_value
        )
    ] = np.nan

    group[
        "sector_proxy_corr_margin_252d"
    ] = margin

    group[
        "sector_proxy_confident"
    ] = (
        group[
            "sector_proxy_corr_252d"
        ].ge(
            0.20
        )
    )

    group[
        "sector_proxy_distinct"
    ] = (
        group[
            "sector_proxy_corr_margin_252d"
        ].ge(
            0.05
        )
    )

    for horizon in (
        RETURN_HORIZONS
    ):
        matrix = np.full(
            (
                n,
                m,
            ),
            np.nan,
            dtype=float,
        )

        for j, sector in enumerate(
            sector_keys
        ):
            matrix[
                :,
                j,
            ] = pd.to_numeric(
                group[
                    (
                        f"sector__{sector}"
                        f"__return_{horizon}d"
                    )
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

        sector_return = choose(
            matrix
        )

        group[
            f"sector_proxy_return_{horizon}d"
        ] = sector_return

        group[
            (
                "stock_minus_sector_"
                f"{horizon}d"
            )
        ] = (
            pd.to_numeric(
                group[
                    f"return_{horizon}d"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            - sector_return
        )

        nifty = pd.to_numeric(
            group[
                f"nifty50_return_{horizon}d"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        group[
            (
                "sector_minus_nifty50_"
                f"{horizon}d"
            )
        ] = (
            sector_return
            - nifty
        )

        # Same-date cross-sector leadership percentile.
        sector_values = np.full(
            (
                n,
                m,
            ),
            np.nan,
            dtype=float,
        )

        for j, sector in enumerate(
            sector_keys
        ):
            sector_values[
                :,
                j,
            ] = pd.to_numeric(
                group[
                    (
                        f"sector__{sector}"
                        f"__return_{horizon}d"
                    )
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

        ranks = pd.DataFrame(
            sector_values
        ).rank(
            axis=1,
            method="average",
            pct=True,
        ).to_numpy(
            dtype=float
        )

        group[
            (
                "sector_strength_rank_"
                f"{horizon}d"
            )
        ] = choose(
            ranks
        )

    current_sector_1d = pd.to_numeric(
        group[
            "sector_proxy_return_1d"
        ],
        errors="coerce",
    )

    group[
        "sector_response_gap_120d"
    ] = (
        pd.to_numeric(
            group[
                "return_1d"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            group[
                "sector_proxy_beta_120d"
            ],
            errors="coerce",
        )
        * current_sector_1d
    )

    group[
        "sector_response_gap_252d"
    ] = (
        pd.to_numeric(
            group[
                "return_1d"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            group[
                "sector_proxy_beta_252d"
            ],
            errors="coerce",
        )
        * current_sector_1d
    )

    for lag in LEAD_LAGS:
        group[
            (
                "sector_leads_stock_corr_"
                f"lag{lag}_120d"
            )
        ] = choose(
            lead_corr_pit[
                lag
            ]
        )

    for sector in sector_keys:
        group[
            (
                "sector_proxy_is_"
                + sector
            )
        ] = (
            group[
                "sector_proxy_name"
            ].eq(
                sector
            )
        ).astype(
            "int8"
        )

    return group


def build_sector_context(
    root: Path,
) -> dict:
    root = Path(
        root
    ).resolve()

    metadata_root = (
        root
        / "data/processed/"
        "pit_metadata_v2"
    )
    panel_root = (
        root
        / "data/processed/"
        "research_panel"
    )
    sector_path = (
        root
        / "data/processed/"
        "sector_benchmarks/"
        "nse_sector_indices.parquet"
    )
    output_root = (
        root
        / "data/processed/"
        "sector_context_v2"
    )
    reports_root = (
        root
        / "reports/"
        "sector_context_v2"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    largecap_keys = (
        load_largecap_keys(
            metadata_root
        )
    )

    securities = set(
        largecap_keys[
            "canonical_security_id"
        ].astype(
            str
        ).unique()
    )

    first_largecap_date = (
        pd.Timestamp(
            largecap_keys[
                "date"
            ].min()
        ).normalize()
    )

    history_start = (
        first_largecap_date
        - pd.Timedelta(
            days=430
        )
    )

    history = (
        load_research_history(
            panel_root,
            securities=securities,
            start=history_start,
        )
    )

    sector_wide, sector_keys = (
        load_sector_returns(
            sector_path
        )
    )

    broad = load_broad_market_returns(
        root
    )

    history = history.merge(
        sector_wide,
        on="date",
        how="left",
        validate="many_to_one",
    )
    history = history.merge(
        broad,
        on="date",
        how="left",
        validate="many_to_one",
    )

    print(
        f"Computing market-implied sector "
        f"context for {len(securities):,} "
        f"ever-large-cap securities across "
        f"{len(sector_keys)} sector baskets..."
    )

    pieces = []

    groups = list(
        history.groupby(
            "canonical_security_id",
            sort=False,
        )
    )

    for i, (
        _,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):
        pieces.append(
            compute_security_context(
                group,
                sector_keys=(
                    sector_keys
                ),
            )
        )

        if (
            i % 25 == 0
            or i == len(
                groups
            )
        ):
            print(
                f"  computed {i:,}/"
                f"{len(groups):,} securities",
                flush=True,
            )

    enriched = pd.concat(
        pieces,
        ignore_index=True,
    )

    # Keep only dates where the security is actually in the point-in-time
    # large-cap trading universe. Historical pre-membership rows were loaded
    # solely to make trailing affinity estimates possible.
    output = largecap_keys.merge(
        enriched,
        on=[
            "date",
            "canonical_security_id",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_research",
        ),
    )

    output[
        "amfi_largecap_rank_pct"
    ] = (
        (
            pd.to_numeric(
                output[
                    "amfi_market_cap_rank"
                ],
                errors="coerce",
            )
            - 1.0
        )
        / 99.0
    )

    output[
        "log_amfi_average_market_cap_cr"
    ] = np.log1p(
        pd.to_numeric(
            output[
                "amfi_average_market_cap_cr"
            ],
            errors="coerce",
        ).where(
            pd.to_numeric(
                output[
                    "amfi_average_market_cap_cr"
                ],
                errors="coerce",
            )
            >= 0
        )
    )

    daily_median_cap = (
        output.groupby(
            "date",
            sort=False,
        )[
            "amfi_average_market_cap_cr"
        ].transform(
            "median"
        )
    )

    output[
        "amfi_mcap_vs_largecap_median"
    ] = np.log(
        pd.to_numeric(
            output[
                "amfi_average_market_cap_cr"
            ],
            errors="coerce",
        )
        / pd.to_numeric(
            daily_median_cap,
            errors="coerce",
        ).where(
            pd.to_numeric(
                daily_median_cap,
                errors="coerce",
            )
            > 0
        )
    )

    base_keep = [
        "date",
        "market_day_index",
        "canonical_security_id",
        "symbol",
        "amfi_market_cap_rank",
        "amfi_average_market_cap_cr",
        "amfi_largecap_rank_pct",
        "log_amfi_average_market_cap_cr",
        "amfi_mcap_vs_largecap_median",
        "sector_proxy_name",
        "sector_proxy_corr_120d",
        "sector_proxy_corr_252d",
        "sector_proxy_beta_120d",
        "sector_proxy_beta_252d",
        "sector_proxy_corr_margin_252d",
        "sector_proxy_confident",
        "sector_proxy_distinct",
        "sector_response_gap_120d",
        "sector_response_gap_252d",
    ]

    feature_keep = []

    for horizon in (
        RETURN_HORIZONS
    ):
        feature_keep.extend([
            (
                "sector_proxy_return_"
                f"{horizon}d"
            ),
            (
                "stock_minus_sector_"
                f"{horizon}d"
            ),
            (
                "sector_minus_nifty50_"
                f"{horizon}d"
            ),
            (
                "sector_strength_rank_"
                f"{horizon}d"
            ),
        ])

    for lag in LEAD_LAGS:
        feature_keep.append(
            (
                "sector_leads_stock_corr_"
                f"lag{lag}_120d"
            )
        )

    for horizon in (
        RETURN_HORIZONS
    ):
        feature_keep.append(
            (
                "nifty50_minus_nifty500_"
                f"{horizon}d"
            )
        )

    one_hot = [
        (
            "sector_proxy_is_"
            + sector
        )
        for sector in sector_keys
    ]

    final_columns = [
        column
        for column in (
            base_keep
            + feature_keep
            + one_hot
        )
        if column
        in output.columns
    ]

    output = output[
        final_columns
    ].copy()

    numeric_columns = [
        column
        for column in output.columns
        if column not in {
            "date",
            "canonical_security_id",
            "symbol",
            "sector_proxy_name",
            "sector_proxy_confident",
            "sector_proxy_distinct",
        }
    ]

    for column in numeric_columns:
        output[
            column
        ] = pd.to_numeric(
            output[
                column
            ],
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    duplicate = output.duplicated(
        [
            "date",
            "canonical_security_id",
        ],
        keep=False,
    )

    if duplicate.any():
        raise RuntimeError(
            "Sector context contains duplicate "
            "date/security rows."
        )

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for date, group in output.groupby(
        "date",
        sort=True,
    ):
        partition = (
            output_root
            / (
                "date="
                + pd.Timestamp(
                    date
                ).strftime(
                    "%Y-%m-%d"
                )
            )
        )
        partition.mkdir(
            parents=True,
            exist_ok=True,
        )

        group.to_parquet(
            partition
            / "data.parquet",
            index=False,
            compression="zstd",
        )

    coverage = []

    feature_columns = [
        column
        for column in final_columns
        if column not in {
            "date",
            "market_day_index",
            "canonical_security_id",
            "symbol",
            "amfi_market_cap_rank",
            "amfi_average_market_cap_cr",
            "amfi_largecap_rank_pct",
            "log_amfi_average_market_cap_cr",
            "amfi_mcap_vs_largecap_median",
            "sector_proxy_name",
        }
    ]

    for column in feature_columns:
        coverage.append({
            "feature": column,
            "non_null_fraction": float(
                output[
                    column
                ].notna().mean()
            ),
        })

    pd.DataFrame(
        coverage
    ).to_csv(
        reports_root
        / "feature_coverage.csv",
        index=False,
    )

    proxy_counts = (
        output[
            "sector_proxy_name"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "sector_proxy_name"
        )
        .reset_index(
            name="rows"
        )
    )

    proxy_counts.to_csv(
        reports_root
        / "proxy_counts.csv",
        index=False,
    )

    annual = (
        output.assign(
            year=pd.to_datetime(
                output[
                    "date"
                ]
            ).dt.year
        )
        .groupby(
            "year",
            sort=True,
        )
        .agg(
            rows=(
                "canonical_security_id",
                "size",
            ),
            securities=(
                "canonical_security_id",
                "nunique",
            ),
            proxy_known_fraction=(
                "sector_proxy_name",
                lambda values: float(
                    values.notna().mean()
                ),
            ),
            confident_fraction=(
                "sector_proxy_confident",
                lambda values: float(
                    values.fillna(
                        False
                    ).mean()
                ),
            ),
            distinct_fraction=(
                "sector_proxy_distinct",
                lambda values: float(
                    values.fillna(
                        False
                    ).mean()
                ),
            ),
            median_proxy_corr_252d=(
                "sector_proxy_corr_252d",
                "median",
            ),
            median_proxy_margin_252d=(
                "sector_proxy_corr_margin_252d",
                "median",
            ),
        )
        .reset_index()
    )

    annual.to_csv(
        reports_root
        / "annual_coverage.csv",
        index=False,
    )

    summary = {
        "rows": int(
            len(
                output
            )
        ),
        "dates": int(
            output[
                "date"
            ].nunique()
        ),
        "securities": int(
            output[
                "canonical_security_id"
            ].nunique()
        ),
        "sector_baskets": (
            sector_keys
        ),
        "sector_basket_count": int(
            len(
                sector_keys
            )
        ),
        "proxy_known_fraction": float(
            output[
                "sector_proxy_name"
            ].notna().mean()
        ),
        "proxy_confident_fraction": float(
            output[
                "sector_proxy_confident"
            ].fillna(
                False
            ).mean()
        ),
        "proxy_distinct_fraction": float(
            output[
                "sector_proxy_distinct"
            ].fillna(
                False
            ).mean()
        ),
        "median_proxy_corr_252d": float(
            output[
                "sector_proxy_corr_252d"
            ].median()
        ),
        "median_proxy_margin_252d": float(
            output[
                "sector_proxy_corr_margin_252d"
            ].median()
        ),
        "selection_rule": (
            "highest positive trailing "
            "252-session stock/sector-index "
            "return correlation estimated "
            "through t-1; no same-day adaptive "
            "basket selection"
        ),
        "taxonomy_claim": (
            "market-implied sector basket; "
            "not official company industry classification"
        ),
        "output_root": str(
            output_root
        ),
    }

    (
        reports_root
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
    dates = pd.date_range(
        "2025-01-01",
        periods=300,
        freq="D",
    )

    base = pd.Series(
        np.sin(
            np.arange(
                300
            )
            / 10.0
        )
        / 100.0
    )

    group = pd.DataFrame({
        "date": dates,
        "return_1d": (
            base
            + 0.0001
        ),
        "return_5d": base,
        "return_20d": base,
        "return_60d": base,
        "nifty50_return_1d": (
            base * 0.1
        ),
        "nifty50_return_5d": (
            base * 0.1
        ),
        "nifty50_return_20d": (
            base * 0.1
        ),
        "nifty50_return_60d": (
            base * 0.1
        ),
    })

    sectors = [
        "sector_a",
        "sector_b",
    ]

    for horizon in (
        1,
        2,
        3,
        5,
        20,
        60,
    ):
        group[
            (
                "sector__sector_a"
                f"__return_{horizon}d"
            )
        ] = base

        group[
            (
                "sector__sector_b"
                f"__return_{horizon}d"
            )
        ] = (
            -base
        )

    out = compute_security_context(
        group,
        sector_keys=sectors,
    )

    assert (
        out[
            "sector_proxy_name"
        ].dropna()
        .tail(
            20
        )
        .eq(
            "sector_a"
        )
        .all()
    )

    print(
        "Sector-context-v2 self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build point-in-time market-implied "
            "sector context for the official "
            "AMFI large-cap universe."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    summary = build_sector_context(
        Path(
            args.root
        )
    )

    print(
        "\n=== SECTOR CONTEXT V2 COMPLETE ==="
    )
    print(
        f"Rows:              "
        f"{summary['rows']:,}"
    )
    print(
        f"Dates:             "
        f"{summary['dates']:,}"
    )
    print(
        f"Securities:        "
        f"{summary['securities']:,}"
    )
    print(
        f"Sector baskets:    "
        f"{summary['sector_basket_count']:,}"
    )
    print(
        f"Proxy known:       "
        f"{summary['proxy_known_fraction']:.2%}"
    )
    print(
        f"Proxy confident:   "
        f"{summary['proxy_confident_fraction']:.2%}"
    )
    print(
        f"Proxy distinct:    "
        f"{summary['proxy_distinct_fraction']:.2%}"
    )
    print(
        f"Median corr 252d:  "
        f"{summary['median_proxy_corr_252d']:.3f}"
    )
    print(
        f"Median margin:      "
        f"{summary['median_proxy_margin_252d']:.3f}"
    )
    print(
        f"Output:            "
        f"{summary['output_root']}"
    )


if __name__ == "__main__":
    main()
