from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering_v2 import (
    SOURCE_COLUMNS,
    add_calendar_features,
    add_cross_section_features,
    add_market_features,
    add_ohlc_features,
    add_stock_tendencies,
    add_streak_features,
    load_market_context,
)


def partition_date(
    path: Path,
) -> pd.Timestamp:
    name = path.parent.name

    if not name.startswith(
        "date="
    ):
        raise ValueError(
            f"Unexpected partition path: {path}"
        )

    return pd.Timestamp(
        name.split(
            "=",
            1,
        )[
            1
        ]
    ).normalize()


def select_window(
    panel_root: Path,
    *,
    requested_cutoff: pd.Timestamp,
    history_sessions: int,
    future_sessions: int,
) -> tuple[
    list[Path],
    pd.Timestamp,
]:
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

    dates = pd.Index([
        partition_date(
            path
        )
        for path in files
    ])

    eligible = np.flatnonzero(
        dates
        <= requested_cutoff
    )

    if len(
        eligible
    ) == 0:
        raise RuntimeError(
            "Requested cutoff precedes "
            "the research panel."
        )

    cutoff_pos = int(
        eligible[
            -1
        ]
    )
    cutoff = pd.Timestamp(
        dates[
            cutoff_pos
        ]
    ).normalize()

    start = max(
        0,
        cutoff_pos
        - history_sessions
        + 1,
    )
    stop = min(
        len(
            files
        ),
        cutoff_pos
        + future_sessions
        + 1,
    )

    return (
        files[
            start:stop
        ],
        cutoff,
    )


def load_real_window(
    files: list[Path],
    *,
    max_securities: int,
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            path,
            columns=SOURCE_COLUMNS,
        )
        for path in files
    ]

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

    if max_securities > 0:
        counts = (
            df.groupby(
                "canonical_security_id",
                sort=False,
            )[
                "date"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        selected = set(
            counts.head(
                max_securities
            ).index
        )

        df = df.loc[
            df[
                "canonical_security_id"
            ].isin(
                selected
            )
        ].copy()

    return (
        df.sort_values(
            [
                "canonical_security_id",
                "date",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def compute_features(
    source: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    df = source.copy()
    registry: list[
        dict
    ] = []

    add_ohlc_features(
        df,
        registry,
    )
    add_streak_features(
        df,
        registry,
    )
    add_stock_tendencies(
        df,
        registry,
    )
    add_market_features(
        df,
        market,
        registry,
    )
    add_cross_section_features(
        df,
        registry,
    )
    add_calendar_features(
        df,
        registry,
    )

    features = list(
        dict.fromkeys(
            row[
                "feature"
            ]
            for row in registry
            if row.get(
                "status",
                "implemented",
            )
            == "implemented"
        )
    )

    return (
        df,
        features,
    )


def corrupt_future_sources(
    source: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    out = source.copy()

    future = out[
        "date"
    ].gt(
        cutoff
    )

    positions = np.arange(
        int(
            future.sum()
        ),
        dtype=float,
    )

    if not len(
        positions
    ):
        raise RuntimeError(
            "Leakage test window contains "
            "no post-cutoff rows."
        )

    price_factor = (
        3.0
        + (
            positions
            % 17
        )
        / 10.0
    )

    for column in (
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
    ):
        values = pd.to_numeric(
            out.loc[
                future,
                column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        out.loc[
            future,
            column,
        ] = (
            values
            * price_factor
        )

    for column in (
        "adj_volume",
        "turnover",
    ):
        values = pd.to_numeric(
            out.loc[
                future,
                column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        out.loc[
            future,
            column,
        ] = (
            values
            * (
                11.0
                + (
                    positions
                    % 7
                )
            )
        )

    for column in (
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
    ):
        values = pd.to_numeric(
            out.loc[
                future,
                column,
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        out.loc[
            future,
            column,
        ] = (
            -7.0
            * values
            + (
                positions
                % 13
            )
            / 100.0
        )

    return out


def corrupt_future_market(
    market: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    out = market.copy()

    future = out[
        "date"
    ].gt(
        cutoff
    )

    for column in out.columns:
        if column == "date":
            continue

        values = pd.to_numeric(
            out.loc[
                future,
                column,
            ],
            errors="coerce",
        )

        out.loc[
            future,
            column,
        ] = (
            -9.0
            * values
            + 0.777
        )

    return out


def compare_pre_cutoff(
    baseline: pd.DataFrame,
    mutated: pd.DataFrame,
    *,
    features: list[str],
    cutoff: pd.Timestamp,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    key_columns = [
        "date",
        "canonical_security_id",
    ]

    left = (
        baseline.loc[
            baseline[
                "date"
            ].le(
                cutoff
            ),
            [
                *key_columns,
                *features,
            ],
        ]
        .sort_values(
            key_columns,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    right = (
        mutated.loc[
            mutated[
                "date"
            ].le(
                cutoff
            ),
            [
                *key_columns,
                *features,
            ],
        ]
        .sort_values(
            key_columns,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if not left[
        key_columns
    ].equals(
        right[
            key_columns
        ]
    ):
        raise RuntimeError(
            "Baseline/mutated pre-cutoff "
            "keys do not match."
        )

    mismatches: list[
        dict
    ] = []

    for feature in features:
        a = pd.to_numeric(
            left[
                feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
        b = pd.to_numeric(
            right[
                feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        equal = np.isclose(
            a,
            b,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )

        bad = np.flatnonzero(
            ~equal
        )

        for position in bad[
            :100
        ]:
            mismatches.append({
                "date": str(
                    pd.Timestamp(
                        left.loc[
                            position,
                            "date",
                        ]
                    ).date()
                ),
                "canonical_security_id": (
                    left.loc[
                        position,
                        "canonical_security_id",
                    ]
                ),
                "feature": feature,
                "baseline": (
                    None
                    if np.isnan(
                        a[
                            position
                        ]
                    )
                    else float(
                        a[
                            position
                        ]
                    )
                ),
                "future_mutated": (
                    None
                    if np.isnan(
                        b[
                            position
                        ]
                    )
                    else float(
                        b[
                            position
                        ]
                    )
                ),
            })

        if len(
            mismatches
        ) >= 1000:
            break

    return pd.DataFrame(
        mismatches
    )


def validate_registry(
    root: Path,
    generated_features: list[str],
) -> None:
    path = (
        root
        / "reports/features_v2/"
        "feature_registry.csv"
    )

    if not path.is_file():
        return

    registry = pd.read_csv(
        path
    )
    expected = set(
        registry.loc[
            registry[
                "status"
            ].eq(
                "implemented"
            ),
            "feature",
        ].astype(
            str
        )
    )

    actual = set(
        generated_features
    )

    if expected != actual:
        raise RuntimeError(
            "Leakage test feature set does "
            "not match built v2 registry. "
            f"Missing={sorted(expected - actual)} "
            f"Extra={sorted(actual - expected)}"
        )


def run_test(
    root: Path,
    *,
    requested_cutoff: pd.Timestamp,
    history_sessions: int,
    future_sessions: int,
    max_securities: int,
    atol: float,
    rtol: float,
) -> dict:
    root = Path(
        root
    ).resolve()

    panel_root = (
        root
        / "data/processed/"
        "research_panel"
    )
    reports_root = (
        root
        / "reports/features_v2/"
        "leakage_regression"
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files, cutoff = (
        select_window(
            panel_root,
            requested_cutoff=(
                requested_cutoff
            ),
            history_sessions=(
                history_sessions
            ),
            future_sessions=(
                future_sessions
            ),
        )
    )

    source = load_real_window(
        files,
        max_securities=(
            max_securities
        ),
    )

    market = load_market_context(
        root
    )

    min_date = source[
        "date"
    ].min()
    max_date = source[
        "date"
    ].max()

    market = market.loc[
        market[
            "date"
        ].between(
            min_date,
            max_date,
            inclusive="both",
        )
    ].copy()

    print(
        "Building baseline features..."
    )
    baseline, features = (
        compute_features(
            source,
            market,
        )
    )

    validate_registry(
        root,
        features,
    )

    print(
        "Corrupting every source after "
        f"{cutoff.date()}..."
    )
    mutated_source = (
        corrupt_future_sources(
            source,
            cutoff=cutoff,
        )
    )
    mutated_market = (
        corrupt_future_market(
            market,
            cutoff=cutoff,
        )
    )

    print(
        "Rebuilding from corrupted future..."
    )
    mutated, mutated_features = (
        compute_features(
            mutated_source,
            mutated_market,
        )
    )

    if features != mutated_features:
        raise RuntimeError(
            "Feature ordering changed after "
            "future mutation."
        )

    print(
        "Comparing all pre-cutoff "
        f"values across {len(features)} "
        "features..."
    )
    mismatches = compare_pre_cutoff(
        baseline,
        mutated,
        features=features,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
    )

    mismatch_path = (
        reports_root
        / "mismatches.csv"
    )
    mismatches.to_csv(
        mismatch_path,
        index=False,
    )

    summary = {
        "requested_cutoff": str(
            requested_cutoff.date()
        ),
        "actual_cutoff": str(
            cutoff.date()
        ),
        "window_start": str(
            pd.Timestamp(
                min_date
            ).date()
        ),
        "window_end": str(
            pd.Timestamp(
                max_date
            ).date()
        ),
        "source_rows": int(
            len(
                source
            )
        ),
        "securities": int(
            source[
                "canonical_security_id"
            ].nunique()
        ),
        "features_checked": int(
            len(
                features
            )
        ),
        "pre_cutoff_rows": int(
            baseline[
                "date"
            ].le(
                cutoff
            ).sum()
        ),
        "future_rows_corrupted": int(
            source[
                "date"
            ].gt(
                cutoff
            ).sum()
        ),
        "mismatch_rows_reported": int(
            len(
                mismatches
            )
        ),
        "passed": bool(
            mismatches.empty
        ),
        "atol": float(
            atol
        ),
        "rtol": float(
            rtol
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

    if not mismatches.empty:
        sample = mismatches.head(
            20
        )

        raise RuntimeError(
            "FUTURE-MUTATION LEAKAGE TEST "
            "FAILED. Sample:\n"
            + sample.to_string(
                index=False
            )
            + "\nFull mismatch report: "
            + str(
                mismatch_path
            )
        )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Real-data future-mutation "
            "regression test for every "
            "implemented feature-v2 column."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--cutoff",
        default="2023-12-29",
    )
    ap.add_argument(
        "--history-sessions",
        type=int,
        default=400,
    )
    ap.add_argument(
        "--future-sessions",
        type=int,
        default=40,
    )
    ap.add_argument(
        "--securities",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=1e-12,
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
    )
    args = ap.parse_args()

    summary = run_test(
        Path(
            args.root
        ),
        requested_cutoff=pd.Timestamp(
            args.cutoff
        ).normalize(),
        history_sessions=(
            args.history_sessions
        ),
        future_sessions=(
            args.future_sessions
        ),
        max_securities=(
            args.securities
        ),
        atol=args.atol,
        rtol=args.rtol,
    )

    print(
        "\n=== FEATURE V2 FUTURE-MUTATION TEST ==="
    )
    print(
        f"Cutoff:                "
        f"{summary['actual_cutoff']}"
    )
    print(
        f"Rows tested:           "
        f"{summary['pre_cutoff_rows']:,}"
    )
    print(
        f"Future rows corrupted: "
        f"{summary['future_rows_corrupted']:,}"
    )
    print(
        f"Features checked:      "
        f"{summary['features_checked']:,}"
    )
    print(
        "Result:                PASS"
    )


if __name__ == "__main__":
    main()
