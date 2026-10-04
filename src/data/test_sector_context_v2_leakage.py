from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sector_context_v2 import (
    compute_security_context,
    load_broad_market_returns,
    load_research_history,
    load_sector_returns,
)


def load_candidate_securities(
    root: Path,
    *,
    cutoff: pd.Timestamp,
    limit: int,
) -> list[str]:
    files = sorted(
        (
            root
            / "data/processed/"
            "pit_metadata_v2"
        ).glob(
            "date=*/data.parquet"
        )
    )

    counts: dict[
        str,
        int
    ] = {}

    for path in files:
        date = pd.Timestamp(
            path.parent.name.split(
                "=",
                1,
            )[
                1
            ]
        ).normalize()

        if date > cutoff:
            break

        frame = pd.read_parquet(
            path,
            columns=[
                "canonical_security_id",
                "pit_largecap_universe",
            ],
        )

        frame = frame.loc[
            frame[
                "pit_largecap_universe"
            ].fillna(
                False
            )
        ]

        for security in frame[
            "canonical_security_id"
        ].astype(
            str
        ):
            counts[
                security
            ] = counts.get(
                security,
                0,
            ) + 1

    ranked = sorted(
        counts,
        key=lambda security: (
            -counts[
                security
            ],
            security,
        ),
    )

    return ranked[
        :limit
    ]


def prepare_source(
    root: Path,
    *,
    cutoff: pd.Timestamp,
    history_days: int,
    future_days: int,
    securities: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    start = (
        cutoff
        - pd.Timedelta(
            days=history_days
        )
    )
    end = (
        cutoff
        + pd.Timedelta(
            days=future_days
        )
    )

    history = load_research_history(
        root
        / "data/processed/"
        "research_panel",
        securities=set(
            securities
        ),
        start=start,
    )

    history = history.loc[
        history[
            "date"
        ].le(
            end
        )
    ].copy()

    sector_wide, sector_keys = (
        load_sector_returns(
            root
            / "data/processed/"
            "sector_benchmarks/"
            "nse_sector_indices.parquet"
        )
    )

    broad = (
        load_broad_market_returns(
            root
        )
    )

    source = history.merge(
        sector_wide,
        on="date",
        how="left",
        validate="many_to_one",
    ).merge(
        broad,
        on="date",
        how="left",
        validate="many_to_one",
    )

    return (
        source,
        sector_keys,
    )


def compute_all(
    source: pd.DataFrame,
    *,
    sector_keys: list[str],
) -> pd.DataFrame:
    pieces = []

    for _, group in source.groupby(
        "canonical_security_id",
        sort=False,
    ):
        pieces.append(
            compute_security_context(
                group,
                sector_keys=(
                    sector_keys
                ),
            )
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    ).sort_values(
        [
            "canonical_security_id",
            "date",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )


def corrupt_future(
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

    numeric = [
        column
        for column in out.columns
        if (
            column.startswith(
                "return_"
            )
            or column.startswith(
                "sector__"
            )
            or column.startswith(
                "nifty50_"
            )
            or column.startswith(
                "nifty500_"
            )
        )
    ]

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
            "No post-cutoff rows available "
            "for future corruption."
        )

    perturbation = (
        (
            positions
            % 17
        )
        / 10.0
        + 2.5
    )

    for column in numeric:
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
            -11.0
            * values
            + perturbation
        )

    return out


def feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    prefixes = (
        "sector_proxy_",
        "stock_minus_sector_",
        "sector_minus_nifty50_",
        "sector_strength_rank_",
        "sector_leads_stock_corr_",
        "nifty50_minus_nifty500_",
    )

    return [
        column
        for column in frame.columns
        if column.startswith(
            prefixes
        )
    ]


def compare(
    baseline: pd.DataFrame,
    mutated: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    features: list[str],
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    keys = [
        "date",
        "canonical_security_id",
    ]

    left = baseline.loc[
        baseline[
            "date"
        ].le(
            cutoff
        ),
        [
            *keys,
            *features,
        ],
    ].sort_values(
        keys
    ).reset_index(
        drop=True
    )

    right = mutated.loc[
        mutated[
            "date"
        ].le(
            cutoff
        ),
        [
            *keys,
            *features,
        ],
    ].sort_values(
        keys
    ).reset_index(
        drop=True
    )

    if not left[
        keys
    ].equals(
        right[
            keys
        ]
    ):
        raise RuntimeError(
            "Pre-cutoff keys differ."
        )

    rows = []

    for feature in features:
        if (
            left[
                feature
            ].dtype
            == object
            or str(
                left[
                    feature
                ].dtype
            )
            in {
                "string",
                "boolean",
            }
        ):
            a = left[
                feature
            ].astype(
                "string"
            )
            b = right[
                feature
            ].astype(
                "string"
            )

            equal = (
                a.fillna(
                    "__NA__"
                )
                == b.fillna(
                    "__NA__"
                )
            ).to_numpy()
        else:
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
            :50
        ]:
            rows.append({
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
                "baseline": str(
                    left.loc[
                        position,
                        feature,
                    ]
                ),
                "future_mutated": str(
                    right.loc[
                        position,
                        feature,
                    ]
                ),
            })

        if len(
            rows
        ) >= 500:
            break

    return pd.DataFrame(
        rows
    )


def run_test(
    root: Path,
    *,
    cutoff: pd.Timestamp,
    securities: int,
    history_days: int,
    future_days: int,
    atol: float,
    rtol: float,
) -> dict:
    root = Path(
        root
    ).resolve()

    reports = (
        root
        / "reports/"
        "sector_context_v2/"
        "leakage_regression"
    )
    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = (
        load_candidate_securities(
            root,
            cutoff=cutoff,
            limit=securities,
        )
    )

    print(
        f"Selected {len(selected)} "
        "large-cap securities."
    )

    source, sector_keys = (
        prepare_source(
            root,
            cutoff=cutoff,
            history_days=(
                history_days
            ),
            future_days=(
                future_days
            ),
            securities=selected,
        )
    )

    print(
        "Computing baseline F5 context..."
    )
    baseline = compute_all(
        source,
        sector_keys=(
            sector_keys
        ),
    )

    print(
        "Corrupting all post-cutoff stock, "
        "sector and broad-market returns..."
    )
    corrupted = corrupt_future(
        source,
        cutoff=cutoff,
    )

    print(
        "Recomputing F5 context..."
    )
    mutated = compute_all(
        corrupted,
        sector_keys=(
            sector_keys
        ),
    )

    features = feature_columns(
        baseline
    )

    print(
        f"Comparing {len(features)} F5 "
        "features at/before cutoff..."
    )

    mismatches = compare(
        baseline,
        mutated,
        cutoff=cutoff,
        features=features,
        atol=atol,
        rtol=rtol,
    )

    mismatches.to_csv(
        reports
        / "mismatches.csv",
        index=False,
    )

    summary = {
        "cutoff": str(
            cutoff.date()
        ),
        "securities": int(
            len(
                selected
            )
        ),
        "source_rows": int(
            len(
                source
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
        "features_checked": int(
            len(
                features
            )
        ),
        "mismatch_rows_reported": int(
            len(
                mismatches
            )
        ),
        "passed": bool(
            mismatches.empty
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

    if not mismatches.empty:
        raise RuntimeError(
            "F5 future-mutation leakage test "
            "FAILED. See "
            + str(
                reports
                / "mismatches.csv"
            )
        )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Future-mutation leakage regression "
            "for sector-context-v2."
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
        "--securities",
        type=int,
        default=48,
    )
    ap.add_argument(
        "--history-days",
        type=int,
        default=700,
    )
    ap.add_argument(
        "--future-days",
        type=int,
        default=90,
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
        cutoff=pd.Timestamp(
            args.cutoff
        ).normalize(),
        securities=args.securities,
        history_days=(
            args.history_days
        ),
        future_days=(
            args.future_days
        ),
        atol=args.atol,
        rtol=args.rtol,
    )

    print(
        "\n=== F5 FUTURE-MUTATION TEST ==="
    )
    print(
        f"Features checked:      "
        f"{summary['features_checked']}"
    )
    print(
        f"Pre-cutoff rows:       "
        f"{summary['pre_cutoff_rows']:,}"
    )
    print(
        f"Future rows corrupted: "
        f"{summary['future_rows_corrupted']:,}"
    )
    print(
        "Result:                PASS"
    )


if __name__ == "__main__":
    main()
