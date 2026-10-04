from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATE_RE = re.compile(
    r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)"
)

ALIASES = {
    "asof_date": [
        "asof_date",
        "date",
        "effective_date",
    ],
    "canonical_security_id": [
        "canonical_security_id",
    ],
    "isin": [
        "isin",
        "isin_code",
        "isin number",
        "isin number ",
        "isin_code_",
    ],
    "symbol": [
        "symbol",
        "ticker",
    ],
    "sector": [
        "sector",
        "macro economic sector",
    ],
    "industry": [
        "industry",
        "basic industry",
    ],
    "sector_index_name": [
        "sector_index_name",
        "sector index",
    ],
    "market_cap": [
        "market_cap",
        "market capitalization",
        "market_capitalisation",
        "full market capitalisation",
        "free float market capitalization",
    ],
    "largecap_flag": [
        "largecap_flag",
        "large_cap",
        "nifty100_member",
    ],
}


def _normalized_name(
    value: object,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def normalize_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()
    lookup = {
        _normalized_name(
            column
        ): column
        for column in frame.columns
    }

    rename: dict[
        str,
        str,
    ] = {}

    for target, candidates in (
        ALIASES.items()
    ):
        for candidate in candidates:
            source = lookup.get(
                _normalized_name(
                    candidate
                )
            )
            if source is not None:
                rename[
                    source
                ] = target
                break

    return frame.rename(
        columns=rename
    )


def date_from_filename(
    path: Path,
) -> pd.Timestamp | None:
    match = DATE_RE.search(
        path.stem
    )
    if match is None:
        return None

    year, month, day = (
        match.groups()
    )
    parsed = pd.to_datetime(
        f"{year}-{month}-{day}",
        errors="coerce",
    )
    if pd.isna(
        parsed
    ):
        return None
    return pd.Timestamp(
        parsed
    ).normalize()


def read_snapshot(
    path: Path,
) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(
            path,
            low_memory=False,
        )
    elif suffix in {
        ".parquet",
        ".pq",
    }:
        frame = pd.read_parquet(
            path
        )
    else:
        raise ValueError(
            f"Unsupported snapshot: {path}"
        )

    frame = normalize_columns(
        frame
    )

    if "asof_date" in frame.columns:
        frame[
            "asof_date"
        ] = pd.to_datetime(
            frame[
                "asof_date"
            ],
            errors="coerce",
        ).dt.normalize()
    else:
        inferred = date_from_filename(
            path
        )
        if inferred is None:
            raise RuntimeError(
                "Snapshot needs an asof_date "
                "column or YYYY-MM-DD in its "
                f"filename: {path}"
            )
        frame[
            "asof_date"
        ] = inferred

    if (
        "canonical_security_id"
        not in frame.columns
        and "isin"
        not in frame.columns
    ):
        raise RuntimeError(
            "Snapshot must contain either "
            "canonical_security_id or ISIN. "
            "Symbol-only historical mapping is "
            "deliberately rejected."
        )

    return frame


def build_isin_identity_map(
    panel_root: Path,
) -> dict[
    str,
    str,
]:
    files = sorted(
        Path(
            panel_root
        ).glob(
            "date=*/data.parquet"
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No research panel under {panel_root}"
        )

    schema = set(
        pq.read_schema(
            files[0]
        ).names
    )
    isin_column = (
        "isin_resolved"
        if "isin_resolved"
        in schema
        else (
            "isin"
            if "isin" in schema
            else None
        )
    )

    if isin_column is None:
        raise RuntimeError(
            "Research panel has no ISIN column "
            "for context identity resolution."
        )

    pairs: list[
        pd.DataFrame
    ] = []

    print(
        "Building ISIN -> canonical identity map..."
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        part = pd.read_parquet(
            path,
            columns=[
                "canonical_security_id",
                isin_column,
            ],
        )
        part = part.dropna(
            subset=[
                "canonical_security_id",
                isin_column,
            ]
        )
        if not part.empty:
            pairs.append(
                part.drop_duplicates()
            )

        if (
            number % 1000 == 0
            or number == len(files)
        ):
            print(
                f"  scanned {number:,}/"
                f"{len(files):,}",
                flush=True,
            )

    unique = (
        pd.concat(
            pairs,
            ignore_index=True,
        )
        .assign(
            **{
                isin_column: (
                    lambda x: (
                        x[
                            isin_column
                        ]
                        .astype(str)
                        .str.strip()
                    )
                ),
                "canonical_security_id": (
                    lambda x: (
                        x[
                            "canonical_security_id"
                        ]
                        .astype(str)
                        .str.strip()
                    )
                ),
            }
        )
        .drop_duplicates()
    )

    ambiguous = (
        unique.groupby(
            isin_column
        )[
            "canonical_security_id"
        ]
        .nunique()
    )
    bad = set(
        ambiguous[
            ambiguous > 1
        ].index
    )
    if bad:
        unique = unique.loc[
            ~unique[
                isin_column
            ].isin(
                bad
            )
        ]

    return dict(
        zip(
            unique[
                isin_column
            ],
            unique[
                "canonical_security_id"
            ],
            strict=False,
        )
    )


def resolve_snapshot_ids(
    frame: pd.DataFrame,
    *,
    isin_map: dict[
        str,
        str,
    ] | None,
) -> pd.DataFrame:
    frame = frame.copy()

    if (
        "canonical_security_id"
        in frame.columns
    ):
        cid = (
            frame[
                "canonical_security_id"
            ]
            .astype("string")
            .str.strip()
        )
    else:
        cid = pd.Series(
            pd.NA,
            index=frame.index,
            dtype="string",
        )

    if (
        cid.isna().any()
        and "isin"
        in frame.columns
    ):
        if isin_map is None:
            raise RuntimeError(
                "ISIN resolution was not built."
            )
        isin = (
            frame[
                "isin"
            ]
            .astype("string")
            .str.strip()
        )
        mapped = isin.map(
            isin_map
        )
        cid = cid.fillna(
            mapped
        )

    frame[
        "canonical_security_id"
    ] = cid

    unresolved = frame[
        "canonical_security_id"
    ].isna()

    if unresolved.any():
        sample_columns = [
            column
            for column in (
                "symbol",
                "isin",
                "asof_date",
            )
            if column in frame.columns
        ]
        sample = frame.loc[
            unresolved,
            sample_columns,
        ].head(
            20
        )
        raise RuntimeError(
            "Unresolved historical context "
            f"identities: {int(unresolved.sum()):,}\n"
            + sample.to_string(
                index=False
            )
        )

    return frame


def build_membership_context(
    snapshots: list[
        pd.DataFrame
    ],
) -> pd.DataFrame:
    if not snapshots:
        raise RuntimeError(
            "No snapshots supplied."
        )

    all_members = set(
        pd.concat(
            [
                frame[
                    "canonical_security_id"
                ]
                for frame
                in snapshots
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
    )

    rows: list[
        pd.DataFrame
    ] = []

    for frame in snapshots:
        dates = (
            frame[
                "asof_date"
            ]
            .dropna()
            .unique()
        )
        if len(dates) != 1:
            raise RuntimeError(
                "Each membership snapshot file "
                "must represent exactly one "
                "as-of date."
            )

        asof = pd.Timestamp(
            dates[0]
        ).normalize()
        current = set(
            frame[
                "canonical_security_id"
            ]
            .dropna()
            .astype(str)
        )

        base = pd.DataFrame({
            "asof_date": asof,
            "canonical_security_id": (
                sorted(
                    all_members
                )
            ),
        })
        base[
            "largecap_flag"
        ] = base[
            "canonical_security_id"
        ].isin(
            current
        )

        metadata_columns = [
            column
            for column in (
                "sector",
                "industry",
                "sector_index_name",
                "market_cap",
                "symbol",
                "isin",
            )
            if column in frame.columns
        ]

        if metadata_columns:
            current_meta = (
                frame[
                    [
                        "canonical_security_id",
                        *metadata_columns,
                    ]
                ]
                .drop_duplicates(
                    "canonical_security_id",
                    keep="last",
                )
            )
            base = base.merge(
                current_meta,
                on="canonical_security_id",
                how="left",
                validate="one_to_one",
            )

        rows.append(
            base
        )

    context = pd.concat(
        rows,
        ignore_index=True,
    )

    return (
        context.sort_values(
            [
                "asof_date",
                "canonical_security_id",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize dated point-in-time "
            "large-cap/security-context snapshots."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--input-dir",
        default=(
            "data/reference/"
            "security_context_snapshots"
        ),
    )
    ap.add_argument(
        "--panel",
        default=(
            "data/processed/"
            "research_panel"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "data/processed/context/"
            "security_context.parquet"
        ),
    )
    args = ap.parse_args()

    root = Path(
        args.root
    ).resolve()
    input_dir = (
        root
        / args.input_dir
    )

    files = sorted(
        [
            *input_dir.glob(
                "*.csv"
            ),
            *input_dir.glob(
                "*.parquet"
            ),
            *input_dir.glob(
                "*.pq"
            ),
        ]
    )
    if not files:
        raise FileNotFoundError(
            "No dated context snapshots under "
            f"{input_dir}"
        )

    raw = [
        read_snapshot(
            path
        )
        for path in files
    ]

    needs_isin = any(
        "canonical_security_id"
        not in frame.columns
        or frame[
            "canonical_security_id"
        ].isna().any()
        for frame in raw
    )
    isin_map = (
        build_isin_identity_map(
            root
            / args.panel
        )
        if needs_isin
        else None
    )

    resolved = [
        resolve_snapshot_ids(
            frame,
            isin_map=isin_map,
        )
        for frame in raw
    ]

    context = (
        build_membership_context(
            resolved
        )
    )

    duplicate = context.duplicated(
        [
            "asof_date",
            "canonical_security_id",
        ],
        keep=False,
    )
    if duplicate.any():
        raise RuntimeError(
            "Context build produced "
            "duplicate as-of/security rows."
        )

    output = (
        root
        / args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    context.to_parquet(
        output,
        index=False,
        compression="zstd",
    )

    summary = (
        context.groupby(
            "asof_date"
        )
        .agg(
            rows=(
                "canonical_security_id",
                "size",
            ),
            largecaps=(
                "largecap_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )
    print(
        f"\nOutput: {output}"
    )
    print(
        "No context is backfilled before "
        "its first dated snapshot."
    )


if __name__ == "__main__":
    main()
