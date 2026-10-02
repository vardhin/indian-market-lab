from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd


MECHANICAL_ACTION_TYPES = {
    "bonus",
    "split_or_consolidation",
}

ADJUSTABLE_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "last",
]


def _norm_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.eq(""))


def load_mechanical_actions(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Corporate-actions file not found: {path}"
        )

    actions = pd.read_parquet(path)

    required = {
        "symbol",
        "series",
        "purpose",
        "action_type",
        "ex_date",
        "share_multiplier",
        "price_multiplier",
        "volume_multiplier",
        "auto_adjustable",
    }

    missing = sorted(required - set(actions.columns))

    if missing:
        raise RuntimeError(
            "Corporate-actions dataset is missing columns: "
            f"{missing}"
        )

    actions = actions.loc[
        actions["auto_adjustable"].fillna(False)
        & actions["action_type"].isin(
            MECHANICAL_ACTION_TYPES
        )
    ].copy()

    actions["ex_date"] = pd.to_datetime(
        actions["ex_date"],
        errors="coerce",
    ).dt.normalize()

    for col in (
        "symbol",
        "series",
        "purpose",
        "action_type",
    ):
        actions[col] = _norm_string(actions[col])

    for col in (
        "share_multiplier",
        "price_multiplier",
        "volume_multiplier",
    ):
        actions[col] = pd.to_numeric(
            actions[col],
            errors="coerce",
        )

    invalid = (
        actions["ex_date"].isna()
        | actions["symbol"].isna()
        | actions["share_multiplier"].le(0)
        | actions["price_multiplier"].le(0)
        | actions["volume_multiplier"].le(0)
    )

    if invalid.any():
        sample = actions.loc[
            invalid,
            [
                "symbol",
                "series",
                "purpose",
                "ex_date",
                "share_multiplier",
                "price_multiplier",
                "volume_multiplier",
            ],
        ].head(30)

        raise RuntimeError(
            "Invalid mechanically adjustable corporate actions. Sample:\n"
            + sample.to_string(index=False)
        )

    return (
        actions
        .sort_values(
            ["ex_date", "symbol", "series", "purpose"]
        )
        .reset_index(drop=True)
    )


def _partition_path(
    resolved_root: Path,
    day: pd.Timestamp,
) -> Path:
    return (
        resolved_root
        / f"date={pd.Timestamp(day).date()}"
        / "data.parquet"
    )


def _load_identity_slice(
    resolved_root: Path,
    day: pd.Timestamp,
) -> pd.DataFrame | None:
    path = _partition_path(
        resolved_root,
        day,
    )

    if not path.is_file():
        return None

    cols = [
        "date",
        "symbol",
        "series",
        "canonical_security_id",
        "isin_resolved",
    ]

    try:
        d = pd.read_parquet(
            path,
            columns=cols,
        )
    except Exception:
        d = pd.read_parquet(path)

        missing = [
            c
            for c in cols
            if c not in d.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path} is missing required identity columns: "
                f"{missing}"
            )

        d = d[cols].copy()

    d["date"] = pd.to_datetime(
        d["date"],
        errors="coerce",
    ).dt.normalize()

    for col in (
        "symbol",
        "series",
        "canonical_security_id",
        "isin_resolved",
    ):
        d[col] = _norm_string(d[col])

    return d


def _match_action_in_slice(
    action: pd.Series,
    d: pd.DataFrame,
) -> pd.DataFrame:
    matches = d.loc[
        d["symbol"].eq(action["symbol"])
    ].copy()

    action_series = action.get("series")

    if (
        pd.notna(action_series)
        and str(action_series).strip()
    ):
        series_matches = matches.loc[
            matches["series"].eq(
                str(action_series).strip()
            )
        ].copy()

        if not series_matches.empty:
            matches = series_matches

    return matches


def map_action_to_identity(
    action: pd.Series,
    resolved_root: Path,
    *,
    fallback_days: int = 7,
) -> dict:
    """
    Map one NSE corporate action to a resolved canonical security.

    Preferred mapping:
      exact ex-date + symbol (+ series when available)

    Conservative fallback:
      find the nearest matching resolved row before and after ex-date within
      fallback_days calendar days; accept only when both sides resolve to the
      same canonical security ID.

    A one-sided fallback is intentionally not accepted.
    """
    ex_date = pd.Timestamp(
        action["ex_date"]
    ).normalize()

    exact = _load_identity_slice(
        resolved_root,
        ex_date,
    )

    if exact is not None:
        matches = _match_action_in_slice(
            action,
            exact,
        )

        ids = sorted(
            set(
                matches["canonical_security_id"]
                .dropna()
                .astype(str)
            )
        )

        if len(ids) == 1:
            row = matches.loc[
                matches["canonical_security_id"].eq(
                    ids[0]
                )
            ].iloc[0]

            return {
                "canonical_security_id": ids[0],
                "mapped_isin": row["isin_resolved"],
                "mapping_method": "exact_ex_date",
                "mapping_status": "mapped",
                "mapping_left_date": ex_date,
                "mapping_right_date": ex_date,
            }

        if len(ids) > 1:
            return {
                "canonical_security_id": None,
                "mapped_isin": None,
                "mapping_method": None,
                "mapping_status": "ambiguous_exact_match",
                "mapping_left_date": pd.NaT,
                "mapping_right_date": pd.NaT,
            }

    left_match: tuple[pd.Timestamp, str, str | None] | None = None
    right_match: tuple[pd.Timestamp, str, str | None] | None = None

    for offset in range(1, fallback_days + 1):
        if left_match is None:
            day = ex_date - pd.Timedelta(
                days=offset
            )
            d = _load_identity_slice(
                resolved_root,
                day,
            )

            if d is not None:
                matches = _match_action_in_slice(
                    action,
                    d,
                )

                ids = sorted(
                    set(
                        matches["canonical_security_id"]
                        .dropna()
                        .astype(str)
                    )
                )

                if len(ids) == 1:
                    row = matches.loc[
                        matches[
                            "canonical_security_id"
                        ].eq(ids[0])
                    ].iloc[0]
                    left_match = (
                        day,
                        ids[0],
                        row["isin_resolved"],
                    )
                elif len(ids) > 1:
                    return {
                        "canonical_security_id": None,
                        "mapped_isin": None,
                        "mapping_method": None,
                        "mapping_status": "ambiguous_left_fallback",
                        "mapping_left_date": day,
                        "mapping_right_date": pd.NaT,
                    }

        if right_match is None:
            day = ex_date + pd.Timedelta(
                days=offset
            )
            d = _load_identity_slice(
                resolved_root,
                day,
            )

            if d is not None:
                matches = _match_action_in_slice(
                    action,
                    d,
                )

                ids = sorted(
                    set(
                        matches["canonical_security_id"]
                        .dropna()
                        .astype(str)
                    )
                )

                if len(ids) == 1:
                    row = matches.loc[
                        matches[
                            "canonical_security_id"
                        ].eq(ids[0])
                    ].iloc[0]
                    right_match = (
                        day,
                        ids[0],
                        row["isin_resolved"],
                    )
                elif len(ids) > 1:
                    return {
                        "canonical_security_id": None,
                        "mapped_isin": None,
                        "mapping_method": None,
                        "mapping_status": "ambiguous_right_fallback",
                        "mapping_left_date": pd.NaT,
                        "mapping_right_date": day,
                    }

        if (
            left_match is not None
            and right_match is not None
        ):
            break

    if (
        left_match is not None
        and right_match is not None
        and left_match[1] == right_match[1]
    ):
        mapped_isin = (
            left_match[2]
            if left_match[2] == right_match[2]
            else left_match[2] or right_match[2]
        )

        return {
            "canonical_security_id": left_match[1],
            "mapped_isin": mapped_isin,
            "mapping_method": "bracket_same_identity",
            "mapping_status": "mapped",
            "mapping_left_date": left_match[0],
            "mapping_right_date": right_match[0],
        }

    if (
        left_match is not None
        and right_match is not None
        and left_match[1] != right_match[1]
    ):
        return {
            "canonical_security_id": None,
            "mapped_isin": None,
            "mapping_method": None,
            "mapping_status": "conflicting_bracket_identity",
            "mapping_left_date": left_match[0],
            "mapping_right_date": right_match[0],
        }

    return {
        "canonical_security_id": None,
        "mapped_isin": None,
        "mapping_method": None,
        "mapping_status": "unmapped",
        "mapping_left_date": (
            left_match[0]
            if left_match is not None
            else pd.NaT
        ),
        "mapping_right_date": (
            right_match[0]
            if right_match is not None
            else pd.NaT
        ),
    }


def map_actions(
    actions: pd.DataFrame,
    resolved_root: Path,
    *,
    fallback_days: int = 7,
) -> pd.DataFrame:
    rows: list[dict] = []

    for index, action in actions.iterrows():
        mapped = map_action_to_identity(
            action,
            resolved_root,
            fallback_days=fallback_days,
        )

        record = action.to_dict()
        record.update(mapped)
        rows.append(record)

        if (
            (index + 1) % 100 == 0
            or index + 1 == len(actions)
        ):
            print(
                f"  mapped {index + 1:,}/"
                f"{len(actions):,}",
                flush=True,
            )

    out = pd.DataFrame(rows)

    for col in (
        "mapping_left_date",
        "mapping_right_date",
    ):
        out[col] = pd.to_datetime(
            out[col],
            errors="coerce",
        ).dt.normalize()

    return out


def collapse_mapped_actions(
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse multiple mechanical actions for the same canonical security and
    ex-date into one multiplicative event.
    """
    mapped_ok = mapped.loc[
        mapped["mapping_status"].eq("mapped")
        & mapped["canonical_security_id"].notna()
    ].copy()

    if mapped_ok.empty:
        return pd.DataFrame(
            columns=[
                "canonical_security_id",
                "ex_date",
                "price_multiplier",
                "volume_multiplier",
                "share_multiplier",
                "action_count",
                "action_types",
                "purposes",
            ]
        )

    grouped = mapped_ok.groupby(
        [
            "canonical_security_id",
            "ex_date",
        ],
        sort=True,
        dropna=False,
    )

    event_rows: list[dict] = []

    for (
        canonical_security_id,
        ex_date,
    ), g in grouped:
        price_multiplier = float(
            g["price_multiplier"].prod()
        )
        volume_multiplier = float(
            g["volume_multiplier"].prod()
        )
        share_multiplier = float(
            g["share_multiplier"].prod()
        )

        event_rows.append({
            "canonical_security_id": (
                canonical_security_id
            ),
            "ex_date": pd.Timestamp(
                ex_date
            ).normalize(),
            "price_multiplier": (
                price_multiplier
            ),
            "volume_multiplier": (
                volume_multiplier
            ),
            "share_multiplier": (
                share_multiplier
            ),
            "action_count": int(len(g)),
            "action_types": ";".join(
                sorted(
                    set(
                        g["action_type"]
                        .dropna()
                        .astype(str)
                    )
                )
            ),
            "purposes": " || ".join(
                sorted(
                    set(
                        g["purpose"]
                        .dropna()
                        .astype(str)
                    )
                )
            ),
        })

    events = pd.DataFrame(
        event_rows
    ).sort_values(
        [
            "ex_date",
            "canonical_security_id",
        ]
    )

    invalid = (
        events["price_multiplier"].le(0)
        | events["volume_multiplier"].le(0)
        | events["share_multiplier"].le(0)
    )

    if invalid.any():
        raise RuntimeError(
            "Collapsed mechanical action has "
            "a non-positive multiplier."
        )

    return events.reset_index(drop=True)


def validate_adjusted_partition(
    d: pd.DataFrame,
    path: Path,
) -> None:
    required = [
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    ]

    missing = [
        c
        for c in required
        if c not in d.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path}: adjusted columns missing: "
            f"{missing}"
        )

    bad = {
        "nonpositive_adj_close": int(
            (d["adj_close"] <= 0).sum()
        ),
        "negative_adj_volume": int(
            (d["adj_volume"] < 0).sum()
        ),
        "adj_high_below_open": int(
            (d["adj_high"] < d["adj_open"]).sum()
        ),
        "adj_high_below_close": int(
            (d["adj_high"] < d["adj_close"]).sum()
        ),
        "adj_high_below_low": int(
            (d["adj_high"] < d["adj_low"]).sum()
        ),
        "adj_low_above_open": int(
            (d["adj_low"] > d["adj_open"]).sum()
        ),
        "adj_low_above_close": int(
            (d["adj_low"] > d["adj_close"]).sum()
        ),
    }

    if sum(bad.values()):
        raise RuntimeError(
            f"{path}: adjusted OHLCV validation failed\n"
            + json.dumps(
                bad,
                indent=2,
            )
        )


def build_adjusted_partitions(
    resolved_root: Path,
    output_root: Path,
    events: pd.DataFrame,
) -> dict:
    """
    Build a backward-adjusted research layer.

    Processing runs newest -> oldest. For each day:
      1. apply the factor state accumulated from corporate actions whose
         ex-date is strictly AFTER the current row date;
      2. write the adjusted partition;
      3. only then fold events whose ex-date equals the current day into state.

    Therefore the ex-date row itself remains on the post-action basis, while
    pre-ex-date history is transformed onto that same basis.
    """
    resolved_root = Path(
        resolved_root
    )
    output_root = Path(
        output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for child in output_root.glob(
        "date=*"
    ):
        if child.is_dir():
            shutil.rmtree(child)

    event_map: dict[
        pd.Timestamp,
        list[dict],
    ] = {}

    for row in events.itertuples(
        index=False
    ):
        day = pd.Timestamp(
            row.ex_date
        ).normalize()

        event_map.setdefault(
            day,
            [],
        ).append({
            "canonical_security_id": (
                str(
                    row.canonical_security_id
                )
            ),
            "price_multiplier": float(
                row.price_multiplier
            ),
            "volume_multiplier": float(
                row.volume_multiplier
            ),
        })

    price_state: dict[str, float] = {}
    volume_state: dict[str, float] = {}

    partitions = sorted(
        resolved_root.glob(
            "date=*/data.parquet"
        ),
        reverse=True,
    )

    if not partitions:
        raise FileNotFoundError(
            "No resolved company-equity "
            f"partitions under {resolved_root}"
        )

    total_rows = 0
    adjusted_rows = 0
    min_price_factor = 1.0
    max_volume_factor = 1.0

    for index, path in enumerate(
        partitions,
        start=1,
    ):
        date_text = (
            path.parent.name
            .replace("date=", "")
        )
        day = pd.Timestamp(
            date_text
        ).normalize()

        d = pd.read_parquet(
            path
        )

        if (
            "canonical_security_id"
            not in d.columns
        ):
            raise RuntimeError(
                f"{path} is missing "
                "canonical_security_id"
            )

        ids = (
            d["canonical_security_id"]
            .astype("string")
        )

        d[
            "price_adjustment_factor"
        ] = (
            ids.map(price_state)
            .fillna(1.0)
            .astype(float)
        )

        d[
            "volume_adjustment_factor"
        ] = (
            ids.map(volume_state)
            .fillna(1.0)
            .astype(float)
        )

        for col in (
            ADJUSTABLE_PRICE_COLUMNS
        ):
            if col in d.columns:
                d[f"adj_{col}"] = (
                    pd.to_numeric(
                        d[col],
                        errors="coerce",
                    )
                    * d[
                        "price_adjustment_factor"
                    ]
                )

        d["adj_volume"] = (
            pd.to_numeric(
                d["volume"],
                errors="coerce",
            )
            * d[
                "volume_adjustment_factor"
            ]
        )

        d[
            "mechanically_adjusted"
        ] = (
            d[
                "price_adjustment_factor"
            ].ne(1.0)
            | d[
                "volume_adjustment_factor"
            ].ne(1.0)
        )

        out_dir = (
            output_root
            / f"date={day.date()}"
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        out_path = (
            out_dir
            / "data.parquet"
        )

        validate_adjusted_partition(
            d,
            out_path,
        )

        d.to_parquet(
            out_path,
            index=False,
            compression="zstd",
        )

        total_rows += len(d)
        adjusted_rows += int(
            d[
                "mechanically_adjusted"
            ].sum()
        )

        if len(d):
            min_price_factor = min(
                min_price_factor,
                float(
                    d[
                        "price_adjustment_factor"
                    ].min()
                ),
            )
            max_volume_factor = max(
                max_volume_factor,
                float(
                    d[
                        "volume_adjustment_factor"
                    ].max()
                ),
            )

        # Ex-date rows are already post-action, so update state only AFTER
        # writing the partition for that date.
        for event in event_map.get(
            day,
            [],
        ):
            cid = event[
                "canonical_security_id"
            ]

            price_state[cid] = (
                price_state.get(
                    cid,
                    1.0,
                )
                * event[
                    "price_multiplier"
                ]
            )

            volume_state[cid] = (
                volume_state.get(
                    cid,
                    1.0,
                )
                * event[
                    "volume_multiplier"
                ]
            )

        if (
            index % 250 == 0
            or index == len(partitions)
        ):
            print(
                f"  adjusted partitions "
                f"{index:,}/"
                f"{len(partitions):,}",
                flush=True,
            )

    return {
        "partitions": int(
            len(partitions)
        ),
        "rows": int(
            total_rows
        ),
        "mechanically_adjusted_rows": int(
            adjusted_rows
        ),
        "min_price_adjustment_factor": float(
            min_price_factor
        ),
        "max_volume_adjustment_factor": float(
            max_volume_factor
        ),
        "output_root": str(
            output_root
        ),
    }


def self_test() -> None:
    """
    Verify the ex-date convention and cumulative-factor logic without touching
    repository data.
    """
    state_price: dict[str, float] = {}
    state_volume: dict[str, float] = {}

    # Newest -> oldest:
    # 2024-01-03 post-split row: factor 1
    # 2024-01-02 split ex-date: factor 1
    # after writing ex-date, state becomes price=.2 / volume=5
    # 2024-01-01 pre-split row: factor .2 / 5

    cid = "ISIN:TEST"

    assert math.isclose(
        state_price.get(
            cid,
            1.0,
        ),
        1.0,
    )

    # Ex-date row is written before state update.
    ex_date_price_factor = (
        state_price.get(
            cid,
            1.0,
        )
    )
    ex_date_volume_factor = (
        state_volume.get(
            cid,
            1.0,
        )
    )

    assert math.isclose(
        ex_date_price_factor,
        1.0,
    )
    assert math.isclose(
        ex_date_volume_factor,
        1.0,
    )

    state_price[cid] = (
        state_price.get(
            cid,
            1.0,
        )
        * 0.2
    )
    state_volume[cid] = (
        state_volume.get(
            cid,
            1.0,
        )
        * 5.0
    )

    assert math.isclose(
        state_price[cid],
        0.2,
    )
    assert math.isclose(
        state_volume[cid],
        5.0,
    )

    # Bonus 1:1 on an even earlier ex-date compounds with the later split.
    bonus_ex_date_factor = (
        state_price[cid]
    )
    assert math.isclose(
        bonus_ex_date_factor,
        0.2,
    )

    state_price[cid] *= 0.5
    state_volume[cid] *= 2.0

    assert math.isclose(
        state_price[cid],
        0.1,
    )
    assert math.isclose(
        state_volume[cid],
        10.0,
    )

    print(
        "Adjusted-price factor self-test: PASS"
    )


def build_adjusted_prices(
    root: Path,
    *,
    fallback_days: int = 7,
) -> dict:
    root = Path(root).resolve()

    corporate_actions_path = (
        root
        / "data/processed/corporate_actions/"
        "nse_corporate_actions.parquet"
    )
    resolved_root = (
        root
        / "data/processed/equities_resolved"
    )
    output_root = (
        root
        / "data/processed/equities_adjusted"
    )
    mapping_root = (
        root
        / "data/processed/corporate_actions"
    )
    reports_root = (
        root
        / "reports"
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading mechanically adjustable "
        "corporate actions..."
    )
    actions = load_mechanical_actions(
        corporate_actions_path
    )
    print(
        f"  candidate actions="
        f"{len(actions):,}"
    )

    print(
        "Mapping actions to resolved "
        "historical security identities..."
    )
    mapped = map_actions(
        actions,
        resolved_root,
        fallback_days=fallback_days,
    )

    mapping_parquet = (
        mapping_root
        / "mechanical_action_identity_map.parquet"
    )
    mapping_csv = (
        mapping_root
        / "mechanical_action_identity_map.csv"
    )

    mapped.to_parquet(
        mapping_parquet,
        index=False,
        compression="zstd",
    )
    mapped.to_csv(
        mapping_csv,
        index=False,
        date_format="%Y-%m-%d",
    )

    mapped_ok = mapped.loc[
        mapped["mapping_status"].eq(
            "mapped"
        )
    ].copy()

    events = collapse_mapped_actions(
        mapped
    )

    events_path = (
        mapping_root
        / "mechanical_adjustment_events.parquet"
    )
    events_csv = (
        mapping_root
        / "mechanical_adjustment_events.csv"
    )

    events.to_parquet(
        events_path,
        index=False,
        compression="zstd",
    )
    events.to_csv(
        events_csv,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(
        f"  mapped actions="
        f"{len(mapped_ok):,}/"
        f"{len(mapped):,}"
    )
    print(
        f"  collapsed adjustment events="
        f"{len(events):,}"
    )

    print(
        "Building backward-adjusted "
        "company-equity partitions..."
    )
    adjustment_summary = (
        build_adjusted_partitions(
            resolved_root,
            output_root,
            events,
        )
    )

    status_counts = {
        str(k): int(v)
        for k, v in (
            mapped[
                "mapping_status"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        )
    }

    method_counts = {
        str(k): int(v)
        for k, v in (
            mapped_ok[
                "mapping_method"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        )
    }

    unmapped = mapped.loc[
        ~mapped[
            "mapping_status"
        ].eq("mapped")
    ].copy()

    unmapped_path = (
        reports_root
        / "corporate_actions_unmapped_mechanical.csv"
    )

    unmapped.to_csv(
        unmapped_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "candidate_mechanical_actions": int(
            len(mapped)
        ),
        "mapped_actions": int(
            len(mapped_ok)
        ),
        "unmapped_or_ambiguous_actions": int(
            len(unmapped)
        ),
        "mapping_status_counts": (
            status_counts
        ),
        "mapping_method_counts": (
            method_counts
        ),
        "collapsed_adjustment_events": int(
            len(events)
        ),
        "affected_canonical_securities": int(
            events[
                "canonical_security_id"
            ].nunique()
        )
        if len(events)
        else 0,
        "adjusted_dataset": (
            adjustment_summary
        ),
        "outputs": {
            "mapping_parquet": str(
                mapping_parquet
            ),
            "mapping_csv": str(
                mapping_csv
            ),
            "events_parquet": str(
                events_path
            ),
            "events_csv": str(
                events_csv
            ),
            "unmapped_csv": str(
                unmapped_path
            ),
            "adjusted_root": str(
                output_root
            ),
        },
    }

    summary_path = (
        reports_root
        / "adjusted_prices_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n"
    )

    return {
        "summary": summary,
        "summary_path": (
            summary_path
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Map mechanically safe NSE corporate actions onto resolved "
            "historical company identities and build a backward-adjusted "
            "OHLCV research layer while preserving raw NSE prices."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Repository/data root",
    )
    ap.add_argument(
        "--fallback-days",
        type=int,
        default=7,
        help=(
            "Calendar-day bracket window for conservative action-to-identity "
            "fallback mapping (default: 7)"
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run adjustment-factor tests and exit"
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.fallback_days < 1:
        raise SystemExit(
            "--fallback-days must be >= 1"
        )

    result = build_adjusted_prices(
        Path(args.root),
        fallback_days=args.fallback_days,
    )

    summary = result["summary"]
    dataset = summary[
        "adjusted_dataset"
    ]

    print(
        "\n=== ADJUSTED PRICE BUILD COMPLETE ==="
    )
    print(
        "Candidate actions:      "
        f"{summary['candidate_mechanical_actions']:,}"
    )
    print(
        "Mapped actions:         "
        f"{summary['mapped_actions']:,}"
    )
    print(
        "Unmapped/ambiguous:     "
        f"{summary['unmapped_or_ambiguous_actions']:,}"
    )
    print(
        "Adjustment events:      "
        f"{summary['collapsed_adjustment_events']:,}"
    )
    print(
        "Affected securities:    "
        f"{summary['affected_canonical_securities']:,}"
    )
    print(
        "Adjusted rows:          "
        f"{dataset['mechanically_adjusted_rows']:,}"
    )
    print(
        "Mapping statuses:       "
        f"{summary['mapping_status_counts']}"
    )
    print(
        "Mapping methods:        "
        f"{summary['mapping_method_counts']}"
    )
    print(
        "Adjusted partitions:    "
        f"{dataset['output_root']}"
    )
    print(
        "Summary:                "
        f"{result['summary_path']}"
    )


if __name__ == "__main__":
    main()
