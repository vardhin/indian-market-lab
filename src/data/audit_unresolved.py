from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = [
    "date",
    "symbol",
    "series",
    "original_isin",
    "resolved_isin",
    "resolution_method",
    "resolution_confidence",
    "episode_id",
    "episode_start",
    "episode_end",
    "episode_known_isin_count",
    "episode_multiple_isins",
]

MARKET_COLUMNS = [
    "date",
    "symbol",
    "series",
    "close",
    "volume",
    "turnover",
]


def _norm_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.eq(""))


def _join_unique(s: pd.Series) -> str:
    vals = sorted(set(s.dropna().astype(str)))
    return ";".join(vals)


def load_identity_map(identity_root: Path) -> pd.DataFrame:
    """
    Load the row-level identity map.

    Prefer the consolidated historical_identity_map.parquet written by the
    resolver. Fall back to its daily partitions so the audit also works when
    --no-consolidated-map was used.
    """
    identity_root = Path(identity_root)
    consolidated = identity_root / "historical_identity_map.parquet"

    if consolidated.is_file():
        df = pd.read_parquet(consolidated, columns=IDENTITY_COLUMNS)
    else:
        files = sorted((identity_root / "daily").glob("date=*/data.parquet"))
        if not files:
            raise FileNotFoundError(
                "No identity map found. Expected either "
                f"{consolidated} or daily partitions under "
                f"{identity_root / 'daily'}"
            )

        frames = [
            pd.read_parquet(path, columns=IDENTITY_COLUMNS)
            for path in files
        ]
        df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["episode_start"] = pd.to_datetime(
        df["episode_start"], errors="coerce"
    ).dt.normalize()
    df["episode_end"] = pd.to_datetime(
        df["episode_end"], errors="coerce"
    ).dt.normalize()

    for col in (
        "symbol",
        "series",
        "original_isin",
        "resolved_isin",
        "resolution_method",
        "episode_id",
    ):
        df[col] = _norm_string(df[col])

    bad = int(
        df[["date", "symbol", "series", "episode_id"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    if bad:
        raise RuntimeError(
            f"Identity map contains {bad:,} rows with unusable audit keys."
        )

    return df


def unresolved_episode_rows(identity: pd.DataFrame) -> pd.DataFrame:
    """Keep every row belonging to an episode that is not fully resolved."""
    unresolved_ids = set(
        identity.loc[
            identity["resolved_isin"].isna(),
            "episode_id",
        ].astype(str)
    )

    if not unresolved_ids:
        return identity.iloc[0:0].copy()

    return (
        identity.loc[identity["episode_id"].isin(unresolved_ids)]
        .sort_values(["episode_id", "date"])
        .reset_index(drop=True)
    )


def attach_anchor_context(ep: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, attach the nearest explicit ISIN anchor on each side.

    This makes unresolved rows directly interpretable:
      - same left/right anchor -> potentially recoverable evidence gap
      - different left/right anchors -> intentional ISIN-change ambiguity
      - no anchors -> genuinely unidentified historical episode
    """
    ep = ep.sort_values("date").copy()

    explicit = ep["original_isin"]

    ep["left_anchor_isin"] = explicit.ffill()
    ep["right_anchor_isin"] = explicit.bfill()

    explicit_date = ep["date"].where(explicit.notna())
    ep["left_anchor_date"] = explicit_date.ffill()
    ep["right_anchor_date"] = explicit_date.bfill()

    return ep


def collect_market_rows(
    eq_all_root: Path,
    episode_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load only market rows needed by unresolved episodes.

    The full EQ history is millions of rows, but the audit needs only the
    date/symbol/series keys belonging to the  unresolved episodes. We therefore
    project six columns from only the required daily partitions.
    """
    eq_all_root = Path(eq_all_root)

    keys_by_date: dict[pd.Timestamp, set[tuple[str, str]]] = {}
    for day, g in episode_rows.groupby("date", sort=True):
        keys_by_date[pd.Timestamp(day).normalize()] = set(
            zip(
                g["symbol"].astype(str),
                g["series"].astype(str),
            )
        )

    frames: list[pd.DataFrame] = []
    missing_partitions: list[str] = []

    for day, wanted in keys_by_date.items():
        path = (
            eq_all_root
            / f"date={day.date()}"
            / "data.parquet"
        )

        if not path.is_file():
            missing_partitions.append(str(day.date()))
            continue

        d = pd.read_parquet(path, columns=MARKET_COLUMNS)
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d["symbol"] = _norm_string(d["symbol"])
        d["series"] = _norm_string(d["series"])

        key = list(zip(d["symbol"].astype(str), d["series"].astype(str)))
        keep = pd.Series(
            [x in wanted for x in key],
            index=d.index,
        )

        d = d.loc[keep].copy()

        for col in ("close", "volume", "turnover"):
            d[col] = pd.to_numeric(d[col], errors="coerce")

        frames.append(d)

    if not frames:
        return pd.DataFrame(columns=MARKET_COLUMNS), missing_partitions

    out = pd.concat(frames, ignore_index=True)

    if out.duplicated(["date", "symbol", "series"]).any():
        sample = out.loc[
            out.duplicated(
                ["date", "symbol", "series"],
                keep=False,
            ),
            ["date", "symbol", "series"],
        ].head(20)
        raise RuntimeError(
            "Duplicate market rows found while auditing unresolved identity. "
            "Sample:\n"
            + sample.to_string(index=False)
        )

    return out, missing_partitions


def anchor_transition_pairs(ep: pd.DataFrame) -> str:
    """Summarize left->right explicit anchor pairs seen on unresolved rows."""
    u = ep.loc[ep["resolved_isin"].isna()].copy()
    if u.empty:
        return ""

    pairs: set[str] = set()

    for row in u.itertuples(index=False):
        left = (
            str(row.left_anchor_isin)
            if pd.notna(row.left_anchor_isin)
            else "<NONE>"
        )
        right = (
            str(row.right_anchor_isin)
            if pd.notna(row.right_anchor_isin)
            else "<NONE>"
        )
        pairs.add(f"{left}->{right}")

    return ";".join(sorted(pairs))


def summarize_episode(ep: pd.DataFrame) -> dict:
    ep = attach_anchor_context(ep)

    unresolved = ep.loc[ep["resolved_isin"].isna()].copy()
    explicit = ep.loc[ep["original_isin"].notna()].copy()

    methods = _join_unique(unresolved["resolution_method"])

    ambiguous_change_rows = int(
        unresolved["resolution_method"]
        .eq("unresolved_between_isin_change")
        .sum()
    )
    no_anchor_rows = int(
        unresolved["resolution_method"]
        .eq("unresolved_no_anchor")
        .sum()
    )

    if ambiguous_change_rows and no_anchor_rows:
        identity_risk = "mixed"
    elif ambiguous_change_rows:
        identity_risk = "between_isin_change"
    elif no_anchor_rows:
        identity_risk = "no_anchor"
    else:
        identity_risk = "other_unresolved"

    first_explicit_date = (
        explicit["date"].min()
        if not explicit.empty
        else pd.NaT
    )
    last_explicit_date = (
        explicit["date"].max()
        if not explicit.empty
        else pd.NaT
    )

    return {
        "episode_id": str(ep["episode_id"].iloc[0]),
        "symbol": str(ep["symbol"].iloc[0]),
        "series": str(ep["series"].iloc[0]),
        "episode_start": ep["episode_start"].iloc[0],
        "episode_end": ep["episode_end"].iloc[0],
        "episode_days": int(
            (
                ep["episode_end"].iloc[0]
                - ep["episode_start"].iloc[0]
            ).days
            + 1
        ),
        "episode_rows": int(len(ep)),
        "unresolved_rows": int(len(unresolved)),
        "unresolved_start": unresolved["date"].min(),
        "unresolved_end": unresolved["date"].max(),
        "explicit_isin_rows": int(ep["original_isin"].notna().sum()),
        "resolved_isin_rows": int(ep["resolved_isin"].notna().sum()),
        "known_isins": _join_unique(ep["original_isin"]),
        "resolved_isins": _join_unique(ep["resolved_isin"]),
        "known_isin_count": int(ep["episode_known_isin_count"].max()),
        "multiple_isins": bool(ep["episode_multiple_isins"].max()),
        "unresolved_methods": methods,
        "identity_risk": identity_risk,
        "ambiguous_change_rows": ambiguous_change_rows,
        "no_anchor_rows": no_anchor_rows,
        "anchor_transition_pairs": anchor_transition_pairs(ep),
        "first_explicit_isin_date": first_explicit_date,
        "last_explicit_isin_date": last_explicit_date,
    }


def add_market_statistics(
    episode_summary: pd.DataFrame,
    episode_rows: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    identity_market = episode_rows.merge(
        market,
        on=["date", "symbol", "series"],
        how="left",
        validate="one_to_one",
    )

    identity_market["is_unresolved"] = identity_market["resolved_isin"].isna()

    rows: list[dict] = []

    for episode_id, g in identity_market.groupby("episode_id", sort=True):
        u = g.loc[g["is_unresolved"]].copy()

        turnover = pd.to_numeric(u["turnover"], errors="coerce")
        volume = pd.to_numeric(u["volume"], errors="coerce")
        close = pd.to_numeric(u["close"], errors="coerce")

        rows.append({
            "episode_id": str(episode_id),
            "market_rows_found": int(u["turnover"].notna().sum()),
            "market_rows_missing": int(u["turnover"].isna().sum()),
            "unresolved_total_turnover": float(turnover.fillna(0).sum()),
            "unresolved_mean_daily_turnover": float(turnover.mean())
            if turnover.notna().any()
            else 0.0,
            "unresolved_median_daily_turnover": float(turnover.median())
            if turnover.notna().any()
            else 0.0,
            "unresolved_max_daily_turnover": float(turnover.max())
            if turnover.notna().any()
            else 0.0,
            "unresolved_total_volume": float(volume.fillna(0).sum()),
            "unresolved_median_volume": float(volume.median())
            if volume.notna().any()
            else 0.0,
            "unresolved_median_close": float(close.median())
            if close.notna().any()
            else 0.0,
            "unresolved_min_close": float(close.min())
            if close.notna().any()
            else 0.0,
            "unresolved_max_close": float(close.max())
            if close.notna().any()
            else 0.0,
            "unresolved_zero_volume_days": int(volume.fillna(0).eq(0).sum()),
            "unresolved_active_days": int(volume.fillna(0).gt(0).sum()),
        })

    stats = pd.DataFrame(rows)

    out = episode_summary.merge(
        stats,
        on="episode_id",
        how="left",
        validate="one_to_one",
    )

    numeric_fill = [
        "market_rows_found",
        "market_rows_missing",
        "unresolved_total_turnover",
        "unresolved_mean_daily_turnover",
        "unresolved_median_daily_turnover",
        "unresolved_max_daily_turnover",
        "unresolved_total_volume",
        "unresolved_median_volume",
        "unresolved_median_close",
        "unresolved_min_close",
        "unresolved_max_close",
        "unresolved_zero_volume_days",
        "unresolved_active_days",
    ]
    out[numeric_fill] = out[numeric_fill].fillna(0)

    return out


def add_economic_relevance(audit: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by actual turnover trapped in unresolved rows.

    Bands are deliberately relative rather than arbitrary rupee thresholds:
      A: episodes accounting for the first 80% of unresolved turnover
      B: next 15% (80-95%)
      C: final 5%

    Identity-risk flags remain separate, so a low-turnover but structurally
    ambiguous ISIN-change episode is still easy to find.
    """
    audit = audit.sort_values(
        [
            "unresolved_total_turnover",
            "unresolved_rows",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    total = float(audit["unresolved_total_turnover"].sum())

    if total > 0:
        audit["turnover_share"] = (
            audit["unresolved_total_turnover"] / total
        )
        audit["cumulative_turnover_share"] = audit["turnover_share"].cumsum()
    else:
        audit["turnover_share"] = 0.0
        audit["cumulative_turnover_share"] = 0.0

    before = audit["cumulative_turnover_share"] - audit["turnover_share"]

    audit["economic_relevance_band"] = "C_tail_5pct"
    audit.loc[before < 0.95, "economic_relevance_band"] = "B_next_15pct"
    audit.loc[before < 0.80, "economic_relevance_band"] = "A_top_80pct"

    def priority(row: pd.Series) -> str:
        high_economic = row["economic_relevance_band"] == "A_top_80pct"
        ambiguous = row["identity_risk"] in {
            "between_isin_change",
            "mixed",
        }

        if high_economic and ambiguous:
            return "P0_economic_and_identity_risk"
        if ambiguous:
            return "P1_identity_risk"
        if high_economic:
            return "P1_economic_relevance"
        if row["economic_relevance_band"] == "B_next_15pct":
            return "P2"
        return "P3"

    audit["review_priority"] = audit.apply(priority, axis=1)

    return audit


def write_markdown(
    audit: pd.DataFrame,
    output_path: Path,
    *,
    total_unresolved_rows: int,
    total_unresolved_turnover: float,
    missing_partitions: list[str],
    top_n: int,
) -> None:
    top = audit.head(top_n).copy()

    lines = [
        "# Unresolved Historical Identity Audit",
        "",
        "This report ranks unresolved NSE identity episodes by the actual trading "
        "turnover occurring in unresolved rows. It does not force a mapping; "
        "the underlying conservative identity rules remain unchanged.",
        "",
        "## Summary",
        "",
        f"- Unresolved episodes: {len(audit):,}",
        f"- Unresolved rows: {total_unresolved_rows:,}",
        f"- Turnover inside unresolved rows: ₹{total_unresolved_turnover:,.2f}",
        f"- Missing EQ-all partitions during audit: {len(missing_partitions):,}",
        "",
        "Economic relevance bands are relative to total turnover inside the "
        "unresolved set: A covers the first 80%, B the next 15%, and C the "
        "remaining tail.",
        "",
        f"## Top {min(top_n, len(top))} episodes by unresolved turnover",
        "",
        "| Priority | Symbol | Start | End | Unresolved rows | Identity risk | "
        "Unresolved turnover | Share | Anchor transitions |",
        "|---|---|---|---|---:|---|---:|---:|---|",
    ]

    for row in top.itertuples(index=False):
        lines.append(
            "| "
            f"{row.review_priority} | "
            f"{row.symbol} | "
            f"{pd.Timestamp(row.unresolved_start).date()} | "
            f"{pd.Timestamp(row.unresolved_end).date()} | "
            f"{int(row.unresolved_rows):,} | "
            f"{row.identity_risk} | "
            f"₹{float(row.unresolved_total_turnover):,.2f} | "
            f"{float(row.turnover_share):.2%} | "
            f"{row.anchor_transition_pairs or '-'} |"
        )

    if missing_partitions:
        lines += [
            "",
            "## Missing partitions",
            "",
            "These EQ-all daily partitions were required by unresolved episodes "
            "but were not found:",
            "",
        ]
        lines.extend(f"- {x}" for x in missing_partitions)

    output_path.write_text("\n".join(lines) + "\n")


def audit_unresolved_identities(
    identity_root: Path,
    eq_all_root: Path,
    output_root: Path,
    *,
    top_n: int = 30,
) -> dict:
    identity_root = Path(identity_root)
    eq_all_root = Path(eq_all_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading identity map...")
    identity = load_identity_map(identity_root)

    unresolved_rows = identity.loc[identity["resolved_isin"].isna()].copy()
    episode_rows = unresolved_episode_rows(identity)

    if unresolved_rows.empty:
        summary = {
            "identity_rows": int(len(identity)),
            "unresolved_rows": 0,
            "unresolved_episodes": 0,
            "message": "Identity map is fully resolved.",
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("Identity map is fully resolved; nothing to audit.")
        return {
            "summary": summary,
            "summary_path": summary_path,
        }

    print(
        f"  unresolved rows={len(unresolved_rows):,}, "
        f"episodes={episode_rows['episode_id'].nunique():,}"
    )

    print("Building episode/anchor diagnostics...")
    episode_summary = pd.DataFrame([
        summarize_episode(g)
        for _, g in episode_rows.groupby("episode_id", sort=True)
    ])

    print("Loading market activity for unresolved episodes...")
    market, missing_partitions = collect_market_rows(
        eq_all_root,
        episode_rows,
    )

    audit = add_market_statistics(
        episode_summary,
        episode_rows,
        market,
    )
    audit = add_economic_relevance(audit)

    parquet_path = output_root / "unresolved_identity_audit.parquet"
    csv_path = output_root / "unresolved_identity_audit.csv"
    markdown_path = output_root / "unresolved_identity_audit.md"
    summary_path = output_root / "summary.json"

    audit.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )
    audit.to_csv(csv_path, index=False)

    total_turnover = float(audit["unresolved_total_turnover"].sum())

    priority_counts = {
        str(k): int(v)
        for k, v in audit["review_priority"].value_counts().items()
    }
    risk_counts = {
        str(k): int(v)
        for k, v in audit["identity_risk"].value_counts().items()
    }
    band_counts = {
        str(k): int(v)
        for k, v in audit["economic_relevance_band"].value_counts().items()
    }

    summary = {
        "identity_rows": int(len(identity)),
        "unresolved_rows": int(len(unresolved_rows)),
        "unresolved_episodes": int(audit["episode_id"].nunique()),
        "unresolved_symbols": int(audit["symbol"].nunique()),
        "unresolved_total_turnover": total_turnover,
        "priority_counts": priority_counts,
        "identity_risk_counts": risk_counts,
        "economic_relevance_band_counts": band_counts,
        "missing_market_partitions": missing_partitions,
        "outputs": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    write_markdown(
        audit,
        markdown_path,
        total_unresolved_rows=len(unresolved_rows),
        total_unresolved_turnover=total_turnover,
        missing_partitions=missing_partitions,
        top_n=top_n,
    )

    return {
        "summary": summary,
        "summary_path": summary_path,
        "parquet_path": parquet_path,
        "csv_path": csv_path,
        "markdown_path": markdown_path,
        "audit": audit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit unresolved historical NSE identity episodes and rank them "
            "by economic relevance without forcing ambiguous mappings."
        )
    )
    ap.add_argument(
        "--identity-root",
        default="data/processed/identity",
        help="Identity resolver output root",
    )
    ap.add_argument(
        "--eq-all-root",
        default="data/processed/eq_all",
        help="Partitioned EQ-all dataset root",
    )
    ap.add_argument(
        "--output",
        default="reports/identity_audit",
        help="Audit output directory",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top-turnover episodes to include in Markdown report",
    )
    args = ap.parse_args()

    if args.top < 1:
        raise SystemExit("--top must be >= 1")

    result = audit_unresolved_identities(
        Path(args.identity_root),
        Path(args.eq_all_root),
        Path(args.output),
        top_n=args.top,
    )

    summary = result["summary"]

    print("\n=== UNRESOLVED IDENTITY AUDIT COMPLETE ===")
    print(f"Unresolved rows:      {summary['unresolved_rows']:,}")
    print(f"Unresolved episodes:  {summary['unresolved_episodes']:,}")

    if summary["unresolved_rows"]:
        print(
            "Unresolved turnover:  "
            f"₹{summary['unresolved_total_turnover']:,.2f}"
        )
        print(f"Priority counts:      {summary['priority_counts']}")
        print(f"Identity risks:       {summary['identity_risk_counts']}")
        print(f"Summary:              {result['summary_path']}")
        print(f"Audit Parquet:        {result['parquet_path']}")
        print(f"Audit CSV:            {result['csv_path']}")
        print(f"Human report:         {result['markdown_path']}")


if __name__ == "__main__":
    main()
