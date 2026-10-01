from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "symbol", "series", "isin"]
OPTIONAL_COLUMNS = ["source_format", "asset_class", "classification_source"]


def _norm_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.eq(""))


def load_identity_frame(input_root: Path) -> pd.DataFrame:
    """Load only the columns needed to resolve historical security identity."""
    input_root = Path(input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input EQ-all root does not exist: {input_root}")

    files = sorted(input_root.glob("date=*/data.parquet"))
    if not files:
        files = sorted(input_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet partitions found under: {input_root}")

    frames: list[pd.DataFrame] = []
    requested = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    for path in files:
        # Fast path: our builder always writes these columns, so Parquet can
        # project only the identity payload rather than reading prices/volume.
        try:
            probe = pd.read_parquet(path, columns=requested)
        except Exception:
            # Compatibility path for an older partition schema.
            probe = pd.read_parquet(path)

        available = set(probe.columns)
        missing = [c for c in REQUIRED_COLUMNS if c not in available]
        if missing:
            raise RuntimeError(
                f"{path} is missing required columns {missing}. "
                f"Available: {sorted(available)}"
            )
        cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in available]
        frames.append(probe[cols].copy())

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ("symbol", "series", "isin"):
        df[c] = _norm_string(df[c])

    bad_date = int(df["date"].isna().sum())
    bad_symbol = int(df["symbol"].isna().sum())
    bad_series = int(df["series"].isna().sum())
    if bad_date or bad_symbol or bad_series:
        raise RuntimeError(
            "Identity input has unusable keys: "
            f"missing_date={bad_date}, missing_symbol={bad_symbol}, "
            f"missing_series={bad_series}"
        )

    dupes = df.duplicated(["date", "symbol", "series"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["date", "symbol", "series", "isin"]].head(50)
        raise RuntimeError(
            "Identity input has duplicate date/symbol/series rows. Sample:\n"
            + sample.to_string(index=False)
        )

    return df.sort_values(["symbol", "series", "date"]).reset_index(drop=True)


def _stable_lineage_id(symbol: str, series: str, start: pd.Timestamp, episode: int) -> str:
    payload = f"{symbol}|{series}|{start.date()}|{episode}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"NSELINEAGE:{digest}"


def _episode_assignments(g: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Resolve one symbol+series timeline conservatively."""
    g = g.sort_values("date").copy()
    gap = g["date"].diff().dt.days.fillna(0)
    g["episode_no"] = gap.gt(max_gap_days).cumsum().astype(int)

    outputs: list[pd.DataFrame] = []

    for episode_no, ep in g.groupby("episode_no", sort=True):
        ep = ep.copy()
        ep_start = ep["date"].min()
        ep_end = ep["date"].max()
        symbol = str(ep["symbol"].iloc[0])
        series = str(ep["series"].iloc[0])
        lineage_id = _stable_lineage_id(symbol, series, ep_start, int(episode_no))

        original = ep["isin"].copy()
        known_mask = original.notna()
        unique_known = list(pd.unique(original[known_mask]))

        ep["resolved_isin"] = original
        ep["resolution_method"] = "unresolved_no_anchor"
        ep["resolution_confidence"] = 0.0
        ep.loc[known_mask, "resolution_method"] = "exact_isin"
        ep.loc[known_mask, "resolution_confidence"] = 1.0

        if len(unique_known) == 1:
            anchor = unique_known[0]
            ep.loc[~known_mask, "resolved_isin"] = anchor

            known_dates = ep.loc[known_mask, "date"]
            first_known = known_dates.min()
            last_known = known_dates.max()

            lead = ~known_mask & ep["date"].lt(first_known)
            trail = ~known_mask & ep["date"].gt(last_known)
            bridge = ~known_mask & ~(lead | trail)

            ep.loc[lead, "resolution_method"] = "backward_single_isin_episode"
            ep.loc[lead, "resolution_confidence"] = 0.98
            ep.loc[bridge, "resolution_method"] = "bridge_single_isin_episode"
            ep.loc[bridge, "resolution_confidence"] = 0.99
            ep.loc[trail, "resolution_method"] = "forward_single_isin_episode"
            ep.loc[trail, "resolution_confidence"] = 0.98

        elif len(unique_known) > 1:
            # Missing rows are resolved only when the nearest explicit ISIN
            # evidence on both sides agrees. Leading/trailing missing rows are
            # mapped to the nearest anchor with lower confidence. Missing rows
            # between different ISIN anchors remain unresolved.
            idx = np.arange(len(ep))
            known_pos = np.flatnonzero(known_mask.to_numpy())
            known_values = original.iloc[known_pos].astype(str).to_numpy()

            for pos in idx[~known_mask.to_numpy()]:
                left_ix = np.searchsorted(known_pos, pos, side="right") - 1
                right_ix = np.searchsorted(known_pos, pos, side="left")

                left_pos = known_pos[left_ix] if left_ix >= 0 else None
                right_pos = known_pos[right_ix] if right_ix < len(known_pos) else None

                if left_pos is None and right_pos is not None:
                    value = str(original.iloc[right_pos])
                    ep.iloc[pos, ep.columns.get_loc("resolved_isin")] = value
                    ep.iloc[pos, ep.columns.get_loc("resolution_method")] = (
                        "backward_first_anchor_multi_isin_episode"
                    )
                    ep.iloc[pos, ep.columns.get_loc("resolution_confidence")] = 0.85
                elif right_pos is None and left_pos is not None:
                    value = str(original.iloc[left_pos])
                    ep.iloc[pos, ep.columns.get_loc("resolved_isin")] = value
                    ep.iloc[pos, ep.columns.get_loc("resolution_method")] = (
                        "forward_last_anchor_multi_isin_episode"
                    )
                    ep.iloc[pos, ep.columns.get_loc("resolution_confidence")] = 0.85
                elif left_pos is not None and right_pos is not None:
                    left_value = str(original.iloc[left_pos])
                    right_value = str(original.iloc[right_pos])
                    if left_value == right_value:
                        ep.iloc[pos, ep.columns.get_loc("resolved_isin")] = left_value
                        ep.iloc[pos, ep.columns.get_loc("resolution_method")] = (
                            "bridge_matching_anchors_multi_isin_episode"
                        )
                        ep.iloc[pos, ep.columns.get_loc("resolution_confidence")] = 0.95
                    else:
                        ep.iloc[pos, ep.columns.get_loc("resolution_method")] = (
                            "unresolved_between_isin_change"
                        )
                        ep.iloc[pos, ep.columns.get_loc("resolution_confidence")] = 0.0

        ep["episode_id"] = lineage_id
        ep["episode_start"] = ep_start
        ep["episode_end"] = ep_end
        ep["episode_known_isin_count"] = len(unique_known)
        ep["episode_multiple_isins"] = len(unique_known) > 1

        ep["canonical_security_id"] = lineage_id
        resolved_mask = ep["resolved_isin"].notna()
        ep.loc[resolved_mask, "canonical_security_id"] = (
            "ISIN:" + ep.loc[resolved_mask, "resolved_isin"].astype(str)
        )

        ep["resolved_asset_class"] = "unresolved"
        ep.loc[resolved_mask, "resolved_asset_class"] = "company_equity"
        inf = ep["resolved_isin"].fillna("").str.startswith("INF")
        ep.loc[resolved_mask & inf, "resolved_asset_class"] = "etf"

        outputs.append(ep)

    return pd.concat(outputs, ignore_index=True)


def resolve_identities(df: pd.DataFrame, max_gap_days: int = 120) -> pd.DataFrame:
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be >= 1")

    chunks: list[pd.DataFrame] = []
    for (_, _), g in df.groupby(["symbol", "series"], sort=True, dropna=False):
        chunks.append(_episode_assignments(g, max_gap_days=max_gap_days))

    out = pd.concat(chunks, ignore_index=True)
    out = out.rename(columns={"isin": "original_isin"})
    out = out.sort_values(["date", "symbol", "series"]).reset_index(drop=True)
    return out


def build_episode_summary(row_map: pd.DataFrame) -> pd.DataFrame:
    def _join_isins(s: pd.Series) -> str:
        vals = sorted(set(s.dropna().astype(str)))
        return ";".join(vals)

    grouped = row_map.groupby(
        ["episode_id", "symbol", "series", "episode_start", "episode_end"],
        dropna=False,
        sort=True,
    )

    summary = grouped.agg(
        rows=("date", "size"),
        exact_isin_rows=("original_isin", lambda s: int(s.notna().sum())),
        resolved_rows=("resolved_isin", lambda s: int(s.notna().sum())),
        unresolved_rows=("resolved_isin", lambda s: int(s.isna().sum())),
        known_isins=("original_isin", _join_isins),
        resolved_isins=("resolved_isin", _join_isins),
        min_confidence=("resolution_confidence", "min"),
        max_confidence=("resolution_confidence", "max"),
        episode_known_isin_count=("episode_known_isin_count", "max"),
        episode_multiple_isins=("episode_multiple_isins", "max"),
    ).reset_index()

    summary["fully_resolved"] = summary["unresolved_rows"].eq(0)
    return summary


def write_identity_outputs(
    row_map: pd.DataFrame,
    output_root: Path,
    *,
    write_consolidated: bool = True,
) -> dict:
    output_root = Path(output_root)
    daily_root = output_root / "daily"
    daily_root.mkdir(parents=True, exist_ok=True)

    # Replace daily identity partitions so re-running after more backfill cannot
    # leave stale mappings behind.
    if daily_root.exists():
        for child in daily_root.glob("date=*"):
            if child.is_dir():
                shutil.rmtree(child)

    cols = [
        "date",
        "symbol",
        "series",
        "original_isin",
        "resolved_isin",
        "canonical_security_id",
        "resolved_asset_class",
        "resolution_method",
        "resolution_confidence",
        "episode_id",
        "episode_start",
        "episode_end",
        "episode_known_isin_count",
        "episode_multiple_isins",
    ]

    for day, g in row_map.groupby("date", sort=True):
        out_dir = daily_root / f"date={pd.Timestamp(day).date()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        g[cols].to_parquet(out_dir / "data.parquet", index=False, compression="zstd")

    episodes = build_episode_summary(row_map)
    episodes_path = output_root / "identity_episodes.parquet"
    episodes.to_parquet(episodes_path, index=False, compression="zstd")

    unresolved = episodes.loc[~episodes["fully_resolved"]].copy()
    unresolved_path = output_root / "unresolved_episodes.parquet"
    unresolved.to_parquet(unresolved_path, index=False, compression="zstd")

    consolidated_path = output_root / "historical_identity_map.parquet"
    if write_consolidated:
        row_map[cols].to_parquet(consolidated_path, index=False, compression="zstd")

    summary = {
        "rows": int(len(row_map)),
        "episodes": int(len(episodes)),
        "date_min": str(row_map["date"].min().date()),
        "date_max": str(row_map["date"].max().date()),
        "original_isin_rows": int(row_map["original_isin"].notna().sum()),
        "resolved_isin_rows": int(row_map["resolved_isin"].notna().sum()),
        "propagated_rows": int(
            row_map["original_isin"].isna().sum()
            - row_map["resolved_isin"].isna().sum()
        ),
        "unresolved_rows": int(row_map["resolved_isin"].isna().sum()),
        "fully_resolved_episodes": int(episodes["fully_resolved"].sum()),
        "unresolved_episodes": int((~episodes["fully_resolved"]).sum()),
        "multi_isin_episodes": int(episodes["episode_multiple_isins"].sum()),
        "resolution_methods": {
            str(k): int(v)
            for k, v in row_map["resolution_method"].value_counts().items()
        },
    }

    summary_path = output_root / "identity_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "episodes_path": episodes_path,
        "unresolved_path": unresolved_path,
        "consolidated_path": consolidated_path if write_consolidated else None,
        "daily_root": daily_root,
    }


def apply_identity_map(
    eq_all_root: Path,
    identity_root: Path,
    resolved_root: Path,
) -> dict:
    """Apply daily identity maps and write a strict resolved company-equity view."""
    eq_all_root = Path(eq_all_root)
    identity_root = Path(identity_root)
    resolved_root = Path(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)

    # Rebuild deterministically.
    for child in resolved_root.glob("date=*"):
        if child.is_dir():
            shutil.rmtree(child)

    day_dirs = sorted(eq_all_root.glob("date=*"))
    total_rows = 0
    company_rows = 0
    etf_rows = 0
    unresolved_rows = 0

    for day_dir in day_dirs:
        data_path = day_dir / "data.parquet"
        if not data_path.exists():
            continue

        date_part = day_dir.name
        map_path = identity_root / "daily" / date_part / "data.parquet"
        if not map_path.exists():
            raise RuntimeError(f"Missing identity map for {date_part}: {map_path}")

        d = pd.read_parquet(data_path)
        m = pd.read_parquet(map_path)

        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        m["date"] = pd.to_datetime(m["date"]).dt.normalize()

        merge_cols = ["date", "symbol", "series"]
        identity_cols = merge_cols + [
            "original_isin",
            "resolved_isin",
            "canonical_security_id",
            "resolved_asset_class",
            "resolution_method",
            "resolution_confidence",
            "episode_id",
        ]

        merged = d.merge(
            m[identity_cols],
            on=merge_cols,
            how="left",
            validate="one_to_one",
        )

        if merged["canonical_security_id"].isna().any():
            bad = merged.loc[
                merged["canonical_security_id"].isna(), merge_cols
            ].head(20)
            raise RuntimeError(
                f"Identity join failed for {date_part}. Sample:\n"
                + bad.to_string(index=False)
            )

        # Keep raw/original identity columns untouched for auditability.
        merged["isin_resolved"] = merged["resolved_isin"]
        merged["isin_resolution_method"] = merged["resolution_method"]
        merged["isin_resolution_confidence"] = merged["resolution_confidence"]

        company = merged.loc[
            merged["resolved_asset_class"].eq("company_equity")
            & merged["isin_resolved"].notna()
        ].copy()

        company = company.sort_values(["date", "canonical_security_id"])
        if company.duplicated(["date", "canonical_security_id"]).any():
            dup = company.loc[
                company.duplicated(["date", "canonical_security_id"], keep=False),
                ["date", "symbol", "series", "isin_resolved", "canonical_security_id"],
            ].head(30)
            raise RuntimeError(
                f"Resolved company universe has duplicate canonical IDs on {date_part}.\n"
                + dup.to_string(index=False)
            )

        out_dir = resolved_root / date_part
        out_dir.mkdir(parents=True, exist_ok=True)
        company.to_parquet(out_dir / "data.parquet", index=False, compression="zstd")

        total_rows += len(merged)
        company_rows += len(company)
        etf_rows += int(merged["resolved_asset_class"].eq("etf").sum())
        unresolved_rows += int(merged["resolved_asset_class"].eq("unresolved").sum())

    return {
        "input_rows": int(total_rows),
        "resolved_company_rows": int(company_rows),
        "resolved_etf_rows": int(etf_rows),
        "unresolved_rows": int(unresolved_rows),
        "output_root": str(resolved_root),
    }


def resolve_identity_dataset(
    input_root: Path,
    output_root: Path,
    *,
    resolved_root: Path | None = None,
    max_gap_days: int = 120,
    write_consolidated: bool = True,
) -> dict:
    print("Loading EQ-all identity timeline...")
    df = load_identity_frame(Path(input_root))
    print(
        f"  rows={len(df):,}, dates={df['date'].nunique():,}, "
        f"symbols={df['symbol'].nunique():,}, explicit_ISIN={df['isin'].notna().sum():,}"
    )

    print(f"Resolving symbol/series episodes (max gap {max_gap_days} days)...")
    row_map = resolve_identities(df, max_gap_days=max_gap_days)

    outputs = write_identity_outputs(
        row_map,
        Path(output_root),
        write_consolidated=write_consolidated,
    )

    apply_summary = None
    if resolved_root is not None:
        print("Applying identity map to daily EQ partitions...")
        apply_summary = apply_identity_map(
            Path(input_root),
            Path(output_root),
            Path(resolved_root),
        )

    result = {
        **outputs,
        "apply_summary": apply_summary,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Resolve missing historical NSE ISINs conservatively from "
            "continuous symbol/series timelines."
        )
    )
    ap.add_argument(
        "--input",
        default="data/processed/eq_all",
        help="Partitioned EQ-all root",
    )
    ap.add_argument(
        "--output",
        default="data/processed/identity",
        help="Identity-map output root",
    )
    ap.add_argument(
        "--resolved-output",
        default="data/processed/equities_resolved",
        help="Resolved company-equity partition root",
    )
    ap.add_argument(
        "--max-gap-days",
        type=int,
        default=120,
        help=(
            "Split a symbol timeline after this many calendar days without an "
            "observation. Conservative protection against symbol reuse."
        ),
    )
    ap.add_argument(
        "--no-apply",
        action="store_true",
        help="Build identity maps only; do not write resolved equity partitions",
    )
    ap.add_argument(
        "--no-consolidated-map",
        action="store_true",
        help="Skip the single consolidated historical_identity_map.parquet",
    )
    args = ap.parse_args()

    result = resolve_identity_dataset(
        Path(args.input),
        Path(args.output),
        resolved_root=None if args.no_apply else Path(args.resolved_output),
        max_gap_days=args.max_gap_days,
        write_consolidated=not args.no_consolidated_map,
    )

    summary = result["summary"]
    print("\n=== IDENTITY RESOLUTION COMPLETE ===")
    print(f"Rows:                 {summary['rows']:,}")
    print(f"Episodes:             {summary['episodes']:,}")
    print(f"Original ISIN rows:   {summary['original_isin_rows']:,}")
    print(f"Resolved ISIN rows:   {summary['resolved_isin_rows']:,}")
    print(f"Propagated rows:      {summary['propagated_rows']:,}")
    print(f"Unresolved rows:      {summary['unresolved_rows']:,}")
    print(f"Unresolved episodes:  {summary['unresolved_episodes']:,}")
    print(f"Multi-ISIN episodes:  {summary['multi_isin_episodes']:,}")
    print(f"Summary:              {result['summary_path']}")
    print(f"Episodes:             {result['episodes_path']}")
    print(f"Unresolved:           {result['unresolved_path']}")
    if result["consolidated_path"] is not None:
        print(f"Row map:              {result['consolidated_path']}")

    if result["apply_summary"] is not None:
        a = result["apply_summary"]
        print(f"Resolved companies:   {a['resolved_company_rows']:,}")
        print(f"Resolved ETFs:        {a['resolved_etf_rows']:,}")
        print(f"Still unresolved:     {a['unresolved_rows']:,}")
        print(f"Resolved partitions:  {a['output_root']}")


if __name__ == "__main__":
    main()
