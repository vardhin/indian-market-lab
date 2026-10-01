from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from bhavcopy import (
    UDIFF_START,
    fetch_bhavcopy,
    parse_bhavcopy,
    validate_bhavcopy,
)


UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

SECURITY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "NSE_CM_security_{dmy}.csv.gz"
)

EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

ETF_URL = (
    "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def download_plain(url: str, dest: Path, required: bool = False) -> bool:
    """Download non-ZIP NSE files via aria2c."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return True

    cmd = [
        "aria2c",
        "-x", "1",
        "-s", "1",
        "--disable-ipv6=true",
        "--connect-timeout=15",
        "--timeout=30",
        "--max-tries=2",
        "--retry-wait=1",
        "--file-allocation=none",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--summary-interval=0",
        "--console-log-level=warn",
        f"--user-agent={UA}",
        "--header=Accept-Language: en-US,en;q=0.9",
        "--referer=https://www.nseindia.com/",
        "--dir", str(dest.parent),
        "--out", dest.name,
        url,
    ]

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    ok = p.returncode == 0 and dest.exists() and dest.stat().st_size > 0

    if ok:
        return True

    dest.unlink(missing_ok=True)
    Path(str(dest) + ".aria2").unlink(missing_ok=True)

    if required:
        raise RuntimeError(
            f"Download failed:\n{url}\n\n{p.stdout}"
        )

    return False


def load_reference_sets(
    ref_dir: Path,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Current lists are classification hints only.

    They MUST NOT be used to discard historical symbols, otherwise we would
    introduce survivorship bias.
    """
    eq_path = ref_dir / "EQUITY_L.csv"
    etf_path = ref_dir / "eq_etfseclist.csv"

    download_plain(EQUITY_URL, eq_path, required=True)
    download_plain(ETF_URL, etf_path, required=True)

    eq = clean_columns(pd.read_csv(eq_path, low_memory=False))
    etf = clean_columns(pd.read_csv(etf_path, low_memory=False))

    eq_isins = set(
        eq["ISIN NUMBER"].dropna().astype(str).str.strip()
    )
    etf_isins = set(
        etf["ISINNumber"].dropna().astype(str).str.strip()
    )

    eq_symbols = set(
        eq["SYMBOL"].dropna().astype(str).str.strip()
    )
    etf_symbols = set(
        etf["Symbol"].dropna().astype(str).str.strip()
    )

    return eq_isins, etf_isins, eq_symbols, etf_symbols


def parse_master_date(s: pd.Series) -> pd.Series:
    """
    Numeric dates in the MII security master are seconds from 1980-01-01.
    Zero means no date.
    """
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask = n.gt(0)

    out.loc[mask] = (
        pd.Timestamp("1980-01-01")
        + pd.to_timedelta(n.loc[mask], unit="s")
    )
    return out


def get_security_master(
    day: pd.Timestamp,
    raw_dir: Path,
) -> Path | None:
    """
    We only request the MII security master in the UDiFF era.

    Historical legacy bhavcopies predate this modern daily master, so trying
    this URL thousands of times for 2010-2024 is just noise.
    """
    if day < UDIFF_START:
        return None

    dmy = day.strftime("%d%m%Y")
    path = raw_dir / f"NSE_CM_security_{dmy}.csv.gz"

    return (
        path
        if download_plain(
            SECURITY_URL.format(dmy=dmy),
            path,
            required=False,
        )
        else None
    )


def load_security_master(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    m = clean_columns(
        pd.read_csv(
            path,
            compression="gzip",
            low_memory=False,
        )
    )

    wanted = [
        "FinInstrmId",
        "TckrSymb",
        "SctySrs",
        "ISIN",
        "NewBrdLotQty",
        "ParVal",
        "TickSz",
        "IssdCptl",
        "FreeFltCptl",
        "ListgDt",
        "RmvlDt",
        "RadmssnDt",
        "DelFlg",
        "SctyStsNrmlMkt",
        "ElgbltyNrmlMkt",
        "PreOpnAllwdFlg",
        "SLBMElgblty",
        "TradToTradInd",
    ]

    m = m[[c for c in wanted if c in m.columns]].copy()

    m = m.rename(columns={
        "FinInstrmId": "instrument_id",
        "TckrSymb": "symbol",
        "SctySrs": "series",
        "ISIN": "isin",
        "NewBrdLotQty": "board_lot",
        "ParVal": "par_value",
        "TickSz": "tick_size",
        "IssdCptl": "issued_capital",
        "FreeFltCptl": "free_float_capital",
        "ListgDt": "listing_date_raw",
        "RmvlDt": "removal_date_raw",
        "RadmssnDt": "readmission_date_raw",
        "DelFlg": "deleted_flag",
        "SctyStsNrmlMkt": "normal_market_status",
        "ElgbltyNrmlMkt": "normal_market_eligible",
        "PreOpnAllwdFlg": "preopen_allowed",
        "SLBMElgblty": "slbm_eligible",
        "TradToTradInd": "trade_to_trade",
    })

    for c in ("isin", "symbol", "series"):
        if c in m:
            m[c] = m[c].astype("string").str.strip()

    if "instrument_id" in m:
        m["instrument_id"] = pd.to_numeric(
            m["instrument_id"],
            errors="coerce",
        ).astype("Int64")

    for raw, parsed in (
        ("listing_date_raw", "listing_date"),
        ("removal_date_raw", "removal_date"),
        ("readmission_date_raw", "readmission_date"),
    ):
        if raw in m:
            m[parsed] = parse_master_date(m[raw])

    return m


def classify_eq_rows(
    d: pd.DataFrame,
    eq_isins: set[str],
    etf_isins: set[str],
    eq_symbols: set[str],
    etf_symbols: set[str],
) -> pd.DataFrame:
    """
    Classify EQ-series rows without deleting unresolved history.

    ISIN is authoritative enough for the basic ETF/company split:
      INF... -> mutual-fund/ETF security
      other EQ ISIN -> company-equity candidate

    For rows without ISIN (common in old bhavcopies), current symbol lists are
    only hints. Unknown historical symbols stay UNKNOWN and remain preserved in
    eq_all rather than being silently discarded.
    """
    d = d.copy()

    has_isin = d["isin"].notna()
    isin_inf = d["isin"].fillna("").str.startswith("INF")

    current_eq_isin = d["isin"].isin(eq_isins)
    current_etf_isin = d["isin"].isin(etf_isins)

    current_eq_symbol = d["symbol"].isin(eq_symbols)
    current_etf_symbol = d["symbol"].isin(etf_symbols)

    d["is_current_equity_reference"] = current_eq_isin
    d["is_current_etf_reference"] = current_etf_isin

    d["asset_class"] = "equity_or_etf_unknown"
    d["classification_source"] = "unresolved_legacy_identity"

    # Exact/current ETF reference has highest priority.
    mask = current_etf_isin
    d.loc[mask, "asset_class"] = "etf"
    d.loc[mask, "classification_source"] = "current_etf_isin_reference"

    # INF is the Indian mutual-fund ISIN prefix and catches historical ETFs
    # that are no longer in today's ETF list.
    mask = has_isin & isin_inf & ~current_etf_isin
    d.loc[mask, "asset_class"] = "etf"
    d.loc[mask, "classification_source"] = "isin_INF"

    # Any non-INF ISIN in EQ series is retained as a company-equity candidate.
    mask = has_isin & ~isin_inf & ~current_etf_isin
    d.loc[mask, "asset_class"] = "company_equity"
    d.loc[mask, "classification_source"] = "eq_series_non_INF_isin"

    # Current reference can strengthen that classification.
    mask = current_eq_isin & ~current_etf_isin
    d.loc[mask, "asset_class"] = "company_equity"
    d.loc[mask, "classification_source"] = "current_equity_isin_reference"

    # For missing-ISIN legacy rows, current symbol matches are useful labels,
    # but these rows are explicitly marked as symbol-based so we never confuse
    # them with point-in-time identity.
    no_isin = ~has_isin

    mask = no_isin & current_etf_symbol
    d.loc[mask, "asset_class"] = "etf"
    d.loc[mask, "classification_source"] = "current_etf_symbol_hint"

    mask = no_isin & current_eq_symbol & ~current_etf_symbol
    d.loc[mask, "asset_class"] = "company_equity"
    d.loc[mask, "classification_source"] = "current_equity_symbol_hint"

    d["is_etf"] = d["asset_class"].eq("etf")
    d["is_company_equity"] = d["asset_class"].eq("company_equity")
    d["classification_resolved"] = ~d["asset_class"].eq(
        "equity_or_etf_unknown"
    )

    return d


def add_modern_security_metadata(
    d: pd.DataFrame,
    sec: pd.DataFrame | None,
    day: pd.Timestamp,
    processed_root: Path,
) -> pd.DataFrame:
    """Attach same-day MII metadata when available."""
    d = d.copy()

    if sec is not None:
        inst_dir = (
            processed_root
            / "instruments"
            / f"date={day.date()}"
        )
        inst_dir.mkdir(parents=True, exist_ok=True)

        sec.assign(as_of_date=day).to_parquet(
            inst_dir / "data.parquet",
            index=False,
            compression="zstd",
        )

        keys = [
            "instrument_id",
            "symbol",
            "series",
            "isin",
        ]

        sec_eq = (
            sec.loc[sec["series"].eq("EQ")]
            .drop_duplicates(keys, keep="last")
        )

        d = d.merge(
            sec_eq,
            how="left",
            on=keys,
            validate="one_to_one",
        )

    optional = [
        "board_lot",
        "par_value",
        "tick_size",
        "issued_capital",
        "free_float_capital",
        "listing_date_raw",
        "removal_date_raw",
        "readmission_date_raw",
        "listing_date",
        "removal_date",
        "readmission_date",
        "deleted_flag",
        "normal_market_status",
        "normal_market_eligible",
        "preopen_allowed",
        "slbm_eligible",
        "trade_to_trade",
    ]

    for c in optional:
        if c not in d:
            d[c] = pd.NA

    return d


def add_derived_fields(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    d["daily_return"] = (
        d["close"] / d["prev_close"] - 1
    )

    d["intraday_return"] = (
        d["close"] / d["open"].where(d["open"] != 0) - 1
    )

    d["range_pct"] = (
        (d["high"] - d["low"])
        / d["open"].where(d["open"] != 0)
    )

    d["is_test_instrument"] = (
        d["symbol"]
        .fillna("")
        .str.contains(
            "NSETEST",
            case=False,
            regex=False,
        )
        |
        d["isin"]
        .fillna("")
        .str.startswith("DUMMY")
    )

    deleted = (
        d["deleted_flag"]
        .astype("string")
        .str.upper()
        .eq("Y")
        .fillna(False)
    )

    normal_elig = pd.to_numeric(
        d["normal_market_eligible"],
        errors="coerce",
    )

    normal_ok = (
        normal_elig.isna()
        | normal_elig.eq(1)
    )

    d["eligible_basic_equity"] = (
        d["is_company_equity"]
        & ~d["is_test_instrument"]
        & ~deleted
        & normal_ok
    )

    return d


def validate_company_equities(d: pd.DataFrame) -> dict:
    """Strict validation for the resolved EQ company-equity view."""
    report = validate_bhavcopy(
        d,
        strict_market_rows=True,
        require_series=True,
    )

    report["unknown_asset_class"] = int(
        d["asset_class"].eq("equity_or_etf_unknown").sum()
    )
    report["etf_rows"] = int(d["is_etf"].sum())

    # Those should both be zero in this filtered view.
    report["critical_error_count"] += (
        report["unknown_asset_class"]
        + report["etf_rows"]
    )

    return report


def build_day(
    day: pd.Timestamp,
    raw_root: Path,
    processed_root: Path,
    refs: tuple[set[str], set[str], set[str], set[str]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    raw_dir = raw_root / day.strftime("%Y/%m/%d")

    try:
        csv_path = fetch_bhavcopy(day, raw_root)
    except FileNotFoundError:
        return None, None, {
            "date": str(day.date()),
            "status": "no_bhavcopy",
        }

    parsed = parse_bhavcopy(
        csv_path,
        expected_day=day,
    )

    raw_report = validate_bhavcopy(
        parsed,
        strict_market_rows=False,
        require_series=False,
    )
    if raw_report["critical_error_count"]:
        raise RuntimeError(
            f"{day.date()}: raw bhavcopy validation failed\n"
            + json.dumps(raw_report, indent=2)
        )

    eq_all = (
        parsed.loc[parsed["series"].eq("EQ")]
        .copy()
        .reset_index(drop=True)
    )

    eq_isins, etf_isins, eq_symbols, etf_symbols = refs

    eq_all = classify_eq_rows(
        eq_all,
        eq_isins,
        etf_isins,
        eq_symbols,
        etf_symbols,
    )

    sec_path = get_security_master(
        day,
        raw_dir,
    )

    sec = load_security_master(sec_path)

    eq_all = add_modern_security_metadata(
        eq_all,
        sec,
        day,
        processed_root,
    )

    eq_all = add_derived_fields(eq_all)

    eq_all_dir = (
        processed_root
        / "eq_all"
        / f"date={day.date()}"
    )
    eq_all_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eq_all.to_parquet(
        eq_all_dir / "data.parquet",
        index=False,
        compression="zstd",
    )

    equities = (
        eq_all.loc[eq_all["eligible_basic_equity"]]
        .sort_values(["date", "market_row_key"])
        .reset_index(drop=True)
    )

    company_report = validate_company_equities(equities)

    if company_report["critical_error_count"]:
        raise RuntimeError(
            f"{day.date()}: company-equity validation failed\n"
            + json.dumps(company_report, indent=2)
        )

    equities_dir = (
        processed_root
        / "equities"
        / f"date={day.date()}"
    )
    equities_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    equities.to_parquet(
        equities_dir / "data.parquet",
        index=False,
        compression="zstd",
    )

    status = {
        "date": str(day.date()),
        "status": "ok",
        "source_format": (
            str(parsed["source_format"].iloc[0])
            if len(parsed)
            else None
        ),
        "all_bhav_rows": int(len(parsed)),
        "eq_rows": int(len(eq_all)),
        "company_equity_rows": int(len(equities)),
        "etf_rows": int(eq_all["is_etf"].sum()),
        "unresolved_eq_rows": int(
            eq_all["asset_class"]
            .eq("equity_or_etf_unknown")
            .sum()
        ),
        "exact_isin_eq_rows": int(
            eq_all["isin"].notna().sum()
        ),
        "fallback_identity_eq_rows": int(
            eq_all["isin"].isna().sum()
        ),
        "security_master": sec is not None,
        "raw_validation": raw_report,
        "company_validation": company_report,
    }

    return eq_all, equities, status


def consolidate(
    frames: list[pd.DataFrame],
    out: Path,
) -> pd.DataFrame:
    full = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "market_row_key"])
        .reset_index(drop=True)
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full.to_parquet(
        out,
        index=False,
        compression="zstd",
    )

    return full


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build normalized NSE daily cash-market data across both "
            "legacy and UDiFF bhavcopy eras."
        )
    )

    ap.add_argument(
        "--from",
        dest="start",
        required=True,
        help="YYYY-MM-DD",
    )

    ap.add_argument(
        "--to",
        dest="end",
        required=True,
        help="YYYY-MM-DD",
    )

    ap.add_argument(
        "--root",
        default=".",
    )

    ap.add_argument(
        "--no-consolidate",
        action="store_true",
        help=(
            "Write daily partitions only. Recommended for very large "
            "multi-year backfills."
        ),
    )

    ap.add_argument(
        "--resolve-identities",
        action="store_true",
        help=(
            "After the backfill, build the conservative historical symbol/ISIN "
            "identity map and resolved company-equity partitions."
        ),
    )

    ap.add_argument(
        "--identity-max-gap-days",
        type=int,
        default=120,
        help=(
            "Split a symbol lineage after this many calendar days without a "
            "trading observation (default: 120)."
        ),
    )

    args = ap.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    if end < start:
        raise SystemExit(
            "--to must be >= --from"
        )

    root = Path(args.root).resolve()

    raw = root / "data/raw"
    ref = root / "data/reference"
    processed = root / "data/processed"
    reports = root / "reports"

    for p in (
        raw,
        ref,
        processed,
        reports,
    ):
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Loading NSE reference lists...")

    refs = load_reference_sets(ref)

    print(
        f"  current company ISIN refs: "
        f"{len(refs[0]):,}"
    )
    print(
        f"  current ETF ISIN refs:     "
        f"{len(refs[1]):,}"
    )
    print()

    eq_all_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    statuses: list[dict] = []

    for day in pd.date_range(
        start,
        end,
        freq="D",
    ):
        if day.weekday() >= 5:
            statuses.append({
                "date": str(day.date()),
                "status": "weekend",
            })
            continue

        print(
            f"[{day.date()}] ",
            end="",
            flush=True,
        )

        eq_all, equities, status = build_day(
            day,
            raw,
            processed,
            refs,
        )

        statuses.append(status)

        if status["status"] != "ok":
            print("no bhavcopy")
            continue

        print(
            f"{status['source_format']} | "
            f"EQ={status['eq_rows']:,} | "
            f"companies={status['company_equity_rows']:,} | "
            f"ETF={status['etf_rows']:,} | "
            f"unresolved={status['unresolved_eq_rows']:,} | "
            f"ISIN-missing={status['fallback_identity_eq_rows']:,}"
        )

        if not args.no_consolidate:
            eq_all_frames.append(eq_all)
            equity_frames.append(equities)

    ok_days = [
        s for s in statuses
        if s["status"] == "ok"
    ]

    if not ok_days:
        raise SystemExit(
            "No trading days were built."
        )

    if not args.no_consolidate:
        range_tag = (
            f"{start:%Y%m%d}_{end:%Y%m%d}"
        )

        eq_all_out = (
            processed
            / f"nse_eq_all_{range_tag}.parquet"
        )

        company_out = (
            processed
            / f"nse_daily_{range_tag}.parquet"
        )

        eq_all_full = consolidate(
            eq_all_frames,
            eq_all_out,
        )

        company_full = consolidate(
            equity_frames,
            company_out,
        )

        # Keep the convenient canonical company-equity path for short builds.
        company_full.to_parquet(
            processed / "nse_daily.parquet",
            index=False,
            compression="zstd",
        )

        summary = {
            "eq_all_rows": int(len(eq_all_full)),
            "company_equity_rows": int(len(company_full)),
            "unique_eq_security_keys": int(
                eq_all_full["security_key"].nunique()
            ),
            "unique_company_security_keys": int(
                company_full["security_key"].nunique()
            ),
        }
    else:
        eq_all_out = None
        company_out = None
        summary = {}

    report = {
        "requested_from": str(start.date()),
        "requested_to": str(end.date()),
        "udiff_start": str(UDIFF_START.date()),
        "successful_trading_days": len(ok_days),
        "legacy_days": sum(
            s.get("source_format") == "legacy"
            for s in ok_days
        ),
        "udiff_days": sum(
            s.get("source_format") == "udiff"
            for s in ok_days
        ),
        "days": statuses,
        **summary,
    }

    report_path = (
        reports
        / f"data_quality_{start:%Y%m%d}_{end:%Y%m%d}.json"
    )

    with report_path.open("w") as f:
        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    missing_weekdays = [
        s["date"]
        for s in statuses
        if s["status"] == "no_bhavcopy"
    ]

    unresolved_days = [
        (
            s["date"],
            s["unresolved_eq_rows"],
            s["fallback_identity_eq_rows"],
        )
        for s in ok_days
        if s["unresolved_eq_rows"] > 0
        or s["fallback_identity_eq_rows"] > 0
    ]

    print("\n=== BUILD COMPLETE ===")
    print(
        f"Trading days: {len(ok_days):,}"
    )
    print(
        f"Legacy days:  {report['legacy_days']:,}"
    )
    print(
        f"UDiFF days:   {report['udiff_days']:,}"
    )

    if eq_all_out is not None:
        print(
            f"EQ-all:       {eq_all_out}"
        )
        print(
            f"Companies:    {company_out}"
        )

    print(
        f"Quality:      {report_path}"
    )

    if missing_weekdays:
        print(
            "\nWeekdays with no bhavcopy "
            "(usually holidays; review):"
        )
        for x in missing_weekdays:
            print(" ", x)

    if unresolved_days:
        print(
            "\nHistorical identity/classification gaps preserved "
            "for later resolution:"
        )
        for date, unresolved, missing_isin in unresolved_days:
            print(
                f"  {date}: "
                f"unresolved={unresolved:,}, "
                f"missing_ISIN={missing_isin:,}"
            )

    if args.resolve_identities:
        from resolve_identity import resolve_identity_dataset

        print("\n=== HISTORICAL IDENTITY RESOLUTION ===")
        identity_result = resolve_identity_dataset(
            processed / "eq_all",
            processed / "identity",
            resolved_root=processed / "equities_resolved",
            max_gap_days=args.identity_max_gap_days,
            write_consolidated=True,
        )
        ident = identity_result["summary"]
        print(f"Propagated ISIN rows: {ident['propagated_rows']:,}")
        print(f"Unresolved rows:      {ident['unresolved_rows']:,}")
        print(f"Unresolved episodes:  {ident['unresolved_episodes']:,}")
        print(f"Identity summary:      {identity_result['summary_path']}")
        if identity_result["apply_summary"] is not None:
            a = identity_result["apply_summary"]
            print(f"Resolved companies:   {a['resolved_company_rows']:,}")
            print(f"Resolved partitions:  {a['output_root']}")


if __name__ == "__main__":
    main()
