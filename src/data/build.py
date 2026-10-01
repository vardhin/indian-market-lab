from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

BHAV_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)
SECURITY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "NSE_CM_security_{dmy}.csv.gz"
)
EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
ETF_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def download(url: str, dest: Path, required: bool = False) -> bool:
    """Download with aria2c using headers NSE accepts."""
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
        "--max-tries=3",
        "--retry-wait=2",
        "--continue=true",
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
        raise RuntimeError(f"Download failed:\n{url}\n\n{p.stdout}")
    return False


def reference_sets(ref_dir: Path) -> tuple[set[str], set[str]]:
    """Current NSE lists are used as labels/checks, not as a historical universe filter."""
    eq_path = ref_dir / "EQUITY_L.csv"
    etf_path = ref_dir / "eq_etfseclist.csv"

    download(EQUITY_URL, eq_path, required=True)
    download(ETF_URL, etf_path, required=True)

    eq = clean_cols(pd.read_csv(eq_path, low_memory=False))
    etf = clean_cols(pd.read_csv(etf_path, low_memory=False))

    eq_isins = set(eq["ISIN NUMBER"].dropna().astype(str).str.strip())
    etf_isins = set(etf["ISINNumber"].dropna().astype(str).str.strip())
    return eq_isins, etf_isins


def parse_master_date(s: pd.Series) -> pd.Series:
    """
    NSE CM security-master dates are seconds from 1980-01-01.
    0 means no date.

    Verified against:
      INFY      476668800 -> 1995-02-08
      RELIANCE  502070400 -> 1995-11-29
      TCS       777859200 -> 2004-08-25
    """
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask = n.gt(0)
    out.loc[mask] = pd.Timestamp("1980-01-01") + pd.to_timedelta(
        n.loc[mask], unit="s"
    )
    return out


def get_bhav_csv(day: pd.Timestamp, raw_dir: Path) -> Path | None:
    ymd = day.strftime("%Y%m%d")
    name = f"BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
    zpath = raw_dir / name

    if not download(BHAV_URL.format(ymd=ymd), zpath):
        return None

    with ZipFile(zpath) as z:
        csvs = [x for x in z.namelist() if x.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise RuntimeError(f"{zpath}: expected 1 CSV, found {len(csvs)}")

        out = raw_dir / Path(csvs[0]).name
        if not out.exists():
            z.extract(csvs[0], raw_dir)
            extracted = raw_dir / csvs[0]
            if extracted != out:
                extracted.replace(out)
        return out


def get_security_master(day: pd.Timestamp, raw_dir: Path) -> Path | None:
    dmy = day.strftime("%d%m%Y")
    path = raw_dir / f"NSE_CM_security_{dmy}.csv.gz"
    return path if download(SECURITY_URL.format(dmy=dmy), path) else None


def load_security(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    m = clean_cols(pd.read_csv(path, compression="gzip", low_memory=False))
    wanted = [
        "FinInstrmId", "TckrSymb", "SctySrs", "ISIN",
        "NewBrdLotQty", "ParVal", "TickSz",
        "IssdCptl", "FreeFltCptl",
        "ListgDt", "RmvlDt", "RadmssnDt",
        "DelFlg", "SctyStsNrmlMkt", "ElgbltyNrmlMkt",
        "PreOpnAllwdFlg", "SLBMElgblty", "TradToTradInd",
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
        m["instrument_id"] = pd.to_numeric(m["instrument_id"], errors="coerce")

    for raw, parsed in (
        ("listing_date_raw", "listing_date"),
        ("removal_date_raw", "removal_date"),
        ("readmission_date_raw", "readmission_date"),
    ):
        if raw in m:
            m[parsed] = parse_master_date(m[raw])

    return m


def validate(df: pd.DataFrame) -> dict:
    r = {
        "rows": int(len(df)),
        "unique_isins": int(df["isin"].nunique()),
        "duplicate_date_isin": int(df.duplicated(["date", "isin"]).sum()),
        "missing_isin": int(df["isin"].isna().sum()),
        "missing_symbol": int(df["symbol"].isna().sum()),
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "negative_volume": int((df["volume"] < 0).sum()),
        "negative_turnover": int((df["turnover"] < 0).sum()),
        "high_below_open": int((df["high"] < df["open"]).sum()),
        "high_below_close": int((df["high"] < df["close"]).sum()),
        "high_below_low": int((df["high"] < df["low"]).sum()),
        "low_above_open": int((df["low"] > df["open"]).sum()),
        "low_above_close": int((df["low"] > df["close"]).sum()),
    }

    for c in ("open", "high", "low", "close", "prev_close", "volume"):
        r[f"missing_{c}"] = int(df[c].isna().sum())

    ignored = {"rows", "unique_isins"}
    r["critical_error_count"] = int(
        sum(v for k, v in r.items() if k not in ignored)
    )
    return r


def build_day(
    day: pd.Timestamp,
    raw_root: Path,
    processed_root: Path,
    eq_refs: set[str],
    etf_refs: set[str],
) -> tuple[pd.DataFrame | None, dict]:
    raw_dir = raw_root / day.strftime("%Y/%m/%d")
    raw_dir.mkdir(parents=True, exist_ok=True)

    bhav_csv = get_bhav_csv(day, raw_dir)
    if bhav_csv is None:
        return None, {"date": str(day.date()), "status": "no_bhavcopy"}

    sec_path = get_security_master(day, raw_dir)
    sec = load_security(sec_path)

    b = clean_cols(pd.read_csv(bhav_csv, low_memory=False))
    needed = [
        "TradDt", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs", "FinInstrmNm",
        "OpnPric", "HghPric", "LwPric", "ClsPric", "LastPric", "PrvsClsgPric",
        "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd",
    ]
    missing = [c for c in needed if c not in b.columns]
    if missing:
        raise RuntimeError(f"{day.date()}: missing bhavcopy columns: {missing}")

    d = b.loc[b["SctySrs"].eq("EQ"), needed].copy()
    d = d.rename(columns={
        "TradDt": "date",
        "FinInstrmId": "instrument_id",
        "ISIN": "isin",
        "TckrSymb": "symbol",
        "SctySrs": "series",
        "FinInstrmNm": "name",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "LastPric": "last",
        "PrvsClsgPric": "prev_close",
        "TtlTradgVol": "volume",
        "TtlTrfVal": "turnover",
        "TtlNbOfTxsExctd": "trades",
    })

    d["date"] = pd.to_datetime(d["date"])
    for c in ("isin", "symbol", "series", "name"):
        d[c] = d[c].astype("string").str.strip()

    d["instrument_id"] = pd.to_numeric(d["instrument_id"], errors="coerce")
    for c in ("open", "high", "low", "close", "last", "prev_close",
              "volume", "turnover", "trades"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Classification:
    # - current reference lists confirm current securities;
    # - INF prefix catches historical ETFs no longer present in today's ETF list;
    # - non-ETF EQ rows are retained so historical delisted equities are not lost.
    in_eq_ref = d["isin"].isin(eq_refs)
    in_etf_ref = d["isin"].isin(etf_refs)
    inf_prefix = d["isin"].fillna("").str.startswith("INF")

    d["is_current_equity_reference"] = in_eq_ref
    d["is_current_etf_reference"] = in_etf_ref
    d["is_etf"] = in_etf_ref | (~in_eq_ref & inf_prefix)
    d["is_company_equity"] = ~d["is_etf"]
    d["asset_class"] = d["is_etf"].map({True: "etf", False: "company_equity"})

    d["classification_source"] = "eq_non_etf_point_in_time"
    d.loc[in_eq_ref, "classification_source"] = "current_equity_reference"
    d.loc[in_etf_ref, "classification_source"] = "current_etf_reference"
    d.loc[
        ~in_eq_ref & ~in_etf_ref & inf_prefix,
        "classification_source",
    ] = "isin_INF_fallback"

    d["is_test_instrument"] = (
        d["symbol"].fillna("").str.contains("NSETEST", case=False, regex=False)
        | d["isin"].fillna("").str.startswith("DUMMY")
    )

    if sec is not None:
        # Preserve a point-in-time exchange snapshot separately.
        inst_dir = processed_root / "instruments" / f"date={day.date()}"
        inst_dir.mkdir(parents=True, exist_ok=True)
        sec.assign(as_of_date=day).to_parquet(
            inst_dir / "data.parquet", index=False, compression="zstd"
        )

        keys = ["instrument_id", "symbol", "series", "isin"]
        sec_eq = sec.loc[sec["series"].eq("EQ")].drop_duplicates(keys, keep="last")
        d = d.merge(sec_eq, how="left", on=keys, validate="one_to_one")

    optional = [
        "board_lot", "par_value", "tick_size", "issued_capital",
        "free_float_capital", "listing_date_raw", "removal_date_raw",
        "readmission_date_raw", "listing_date", "removal_date",
        "readmission_date", "deleted_flag", "normal_market_status",
        "normal_market_eligible", "preopen_allowed", "slbm_eligible",
        "trade_to_trade",
    ]
    for c in optional:
        if c not in d:
            d[c] = pd.NA

    deleted = (
        d["deleted_flag"].astype("string").str.upper().eq("Y").fillna(False)
    )
    normal_elig = pd.to_numeric(d["normal_market_eligible"], errors="coerce")
    normal_ok = normal_elig.isna() | normal_elig.eq(1)

    d["eligible_basic_equity"] = (
        d["is_company_equity"]
        & ~d["is_test_instrument"]
        & ~deleted
        & normal_ok
    )

    d["daily_return"] = d["close"] / d["prev_close"] - 1
    d["intraday_return"] = d["close"] / d["open"] - 1
    d["range_pct"] = (
        (d["high"] - d["low"]) / d["open"].where(d["open"] != 0)
    )

    total_eq = len(d)
    etfs = int(d["is_etf"].sum())

    d = (
        d.loc[d["eligible_basic_equity"]]
        .sort_values(["date", "isin"])
        .reset_index(drop=True)
    )

    report = validate(d)
    if report["critical_error_count"]:
        raise RuntimeError(
            f"{day.date()}: validation failed\n{json.dumps(report, indent=2)}"
        )

    out_dir = processed_root / "equities" / f"date={day.date()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out_dir / "data.parquet", index=False, compression="zstd")

    return d, {
        "date": str(day.date()),
        "status": "ok",
        "bhav_eq_rows": int(total_eq),
        "etf_rows": etfs,
        "company_equity_rows": int(len(d)),
        "security_master": sec is not None,
        "validation": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise SystemExit("--to must be >= --from")

    root = Path(args.root).resolve()
    raw = root / "data/raw"
    ref = root / "data/reference"
    processed = root / "data/processed"
    reports = root / "reports"
    for p in (raw, ref, processed, reports):
        p.mkdir(parents=True, exist_ok=True)

    print("Loading NSE reference lists...")
    eq_refs, etf_refs = reference_sets(ref)
    print(f"  company refs: {len(eq_refs):,}")
    print(f"  ETF refs:     {len(etf_refs):,}\n")

    frames: list[pd.DataFrame] = []
    statuses: list[dict] = []

    for day in pd.date_range(start, end, freq="D"):
        if day.weekday() >= 5:
            statuses.append({"date": str(day.date()), "status": "weekend"})
            continue

        print(f"[{day.date()}] ", end="", flush=True)
        frame, status = build_day(day, raw, processed, eq_refs, etf_refs)
        statuses.append(status)

        if frame is None:
            print("no bhavcopy")
        else:
            frames.append(frame)
            print(
                f"{len(frame):,} equities "
                f"(security master: {'yes' if status['security_master'] else 'no'})"
            )

    if not frames:
        raise SystemExit("No trading days were built.")

    full = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "isin"])
        .reset_index(drop=True)
    )

    range_report = validate(full)
    range_report.update({
        "dates": int(full["date"].nunique()),
        "first_date": str(full["date"].min().date()),
        "last_date": str(full["date"].max().date()),
        "unique_symbols": int(full["symbol"].nunique()),
    })
    if range_report["critical_error_count"]:
        raise RuntimeError(
            "Range validation failed\n" + json.dumps(range_report, indent=2)
        )

    snapshot = (
        processed
        / f"nse_daily_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    )
    canonical = processed / "nse_daily.parquet"

    full.to_parquet(snapshot, index=False, compression="zstd")
    full.to_parquet(canonical, index=False, compression="zstd")

    report_path = reports / f"data_quality_{start:%Y%m%d}_{end:%Y%m%d}.json"
    with report_path.open("w") as f:
        json.dump(
            {
                "requested_from": str(start.date()),
                "requested_to": str(end.date()),
                "range_validation": range_report,
                "days": statuses,
            },
            f,
            indent=2,
            default=str,
        )

    missing_weekdays = [
        x["date"] for x in statuses if x["status"] == "no_bhavcopy"
    ]

    print("\n=== BUILD COMPLETE ===")
    print(f"Trading days:   {full['date'].nunique():,}")
    print(f"Rows:           {len(full):,}")
    print(f"Unique ISINs:   {full['isin'].nunique():,}")
    print(f"Range:          {full['date'].min().date()} -> {full['date'].max().date()}")
    print(f"Parquet:        {snapshot}")
    print(f"Canonical:      {canonical}")
    print(f"Quality report: {report_path}")

    if missing_weekdays:
        print("\nWeekdays with no bhavcopy (usually NSE holidays; review):")
        for x in missing_weekdays:
            print(" ", x)


if __name__ == "__main__":
    main()
