from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import shutil
import pandas as pd


# NSE officially discontinued the legacy cash-market bhavcopy effective
# 2024-07-08 and moved to UDiFF.
UDIFF_START = pd.Timestamp("2024-07-08")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

MODERN_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)

# Both hosts have historically been used for NSE archives. Try the canonical
# archives host first and nsearchives as a fallback.
LEGACY_URLS = (
    "https://archives.nseindia.com/content/historical/EQUITIES/"
    "{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip",
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip",
)


CANONICAL_COLUMNS = [
    "date",
    "source_format",
    "instrument_id",
    "isin",
    "security_key",
    "market_row_key",
    "identity_quality",
    "symbol",
    "series",
    "name",
    "open",
    "high",
    "low",
    "close",
    "last",
    "prev_close",
    "volume",
    "turnover",
    "trades",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace and drop empty trailing CSV columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    drop = [
        c for c in df.columns
        if not c or c.lower().startswith("unnamed:")
    ]
    if drop:
        df = df.drop(columns=drop)

    return df


def _aria2(url: str, dest: Path) -> tuple[bool, str]:
    """Download one file. Returns (success, captured output)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # A previously validated zip can be reused.
    if dest.exists() and dest.stat().st_size > 0:
        try:
            with ZipFile(dest):
                return True, "cached"
        except BadZipFile:
            dest.unlink(missing_ok=True)

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

    # Do not trust HTTP success blindly: a CDN/error page must not survive as
    # our "zip".
    if ok:
        try:
            with ZipFile(dest):
                return True, p.stdout
        except BadZipFile:
            ok = False

    dest.unlink(missing_ok=True)
    Path(str(dest) + ".aria2").unlink(missing_ok=True)
    return False, p.stdout


def _extract_single_csv(zpath: Path, raw_dir: Path) -> Path:
    with ZipFile(zpath) as zf:
        csvs = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv")
        ]

        if len(csvs) != 1:
            raise RuntimeError(
                f"{zpath}: expected exactly one CSV, found {csvs}"
            )

        member = csvs[0]

        # Always extract into a temporary directory
        tmp = raw_dir / "_extract_tmp"
        tmp.mkdir(parents=True, exist_ok=True)

        zf.extract(member, tmp)

        extracted = tmp / member

        # Find the actual file if archive has nested folders
        if extracted.is_file():
            source = extracted
        else:
            matches = list(tmp.rglob("*.csv"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Could not resolve extracted CSV from {zpath}"
                )
            source = matches[0]

        out = raw_dir / Path(member).name

        # If a directory with same name exists, remove it
        if out.exists() and out.is_dir():
            shutil.rmtree(out)

        if source != out:
            source.replace(out)

        shutil.rmtree(tmp, ignore_errors=True)

        return out

def fetch_bhavcopy(day: pd.Timestamp, raw_root: Path) -> Path:
    """
    Download and extract the appropriate NSE CM bhavcopy for one trading day.

    Raises FileNotFoundError for weekends/holidays/unavailable dates.
    """
    day = pd.Timestamp(day).normalize()
    raw_dir = raw_root / day.strftime("%Y/%m/%d")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if day >= UDIFF_START:
        ymd = day.strftime("%Y%m%d")
        zip_name = f"BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
        zpath = raw_dir / zip_name
        url = MODERN_URL.format(ymd=ymd)

        ok, output = _aria2(url, zpath)
        if not ok:
            raise FileNotFoundError(
                f"No UDiFF bhavcopy available for {day.date()}.\n"
                f"URL: {url}\n"
                f"aria2c output:\n{output}"
            )

        return _extract_single_csv(zpath, raw_dir)

    yyyy = day.strftime("%Y")
    mmm = day.strftime("%b").upper()
    dd = day.strftime("%d")

    zip_name = f"cm{dd}{mmm}{yyyy}bhav.csv.zip"
    zpath = raw_dir / zip_name

    errors: list[str] = []

    for template in LEGACY_URLS:
        url = template.format(
            yyyy=yyyy,
            mmm=mmm,
            dd=dd,
        )
        ok, output = _aria2(url, zpath)

        if ok:
            return _extract_single_csv(zpath, raw_dir)

        errors.append(f"{url}\n{output}")

    raise FileNotFoundError(
        f"No legacy bhavcopy available for {day.date()}.\n\n"
        + "\n\n".join(errors)
    )


def _normalise_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("isin", "symbol", "series", "name"):
        if col not in df:
            df[col] = pd.NA

        df[col] = df[col].astype("string").str.strip()

        # NSE legacy files may represent empty cells as blank strings.
        df.loc[df[col].eq(""), col] = pd.NA

    return df


def _add_security_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build two different keys on purpose.

    security_key:
        Identity of the underlying security across NSE market/series rows.
        When ISIN exists, this is ISIN-based.

    market_row_key:
        Identity of one row in a daily bhavcopy. The same ISIN may legitimately
        appear in more than one NSE series on a day, so series belongs here,
        not in the uniqueness rule for security identity.

    For old rows without ISIN we only have symbol+series, so the fallback
    security identity remains explicitly weaker.
    """
    has_isin = df["isin"].notna()

    df["security_key"] = (
        "NSE:"
        + df["symbol"].fillna("<MISSING>")
        + ":"
        + df["series"].fillna("<MISSING>")
    )

    df.loc[has_isin, "security_key"] = (
        "ISIN:" + df.loc[has_isin, "isin"]
    )

    df["market_row_key"] = (
        df["security_key"]
        + "|SERIES:"
        + df["series"].fillna("<MISSING>")
    )

    df["identity_quality"] = "symbol_series_fallback"
    df.loc[has_isin, "identity_quality"] = "exact_isin"

    return df


def parse_udiff(df: pd.DataFrame) -> pd.DataFrame:
    """Parse NSE UDiFF CM bhavcopy (2024-07-08 onward)."""
    required = {
        "TradDt",
        "FinInstrmId",
        "ISIN",
        "TckrSymb",
        "SctySrs",
        "FinInstrmNm",
        "OpnPric",
        "HghPric",
        "LwPric",
        "ClsPric",
        "LastPric",
        "PrvsClsgPric",
        "TtlTradgVol",
        "TtlTrfVal",
        "TtlNbOfTxsExctd",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"UDiFF bhavcopy missing columns: {missing}")

    rename = {
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
    }

    out = df[list(rename)].rename(columns=rename).copy()
    out["source_format"] = "udiff"
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["instrument_id"] = pd.to_numeric(
        out["instrument_id"], errors="coerce"
    ).astype("Int64")

    return out


def parse_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the pre-UDiFF NSE cash-market bhavcopy.

    TOTALTRADES and ISIN are optional because older legacy files (notably
    pre-2012 files) may not contain them.
    """
    required = {
        "SYMBOL",
        "SERIES",
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "LAST",
        "PREVCLOSE",
        "TOTTRDQTY",
        "TOTTRDVAL",
        "TIMESTAMP",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Legacy bhavcopy missing columns: {missing}")

    rename = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "LAST": "last",
        "PREVCLOSE": "prev_close",
        "TOTTRDQTY": "volume",
        "TOTTRDVAL": "turnover",
        "TIMESTAMP": "date",
        "TOTALTRADES": "trades",
        "ISIN": "isin",
    }

    existing = [c for c in rename if c in df.columns]
    out = df[existing].rename(columns=rename).copy()

    # Fields that did not exist in the legacy format.
    out["instrument_id"] = pd.Series(
        pd.NA, index=out.index, dtype="Int64"
    )
    out["name"] = pd.Series(
        pd.NA, index=out.index, dtype="string"
    )

    if "isin" not in out:
        out["isin"] = pd.Series(
            pd.NA, index=out.index, dtype="string"
        )

    if "trades" not in out:
        out["trades"] = pd.Series(
            pd.NA, index=out.index, dtype="Float64"
        )

    out["source_format"] = "legacy"

    # NSE legacy TIMESTAMP formats vary across years:
    #
    #   04-JAN-2010
    #   26-OCT-2023
    #   13-JUL-20
    #
    # Parse multiple known formats instead of assuming only YYYY years.

    timestamp = (
        out["date"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    parsed = pd.Series(
        pd.NaT,
        index=timestamp.index,
        dtype="datetime64[ns]",
    )

    for fmt in (
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d/%m/%Y",
        "%d/%m/%y",
    ):
        mask = parsed.isna()

        if not mask.any():
            break

        parsed.loc[mask] = pd.to_datetime(
            timestamp.loc[mask],
            format=fmt,
            errors="coerce",
        )

    # Last-resort parser for future NSE variations.
    mask = parsed.isna()

    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            timestamp.loc[mask],
            errors="coerce",
            dayfirst=True,
        )

    out["date"] = parsed

    return out


def parse_bhavcopy(
    csv_path: Path,
    expected_day: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Parse either NSE cash-market schema into one stable canonical schema.
    """
    raw = clean_columns(
        pd.read_csv(csv_path, low_memory=False)
    )

    cols = set(raw.columns)

    if "TradDt" in cols:
        out = parse_udiff(raw)
    elif {"SYMBOL", "TIMESTAMP"} <= cols:
        out = parse_legacy(raw)
    else:
        raise ValueError(
            f"Unrecognised NSE bhavcopy schema in {csv_path}.\n"
            f"Columns: {sorted(raw.columns)}"
        )

    out = _normalise_identifiers(out)

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "last",
        "prev_close",
        "volume",
        "turnover",
        "trades",
    ]

    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _add_security_key(out)

    # Keep the schema and column order invariant across 2010 -> present.
    for col in CANONICAL_COLUMNS:
        if col not in out:
            out[col] = pd.NA

    out = out[CANONICAL_COLUMNS].copy()

    if expected_day is not None:
        expected_day = pd.Timestamp(expected_day).normalize()
        actual_dates = set(out["date"].dropna().dt.normalize().unique())

        if actual_dates != {expected_day}:
            raise ValueError(
                f"{csv_path}: expected trading date {expected_day.date()}, "
                f"found {sorted(str(x) for x in actual_dates)}"
            )

    return out


def validate_bhavcopy(
    df: pd.DataFrame,
    *,
    strict_market_rows: bool = False,
    require_series: bool = False,
) -> dict:
    """
    Validate a normalized bhavcopy.

    Raw NSE bhavcopies contain more than the EQ research universe. A few
    non-standard rows can legitimately have blank series or unusual OHLC
    relationships. Those are reported as diagnostics but do not abort a raw
    import.

    strict_market_rows=True is used after filtering to the actual research
    universe, where OHLC consistency is required.
    """
    report = {
        "rows": int(len(df)),
        "dates": int(df["date"].nunique()),
        "unique_security_keys": int(df["security_key"].nunique()),
        "unique_market_row_keys": int(df["market_row_key"].nunique()),
        "exact_isin_rows": int(df["isin"].notna().sum()),
        "fallback_identity_rows": int(df["isin"].isna().sum()),
        "missing_date": int(df["date"].isna().sum()),
        "missing_symbol": int(df["symbol"].isna().sum()),
        "missing_series": int(df["series"].isna().sum()),
        "duplicate_date_market_row_key": int(
            df.duplicated(["date", "market_row_key"]).sum()
        ),
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "negative_volume": int((df["volume"] < 0).sum()),
        "negative_turnover": int((df["turnover"] < 0).sum()),
        "high_below_open": int((df["high"] < df["open"]).sum()),
        "high_below_close": int((df["high"] < df["close"]).sum()),
        "high_below_low": int((df["high"] < df["low"]).sum()),
        "low_above_open": int((df["low"] > df["open"]).sum()),
        "low_above_close": int((df["low"] > df["close"]).sum()),
    }

    for col in ("open", "high", "low", "close", "prev_close", "volume"):
        report[f"missing_{col}"] = int(df[col].isna().sum())

    # These indicate parser/data corruption regardless of market series.
    critical_fields = [
        "missing_date",
        "missing_symbol",
        "duplicate_date_market_row_key",
        "nonpositive_close",
        "negative_volume",
        "negative_turnover",
        "missing_open",
        "missing_high",
        "missing_low",
        "missing_close",
        "missing_prev_close",
        "missing_volume",
    ]

    if require_series:
        critical_fields.append("missing_series")

    # Enforce strict OHLC geometry only on the researchable market view.
    if strict_market_rows:
        critical_fields.extend([
            "high_below_open",
            "high_below_close",
            "high_below_low",
            "low_above_open",
            "low_above_close",
        ])

    report["critical_error_count"] = int(
        sum(report[k] for k in critical_fields)
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download and parse one NSE cash-market bhavcopy using either "
            "the legacy or UDiFF schema."
        )
    )
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--raw-root",
        default="data/raw",
        help="Raw data root (default: data/raw)",
    )
    ap.add_argument(
        "--series",
        default=None,
        help="Optional series filter, e.g. EQ",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional Parquet output path",
    )
    args = ap.parse_args()

    day = pd.Timestamp(args.date).normalize()
    raw_root = Path(args.raw_root)

    csv_path = fetch_bhavcopy(day, raw_root)
    df = parse_bhavcopy(csv_path, expected_day=day)

    if args.series:
        df = df.loc[df["series"].eq(args.series)].copy()

    report = validate_bhavcopy(
        df,
        strict_market_rows=bool(args.series),
        require_series=bool(args.series),
    )

    print(f"Date:              {day.date()}")
    print(f"Source CSV:        {csv_path}")
    print(f"Format:            {df['source_format'].iloc[0] if len(df) else 'n/a'}")
    print(f"Rows:              {len(df):,}")
    print(f"Series:            {args.series or 'ALL'}")
    print(f"Unique securities: {df['security_key'].nunique():,}")
    print(f"Rows with ISIN:    {df['isin'].notna().sum():,}")
    print(f"Fallback IDs:      {df['isin'].isna().sum():,}")
    print(f"Critical errors:   {report['critical_error_count']:,}")

    if report["critical_error_count"]:
        print("\nValidation report:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        raise SystemExit(2)

    sample_cols = [
        "date",
        "symbol",
        "series",
        "isin",
        "security_key",
        "market_row_key",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
    ]

    print("\nSample:")
    print(df[sample_cols].head(15).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="zstd")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
