from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


NSE_HOME = "https://www.nseindia.com/"
CORPORATE_ACTIONS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
)

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": (
        "https://www.nseindia.com/"
        "companies-listing/corporate-filings-actions"
    ),
    "Connection": "keep-alive",
}

BONUS_RE = re.compile(
    r"\bbonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

SPLIT_RE = re.compile(
    r"\bfrom\s+(?:rs|re)\.?\s*(\d+(?:\.\d+)?)"
    r"\s*/?-?\s*per\s+share\s+to\s+"
    r"(?:rs|re)\.?\s*(\d+(?:\.\d+)?)"
    r"\s*/?-?\s*per\s+share\b",
    re.IGNORECASE,
)

DIVIDEND_PER_SHARE_RE = re.compile(
    r"(?:rs|re)\.?\s*(\d+(?:\.\d+)?)\s*per\s+share",
    re.IGNORECASE,
)


NORMALIZED_COLUMNS = [
    "symbol",
    "company_name",
    "series",
    "purpose",
    "action_type",
    "face_value",
    "ex_date",
    "record_date",
    "book_closure_start",
    "book_closure_end",
    "payment_date",
    "remarks",
    "share_multiplier",
    "price_multiplier",
    "volume_multiplier",
    "cash_dividend_per_share",
    "auto_adjustable",
    "parse_status",
    "source",
]


def _norm_text(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text in {"-", "--", "None", "nan", "NaN"}:
        return None

    return text


def _parse_date(value: object) -> pd.Timestamp:
    text = _norm_text(value)

    if text is None:
        return pd.NaT

    for fmt in (
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ):
        parsed = pd.to_datetime(
            text,
            format=fmt,
            errors="coerce",
        )

        if not pd.isna(parsed):
            return pd.Timestamp(parsed).normalize()

    return pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True,
    ).normalize()


def classify_action(purpose: str | None) -> str:
    text = (purpose or "").lower()

    if "bonus" in text:
        return "bonus"

    if (
        "split" in text
        or "sub-division" in text
        or "subdivision" in text
        or "consolidation" in text
    ):
        return "split_or_consolidation"

    if "right" in text:
        return "rights"

    if "demerger" in text:
        return "demerger"

    if (
        "merger" in text
        or "amalgamation" in text
        or "scheme of arrangement" in text
    ):
        return "merger_or_scheme"

    if "dividend" in text:
        return "dividend"

    if "buyback" in text or "buy back" in text:
        return "buyback"

    if "interest" in text:
        return "interest"

    if (
        "annual general meeting" in text
        or "extra ordinary general meeting" in text
        or "extraordinary general meeting" in text
        or text.strip() in {"agm", "egm"}
    ):
        return "general_meeting"

    return "other"


def parse_share_multiplier(
    action_type: str,
    purpose: str | None,
) -> tuple[float | None, str]:
    """
    Return post-action shares per pre-action share.

    Examples:
      Bonus 1:1 -> 2.0
      Bonus 1:2 -> 1.5
      Split Rs 10 -> Rs 2 -> 5.0
      Consolidation Rs 1 -> Rs 10 -> 0.1

    Only bonus and face-value split/consolidation events are considered
    mechanically safe enough for automatic price/volume adjustment.
    """
    text = purpose or ""

    if action_type == "bonus":
        match = BONUS_RE.search(text)

        if match is None:
            return None, "bonus_unparsed"

        new_shares = float(match.group(1))
        existing_shares = float(match.group(2))

        if existing_shares <= 0:
            return None, "bonus_invalid_ratio"

        return (
            (new_shares + existing_shares) / existing_shares,
            "bonus_parsed",
        )

    if action_type == "split_or_consolidation":
        match = SPLIT_RE.search(text)

        if match is None:
            return None, "split_unparsed"

        old_face_value = float(match.group(1))
        new_face_value = float(match.group(2))

        if old_face_value <= 0 or new_face_value <= 0:
            return None, "split_invalid_face_value"

        return (
            old_face_value / new_face_value,
            "split_parsed",
        )

    return None, "not_share_count_action"


def parse_cash_dividend_per_share(
    action_type: str,
    purpose: str | None,
) -> float | None:
    """
    Extract explicit cash-dividend amounts.

    NSE occasionally combines multiple dividends in one PURPOSE field, e.g.
    interim + special dividend. We sum all "... Rs X Per Share" amounts.

    This value is stored for later portfolio cash-flow modelling. It is NOT
    used to mechanically adjust OHLC prices.
    """
    if action_type != "dividend":
        return None

    amounts = [
        float(x)
        for x in DIVIDEND_PER_SHARE_RE.findall(purpose or "")
    ]

    if not amounts:
        return None

    return float(sum(amounts))


def normalize_record(row: dict) -> dict:
    symbol = _norm_text(
        row.get("symbol")
        or row.get("SYMBOL")
    )
    company = _norm_text(
        row.get("comp")
        or row.get("companyName")
        or row.get("COMPANY NAME")
    )
    series = _norm_text(
        row.get("series")
        or row.get("SERIES")
    )
    purpose = _norm_text(
        row.get("subject")
        or row.get("purpose")
        or row.get("PURPOSE")
    )

    action_type = classify_action(purpose)
    share_multiplier, parse_status = parse_share_multiplier(
        action_type,
        purpose,
    )

    if share_multiplier is not None and share_multiplier > 0:
        price_multiplier = 1.0 / share_multiplier
        volume_multiplier = share_multiplier
        auto_adjustable = True
    else:
        price_multiplier = None
        volume_multiplier = None
        auto_adjustable = False

    face_value = pd.to_numeric(
        row.get("faceVal")
        or row.get("faceValue")
        or row.get("FACE VALUE"),
        errors="coerce",
    )
    face_value = (
        None
        if pd.isna(face_value)
        else float(face_value)
    )

    return {
        "symbol": symbol,
        "company_name": company,
        "series": series,
        "purpose": purpose,
        "action_type": action_type,
        "face_value": face_value,
        "ex_date": _parse_date(
            row.get("exDate")
            or row.get("EX-DATE")
        ),
        "record_date": _parse_date(
            row.get("recDate")
            or row.get("recordDate")
            or row.get("RECORD DATE")
        ),
        "book_closure_start": _parse_date(
            row.get("bcStartDate")
            or row.get("BOOK CLOSURE START DATE")
        ),
        "book_closure_end": _parse_date(
            row.get("bcEndDate")
            or row.get("BOOK CLOSURE END DATE")
        ),
        "payment_date": _parse_date(
            row.get("payDate")
            or row.get("paymentDate")
        ),
        "remarks": _norm_text(
            row.get("remarks")
        ),
        "share_multiplier": share_multiplier,
        "price_multiplier": price_multiplier,
        "volume_multiplier": volume_multiplier,
        "cash_dividend_per_share": parse_cash_dividend_per_share(
            action_type,
            purpose,
        ),
        "auto_adjustable": auto_adjustable,
        "parse_status": parse_status,
        "source": "NSE_corporates-corporateActions",
    }


def month_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if months < 1:
        raise ValueError("months must be >= 1")

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    while cursor <= end:
        next_start = cursor + pd.DateOffset(months=months)
        chunk_end = min(
            end,
            next_start - pd.Timedelta(days=1),
        )

        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)

    return chunks


class NSECorporateActionsClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 4,
        pause_seconds: float = 0.35,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.pause_seconds = pause_seconds
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def warm(self) -> None:
        response = self.session.get(
            NSE_HOME,
            timeout=self.timeout,
        )
        response.raise_for_status()

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict]:
        params = {
            "index": "equities",
            "from_date": pd.Timestamp(start).strftime("%d-%m-%Y"),
            "to_date": pd.Timestamp(end).strftime("%d-%m-%Y"),
        }

        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                if attempt == 1 or attempt > 2:
                    self.warm()

                response = self.session.get(
                    CORPORATE_ACTIONS_URL,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code in {401, 403, 429}:
                    raise RuntimeError(
                        f"NSE returned HTTP {response.status_code}"
                    )

                response.raise_for_status()
                payload = response.json()

                if isinstance(payload, list):
                    rows = payload
                elif isinstance(payload, dict):
                    rows = (
                        payload.get("data")
                        or payload.get("records")
                        or payload.get("result")
                    )

                    if rows is None:
                        raise RuntimeError(
                            "Unexpected NSE corporate-actions JSON object "
                            f"keys: {sorted(payload)}"
                        )
                else:
                    raise RuntimeError(
                        "Unexpected NSE corporate-actions JSON type: "
                        f"{type(payload).__name__}"
                    )

                if not isinstance(rows, list):
                    raise RuntimeError(
                        "NSE corporate-actions payload is not a list."
                    )

                if not all(isinstance(x, dict) for x in rows):
                    raise RuntimeError(
                        "NSE corporate-actions payload contains non-object rows."
                    )

                time.sleep(self.pause_seconds)
                return rows

            except Exception as exc:
                last_error = exc

                if attempt >= self.retries:
                    break

                time.sleep(
                    min(
                        8.0,
                        self.pause_seconds
                        * (2 ** attempt),
                    )
                )

        raise RuntimeError(
            "Failed to fetch NSE corporate actions for "
            f"{start.date()} -> {end.date()}: {last_error}"
        )


def load_or_fetch_chunk(
    client: NSECorporateActionsClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    raw_root: Path,
    *,
    refresh: bool,
) -> list[dict]:
    raw_root.mkdir(parents=True, exist_ok=True)

    name = (
        f"corporate_actions_"
        f"{start:%Y%m%d}_{end:%Y%m%d}.json"
    )
    path = raw_root / name

    if path.is_file() and not refresh:
        with path.open() as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return payload

        raise RuntimeError(
            f"Cached corporate-action file is not a JSON list: {path}"
        )

    rows = client.fetch(start, end)

    with path.open("w") as f:
        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return rows


def build_corporate_actions(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path,
    chunk_months: int = 12,
    refresh: bool = False,
) -> dict:
    root = Path(root).resolve()
    raw_root = root / "data/raw/corporate_actions"
    output_root = root / "data/processed/corporate_actions"
    reports_root = root / "reports"

    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    client = NSECorporateActionsClient()

    chunks = month_chunks(
        pd.Timestamp(start),
        pd.Timestamp(end),
        chunk_months,
    )

    normalized_frames: list[pd.DataFrame] = []

    for index, (chunk_start, chunk_end) in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(chunks):02d}] "
            f"{chunk_start.date()} -> {chunk_end.date()} ",
            end="",
            flush=True,
        )

        rows = load_or_fetch_chunk(
            client,
            chunk_start,
            chunk_end,
            raw_root,
            refresh=refresh,
        )

        normalized = [
            normalize_record(row)
            for row in rows
        ]

        if normalized:
            frame = pd.DataFrame(normalized)
            normalized_frames.append(frame)

        print(f"rows={len(rows):,}")

    if normalized_frames:
        actions = pd.concat(
            normalized_frames,
            ignore_index=True,
        )
    else:
        actions = pd.DataFrame(
            columns=NORMALIZED_COLUMNS
        )

    for col in NORMALIZED_COLUMNS:
        if col not in actions:
            actions[col] = pd.NA

    actions = actions[NORMALIZED_COLUMNS].copy()

    for col in (
        "symbol",
        "company_name",
        "series",
        "purpose",
        "action_type",
        "remarks",
        "parse_status",
        "source",
    ):
        actions[col] = (
            actions[col]
            .astype("string")
            .str.strip()
        )

    for col in (
        "ex_date",
        "record_date",
        "book_closure_start",
        "book_closure_end",
        "payment_date",
    ):
        actions[col] = pd.to_datetime(
            actions[col],
            errors="coerce",
        ).dt.normalize()

    actions = actions.loc[
        actions["ex_date"].notna()
    ].copy()

    actions = actions.loc[
        actions["ex_date"].between(
            pd.Timestamp(start),
            pd.Timestamp(end),
            inclusive="both",
        )
    ].copy()

    # Corporate actions can appear more than once in adjacent API windows or
    # repeated exchange records. Keep distinct PURPOSE entries but remove exact
    # duplicates.
    dedupe_cols = [
        "symbol",
        "series",
        "purpose",
        "ex_date",
        "record_date",
    ]
    actions = (
        actions
        .drop_duplicates(dedupe_cols, keep="last")
        .sort_values(
            [
                "ex_date",
                "symbol",
                "series",
                "purpose",
            ]
        )
        .reset_index(drop=True)
    )

    parquet_path = (
        output_root
        / "nse_corporate_actions.parquet"
    )
    csv_path = (
        output_root
        / "nse_corporate_actions.csv"
    )

    actions.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )
    actions.to_csv(
        csv_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    counts = {
        str(k): int(v)
        for k, v in (
            actions["action_type"]
            .value_counts(dropna=False)
            .items()
        )
    }

    parse_counts = {
        str(k): int(v)
        for k, v in (
            actions["parse_status"]
            .value_counts(dropna=False)
            .items()
        )
    }

    share_count_actions = actions.loc[
        actions["action_type"].isin(
            ["bonus", "split_or_consolidation"]
        )
    ]

    unparsed_share_count = share_count_actions.loc[
        ~share_count_actions["auto_adjustable"]
    ]

    summary = {
        "requested_from": str(
            pd.Timestamp(start).date()
        ),
        "requested_to": str(
            pd.Timestamp(end).date()
        ),
        "rows": int(len(actions)),
        "symbols": int(
            actions["symbol"].nunique()
        ),
        "ex_date_min": (
            str(actions["ex_date"].min().date())
            if len(actions)
            else None
        ),
        "ex_date_max": (
            str(actions["ex_date"].max().date())
            if len(actions)
            else None
        ),
        "action_type_counts": counts,
        "parse_status_counts": parse_counts,
        "share_count_actions": int(
            len(share_count_actions)
        ),
        "auto_adjustable_share_count_actions": int(
            share_count_actions["auto_adjustable"].sum()
        ),
        "unparsed_share_count_actions": int(
            len(unparsed_share_count)
        ),
        "cash_dividend_rows_with_amount": int(
            actions["cash_dividend_per_share"]
            .notna()
            .sum()
        ),
        "outputs": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "raw_cache": str(raw_root),
        },
    }

    if len(unparsed_share_count):
        failures_path = (
            reports_root
            / "corporate_actions_unparsed_share_count.csv"
        )
        unparsed_share_count.to_csv(
            failures_path,
            index=False,
            date_format="%Y-%m-%d",
        )
        summary["unparsed_share_count_path"] = str(
            failures_path
        )

    summary_path = (
        reports_root
        / "corporate_actions_summary.json"
    )

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    return {
        "actions": actions,
        "summary": summary,
        "summary_path": summary_path,
        "parquet_path": parquet_path,
        "csv_path": csv_path,
    }


def self_test() -> None:
    samples = [
        (
            "Bonus 1:1",
            "bonus",
            2.0,
        ),
        (
            "Bonus 1:2",
            "bonus",
            1.5,
        ),
        (
            (
                "Face Value Split (Sub-Division) - "
                "From Rs 10/- Per Share To Rs 2/- Per Share"
            ),
            "split_or_consolidation",
            5.0,
        ),
        (
            (
                "Face Value Split (Sub-Division) - "
                "From Rs 10/- Per Share To Re 1/- Per Share"
            ),
            "split_or_consolidation",
            10.0,
        ),
    ]

    for purpose, expected_type, expected_multiplier in samples:
        action_type = classify_action(purpose)
        multiplier, status = parse_share_multiplier(
            action_type,
            purpose,
        )

        assert action_type == expected_type
        assert multiplier is not None
        assert math.isclose(
            multiplier,
            expected_multiplier,
            rel_tol=1e-12,
        )
        assert status.endswith("_parsed")

    dividend = (
        "Interim Dividend Rs 11 Per Share/ "
        "Special Dividend Rs 46 Per Share"
    )
    amount = parse_cash_dividend_per_share(
        "dividend",
        dividend,
    )

    assert amount is not None
    assert math.isclose(amount, 57.0)

    print("Corporate-action parser self-test: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download, normalize and classify NSE equity corporate actions. "
            "No market prices are modified by this command."
        )
    )
    ap.add_argument(
        "--from",
        dest="start",
        default="2010-01-01",
        help="Start ex-date, YYYY-MM-DD (default: 2010-01-01)",
    )
    ap.add_argument(
        "--to",
        dest="end",
        default=str(date.today()),
        help="End ex-date, YYYY-MM-DD (default: today)",
    )
    ap.add_argument(
        "--root",
        default=".",
        help="Repository/data root",
    )
    ap.add_argument(
        "--chunk-months",
        type=int,
        default=12,
        help="NSE API request window in months (default: 12)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached raw JSON and refetch all requested windows",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser tests and exit",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    if end < start:
        raise SystemExit("--to must be >= --from")

    result = build_corporate_actions(
        start,
        end,
        root=Path(args.root),
        chunk_months=args.chunk_months,
        refresh=args.refresh,
    )

    summary = result["summary"]

    print("\n=== CORPORATE ACTION INGEST COMPLETE ===")
    print(f"Actions:              {summary['rows']:,}")
    print(f"Symbols:              {summary['symbols']:,}")
    print(
        "Share-count actions:  "
        f"{summary['share_count_actions']:,}"
    )
    print(
        "Auto-adjustable:       "
        f"{summary['auto_adjustable_share_count_actions']:,}"
    )
    print(
        "Unparsed split/bonus:  "
        f"{summary['unparsed_share_count_actions']:,}"
    )
    print(
        "Dividend amounts:      "
        f"{summary['cash_dividend_rows_with_amount']:,}"
    )
    print(f"Action types:         {summary['action_type_counts']}")
    print(f"Parquet:              {result['parquet_path']}")
    print(f"CSV:                  {result['csv_path']}")
    print(f"Summary:              {result['summary_path']}")


if __name__ == "__main__":
    main()
