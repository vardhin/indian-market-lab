from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


NSE_HOME = "https://www.nseindia.com/"
NSE_INDEX_REPORT = (
    "https://www.nseindia.com/"
    "reports-indices-historical-index-data"
)
NSE_INDEX_HISTORY = (
    "https://www.nseindia.com/"
    "api/historicalOR/indicesHistory"
)

DEFAULT_INDICES = [
    "NIFTY 50",
    "NIFTY 100",
    "NIFTY 500",
]

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_INDEX_REPORT,
    "Connection": "keep-alive",
}


def _norm_text(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if (
        not text
        or text in {
            "-",
            "--",
            "None",
            "nan",
            "NaN",
        }
    ):
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
        "%d/%m/%y",
    ):
        parsed = pd.to_datetime(
            text,
            format=fmt,
            errors="coerce",
        )

        if not pd.isna(parsed):
            return pd.Timestamp(
                parsed
            ).normalize()

    return pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True,
    ).normalize()


def _num(value: object) -> float | None:
    if value is None:
        return None

    parsed = pd.to_numeric(
        str(value).replace(
            ",",
            "",
        ),
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return float(parsed)


def chunk_ranges(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    days: int,
) -> list[
    tuple[
        pd.Timestamp,
        pd.Timestamp,
    ]
]:
    if days < 1:
        raise ValueError(
            "days must be >= 1"
        )

    start = pd.Timestamp(
        start
    ).normalize()
    end = pd.Timestamp(
        end
    ).normalize()

    chunks: list[
        tuple[
            pd.Timestamp,
            pd.Timestamp,
        ]
    ] = []

    cursor = start

    while cursor <= end:
        chunk_end = min(
            end,
            cursor
            + pd.Timedelta(
                days=days - 1
            ),
        )

        chunks.append(
            (
                cursor,
                chunk_end,
            )
        )
        cursor = (
            chunk_end
            + pd.Timedelta(days=1)
        )

    return chunks


class NSEIndexHistoryClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 4,
        pause_seconds: float = 0.35,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.pause_seconds = (
            pause_seconds
        )

        self.session = (
            requests.Session()
        )
        self.session.headers.update(
            HEADERS
        )

    def warm(self) -> None:
        for url in (
            NSE_HOME,
            NSE_INDEX_REPORT,
        ):
            response = (
                self.session.get(
                    url,
                    timeout=self.timeout,
                )
            )
            response.raise_for_status()

    @staticmethod
    def _extract_rows(
        payload: object,
    ) -> list[dict]:
        if isinstance(
            payload,
            list,
        ):
            rows = payload

        elif isinstance(
            payload,
            dict,
        ):
            data = payload.get(
                "data"
            )

            if isinstance(
                data,
                list,
            ):
                rows = data
            elif isinstance(
                data,
                dict,
            ):
                for key in (
                    "indexCloseOnlineRecords",
                    "records",
                    "data",
                ):
                    candidate = (
                        data.get(key)
                    )

                    if isinstance(
                        candidate,
                        list,
                    ):
                        rows = candidate
                        break
                else:
                    raise RuntimeError(
                        "Unexpected NSE index-history "
                        "data object keys: "
                        f"{sorted(data)}"
                    )
            else:
                for key in (
                    "records",
                    "result",
                ):
                    candidate = (
                        payload.get(key)
                    )

                    if isinstance(
                        candidate,
                        list,
                    ):
                        rows = candidate
                        break
                else:
                    raise RuntimeError(
                        "Unexpected NSE index-history "
                        "JSON keys: "
                        f"{sorted(payload)}"
                    )
        else:
            raise RuntimeError(
                "Unexpected NSE index-history "
                f"JSON type: "
                f"{type(payload).__name__}"
            )

        if not all(
            isinstance(
                row,
                dict,
            )
            for row in rows
        ):
            raise RuntimeError(
                "NSE index-history payload "
                "contains non-object rows."
            )

        return rows

    def fetch(
        self,
        index_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict]:
        params = {
            "indexType": index_name,
            "from": (
                pd.Timestamp(start)
                .strftime("%d-%m-%Y")
            ),
            "to": (
                pd.Timestamp(end)
                .strftime("%d-%m-%Y")
            ),
        }

        last_error: (
            Exception | None
        ) = None

        for attempt in range(
            1,
            self.retries + 1,
        ):
            try:
                if (
                    attempt == 1
                    or attempt > 2
                ):
                    self.warm()

                response = (
                    self.session.get(
                        NSE_INDEX_HISTORY,
                        params=params,
                        timeout=(
                            self.timeout
                        ),
                    )
                )

                if (
                    response.status_code
                    in {
                        401,
                        403,
                        429,
                    }
                ):
                    raise RuntimeError(
                        "NSE returned HTTP "
                        f"{response.status_code}"
                    )

                response.raise_for_status()

                content_type = (
                    response.headers.get(
                        "content-type",
                        "",
                    )
                ).lower()

                if (
                    "json"
                    not in content_type
                ):
                    prefix = (
                        response.text[:120]
                        .replace(
                            "\n",
                            " ",
                        )
                    )

                    raise RuntimeError(
                        "NSE index-history "
                        "response was not JSON: "
                        f"{content_type!r}, "
                        f"prefix={prefix!r}"
                    )

                rows = (
                    self._extract_rows(
                        response.json()
                    )
                )

                time.sleep(
                    self.pause_seconds
                )
                return rows

            except Exception as exc:
                last_error = exc

                if (
                    attempt
                    >= self.retries
                ):
                    break

                time.sleep(
                    min(
                        8.0,
                        self.pause_seconds
                        * (
                            2 ** attempt
                        ),
                    )
                )

        raise RuntimeError(
            "Failed to fetch NSE "
            f"index history for "
            f"{index_name}: "
            f"{start.date()} -> "
            f"{end.date()}: "
            f"{last_error}"
        )


def normalize_record(
    row: dict,
    requested_index: str,
) -> dict:
    return {
        "requested_index": (
            requested_index
        ),
        "index_name": (
            _norm_text(
                row.get(
                    "EOD_INDEX_NAME"
                )
                or row.get(
                    "INDEX_NAME"
                )
                or row.get(
                    "indexName"
                )
            )
            or requested_index
        ),
        "date": _parse_date(
            row.get(
                "EOD_TIMESTAMP"
            )
            or row.get("Date")
            or row.get("date")
            or row.get(
                "TIMESTAMP"
            )
        ),
        "open": _num(
            row.get(
                "EOD_OPEN_INDEX_VAL"
            )
            or row.get("OPEN")
            or row.get("open")
        ),
        "high": _num(
            row.get(
                "EOD_HIGH_INDEX_VAL"
            )
            or row.get("HIGH")
            or row.get("high")
        ),
        "low": _num(
            row.get(
                "EOD_LOW_INDEX_VAL"
            )
            or row.get("LOW")
            or row.get("low")
        ),
        "close": _num(
            row.get(
                "EOD_CLOSE_INDEX_VAL"
            )
            or row.get("CLOSE")
            or row.get("close")
        ),
        "shares_traded": _num(
            row.get(
                "EOD_TRADED_QTY"
            )
            or row.get(
                "SHARES_TRADED"
            )
            or row.get(
                "sharesTraded"
            )
        ),
        "turnover_crore": _num(
            row.get(
                "EOD_TURN_OVER"
            )
            or row.get(
                "TURNOVER"
            )
            or row.get(
                "turnover"
            )
        ),
        "source": (
            "NSE_historicalOR_indicesHistory"
        ),
    }


def cache_path(
    raw_root: Path,
    index_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    safe_index = (
        index_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    return (
        raw_root
        / safe_index
        / (
            f"{start:%Y%m%d}_"
            f"{end:%Y%m%d}.json"
        )
    )


def load_or_fetch(
    client: NSEIndexHistoryClient,
    index_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    raw_root: Path,
    *,
    refresh: bool,
) -> list[dict]:
    path = cache_path(
        raw_root,
        index_name,
        start,
        end,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        path.is_file()
        and not refresh
    ):
        payload = json.loads(
            path.read_text()
        )

        if not isinstance(
            payload,
            list,
        ):
            raise RuntimeError(
                "Cached index-history "
                "payload is not a list: "
                f"{path}"
            )

        return payload

    rows = client.fetch(
        index_name,
        start,
        end,
    )

    path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    return rows


def validate_index(
    frame: pd.DataFrame,
    *,
    requested_index: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    if frame.empty:
        raise RuntimeError(
            f"No rows returned for "
            f"{requested_index}"
        )

    frame = frame.sort_values(
        "date"
    )

    duplicate_dates = int(
        frame.duplicated(
            "date",
            keep=False,
        ).sum()
    )

    if duplicate_dates:
        raise RuntimeError(
            f"{requested_index} has "
            f"{duplicate_dates:,} "
            "duplicate-date rows."
        )

    bad_ohlc = (
        frame["close"].le(0)
        | frame["open"].le(0)
        | frame["high"].lt(
            frame[
                [
                    "open",
                    "close",
                    "low",
                ]
            ].max(axis=1)
        )
        | frame["low"].gt(
            frame[
                [
                    "open",
                    "close",
                    "high",
                ]
            ].min(axis=1)
        )
    )

    if bad_ohlc.any():
        sample = frame.loc[
            bad_ohlc,
            [
                "date",
                "open",
                "high",
                "low",
                "close",
            ],
        ].head(20)

        raise RuntimeError(
            f"{requested_index}: invalid "
            "OHLC rows. Sample:\n"
            + sample.to_string(
                index=False
            )
        )

    expected_end = min(
        pd.Timestamp(end).normalize(),
        pd.Timestamp.today().normalize(),
    )
    end_gap_days = int(
        (
            expected_end
            - frame["date"].max()
        ).days
    )

    if end_gap_days > 10:
        raise RuntimeError(
            f"{requested_index}: benchmark history stops at "
            f"{frame['date'].max().date()}, which is {end_gap_days} days "
            f"before expected end {expected_end.date()}. "
            "This usually indicates a truncated NSE API response."
        )

    weekday_count = len(
        pd.bdate_range(
            max(
                pd.Timestamp(start),
                frame["date"].min(),
            ),
            expected_end,
        )
    )
    weekday_coverage = (
        len(frame) / weekday_count
        if weekday_count
        else 1.0
    )

    if weekday_coverage < 0.75:
        raise RuntimeError(
            f"{requested_index}: only {len(frame):,} rows for roughly "
            f"{weekday_count:,} weekday sessions "
            f"({weekday_coverage:.1%} coverage). "
            "Benchmark history appears incomplete."
        )

    return {
        "requested_index": (
            requested_index
        ),
        "rows": int(
            len(frame)
        ),
        "weekday_coverage": float(
            weekday_coverage
        ),
        "end_gap_days": int(
            end_gap_days
        ),
        "date_min": str(
            frame["date"]
            .min()
            .date()
        ),
        "date_max": str(
            frame["date"]
            .max()
            .date()
        ),
        "requested_from": str(
            pd.Timestamp(
                start
            ).date()
        ),
        "requested_to": str(
            pd.Timestamp(
                end
            ).date()
        ),
        "first_close": float(
            frame["close"]
            .iloc[0]
        ),
        "last_close": float(
            frame["close"]
            .iloc[-1]
        ),
    }


def build_indices(
    *,
    root: Path,
    indices: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 60,
    refresh: bool = False,
) -> dict:
    root = Path(root).resolve()

    raw_root = (
        root
        / "data/raw/index_benchmarks"
    )
    output_root = (
        root
        / "data/processed/index_benchmarks"
    )
    reports_root = (
        root
        / "reports"
    )

    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = chunk_ranges(
        start,
        end,
        days=chunk_days,
    )

    client = (
        NSEIndexHistoryClient()
    )

    all_frames: list[
        pd.DataFrame
    ] = []
    summaries: dict[
        str,
        dict,
    ] = {}

    for index_name in indices:
        normalized_rows: list[
            dict
        ] = []

        print(
            f"\n=== {index_name} ==="
        )

        for number, (
            chunk_start,
            chunk_end,
        ) in enumerate(
            chunks,
            start=1,
        ):
            print(
                f"[{number:02d}/"
                f"{len(chunks):02d}] "
                f"{chunk_start.date()} "
                f"-> {chunk_end.date()} ",
                end="",
                flush=True,
            )

            rows = load_or_fetch(
                client,
                index_name,
                chunk_start,
                chunk_end,
                raw_root,
                refresh=refresh,
            )

            requested_span_days = (
                chunk_end
                - chunk_start
            ).days + 1

            # NSE's endpoint can silently truncate large date windows rather
            # than returning every requested trading day. The observed failure
            # mode is exactly 70 rows from year-long requests. Never accept
            # that shape silently.
            if (
                len(rows) >= 70
                and requested_span_days > 120
            ):
                raise RuntimeError(
                    "Suspiciously capped NSE index-history response: "
                    f"{len(rows)} rows for a {requested_span_days}-day "
                    f"window ({chunk_start.date()} -> {chunk_end.date()}). "
                    "Use --chunk-days 60 (default) or smaller."
                )

            normalized_rows.extend(
                normalize_record(
                    row,
                    index_name,
                )
                for row in rows
            )

            print(
                f"rows={len(rows):,}"
            )

        frame = pd.DataFrame(
            normalized_rows
        )

        if frame.empty:
            raise RuntimeError(
                f"NSE returned no data "
                f"for {index_name}."
            )

        frame["date"] = (
            pd.to_datetime(
                frame["date"],
                errors="coerce",
            ).dt.normalize()
        )

        for col in (
            "open",
            "high",
            "low",
            "close",
            "shares_traded",
            "turnover_crore",
        ):
            frame[col] = (
                pd.to_numeric(
                    frame[col],
                    errors="coerce",
                )
            )

        frame = (
            frame.loc[
                frame["date"].notna()
                & frame["date"].between(
                    start,
                    end,
                    inclusive="both",
                )
            ]
            .drop_duplicates(
                [
                    "requested_index",
                    "date",
                ],
                keep="last",
            )
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        summary = (
            validate_index(
                frame,
                requested_index=(
                    index_name
                ),
                start=start,
                end=end,
            )
        )

        summaries[
            index_name
        ] = summary

        safe_index = (
            index_name.lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
        )

        parquet_path = (
            output_root
            / f"{safe_index}.parquet"
        )
        csv_path = (
            output_root
            / f"{safe_index}.csv"
        )

        frame.to_parquet(
            parquet_path,
            index=False,
            compression="zstd",
        )
        frame.to_csv(
            csv_path,
            index=False,
            date_format="%Y-%m-%d",
        )

        all_frames.append(
            frame
        )

        print(
            "  normalized="
            f"{len(frame):,}, "
            f"{summary['date_min']} "
            f"-> {summary['date_max']}"
        )

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    combined_path = (
        output_root
        / "nse_price_indices.parquet"
    )
    combined_csv = (
        output_root
        / "nse_price_indices.csv"
    )

    combined.to_parquet(
        combined_path,
        index=False,
        compression="zstd",
    )
    combined.to_csv(
        combined_csv,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary = {
        "index_return_type": (
            "price_index"
        ),
        "dividends_included": False,
        "indices": summaries,
        "combined_parquet": str(
            combined_path
        ),
        "combined_csv": str(
            combined_csv
        ),
        "raw_cache": str(
            raw_root
        ),
    }

    summary_path = (
        reports_root
        / "index_benchmarks_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return {
        "summary": summary,
        "summary_path": (
            summary_path
        ),
    }


def self_test() -> None:
    sample = {
        "EOD_INDEX_NAME": (
            "NIFTY 500"
        ),
        "EOD_TIMESTAMP": (
            "02-Oct-2026"
        ),
        "EOD_OPEN_INDEX_VAL": (
            "25,100.25"
        ),
        "EOD_HIGH_INDEX_VAL": (
            "25,250.50"
        ),
        "EOD_LOW_INDEX_VAL": (
            "25,010.10"
        ),
        "EOD_CLOSE_INDEX_VAL": (
            "25,200.75"
        ),
        "EOD_TRADED_QTY": (
            "123456789"
        ),
        "EOD_TURN_OVER": (
            "12345.67"
        ),
    }

    row = normalize_record(
        sample,
        "NIFTY 500",
    )

    assert (
        row["index_name"]
        == "NIFTY 500"
    )
    assert (
        row["date"]
        == pd.Timestamp(
            "2026-10-02"
        )
    )
    assert (
        row["close"]
        == 25200.75
    )

    chunks = chunk_ranges(
        pd.Timestamp(
            "2026-01-01"
        ),
        pd.Timestamp(
            "2026-01-10"
        ),
        days=4,
    )

    assert chunks == [
        (
            pd.Timestamp(
                "2026-01-01"
            ),
            pd.Timestamp(
                "2026-01-04"
            ),
        ),
        (
            pd.Timestamp(
                "2026-01-05"
            ),
            pd.Timestamp(
                "2026-01-08"
            ),
        ),
        (
            pd.Timestamp(
                "2026-01-09"
            ),
            pd.Timestamp(
                "2026-01-10"
            ),
        ),
    ]

    assert len(
        chunk_ranges(
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-12-31"),
            days=60,
        )
    ) == 7

    print(
        "Index benchmark ingest "
        "self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download and normalize official NSE "
            "historical price-index data for "
            "market benchmarks."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--indices",
        nargs="+",
        default=DEFAULT_INDICES,
    )
    ap.add_argument(
        "--from",
        dest="start",
        default="2010-01-01",
    )
    ap.add_argument(
        "--to",
        dest="end",
        default="2026-09-30",
    )
    ap.add_argument(
        "--chunk-days",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    start = pd.Timestamp(
        args.start
    ).normalize()
    end = pd.Timestamp(
        args.end
    ).normalize()

    if end < start:
        raise SystemExit(
            "--to must be >= --from"
        )

    result = build_indices(
        root=Path(args.root),
        indices=list(
            args.indices
        ),
        start=start,
        end=end,
        chunk_days=(
            args.chunk_days
        ),
        refresh=args.refresh,
    )

    print(
        "\n=== INDEX BENCHMARK INGEST "
        "COMPLETE ==="
    )

    for index_name, summary in (
        result["summary"][
            "indices"
        ].items()
    ):
        print(
            f"{index_name:12s} "
            f"rows={summary['rows']:,} "
            f"{summary['date_min']} -> "
            f"{summary['date_max']}"
        )

    print(
        "Combined: "
        f"{result['summary']['combined_parquet']}"
    )
    print(
        "Summary:  "
        f"{result['summary_path']}"
    )


if __name__ == "__main__":
    main()
