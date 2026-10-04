from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests


ARCHIVE_URL = (
    "https://www.amfiindia.com/"
    "otherdata/categorisation-of-stocks"
)

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/136 Safari/537.36"
)

HEADERS = {
    "User-Agent": UA,
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# AMFI's archive page has historically exposed a couple of workbook
# anchors inconsistently to non-browser clients.  Keep the exact
# official workbook URLs as discovery fallbacks so a missing half-year
# can never silently stretch the previous PIT classification forward.
KNOWN_OFFICIAL_WORKBOOKS = {
    pd.Timestamp("2020-12-31"): (
        "https://www.amfiindia.com/Themes/Theme1/downloads/"
        "Average%20Market%20Capitalization%20of%20Listed%20Companies%20"
        "during%20Jul%20-%20Dec%202020_Final.xlsx"
    ),
    pd.Timestamp("2024-06-30"): (
        "https://www.amfiindia.com/uploads/"
        "Average_Market_Capitalization_30_Jun2024_2a1ab4c1d8.xlsx"
    ),
}

ISIN_RE = re.compile(
    r"^[A-Z]{2}[A-Z0-9]{10}$"
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[
            tuple[str, str | None]
        ],
    ) -> None:
        if tag.lower() != "a":
            return

        href = dict(
            attrs
        ).get(
            "href"
        )
        if href:
            self.links.append(
                href
            )


def _norm_text(
    value: object,
) -> str | None:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if (
        not text
        or text.lower()
        in {
            "nan",
            "none",
            "nat",
            "-",
            "--",
        }
    ):
        return None

    return text


def _norm_header(
    value: object,
) -> str:
    text = (
        _norm_text(
            value
        )
        or ""
    ).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    text = (
        text.replace(
            "\n",
            " ",
        )
        .replace(
            "\r",
            " ",
        )
    )
    return text.strip()


def _to_num(
    value: object,
) -> float | None:
    text = _norm_text(
        value
    )
    if text is None:
        return None

    text = (
        text.replace(
            ",",
            "",
        )
        .replace(
            "₹",
            "",
        )
    )

    parsed = pd.to_numeric(
        text,
        errors="coerce",
    )
    if pd.isna(
        parsed
    ):
        return None

    return float(
        parsed
    )


def _normalise_isin(
    value: object,
) -> str | None:
    text = _norm_text(
        value
    )
    if text is None:
        return None

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )

    if not ISIN_RE.match(
        text
    ):
        return None

    return text


def _normalise_category(
    value: object,
) -> str | None:
    text = _norm_text(
        value
    )
    if text is None:
        return None

    low = text.lower()

    if "large" in low:
        return "large_cap"
    if "mid" in low:
        return "mid_cap"
    if "small" in low:
        return "small_cap"

    return None


def discover_excel_urls(
    *,
    timeout: float = 30.0,
) -> list[str]:
    response = requests.get(
        ARCHIVE_URL,
        headers=HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(
        response.text
    )

    urls: list[str] = []

    for href in parser.links:
        absolute = urljoin(
            ARCHIVE_URL,
            href,
        )
        path = unquote(
            urlparse(
                absolute
            ).path
        ).lower()

        if path.endswith(
            ".xlsx"
        ):
            urls.append(
                absolute
            )

    return list(
        dict.fromkeys(
            urls
        )
    )


def expected_period_ends(
    first: pd.Timestamp,
    last: pd.Timestamp,
) -> list[pd.Timestamp]:
    first = pd.Timestamp(
        first
    ).normalize()
    last = pd.Timestamp(
        last
    ).normalize()

    if first.month not in {
        6,
        12,
    }:
        raise ValueError(
            "first must be a June/December "
            "half-year endpoint"
        )

    if last.month not in {
        6,
        12,
    }:
        raise ValueError(
            "last must be a June/December "
            "half-year endpoint"
        )

    out: list[pd.Timestamp] = []
    cursor = first

    while cursor <= last:
        out.append(
            cursor
        )

        if cursor.month == 6:
            cursor = pd.Timestamp(
                year=cursor.year,
                month=12,
                day=31,
            )
        else:
            cursor = pd.Timestamp(
                year=cursor.year + 1,
                month=6,
                day=30,
            )

    return out


def validate_period_completeness(
    sources: list[dict],
) -> None:
    if not sources:
        raise RuntimeError(
            "No AMFI sources discovered."
        )

    discovered = sorted({
        pd.Timestamp(
            row[
                "period_end"
            ]
        ).normalize()
        for row in sources
    })

    first = min(
        discovered
    )
    last = max(
        discovered
    )

    expected = set(
        expected_period_ends(
            first,
            last,
        )
    )
    actual = set(
        discovered
    )

    missing = sorted(
        expected
        - actual
    )

    if missing:
        raise RuntimeError(
            "AMFI half-year archive is "
            "incomplete. Missing period ends: "
            + ", ".join(
                str(
                    value.date()
                )
                for value in missing
            )
        )


def infer_period_end(
    url: str,
) -> pd.Timestamp:
    text = unquote(
        url
    )

    patterns = [
        (
            r"30\s*jun(?:e)?\s*(20\d{2})",
            6,
            30,
        ),
        (
            r"31\s*dec(?:ember)?\s*(20\d{2})",
            12,
            31,
        ),
        (
            r"jan[^0-9]{0,20}jun(?:e)?[^0-9]{0,20}(20\d{2})",
            6,
            30,
        ),
        (
            r"jul(?:y)?[^0-9]{0,20}dec(?:ember)?[^0-9]{0,20}(20\d{2})",
            12,
            31,
        ),
    ]

    low = text.lower()

    for (
        pattern,
        month,
        day,
    ) in patterns:
        match = re.search(
            pattern,
            low,
        )
        if match:
            return pd.Timestamp(
                year=int(
                    match.group(
                        1
                    )
                ),
                month=month,
                day=day,
            )

    raise ValueError(
        "Could not infer AMFI six-month "
        f"period from URL: {url}"
    )


def measurement_start(
    period_end: pd.Timestamp,
) -> pd.Timestamp:
    end = pd.Timestamp(
        period_end
    ).normalize()

    if end.month == 6:
        return pd.Timestamp(
            year=end.year,
            month=1,
            day=1,
        )

    if end.month == 12:
        return pd.Timestamp(
            year=end.year,
            month=7,
            day=1,
        )

    raise ValueError(
        f"Unexpected period end: {end}"
    )


def conservative_effective_from(
    period_end: pd.Timestamp,
) -> pd.Timestamp:
    """
    Conservative point-in-time availability.

    We do not assume the completed six-month AMFI file was
    tradably available on the period-end date. The new list
    becomes usable one full calendar month later:
      Jan-Jun  -> Aug 1
      Jul-Dec  -> Feb 1 next year
    """
    end = pd.Timestamp(
        period_end
    ).normalize()

    if end.month == 6:
        return pd.Timestamp(
            year=end.year,
            month=8,
            day=1,
        )

    if end.month == 12:
        return pd.Timestamp(
            year=end.year + 1,
            month=2,
            day=1,
        )

    raise ValueError(
        f"Unexpected period end: {end}"
    )


def cache_name(
    url: str,
    period_end: pd.Timestamp,
) -> str:
    digest = hashlib.sha256(
        url.encode(
            "utf-8"
        )
    ).hexdigest()[
        :10
    ]

    return (
        "amfi_market_cap_"
        f"{period_end:%Y%m%d}_"
        f"{digest}.xlsx"
    )


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float = 60.0,
    retries: int = 4,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    error: Exception | None = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            response = requests.get(
                url,
                headers={
                    **HEADERS,
                    "Referer": (
                        ARCHIVE_URL
                    ),
                },
                timeout=timeout,
            )
            response.raise_for_status()

            if len(
                response.content
            ) < 5000:
                raise RuntimeError(
                    "Downloaded AMFI workbook "
                    "is unexpectedly small."
                )

            destination.write_bytes(
                response.content
            )
            return

        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(
                    1.5 * attempt
                )

    assert error is not None
    raise error


def _find_header_row(
    raw: pd.DataFrame,
) -> int:
    limit = min(
        40,
        len(
            raw
        ),
    )

    for row_index in range(
        limit
    ):
        values = [
            _norm_header(
                value
            )
            for value in raw.iloc[
                row_index
            ].tolist()
        ]

        joined = " | ".join(
            values
        )

        if (
            "isin" in joined
            and "company" in joined
            and (
                "categor" in joined
                or "classification"
                in joined
            )
        ):
            return row_index

    raise RuntimeError(
        "Could not locate AMFI workbook "
        "header row."
    )


def _make_unique_headers(
    values: list[object],
) -> list[str]:
    out: list[str] = []
    seen: dict[
        str,
        int,
    ] = {}

    for i, value in enumerate(
        values
    ):
        base = (
            _norm_header(
                value
            )
            or f"unnamed_{i}"
        )

        count = seen.get(
            base,
            0,
        )
        seen[
            base
        ] = count + 1

        out.append(
            base
            if count == 0
            else (
                f"{base}__{count + 1}"
            )
        )

    return out


def _pick_column(
    columns: list[str],
    predicates: list[
        tuple[str, ...]
    ],
    *,
    exclude: tuple[str, ...] = (),
) -> str | None:
    for required in predicates:
        for column in columns:
            low = column.lower()

            if any(
                bad in low
                for bad in exclude
            ):
                continue

            if all(
                token in low
                for token in required
            ):
                return column

    return None


def parse_workbook(
    path: Path,
    *,
    url: str,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    raw = pd.read_excel(
        path,
        sheet_name=0,
        header=None,
        dtype=object,
        engine="openpyxl",
    )

    header_row = (
        _find_header_row(
            raw
        )
    )

    headers = (
        _make_unique_headers(
            raw.iloc[
                header_row
            ].tolist()
        )
    )

    data = raw.iloc[
        header_row + 1:
    ].copy()
    data.columns = headers

    columns = list(
        data.columns
    )

    rank_col = _pick_column(
        columns,
        [
            (
                "sr",
                "no",
            ),
            (
                "serial",
            ),
            (
                "rank",
            ),
        ],
    )
    company_col = _pick_column(
        columns,
        [
            (
                "company",
                "name",
            ),
            (
                "company",
            ),
        ],
    )
    isin_col = _pick_column(
        columns,
        [
            (
                "isin",
            ),
        ],
    )
    nse_symbol_col = (
        _pick_column(
            columns,
            [
                (
                    "nse",
                    "symbol",
                ),
            ],
        )
    )
    category_col = _pick_column(
        columns,
        [
            (
                "categor",
            ),
            (
                "classification",
            ),
        ],
    )
    average_col = _pick_column(
        columns,
        [
            (
                "average",
                "all",
                "exchange",
            ),
            (
                "average",
                "market",
                "cap",
            ),
            (
                "average",
                "market",
                "capital",
            ),
        ],
        exclude=(
            "bse",
            "nse",
            "msei",
        ),
    )

    required = {
        "rank": rank_col,
        "company": (
            company_col
        ),
        "isin": isin_col,
        "category": (
            category_col
        ),
    }

    missing = [
        name
        for name, column
        in required.items()
        if column is None
    ]

    if missing:
        raise RuntimeError(
            f"{path.name}: unable to map "
            "required AMFI columns: "
            f"{missing}. Headers={columns}"
        )

    rows = pd.DataFrame({
        "amfi_market_cap_rank": (
            data[
                rank_col
            ].map(
                _to_num
            )
        ),
        "company_name": (
            data[
                company_col
            ].map(
                _norm_text
            )
        ),
        "isin": (
            data[
                isin_col
            ].map(
                _normalise_isin
            )
        ),
        "nse_symbol": (
            data[
                nse_symbol_col
            ].map(
                _norm_text
            )
            if nse_symbol_col
            else None
        ),
        "amfi_average_market_cap_cr": (
            data[
                average_col
            ].map(
                _to_num
            )
            if average_col
            else None
        ),
        "amfi_cap_bucket": (
            data[
                category_col
            ].map(
                _normalise_category
            )
        ),
    })

    rows = rows.loc[
        rows[
            "isin"
        ].notna()
        & rows[
            "amfi_market_cap_rank"
        ].notna()
    ].copy()

    rows[
        "amfi_market_cap_rank"
    ] = rows[
        "amfi_market_cap_rank"
    ].round().astype(
        "int32"
    )

    rows[
        "measurement_start"
    ] = measurement_start(
        period_end
    )
    rows[
        "measurement_end"
    ] = pd.Timestamp(
        period_end
    ).normalize()
    rows[
        "effective_from"
    ] = (
        conservative_effective_from(
            period_end
        )
    )
    rows[
        "source_url"
    ] = url
    rows[
        "source_file"
    ] = path.name
    rows[
        "availability_policy"
    ] = (
        "conservative_one_full_month_after_period_end"
    )

    rows[
        "rank_implied_bucket"
    ] = pd.cut(
        rows[
            "amfi_market_cap_rank"
        ],
        bins=[
            0,
            100,
            250,
            float(
                "inf"
            ),
        ],
        labels=[
            "large_cap",
            "mid_cap",
            "small_cap",
        ],
        right=True,
    ).astype(
        "string"
    )

    rows[
        "category_rank_match"
    ] = (
        rows[
            "amfi_cap_bucket"
        ].astype(
            "string"
        )
        == rows[
            "rank_implied_bucket"
        ].astype(
            "string"
        )
    )

    duplicated = rows.duplicated(
        "isin",
        keep=False,
    )

    rows[
        "duplicate_isin_count"
    ] = (
        rows.groupby(
            "isin",
            sort=False,
        )[
            "isin"
        ].transform(
            "size"
        ).astype(
            "int16"
        )
    )

    rows[
        "duplicate_rank_min"
    ] = (
        rows.groupby(
            "isin",
            sort=False,
        )[
            "amfi_market_cap_rank"
        ].transform(
            "min"
        )
    )
    rows[
        "duplicate_rank_max"
    ] = (
        rows.groupby(
            "isin",
            sort=False,
        )[
            "amfi_market_cap_rank"
        ].transform(
            "max"
        )
    )

    category_nunique = (
        rows[
            "amfi_cap_bucket"
        ]
        .fillna(
            "__MISSING__"
        )
        .groupby(
            rows[
                "isin"
            ],
            sort=False,
        )
        .transform(
            "nunique"
        )
    )

    rows[
        "duplicate_category_disagreement"
    ] = (
        rows[
            "duplicate_isin_count"
        ].gt(
            1
        )
        & category_nunique.gt(
            1
        )
    )

    market_cap_nunique = (
        rows[
            "amfi_average_market_cap_cr"
        ]
        .round(
            8
        )
        .fillna(
            float(
                "-inf"
            )
        )
        .groupby(
            rows[
                "isin"
            ],
            sort=False,
        )
        .transform(
            "nunique"
        )
    )

    rows[
        "duplicate_market_cap_disagreement"
    ] = (
        rows[
            "duplicate_isin_count"
        ].gt(
            1
        )
        & market_cap_nunique.gt(
            1
        )
    )

    # AMFI workbooks can contain the same ISIN more than once
    # (typically duplicate exchange/name rows).  Do not duplicate
    # one security in our PIT join.  Use the best/smallest published
    # rank as the representative row, but treat category disagreement
    # as unresolved rather than silently choosing a side.
    if duplicated.any():
        rows = (
            rows.sort_values(
                [
                    "isin",
                    "amfi_market_cap_rank",
                ],
                kind="stable",
            )
            .drop_duplicates(
                "isin",
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

        conflict = rows[
            "duplicate_category_disagreement"
        ]

        rows.loc[
            conflict,
            "amfi_cap_bucket",
        ] = pd.NA
        rows.loc[
            conflict,
            "category_rank_match",
        ] = False

    return rows.sort_values(
        "amfi_market_cap_rank"
    ).reset_index(
        drop=True
    )


def build_archive(
    root: Path,
    *,
    refresh: bool,
    offline: bool,
) -> dict:
    root = Path(
        root
    ).resolve()

    raw_root = (
        root
        / "data/raw/"
        "amfi_market_cap"
    )
    processed_root = (
        root
        / "data/processed/"
        "amfi_market_cap"
    )
    reports_root = (
        root
        / "reports/"
        "amfi_market_cap"
    )

    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        raw_root
        / "manifest.json"
    )

    if offline:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "--offline requires an existing "
                f"manifest: {manifest_path}"
            )
        manifest = json.loads(
            manifest_path.read_text()
        )
        sources = manifest[
            "sources"
        ]
        validate_period_completeness(
            sources
        )
    else:
        print(
            "Discovering AMFI historical "
            "market-cap workbooks..."
        )
        urls = discover_excel_urls()

        # Add exact official AMFI fallbacks for historical anchors
        # that have been observed to disappear from simple HTTP
        # discovery even though they remain listed on the archive page.
        urls.extend(
            KNOWN_OFFICIAL_WORKBOOKS.values()
        )
        urls = list(
            dict.fromkeys(
                urls
            )
        )

        if not urls:
            raise RuntimeError(
                "AMFI archive page returned "
                "no Excel links."
            )

        sources = []

        for url in urls:
            try:
                end = infer_period_end(
                    url
                )
            except ValueError:
                print(
                    "  skipping unrecognised "
                    f"Excel link: {url}"
                )
                continue

            name = cache_name(
                url,
                end,
            )
            sources.append({
                "url": url,
                "period_end": str(
                    end.date()
                ),
                "cache_file": (
                    name
                ),
            })

        # If discovery omitted a known official historical workbook,
        # inject it by its authoritative period endpoint.
        existing_periods = {
            pd.Timestamp(
                row[
                    "period_end"
                ]
            ).normalize()
            for row in sources
        }

        for (
            period_end,
            url,
        ) in (
            KNOWN_OFFICIAL_WORKBOOKS.items()
        ):
            if period_end in existing_periods:
                continue

            sources.append({
                "url": url,
                "period_end": str(
                    period_end.date()
                ),
                "cache_file": cache_name(
                    url,
                    period_end,
                ),
            })

        sources = sorted(
            sources,
            key=lambda row: (
                row[
                    "period_end"
                ]
            ),
        )

        validate_period_completeness(
            sources
        )

        manifest = {
            "archive_url": (
                ARCHIVE_URL
            ),
            "availability_policy": (
                "conservative_one_full_month_after_period_end"
            ),
            "sources": sources,
        }

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            )
            + "\n"
        )

    frames: list[
        pd.DataFrame
    ] = []
    summaries: list[
        dict
    ] = []

    print(
        f"AMFI snapshots discovered: "
        f"{len(sources):,}"
    )

    for i, source in enumerate(
        sources,
        start=1,
    ):
        url = source[
            "url"
        ]
        end = pd.Timestamp(
            source[
                "period_end"
            ]
        ).normalize()
        path = (
            raw_root
            / source[
                "cache_file"
            ]
        )

        if (
            not path.is_file()
            or refresh
        ):
            if offline:
                raise FileNotFoundError(
                    f"Missing cached workbook: {path}"
                )

            print(
                f"  [{i}/{len(sources)}] "
                f"downloading {end.date()}..."
            )
            download_file(
                url,
                path,
            )
        else:
            print(
                f"  [{i}/{len(sources)}] "
                f"cached {end.date()}"
            )

        parsed = parse_workbook(
            path,
            url=url,
            period_end=end,
        )

        frames.append(
            parsed
        )

        summaries.append({
            "measurement_end": (
                str(
                    end.date()
                )
            ),
            "effective_from": str(
                conservative_effective_from(
                    end
                ).date()
            ),
            "rows": int(
                len(
                    parsed
                )
            ),
            "large_cap_rows": int(
                parsed[
                    "amfi_cap_bucket"
                ].eq(
                    "large_cap"
                ).sum()
            ),
            "mid_cap_rows": int(
                parsed[
                    "amfi_cap_bucket"
                ].eq(
                    "mid_cap"
                ).sum()
            ),
            "small_cap_rows": int(
                parsed[
                    "amfi_cap_bucket"
                ].eq(
                    "small_cap"
                ).sum()
            ),
            "duplicate_isin_groups": int(
                parsed.loc[
                    parsed[
                        "duplicate_isin_count"
                    ].gt(
                        1
                    ),
                    "isin",
                ].nunique()
            ),
            "duplicate_category_conflicts": int(
                parsed[
                    "duplicate_category_disagreement"
                ].sum()
            ),
            "duplicate_market_cap_conflicts": int(
                parsed[
                    "duplicate_market_cap_disagreement"
                ].sum()
            ),
            "category_rank_mismatches": int(
                (
                    ~parsed[
                        "category_rank_match"
                    ].fillna(
                        False
                    )
                ).sum()
            ),
            "average_market_cap_available_fraction": float(
                parsed[
                    "amfi_average_market_cap_cr"
                ].notna().mean()
            ),
            "source_url": url,
        })

    if not frames:
        raise RuntimeError(
            "No AMFI snapshots parsed."
        )

    snapshots = pd.concat(
        frames,
        ignore_index=True,
    )

    periods = (
        snapshots[
            [
                "measurement_start",
                "measurement_end",
                "effective_from",
                "source_url",
                "availability_policy",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "effective_from"
        )
        .reset_index(
            drop=True
        )
    )

    periods[
        "effective_until_exclusive"
    ] = periods[
        "effective_from"
    ].shift(
        -1
    )

    snapshots = snapshots.merge(
        periods[
            [
                "measurement_end",
                "effective_until_exclusive",
            ]
        ],
        on="measurement_end",
        how="left",
        validate="many_to_one",
    )

    snapshots.to_parquet(
        processed_root
        / "amfi_market_cap_snapshots.parquet",
        index=False,
        compression="zstd",
    )
    periods.to_parquet(
        processed_root
        / "amfi_market_cap_periods.parquet",
        index=False,
        compression="zstd",
    )

    summary_frame = (
        pd.DataFrame(
            summaries
        )
    )
    summary_frame.to_csv(
        reports_root
        / "snapshot_summary.csv",
        index=False,
    )

    summary = {
        "snapshots": int(
            periods.shape[
                0
            ]
        ),
        "rows": int(
            len(
                snapshots
            )
        ),
        "first_measurement_end": str(
            pd.Timestamp(
                periods[
                    "measurement_end"
                ].min()
            ).date()
        ),
        "last_measurement_end": str(
            pd.Timestamp(
                periods[
                    "measurement_end"
                ].max()
            ).date()
        ),
        "first_effective_from": str(
            pd.Timestamp(
                periods[
                    "effective_from"
                ].min()
            ).date()
        ),
        "availability_policy": (
            "conservative_one_full_month_after_period_end"
        ),
        "duplicate_isin_groups": int(
            snapshots.loc[
                snapshots[
                    "duplicate_isin_count"
                ].gt(
                    1
                ),
                [
                    "measurement_end",
                    "isin",
                ],
            ].drop_duplicates().shape[
                0
            ]
        ),
        "duplicate_category_conflicts": int(
            snapshots[
                "duplicate_category_disagreement"
            ].sum()
        ),
        "duplicate_market_cap_conflicts": int(
            snapshots[
                "duplicate_market_cap_disagreement"
            ].sum()
        ),
        "rank_category_mismatches": int(
            (
                ~snapshots[
                    "category_rank_match"
                ].fillna(
                    False
                )
            ).sum()
        ),
        "processed_root": str(
            processed_root
        ),
    }

    (
        reports_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return summary


def self_test() -> None:
    assert infer_period_end(
        "https://x/Avg.%20Market%20Capitalization%20during%20-Jan-June%202018.xlsx"
    ) == pd.Timestamp(
        "2018-06-30"
    )
    assert infer_period_end(
        "https://x/AverageMarketCapitalization30Jun2026.xlsx"
    ) == pd.Timestamp(
        "2026-06-30"
    )
    assert conservative_effective_from(
        pd.Timestamp(
            "2024-06-30"
        )
    ) == pd.Timestamp(
        "2024-08-01"
    )
    assert conservative_effective_from(
        pd.Timestamp(
            "2024-12-31"
        )
    ) == pd.Timestamp(
        "2025-02-01"
    )
    expected = expected_period_ends(
        pd.Timestamp(
            "2017-12-31"
        ),
        pd.Timestamp(
            "2026-06-30"
        ),
    )
    assert len(
        expected
    ) == 18
    assert pd.Timestamp(
        "2020-12-31"
    ) in expected
    assert pd.Timestamp(
        "2024-06-30"
    ) in expected

    assert _normalise_category(
        "Large Cap"
    ) == "large_cap"
    assert _normalise_isin(
        "INE002A01018"
    ) == "INE002A01018"

    print(
        "AMFI market-cap self-test: PASS"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download and parse AMFI's official "
            "historical large/mid/small-cap "
            "classification workbooks."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
    )
    ap.add_argument(
        "--offline",
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

    summary = build_archive(
        Path(
            args.root
        ),
        refresh=args.refresh,
        offline=args.offline,
    )

    print(
        "\n=== AMFI MARKET-CAP ARCHIVE COMPLETE ==="
    )
    print(
        f"Snapshots:             "
        f"{summary['snapshots']:,}"
    )
    print(
        f"Rows:                  "
        f"{summary['rows']:,}"
    )
    print(
        f"First effective date:  "
        f"{summary['first_effective_from']}"
    )
    print(
        f"Rank/category mismatch:"
        f" {summary['rank_category_mismatches']:,}"
    )
    print(
        f"Output:                "
        f"{summary['processed_root']}"
    )


if __name__ == "__main__":
    main()
