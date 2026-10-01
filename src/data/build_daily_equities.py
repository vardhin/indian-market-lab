from pathlib import Path

import pandas as pd


DATE = "20260925"

BHAV = Path(
    f"data/raw/BhavCopy_NSE_CM_0_0_0_{DATE}_F_0000.csv"
)

EQUITY_LIST = Path("data/raw/EQUITY_L.csv")
ETF_LIST = Path("data/raw/eq_etfseclist.csv")

OUT = Path(
    f"data/processed/equities/date=2026-09-25/data.parquet"
)


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


# --------------------------------------------------
# Load
# --------------------------------------------------

bhav = clean_columns(
    pd.read_csv(BHAV, low_memory=False)
)

equities = clean_columns(
    pd.read_csv(EQUITY_LIST, low_memory=False)
)

etfs = clean_columns(
    pd.read_csv(ETF_LIST, low_memory=False)
)


# --------------------------------------------------
# Reference sets
# --------------------------------------------------

company_isins = set(
    equities["ISIN NUMBER"]
    .dropna()
    .astype(str)
    .str.strip()
)

etf_isins = set(
    etfs["ISINNumber"]
    .dropna()
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# Start with actual EQ-series trades
# --------------------------------------------------

df = bhav[
    bhav["SctySrs"].eq("EQ")
].copy()

df["ISIN"] = (
    df["ISIN"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# Classify
# --------------------------------------------------

df["is_etf"] = df["ISIN"].isin(etf_isins)

df["is_company_equity"] = (
    df["ISIN"].isin(company_isins)
    & ~df["is_etf"]
)

df = df[
    df["is_company_equity"]
].copy()


# --------------------------------------------------
# Normalize
# --------------------------------------------------

rename = {
    "TradDt": "date",
    "FinInstrmId": "instrument_id",
    "ISIN": "isin",
    "TckrSymb": "symbol",
    "FinInstrmNm": "name",
    "SctySrs": "series",

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

df = df[list(rename)].rename(columns=rename)

df["date"] = pd.to_datetime(df["date"])

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

for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# --------------------------------------------------
# Basic derived values
# --------------------------------------------------

df["daily_return"] = (
    df["close"] / df["prev_close"] - 1
)

df["intraday_return"] = (
    df["close"] / df["open"] - 1
)

df["range_pct"] = (
    (df["high"] - df["low"])
    / df["open"]
)


# --------------------------------------------------
# Ensure one row per security/day
# --------------------------------------------------

dupes = df.duplicated(
    ["date", "isin"],
    keep=False,
)

if dupes.any():
    print(
        df.loc[
            dupes,
            ["date", "isin", "symbol", "series"]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Duplicate date/ISIN rows found."
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

df = df.sort_values(
    ["date", "isin"]
).reset_index(drop=True)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUT,
    index=False,
    compression="zstd",
)

print("Date:", df.date.iloc[0])
print("Company equities:", len(df))
print("Unique ISINs:", df["isin"].nunique())

print("\nLargest by turnover:")
print(
    df.sort_values(
        "turnover",
        ascending=False,
    )[
        [
            "symbol",
            "close",
            "volume",
            "turnover",
            "trades",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSaved:", OUT)
