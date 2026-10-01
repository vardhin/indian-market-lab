from pathlib import Path
import pandas as pd

RAW = Path(
    "data/raw/"
    "BhavCopy_NSE_CM_0_0_0_20260925_F_0000.csv"
)

OUT = Path("data/processed/nse_daily.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# ----------------------------------------
# 1. Ordinary NSE equity series only
# ----------------------------------------

df = df[df["SctySrs"] == "EQ"].copy()

# ----------------------------------------
# 2. Select useful fields
# ----------------------------------------

columns = {
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

    "NewBrdLotQty": "board_lot",
}

df = df[list(columns)].rename(columns=columns)

# ----------------------------------------
# 3. Types
# ----------------------------------------

df["date"] = pd.to_datetime(df["date"])

numeric_cols = [
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "last",
    "prev_close",
    "volume",
    "turnover",
    "trades",
    "board_lot",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------
# 4. Derived fields
# ----------------------------------------

df["daily_return"] = (
    df["close"] / df["prev_close"] - 1
)

df["intraday_return"] = (
    df["close"] / df["open"] - 1
)

df["range_pct"] = (
    (df["high"] - df["low"]) / df["open"]
)

# ----------------------------------------
# 5. Stable ordering
# ----------------------------------------

df = (
    df.sort_values(["date", "isin"])
      .reset_index(drop=True)
)

# ----------------------------------------
# 6. Save
# ----------------------------------------

df.to_parquet(
    OUT,
    index=False,
    compression="zstd",
)

print(df.head(10).to_string(index=False))

print("\n--- Summary ---")
print("Rows:", len(df))
print("Stocks:", df["isin"].nunique())
print("Date:", df["date"].min())
print("Output:", OUT)
print(
    "Size:",
    round(OUT.stat().st_size / 1024, 2),
    "KiB",
)
