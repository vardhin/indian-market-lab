import json
from pathlib import Path

import pandas as pd

PATH = Path("data/processed/nse_daily.parquet")
REPORT = Path("reports/data_quality.json")

df = pd.read_parquet(PATH)

report = {}

report["rows"] = len(df)
report["stocks"] = int(df["isin"].nunique())
report["dates"] = int(df["date"].nunique())

# Identity
report["missing_isin"] = int(df["isin"].isna().sum())
report["missing_symbol"] = int(df["symbol"].isna().sum())

report["duplicate_date_isin"] = int(
    df.duplicated(["date", "isin"]).sum()
)

# OHLC sanity
report["high_below_open"] = int(
    (df["high"] < df["open"]).sum()
)

report["high_below_close"] = int(
    (df["high"] < df["close"]).sum()
)

report["low_above_open"] = int(
    (df["low"] > df["open"]).sum()
)

report["low_above_close"] = int(
    (df["low"] > df["close"]).sum()
)

report["high_below_low"] = int(
    (df["high"] < df["low"]).sum()
)

# Price/volume sanity
report["nonpositive_close"] = int(
    (df["close"] <= 0).sum()
)

report["negative_volume"] = int(
    (df["volume"] < 0).sum()
)

report["negative_turnover"] = int(
    (df["turnover"] < 0).sum()
)

# Missing values
for col in [
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
]:
    report[f"missing_{col}"] = int(
        df[col].isna().sum()
    )

print(json.dumps(report, indent=2))

REPORT.parent.mkdir(exist_ok=True)

with REPORT.open("w") as f:
    json.dump(report, f, indent=2)
