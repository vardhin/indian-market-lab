from pathlib import Path

import pandas as pd


DATE = "2026-09-25"

SECURITY_MASTER = Path(
    "data/raw/NSE_CM_security_25092026.csv"
)

EQUITY_LIST = Path(
    "data/raw/EQUITY_L.csv"
)

ETF_LIST = Path(
    "data/raw/eq_etfseclist.csv"
)

OUT = Path(
    "data/processed/instruments.parquet"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_nse_date(series: pd.Series) -> pd.Series:
    """
    NSE security master may encode dates as Unix timestamps,
    zeroes, strings, or NaN depending on field/instrument.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    # Values around 1e9 are Unix seconds.
    mask_epoch = numeric > 100_000_000

    if mask_epoch.any():
        result.loc[mask_epoch] = pd.to_datetime(
            numeric.loc[mask_epoch],
            unit="s",
            errors="coerce",
        )

    # Try parsing remaining non-empty values as strings.
    mask_other = ~mask_epoch & series.notna()

    if mask_other.any():
        result.loc[mask_other] = pd.to_datetime(
            series.loc[mask_other],
            errors="coerce",
            dayfirst=True,
        )

    return result


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

master = clean_columns(
    pd.read_csv(
        SECURITY_MASTER,
        low_memory=False,
    )
)

equities = clean_columns(
    pd.read_csv(
        EQUITY_LIST,
        low_memory=False,
    )
)

etfs = clean_columns(
    pd.read_csv(
        ETF_LIST,
        low_memory=False,
    )
)


# ------------------------------------------------------------
# Normalize reference lists
# ------------------------------------------------------------

equities = equities.rename(
    columns={
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "company_name",
        "SERIES": "equity_list_series",
        "DATE OF LISTING": "equity_listing_date",
        "PAID UP VALUE": "paid_up_value",
        "MARKET LOT": "equity_market_lot",
        "ISIN NUMBER": "isin",
        "FACE VALUE": "face_value",
    }
)

etfs = etfs.rename(
    columns={
        "Symbol": "symbol",
        "Underlying Asset": "underlying_asset",
        "SecurityName": "security_name",
        "DateofListing": "etf_listing_date",
        "MarketLot": "etf_market_lot",
        "ISINNumber": "isin",
        "FaceValue": "etf_face_value",
        "ETF Underlying": "etf_underlying_type",
        "Underlying Key": "underlying_key",
    }
)

equity_isins = set(
    equities["isin"]
    .dropna()
    .astype(str)
    .str.strip()
)

etf_isins = set(
    etfs["isin"]
    .dropna()
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Select security-master fields
# ------------------------------------------------------------

keep = [
    "FinInstrmId",
    "TckrSymb",
    "SctySrs",
    "FinInstrmNm",
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

    "FinInstrmTp",
    "InstrmTp",
    "AsstClss",
]

# Some columns may be missing/change in future NSE files.
keep = [c for c in keep if c in master.columns]

df = master[keep].copy()

df = df.rename(
    columns={
        "FinInstrmId": "instrument_id",
        "TckrSymb": "symbol",
        "SctySrs": "series",
        "FinInstrmNm": "name",
        "ISIN": "isin",

        "NewBrdLotQty": "board_lot",
        "ParVal": "par_value",
        "TickSz": "tick_size",

        "IssdCptl": "issued_capital",
        "FreeFltCptl": "free_float_capital",

        "ListgDt": "listing_date",
        "RmvlDt": "removal_date",
        "RadmssnDt": "readmission_date",

        "DelFlg": "deleted_flag",

        "SctyStsNrmlMkt": "normal_market_status",
        "ElgbltyNrmlMkt": "normal_market_eligible",

        "PreOpnAllwdFlg": "preopen_allowed",
        "SLBMElgblty": "slbm_eligible",

        "TradToTradInd": "trade_to_trade",

        "FinInstrmTp": "financial_instrument_type",
        "InstrmTp": "instrument_type",
        "AsstClss": "asset_class_raw",
    }
)


# ------------------------------------------------------------
# Normalize identifiers
# ------------------------------------------------------------

for col in ["isin", "symbol", "series", "name"]:
    if col in df:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )


# ------------------------------------------------------------
# Dates
# ------------------------------------------------------------

for col in [
    "listing_date",
    "removal_date",
    "readmission_date",
]:
    if col in df.columns:
        df[col] = parse_nse_date(df[col])


df["as_of_date"] = pd.Timestamp(DATE)


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

df["is_etf"] = df["isin"].isin(etf_isins)

df["is_company_equity"] = (
    df["isin"].isin(equity_isins)
    &
    ~df["is_etf"]
)


def classify(row):
    # ETF check comes first deliberately.
    if row["is_etf"]:
        return "etf"

    if row["is_company_equity"]:
        return "company_equity"

    series = row["series"]

    mapping = {
        "GB": "sovereign_gold_bond",
        "GS": "government_security",
        "TB": "treasury_bill",
        "IV": "invit",
        "RR": "reit",
    }

    return mapping.get(series, "other")


df["asset_class"] = df.apply(
    classify,
    axis=1,
)


# ------------------------------------------------------------
# Identify obvious NSE dummy/test instruments
# ------------------------------------------------------------

df["is_test_instrument"] = (
    df["symbol"]
    .fillna("")
    .str.contains(
        "NSETEST",
        case=False,
        regex=False,
    )
    |
    df["isin"]
    .fillna("")
    .str.startswith("DUMMY")
)


# ------------------------------------------------------------
# Initial research/trading eligibility
# ------------------------------------------------------------
#
# Very intentionally conservative.
#
# For our first swing-trading experiments:
#   company equity
#   EQ series
#   not test/dummy
#   not deleted
#
# We can add liquidity constraints later.
#

deleted = (
    df["deleted_flag"]
    .astype("string")
    .str.upper()
    .eq("Y")
)

df["eligible_basic_equity"] = (
    df["is_company_equity"]
    &
    df["series"].eq("EQ")
    &
    ~df["is_test_instrument"]
    &
    ~deleted
)


# ------------------------------------------------------------
# Enrich ETFs
# ------------------------------------------------------------

etf_meta = etfs[
    [
        "isin",
        "underlying_asset",
        "etf_underlying_type",
        "underlying_key",
    ]
].drop_duplicates("isin")

df = df.merge(
    etf_meta,
    how="left",
    on="isin",
)


# ------------------------------------------------------------
# Enrich company equities
# ------------------------------------------------------------

equity_meta = equities[
    [
        "isin",
        "company_name",
        "equity_list_series",
        "equity_listing_date",
        "face_value",
    ]
].drop_duplicates("isin")

df = df.merge(
    equity_meta,
    how="left",
    on="isin",
)


# ------------------------------------------------------------
# Sort + save
# ------------------------------------------------------------

df = (
    df.sort_values(
        [
            "asset_class",
            "isin",
            "series",
            "symbol",
        ]
    )
    .reset_index(drop=True)
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUT,
    index=False,
    compression="zstd",
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n=== Instrument Master ===")
print("Rows:", len(df))
print("Unique ISINs:", df["isin"].nunique())

print("\nAsset classes:")
print(
    df["asset_class"]
    .value_counts(dropna=False)
)

print("\nCompany equities:")
print(int(df["is_company_equity"].sum()))

print("ETFs:")
print(int(df["is_etf"].sum()))

print("Test instruments:")
print(int(df["is_test_instrument"].sum()))

print("Basic eligible EQ equities:")
print(int(df["eligible_basic_equity"].sum()))

print("\nSeries among company equities:")
print(
    df.loc[
        df["is_company_equity"],
        "series",
    ]
    .value_counts()
    .head(20)
)

print("\nExample eligible stocks:")
print(
    df.loc[
        df["eligible_basic_equity"],
        [
            "symbol",
            "isin",
            "name",
            "series",
            "listing_date",
        ],
    ]
    .head(20)
    .to_string(index=False)
)

print("\nExample ETFs:")
print(
    df.loc[
        df["is_etf"],
        [
            "symbol",
            "isin",
            "name",
            "underlying_asset",
        ],
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSaved:", OUT)
