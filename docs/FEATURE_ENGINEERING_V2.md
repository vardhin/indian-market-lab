# Feature Engineering V2

## Research boundary

The decision timestamp is **after the close of trading session t**.

The earliest permitted execution is **the next trading session open, t+1**.

Every predictive feature attached to row (security i, session t) must therefore be
fully observable by the close of t. Future opens, closes, corporate actions,
future index membership, future sector classifications, and any statistic using
future outcomes are forbidden.

The frozen `research_panel` remains unchanged. V2 features are written as a
keyed sidecar under:

```text
data/processed/feature_panel_v2/date=YYYY-MM-DD/data.parquet
```

with keys:

```text
date
market_day_index
canonical_security_id
symbol
eligible_universe
```

This keeps target/execution semantics separate from experimental features.

## Feature families

### F1 — OHLC geometry

Implemented from adjusted OHLC known by close(t):

- candle body as a fraction of open
- upper/lower wick as a fraction of open
- close location within the daily range
- body/range ratio
- true-range percentage
- position within trailing 20/60/252-session ranges

Absolute price levels are not used as predictive features here.

### F2 — streak / discrete behavioral state

Implemented:

- signed consecutive close-direction streak
- signed consecutive intraday-direction streak
- higher/lower-high streak
- higher/lower-low streak
- gap-direction streak

Positive values denote consecutive positive/up states and negative values denote
consecutive negative/down states.

### F3 — stock-specific historical tendencies

Implemented point-in-time expanding estimates conditional on the stock's current
signed close streak (capped at +/-5):

- historical mean next-session close-to-close return
- historical probability of a positive next-session return
- count of prior occurrences

The current event's own future outcome is excluded. Only previous state
occurrences whose t+1 outcome is already knowable by the current close may
contribute.

### F4 — broad-market context

Implemented using the existing point-in-time daily NIFTY 50 and NIFTY 500 price
index files:

- stock minus index return over 1/5/20/60 sessions
- rolling stock/index correlation over 60/252 observations
- rolling stock/index beta over 60/252 observations

### F5 — sector / industry context

Pending.

Do not backfill a current sector or industry classification into historical
dates without an effective-date policy. Historical point-in-time classifications
or a defensible stable-taxonomy mapping are required first.

### F6 — lead/lag and response

Implemented at daily/session resolution:

- market(t-k) vs stock(t) rolling relationships for k = 0,1,2,3,5
- stock(t-k) vs market(t) rolling relationships for k = 1,2,3,5
- current stock return minus rolling-beta-expected market response

These features describe same-session to five-session response structure. They do
not make intraday or HFT claims.

### F7 — size / large-cap context

Pending.

The primary intended universe is point-in-time Indian large cap. Current 2026
membership must never be projected backward. Before enabling this family we
need either historical official index membership or point-in-time market
capitalization derived from historically valid shares outstanding.

### F8 — cross-sectional state

Implemented same-date percentile ranks using only information known at close(t):

- 5-session return
- 20-session return
- 60-session return
- turnover

### F9 — calendar / seasonality

Implemented:

- weekday
- month
- quarter
- month-end
- quarter-end
- March/fiscal-year-end-month indicator

Absolute year is deliberately excluded from the predictive feature set. Year is
an attribution and diagnostic dimension because feeding it to a tree can
encourage historical-regime memorization with no natural rule for unseen years.

## Leakage rules

Examples of legal features for signal date t:

```text
open(t)
high(t)
low(t)
close(t)
volume(t)
NIFTY close(t)
trailing statistics ending at t
same-date cross-sectional ranks
rolling beta estimated through t
```

Forbidden:

```text
open(t+1)
close(t+1)
future index constituents
future market-cap bucket
future sector mapping
future corporate-action knowledge used to decide eligibility
full-sample conditional statistics pasted backward
normalization fit on future observations
feature selection using the held-out evaluation period without nesting
```

## Hindsight / researcher overfitting

Repeatedly observing 2019–2026 while inventing features makes that period an
informal training set even when the model itself never sees future labels.

Feature-family development must therefore be evaluated through predeclared
ablations and eventually nested or genuinely untouched evaluation.

The intended comparison sequence is:

```text
F0
F0+F1
F0+F1+F2
F0+F1+F2+F3
...
```

rather than selecting individual features after inspecting the final test
period.

## Multicollinearity and redundancy

Tree models do not require classical linear-regression multicollinearity
elimination, but highly redundant features can destabilize importance,
encourage unnecessary search, and increase overfitting.

Use sparse, interpretable horizon grids such as:

```text
1 / 5 / 20 / 60 / 120 / 252
```

rather than every possible lookback.

## Explainability

Predictor explainability and economic attribution are separate requirements.

Model-level:

- SHAP / feature importance
- interaction stability
- year-to-year feature stability

Strategy-level:

- selected securities
- sector / industry exposure
- market-cap bucket
- year / month / weekday
- volatility regime
- market regime
- contribution to P&L
- contribution to drawdown
- turnover and transaction costs

A strong CAGR without knowing where the returns came from is not treated as a
finished result.

## Current status

Implemented families can be built with:

```bash
uv run python src/data/feature_engineering_v2.py --self-test
uv run python src/data/feature_engineering_v2.py
```

The next gated step is to source and validate point-in-time large-cap and
sector/industry data before F5/F7 are allowed into model training.


## Point-in-time large-cap metadata

AMFI ingestion is implemented in:

```text
src/data/amfi_market_cap.py
```

It discovers and caches the official AMFI historical Excel workbooks, parses
ISIN, rank, average market capitalization, NSE symbol and SEBI cap bucket, and
writes normalized snapshots under:

```text
data/processed/amfi_market_cap/
```

Historical publication timestamps are not assumed from the six-month
measurement end. The default research availability policy is deliberately
conservative:

```text
Jan-Jun measurement  -> effective Aug 1
Jul-Dec measurement  -> effective Feb 1 of the following year
```

The prior snapshot remains active until the next conservative effective date.

The research-panel join is implemented as a second sidecar:

```text
src/data/build_pit_metadata_v2.py
data/processed/pit_metadata_v2/
```

AMFI membership is joined by resolved ISIN only, with a canonical ISIN fallback.
There is deliberately no symbol-only classification fallback. Unknown
classification is represented as unknown rather than silently treating the
security as small cap.

The intended large-cap research flag is:

```text
pit_largecap_universe
  = eligible_universe
    AND point-in-time AMFI large-cap classification
```

This makes the large-cap experiment available only once a defensible AMFI
classification is point-in-time available.

## Future-mutation leakage regression

The strongest feature-v2 leakage regression is:

```text
src/data/test_feature_engineering_v2_leakage.py
```

It loads real research-panel rows around a historical cutoff, computes every
implemented v2 feature, then deliberately corrupts all stock and broad-market
source values after the cutoff and rebuilds the features.

Every feature at or before the cutoff must remain numerically identical. Any
difference fails the test and writes the exact date/security/feature mismatch.

This is complementary to construction-time reasoning: it catches accidental
future dependence introduced by shifts, joins, expanding statistics or later
refactors.
