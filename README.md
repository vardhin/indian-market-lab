# Indian Market Lab

Point-in-time Indian cash-equity research pipeline built from official NSE
archives.

## Current dataset stage

The historical backfill currently spans 2010 through 2026 and normalizes both
NSE cash-market formats:

- legacy CM bhavcopy before 2024-07-08
- UDiFF bhavcopy from 2024-07-08 onward

The identity layer preserves historical securities before applying present-day
classification hints, avoiding the common mistake of constructing old
universes from today's survivor list.

Current completed identity run:

- 6,605,639 EQ-series rows
- 4,136 trading dates
- 4,120 symbols
- 6,115,762 rows with explicit NSE ISINs
- 472,405 missing historical ISIN rows conservatively propagated
- 17,472 unresolved rows
- 136 unresolved identity episodes
- 6,193,122 resolved company-equity rows

## Resume a historical build

Daily partitions are checkpointed. A failed historical build can be resumed
without rebuilding completed years:

```bash
uv run python src/data/build.py \
  --from 2010-01-01 \
  --to 2026-09-30 \
  --no-consolidate \
  --resolve-identities \
  --resume
```

`--resume` is coverage-aware. It does not simply jump to the newest existing
partition, because isolated later test partitions may exist beyond an unfinished
historical range.

## Audit unresolved identities

After identity resolution, rank the remaining ambiguous episodes by actual
market activity:

```bash
uv run python src/data/audit_unresolved.py
```

Outputs:

```text
reports/identity_audit/
├── unresolved_identity_audit.parquet
├── unresolved_identity_audit.csv
├── unresolved_identity_audit.md
└── summary.json
```

The audit reports:

- unresolved date range and row count
- explicit and resolved ISIN evidence
- nearest left/right anchor transitions
- whether ambiguity occurs across an ISIN change or because no anchor exists
- volume, turnover, and price statistics for unresolved rows
- each episode's share of all unresolved turnover
- transparent economic-relevance bands and manual-review priorities

No ambiguous identity is force-resolved by the audit.

## Data layers

```text
data/processed/
├── eq_all/                 # historical EQ-series universe
├── equities/               # immediately classifiable company equities
├── identity/               # row-level identity maps + episode summaries
└── equities_resolved/      # resolved company-equity research universe
```


## Corporate actions

Corporate actions are ingested separately from raw prices. The first stage
downloads and normalizes NSE equity corporate actions without modifying the
historical OHLCV dataset:

```bash
uv sync
uv run python src/data/corporate_actions.py \
  --from 2010-01-01 \
  --to 2026-09-30
```

Before the full fetch, the parser can be checked locally:

```bash
uv run python src/data/corporate_actions.py --self-test
```

Outputs:

```text
data/raw/corporate_actions/                       # cached NSE JSON windows
data/processed/corporate_actions/
├── nse_corporate_actions.parquet
└── nse_corporate_actions.csv
reports/
├── corporate_actions_summary.json
└── corporate_actions_unparsed_share_count.csv   # only when needed
```

The ingestion layer classifies dividends, bonuses, splits/consolidations,
rights, demergers, mergers/schemes, buybacks and other actions.

Only mechanically deterministic share-count actions are marked
`auto_adjustable`:

- bonus issue: post-action share multiplier = `(new + existing) / existing`
- split/consolidation: post-action share multiplier =
  `old face value / new face value`

Dividends are stored as cash-flow information rather than silently baked into
OHLC prices. Rights, demergers, mergers and schemes are preserved and flagged
for separate treatment rather than assigned guessed adjustment factors.

The next stage maps these events onto the resolved historical security
lineages and writes a separate adjusted-price research layer while preserving
the raw NSE prices unchanged.


### Build mechanically adjusted OHLCV

After the corporate-action ingest, map split/bonus events to the resolved
historical security lineage and build a separate backward-adjusted research
layer:

```bash
uv run python src/data/adjust_prices.py --self-test
uv run python src/data/adjust_prices.py
```

The builder:

- maps each split/bonus to a resolved `canonical_security_id`
- prefers an exact ex-date match
- uses only a two-sided same-identity bracket as fallback
- never accepts a one-sided fallback
- compounds multiple same-day mechanical actions multiplicatively
- applies factors only to rows strictly before the action ex-date
- handles missing ex-date partitions by date ordering
- preserves every raw NSE price/volume column
- writes separate `adj_open`, `adj_high`, `adj_low`, `adj_close`,
  `adj_last`, and `adj_volume` columns
- writes diagnostics for unmapped or ambiguous actions

Outputs:

```text
data/processed/corporate_actions/
├── mechanical_action_identity_map.parquet
├── mechanical_action_identity_map.csv
├── mechanical_adjustment_events.parquet
└── mechanical_adjustment_events.csv

data/processed/equities_adjusted/
└── date=YYYY-MM-DD/data.parquet

reports/
├── adjusted_prices_summary.json
└── corporate_actions_unmapped_mechanical.csv
```

The raw `equities_resolved` dataset is never modified. Dividends, rights,
demergers and merger/scheme events are also not folded into these mechanical
OHLCV factors; they remain separate event/cash-flow layers.


## Build the point-in-time research panel

After adjusted OHLCV is available:

```bash
uv run python src/data/build_research_panel.py --self-test
uv run python src/data/build_research_panel.py
```

The default universe is designed for an after-close signal and next-session-open
entry:

- at least 120 prior trading observations
- raw/as-traded close >= INR 10
- trailing 20-observation median turnover >= INR 5,000,000
- positive-volume ratio over the trailing 20 observations >= 90%
- complete observation coverage across the most recent 20 NSE market sessions
- no unresolved split/bonus event inside the trailing feature-history window

Future execution availability is deliberately NOT part of `eligible_universe`.
A stock missing the next session is an execution/label outcome, not information
known at close(t).

All thresholds remain explicit CLI parameters.

Features use only information available through the close of date `t`. Lagged
returns are accepted only when they line up with the actual NSE market-session
index, so a suspension or missing security row cannot masquerade as a normal
1/5/20-day return.

Targets are generated separately:

```text
signal: close(t)
entry:  open(t+1)
exit:   close(t+h), h in {1,3,5,10,20}
```

The builder also writes close-to-close targets for forecasting diagnostics.
Neither target family is used in feature calculation or universe selection.

Unparsed or unmapped split/bonus events are not manually guessed. Instead the
builder separates two cases:

- `unsafe_feature_window=True` for rows after an unresolved mechanical event
  whose trailing 60-session features can be contaminated; these rows are
  excluded from the point-in-time universe.
- `unsafe_target_window_{h}d=True` for rows whose future h-session label
  crosses such an event; these rows may still have been valid point-in-time
  trading candidates, but that particular historical target is excluded from
  supervised training.

Accordingly, `eligible_universe` contains only information available through
close(t), while `training_eligible_{h}d` may additionally depend on future
label availability.

Outputs:

```text
data/processed/research_panel/
└── date=YYYY-MM-DD/data.parquet

reports/
├── research_panel_summary.json
└── research_universe_by_year.csv
```


## Deterministic baseline backtests

The first strategy suite is deliberately simple and auditable:

```text
momentum_5d
momentum_20d
momentum_60d
mean_reversion_5d
risk_adjusted_momentum_20d
```

Run the synthetic execution/cost sanity test first:

```bash
uv run python src/backtest/baselines.py --self-test
```

Then run the default INR 50,000 baseline suite:

```bash
uv run python src/backtest/baselines.py
```

Default execution convention:

```text
signal: after close(t)
entry:  next NSE session open
exit:   close(t + 5 market sessions)
portfolio: top 5, equal slot budgets, integer initial share quantities
cohorts: non-overlapping
```

The default cost profile is a constant 2026 delivery-equity friction model:

- STT: 0.1% on delivery buy and 0.1% on delivery sell
- stamp duty: 0.015% on the buy side
- SEBI turnover fee: INR 10/crore each side
- NSE cash-market outflow: INR 307/crore each side
- GST: 18% on brokerage + exchange + SEBI service charges
- brokerage assumption: INR 15 per executed order, configurable
- slippage assumption: 5 bps per side, configurable
- DP charge: configurable, default zero because it is broker-specific

This is intentionally a constant-current-cost experiment, not a historical
claim about transaction levies in 2010-2025. It answers whether historical
signals survive a current friction model. A point-in-time historical fee
schedule can be added as a later robustness test.

Examples:

```bash
# Gross strategy signal baseline
uv run python src/backtest/baselines.py --zero-costs

# 10-stock portfolio, 20-session holding period
uv run python src/backtest/baselines.py --top-k 10 --holding-sessions 20

# Broker-specific assumptions
uv run python src/backtest/baselines.py \
  --brokerage-per-order 0 \
  --dp-charge-per-sell 15.34 \
  --slippage-bps 5
```

Mapped split/bonus events update the share quantity of an already-held
position before trading on the ex-date. If a selected historical trade's
forward outcome crosses an unresolved mechanical event, that selected slot is
left in cash for data-quality reasons and is not replaced by a lower-ranked
security. This is reported separately rather than hidden.

Outputs are written under:

```text
reports/backtests/
├── baseline_comparison_*.csv
├── run_summary_*.json
└── <strategy>/
    └── <run configuration>/
        ├── metrics.json
        ├── trades.csv
        └── equity_curve.csv
```


### Robustness grid

After the single-run baselines, sweep a compact deterministic surface without
changing the data or inventing extra features:

```bash
uv run python src/backtest/grid.py
```

Default grid:

```text
strategies:
  momentum_20d
  momentum_60d
  risk_adjusted_momentum_20d
  risk_adjusted_momentum_60d
  liquidity_control

holding horizons: 5, 10, 20 market sessions
portfolio breadth: top 3, 5, 10
cost profile: current 2026 delivery-equity model
```

The `liquidity_control` is intentionally dumb: it simply owns the most liquid
eligible names by trailing 20-day median turnover. It helps show whether a
momentum rule is adding anything beyond repeatedly selecting large/liquid
stocks.

To produce matched gross-vs-net results in the same sweep:

```bash
uv run python src/backtest/grid.py --cost-profiles current zero
```

Outputs:

```text
reports/backtests/grid/
├── grid_*.csv
├── grid_*_net_cagr_percent.csv
├── grid_*_gross_vs_net.csv       # when both cost profiles are requested
└── grid_*.json
```

The grid also reports Calmar, turnover multiples, fees, failed entries, delayed
exits and data-quality exclusions for each configuration.


## Market benchmark comparison

Official NSE historical price-index data is ingested separately from security
history:

```bash
uv run python src/data/index_benchmarks.py --self-test
uv run python src/data/index_benchmarks.py
```

Default indices:

```text
NIFTY 50
NIFTY 500
```

The downloader uses the NSE historical-index report endpoint and defaults to
60-calendar-day request windows. NSE can silently cap oversized historical
requests (observed as exactly 70 rows from year-long windows), so the ingest
now rejects suspicious capped responses, incomplete weekday coverage and data
that stops materially before the requested end date. Each request window is
cached under `data/raw/index_benchmarks/`.

It validates OHLC integrity and writes normalized price-index history to:

```text
data/processed/index_benchmarks/
├── nifty_50.parquet
├── nifty_50.csv
├── nifty_500.parquet
├── nifty_500.csv
├── nse_price_indices.parquet
└── nse_price_indices.csv
```

The first comparison deliberately uses price indices rather than total-return
indices. The stock simulator does not yet credit cash dividends, so comparing
price-only strategy wealth to a dividend-reinvested TRI would be asymmetric.
Dividend-inclusive strategy and TRI comparison belongs in the later cash-flow
stage.

Then generate the frozen deterministic benchmark report:

```bash
uv run python src/backtest/benchmark_report.py --self-test
uv run python src/backtest/benchmark_report.py
```

The evaluation window starts on the first date with at least one
`eligible_universe=True` row. The 120-session feature warm-up period is
excluded from both strategies and indices.

Frozen strategy comparators:

```text
liquidity_control        h20 / k5
momentum_20d             h20 / k5
momentum_60d             h20 / k5
momentum_60d             h20 / k10
risk_adjusted_momentum_60d h20 / k10
```

They are compared with NIFTY 50, NIFTY 500 and a zero-return cash baseline on:

- terminal wealth and total return
- CAGR
- annualized volatility
- Sharpe
- maximum drawdown
- Calmar
- calendar-year returns
- rolling 1-year and 3-year returns
- largest drawdown episodes
- CAGR difference versus NIFTY 50 and NIFTY 500

Outputs:

```text
reports/benchmarks/
├── benchmark_comparison.csv
├── calendar_year_returns_percent.csv
├── rolling_return_summary.csv
├── largest_drawdowns.csv
├── aligned_equity_curves.csv
└── benchmark_report_summary.json
```


## Classical ML temporal experiment

The first learned-model experiment is deliberately classical and
cross-sectional. It predicts the relative attractiveness of each eligible NSE
security rather than trying to forecast an exact rupee price.

Target:

```text
signal time: close(t)
entry:       open(t+1)
exit:        close(t+20)
label:       within-date percentile rank of next-open -> close(t+20) return
```

Feature set is point-in-time only and currently includes short/medium-term
returns, moving-average distance, high/low distance, volatility, range,
intraday/gap return, turnover/volume z-scores, activity ratio and log trailing
turnover.

The fixed temporal protocol is:

```text
TRAIN
2010-06-28 -> 2018-12-31
purge final 20 market sessions from labels

VALIDATION
2019-01-01 -> 2021-12-31
purge final 20 market sessions from labels

HELD-OUT TEST
2022-01-01 -> 2026-09-30
```

The split is never random. Boundary purging prevents a training or validation
label from using an exit price that lies beyond that split's information
cutoff.

Default candidate models:

```text
ridge
hist_gb
```

`random_forest` is implemented as an optional heavier baseline.

Model selection uses only validation **mean daily cross-sectional Spearman IC**.
After selection, that algorithm is refit on purged train+validation data and
scored once on the 2022-2026 held-out test period.

The final test predictions are run through the same delivery-equity backtester
used by deterministic strategies: INR 50,000 initial capital, integer shares,
next-session-open entry, 20-session close exit, mapped split/bonus handling,
current-2026 transaction-cost assumptions and the same unresolved-action
censor.

Run:

```bash
uv sync

uv run python src/backtest/baselines.py --self-test
uv run python src/ml/classical.py --self-test
uv run python src/ml/classical.py
```

The default final portfolio comparison evaluates the selected model at top-5
and top-10 against matching 60-day momentum and liquidity controls plus NIFTY
50 and NIFTY 500 price-index benchmarks.

For an optional random-forest candidate:

```bash
uv run python src/ml/classical.py \
  --models ridge hist_gb random_forest
```

By default all available training rows are used. For a quick computational
smoke test without changing the experiment code:

```bash
uv run python src/ml/classical.py \
  --max-train-rows 250000 \
  --max-refit-rows 400000
```

Do not use capped-row results as the final paper result.

Outputs:

```text
reports/ml/classical/
├── validation_model_comparison.csv
├── test_portfolio_comparison.csv
├── test_predictions.parquet
├── test_equity_curves_long.csv
├── experiment_summary.json
└── models/
    └── <selected_model>_selected.joblib
```

The held-out test comparison, rather than training fit statistics, is the
primary result.

## Research roadmap

1. Audit unresolved identity episodes.
2. Build point-in-time corporate-action adjustments.
3. Add market, sector and volatility benchmarks.
4. Define a point-in-time tradeable universe.
5. Generate leakage-safe features and 1/3/5/10/20-day targets.
6. Build a realistic INR 50,000 cash-equity simulator with Indian transaction
   costs and integer-share constraints.
7. Establish deterministic momentum/mean-reversion/ranking baselines.
8. Compare classical ML and temporal/deep forecasting models.
9. Test uncertainty-aware selective trading.
10. Compare deterministic allocation with offline RL.
11. Replicate key findings across additional markets.
