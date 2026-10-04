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


## ML portfolio diagnostics and walk-forward construction

The first held-out classical-ML run showed positive cross-sectional information
coefficient but negative concentrated top-k portfolio returns. The next stage
therefore diagnoses the score distribution and portfolio-construction layer
without tuning directly on the 2022-2026 held-out result.

Run:

```bash
uv run python src/ml/portfolio_diagnostics.py --self-test
uv run python src/ml/portfolio_diagnostics.py
```

The default model family is `hist_gb`, frozen from the prior 2019-2021
validation model-selection experiment.

### Validation-only diagnostics

Using a model fit only through 2018, the 2019-2021 validation block produces:

- score-decile realized-return monotonicity
- top-tail realized returns for top 20%, 10%, 5%, 2%, 1%, top 10 names and
  top 5 names
- daily score-rank persistence at 1-session and 20-session lags
- top-5/10/20/50 membership overlap and turnover proxies
- pure-ML gross-versus-net portfolio decomposition

### ML + momentum ensemble

For every signal date:

```text
ensemble_score =
    alpha * percentile(ML score)
  + (1 - alpha) * percentile(60d momentum)
```

Default candidate weights are:

```text
alpha in {0.00, 0.25, 0.50, 0.75, 1.00}
```

`alpha=0` is pure 60-day momentum rank and `alpha=1` is pure ML rank.

The ensemble weight is selected **only on 2019-2021 validation data**. The
selection criterion is mean net Sharpe across top-5 and top-10 portfolios, with
mean net CAGR as the tie-breaker. The selected alpha is then frozen.

A separate, non-tuned confirmation construction is also evaluated: ML ranking
among eligible securities whose 60-day return is positive.

### Expanding walk-forward test

The frozen model family and validation-selected ensemble weight are evaluated
with annual expanding fits:

```text
2022 score <- train through 2021, purged 20 market sessions
2023 score <- train through 2022, purged 20 market sessions
2024 score <- train through 2023, purged 20 market sessions
2025 score <- train through 2024, purged 20 market sessions
2026 score <- train through 2025, purged 20 market sessions
```

Each fold is fit only from labels whose full 20-session target is known before
the fold starts. The ensemble alpha is never retuned in the walk-forward
period.

The combined out-of-sample scores are run through the same INR 50,000,
integer-share, next-open-entry, current-cost delivery simulator used by the
deterministic baselines. The final comparison includes:

```text
walk-forward pure ML
validation-selected ML + 60d momentum ensemble
ML with positive-60d-momentum confirmation
60d momentum
liquidity control
NIFTY 50 price index
NIFTY 500 price index
```

Both top-5 and top-10 portfolios are reported. Pure ML and the selected
ensemble also receive matched zero-cost/current-cost decompositions.

Outputs:

```text
reports/ml/portfolio_diagnostics/
├── validation_decile_daily.csv
├── validation_decile_summary.csv
├── validation_top_tail_daily.csv
├── validation_top_tail_summary.csv
├── validation_score_persistence_daily.csv
├── validation_score_persistence_summary.csv
├── validation_ensemble_search.csv
├── validation_gross_vs_net.csv
├── walkforward_fold_diagnostics.csv
├── walkforward_decile_daily.csv
├── walkforward_decile_summary.csv
├── walkforward_top_tail_daily.csv
├── walkforward_top_tail_summary.csv
├── walkforward_score_persistence_daily.csv
├── walkforward_score_persistence_summary.csv
├── walkforward_portfolio_comparison.csv
├── walkforward_gross_vs_net.csv
├── walkforward_predictions.parquet
├── diagnostics_summary.json
└── walkforward_models/
    └── *.joblib
```

For a computational smoke test only:

```bash
uv run python src/ml/portfolio_diagnostics.py \
  --max-train-rows 250000 \
  --max-walkforward-train-rows 400000
```

Capped-row runs are diagnostic only and should not be used as paper results.


## Quant performance dashboard and jargon glossary

Backtests now use a shared metrics layer in:

```text
src/backtest/quant_metrics.py
```

Every strategy run inherits a richer performance dashboard including, where
the required data exists:

```text
total return
CAGR
annualized arithmetic return
annualized volatility
Sharpe
Sortino
maximum drawdown
drawdown duration
high-water mark
Calmar
recovery factor
95% historical VaR
95% CVaR / expected shortfall
skewness
excess kurtosis
best/worst day
positive-day rate
win rate
average win/loss
payoff ratio
expectancy
profit factor
average trade return
transaction costs
turnover and annualized turnover
average/max exposure
```

Benchmark reports additionally compute aligned NIFTY-relative statistics:

```text
benchmark return/CAGR
excess CAGR
beta
CAPM alpha
tracking error
information ratio
benchmark correlation
```

ML reports retain ranking diagnostics such as Rank IC, ICIR, top-tail/top-K
returns and decile analysis. Rolling risk output includes rolling Sharpe and
drawdown series.

Each baseline strategy directory now writes:

```text
metrics.json
metrics_captioned.csv
trades.csv
equity_curve.csv
```

`metrics_captioned.csv` stores the machine metric name, readable label,
formatted value, one-line explanation, category and implementation status.

The complete jargon glossary can be printed without running any backtest:

```bash
uv run python src/backtest/show_glossary.py
```

or filtered:

```bash
uv run python src/backtest/show_glossary.py --category benchmark
uv run python src/backtest/show_glossary.py --category ml
uv run python src/backtest/show_glossary.py --category research_design
```

The glossary explicitly distinguishes `computed`, `diagnostic/report`,
`concept` and `not_yet_modeled` items. For example, MAE/MFE, separate
bid-ask-spread modeling, market impact and capacity are not assigned invented
numbers until the data/simulator supports them.

Sharpe and Sortino keep a zero annual risk-free rate by default for backward
comparability with earlier experiments. Baseline runs can override it:

```bash
uv run python src/backtest/baselines.py --risk-free-rate 0.065
```

where `0.065` means 6.5% annualized.

The benchmark report also writes:

```text
reports/benchmarks/
├── benchmark_relative_metrics.csv
├── rolling_risk_metrics.csv
├── metric_glossary.csv
└── captioned_metrics_long.csv
```


## Nested generalization-first persistence experiment

The single-year 2021 persistence search is useful diagnostically, but its very
large validation returns do not establish generalization. The primary
persistence experiment therefore uses annual out-of-sample model predictions
and nested policy selection across multiple market years and multiple portfolio
breadths.

Run:

```bash
uv run python src/ml/nested_generalization.py --self-test
uv run python src/ml/nested_generalization.py
```

### Annual base-model OOS folds

By default the HistGradientBoosting ranker produces one genuinely annual OOS
score set for every year from 2016 through 2026:

```text
2016 scores <- train through 2015, with 20-session label purge
2017 scores <- train through 2016, with 20-session label purge
...
2026 scores <- train through 2025, with 20-session label purge
```

Predictions are cached by model family, random seed and year so interrupted or
follow-up analyses do not need to refit unchanged annual folds.

### Coarse persistence surface

The portfolio-construction search is intentionally small:

```text
top-K                 = {5, 10, 15, 20}
hold-band multiplier  = {1x, 2x, 3x}
score-gap hurdle      = {0.00, 0.05, 0.10}
```

Both current-cost and zero-cost surfaces are generated. Each fixed rule is run
continuously over the complete OOS score history and then decomposed into
calendar-year performance. This avoids isolated one-year backtests whose forced
year-end resets could distort turnover or holding persistence.

### Universal nested rule

For evaluation year Y, only complete OOS years strictly before Y are eligible
for persistence-rule selection. The selector aggregates each
`hold_multiplier + score_gap` rule across **all four K values and all prior
years together**.

The robust ordering is:

```text
1. higher 25th-percentile annual Sharpe
2. higher median annual Sharpe
3. higher fraction of positive-CAGR observations
4. higher 25th-percentile annual CAGR
5. higher median annual CAGR
6. lower median turnover
```

This deliberately favors rules that remain acceptable in weaker years and at
different portfolio breadths rather than whichever rule produced the single
largest historical CAGR.

The first default evaluation year is 2019, so its policy can use only the
2016-2018 OOS history. The selected universal rule is updated annually using
only prior complete OOS years.

### Controls

The executable 2019-2026 nested evaluation compares:

```text
nested_dynamic
    universal rule re-selected annually from prior OOS years only

frozen_pre_eval
    one universal rule selected from pre-2019 OOS years and never changed

buffer_only
    hold an incumbent only while it remains inside the fresh top-K

cohort_ml
    frozen 20-session liquidate-and-rebuild ML portfolio

momentum_60d
    deterministic medium-term momentum control

NIFTY 50 / NIFTY 500
    official NSE price-index benchmarks
```

Active strategies are reported both gross and net of the current delivery-cost
model. The final output therefore separates model/portfolio quality from
friction sensitivity.

### Outputs

```text
reports/ml/nested_generalization/
├── annual_base_model_diagnostics.csv
├── annual_config_surface_net.csv
├── annual_config_surface_gross.csv
├── nested_policy_by_year.csv
├── nested_policy_candidate_scores.csv
├── policy_stability.csv
├── final_generalization_comparison.csv
├── annual_generalization_metrics.csv
├── aggregate_generalization.csv
├── gross_net_decomposition.csv
├── final_generalization_captioned.csv
├── nested_generalization_summary.json
└── predictions/
    └── <model>_rs<seed>/
        ├── 2016.parquet
        ├── ...
        └── 2026.parquet
```

`aggregate_generalization.csv` excludes partial calendar years and reports
the median and lower-quartile CAGR/Sharpe, positive-year fractions, worst
drawdown and median turnover. This is the main robustness table; terminal CAGR
alone is not treated as sufficient evidence of generalization.

The 2022-2026 interval has already been inspected during development, so this
is correctly described as nested walk-forward generalization analysis rather
than a pristine never-seen final holdout.


## Multi-axis robustness cube

After the nested generalization experiment, the persistence rule is no longer
retuned. The robustness battery freezes the simplest construction:

```text
buy fresh top-K
keep an incumbent while rank <= K
replace after it leaves K
score-gap hurdle = 0
```

and asks whether that same rule survives changes in the environment rather than
searching for another historical optimum.

Run:

```bash
uv run python src/ml/robustness_cube.py --self-test
uv run python src/ml/robustness_cube.py
```

The default suite evaluates every stress across:

```text
K = {5, 10, 15, 20}
evaluation = aligned annual OOS predictions, 2019-2026
base model = HistGradientBoosting
base seed = 42
base capital = INR 50,000
base liquidity floor = INR 5,000,000 trailing-20d median turnover
risk-free rate = 6.5%
```

### Independent robustness axes

Capital:

```text
INR 25k
INR 50k
INR 100k
INR 250k
INR 500k
```

Cost stress:

```text
0.0x
0.5x
1.0x
1.5x
2.0x
```

The friction multiplier scales STT, stamp duty, SEBI/exchange charges,
brokerage, DP charges and slippage. GST remains the statutory percentage; its
taxable base is what scales.

Liquidity / execution-universe stress:

```text
INR 5M
INR 10M
INR 25M
INR 50M
trailing-20d median turnover
```

This is deliberately an inference/execution-universe stress. It does not
retroactively retrain the model on a newly invented universe.

Model-specification robustness:

```text
Ridge
HistGradientBoosting with the original internal early stopping
HistGradientBoosting with fixed 250 iterations and early_stopping=False
```

The fixed-iteration HistGB variant exists specifically to remove the random
internal validation fraction from the model specification while preserving the
same tree depth, learning rate, leaf-size and L2 settings.

Random-seed robustness for the base HistGB:

```text
7
19
42
123
2026
```

The seed is never selected by performance; the distribution is reported.

A small joint-stress set combines lower capital, higher friction and stricter
liquidity without evaluating the full Cartesian product.

### Generalization-first reporting

Every run is retained per K, but each stress setting is also summarized over
all complete calendar years and all K values together. The main family-level
statistics are:

```text
median annual CAGR
25th-percentile annual CAGR
worst annual CAGR
positive-CAGR fraction

median annual Sharpe
25th-percentile annual Sharpe
positive-Sharpe fraction

worst annual drawdown
median annual turnover

median terminal CAGR across K
minimum terminal CAGR across K

failed-entry count
average exposure
```

The purpose is to reject a result that survives only one breadth or one market
year even when its terminal CAGR looks attractive.

### Statistical uncertainty

For the frozen base persistence family `K={5,10,15,20}`, the suite aligns
daily strategy returns with the NIFTY 500 price index and runs a 20-session
moving-block bootstrap by default.

It reports 95% bootstrap intervals for:

```text
CAGR
Sharpe
maximum drawdown
CAGR minus NIFTY 500 CAGR
```

It also runs a joint max-statistic test on active daily returns. The four K
active-return series are centered under the null and block-resampled jointly,
preserving cross-K dependence. The resulting familywise p-value controls the
choice of the best K **inside this four-breadth frozen persistence family**.

It does not claim to correct for every strategy, model or exploratory decision
ever examined during the project.

### Outputs

```text
reports/ml/robustness_cube/
├── robustness_terminal_runs.csv
├── robustness_annual_runs.csv
├── robustness_axis_summary.csv
├── model_seed_oos_diagnostics.csv
├── benchmark_reference.csv
├── bootstrap_aligned_returns.csv
├── bootstrap_confidence_intervals.csv
├── familywise_block_bootstrap_test.json
├── robustness_captioned_metrics.csv
└── robustness_cube_summary.json
```

Prediction files are reused from the nested-generalization cache where model,
seed and year match. Additional Ridge, fixed-HistGB or seed variants are cached
once generated.

The default command runs every axis. Sections can be run separately without
changing the methodology:

```bash
uv run python src/ml/robustness_cube.py --sections capital cost liquidity
uv run python src/ml/robustness_cube.py --sections model seed
uv run python src/ml/robustness_cube.py --sections joint bootstrap
```


## India model tournament

The next research stage is a fixed-protocol model tournament. The target is not
merely positive returns:

```text
economic target:
    net CAGR - NIFTY 500 CAGR >= 5 percentage points
    Sharpe > 1

factor-adjusted target:
    CAPM-like annual alpha >= 5%
    Sharpe > 1
```

The portfolio/execution protocol is frozen across models:

```text
annual expanding walk-forward fit
20-session label purge
cross-sectional rank target
buy fresh top-K
keep incumbent while rank <= K
gap = 0
next-open execution
20-session rebalance cadence
same Indian delivery-equity cost model
K = {5, 10, 15, 20}
```

### Tabular nonlinear tier

Install:

```bash
uv sync --extra tournament
```

Run:

```bash
uv run python src/ml/model_tournament.py --self-test
uv run python src/ml/model_tournament.py
```

Default fixed configurations:

```text
HistGradientBoosting
HistGradientBoosting, fixed iterations / no internal early stopping
ExtraTrees
XGBoost
LightGBM
CatBoost
```

A predeclared equal-rank tree consensus is also evaluated when at least three of:

```text
HistGB-fixed
ExtraTrees
XGBoost
LightGBM
CatBoost
```

complete successfully. Ensemble weights are not tuned on the evaluation period.

Outputs:

```text
reports/ml/model_tournament/
├── leaderboard.csv
├── annual_portfolio_metrics.csv
├── annual_prediction_diagnostics.csv
├── prediction_summary.csv
├── equity_curves.parquet
├── benchmark_reference.csv
├── tree_consensus_score_correlation.csv
├── failures.csv
└── tournament_summary.json
```

### Temporal nonlinear tier

Install:

```bash
uv sync --extra deep
```

Run the self-test:

```bash
uv run python src/ml/temporal_tournament.py --self-test
```

Then run one or more sequence models:

```bash
uv run python src/ml/temporal_tournament.py \
  --models seq_mlp lstm gru tcn transformer patch_transformer itransformer nhits_style
```

Every temporal sample is a leakage-safe trailing sequence for one security.
The default lookback is 60 market sessions, and a sequence is accepted only if
those sessions are contiguous in the market-day index. Each annual fold fits
only labels that have matured before that fold.

The temporal implementations are supervised rank regressors:

```text
seq_mlp             flattened sequence MLP
lstm                LSTM sequence encoder
gru                 GRU sequence encoder
tcn                 dilated temporal convolution encoder
transformer         temporal-token Transformer
patch_transformer   patch-token Transformer ranker
itransformer        variate-token Transformer ranker
nhits_style         N-HiTS-inspired multiresolution pooling/MLP ranker
mamba               optional mamba_ssm selective state-space ranker
```

`nhits_style`, `patch_transformer`, and `itransformer` are adaptations to
the project's supervised cross-sectional rank target; they are not claimed to
be drop-in reproductions of the original multi-step forecasting tasks.

Mamba is isolated because `mamba-ssm` is source/CUDA dependent:

```bash
uv sync --extra deep --extra mamba
uv run python src/ml/temporal_tournament.py --models mamba
```

Temporal predictions are cached by model, seed, lookback, epoch count and
training-sequence cap.


### Boosting complementarity and selective ensemble

After the architecture tournament, the strongest observed tabular families can be
studied without refitting them. This experiment is intentionally cache-first and
focuses on XGBoost, LightGBM, and CatBoost.

It measures:

```text
daily cross-sectional Spearman rank correlation
top-K overlap and Jaccard overlap
standalone portfolio-return correlation
coincident 10% / 20% drawdown periods
annual winners by K
```

It then evaluates only predeclared equal-rank signals:

```text
XGBoost
LightGBM
CatBoost
XGBoost + LightGBM
XGBoost + CatBoost
LightGBM + CatBoost
XGBoost + LightGBM + CatBoost
```

No ensemble weights are fitted. Each ensemble is the equal-weight mean of the
members' within-date percentile ranks, and every signal is evaluated at the same
K = {5, 10, 15, 20} with the frozen persistent execution/backtest protocol.

Run:

```bash
uv run python src/ml/boosting_ensemble.py --self-test
uv run python src/ml/boosting_ensemble.py
```

By default the script refuses to refit missing annual prediction caches. Use
`--allow-fit` only when CPU refitting is intentional.

Outputs:

```text
reports/ml/boosting_ensemble/
├── leaderboard.csv
├── annual_metrics.csv
├── annual_prediction_diagnostics.csv
├── daily_score_correlations.csv
├── score_correlation_summary.csv
├── daily_topk_overlap.csv
├── topk_overlap_summary.csv
├── portfolio_pair_diagnostics.csv
├── portfolio_return_correlation.csv
├── annual_winners_by_k.csv
├── equity_curves.parquet
├── benchmark_reference.csv
└── summary.json
```

This stage is exploratory because the choice to focus on these three boosting
models was informed by previously observed OOS tournament results. Any promising
ensemble must later be validated with nested selection or an independent test.

### SHAP / nonlinear interaction stability

SHAP is an explanation stage, not a predictive model.

After choosing a tree model worth interpreting:

```bash
uv run python src/ml/explain_tournament.py --model hist_gb_fixed
```

or, for example:

```bash
uv run python src/ml/explain_tournament.py --model xgboost
```

The explanation model is refit separately for each annual fold. The report
tracks mean absolute SHAP value, signed SHAP value, feature rank stability and,
where TreeExplainer supports it, pairwise SHAP-interaction strength.

Outputs:

```text
reports/ml/model_tournament/explanations/<model>/
├── annual_feature_shap.csv
├── feature_stability.csv
├── annual_interactions.csv
├── interaction_stability.csv
└── explanation_summary.json
```

The intended progression is:

```text
tabular nonlinear tournament
        ↓
temporal nonlinear tournament
        ↓
predeclared rank ensembles
        ↓
SHAP / interaction stability on credible models
        ↓
only then architecture/hyperparameter refinement
```

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


## Leakage-audited feature engineering reset

The next research phase treats the old model tournament as a baseline, not as a
finished predictor. The information boundary is frozen as:

```text
observe all information through close(t)
        ↓
score / select after close(t)
        ↓
earliest execution = open(t+1)
```

No feature may depend on `open(t+1)`, future corporate actions, future index
membership, future sector labels, or a present-day security snapshot copied
backward through history.

The feature registry is in:

```text
src/features/feature_registry.py
```

The predeclared feature families are:

```text
F0   frozen baseline features
F1   OHLC geometry
F2   streak / discrete price state
F3   stock-specific historical tendencies
F4   broad-market context
F5   point-in-time large-cap / market-cap context
F6   market→stock lead-lag and response residuals
F8   same-close cross-sectional ranks
F9   recurring calendar / seasonality
F10  liquidity and volatility state
```

Absolute calendar year is kept for attribution and diagnostics, not as a default
model feature.

### Build the enriched panel

NIFTY 100 is now included in the default historical-index download. Rebuild the
index panel first if the local copy predates that change:

```bash
uv run python src/data/index_benchmarks.py
```

Run the feature leakage self-test:

```bash
uv run python src/features/build_enriched_panel.py --self-test
```

Then build the enriched daily panel:

```bash
uv run python src/features/build_enriched_panel.py --overwrite
```

Outputs include:

```text
data/processed/research_panel_enriched/
reports/features/feature_registry.json
reports/features/enriched_panel_metadata.json
```

### Point-in-time large-cap context

The current 2026 security master must not be used to label historical rows as
large-cap, sector or industry members. Historical context is ingested only from
dated snapshots.

Place dated constituent/context snapshots under:

```text
data/reference/security_context_snapshots/
```

Each file must contain an `asof_date` column or a YYYY-MM-DD date in the
filename, plus either `canonical_security_id` or ISIN. Optional metadata
includes sector, industry, sector index and market capitalization.

Build the context table with:

```bash
uv run python src/features/build_security_context.py
```

The importer explicitly writes false membership rows when a security disappears
from a later large-cap snapshot, so old membership cannot silently persist.

After context is available, rebuild the enriched panel.

### Feature-family ablation protocol

Development is intentionally separated from the later evaluation block.

```text
development OOS: 2019–2023
evaluation OOS:  2024–2026
```

Development runs cumulative, predeclared feature sets only:

```bash
uv run python src/ml/feature_ablation.py \
  --stage development \
  --universe largecap \
  --models xgboost
```

The evaluation stage refuses to run an unspecified feature search. Freeze one
development feature set first, then evaluate exactly that set:

```bash
uv run python src/ml/feature_ablation.py \
  --stage evaluation \
  --universe largecap \
  --feature-set 'F0+F1+F2+F3+F4+F6' \
  --models xgboost
```

Do not repeatedly alter the feature set after inspecting the evaluation block
without declaring a new research cycle.

### Strategy attribution

`src/ml/strategy_attribution.py` rebuilds the actual persistent portfolio from
saved annual prediction caches and writes:

```text
top-K signal ledger
actual trade ledger
rebalance ledger
equity curve
P&L by stock
P&L by sector
P&L by industry
P&L by year
P&L by month
P&L by weekday
```

This separates model explainability from economic attribution: SHAP can explain
why a model produced a score, while the attribution ledger explains where the
portfolio actually made or lost money.

