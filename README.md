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
