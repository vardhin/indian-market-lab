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
