# Regime refresh performance: commit summary

- Base commit: `0269609`
- Scope: working-tree changes since that commit.
- Date: 2026-11-04

## Suggested commit message

```text
Optimize Trend05 scans and stream regime observations

- Read cached Trend05 regression vectors directly during target scans.
- Replace eligibility and market timestamp dictionaries with sorted walks.
- Append hourly observations to a temporary Arrow file for mmap-backed aggregation.
- Preserve full-universe market weighting and cutoff-aware sample selection.
- Test target deadlines, join equivalence, observation lifetimes and fold counts.
- Document the refresh storage lifecycle and bump TradingStrategy to 0.23.5.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

## Changes

### Trend05 target generation

- [Targets](../Targets/src/Targets.jl) resolves the lead, support and trend
  regression columns and timestamps once before scanning. Candidate entries use
  these vectors directly; forward hold checks read only the support gradient.
- The target scan retains elapsed-time deadlines, uninterrupted holds,
  unresolved decisions, timestamp offsets and incremental append behavior.
- [Trend05 tests](../Targets/test/trend05_test.jl) adapt to the vector-based
  helpers and add numeric support-gradient checks.

### Eligibility and market calendar

- [RegimePipeline](../TradingStrategy/src/RegimePipeline.jl) replaces per-coin
  feature and target timestamp dictionaries with a typed, sorted walk over
  history, feature and target timestamps. Feature finiteness is calculated
  column-wise and asserted for eligible joined rows; liquidity is still
  recomputed from the chronological prefix when a cutoff is supplied.
- [RegimeCalendar](../TradingStrategy/src/RegimeCalendar.jl) uses one sorted
  cursor per coin instead of observation timestamp dictionaries. It retains
  full-universe, volume-capped market aggregation and the existing regime logic.
- History, feature, target and observation timestamps must now be strictly
  increasing and unique. Duplicate or unsorted inputs fail explicitly rather
  than being handled by a dictionary.

### Disk-backed observations

- [Refresh](../scripts/refresh.jl) appends each accepted coin's hourly
  observations to one temporary Arrow file under the debug directory. It records
  per-coin row ranges, releases the source DataFrame, and maps the combined file
  without copying columns when constructing the shared market calendar.
- The mapped observations are released after calendar construction. Existing
  per-coin eligible timestamp files remain available through fold balancing
  and counting; an empty timestamp file gives zero counts, while a missing
  file remains an error. Temporary files are cleaned up when preparation
  completes or throws.
- [README](../README.md) describes the storage and timestamp lifetime.
  [TradingStrategy's package version](../TradingStrategy/Project.toml) changes
  from `0.23.4` to `0.23.5`.
- Shared full-history feature, target and regression stores remain in use;
  generating them on demand is not part of this change.

## Validation performed

The following focused runs passed during this change sequence:

- Targets: 254 assertions, including the Trend04 cross-check fixture and
  Trend05 deadline, hold and incremental-label tests.
- TradingStrategy: 4,430 assertions across appended observations, market and
  fold contracts, sorted eligibility joins, history boundaries and synthetic
  experiment preparation.
- Script integration: 162 assertions, including a two-coin comparison of
  streamed and pooled market results and train/eval/test counts. Tests also
  check source DataFrame collection, timestamp-file lifetime, empty files and
  missing-file errors.

The new tests compare the sorted implementations against the previous
dictionary-based algorithms, including gaps, cutoffs, ignored labels, excluded
non-finite features, sparse or off-grid observations and retrospective modes.
No full workspace test suite or end-to-end live refresh benchmark was run.

## Bounded performance measurements

Median of five candidate scans on recent 2,000-row slices, with old and new
results compared:

| Coin | Old scan | New scan | Old allocations | New allocations |
|---|---:|---:|---:|---:|
| POKT | 86.023 ms | 0.079 ms | 23.96 MiB | 0.44 KiB |
| POL | 37.802 ms | 0.044 ms | 11.46 MiB | 0.44 KiB |
| PONKE | 86.776 ms | 0.084 ms | 24.98 MiB | 0.44 KiB |

For one synthetic coin with 60,000 history rows and two numeric feature columns,
the complete eligibility calculation, including liquidity, took a median of
78.84 ms and allocated 71.99 MiB with dictionaries, versus 14.45 ms and
11.12 MiB with the sorted walk. Results were compared exactly over five runs.

These are component measurements, not full-refresh speedups or peak-memory
measurements. Mapped file pages and Julia's allocator can still contribute to
resident memory. Julia processes started before these source changes continue
using their previously loaded code.
