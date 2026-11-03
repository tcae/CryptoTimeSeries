# Streaming RegimeTraining and refresh: commit summary

Base commit: `2c94793`  
Scope: tracked working-tree changes and the new sets-store regression test since that commit.  
Date: 2026-11-03

## Suggested commit message

```text
Stream regime training inputs and refresh eligibility per coin

- Build compact mmap features from lazy per-coin store loaders.
- Release preparation tables before compaction and reload held-out inputs on demand.
- Skip insufficient-history coins with explicit diagnostics.
- Rebuild requested partition stores and report empty assignments.
- Evaluate multiple strategy configurations and report common regime/fold eligibility.
- Stream refresh preparation, disk-back eligible timestamps and retain only counts.
- Preserve global progress counters during single-coin store updates.
- Add CS02 and updated SimpleChains/S09 strategy registrations.
- Add regression tests for rebuilds, streaming equivalence and eligibility boundaries.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

## Changes

### Compact training sources and lifecycle

- [Classify SampleSource](../Classify/src/SampleSource.jl) adds a lazy-loader
  `CompactMmapSource` constructor. It loads one coin's feature table, copies the
  selected rows directly into the final Float32 mmap, releases source references,
  and collects garbage before loading the next coin.
- The lazy path requires coin-grouped indices and preserves feature order, sample
  order, UInt8 label storage, class balancing and existing batch behavior.
  It avoids intermediate per-coin matrices and a cross-coin concatenation copy.
- [RegimeTraining](../scripts/RegimeTraining.jl) releases histories, shared-store
  windows, experiment feature/target tables and preparation calendars before
  streaming compaction in the pooled CLI path. Train indices are released after
  compaction; held-out test indices are persisted and reloaded on demand.
- Evaluation predictions use compact source batches through
  [RegimePipeline](../TradingStrategy/src/RegimePipeline.jl), rather than requiring
  all feature stores to remain loaded. Test/gain prediction sources also load
  features per coin, and gain replay reloads individual histories.
- Walk-forward compaction uses the lazy writer but retains histories needed for
  chronological memberships and reloads experiment stores between windows.
  It is not an entirely disk-streamed walk-forward preparation pipeline.

### History eligibility and shared preparation

- Shared helpers implement the regime warmup calculation, experiment-window
  selection, aligned shared-store slicing, target-tail purging, and regime/fold
  assignments.
- Coins with insufficient window/context rows, such as CAMP with 26,679 rows
  against a 28,800-row warmup, are warned about and skipped before store refresh.
- Missing/corrupt inputs and alignment errors still fail explicitly. No usable
  histories, calendar blocks or required training populations remain fatal.
- Full-history shared feature/target stores remain distinct from experiment
  windows; window slicing does not overwrite reusable shared stores.

### Partition refresh and eligibility reporting

- [SetsStore](../Classify/src/SetsStore.jl) adds explicit `rebuild=true` support.
  Normal reuse and hash checks remain unchanged; explicit rebuild replaces the
  existing store only after a replacement has been generated and written.
- [Refresh](../scripts/refresh.jl) accepts partition/date-range lists, rebuilds
  requested sets stores, verifies their shape and timestamps, and reports failures
  and successfully rebuilt empty stores separately.
- Partition output prints copy-pasteable coin lists with train/eval/test
  assignments. These are explicitly distinct from RegimeTraining eligibility.
- `ts=062,066,067` requests full eligibility preparation for each selected
  classifier and date range, using Bybit USDT histories and default detector/folds.
  Per-config counts and exclusions precede a common coin list requiring all three
  populations under every selected config. No classifier is trained.
- Refresh processes source histories and feature/target tables one coin at a time.
  Eligible minute timestamps are stored in temporary Arrow files and reloaded
  per coin for shared fold balancing and final membership counts.
- Only compact hourly observations are pooled for volume-capped cross-coin market
  calculations. After calendar construction they are released; only three counts
  per coin are retained across configurations. Temporary files are cleaned up on
  completion or exception.
- [RegimeCalendar](../TradingStrategy/src/RegimeCalendar.jl) adds observation-based
  market construction, timestamp-loader fold construction, shared block balancing,
  and membership counting without allocating minute-level projected tables.
- [ArtifactPipeline](../TradingStrategy/src/artifactpipeline.jl) accepts progress
  offsets/totals so single-coin updates display their full-universe position,
  instead of always `1/1`. Skipped coins remain part of the denominator.
- [README](../README.md) documents commands, eligibility distinctions and memory
  lifecycle constraints.

### Registry and package versions

- [Strategy registry](../TradingStrategy/src/tradingstrategyconfig.jl) adds the
  expanded `traincoins_v2()` universe as CS02 and makes it the non-test default.
  CS00 and CS01 remain registered.
- Removes strategy registrations 063/064 and adds 066/067/068 using SimpleChains
  M011 with S09. Existing 062/065 remain registered.
- Classify version changes from 0.38.0 to 0.39.1; TradingStrategy changes from
  0.23.0 to 0.23.4.

## Validation performed

- Classify package tests passed earlier in this change sequence, including the
  new explicit sets-store rebuild tests. Focused classifier-store tests passed:
  51 store/source tests and 19 inference regressions.
- TradingStrategy focused calendar/pipeline tests passed: 4,355 calendar checks,
  16 history-boundary checks and 27 experiment-pipeline checks.
- The final refresh/script-help suite passed, including 18 full-preparation/progress
  checks and exact streamed-versus-pooled sample-count equivalence with nonempty
  train/eval/test populations.
- Tests cover strict multi-config intersections, invalid configs, source batch
  equivalence, warmup boundaries, stricter feature/target history requirements,
  and collection of loaded timestamp vectors before loading the next coin.
- `git diff --check` passed during implementation.
- The last RegimeTraining integration run had 109 passes and one failure from the
  stale CS01 expectation after the default changed to CS02. Later removal of
  063/064 also leaves RegimeTraining's default `ts=063` and some existing tests/help
  stale. These registry/default mismatches have not been fixed in this change sequence.
- No live market-run peak-memory measurement or final full-workspace test run was
  performed. Source retention is reduced, but Julia allocator retention and
  OS-managed mmap pages can still contribute to reported process footprint.

## Artifact and reproducibility implications

- Changing a coin universe changes classifier identity; do not reuse classifiers
  trained against a different universe. Merely changing a coin list does not alter
  feature, target or partition configuration hashes.
- Explicit partition rebuild can change memberships under an existing config ID.
  Preserve old results or use new IDs/run identities for reproducible comparisons.
- Regime/fold coverage depends on the evaluated universe. Copying the common
  eligibility list into a coin-set configuration can change market regimes and
  therefore needs a fresh coverage check.
