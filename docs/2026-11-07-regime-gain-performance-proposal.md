# Regime gain-stage performance assessment and optimization proposal

Run: `rtrain train ts=068 stage=gain runid=F09-T09-P02-M011-B1-CS02-D01-dfe00fd9a00e`.

Assessed November 7, 2026, using the running process, artifact timestamps,
source inspection and bounded, warmed in-memory benchmarks. No implementation
changes or interruption of the running job were made for this assessment.

## Findings

This is not just an inference job. The gain invocation currently repeats dataset
preparation, creates training and evaluation sources, creates a test source,
predicts test samples, then runs minute-by-minute trading simulation twice:
once with predicted labels and once with target labels. Finally it compiles and
reports completed trades.

### Observed resource use and phase boundaries

PID 33415 started at 09:07:20. At 10:45 it had run for about 98 minutes,
consumed about 134 CPU-minutes, and was using approximately 132% CPU.
A three-second macOS process sample reported a physical footprint of `7.4G`
and a peak of `15.1G`. The host has 16 GiB RAM. RSS was about 5.8 GiB,
subsequently about 6.3 GiB. RSS and physical footprint are different measures;
neither is the same as cumulative allocation.

System swap usage was about 428 MiB. System compression and swap counters are
cumulative and shared by other processes; they do not establish how much
slowdown this job alone caused.

Approximate boundaries from output timestamps:

| Phase | Evidence | Elapsed contribution |
|---|---|---|
| Dataset preparation | Market/calendar saved 09:25; coverage saved 09:31:35 | About 24 minutes from launch |
| Index construction and identity verification | Train/eval indices published 10:08:32-34 | About 37 additional minutes |
| Train/eval/test compaction, inference and prediction persistence | Test predictions saved 10:17:48 | About 9 additional minutes |
| Trading replay and per-coin output | BTC saved 10:24, OM 10:41; further coins saved 10:45-46 | Still running at inspection |

These are file-completion boundaries, not instrumented function timings.
In particular, the nine-minute interval does not isolate inference from
compaction, identity work and serialization. However, approximately 70 minutes
elapsed **before test predictions were available**.

At a later inspection, 269 predicted per-coin files existed. Output continued
to advance, so the process was making progress rather than hanging.

### Actual workload

The compact label files contain one byte per sample:

| Population | Samples | Feature matrix bytes |
|---|---:|---:|
| Train | 97,647,768 | 2,734,137,504 |
| Eval | 24,275,226 | 679,706,328 |
| Test | 24,277,651 | 679,774,228 |

F09 has seven Float32 inputs, hence 28 feature bytes per sample. The test
population requires **48,555,302 simulated rows** for predicted and truth passes.
The train/eval index files alone occupy approximately 8.71 GB on disk, much
larger than their numeric feature matrices.

### Root causes

1. **Gain-only execution still takes the training preparation path.**
   [main](../scripts/RegimeTraining.jl) always calls `prepare!`; market preparation
   rechecks histories and stores, recalculates eligibility and observations,
   rebuilds the universe-wide calendar and memberships, and writes coverage.
   Existing private feature/target tables can be reused, but that does not
   bypass the surrounding preparation work.

2. **Gain unnecessarily constructs train and eval compact sources.**
   [execute! and trainingsources](../scripts/RegimeTraining.jl) construct both
   sources before branching on `stage`. This includes regenerating all three
   indices, verifying training identity, calculating eval weights and copying
   training/evaluation features. Their mmap sources remain referenced while
   gain reporting creates and consumes a third source for test.
   Mmap files are not entirely resident by definition, but touching them adds
   page pressure and unnecessary I/O.

3. **Regression guidance is allocation-heavy inside every simulated row.**
   [gain_limit_reversal_below_regression!](../TradingStrategy/src/TradingStrategy.jl)
   evaluates both regression helpers before checking whether the label is an
   open signal. S09 uses 240 minutes for both gates, yet each helper resolves
   the base and looks up that regression separately.
   [regressionat](../Features/src/Features.jl) repeatedly constructs feature
   descriptors/column names, allocates `names` results, locates a timestamp and
   retrieves dynamically typed DataFrame cells. The classifier's per-base
   dictionary also exposes an abstract NamedTuple value.

   Warmed, 20,000-operation synthetic F09/S09 benchmarks measured:

   | Operation | Allocated bytes per call |
   |---|---:|
   | `Features.regressionat` | 2,672 |
   | Distance helper | 4,160 |
   | Slope helper | 4,192 |
   | Below-regression advice, resetting a short-open label each iteration | 8,352 |

   A second benchmark consumed changing sample values and returned identical
   checksums: current regression reads allocated 55,130,560 bytes over 20,000
   reads, versus 16 bytes total for pre-resolved, typed timestamp/price/slope
   vectors. The latter is an access-pattern control, not a complete replacement
   algorithm or an end-to-end speedup claim.

   Extrapolating the 8,352-byte gate measurement to both full passes gives
   **approximately 405.5 GB of transient allocations**. This is not measured
   whole-run allocation and is not retained heap size, but it explains
   substantial allocation/GC pressure. The live sample also showed Float32
   boxing and garbage collection; Julia JIT frames were mostly unsymbolized,
   so that sample cannot attribute all runtime to a particular Julia function.

4. **Replay reconstructs features that inference has already consumed.**
   [gainreport!](../scripts/RegimeTraining.jl) loads and copies a full coin
   history, then attaches a runtime classifier.
   [TrendClassifier001.addbase!](../Classify/src/TrendClassifier001.jl) initializes
   the complete F09 feature configuration.
   [Features.read! and supplement!](../Features/src/Features.jl) copy regression
   columns and materialize feature arrays, including distance and relative-volume
   features. Replay already has positive scores and labels; S09 only needs its
   240-minute regression endpoint and gradient. It does not need a second F09
   feature matrix for neural-network inference.

5. **Global tables and per-coin copies coexist.**
   Gain retains the full test index and predictions, repeatedly scans the full
   predictions table with `predictions.coin .== coin`, joins each coin against
   history, builds full-schema Trades frames per range, retains snapshot parts
   and concatenates them before saving and compiling gains.
   This adds copies and allocation traffic beyond the model's small batch arrays.

These differences explain why gain can consume more memory than adaptation:
adaptation deliberately releases preparation/index tables and executes bounded
numeric batches, while gain combines metadata, histories, regenerated features,
mutable simulation frames and output assembly. The measured peak cannot be
assigned to one phase without additional instrumentation; this is evidence of
excess working sets and allocation, not proof of an unbounded memory leak.

## Recommended implementation order

### 1. Make gain/test stage setup independent of training sources

Introduce a narrowly scoped replay-source setup in the runner:

- Load the checkpoint through the existing verified manifest/checkpoint contract.
- Verify the saved population/configuration identity and selected folds/regimes.
- Load only the required test membership and feature data.
- Do not build train/eval compact files or recompute eval weights for gain.
- Reuse prepared calendar, memberships and indices only under verified dependency
  identities. `runid` or the existence of a file alone is insufficient.
- Separate an immutable saved-experiment replay from refreshing current mutable
  market inputs. Replaying current inputs must still check their content identity;
  a mismatch must fail explicitly, not silently reuse stale preparation.
- Add a prediction identity covering checkpoint digest, test membership,
  feature content/order and label order. Reuse predictions across strategy-only
  reruns when it matches; strategy settings belong to the replay identity.
- Publish cache completion atomically and fail clearly for incomplete or
  mismatched requested artifacts. Do not add a silent compatibility fallback.

This is the largest observed avoidable wall-time opportunity: it addresses the
roughly 70-minute pre-replay path, although verification and first-time test
prediction still have a real cost.

### 2. Resolve regression inputs outside the row loop

Use the existing typed-column/function-barrier pattern established by
[TSM.TradesColumns and the replay loop](../TradingStrategy/src/TradingStrategy.jl):

- Resolve the coin configuration, enabled windows and typed timestamp/endpoint/
  gradient columns once per coin or replay range.
- Validate source types, sorted timestamps and exact required timestamp coverage
  during setup. Never replace missing coverage with the nearest sample.
- Align regression rows by a sorted cursor or exact timestamp join.
- Read a shared window once when trigger and trend windows match.
- Evaluate numeric gates only for open/strong-open labels. Validate configuration
  and required coverage outside the loop so skipping hold-label arithmetic does
  not hide configuration errors.
- Preserve nonpositive/non-finite endpoint rejection, matching hold labels,
  disabled-window behavior and all existing close/stop/reversal management.
- Keep live and replay gate arithmetic shared; do not create a divergent strategy.

Target warmed gate overhead of at most 128 allocated bytes per row, preferably
zero. This target concerns the guidance path, not every output-producing
simulation operation.

### 3. Bound replay data to one coin and one range

- Group predictions once or persist per-coin prediction batches rather than
  rescanning the whole population for every coin.
- On a fresh prediction pass, infer and persist coin/batch-sized results using
  the existing 8,192-sample batching rather than retaining all predictions.
- Load only high/low/close and gate regression guidance for replay, with validated
  context. Do not initialize the full classifier feature pipeline again.
- Reuse canonical regression stores read-only when identity and exact coverage
  match. Missing guidance must be explicit; any calculation from partial history
  must remain private and must never overwrite shared full-history stores.
- Retain the original shared calendar and memberships. Independent per-coin
  regime/fold reconstruction would change the evaluated population.
- Compile gains per independent `(pair, set, rangeid)` replay partition and append
  snapshot batches to the existing Arrow output rather than retaining `parts`
  and making a second full-coin `vcat` copy.
- Release coin/range buffers at clear scope boundaries. Preserve output schema,
  canonical final ordering and dropping unfinished positions at range ends.

Inspect the currently unused `processreplaygains!` return only as a secondary
optimization: gain materialization also drives position transitions, so it
cannot simply be removed because the caller discards its returned DataFrame.

### 4. Add progress, measurements and verified replay checkpoints

Report phase elapsed time, samples/sec, coin/pass/range progress and periodic
memory/allocation/GC measurements. Forward the existing prediction progress
callback. Distinguish preparation, verification, compaction, inference, guidance
setup, simulation, serialization and final reporting.

Record completed per-coin/pass outputs against checkpoint, population, strategy
and replay-semantics identities. Permit skipping only complete, verified outputs
on retry; incomplete or mismatched outputs must not count as success.

Do not start by increasing batch size, adding workers or forcing frequent GC.
Those do not remove redundant work and may worsen peak memory. Consider bounded
parallel coin replay only after the serial working set is measured and controlled.

## Validation and acceptance criteria

- Use one frozen checkpoint and the exact same full-universe test membership.
- Compare prediction labels exactly and scores exactly where batching preserves
  numeric execution; otherwise measure and explain any numeric differences and
  verify that threshold decisions are unchanged.
- Compare per-range final states, open/close timing, amounts, prices, stop behavior,
  completed-trade records and aggregate reports against the guarded baseline.
- Cover long/short and strong-open signals; zero, negative, NaN and infinite
  regression endpoints; invalid gradients; disabled/equal/different windows;
  missing/misaligned timestamps; sparse ranges and unfinished positions.
- Include the OM crash and representative large-history BTC/ETH replay cases.
- Verify no writes to shared OHLCV/features/targets/regression stores and no
  adaptation or use of test rows for model selection.
- Verify gain leaves train/eval numeric artifacts untouched.
- Run affected package tests sequentially, always with `julia --project=.`.
- Measure a warmed guidance microbenchmark, representative complete coin
  replays, then one complete frozen-universe gain run. Subset timing probes must
  not be combined into a replacement eligibility calendar or final assessment.

Provisional optimization targets, to validate rather than promise:

| Metric | Target |
|---|---|
| Warmed guidance allocations | At most 128 bytes per simulated row |
| Peak physical footprint on this host | At most 5 GiB |
| Complete gain run, verified reusable preparation/predictions | At least 3x faster than the corresponding measured baseline |
| Gain-only train/eval source reconstruction | None |

Record cold and warm artifact/page-cache timings separately, with the same
machine load. The current live run is not a clean cold or warm benchmark.

**Recommendation:** implement stages 1 and 2 first, then the streaming/data-lifetime
changes in stage 3. This removes demonstrated redundant work before attempting
parallelism, without changing the strategy or held-out evaluation contract.

## Implemented workflow

The gain CLI now bypasses dataset preparation and training sources entirely.
An explicit one-time `stage=snapshot` verifies original history hashes and test
membership and seals coin-sized test features, prices and active regression
guidance. Existing runs have no such seal and must create it before gain; there
is no automatic unverified migration or current-market-data fallback.

Gain verifies the seal, infers bounded batches and streams independent replay
ranges. Prediction caches include checkpoint and snapshot digests; replay caches
add strategy settings and a replay-semantics version. Content is published before
digest completion markers. Only complete verified outputs are reused.
The original universe is read from saved provenance when no explicit coin
override is supplied, so subsequent edits to default coin-set membership do not
change a saved assessment. Configuration mismatches still fail.

Live and replay use shared gate arithmetic. Replay resolves typed regression
columns once per range, reuses equal windows, and evaluates numeric guidance only
for open signals. The gate allocation regression test enforces the 128-byte/row
target. Snapshot files do not depend on mutable histories or regression stores
after sealing.

The synthetic smoke run sealed 4,800 reserved samples in 3.4 seconds. A first
gain run, including JIT compilation and reporting, took 10.8 seconds; a fully
verified cache rerun took 1.2 seconds. Process peak RSS was approximately 2.2 GiB,
including preparation and adaptation performed in the same smoke process.
These are fixture measurements, not a full-market runtime or memory guarantee.
The complete market-run targets above still require measurement on a sealed
full-universe run.
