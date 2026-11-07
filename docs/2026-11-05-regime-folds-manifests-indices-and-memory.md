# Regime folds, manifests, indices and memory lifetime

This describes the implementation as of November 5, 2026, including lazy epoch
batches, weights-only epoch evaluation, index release/reload and compact
checkpoint references.

Update: input and index identities now use row counts and timestamp bounds rather
than content hashes. Interior changes with unchanged coverage are not detected;
regenerate run artifacts after such changes. Older identities require regeneration.

November 7 market-path update: eligibility and observations are prepared in one
coin pass using complete shared stores; no private feature/target copies or
per-sample `sets` files are persisted. After market detection, a timestamp-only
pass counts samples per block and global balancing assigns whole blocks. One
final coin pass writes train/eval/test compact files and sample metadata directly
from shared stores and eligible timestamps. Sample coordinates now refer to
shared-store rows. Coverage summaries come from pooled block counts. Synthetic/test
and cutoff-safe walk-forward selection retain their existing membership algorithms.
New default market run IDs isolate this layout from older artifacts.
The detailed lifecycle discussion below describes the earlier separate-compaction
path where it refers to private stores or per-set compaction.

## Folds, regimes and population assignment

A fold is a fixed calendar interval shared across the evaluated coin universe.
It is not a separate classifier or a rotating cross-validation split.

Market defaults in [FoldConfig](../TradingStrategy/src/RegimeCalendar.jl) are:

- Anchor: January 1, 2017.
- Fold length: 90 days.
- Maximum block length: one day.
- Boundary gap: 480 minutes.
- Minimum usable block length: 720 minutes.

These are configured defaults, not necessarily every experiment's settings.
Synthetic experiments use shorter explicitly configured intervals.

The shared market detector classifies periods as up, flat, down or unknown.
Blocks stay within a fold and a contiguous known regime interval. Unknown periods
are excluded. The end of each block is purged by the configured gap, which must
be at least the target look-ahead horizon. Short remaining blocks are excluded.
This protects target horizons at assignment boundaries; it is not a claim that
all neighbouring samples or backward-looking feature windows are independent.

Within each represented `(fold, regime)` cell, complete blocks are greedily
assigned to train, eval and test in approximate proportions of 4:1:1, based on
their eligible sample counts across coins. Assignment is shared across coins:
one calendar block cannot be train for one coin and eval for another.
Whole blocks and sparse coverage mean the proportions are not necessarily exact.

See [foldcalendar and balancecalendar!](../TradingStrategy/src/RegimeCalendar.jl).

### What folds do during training

One pooled classifier normally trains across all selected folds and regimes.
There is no automatic rotation of held-out folds and no model per fold.

- Train samples update parameters.
- Eval samples select the best epoch and drive early stopping.
- Test samples remain reserved for subsequent assessment/replay.
- Fold/regime metadata determines fixed eval weights during preparation and
  supports final diagnostic reporting; it is not retained for epoch evaluation.

Fixed eval weights give equal total mass to each represented regime, then equal
mass to each represented fold within that regime, then equal mass to samples
inside that cell:

```text
sample weight = 1 / (
    number of represented regimes
    * number of represented folds in this regime
    * number of eval samples in this regime/fold cell
)
```

See [fixedevalweights and evallosses](../Classify/src/EvalTraining.jl).
Regimes and folds are historical partitioning/reporting metadata, not extra
live classifier inputs.

Train/eval/test blocks can be chronologically interleaved. This is held-out-block
assessment, not a strictly future-only experiment. The separate walk-forward
mode implements chronological train, eval and test windows.

## Files and responsibilities

### Sample indices

The training folder contains `train-index.arrow`, `eval-index.arrow` and
`test-index.arrow`. Each row identifies an eligible coin-minute:

| Columns | Meaning |
|---|---|
| `coin`, `coinix` | Coin name and source-store position |
| `opentime` | Sample timestamp |
| `row` | Feature-store row |
| `labelix` | Target class in classifier label order |
| `set` | Train, eval or test assignment |
| `regime`, `foldid`, `blockid` | Historical grouping and shared block assignment |

Indices establish membership, source coordinates and provenance. Their row counts
and timestamp bounds participate in experiment identity checks, not their full contents.

See [experimentindex and experimentindices](../TradingStrategy/src/RegimePipeline.jl).

### Compact mmap data

The train and eval subfolders each contain:

- `trainfeatures.f32`: the Float32 feature matrix, features by samples.
- `trainlabels.u8`: one target-class code per sample.

The filenames are reused inside both subfolders. Sample columns follow the
corresponding index row order. These files supply numerical data to batches;
the indices retain the coin/time/grouping meaning of those samples.

After compaction, numerical model execution does not need the original source
coordinates or coin/timestamp columns. It needs the compact features/labels and
any configured sample weights. Epoch evaluation needs only the already fixed
weights, not regime/fold membership. The full index is loaded again for final
reporting.

See [CompactMmapSource](../Classify/src/SampleSource.jl).

### Manifests

| File | Purpose |
|---|---|
| Experiment-root `manifest.arrow` | Dataset/preparation identity and configuration description |
| Experiment-root `input-identities.arrow` | Per-coin row counts and timestamp bounds |
| Training-folder `manifest.arrow` | Identity of train/eval index coverage and training-selection configuration |
| Training-folder `input-identities.arrow` | Row counts and timestamp bounds for the train/eval indices |
| `eval-manifest.bson` | Frozen training contract: identity, sample counts, fixed eval weights, labels, architecture, description, batch size and seed |

The manifests reject mismatched coverage or configuration when reusing a run.
They do not detect changed interior data or membership with identical counts and
timestamp bounds, and do not contain the numeric feature matrix.

Best/last checkpoints now store a SHA-256 reference to the separately persisted
`eval-manifest.bson`, rather than serializing its large population metadata in
every checkpoint. The checkpoint loader verifies and resolves that reference.
Keep the manifest alongside the checkpoint files. Existing embedded-manifest
checkpoints remain resumable.

The current v2 contract retains only fixed sample weights, not regime/fold arrays.
Existing v1 manifests remain immutable on disk; loading projects their verified
contract to v2 without retaining obsolete membership arrays. Existing checkpoint
digest references remain valid. The selected membership identity and exact fixed
weights are still checked on resume.

## Index memory lifetime: what is implemented

The production runner requests streaming preparation. Its implementation in
[trainingsources](../scripts/RegimeTraining.jl) releases all three indices:

| Index | Current streaming lifetime |
|---|---|
| Train | Cleared after train mmap construction; the pre-training release function also clears it |
| Test | Cleared before train/eval mmap construction; loaded on demand through `testindexloader` |
| Eval | Fixed weights computed once, then cleared after mmap construction; reloaded only for final eval predictions/reporting |

Clearing here means removing DataFrame columns/references, not deleting the
persisted Arrow file. The train source keeps class weights and, when requested,
balanced per-sample training weights; those are not the full train index.

[releasepreparedmemory!](../scripts/RegimeTraining.jl) also clears preparation-only
histories, loaded stores, eligibility tables and shared calendars, then requests
garbage collection. Nonstreaming preparation does not perform all the same early
index releases, but the pre-adaptation release and `trainreport!` both clear all
three indices before entering the epoch loop.

### Why epoch evaluation no longer retains membership

Regime/fold membership determines fixed sample weights once. After that,
`sum(weight * loss) / sum(weight)` needs no regime/fold arrays.
Per-epoch evaluation now computes only pooled natural and fixed-weight loss.
There are no per-epoch regime/fold loss diagnostics and no new
`epoch-cells.arrow`; an obsolete file at that path is removed after adaptation.

[trainreport!](../scripts/RegimeTraining.jl) passes only `sources.evalweights`
to `traineval!`. It reloads `eval-index.arrow` afterward through `loadevalindex`
to attach timestamps, coins, targets and block metadata to final predictions.
Final regime/fold diagrams and reports remain available; their metadata is not
retained for the epoch loop.

The persisted index order is unchanged, so reloaded report rows remain aligned
with compact sample columns and fixed weights. All eval index columns are
dropped, rather than retaining Arrow-backed regime/fold columns that could keep
the whole shared file buffer alive. Required numerical mmap sources remain live.

### What are eval weights?

**Eval weights determine how much each evaluation sample contributes to the loss
used to select the best classifier.** They do not update model parameters.

Without weights, periods with more samples dominate. Your implementation instead gives:

1. Equal total importance to each represented **regime**.
2. Within each regime, equal importance to each represented **fold**.
3. Within each regime/fold cell, equal importance to each sample.

#### Example

Suppose evaluation contains:

| Regime | Fold | Samples | Combined weight |
|---|---|---:|---:|
| Up | 1 | 10,000 | 25% |
| Up | 2 | 1,000 | 25% |
| Down | 1 | 20,000 | 50% |

Up and down each receive 50%. The two up folds split that share equally.
Consequently, an individual sample in the small up/fold-2 cell has more influence
than one in the larger cells.

The selection metric is:

```text
weighted eval loss = sum(sample weight × sample loss) / sum(sample weights)
```

Weights are **fixed throughout training and resume**, making epoch comparisons
consistent. The unweighted, or natural, eval loss is also reported, but does not
control selection.

These are separate from **training class weights** and **reinforcement weights**,
which influence learning. Eval weights only define the held-out assessment.

### Does evaluation need regime/fold membership?

**No, not during the epoch loop.** Once sample weights are fixed, weighted eval
loss needs only predictions, targets and those weights.

Membership is used at two separate stages:

- **Preparation:** construct balanced sample weights and record train/eval index
  coverage and configuration.
- **Final reporting:** reload the index to group predictions by regime, fold,
  block and coin.

The current implementation does not calculate per-regime/fold losses each epoch
and does not retain membership arrays in the active training manifest.
Checkpoint selection and early stopping use only fixed weighted eval loss.
Resume validation checks index coverage, configuration and exact weights, without
requiring the full index in the epoch loop. Coverage cannot detect every membership change.

Final diagrams and regime/fold reports remain available after the index reload.
This separates useful final diagnostics from the memory required throughout
adaptation.

## Epoch losses, checkpoints and timings

The recorded `trainloss` reuses losses returned by training batches, weighted by
their effective sample-weight mass, including reinforcement when enabled.
It reflects changing model parameters and training-mode BatchNorm; it is not
another inference pass over the full training set. Older records in resumed
runs keep their original loss semantics.

Each epoch calculates pooled natural and fixed-weight held-out eval loss.
Only fixed-weight eval loss controls best-model selection and patience.
Logs separately time adaptation, eval inference, pooled loss aggregation and
checkpoint saving.

- `eval-last.bson` is saved every completed epoch with optimizer/RNG state,
  reinforcement state, complete epoch history and the selected best model.
- `eval-best.bson` is saved only for a strictly lower weighted eval loss or to
  repair a missing/stale best checkpoint. Its history ends at the best epoch.
- Loading a current best checkpoint and ending training do not rewrite it.
- `epoch-losses.arrow` records scalar epoch history. No new
  `epoch-cells.arrow` is produced; final regime/fold reports are separate.

These changes apply when the process loads the updated code. An already-running
process is not automatically restarted or updated.

## Reclaimable memory versus process memory

The current memory optimizations also include:

- One UInt32 shuffle vector and lazy batch views, avoiding millions of separately
  allocated batch index arrays.
- Chunk-local eval losses and pooled accumulators, rather than a full-population
  loss vector, collected tuple array and repeated full-population cell masks.
- Per-cell count-based fixed-weight construction.
- Reuse of training batch losses instead of another full training-set inference pass.
- Best checkpoint writes only on improvement or missing/stale-file repair.

Temporary eval arrays become eligible for reclamation when no longer referenced.
That does not guarantee an immediate reduction in Activity Monitor:

- Julia garbage collection is not tied to each phase boundary.
- Allocators may retain freed storage for reuse.
- Live mmap objects keep their mappings; clean pages may remain resident until
  the operating system reclaims them.
- Process footprint, resident memory and live Julia objects measure different things.

The compact mmap file sizes therefore do not bound total process memory.
Memory claims should distinguish live arrays, allocation peaks, mapped resident
pages and allocator retention rather than treating them as one quantity.

## what files can be deleted after classifier adaptation?

For a **market-mode** run, RegimeTraining’s run folder is `~/crypto/logs/regime-training/<runid>/`; test runs use `~/crypto/debug/regime-training/<runid>/` (`experimentfolder`).

**If you still plan to run `stage=gain`, `stage=test`, or resume adaptation, keep the run data for now.** Keep the checkpoint, manifests, calendar, eligible timestamps and indices, plus the shared feature/target stores until snapshot sealing. A sealed gain snapshot freezes the required test inputs independently of those shared stores.

| Artifact | Cleanup guidance |
|---|---|
| `balanced/{train,eval,test}/` compact files (`trainfeatures.f32`, `trainlabels.u8`) | Rebuildable if calendar, `eligible/`, shared stores, configuration and manifests are retained. Synthetic/test modes instead require their private `stores/`. |
| `balanced/eval-last.bson` | Remove if you’re sure you won’t resume training; it holds the resume state. Keep `eval-best.bson` for test/gain replay and the best checkpoint (`EvalTraining.jl`). |
| Eval reports and predictions (`eval-predictions.arrow`, metrics, plots, `coin-reports/`, `epoch-losses.arrow`) | Optional if you no longer need the reports. You may need to rerun reporting to recover them. |
| `balanced/eval-best.bson`, run indices, calendar, `eligible/`, manifests and sealed snapshots | Keep until any desired test/gain replay is complete. Synthetic/test modes also retain private `stores/`. |
| Shared per-coin feature/target stores under `~/crypto/Bybit/` | Avoid deleting casually: they’re reusable across runs, and rebuilding them requires the full OHLCV history (`ensurefeaturetargetstores!`). |

Keep the **published classifier** in `~/crypto/classifiers/<classifier-name>/` for normal use; test-mode classifiers are under `~/crypto/debug/classifiers/` (`classifierfolder`). If you’ve finished all replay/reporting and only need that published classifier, you can remove the run folder—but its saved metadata points to the run’s best checkpoint, so you’ll lose that checkpoint and its run-level provenance.

## Summary

```text
Calendar   -> shared block assignments
Index      -> membership, source coordinates and reporting metadata
Mmap       -> numerical features and target codes
Manifest   -> immutable experiment/training contract
Checkpoint -> model, optimizer and resumable training state
```

The full indices are not required for numerical adaptation once compaction is
complete. Only fixed per-sample eval weights remain required every epoch.
The implementation releases all sample indices before adaptation and reloads
eval/test metadata only for final reports/replay.
