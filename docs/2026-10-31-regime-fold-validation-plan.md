# Regime, fold, validation and signal-report implementation plan

## Objective and agreed boundaries

Build a reproducible SimpleChains-based training pipeline with:

- Shared market regimes and consecutive-range folds across coins.
- Balanced `(set, regime, fold)` sample exposure without temporal leakage.
- Held-out evaluation-loss stopping and checkpoint selection.
- Trend-entry, trend-hold and no-established-trend precision, recall and coverage reports, overall and per regime/fold,
  to support selection of fixed signal thresholds.
- A separate chronological walk-forward experiment.

Regimes are historical partitioning/reporting metadata, not required live inputs.
Live trading uses one fixed threshold per supported signal type, not per regime.
Classifier trend detection remains separate from strategy execution and
completed-trade economics. No claim of profitability follows from signal precision.

## Implementation status

The synthetic-first experiment pipeline is implemented in
[RegimeCalendar.jl](../TradingStrategy/src/RegimeCalendar.jl),
[RegimePipeline.jl](../TradingStrategy/src/RegimePipeline.jl),
[EvalTraining.jl](../Classify/src/EvalTraining.jl),
[SignalMetrics.jl](../Classify/src/SignalMetrics.jl) and
[RegimeTraining.jl](../scripts/RegimeTraining.jl).

Implemented surfaces:

- Full-content dataset identities and private rebuilt feature/target stores.
- Shared regime/fold/block calendars, balanced complete-block assignments,
  explicit unknown exclusions, fold/regime selection and optional train weights.
- Fixed held-out eval-loss stopping, pooled natural/weighted histories and
  identity-checked best/last checkpoints with resumable optimizer/RNG state.
- Regime/fold-balanced eval weights are computed once. Epoch evaluation retains
  only weights, not membership arrays; full indices are released before adaptation
  and eval metadata is reloaded for final reports. Per-epoch cell-loss diagnostics
  are no longer produced. See the current
  [folds, files and memory-lifetime documentation](2026-11-05-regime-folds-manifests-indices-and-memory.md).
- Train-only soft reinforcement via `reinforce=N`, matching TrendDetector's
  refresh interval and class-normalized 2x hard-example emphasis. Reinforcement
  state/history resumes exactly; eval weighting, patience and total cap remain
  unchanged, including when enabling reinforcement after earlier adaptation.
- Five-label and grouped long/short precision, recall, coverage, opportunity
  episodes, macro/coin diagnostics, threshold candidates and SVG curves.
- Joint shared-block bootstrap minute-metric intervals; explicit insufficient
  independent support. Opportunity reports provide counts/support, not bootstrap
  intervals.
- Separate expanding chronological experiments with cutoff-safe membership,
  horizon purges, fresh models and disjoint aggregated test windows.

Synthetic pooled training, frozen-threshold test reporting and four chronological
two-epoch runs have completed. Focused tests cover exact stopping limits, resume,
BatchNorm isolation, unknown exclusion, shared assignments and metric semantics.
These smoke runs verify execution, not final model quality.

The CLI currently reports **one scalar threshold for all labels and regimes**;
per-signal report curves do not add unsupported live strategy controls. Test
membership and best-checkpoint digest are recorded alongside that threshold.
Changed thresholds/checkpoints and training after test reporting produce warnings,
not policy rejections; threshold evaluation history is retained. Existing P02 and legacy
training-loss adaptation remain available unchanged.

Partial real-data integration has passed for the repaired alphabetical subset
through AVL, using August 1–October 31, 2026 (19:42 UTC) plus 11 days of context.
Of 43 cached symbols, 32 have window coverage and seven contribute eligible
assigned samples. Private F20/T25 stores were rebuilt. The normal ten-coin detector
produced only unknown periods because at most seven coins passed liquidity.
With explicit user approval, a separate subset-only configuration uses five
minimum coins and a 20% coin-weight cap; liquidity criteria and market defaults
remain unchanged. It produced 219,293 train, 56,003 eval and 55,212 reserved test
samples, all in diagnostic fold 40, with all three known regimes represented.
Two M011 epochs and eval signal/event/bootstrap/coin reports completed. No test
threshold was selected and no reserved test predictions were evaluated.

Pending: full real-market history completion, broader real-store audit/rebuild
runs, detector calibration on a declared development period, and full-budget
real-data quality assessment. Market RG01/RF01 defaults are illustrative, not calibrated choices.
The phase descriptions below retain the original design and acceptance criteria;
they are not claims that real-market acceptance has already passed. The current
implementation status above supersedes original per-epoch cell-loss requirements.

The runner is now named `RegimeTraining.jl` and follows TrendDetector's
`test`/`train`, `ts`, `coinset`, `daterange`, `coins`, `retrain`, `runid` and
`folder` conventions. Default test inputs use CS00/D00 and the selected
registered feature/target/model configuration; the small fixture requires
`mode=synthetic`. Adaptation ends with eval diagrams and manual next-step guidance.
A subsequent gain run requires the regime best checkpoint and uses reserved
test membership and the registered strategy, with independent predicted and
target passes. The same or another TS context can select that checkpoint when
its classifier configuration is identical. Strategy settings are recorded
separately with replay history. Changes after test exposure warn but do not
restrict experimental strategy replay; classifier/data compatibility still applies.
TrendDetector remains unchanged. Regime date overrides select experiment
membership, rather than only a gain replay window.

## Development order: synthetic first, real data afterwards

OHLCV history repair and derived-store rebuilding run independently and must not
block implementation. Implement the complete pipeline against SINE and DOUBLESINE
artificial OHLCV from TestOhlcv, including regime detection, shared folds, set
assignment, training, eval-loss stopping, metrics, threshold reporting and
walk-forward execution.

Use deterministic synthetic variants covering up/down trends, no established
trend, missing coverage, regime transitions, liquidity exclusions and unequal
coin histories. Synthetic detector settings and expected regime outcomes are
test fixtures, not production threshold calibration. Test data belongs under
`~/crypto/debug`.

After the complete synthetic pipeline passes and history repair has finished,
audit/rebuild real-market stores and run the real-data coverage, calibration and
training experiments. Only the real-market validation stage depends on history
repair completion; implementation and synthetic validation do not.

## Original baseline (before this implementation)

| Capability | Current implementation | Required change |
|---|---|---|
| P02 partitions | Repeating train/test/train/train/eval/train sequence; 480-minute gaps | Retain as control; add a new partition ID for the new design |
| Target horizon | Trend05 targetminutes=480; gap assertion | Preserve; verify actual boundaries and target dependencies |
| Stopping | 50-epoch patience, 0.05% meaningful improvement, 200-epoch cap | Drive stopping and best checkpoint from eval loss, not training loss |
| Regimes | Diagnostic candidates and historical/online labels | Freeze an explicit versioned detector configuration |
| Fold metadata | Coin-local rangeid only | Add shared foldid and calendar blocks |
| Training index | coinix, row, labelix; set filtering | Add fold/regime selection and frozen weighting metadata |
| Evaluation | Post-training aggregate confusion tables | Correct recall; add signal curves, counts and grouped reports |
| Walk-forward | Not implemented | Separate chronological train/eval/test experiment |

Relevant surfaces:
[sets generation](../Classify/src/SetsStore.jl),
[sample indexing/predictions](../Classify/src/ClassifierStore.jl),
[training loop](../Classify/src/SampleSource.jl),
[model/checkpoint/metrics](../Classify/src/Classify.jl),
[pipeline](../TradingStrategy/src/artifactpipeline.jl),
[configuration](../TradingStrategy/src/tradingstrategyconfig.jl), and
[regime diagnostics](../scripts/diagnose_market_regimes.jl).

## Phase 1: Freeze data and regime identity

1. Establish the data-audit/rebuild mechanism using SINE and DOUBLESINE fixtures.
   After real OHLCV history repairs finish, rebuild affected feature, target and
   sets stores and audit all reused stores, not just deleted caches. The real-data
   rebuild is a gate for real-market validation, not for subsequent implementation.
2. Extend dataset identity with OHLCV content/revision, liquidity settings,
   features/targets, date range, coin universe and partition/calendar identities.
   Coverage endpoints alone must not authorize reuse after interior changes.
3. Persist an immutable run manifest and exact train/eval/test membership hashes.
   Reject incompatible resumed checkpoints. Keep P02 and existing classifier
   artifacts unchanged; new experiment identities receive new IDs.
4. Implement and test liquidity-filtered regime detection on synthetic patterns.
   After history repair, re-run diagnostics with supplemented market history.
   Freeze slope, volatility-strength, breadth, hysteresis and confirmation settings
   using a declared development period, not test performance.
5. Persist shared regime segments with regime, estimated start, confirmation time,
   end, detector ID and data identity. Missing market coverage is explicit
   `unknown`, never silently `flat`. Exclude unknown-regime samples from training,
   evaluation and test in every new experiment mode, including walk-forward.
   Retain exclusion counts and coverage diagnostics, not optimization or metric
   samples. Existing historical P02 artifacts remain unchanged.
6. Historical backdated labels may support descriptive stratification. Persist
   confirmation times and the maximum information cutoff supporting each label.
   Do not use post-cutoff confirmations or retrospectively screened liquidity
   ranges to select or weight training samples in a chronological fold.

**Acceptance:** identical data/config produces identical regimes and identities;
changed liquidity settings or OHLCV invalidate affected derived memberships.
Detector settings are not tuned to improve test results.

## Phase 2: Shared calendar and balanced diagnostic folds

This is the main regime-balanced experiment, not walk-forward validation.

1. Start with shared historical regime segments, then cut them into calendar
   blocks of configured duration. Insert the required gaps at set boundaries and
   at regime boundaries. Short remnants are explicitly omitted and counted.
2. Group adjacent calendar blocks into consecutive folds. Proposed starting point:
   fixed 90-day UTC fold windows with a fixed anchor, independent of coin listings.
   Block boundaries are also cut at fold boundaries. Duration remains a diagnostic
   choice to confirm from real-data sample counts, not an assumed optimum.
3. Assign shared `blockid`, `foldid`, regime and time bounds before projecting onto
   individual coin histories. Missing/liquidity-filtered rows do not shift the
   calendar or restart the fold numbering of a coin.
4. Count eligible samples per block across the frozen coin universe. Deterministically
   assign complete blocks to train/eval/test, targeting P02 proportions (4:1:1)
   within each `(regime, fold)` cell. Adjacent same-set blocks may be represented
   together, but retain their original IDs.
5. Keep the set assignment identical across coins at any shared calendar block.
   Do not split a simultaneous market period into train for one coin and test for
   another to improve balancing.
6. Recount samples after gaps, target-ignore filtering and feature alignment.
   Persist actual counts and deviations from targets. Choose a deterministic
   assignment objective and tie-breaking rule; never silently rerandomize.
7. Do not fabricate missing regimes: a 90-day fold may be entirely up or down.
   Balance available cells, balance regimes across the full experiment, and
   report empty/small cells. Exact equality is not a valid invariant.
8. Exclude unknown-regime periods from train/eval/test in all new experiment
   modes and report their duration/counts. Do not include them in pooled losses,
   precision, recall, coverage denominators or threshold selection.

Per-coin sets gain `blockid`, `foldid`, `regime`, while retaining `rangeid`.
Store the shared calendar separately, with assignment/config hashes.

**Acceptance:** no overlap between sets; actual target lookahead fits every
boundary gap; folds contain consecutive blocks; identical timestamps have
identical assignments across coins. Tests cover missing coins, regime changes,
short segments, absent regimes, fold boundaries and deterministic balancing.

## Phase 3: Fold-aware SimpleChains training and eval-loss stopping

1. Extend sample indexing to select sets, folds and regimes, and retain the metadata
   through compaction. Build distinct train and eval sources; test is not read by
   optimization or early stopping.
2. Initially use all eligible train samples with existing class balancing.
   Add explicit, optional train-only regime/fold weights after the baseline exists.
   Report effective sample mass and maximum weights; reject empty requested cells.
3. Freeze eval membership and weights before epoch 1. Report both:
   - Natural pooled eval loss, representing the observed sample distribution.
   - A fixed regime/fold-weighted eval loss, representing balanced diagnostic cells.
   Proposed stopping loss: average cell mean loss within each regime, then average
   over the represented regimes, with the cell list fixed for the run.
   Keep class-weighted losses separate from natural-distribution diagnostics.
4. Each epoch: train batches, enter inference mode, evaluate eval batches without
   changing parameters or BatchNorm running statistics, restore training mode.
   Record train loss, both eval losses, per-cell losses, duration and counts.
5. Apply existing patience=50, meaningful improvement=0.0005 and total cap=200
   to the chosen fixed eval-loss series. Meaningful improvement resets patience;
   every actual record-low eval loss remains eligible for the best checkpoint.
6. Persist last-epoch and best-eval checkpoints separately, including optimizer
   state, histories and dataset/weight identity. Resume from last epoch; deployment
   and final evaluation load best eval. Log patience exhaustion versus epoch cap.
7. Train one pooled model across diagnostic train folds. "Training by fold" also
   means explicit fold-subset selection and fold-aware weights, not automatically
   training one model for every diagnostic fold.
8. If weighting is later adjusted to help weak regimes, only training weights
   change. Start a distinct experiment; keep eval distribution and metrics fixed.

**Acceptance:** eval/test samples never reach gradient updates; eval leaves
BatchNorm state unchanged; resuming matches uninterrupted stopping decisions;
checkpoint selection follows eval rather than training loss. Test the exact
50-epoch patience, 0.05% boundary and 200-total-epoch limits.

## Phase 4: Signal precision, recall and coverage

### Definitions and label contract

Trend05 produces mutually exclusive `longopen`, `longhold`, `shortopen`,
`shorthold`, and `allclose` labels. At a given minute, an established trend is
either long (`longopen`/`longhold`) or short (`shortopen`/`shorthold`), never both.
`allclose` means no established trend; it is not necessarily a flat market regime.
The classifier has no previous-decision/position input and cannot distinguish
`longclose` from `shortclose`. Do not add directional exit targets to this plan.

- Report all five labels separately, including long-entry and short-entry detection.
- Report `allclose` as no-established-trend detection, not as an exit detector.
- Additionally report established-long and established-short detection, grouping
  the corresponding open/hold labels. Count these using the same mutually
  exclusive classifier decisions, not independent simultaneous binary signals.
- Target-defined trend endings can be described as transitions from established
  long/short to no established trend, or to the opposite trend. Such transition
  diagnostics do not assert that a position was held or a closing action executed.
- Any directional closing action belongs to the strategy and depends on its
  current position. Evaluate that separately from classifier label correctness.
- Freeze mappings for supported strong/normal labels and handling of `ignore`.

At each threshold, evaluate the same argmax-label-plus-score rule used by the
existing classifier consumer. Use one shared decision helper for reporting and
threshold application. If per-class probabilities or grouped probability sums
are explored, retain them as a separate, explicit decision-rule experiment.

For every supported signal:

- TP: signal issued and matching target; FP: signal issued and nonmatching target.
- FN: matching target but signal not issued, including wrong-label predictions.
- Precision = TP / (TP + FP); recall = TP / (TP + FN).
- Coverage = issued signals / eligible evaluated samples.
- Unsupported/empty denominators produce explicit missing metrics plus counts,
  not zero or perfect scores.

Fix the existing extended-confusion recall defect and verify parity with a
one-versus-rest reference implementation.

### Grouping and opportunity dependence

Report by set, signal and threshold:

- Overall pooled counts across regimes (micro metrics).
- Each regime; each fold; each `(regime, fold)` cell.
- Macro summaries separately, without substituting averages for pooled counts.
- Coin diagnostics and label prevalence to reveal concentration.

Minute-level curves are the first required deliverable. Then add target-defined
opportunity-level reporting:

1. Define an opportunity as a contiguous same-signal target interval within the
   same coin, set and block; never bridge missing minutes or split boundaries.
2. Match at most one detected signal to each opportunity and classify other
   unmatched predicted signal episodes explicitly. No executed-trade pairing.
3. Use exact interval matching initially; any early/late timing tolerance is a
   separate documented parameter, fixed before inspecting test results.
4. Report opportunity detection rate, prediction-episode precision, signal timing,
   duplicate signals and episode frequency separately from minute coverage.
5. Account for correlation with block-level resampling, not independent-minute
   confidence intervals. Report insufficient independent blocks explicitly.

**Acceptance:** synthetic tests cover wrong direction, missed no-trend labels, threshold
equality, abstention, ignored labels, repeated signals and missing regimes.
Summed group counts equal pooled counts; one opportunity is never matched twice.
Threshold abstention is reported separately: it must not silently become a
predicted `allclose` or count as a correct no-established-trend classification.

## Phase 5: Threshold report and fixed live operating points

Produce Arrow tables plus a human-readable report/plots under the configured run
folder. Include:

- Full precision/recall/coverage curves and raw counts on a documented threshold grid.
- Overall and regime/fold breakdowns, confidence/insufficient-support indicators.
- Current fixed thresholds as reference rows.
- Comparisons at equal thresholds and comparable coverage.
- Non-dominated candidate operating points and their weak-regime tradeoffs.
- Selected checkpoint, dataset identity, label mappings and exact decision rule.

Do not auto-select an arbitrary required precision. Present choices for the user:
for example maximum recall subject to a specified precision floor and minimum
independent support. Those requirements express tolerable false detections, not
profitability. Regime reports diagnose weaknesses; they do not create regime
thresholds for live trading.

Select one fixed threshold per supported signal type using eval only. If the
current strategy exposes only a shared open threshold or derives exits from
other rules, report that constraint and only wire supported thresholds; adding
new strategy controls is a separate intentional behavior change.

Freeze thresholds and checkpoint, then evaluate test once. Test threshold sweeps
must not become a second selection procedure. Keep completed-trade replay and
economics as a separate classifier-plus-strategy assessment.

**Acceptance:** reporting and inference emit identical signal decisions at the
chosen thresholds; no live regime dependency; test data is not used for selection.

## Phase 6: Chronological walk-forward folds

Use the shared calendar blocks, but a distinct experiment/assignment identity:

1. For each chronological run, choose expanding earlier training windows, a later
   eval window and a still-later test window, separated by horizon-safe purges.
   Proposed starting schedule: two-year minimum training history, three-month eval,
   three-month test, advancing three months; validate availability before running.
2. Persist `walkforwardid` separately from diagnostic `foldid`: a calendar block
   can be training in one run and testing in an earlier run.
3. Never rebalance chronological windows by moving future samples backwards.
   Report missing regimes; apply train-only weighting where explicitly requested.
4. Freeze detector settings before the experiment. Use only regime/eligibility
   information available at each run's training cutoff for training selection.
   Retrospective test labels may be used for later descriptive reporting only.
5. Initialize a fresh M011 model per run; use that run's eval window for stopping
   and fixed thresholds, then evaluate its later test window.
6. Retain non-overlapping test windows and aggregate their counts for out-of-time
   reporting. Keep diagnostic P02/balanced-fold results separate.

**Acceptance:** assertions verify train < eval < test and purges for every run;
no post-cutoff training dependencies; no repeated test samples in pooled results.
Walk-forward results are the evidence for forward generalization, unlike the
interleaved balanced diagnostic experiment.

## Execution order and completion gates

| Step | Depends on | Deliverable |
|---|---|---|
| 1. Data identity and frozen regimes | None | Versioned manifest and regime calendar |
| 2. Shared blocks, diagnostic folds and balancing | 1 | New partition ID and coverage audit |
| 3. Fold-aware training and eval stopping | 2 | Reproducible M011 baseline run |
| 4. Correct signal metrics and grouped reporting | 2; model predictions from 3 for real-data run | Tested counts and threshold curves |
| 5. Threshold report and fixed operating points | 3, 4 | Eval-based selection; frozen test report |
| 6. Chronological walk-forward experiment | 1, 3, 4, 5 | Separate out-of-time results |

Run targeted package tests one package at a time: affected Targets/Ohlcv utilities,
then Classify, then TradingStrategy; run Trade tests if threshold consumers change.
Add functions/types with docstrings and module-local tests; bump package versions
when externally used APIs change. Update related configuration/CLI documentation.

Complete and verify every phase end-to-end with SINE and DOUBLESINE first,
without waiting for real OHLCV history repair. Include assertions that unknown
samples never enter any train/eval/test source or metric denominator.
After repair finalization, audit real-data coverage and derived stores before
real-market calibration and expensive training. Long runs must have Terminal commands, progress logs, Ctrl+C,
checkpoint resume and immutable experiment IDs. Do not start them inside VS Code
without an explicit request.

Completion requires the baseline and walk-forward artifacts to persist their
membership, fold/regime metrics, eval-selected checkpoints, thresholds and test
results. Reporting-only phases can finish before deployment controls are changed.
