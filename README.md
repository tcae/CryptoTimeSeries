# CryptoTimeSeries

## Getting started

<!-- This code base is using the Julia Language ) -->
This code base is using the Julia Language to make a reproducible project named
> CryptoTimeSeries

It is authored by tcae.

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:

   ```julia
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

## What is it all about?

This is a playground Machine Learning project that aims to predict the best trades using popular cryptocurrencies.

Be cautious and don't use it for real trading unless you know what to do. There is no backward compatibility intention, restructuring of project parts may happen without announcement. Any trade signal from this software is no recommendation to trade. Trading cryptocurrencies is high risk.


## Entry points

Each module is located in the src folder and has a corresponding '_test' unti test in the test folder.

- The main module is [Trade](Trade/src/Trade.jl) that is setup via [trade_test](Trade/test/trade_test.jl), which runs a tradeloop
- Machine learning is made available by [Classify](Classify/src/Classify.jl).
- [Targets](Targets/src/Targets.jl) provides target labels to train, evaluate and test machine learning approaches.
- [Features](Features/src/Features.jl) provides OHLCV derived features as input for machine learning and trading.
- [Ohlcv](Ohlcv/src/Ohlcv.jl) provides means to get locally stored historic Open High Low Close Volume data.
- [Xch](Xch/src/Xch.jl) is the abstract layer to [Bybit](Bybit/src/Bybit.jl)
- [Assets](Assets/src/Assets.jl) is selecting those assets from the exchange that are compliant to certain criteria that makes them a trading candidate.
- [EnvConfig](EnvConfig/src/EnvConfig.jl) provides common configuratioin items, like the set of cryptos to use or common folders
- [TestOhlcv](TestOhlcv/src/TestOhlcv.jl) provides periodic OHLCV patterns to test implementations.

Scripts for investigations as well as the GUI ([cryptocockpit](scripts/cryptocockpit.jl)) are located in the scripts folder without a unit test

### Small-network runtime comparison

[SimpleChains versus configuration 059](docs/2026-10-26-simplechains-model008-benchmark.md)
documents the M009 SimpleChains alternative, the M010 matched Flux control, and
the architecture-matched M011 with native BatchNorm, with measured
SINE/DOUBLESINE training and inference runtimes. Reproduce the isolated
test-mode comparison with `julia --project=. scripts/benchmark_model008.jl`.
Add `--batchnorm` to compare M011 directly against 059.
Configuration 059 retains M008 and its BatchNorm layers unchanged.

### Supplement Bybit USDT OHLCV

Run from Terminal in the repository root:

```bash
set -o pipefail
caffeinate -i julia --project=. scripts/supplement.jl 2>&1 | tee "$HOME/crypto/debug/supplement-$(date +%Y%m%d-%H%M%S).log"
```

The script first updates every existing `Bybit/<coin>-USDT/ohlcv.arrow`
cache to the last completed UTC minute. Unavailable/delisted coins are reported
and left unchanged. Existing historical prefixes, including supplemented
Binance candles, are preserved; only the final cached candle may be refreshed.
It then runs the public-market `Trade.tradeselection!` overload to discover
uncached coins passing the trading daily-volume and continuous-liquidity gates.
For those coins it downloads up to two years of available one-minute history.

No credentials, classifier, orders, or VS Code process are required. Ctrl+C
stops the foreground run; completed coin writes are retained and rerunning
continues from the updated caches. Each write is atomic and verified. This
script does not rebuild or delete feature, target, regression, or sets stores.
Unexpected errors abort with diagnostics rather than silently skipping coins.
Bybit kline internal errors (10016) are retried up to four attempts, waiting
1, 2 and 4 seconds. Exhausted retries and other API errors abort explicitly;
failed responses are never interpreted as the end of available history.
Run `julia --project=. scripts/supplement.jl --help` for help.

### Refresh shared stores

Refresh features/targets and explicitly rebuild selected train/test/eval assignment stores
from cached full OHLCV histories:

```bash
julia --project=. scripts/refresh.jl partitions=P02_D01,P02_D00
julia --project=. scripts/refresh.jl features=F19 targets=T24 partitions=P02_D01 pairs=BTC-USDT,ETH-USDT
```

Comma-separated IDs must be one shell argument; quote lists containing spaces, for
example `partitions="P02_D01, P02_D00"`. Without `pairs`, all cached pairs are processed.
The default exchange is Bybit; select another with `xch=KrakenSpot`, for example.

`partitions` always replaces the selected `sets/Pxx_Dxx.arrow` files using current
partition/date-range registry settings, including when their stored hashes differ.
Features and targets retain their normal reuse checks. Classifiers, predictions and
experiment-local regime/fold assignments are not rebuilt or removed. Changed memberships
can invalidate earlier evaluations; prefer new configuration IDs to preserve reproducibility.
Failures are logged and summarized while other pair/config combinations continue.
Empty sets stores are saved with a warning and summarized at exit by coin pair and
partition ID, separately from failures. They contribute no samples through those stores.
For each refreshed partition, the final output includes a copy-pasteable
`traincoins() = [...]` definition limited to coins with train, eval and test assignments.
That partition list does not establish RegimeTraining eligibility. Add `ts=066` to
run full preparation using that strategy's features/targets and the default regime
detector/folds, and print a separate **RegimeTraining eligibility** report at completion:

```bash
julia --project=. scripts/refresh.jl partitions=P02_D01 ts=062,066,067
```

This Bybit-only report evaluates selected USDT pairs, using dates from the requested
partitions (D01 by default when none are requested). It refreshes required shared stores,
but does not train or change classifiers. It reports excluded short histories and per-coin
regime/fold train/eval/test sample counts per configuration, then prints one common
copy-pasteable list: every coin must have complete coverage under **all** selected configs.
Each config is fully evaluated, including its feature history and target lookahead;
the result is not approximated using only the largest warmup value.
Eligibility preparation processes one coin at a time and releases its OHLCV and
feature/target tables before the next coin. Hourly observations are appended to one
temporary Arrow file, then memory-mapped as per-coin views to build the shared market
calendar with sorted cursors rather than timestamp dictionaries. Eligible samples use
an exact sorted walk across history, feature and target timestamps, without per-minute
lookup dictionaries or temporary feature-row objects. Eligible minute timestamps are temporarily
stored on disk and reloaded one coin at a time for fold balancing and membership counts.
Each accepted coin's timestamp file must survive preparation of all later coins.
An empty timestamp table yields zero counts; a missing file is an error, not an exclusion.
No minute-level membership tables are retained across coins or configurations;
the final reports retain only three sample counts per coin. Temporary files are removed
when preparation finishes or throws. Shared market observations cannot be evaluated
independently per coin: volume-capped cross-coin weights require simultaneous coverage.
Coverage is universe-dependent; changing the list can change market regimes.
RegimeTraining warns and skips histories that cannot satisfy its warmup requirement;
it still fails explicitly if no usable histories or shared training blocks remain.
Run `julia --project=. scripts/refresh.jl --help` for details.

### Market regime diagnostics (Phase A)

Run `julia --project=. scripts/diagnose_market_regimes.jl` to inspect CS01/D01
without changing data partitions, target stores or classifiers. Results are saved
under `~/crypto/debug/market-regimes/<timestamp>/`:

- `market.arrow`: hourly market indicators, using exact normalized ten-day
  minute-price regressions, one-day slopes, directional breadth and volatility.
- `coverage.arrow`: explicit missing/invalid coin-history diagnostics.
- `candidates.arrow`: 18 threshold/strength/confirmation comparisons.
- Candidate hourly and segment tables: online labels, retrospective labels,
  estimated starts and confirmation timestamps.
- `report.txt`: methodology, distributions and candidate durations/counts.
- `sample-coverage.arrow`: currently stored eligible samples by feature/target,
  coin, set, illustrative retrospective regime and calendar month. Months are
  reporting buckets, not walk-forward folds. Missing and historical target
  stores are identified separately; none are regenerated.

Volume weights use trailing approximate quote turnover (`pivot * basevolume`),
with a 10% coin cap and at least ten eligible coins. A coin hour is eligible only
if its cutoff minute lies inside an `Ohlcv.liquiditycheck` range (`Ohlcv.ld`
defaults, the same liquidity definition as P02 set generation). All indicator inputs are
past/current completed minutes. Historical regime labels may backdate to the
start of a subsequently confirmed candidate; online labels never backdate.
Volatility-normalized strength is a descriptive heuristic, not a significance
test. Candidate settings are not selected automatically or used for training.

Run focused diagnostic tests with
`julia --project=. -e 'include("test/market_regime_diagnostics_test.jl")'`.

### Trend05 target horizon and partition safety

Trend05 accepts `targetminutes` (480 minutes by default). A candidate must reach
`targetgain`, with uninterrupted holds, at or before its elapsed-time deadline.
Expired candidates produce no entry/hold segment. An incomplete candidate remains
provisional only until its deadline can be observed. T24/T25/T26 explicitly use
480 minutes, matching P02's eight-hour gap.

Classifier configuration and store preparation assert that every target's declared
look-ahead is positive and does not exceed its partition gap. Historical target
configs with longer horizons now fail that check rather than running unsafely.

This changes T24/T25/T26 content hashes. Existing target stores, classifier
metadata and predictions are not silently compatible with bounded targets.
The normal hash checks reject incompatible artifacts; they must be explicitly
rebuilt/replaced before training with the changed targets.

### Training stopping criteria

Legacy classifier adaptation stops after 50 epochs without at least a 0.05% relative
loss reduction from the best loss at the last meaningful improvement. Smaller
record improvements accumulate against that reference; they do not individually
reset patience. Every actual record-low checkpoint is still saved as `nn.bson`,
and `nn-lastepoch.bson` retains the latest model, optimizer and loss history.
The monitored loss is the full class-weighted training loss, not validation loss.

CS01 has a hard total budget of 200 epochs; adaptation also caps any larger
requested budget at 200. Recorded epochs count toward the budget on resume.
`retrain` resumes the last-epoch checkpoint, not the best checkpoint, and does
not restart a converged or exhausted history. CS00 retains its 10-epoch budget.

To train and evaluate the real-data F20/T25 M011 classifier (TS063):

```sh
julia --project=. scripts/TrendDetector.jl train ts=063 coinset=CS01 daterange=D01 stage=evaluate
```

TS060 selects the corresponding M008 classifier. Existing TS060 artifacts are
reused unless explicitly retrained. Comparing a completed historical TS060 run
against TS063 measures classifier quality, but is not a controlled runtime
benchmark: the earlier run used different stopping criteria and may have run
alongside other jobs.

### Regime/fold validation experiments

[RegimeTraining.jl](scripts/RegimeTraining.jl) runs a separate SimpleChains
SimpleChains experiment without altering P02, existing classifiers, live
thresholds or OHLCV caches. Start with SINE/DOUBLESINE from Terminal:

```sh
julia --project=. scripts/RegimeTraining.jl test mode=synthetic runid=synthetic-01 stage=train epochs=200
```

Shared consecutive calendar folds and complete blocks are balanced toward
train/eval/test 4:1:1 within each represented regime/fold. Unknown regimes are
excluded from all three populations and explicitly counted. Experiment inputs and
indices are checked using their row counts and timestamp bounds, together with the
configuration. Interior data changes with unchanged coverage are not detected:
regenerate the run's artifacts after such changes. This coverage-based scheme uses
new default run IDs; older artifacts must be regenerated rather than resumed.

Stopping and best-checkpoint selection use **held-out, fixed regime/fold-weighted
eval loss**, with 50-epoch patience, 0.05% meaningful relative improvement and
a hard 200-total-epoch budget. Natural pooled eval loss is reported separately.
`retrain` resumes the last checkpoint, warning if test results already exist; every actual
eval record low can become the best checkpoint. `eval-last.bson` is saved every epoch
with complete training/evaluation history. `eval-best.bson` is replaced only on a
strictly lower weighted eval loss and retains history through that best epoch.
Loading or resuming repairs a missing/stale best file from the last checkpoint,
but leaves a current best file untouched. `epochs=2` is useful for a smoke
test, not a classifier-quality assessment.

The recorded `trainloss` is the effective-weight-mass-weighted mean of losses
already computed during training batches, including reinforcement when enabled.
It describes changing parameters and training-mode BatchNorm, not a separate
final-model inference pass over the training set. Older records in resumed runs
retain their original full-training-set inference losses. Only held-out weighted
eval loss controls selection and stopping. Epoch logs separate adaptation,
eval inference, eval diagnostic aggregation and checkpoint-saving times.

Epoch shuffling uses one UInt32 index vector; batches are lazy views rather than
millions of separately allocated index arrays. Eval diagnostics reduce losses in
chunks with pooled natural/weighted accumulators, without retaining a population-sized loss
vector or building full-population masks/tuple arrays. Fixed eval weights are
also constructed with per-cell counts instead of repeated masks.
Best/last checkpoints reference the separately persisted `eval-manifest.bson` by
SHA-256 rather than embedding its fixed weights. Keep the manifest
alongside both checkpoints. The Classify checkpoint loader verifies and resolves
the reference; existing embedded-manifest checkpoints remain resumable.
Regime/fold membership is used once to calculate the fixed weights, not during
epoch evaluation. All sample indices are released before adaptation; the eval
index is reloaded only for final predictions/reports. New eval manifests retain
weights but no regime/fold arrays. Existing v1 manifests are verified through
their frozen identity/weights and projected to the smaller active contract.
Per-epoch cell-loss reports are no longer produced; final regime/fold reports
remain available.

Market training outputs are under `~/crypto/logs/regime-training/<run>/`.
Test/synthetic runs use `~/crypto/debug/regime-training/<run>/`. Temporary
preparation files stay inside the run folder and are removed after use; market
preparation therefore also uses logs, never debug. Older/running runs retain their original
debug location: nothing is moved automatically, and resuming an old market run
requires moving its complete run folder to the logs location after it has stopped.

Immediately after adaptation, before predictions or threshold reports, the best-eval
model is also saved as `~/crypto/classifiers/<classifier>-<coinset>-<daterange>/nn.bson`,
with normal loader metadata and experiment/checkpoint provenance in `meta.arrow`.
Threshold selection does not change or delay this trained classifier. Reusing an
existing checkpoint also publishes it without further adaptation. Replacing a
published model invalidates its cached predictions and confusion tables.
Resume regime training with `rtrain ... retrain`, using the intermediate
`eval-last.bson` under logs, not the legacy classifier training command.
Registered test classifiers use `~/crypto/debug/classifiers/`; synthetic fixtures
and chronological walk-forward models remain experiment-local and do not replace
the registered deployment classifier.

- Shared market/calendar, sample coverage/exclusions, immutable identities,
  train/eval indices and compact sources.
- `balanced/epoch-losses.arrow`, best/last checkpoints (no per-epoch cell-loss table)
  and `training-report.txt`.
- Eval minute and target-interval opportunity metrics; overall, regime, fold,
  regime/fold and coin reports; macro summaries, non-dominated threshold
  candidates and `eval-curves.svg`.
- Precision/recall/coverage and counts for the five labels and grouped `long`
  and `short`. Minute metrics include 95% bootstrap intervals from 200 joint
  shared-block resamples; unsupported intervals are missing.

Reporting encodes labels once and accumulates small set/regime/fold/block
confusion tables per threshold. Bootstrap resamples these block counts instead
of rescanning minute rows for every block. Event reporting sorts once (or reuses
already sorted input), walks consecutive coin/metadata segments, and matches
intervals with forward cursors. Only segment-local intervals survive until their
counts are accumulated. Per-coin reports use table views rather than copying and
filtering the complete universe for each coin. Metric definitions, report row
order and seeded bootstrap results are unchanged.

Timestamped, flushed console messages identify the run/output folder, per-coin
preparation, shared calendars, index construction, compaction, adaptation and
reporting. Adaptation displays batch progress and logs every completed checkpoint
with train/eval losses and the best epoch. `eval training stopped` means adaptation
has finished, **not** that reporting has finished. Eval predictions are persisted
before the expensive bootstrap/event calculations; minute metrics are also saved
before event calculations. Reporting logs threshold/group/block progress at
roughly 30-second checkpoints when the current calculation reaches a progress
boundary, and logs per-coin reports. These are not timer-driven heartbeats inside
an individual long operation. Already-running Julia processes do not acquire
source changes automatically.

`allclose` means **no established trend**, not a directional exit. Threshold
abstention is separate. Reports concern opportunities, not executed-trade
economics, and do not introduce live regime-dependent thresholds.

Choose a scalar threshold from eval reports, then record it and the best
checkpoint for test (0.5 below is only an example):

```sh
julia --project=. scripts/RegimeTraining.jl test mode=synthetic runid=synthetic-01 stage=test threshold=0.5
julia --project=. scripts/RegimeTraining.jl test mode=synthetic runid=synthetic-wf-01 stage=walkforward threshold=0.5
```

Test reports contain only the chosen threshold, never an automatic sweep.
Changed thresholds/checkpoints and adaptation after test reporting warn but are
permitted. `test-threshold-history.arrow` preserves evaluation settings;
`fixed-threshold.arrow` records the latest choice. Training-resume compatibility,
patience and the total epoch cap still apply. Walk-forward trains fresh models
on expanding earlier windows, uses cutoff-safe liquidity/regime memberships
and horizon purges, and aggregates disjoint later test windows. Its supplied
threshold is fixed before running the windows.

`folds=1,2`, `regimes=up,down` and `trainweights=balanced` provide diagnostic
selection and optional train-only regime/fold weighting.

Like TrendDetector, `test` (default) selects CS00/D00 and `train` selects CS01/D01.
`ts=063` is the default registered TS context: F20/T25/M011 with its class-balancing
flag. `coinset`, `daterange`, and `coins` are registry selections/overrides.
`startdt`/`enddt` must be supplied together and override experiment dates.
Unlike TrendDetector's replay-only coin override, `coins` selects the entire
regime experiment universe, including training. Default test mode generates
CS00 artificial data over the registered date range using registered features
and targets; `mode=synthetic` is the explicit small smoke fixture.

```sh
julia --project=. scripts/RegimeTraining.jl test ts=063 stage=evaluate
julia --project=. scripts/RegimeTraining.jl test ts=063 stage=snapshot
julia --project=. scripts/RegimeTraining.jl test ts=063 stage=gain
julia --project=. scripts/RegimeTraining.jl train ts=063 coinset=CS01 daterange=D01 stage=train
```

Stages are `stores`, `train`, `predict`, `evaluate`, `snapshot`, `gain`, plus explicit
`test` and `walkforward`. Adaptation ends with eval reports and `next-steps.txt`:
inspect diagrams, select thresholds and update the trading strategy before
starting a subsequent `stage=gain` run. Gain requires an existing adapted
checkpoint and never trains a missing model. `retrain` resumes adaptation only.
Before the first gain run, `stage=snapshot` verifies each original input-history
and reserved test coverage, then seals coin-sized test features, prices
and enabled regression windows in `balanced/gain-snapshot`. This is an explicit
one-time operation for existing runs; it does not rebuild train/eval numeric
sources or adapt the classifier. Changed history coverage fails sealing, but
interior changes with unchanged coverage are not detected. A sealed snapshot is immutable.

`stage=gain` reads that snapshot, not mutable market histories or current default
coin-set membership. Explicit universe/configuration overrides must still match
the original experiment. It verifies snapshot digests, loads the checkpoint,
infers bounded batches and replays both predicted and target-reference signals.
No train/eval source compaction or full classifier feature initialization occurs.
Regression columns are aligned once per range, equal windows share their arrays,
and invalid numerical guidance still downgrades opens to hold while managing
existing positions. Missing guidance remains an explicit error.

Predictions and coin/pass replay files have digest-verified completion markers
under `balanced/gain-cache`, keyed by snapshot/checkpoint and strategy/replay
semantics respectively. Retries reuse only complete matching artifacts. Strategy
changes create separate replay caches; they do not invalidate predictions.
Partial files are recomputed, and changed committed files fail verification.
Canonical prediction/trades/gain reports are still published in their usual
locations. Gain also writes `gain-report.txt` and `gain-report.svg` in the gain
output folder, including on verified cache reruns. These reports separate
classifier predictions from the target-label reference; the SVG shows average
completed-trade gain in percent, not compounded portfolio return. Runs with no
completed trades replace these reports with an explicit empty-result message.
Progress includes coin/pass counts, timings, allocations, GC time,
throughput and peak RSS. See the
[gain performance proposal](docs/2026-11-07-regime-gain-performance-proposal.md).

Market preparation accepts and prepares each coin in one traversal: it loads the
experiment history once, checks its coverage, ensures shared stores from **full**
OHLCV history and computes liquidity once per coin before selecting valid samples.
The same liquidity mask is used for hourly observations. Market mode persists
`eligible/<coin>.arrow`, but does not write experiment-local feature/target copies
or per-sample membership files. Eligibility is staged until the accepted universe
and coverage manifest have been verified.
Shared full-history stores can still be refreshed during staging.
It processes one coin at a time before releasing
the source tables. Hourly observations are appended to a temporary Arrow file
and mapped for the **complete-universe** market calendar; eligible timestamps
remain disk-backed. A subsequent timestamp-only coin pass counts samples within
the known blocks; final pooled counts determine whole-block train/eval/test assignments.
Coverage summaries use these counts without another membership pass.
Scratch files are removed on completion or failure.

Training indices use sorted timestamp cursor joins, not minute-sized dictionaries.
Market training then uses one final coin pass to select eligible timestamps within
assigned blocks from the complete shared feature/target stores. It writes all three
compact feature/label sources (`train/`, `eval/`, `test/`) and their index metadata
together, in bounded batches. Indices retain shared-store row coordinates, labels,
timestamps and regime/fold/block metadata for weighting, diagnostics and snapshot sealing.
No separate membership, index-building or per-set feature-loading pass is needed.
Publication occurs only after output sizes and the training identity have been verified.
Shared stores must remain unchanged until a reserved-test snapshot has been sealed.
This layout uses new default market run IDs; old runs require regeneration with a new
explicit run ID rather than compatibility loading.
Synthetic/test modes retain private stores and the existing separate compaction path.
Train/eval/test indices are read without copying columns. Train and test metadata mappings are released before/after
compaction as appropriate; eval metadata remains necessary for reporting, and
test metadata is reloaded on demand. Sample ordering and weighting are unchanged.
Index batches preserve a fixed native Arrow schema rather
than compacting integer types or independently encoding each coin's dictionary.
Timestamp uniqueness checks walk sorted vectors without allocating minute-sized
sets, and window selection copies each column only once. Walk-forward still
reloads histories and stores needed
for cutoff-safe chronological membership calculations; its initial store ensuring
and calendar preparation are streamed, but its window calculation is not yet
fully disk-streamed. OS-managed mmap pages and Julia's allocator can still
contribute to resident memory.

`reinforce=N` uses the same soft reinforcement as TrendDetector: after every
N epochs, train-only misclassification flags are refreshed. Misclassified samples
receive 2x emphasis, normalized within each class to preserve its existing weight
mass (including optional regime/fold weights). All train samples remain in batches.
It is a refresh interval, not additional training epochs; `reinforce=0` disables it.
Flags, normalizers and refresh history are checkpointed for reproducible resume.
Changing the interval on resume refreshes flags and is logged, without resetting
held-out eval weights, best-loss history, 50-epoch patience or the 200-total-epoch cap.
An already converged/exhausted run does not restart merely because reinforcement
is enabled. Refresh counts are in `balanced/reinforcement-refreshes.arrow`.

```sh
rtrain test ts=063 stage=train reinforce=3
rtrain test ts=063 stage=train retrain reinforce=3
```

Use `retrain` when enabling reinforcement for an existing run; keep its runid
and dataset options. Reinforcement does not change signal thresholds or test data.
`runid` optionally names the immutable experiment; otherwise it is derived
deterministically from configuration, excluding stage/epoch/threshold controls.
`folder` selects the gain output subfolder (market runs under `~/crypto/logs`,
test/smoke runs under the experiment's debug `gain` folder). Gain replay uses reserved test samples and the
registered strategy's execution thresholds, fees, budgets and regression windows,
with separate predicted/target passes and independent contiguous block ranges.
It does not read legacy classifier artifacts or use eval samples for gains.
Keep the same runid and dataset options. The same or another TS context is
accepted when its classifier configuration is identical; TS/strategy identity
is separate from adaptation identity. The first test gain run records the
checkpoint digest, test membership and actual strategy settings in
`test-strategy.arrow`. Later runs warn on changed strategy/checkpoint settings
but permit experimental replay. `test-strategy-history.arrow` retains the settings
used by each replay invocation. This does not lock or modify registry entries.
Results informed by previous test results are not independent held-out assessments.
The reporting `threshold` option is for explicit test/walk-forward evaluation,
not an override of strategy execution thresholds.
Explicit `mincoins`/`weightcap` overrides must be supplied together and recorded
in a distinct experiment identity. M008/Flux contexts fail explicitly; use a
registered SimpleChains context such as TS063.

Real-market execution reads local
Bybit USDT OHLCV only; it does not download data. Real-market validation and
detector calibration for the full universe remain pending history-repair completion.
A two-epoch real-data integration test has passed on the repaired subset through
AVL (August–October 2026), with an explicitly approved subset-only five-coin/20%
detector configuration because fewer than ten coins pass liquidity there.
This is not full-budget model validation. RG01/RF01 market
defaults are illustrative settings, not validated optimal choices; walk-forward
needs two years of initial training plus 90-day eval/test windows and gaps.
See the [implementation plan](docs/2026-10-31-regime-fold-validation-plan.md)
and `julia --project=. scripts/RegimeTraining.jl --help`.

## Design

In order to achieve a seperation of concerns and a self documenting code, the approach is to separate

- the environment configuration [EnvConfig](EnvConfig/src/EnvConfig.jl)
- the used OHLCV data [Ohlcv](Ohlcv/src/Ohlcv.jl) and [TestOhlcv](TestOhlcv/src/TestOhlcv.jl)
- the split into evaluation, test, and training data
- the features derived from OHLCV data [Features](Features/src/Features.jl)
- the machine learning targets [Targets](Targets/src/Targets.jl)
- the classification training and test [Classify](Classify/src/Classify.jl)
- the management of assets through an exchage broker [Xch](Xch/src/Xch.jl), [Assets](Assets/src/Assets.jl)
- the trading loop [Trade](Trade/src/Trade.jl) and [trade_test](Trade/test/trade_test.jl)

In case of variants, e.g. features or classification algorithms, the same function name is implemented by different modules residing in the same , e.g. Features or Classify, package and have to imported via the package name `using Features.Feature100`.

## Under construction

1. [simplegainperformance](scripts/simplegainperformance.jl) nbestgradientgain
2. [targets](src/targets.jl)
3. [notes](notes.md)

## to do

- Track independent work areas and their current status in the [project planning board](docs/project-planning-board.md).
- Trade/preparetradecache! should use Assets.jl instead of creating it's own asset selection

## OHLCV file transfer

OHLCV cache files live under `~/crypto/<exchange>/<BASE>-<QUOTE>/ohlcv.arrow`,
with Bybit as the default exchange root. See [storage and conversion](docs/Jdf2Arrow.md).

- create a tar file of OHLCV Arrow caches:
  - cd ~/crypto/Bybit
  - tar -czvf ohlcv.tar.gz */ohlcv.arrow
  - or
  - tar -czvf ohlcv.tar.gz ~/crypto/Bybit/*/ohlcv.arrow
- list files in `ohlcv.tar.gz`:
  - tar -ztvf ohlcv.tar.gz
- extract files from `ohlcv.tar.gz` to `~/crypto/Bybit`:
  - tar -xzvf ohlcv.tar.gz -C ~/crypto/Bybit
- split a big file into 700 MB chunks:
  - split -b 700m ohlcv.tar.gz ohlcv.tar.gz.parts_
  - creates:
    - `ohlcv.tar.gz`
    - `ohlcv.tar.gz.parts_aa`
    - `ohlcv.tar.gz.parts_ab`
- combine split files
  - cat ohlcv.tar.gz.parts_* > ohlcv.tar.gz
