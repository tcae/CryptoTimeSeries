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
Run `julia --project=. scripts/supplement.jl --help` for help.

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

Audit existing real-data stores without regenerating them:

```sh
julia --project=. scripts/investigate_trend05_horizon.jl
```

Add `--invalidate` to remove only target Arrow stores with proven crossings of
P02 partition gaps. Evidence and each removal are logged under
`~/crypto/debug/trend05-horizon`. The script scans T24/T25/T26 for CS01/D01 and
also distinguishes crossings into a different set type. It reconstructs the
future dependency of successful stored entry/hold segments; failed or unresolved
candidate dependencies are not recorded in target stores, so zero crossings is
not proof of full historical independence. It never removes classifiers.

### Training stopping criteria

Classifier adaptation stops after 50 epochs without at least a 0.05% relative
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

Shared OHLCV cache files now live under `~/crypto/coins/<BASE>-<QUOTE>/ohlcv.arrow`.

- create a tar file of OHLCV Arrow caches:
  - cd ~/crypto/coins
  - tar -czvf ohlcv.tar.gz */ohlcv.arrow
  - or
  - tar -czvf ohlcv.tar.gz ~/crypto/coins/*/ohlcv.arrow
- list files in `ohlcv.tar.gz`:
  - tar -ztvf ohlcv.tar.gz
- extract files from `ohlcv.tar.gz` to `~/crypto/coins`:
  - tar -xzvf ohlcv.tar.gz -C ~/crypto/coins
- split a big file into 700 MB chunks:
  - split -b 700m ohlcv.tar.gz ohlcv.tar.gz.parts_
  - creates:
    - `ohlcv.tar.gz`
    - `ohlcv.tar.gz.parts_aa`
    - `ohlcv.tar.gz.parts_ab`
- combine split files
  - cat ohlcv.tar.gz.parts_* > ohlcv.tar.gz
