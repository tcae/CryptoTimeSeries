# JDF to Arrow conversion

## Current storage

Per-coin artifacts live under `~/crypto/<exchange>/<BASE>-<QUOTE>/`.
The default exchange root is `~/crypto/Bybit`; exchange adapters select their
own roots during initialization. Calling `EnvConfig.setcoinspath!()` without
arguments resets the root to Bybit, not the retired `~/crypto/coins` folder.

For example:

```text
Bybit/BTC-USDT/
  ohlcv.arrow
  regression/
    regr60.arrow
    regrstd60.arrow
  features/
    F09.arrow
  targets/
    T09.arrow
  sets/
    ...
```

Features004 and its cache format, conversion, maintenance scripts, and tests
have been removed. Features006 uses standalone regression stores.
Features005 retains an independent `f5.arrow` regression cache and does not
read or migrate retired feature caches. Cockpit regression bands reuse aligned
standalone regression stores or are calculated on demand from minute OHLCV.

Classifier artifacts live under `~/crypto/classifiers` (or
`~/crypto/debug/classifiers` in test mode). Experiment-specific tables remain
under the configured logs/debug root.

## Legacy shared OHLCV

[Jdf2Arrow](../scripts/Jdf2Arrow.jl) can copy remaining legacy OHLCV JDF caches
from `~/crypto/Features/OHLCV` into `~/crypto/Bybit/<BASE>-<QUOTE>/ohlcv.arrow`.
Shared conversion and cross-check explicitly select Bybit, irrespective of the
previously active exchange root.

Only `ohlcv` is supported for shared conversion/cross-check; unsupported
artifact selections fail explicitly. Conversion reads the selected JDF source,
preserves its base/quote identity, and leaves the source untouched. It refuses to
overwrite existing destination OHLCV caches; failures are logged and included
in the summary as `status=error`.

```sh
julia --project=. scripts/Jdf2Arrow.jl shared=true artifacts=ohlcv bases=BTC,ETH
julia --project=. scripts/Jdf2Arrow.jl crosscheck=true artifacts=ohlcv bases=BTC,ETH savesummary=false
```

By default, summaries are saved as `jdf2arrow_shared_summary.arrow` or
`jdf2arrow_shared_crosscheck.arrow` under the Bybit root. The shared cross-check
reports destination presence and byte-size parity; it is not a value-by-value
verification.

## Experiment tables

Convert JDF artifacts inside a logs/config folder to Arrow copies:

```sh
julia --project=. scripts/Jdf2Arrow.jl folder=Trend-009-training artifacts=features,results,maxpredictions
```

Relative folders are resolved under the logs root; absolute folders are used
directly. Default artifacts are `features`, `results`, and `maxpredictions`.
Outputs use config-scoped subfolders such as `features/all.arrow`,
`results/all.arrow`, and `predictions/maxpredictions.arrow`.

Conversion verifies row count, column count, column names, and available key
columns. Legacy JDF inputs are retained. A conversion summary is written unless
`savesummary=false` is supplied.

## Historical backfill

Inspect missing Arrow copies first:

```sh
julia --project=. scripts/Jdf2Arrow.jl scan=true reportonly=true missingonly=true
```

Then convert selected folders:

```sh
julia --project=. scripts/Jdf2Arrow.jl scan=true folders=Trend-029-test,Bounds-001-test
```

The default scan root is the logs directory. Supply `root=<path>` to select
another root. `missingonly=true` is the default.
