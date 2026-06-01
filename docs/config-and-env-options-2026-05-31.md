# CryptoTimeSeries Config And Env Options

Last updated: 2026-05-31

This document lists externally configurable options discovered from source code:
- Code-level configuration options (public init and runtime knobs)
- Environment variables (ENV) with intent and modules using them

Notes:
- Scope is options that can be changed without editing internal algorithm code.
- "Modules" refers to package/script owners reading each option.
- Test-only env keys are marked accordingly.

## 1) Code Configuration Options

### 1.1 EnvConfig mode and init options

| Option | Allowed values | Intent | Used by modules |
|---|---|---|---|
| EnvConfig.init(mode) | EnvConfig.test, EnvConfig.production, EnvConfig.training | Selects global data/auth mode, base universe, and default data folder naming | EnvConfig, CryptoXch, Ohlcv, Features, Targets, Classify, Trade, scripts |
| EnvConfig.init(; newdatafolder) | Bool | Appends run-id to generated data folder name when true | EnvConfig, CryptoXch tests |
| EnvConfig.init(; authname) | String or nothing | Selects named auth tuple from auth.json/authtest.json | EnvConfig, exchange adapters via EnvConfig.authorization |
| EnvConfig.configmode | Mode enum | Current active mode queried by many modules | Broadly across workspace |

### 1.2 EnvConfig data/path format options

| Option | Allowed values | Intent | Used by modules |
|---|---|---|---|
| EnvConfig.setdfformat!(format) | :jdf, :arrow, :csv | Default dataframe storage format for read/write helpers | EnvConfig and callers of table/read/write helpers |
| EnvConfig.setcoinspath!(folder) | path string | Override coin data root used by coinfile/coinfolder helpers | EnvConfig and data-loading callers |
| EnvConfig.setdebugpath(folder) | path string or nothing | Override debug output path helper target | EnvConfig and debug workflows |
| EnvConfig.cryptoquote | usually "USDT" | Default quote currency used by symbol/path helpers | EnvConfig, Trade, CryptoXch, scripts |

### 1.3 Trade runtime knobs (cache-level)

These are runtime options configured through Trade.TradeCache().mc and are relevant for operational behavior.

| Option | Typical values | Intent | Used by modules |
|---|---|---|---|
| mc[:trademode] | Trade.buysell, Trade.closeonly, Trade.quickexit, Trade.notrade | Enables/disables opening/closing behaviors | Trade |
| mc[:strategy_engine] | :getgainsalgo (legacy value still accepted) | Strategy source metadata; runtime API path is mandatory | Trade |
| mc[:maxassetfraction] | Float | Exposure cap per asset | Trade |
| mc[:maxbudgetquote] | Float or nothing | Global quote budget cap for sizing | Trade |
| mc[:budgetsafetymargin] | Float [0,1) | Safety discount on budget | Trade |
| mc[:reloadtimes] | Time[] | Schedule for trade universe/config refresh | Trade |

### 1.4 Objective 7 runtime-only steady state (2026-06-01)

- Strategy runtime integration in Trade is mandatory; there is no legacy runtime toggle branch.
- `mc[:strategy_engine]` is treated as strategy-source metadata only and does not select between runtime and legacy execution paths.
- Trade runtime path no longer calls `Classify.advice` directly; strategy snapshots are sourced via `TradingStrategy` runtime interfaces.
- Legacy `Classify.TradeAdvice`-typed execution interfaces in Trade were removed in favor of `StrategyAdvice` execution input.

## 2) Environment Variables

### 2.1 Core trading/runtime flags

| Env var | Intent | Used by modules |
|---|---|---|
| CTS_ASYNC_ENGINE_ENABLED | Enables Objective-4 async engine path | Trade |
| CTS_ASYNC_SHADOW_MODE | Enables async-vs-sync shadow compare mode | Trade |
| CTS_WS_MARKETDATA_ENABLED | Enables websocket market-data ownership path | Trade |
| CTS_OHLCV_GAP_BACKFILL_ON_TRADABLE | Enables OHLCV gap backfill in tradable checks | Trade |

### 2.2 Run identity and production test gating

| Env var | Intent | Used by modules |
|---|---|---|
| CTS_RUN_ID | Correlation/run ID used for tradelog/audit partitioning | CryptoXch, scripts/BoundsEstimator, scripts/TrendDetector, scripts/simulationcheck |
| CTS_RUN_PRODUCTION_TESTS | Allows production-only CryptoXch tests when explicitly enabled | CryptoXch tests |

### 2.3 TradeLog persistence controls

| Env var | Intent | Used by modules |
|---|---|---|
| CTS_TRADELOG_ENABLED | Global TradeLog write on/off | TradeLog, scripts/benchmark_tradesim_audit, TradeLog tests |
| CTS_TRADELOG_SIMULATION_ENABLED | TradeLog write on/off for simulation run_mode | TradeLog, scripts/benchmark_tradesim_audit, TradeLog tests |
| CTS_TRADELOG_ROOT | Overrides TradeLog storage root folder | TradeLog, CryptoXch tests |

### 2.4 Exchange credentials and exchange-specific behavior

| Env var | Intent | Used by modules |
|---|---|---|
| BYBIT_APIKEY | Bybit API key (if not resolved from auth tuple) | Bybit |
| BYBIT_SECRET | Bybit API secret (if not resolved from auth tuple) | Bybit |
| KRAKEN_APIKEY | Kraken spot API key; also fallback for Kraken Futures key | KrakenSpot, KrakenFutures |
| KRAKEN_SECRET | Kraken spot secret; also fallback for Kraken Futures secret | KrakenSpot, KrakenFutures |
| KRAKEN_FUTURES_APIKEY | Kraken Futures API key (preferred over KRAKEN_APIKEY) | KrakenFutures |
| KRAKEN_FUTURES_SECRET | Kraken Futures secret (preferred over KRAKEN_SECRET) | KrakenFutures |
| KRAKEN_FUTURES_OMIT_NONCE_READS | Omits nonce on selected private read endpoints | KrakenFutures |
| KRAKEN_FUTURES_INCLUDE_NONCE_IN_POST_BODY | Adds nonce to POST body in addition to header | KrakenFutures |
| KRAKEN_FUTURES_NONCE_MODE | Nonce mode selection (ms or ns) | KrakenFutures |
| KRAKEN_ONLINE_TESTS | Enables/disables Kraken online integration test | KrakenSpot tests |

### 2.5 EnvConfig and filesystem roots

| Env var | Intent | Used by modules |
|---|---|---|
| ONEDRIVE_ROOT | Optional OneDrive root for environment preflight checks and legacy path expectations | EnvConfig |

### 2.6 Script-level simulation/real-trading options

#### tradesim.jl and benchmark_tradesim_audit.jl

| Env var | Intent | Used by modules |
|---|---|---|
| TRADESIM_WHITELIST | Comma-separated base/symbol whitelist for simulation runs | scripts/tradesim, scripts/benchmark_tradesim_audit |
| TRADESIM_STARTDT | Simulation start DateTime override | scripts/tradesim, scripts/benchmark_tradesim_audit |
| TRADESIM_ENDDT | Simulation end DateTime override | scripts/tradesim, scripts/benchmark_tradesim_audit |
| TRADESIM_MAX_BUDGET_QUOTE | Simulation budget cap override | scripts/tradesim |
| TRADESIM_MAX_BUDGET_USDT | Backward-compatible alias for TRADESIM_MAX_BUDGET_QUOTE | scripts/tradesim |
| TRADESIM_BENCH_STARTDT | Benchmark harness start DateTime override | scripts/benchmark_tradesim_audit |
| TRADESIM_BENCH_ENDDT | Benchmark harness end DateTime override | scripts/benchmark_tradesim_audit |
| TRADESIM_BENCH_SHOW_CHILD_OUTPUT | Show child-process output in benchmark harness | scripts/benchmark_tradesim_audit |

#### tradereal.jl

| Env var | Intent | Used by modules |
|---|---|---|
| TRADEREAL_MAX_BUDGET_QUOTE | Live run budget cap override | scripts/tradereal |
| TRADEREAL_MAX_BUDGET_USDT | Backward-compatible alias for TRADEREAL_MAX_BUDGET_QUOTE | scripts/tradereal |
| TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE | Enable/disable KrakenSpot startup capability probe | scripts/tradereal |
| TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE_BASE | Force base coin for KrakenSpot startup probe | scripts/tradereal |
| TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE | Enable/disable KrakenFutures startup capability probe | scripts/tradereal |
| TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE_BASE | Force base coin for KrakenFutures startup probe | scripts/tradereal |

### 2.7 Test-only toggles

| Env var | Intent | Used by modules |
|---|---|---|
| RUN_SLOW_BTC_TREND04 | Enables skipped slow test path for BTC Trend04 test | Targets tests |

## 3) Value conventions

For boolean-like env vars, code commonly accepts these case-insensitive values:
- true set: 1, true, yes, on
- false set: 0, false, no, off

When an env var is not set, modules use their own default values (documented near each read site).

## 4) Operational Classification

Legend:
- production-safe: Intended for normal production/live operation.
- test-only: Intended only for tests, diagnostics, or explicit experimental runs.
- deprecated-alias: Backward-compatibility key; prefer the replacement key.

### 4.1 Environment variables classification

| Env var | Classification | Notes |
|---|---|---|
| BYBIT_APIKEY | production-safe | Live/testnet credential input for Bybit adapter |
| BYBIT_SECRET | production-safe | Live/testnet credential input for Bybit adapter |
| KRAKEN_APIKEY | production-safe | Kraken spot credential; fallback source for KrakenFutures |
| KRAKEN_SECRET | production-safe | Kraken spot credential; fallback source for KrakenFutures |
| KRAKEN_FUTURES_APIKEY | production-safe | Preferred KrakenFutures API key |
| KRAKEN_FUTURES_SECRET | production-safe | Preferred KrakenFutures secret |
| KRAKEN_FUTURES_OMIT_NONCE_READS | production-safe | Exchange-API behavior tuning for private read calls |
| KRAKEN_FUTURES_INCLUDE_NONCE_IN_POST_BODY | production-safe | Exchange-API compatibility toggle |
| KRAKEN_FUTURES_NONCE_MODE | production-safe | Nonce mode selection (ms/ns) |
| CTS_RUN_ID | production-safe | Run correlation/partition identifier |
| CTS_TRADELOG_ENABLED | production-safe | Global TradeLog persistence switch |
| CTS_TRADELOG_SIMULATION_ENABLED | production-safe | TradeLog simulation persistence switch |
| CTS_TRADELOG_ROOT | production-safe | TradeLog root override |
| CTS_ASYNC_ENGINE_ENABLED | test-only | Objective 4 rollout/experiment toggle |
| CTS_ASYNC_SHADOW_MODE | test-only | Shadow-compare safety mode for rollout verification |
| CTS_WS_MARKETDATA_ENABLED | test-only | Objective 4 websocket market-data ownership toggle |
| CTS_OHLCV_GAP_BACKFILL_ON_TRADABLE | test-only | Objective 4 tradable-gap backfill toggle |
| CTS_RUN_PRODUCTION_TESTS | test-only | Allows production test suite execution |
| KRAKEN_ONLINE_TESTS | test-only | Enables online KrakenSpot integration tests |
| RUN_SLOW_BTC_TREND04 | test-only | Enables slow Targets test path |
| TRADESIM_WHITELIST | test-only | Script-level simulation input override |
| TRADESIM_STARTDT | test-only | Script-level simulation interval override |
| TRADESIM_ENDDT | test-only | Script-level simulation interval override |
| TRADESIM_BENCH_STARTDT | test-only | Benchmark harness override |
| TRADESIM_BENCH_ENDDT | test-only | Benchmark harness override |
| TRADESIM_BENCH_SHOW_CHILD_OUTPUT | test-only | Benchmark diagnostics verbosity |
| TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE | test-only | Startup capability probe toggle (operational diagnostic) |
| TRADEREAL_KRAKENSPOT_STARTUP_ORDER_PROBE_BASE | test-only | Startup probe base override |
| TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE | test-only | Startup capability probe toggle (operational diagnostic) |
| TRADEREAL_KRAKENFUTURES_STARTUP_ORDER_PROBE_BASE | test-only | Startup probe base override |
| ONEDRIVE_ROOT | test-only | Used by EnvConfig preflight helper; current runtime path defaults to $HOME/crypto |
| TRADESIM_MAX_BUDGET_USDT | deprecated-alias | Prefer TRADESIM_MAX_BUDGET_QUOTE |
| TRADEREAL_MAX_BUDGET_USDT | deprecated-alias | Prefer TRADEREAL_MAX_BUDGET_QUOTE |
| TRADESIM_MAX_BUDGET_QUOTE | production-safe | Primary budget override key used by simulation script |
| TRADEREAL_MAX_BUDGET_QUOTE | production-safe | Primary budget override key used by live script |

### 4.2 Code-level option classification

| Option | Classification | Notes |
|---|---|---|
| EnvConfig.init(mode) | production-safe | Core runtime mode selector |
| EnvConfig.init(; authname) | production-safe | Auth tuple selection |
| EnvConfig.init(; newdatafolder) | test-only | Mostly used for isolated runs/tests |
| EnvConfig.setdfformat! | production-safe | Storage format preference |
| EnvConfig.setcoinspath! | production-safe | Path override for coin data |
| EnvConfig.setdebugpath | test-only | Debug artifact redirection |
| Trade mc[:trademode] | production-safe | Main runtime behavior selection |
| Trade mc[:strategy_engine] | production-safe | Runtime strategy source metadata (runtime API path is mandatory) |
| Trade mc[:maxassetfraction] | production-safe | Risk/exposure guardrail |
| Trade mc[:maxbudgetquote] | production-safe | Capital cap |
| Trade mc[:budgetsafetymargin] | production-safe | Sizing safety margin |
| Trade mc[:reloadtimes] | production-safe | Operational refresh schedule |

## 5) Source index (primary read sites)

- EnvConfig/src/EnvConfig.jl
- Trade/src/Trade.jl
- TradeLog/src/TradeLog.jl
- CryptoXch/src/CryptoXch.jl
- Bybit/src/Bybit.jl
- KrakenSpot/src/KrakenSpot.jl
- KrakenFutures/src/KrakenFutures.jl
- scripts/tradesim.jl
- scripts/tradereal.jl
- scripts/benchmark_tradesim_audit.jl
