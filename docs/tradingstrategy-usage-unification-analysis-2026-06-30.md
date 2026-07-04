# TradingStrategy Usage Unification Analysis

Date: 2026-06-30
Scope: Compare how TradingStrategy is used by TrendDetector, tradesim, and tradereal.

## Summary

- tradesim and tradereal are unified through the same Trade lifecycle and per-tick loop implementation.
- TrendDetector uses explicit TradingStrategy replay APIs with a different orchestration model (range/threshold batch evaluation), which is intentionally different from trade execution loops.

## A. Preparation and Init Phase Differences

### 1) tradereal

- Initializes production environment and quote setup.
- Loads runtime classifier from configured model folder.
- Builds Xch cache in live mode (enddt is nothing).
- Builds Trade cache and applies strategy via Trade.apply_tradingstrategy!.
- Defers base preparation to Trade runtime initialization.

Call-path details:

- scripts/tradereal.jl lines 157-160: EnvConfig initialization and classifier loading.
- scripts/tradereal.jl lines 186-194: Xch cache, Trade cache, apply_tradingstrategy!.
- Trade/src/Trade.jl lines 2085-2092: _ensure_tradeloop_initialized! triggers tradeselection! and _prepare_strategy_runtime_for_cfg!.
- Trade/src/Trade.jl line 1529: _prepare_strategy_runtime_for_cfg! calls TradingStrategy.preparebases!.

### 2) tradesim

- Initializes test environment and quote setup.
- Loads runtime classifier from configured model folder.
- Builds Xch cache with fixed startdt and enddt for backtest period.
- Seeds and verifies quote budget for simulation.
- Builds Trade cache and applies strategy via Trade.apply_tradingstrategy!.
- Uses same Trade runtime initialization path as tradereal for base preparation.

Call-path details:

- scripts/tradesim.jl lines 450-454: Xch cache with bounded replay window.
- scripts/tradesim.jl lines 458-459: simulation quote balance seeding.
- scripts/tradesim.jl lines 462-463: apply_tradingstrategy!.
- Trade/src/Trade.jl lines 2085-2092 and line 1529: same init and preparebases! path as tradereal.

#### Why are _prepare_strategy_runtime_for_cfg! and _ensure_tradeloop_initialized! different functions while they both belong to teh init phase?

They are split because they serve two different responsibilities, even though both are used during startup.

_ensure_tradeloop_initialized! is orchestration logic
It decides whether initialization is needed at all, and if needed it performs the full bootstrap sequence:
- check if cfg is empty
- run trade selection
- filter cfg to tradable rows
- call strategy-runtime preparation
- See: Trade.jl:2085, Trade.jl:2089, Trade.jl:2090, Trade.jl:2091

_prepare_strategy_runtime_for_cfg! is a focused primitive
- It only takes an already prepared cfg and synchronizes TradingStrategy runtime bases via preparebases!:
- validate cfg has basecoin and rows
- get runtime object
- call TradingStrategy.preparebases!
- See: Trade.jl:1521, Trade.jl:1529

##### Why this split is useful
The focused function is reused outside first-time init, especially on periodic universe refresh:
- _maybe_refresh_tradeselection! reruns selection and then calls the same runtime-prep helper
- See: Trade.jl:1867, Trade.jl:1873, Trade.jl:1875

##### So conceptually:

_ensure_tradeloop_initialized! = “Do we need init, and run the init workflow”  
_prepare_strategy_runtime_for_cfg! = “Given cfg, sync TsCache bases”  
If they were merged, you would duplicate the prep logic in both startup and refresh paths or make refresh call a heavier function with extra side effects.

### 3) TrendDetector

- Does not use Trade cache.
- Creates TsCache and Xch cache directly.
- Uses explicit replay prepare phase per coin/range/threshold via preparereplaytrades!.
- Uses explicit replay process phase via processreplaygains!.

Call-path details:

- scripts/TrendDetector.jl lines 534-542: TsCache and Xch cache creation.
- scripts/TrendDetector.jl lines 645-654 and 673-682: replay preparation calls.
- scripts/TrendDetector.jl lines 655-661 and 683-689: replay processing calls.
- TradingStrategy/src/TradingStrategy.jl lines 1005-1015: preparereplaytrades! contract.
- TradingStrategy/src/TradingStrategy.jl lines 1046-1082: processreplaygains! contract.

#### Why are preparereplaytrades!and and processreplaygains! separated?

They are separated to enforce a clean two-phase lifecycle and make misuse visible.

1) Different responsibilities
    - preparereplaytrades! builds and normalizes replay state: syncs the trades frame, sets label and score, resets replay columns, and injects metadata. See TradingStrategy.jl:1005.
    - processreplaygains! assumes that prepared state exists and only runs gain simulation. See TradingStrategy.jl:1046.
2) Fail-fast contract
    - processreplaygains! explicitly validates that preparation happened (required columns, closeprices coverage, pair consistency) before simulation. See TradingStrategy.jl:1017.
    - This prevents hidden fallback behavior and catches lifecycle bugs early.
3) Clear call-site semantics in TrendDetector
    - TrendDetector now calls prepare then process explicitly for each evaluation lane, so lifecycle is visible in code. See TrendDetector.jl:645 and TrendDetector.jl:655.
    - Same explicit pattern is used for the true-label lane. See TrendDetector.jl:673 and TrendDetector.jl:683.
4) Better reuse and testing
    - You can test preparation and processing independently.
    - You can inspect/diagnose prepared trades state before running gains.
    - The combined convenience wrapper still exists (getgains), but explicit split avoids hidden work and matches the refactor goal. See TradingStrategy.jl:1091.

#### How is in TrendDetector context Xch triggered to create a new row including pair and opentime?

Looking at TrendDetector, here's the flow:

- TrendDetector → TradingStrategy → Xch (row creation)
- TrendDetector TrendDetector.jl:571-576:
- Calls TradingStrategy.preparereplaytrades!(ts, xc, coin, resultsview, ...)
  - resultsview already has :opentime from prediction results
  - TradingStrategy TradingStrategy.jl:607-625:
- preparereplaytrades! calls _synctradesframe!(ts, xc, base, predictionsdf, scores, labels, ...)
  - Creates trades DataFrame from predictionsdf (keeping :opentime)
  - Calls Xch.settrades!(xc, base, quotecoin, tradesdf)
- Xch (XchCore.jl):
  - settrades! calls _ensuretradesidentity!(normalized, pairkey; basekey=basekey)
  - This function populates the :pair column with the trading pair key
  - Stores the dataframe in xc.pairstates[pairkey]
  - Individual rows are created separately:

Xch.ensuretradesrow!(xc, base, quotecoin, opentime) is called from Trade/src during live runtime
This creates one row at a time for the given opentime if it doesn't exist
But in TrendDetector's replay mode, all rows are batch-created upfront via settrades!
So in TrendDetector context: rows with :pair and :opentime are created bulk when preparereplaytrades! calls Xch.settrades!, not one-by-one.

## B. Per-Sample Loop Differences

### 1) tradesim and tradereal (unified loop engine)

Both entrypoints call Trade runners that execute the same internal loop engine and tick-step logic.

- run_backtest! and run_live! both call _ensure_tradeloop_initialized! then _run_tradeloop!.
- _run_tradeloop! iterates over cache.xc and executes _tradestep! once per tick.
- _tradestep! collects strategy rows through TradingStrategy.gettradesrows! and translates rows into trade advice tuples.

Call-path details:

- Trade/src/Trade.jl lines 2147-2150: run_backtest!.
- Trade/src/Trade.jl lines 2158-2161: run_live!.
- Trade/src/Trade.jl lines 2112-2119: _run_tradeloop! loop body.
- Trade/src/Trade.jl lines 1997-2023: _tradestep! core flow.
- Trade/src/Trade.jl line 1820: gettradesrows! call from _collect_strategy_advices.

TradingStrategy per-row semantics in this loop:

- gettradesrow! performs classify phase then apply-row phase.
- gettradesrows! batches that per base.

Call-path details:

- TradingStrategy/src/TradingStrategy.jl lines 562-599: classify phase.
- TradingStrategy/src/TradingStrategy.jl lines 608-671: row-application phase.
- TradingStrategy/src/TradingStrategy.jl lines 675-691: gettradesrow! and gettradesrows! wrappers.

### 2) tradereal versus tradesim behavior inside shared loop

The code path is shared, but runtime behavior differs through simulation mode checks:

- In simulation mode, assets_after is refreshed after trade actions.
- In live mode, open-position safety warnings for missing close orders are evaluated.

Call-path details:

- Trade/src/Trade.jl line 2064: post-trade asset refresh branch by sim mode.
- Trade/src/Trade.jl lines 2068-2075: live-only close-order coverage warning path.

### 3) TrendDetector per-sample orchestration

TrendDetector does not run a minute-by-minute execution loop. It uses batch replay loops:

- Outer loop grouped by rangeid.
- Inner loop over threshold combinations.
- For each combination: prepare replay trades, process replay gains, collect diagnostics.

Call-path details:

- scripts/TrendDetector.jl lines 620-621: outer range loop.
- scripts/TrendDetector.jl line 644: threshold loop.
- scripts/TrendDetector.jl lines 645-661 and 673-689: prepare then process for predicted and true lanes.

## C. Lifecycle Contract Alignment Status

Aligned:

- Trade path now performs base preparation at selection time, not inside minute advice collection.
- tradesim and tradereal are aligned by sharing one Trade execution lifecycle.
- TrendDetector uses explicit replay prepare and replay process APIs.

Intentionally different:

- TrendDetector remains a replay analytics orchestrator, not an execution loop with order and portfolio side effects.

## D. Practical Implication

Unification is strongest at API contract level:

- explicit runtime object ownership through TsCache
- explicit prepare phase versus process phase
- no hidden prepare in per-sample trading advice collection in Trade path

Operationally, two loop families remain by design:

- execution loop family: tradesim and tradereal via Trade
- replay evaluation family: TrendDetector via replay APIs

## E. Decision Logic Anti-Drift Guarantees

This section explains how TradingStrategy decision logic is kept aligned across tradesim/tradereal and TrendDetector.

### Core guardrails

- Shared algorithm contract in both paths:
    - Execution path applies `strategy_config.algorithm` in row update.
    - Replay path runs `simulate_gains!` with `strategy.algorithm`.
- Shared strategy parameter object flow:
    - tradesim/tradereal apply the selected `StrategyConfig` into `TsCache` through `Trade.apply_tradingstrategy!` and `TradingStrategy.apply_strategy!`.
    - TrendDetector passes `cfg.tradingstrategy` directly into `processreplaygains!`.
- Shared schema/state preparation discipline:
    - All paths ensure Trades schema contributors are applied.
    - Replay state is normalized/reset through `preparereplaytrades!` before gain processing.
- Fail-fast validation:
    - `processreplaygains!` asserts prepared replay columns and `closeprices` coverage.
    - Signature mismatch in strategy algorithm raises explicit error instead of drifting silently.

### Why drift risk is reduced

- The decision kernel is not duplicated; both paths call the same algorithm contract.
- Lifecycle is explicit (`prepare` then `process`) and validated.
- Hidden fallback behavior is intentionally removed in replay processing.

### Remaining intentional differences (not considered drift bugs)

- Orchestration goal differs:
    - tradesim/tradereal execute one active runtime policy with order side effects.
    - TrendDetector evaluates batch replay ranges and threshold sweeps.
- State source differs:
    - execution uses evolving exchange/cache state.
    - replay uses frozen prediction/result windows.

### Recommended parity hardening test

- Add a parity test that feeds an identical fixed sample window into:
    - execution row update path (`gettradesrow!`/`gettradesrows!`), and
    - replay processing path (`preparereplaytrades!` + `processreplaygains!`).
- Assert row-level equality (or bounded tolerance) for `label`, `score`, and limit columns before order-execution side effects.

## F. Classifier Call Sequence Guarantees

This section documents how classifier call ordering is enforced across execution and replay flows.

### Sequence control in execution path (tradesim/tradereal)

- Centralized classifier wiring:
    - `TsCache` creates or resolves exactly one classifier instance via `Classify.resolveclassifier`.
- Mandatory prepare-before-process:
    - Trade initialization runs `tradeselection!` and then `_prepare_strategy_runtime_for_cfg!`.
    - `_prepare_strategy_runtime_for_cfg!` calls `TradingStrategy.preparebases!`.
    - `preparebases!` applies `Classify.addbase!` first and `Classify.supplement!` second.
- Per-sample order is fixed:
    - `_collect_strategy_advices` calls `TradingStrategy.gettradesrows!`.
    - `gettradesrow!` runs `_classify_base_advice!` before `_apply_base_advice_row!`.

### Re-classification gating to keep sequencing deterministic

- `_should_skip_classifier` determines whether to reuse prior advice or call `Classify.advice` again, using:
    - classify staleness interval, and
    - relative price-delta trigger.
- Gate state is tracked per base in `classifier_gate_state` and updated only on fresh classifier outputs.

### Sequence control in TrendDetector replay path

- Classifier prediction phase is separate from replay gain phase:
    - `getmaxpredictionsdf` computes labels and scores via classifier prediction.
    - Replay gain path then consumes those prepared labels/scores through:
        - `preparereplaytrades!`, then
        - `processreplaygains!`.
- Replay gain processing does not re-call classifier for each replay row.

### Fail-fast boundaries that prevent sequence drift

- `processreplaygains!` validates replay readiness (required columns and closeprice coverage) before simulation.
- Incorrect algorithm signature raises explicit error instead of silently changing behavior.

### Practical interpretation

- Execution flow guarantees: prepare runtime bases first, then classify/apply per tick.
- Replay flow guarantees: predict first, then prepare replay tables, then process gains.
- These explicit boundaries ensure call order is reproducible and auditable, rather than implicit.
