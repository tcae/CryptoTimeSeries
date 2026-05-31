# Trading Loop Integration Plan (Created 2026-05-10)

## 2026-05-18 Interface Revision Plan (Trade, TradingStrategy, Classify)

### Scope constraints
- Keep `AbstractClassifier` as the only decision support abstraction for now (no `AbstractRegressor` addition in this slice).
- Move score threshold ownership (`openthreshold`, `closethreshold`) to `TradingStrategy` and avoid redefining these values in `Trade` runtime knobs.
- Replace fragile simulated market snapshots from persisted trade config files with an on-the-fly simulated marketview derived from OHLCV at simulation time.
- Keep two explicit config layers configured by scripts:
	- strategy layer: targets, features, classifier, trading strategy
	- trade runtime layer: allocation, refresh cadence, and execution controls while referencing the strategy layer
- Preserve same-minute close-and-open-opposite trend behavior as a first-class use case.

### Progress ledger
- [x] Persist plan and checklist in repository doc
- [x] Add explicit two-layer config structs and script-facing apply functions in `Trade`
- [x] Add on-the-fly simulated marketview provider for `tradeselection!`
- [x] Route `tradeselection!` through live-vs-simulated marketview resolver
- [x] Ensure strategy thresholds are sourced from `TradingStrategy` config application path
- [x] Add/adjust tests for simulated marketview and same-minute reversal expectations
- [x] Validate with workspace tests and record follow-up deltas
- [x] Introduce `StrategyAdvice` bridge with optional strategy-provided limit price
- [x] Route trade-step advice handling through `StrategyAdvice` collection path
- [x] Support same-minute close-then-open reversal expansion in advice handling
- [x] Re-enable periodic trade-selection refresh in trade step with deterministic minute-level cadence tests

### Notes
- First target workflow: TrendDetector config `046` with `gain_limit_reversal!`.
- This section is the active interruption-safe tracker for the current interface refactor.

## 2026-05-30 Progress Update: Audit Renamed to TradeLog (Private Trader Scope)

### Design decision
- Replace "audit trail" terminology with `TradeLog` for the active trading runtime.
- Relax tamper-evidence requirements (no organizational hash-chain guarantee required).
- Keep structured context logging focused on practical trader diagnostics:
	- order intent vs execution outcome
	- fee/cost cross-checks
	- replay against historic OHLCV windows

### Runtime policy
- Canonical log root defaults to `$HOME/crypto/tradelog`.
- Primary enable flags are now:
	- `CTS_TRADELOG_ENABLED`
	- `CTS_TRADELOG_SIMULATION_ENABLED`
- Backward-compatible fallback to old `CTS_AUDIT_*` flags remains temporarily during migration.

### Migration status
- [x] Added new workspace package `TradeLog` with append-only JSONL logging API compatible with current `Trade` and `CryptoXch` call paths.
- [x] Switched `Trade` and `CryptoXch` imports/dependencies from `TradeAudit` to `TradeLog`.
- [x] Disabled hash-chain persistence in the default write path (`writeeventwithhash` now behaves as plain append for compatibility).
- [x] Follow-up: renamed remaining internal helper identifiers containing "audit" to TradeLog-oriented names where this improved readability without behavior changes; kept compatibility aliases for transition.

## 2026-05-30 Progress Update: KrakenFutures Managed Close Routing and Limit Drift Investigation

### Completed in code
- [x] Added routed symbol-aware amend path in `CryptoXch` and switched managed close amend callsites to pass symbol explicitly.
- [x] Added orderid-only `KrakenFutures.amendorder` overload so routed amend calls no longer fail with method dispatch gaps.
- [x] Removed open-order lookup fallback from Kraken futures single-order resolution and switched to `/orders/status` for direct order-status lookup.
- [x] Eliminated the immediate `/openorders?order_id=...` fallback path that had triggered intermittent `authenticationError` during amend handling.
- [x] Added explicit TrendDetector coinspath routing to `coins_bybit` so training/eval reads the active Bybit OHLCV store.

### Validation snapshot
- [x] Method presence checks for routed amend and Kraken futures order/amend overloads passed in workspace runtime checks.
- [x] Static diagnostics for touched files were clean after the amend-path fixes.
- [x] Live tradereal runs now proceed past the earlier amend dispatch/auth lookup failures.

### Current investigation status (open)
- [x] Confirmed repeated warnings in the latest Kraken futures tradereal log: open positions without active full close coverage.
- [x] Confirmed two bases (notably TON and ONDO in the latest run) remained materially above `maxassetquote` across multiple minutes.
- [x] Confirmed open-buy sizing path enforces cap intent via `remaininglongcapacityquote` guard.
- [ ] Resolve root cause of sustained above-cap exposure in live loop (likely close-order fill/maintenance behavior and/or valuation basis mismatch under runtime conditions).
- [ ] Add targeted runtime diagnostics in `Trade` for per-base: exposure quote, required reduction quote, managed close order qty, and observed fill delta.

### Next actions
- [ ] Instrument one focused live diagnostic slice for affected bases only (TON, ONDO) to avoid broad log noise.
- [ ] Decide policy adjustment after diagnostics: either stronger close escalation/execution policy for above-cap inventory, or revised valuation/trigger alignment for cap enforcement and strong-close promotion.
- [ ] Add regression tests that keep a base above cap for multiple ticks and assert reduction trajectory under managed close maintenance.

## Goal
Integrate `algorithm03!` into a production-ready trading loop with exchange abstraction, TradeLog-grade logging, asynchronous orchestration, and a non-blocking dashboard, then extend the exchange layer with Interactive Brokers.

## 2026-05-31 Objective: Exchange-Concept Equity and Mixed Spot+Margin Capacity

### Intent
- Align runtime account semantics with exchange-native concepts for both reporting and sizing.
- Track net worth development using exchange equity semantics instead of reconstructed exposure totals.
- Size new openings from exchange-available capacity with `SAFETY_MARGIN` applied, while retaining exposure metrics for diagnostics.
- Treat mixed spot-long plus spot-margin-short as the default/most general account case.

### Design decision (ownership boundary)
- Exchange adapters own exchange semantics extraction:
	- Canonical account snapshot fields: `equity_quote`, `available_opening_quote`, optional `initial_margin_quote`, optional `maintenance_margin_quote`, timestamp, source metadata.
	- KrakenSpot adapter derives mixed-case capacity (cash long lane and margin short lane) from Kraken account endpoints.
	- KrakenFutures adapter derives collateral/equity/margin capacity from futures account endpoints.
- `Trade` owns strategy/runtime policy:
	- Applies `SAFETY_MARGIN` and global caps.
	- Applies per-asset allocation constraints and directional trade policy.
	- Consumes one normalized account-capacity snapshot regardless of venue.

### Mixed-case policy (KrakenSpot)
- Do not treat `free quote` alone as total opening capacity in mixed long+short spot margin accounts.
- Model capacity as side-aware lanes:
	- long lane: quote cash available for spot buys.
	- short lane: margin availability for new shorts (after existing margin usage).
- For new open orders, choose the lane by intended side and apply safety haircut before final caps.

### Budget formulas (canonical)
- `safe_opening_quote = max(0, available_opening_quote * (1 - SAFETY_MARGIN))`
- `effective_budget_quote = min(safe_opening_quote, MAX_BUDGET_QUOTE)` when cap is configured.
- Keep exposure-oriented values (`project_net_exposure_quote`, `project_gross_exposure_quote`) as secondary diagnostics only.

### Implementation slices
- [ ] Add `CryptoXch.accountcapacity(xc; role=...)` normalized API and snapshot struct/NamedTuple.
- [ ] Implement KrakenSpot mixed-case extractor (cash long lane + margin short lane) and map to normalized snapshot.
- [ ] Implement KrakenFutures extractor and map to normalized snapshot.
- [ ] Add `Trade` budget mode switch with default `:exchange_capacity` for live runtimes and compatibility fallback for simulation/tests.
- [ ] Update startup and periodic logs to show: `equity_quote`, `available_opening_quote`, `safe_opening_quote`, `effective_budget_quote`, and exposure diagnostics.
- [ ] Add tests for mixed spot+margin scenarios and side-aware lane selection behavior.

### Exit criteria
- Net-worth reporting in live loops is sourced from exchange-equity semantics.
- Opening budget uses exchange available capacity with `SAFETY_MARGIN` and cap policy.
- Mixed spot+margin KrakenSpot behavior is validated by tests for both long-open and short-open paths.
- Existing simulation/backtest flows remain available via explicit compatibility mode until migration is complete.

## 2026-05-25 Trade Selection Ownership Integration

### Design intent
Constrain sell-side trading to positions attributable to this trading robot while keeping exchange/account scopes and long-vs-short inventory separate.

### Requirements
- Whitelist constrains opening trades only; it must not implicitly authorize sells of externally acquired positions.
- Sell eligibility must be based on robot-owned exposure, not raw wallet presence.
- Robot-owned exposure must be partitioned by routed trading venue so tradelog fills from different exchanges never mix.
- For the current deployment model, exchange-level separation is sufficient; multiple accounts per exchange are not planned.
- Directional exposure must be tracked separately for long and short inventory because buy-long and short-open fills have different close paths.
- `classifieraccepted` remains required for both buy and sell because the robot needs classifier/strategy output to emit either entry or exit signals.

### First implementation slice
- Add tradelog-derived directional ownership helpers in `Trade` that read only the current routed spot trading partition.
- Store `robotownedlongqty` and `robotownedshortqty` in `TradeCache.cfg`.
- Set `sellenabled` from `(robotownedlongqty > 0 || robotownedshortqty > 0) && classifieraccepted`.
- Cap long-close sizing by `min(freebase, robotownedlongqty)` and short-close sizing by `min(borrowedbase, robotownedshortqty)` so external inventory is ignored.
- Remove the redundant `noinvest` column from trade selection output.

### Follow-up slices
- Generalize ownership reconstruction across spot and futures routing roles.
- Add tests covering mixed external/manual inventory and directional robot-owned close sizing.
- Revisit continuous-liquidity thresholds after the ownership-aware sell path is stable.

## 2026-05-26 Progress Update: Startup Probe and Trade-Loop Cancellation

### Completed in code
- [x] Removed startup probe auto-cancellation scheduling from `scripts/tradereal.jl` for both KrakenSpot and KrakenFutures probes.
- [x] Kept startup capability probes placing long/short validation orders, but now leaves cancellation ownership to the regular trade loop.
- [x] Hardened `_tradestep!` open-order cancellation in `Trade/src/Trade.jl` so one failing cancel/amend does not abort cancellation of remaining open orders.
- [x] Improved cancel symbol resolution in `CryptoXch.cancelorder` to prefer the actual order symbol (from `getorder`) before fallback symbol construction.

### Root-cause notes captured
- A single cancellation exception during `_tradestep!` previously interrupted the loop and left subsequent open orders untouched.
- Base extraction via `basequote(order.symbol)` can be fragile for exchange-specific symbol formats; using the order's native symbol is safer.

### Validation status
- [x] Static editor diagnostics for touched files reported no syntax errors.
- [ ] Workspace "Run tests with coverage" task is currently blocked by package resolution (`CryptoTimeSeries` not found in project/manifest), so full automated validation is pending environment fix.

### Follow-up tasks
- [ ] Add a focused `Trade` test that injects one failing cancel and asserts the loop still attempts cancellation for remaining open orders.
- [ ] Add a `CryptoXch` test that verifies cancel path uses order-native symbol when available.
- [ ] Re-run full workspace tests once project/task package resolution is corrected.

## 2026-05-28 Progress Update: Managed Close Orders for `tradingstrategy03`

### Progress snapshot (as of 2026-05-28 00:20 local)
- [x] Core implementation slice completed (state model, selective reconcile, amend-first flow, restart reconstruction).
- [x] Todo execution for this slice completed end-to-end.
- [x] Latest validation rerun passed: `Trade` test suite `125/125` via `julia --project=. test/runtests.jl`.
- [ ] Follow-up hardening items remain open (adapter fallback edge-case tests, multi-tick scenario assertion, managed-close action counters).

### Policy simplification update (2026-05-28)
- [x] Simplified close-side policy to manage **all open positions** (in-portfolio inventory), removing robot-owned gating from close intent and close sizing.
- [x] Trade-selection retention/candidate logic now keeps in-portfolio assets for close management independently of robot-owned reconstruction.
- [x] Validation rerun passed after the policy change: `Trade` test suite `125/125`.

### Design objective addressed
Keep protective/target close orders active across minute ticks for strategy-driven trading (not canceled and lost each cycle), while remaining restart-safe.

### Completed in code
- [x] Replaced blanket per-minute open-order cancellation in `Trade._tradestep!` with selective cancellation of unmanaged orders.
- [x] Added managed close-order runtime state per base in `TradeCache.mc[:managed_close_orders]`.
- [x] Added per-tick reconstruction of managed close-order state from live open orders plus robot-owned position direction.
- [x] Added amend-first close-order behavior in `trade!` long-close and short-close branches (`changeorder` first, cancel+recreate fallback).
- [x] Added continuous close maintenance pass so one close order remains active for each open robot-owned position even when minute advice is hold/allclose-like.
- [x] Added strategy close-price extraction from `TradingStrategy` lane state (`longta.closeprice` / `shortta.closeprice`) for `:getgainsalgo` flow when available.
- [x] Cleared managed close-order state when applying a new strategy config via `apply_tradingstrategy!`.

### Validation status
- [x] Added focused tests in `Trade/test/managed_close_orders_test.jl`.
- [x] Included new tests in `Trade/test/runtests.jl`.
- [x] Ran `Trade` package tests successfully (`125/125` passing).

### Follow-up tasks
- [ ] Add exchange-facing integration test coverage that validates amend-vs-recreate behavior on adapter responses (including amend rejection fallback).
- [ ] Add one simulation scenario test that runs multiple ticks with persistent open position and asserts exactly one active managed close order per base across ticks.
- [ ] Evaluate whether to persist optional managed-close metadata for observability (restart correctness is already achieved by live reconstruction).

## Trade vs TradingStrategy Regression Harness

The current objective is to keep `Trade.jl` simple and test the behavior that matters directly: compare a bulk strategy evaluation against the minute-by-minute execution path on the same OHLCV window, then separate strategy parity from exchange-rule differences.

### Topic 1: Shared event normalization
- Define a small normalized trade-event shape for both bulk and minute-by-minute runs.
- Keep only the fields needed to compare trade pairs and diagnose mismatches.

### Topic 2: Bulk strategy helper
- Add a pure helper that runs the strategy over a fixed OHLCV window and emits normalized open/close events.
- Keep it isolated from exchange state so the result is deterministic.

### Topic 3: Minute-by-minute execution capture
- Instrument the `Trade.jl` execution path just enough to capture executed trade pairs in the same normalized shape.
- Keep exchange rules and allocation effects visible as data, not as hidden side effects.

### Topic 4: Comparison test
- Add a focused unit/integration test that runs both helpers on the same time range.
- Assert pair equality where the execution model should match bulk strategy behavior.

### Topic 5: Exchange-rule assertions
- Add separate assertions for minimum quantity, allocation limits, and other venue-specific constraints.
- Treat those as expected differences rather than test failures in the parity comparison.

### Topic 6: Diagnostic output
- When the test fails, print missing and extra trade pairs with enough context to explain whether the difference came from strategy logic or execution constraints.
- Keep the diagnostics in test helpers, not in the production `Trade.jl` path.

### Exit criteria
- The regression test can detect unintended `Trade.jl` behavior changes without depending on TrendDetector-specific partition matching.
- Production code stays free of comparison-only trace plumbing.

This harness is intended to replace the TrendDetector-comparison scaffolding that was temporarily added during troubleshooting.

## 2026-05-13 Proposal: Separate Simulation Data from Paper Bookkeeping

### Proposal summary
Use a dedicated simulation exchange `BybitSim` for realistic Bybit-backed market data, symbol metadata, and virtual portfolio bookkeeping. Introduce a dedicated `TestXch` for synthetic OHLCV patterns such as `SINE` and `DOUBLESINE` via `TestOhlcv`. Both simulation exchanges should receive their virtual portfolio when instantiated from the corresponding script, so the simulation bookkeeping lives in the exchange instance rather than in `CryptoXch`.

### Why this matters
- Real simulation needs realistic symbol metadata for `minimumqty`, `validsymbol`, and downstream sizing logic.
- Bybit account/bookkeeping access is not dependable enough for paper trade state, even when market data is usable.
- Synthetic test patterns should be explicit and not inferred from `EnvConfig.test`.
- Exchange choice must become explicit in scripts so the same codebase can target Bybit, Kraken Spot, Kraken Futures, and future venues like IBKR.

### Impact review
- `CryptoXch`: split market-data provider concerns from trade/bookkeeping concerns; keep symbol metadata resolution working in simulation without requiring a real exchange account.
- `CryptoXch`: keep the shared exchange abstraction and routing logic, but move simulation bookkeeping into the simulation exchange implementations instead of `CryptoXch`.
- `Trade`: trade selection and order handling should use the configured exchange identity explicitly, but the virtual portfolio should come from the instantiated simulation exchange.
- Scripts: `tradereal.jl`, `tradesim.jl`, and future entrypoints must pass the exchange explicitly instead of assuming Bybit.
- `TestOhlcv`: remains the provider for explicit synthetic bases only; it should not be implied by `EnvConfig.test`.
- Tests: add coverage for Bybit-seeded simulation, `TestXch` synthetic pattern loading, and quote-asset filtering across exchanges.

### Metadata handling
- `BybitSim` should provide realistic Bybit symbol metadata for simulation sizing and order constraints.
- `TestXch` should reuse copied metadata from Bybit for common symbols and map `XRP` metadata to the synthetic `SINE` and `DOUBLESINE` patterns so sizing and validation behave like a real venue.
- Symbol metadata reuse should stay in the exchange layer so common code can be shared between simulation and real trading.

### Risks and mitigations
- Risk: breaking existing scripts that silently assume Bybit.
	- Mitigation: keep compatibility defaults for one transition period, but emit a clear warning when exchange is omitted.
- Risk: `minimumqty` and symbol validation may still query live metadata in the wrong mode.
	- Mitigation: route symbol metadata through the configured data exchange and add simulation-safe fallbacks.
- Risk: paper bookkeeping accidentally touching venue account state.
	- Mitigation: keep simulation ledger state in `Trade` / `CryptoXch` cache objects only, and block account-dependent paths in `TestXch`.

### Staged plan
1. Split exchange identity from simulation identity in `CryptoXch` so data exchange and trade exchange can be configured independently.
2. Rename the simulation Bybit path to `BybitSim` and add a dedicated `TestXch` adapter for synthetic OHLCV only.
3. Move virtual portfolio/bookkeeping state into the simulation exchange instances and instantiate it from the scripts.
4. Make `minimumqty` and symbol validation simulation-safe while still using real symbol metadata when available.
5. Update `tradereal.jl`, `tradesim.jl`, and helper scripts to declare their exchange explicitly.
6. Add tests covering `BybitSim` market-data/metadata behavior, `TestXch` synthetic data, and non-Bybit live exchanges.

### Exit criteria
- Simulation can run with realistic symbol metadata and a virtual portfolio owned by the simulation exchange instance.
- Synthetic test runs can be targeted explicitly via `TestXch`.
- Scripts no longer silently assume Bybit for all cases, and the simulation exchanges are explicit and script-instantiated.

This plan is intentionally incremental and ordered exactly by the requested priorities. Work on objective `N+1` starts only after objective `N` reaches its exit criteria.

## Current Code Baseline (Relevant Touchpoints)
- `algorithm03!` exists in `TradingStrategy/src/TradingStrategy.jl` and is currently used for gain-segment simulation/evaluation logic.
- Main live/backtest loop exists in `Trade/src/Trade.jl` in `tradeloop(cache::TradeCache)`.
- Exchange abstraction exists in `CryptoXch/src/CryptoXch.jl` with adapters for Bybit, KrakenSpot, KrakenFutures.
- Trend configuration exists in `scripts/TrendDetector.jl` (`TrendDetectorConfig` with `tradingstrategy::TradingStrategy.GainSegment`).
- Existing dashboard foundation exists in `scripts/cryptocockpit.jl`.

## Execution Rules
1. Preserve package boundaries (`Trade` orchestrates, `CryptoXch` abstracts exchanges, `TradingStrategy` evaluates strategy behavior).
2. Keep backward-compatible entry points during migration (feature flags).
3. Every phase produces tests and a resumable artifact (documented below).
4. Bybit is retained for OHLCV updates only after objective 2 is completed.

---

## Objective 1 (First): Integrate `algorithm03!` into `Trade` loop with TrendDetector config, plus backtest and controllable real loop

### Objective 1 status linkage (2026-05-28)
- Managed close-order lifecycle integration for `tradingstrategy03` is now implemented and validated.
- See progress section: **2026-05-28 Progress Update: Managed Close Orders for `tradingstrategy03`**.
- This closes the core gap where per-minute cancellation could leave open positions without active protective/target close orders.
- Latest confirmation run: `Trade` tests passed `125/125` on 2026-05-28 00:20 local.

### Design intent
Turn `algorithm03!` from an offline gain-segment algorithm into a strategy engine that can emit actionable trade intents for both cached backtest and live incremental operation.

### Increment 1.1: Strategy adapter in `Trade`
- Introduce a strategy adapter layer in `Trade` that can invoke `TradingStrategy.GainSegment` and `algorithm03!` on rolling data windows.
- Add a `strategy_mode`/`strategy_engine` config in `TradeCache.mc` with defaults preserving current behavior.
- Add explicit mapping from strategy outputs to executable `TradeLabel` intents.

### Increment 1.2: TrendDetector-configurable strategy runtime
- Add a compact runtime config struct in `Trade` that references the same tuning dimensions as `TrendDetectorConfig.tradingstrategy`.
- Add load/resolve function to pull strategy parameters from a persisted TrendDetector config reference.
- Add validation assertions for parameter ranges and required history length.

### Increment 1.3: Unified backtest/live loop API
- Split `tradeloop` into:
	- deterministic step function (`tradestep!`) for one timestamp/tick
	- runner for cached replay backtest
	- runner for live incremental execution
- Add loop control API for real loop: `start!`, `pause!`, `resume!`, `stop!`, `step!`, and status query.
- Keep old `tradeloop` as compatibility wrapper calling new API.

### Increment 1.4: Tests
- Unit tests for strategy adapter mapping and config loading.
- Integration tests for backtest replay over cached OHLCV.
- Smoke test for live loop control state machine (without real exchange orders).

### Exit criteria
- `algorithm03!` can drive order intents in both backtest and live simulation.
- Runtime strategy settings can be sourced from TrendDetector-style configuration.
- Loop can be paused/resumed/stopped without losing state.

### Deliverables
- Updated `Trade/src/Trade.jl`
- New focused tests in `Trade/test/`
- Migration notes section appended to this file

### Objective 1 follow-up tasks (managed close orders)
- [ ] Add adapter-level tests for amend rejection and explicit cancel+recreate fallback behavior.
- [ ] Add multi-tick simulation test asserting exactly one active managed close order per open base over consecutive ticks.
- [ ] Add structured metrics/log counters for managed-close reconcile actions (`reconstructed`, `amended`, `recreated`, `skipped`) to simplify live monitoring.

---

## Objective 2 (Second): KrakenSpot + KrakenFutures as active trading exchanges through `CryptoXch`; Bybit only for OHLCV updates

### Design intent
Switch active order routing to Kraken via the abstraction layer while preserving Bybit market data ingestion for continuity.

### Progress as of 2026-05-26
- Completed: startup probe order flow in `scripts/tradereal.jl` no longer performs independent delayed cancellation; open-order cleanup ownership is now centered in the trade loop path.
- Completed: cancellation reliability hardened in `Trade._tradestep!` by isolating per-order cancellation/amend exceptions so one failure does not stop cancellation of remaining open orders.
- Completed: `CryptoXch.cancelorder` now prefers the order-native symbol (from `getorder`) before fallback symbol construction to reduce symbol-mapping cancellation misses.
- Pending: add focused Objective 2 regression tests for "one cancel fails, remaining cancels continue" and rerun full workspace tests once task environment/package resolution is fixed.

### Increment 2.1: Explicit trading/data exchange roles
- Extend `XchCache` or add routing config to distinguish:
	- `data_exchange` (OHLCV source)
	- `trade_exchange_spot`
	- `trade_exchange_futures`
- Add policy function in `CryptoXch` deciding adapter per operation (`getklines`, `createorder`, `openorders`, `balances`, etc.).

### Increment 2.2: Kraken-first execution path
- Ensure all order APIs in `Trade` call through KrakenSpot/KrakenFutures routing.
- Add guardrails to block Bybit order placement when policy is `data-only`.
- Add explicit errors with remediation hints if credentials or symbols are missing.

### Increment 2.3: Data continuity
- Keep Bybit `getklines` support active for update/download jobs.
- Add docs and config examples for mixed mode: Bybit data + Kraken trading.
- Example routing setup:
	```julia
	xc = CryptoXch.XchCache()
	setrole!(xc, CryptoXch.data_exchange, CryptoXch.EXCHANGE_BYBIT)
	setrole!(xc, CryptoXch.trade_exchange_spot, CryptoXch.EXCHANGE_KRAKENSPOT, "krakenspot-tcae1")
	setrole!(xc, CryptoXch.trade_exchange_futures, CryptoXch.EXCHANGE_KRAKENFUTURES, "krakenfutures-tcae2")
	```
- `cryptodownload`, `_gethistoryohlcv`, and `downloadupdate!` keep resolving OHLCV via the `data_exchange` role so mixed-mode continuity stays intact.

### Increment 2.4: Base/quote coin pair abstraction
- Introduce explicit `basecoin` and `quotecoin` fields everywhere a trading pair is specified, replacing the concatenated `symbol` string as the primary key in CryptoXch routing.
- The exchange-specific adapter is solely responsible for mapping `(basecoin, quotecoin)` → exchange pair name (e.g. `"BTC"+"USD"` → `"PF_XBTUSD"` for KrakenFutures, `"BTCUSDT"` for KrakenSpot/Bybit).
- This enables:
  1. Trading with non-USDT quote currencies (USD, EUR, USDC) without callers knowing the exact exchange symbol.
  2. Routing the same `(basecoin, quotecoin)` to different exchanges, each applying their own symbol mapping.
- Concrete changes:
  - Add `validsymbol(bc, basecoin, quotecoin)` overloads to KrakenSpot, KrakenFutures, Bybit adapters.
  - Update `CryptoXch` routing helpers to accept `(basecoin, quotecoin)` and resolve to adapter-specific pair names.
  - Update `Trade` callers (portfolio, orders, advice) to pass `basecoin`/`quotecoin` explicitly where it affects routing.
  - Add symbol-mapping unit tests covering at least: BTC/USDT→KrakenSpot, BTC/USD→KrakenFutures, BTC/USDT→Bybit.

### Increment 2.5: Tests
- Adapter routing tests in `CryptoXch/test/`.
- End-to-end simulation test in `Trade/test/` verifying Bybit no-trade enforcement.

### Exit criteria
- Spot and futures orders are Kraken-routed via `CryptoXch`.
- Any Bybit trade attempt is rejected by policy.
- OHLCV update jobs can still use Bybit.
- `(basecoin, quotecoin)` is the canonical pair representation; adapters own the symbol mapping.

### Deliverables
- Updated `CryptoXch/src/CryptoXch.jl`
- Potential updates in `Trade/src/Trade.jl`
- New routing/policy tests

---

## Objective 3 (Third): Multi-exchange TradeLog order/trade logging

### Design intent
Create complete and queryable TradeLog records for trader diagnostics and tax workflows, designed for multiple exchanges in parallel.

### Progress as of 2026-05-26
- Completed: no schema/storage redesign changes in this slice.
- Completed: cancellation-path robustness improvements in Objective 2 reduce risk of missing downstream cancel-state reconciliation for later orders in the same loop iteration.
- Pending: continue planned Objective 3 increments (canonical schema, append-only sink, replay tests) after Objective 2 follow-up cancellation tests are in place.

### Increment 3.1: Canonical TradeLog schema
- Define canonical event model with required fields:
	- event metadata: timestamp (UTC), source module, environment, correlation IDs
	- exchange metadata: exchange name, account/auth alias, routing role used, market type
	- instrument metadata: canonical asset type, venue-specific instrument type, symbol, baseasset, quoteasset, underlying, settlement asset, contract class
	- order metadata: client order id, exchange order id, symbol, side, type, tif, requested qty/price
	- execution metadata: fill qty/price, fees, fee currency, status transitions
	- position/portfolio snapshot deltas before and after action
	- strategy context: config ref, signal label/score, algorithm version
- Introduce an explicit TradeLog classification field set so logs can distinguish at least:
	- crypto spot pair trades
	- crypto perpetual futures on pairs
	- shares against FIAT
	- future asset classes without schema redesign
- Recommended canonical values:
	- `asset_class`: `crypto`, `equity`, later extensible to `fx`, `option`, `future`, `commodity`, ...
	- `instrument_type`: `spot_pair`, `perpetual_future`, `share_fiat`, later extensible
	- `venue_instrument_type`: raw exchange-specific type when available
- The combination of `exchange name + account/auth alias + asset_class + instrument_type + symbol` must be sufficient to disambiguate what was actually traded even when symbols overlap across venues.
- Separate event types: `ORDER_SUBMITTED`, `ORDER_ACK`, `ORDER_PARTIAL_FILL`, `ORDER_FILLED`, `ORDER_CANCELED`, `ORDER_REJECTED`, `POSITION_SNAPSHOT`, `PORTFOLIO_SNAPSHOT`.
- Concrete implementation proposal:
	- Preferred module/package name: `TradeLog` as a dedicated workspace package if reused across `Trade`, dashboard export, and future IB integration; fallback is a focused `Trade/src/tradelog.jl` module if package extraction would slow delivery.
	- Define small canonical enums for:
		- `TradeLogEventType`
		- `TradeLogAssetClass`
		- `TradeLogInstrumentType`
		- `TradeLogMarketType`
		- `TradeLogRoutingRole`
	- Use one canonical flat event row schema for storage and replay rather than multiple incompatible tables. Nested Julia structs may exist in memory, but persisted records should flatten to a single column set.
	- Minimum persisted columns for each event row:
		- event identity: `event_id`, `event_type`, `event_time_utc`, `created_at_utc`, `source_module`, `environment`, `run_id`, `loop_id`, `correlation_id`, `parent_event_id`
		- venue identity: `exchange`, `account_alias`, `routing_role`, `market_type`
		- instrument identity: `asset_class`, `instrument_type`, `venue_instrument_type`, `symbol`, `baseasset`, `quoteasset`, `underlying`, `settlement_asset`, `contract_class`
		- order identity: `client_order_id`, `exchange_order_id`, `exchange_trade_id`, `side`, `order_type`, `time_in_force`, `status`, `status_reason`
		- requested economics: `requested_base_qty`, `requested_quote_qty`, `requested_limit_price`, `requested_stop_price`, `requested_notional`, `leverage`
		- execution economics: `fill_base_qty`, `fill_quote_qty`, `fill_price`, `fill_notional`, `fee_amount`, `fee_currency`, `slippage_bps`
		- state deltas: `position_qty_before`, `position_qty_after`, `cash_before`, `cash_after`, `portfolio_value_before`, `portfolio_value_after`
		- strategy context: `strategy_engine`, `strategy_config_ref`, `signal_label`, `signal_score`, `algorithm_version`, `notes`
	- Event-specific fields that do not apply to a row remain `missing` rather than forcing different schemas.
	- For the first implementation, treat the canonical row schema as the source of truth for replay, export, and dashboard queries.

### Increment 3.2: Append-only log writer
- Add write-once append log sink (Arrow/CSV/JSONL based on EnvConfig format policy).
- Add partitioning by `exchange/account/date` under `$HOME/crypto/tradelog`.
- Add tamper-evidence option via per-file hash chain (stored in companion manifest).
- Concrete storage layout proposal:
	- Root folder: `$HOME/crypto/tradelog/`
	- Partition dimensions in path:
		- `environment=<mode>/`
		- `exchange=<exchange>/`
		- `account=<auth_alias>/`
		- `asset_class=<asset_class>/`
		- `instrument_type=<instrument_type>/`
		- `date=YYYY-MM-DD/`
	- Example paths:
		- `$HOME/crypto/tradelog/environment=production/exchange=KrakenSpot/account=krakenspot-tcae1/asset_class=crypto/instrument_type=spot_pair/date=2026-05-10/events.jsonl`
		- `$HOME/crypto/tradelog/environment=production/exchange=KrakenFutures/account=krakenfutures-tcae2/asset_class=crypto/instrument_type=perpetual_future/date=2026-05-10/events.jsonl`
		- `$HOME/crypto/tradelog/environment=production/exchange=InteractiveBrokers/account=paper/asset_class=equity/instrument_type=share_fiat/date=2026-05-10/events.jsonl`
	- Canonical write path should be append-only `events.jsonl` because JSONL is simple for crash-safe append semantics and diff-friendly diagnostics.
	- Optional read-optimized companions may be produced per partition after rotation:
		- `events.arrow` for analytics/dashboard scans
		- `manifest.json` for row counts, min/max timestamps, and previous-file hash
	- File rotation policy:
		- rotate at UTC day boundary or when file size exceeds a configured threshold
		- never mutate a closed file except to write its final manifest
	- Tamper-evidence proposal:
		- each row carries `event_id` and deterministic serialized payload hash
		- each closed file manifest stores `file_hash`, `previous_file_hash`, `first_event_time_utc`, `last_event_time_utc`, and `row_count`
	- Replay readers should consume JSONL as the authoritative source and prefer Arrow only as an acceleration layer.

### Increment 3.3: Logging integration points
- Instrument `Trade` and `CryptoXch` around all order lifecycle calls and polling updates.
- Ensure failed/rejected/cancelled paths are logged with reason codes.
- Add periodic snapshot events for portfolio state.

### Increment 3.4: Tests and replay utility
- Schema validation tests.
- Replay test that reconstructs order lifecycle from logs.
- Utility script to export tax/TradeLog-friendly reports.

### Exit criteria
- Every order status transition is persisted with traceable identifiers.
- Logs are partitioned for multiple exchanges/accounts.
- TradeLog records identify both the concrete exchange/account used and the traded asset type beyond the raw symbol.
- Replay utility can rebuild per-order lifecycle and daily PnL-relevant events.

### Deliverables
- New TradeLog module in `Trade` or dedicated package (preferred if reused across packages)
- Tests plus report/export script

---

## Objective 4 (Fourth): Asynchronous orchestration for trading loop and exchange interactions

### Design intent
Prevent blocking between data updates, signal evaluation, order management, and portfolio refresh. Also implement dynamic adjustment of adaptive orders.

### 2026-05-30 incremental rollout plan (low-risk)

#### Target architecture contract
- `TradingStrategy` emits action/price intent only.
- `Trade` computes requested amounts from allocation/risk policy.
- Exchange adapters normalize/adjust requests for venue constraints (qty step, min qty, tick size, reduce-only, tif/venue specifics).
- Exchange adapters own balances cache and publish one synchronized snapshot per cycle to `Trade`.
- Exchange adapters own persistent OHLCV update for actively traded bases.
- A base is only `data_ready` for strategy execution after OHLCV gaps to persisted history are closed.

#### Safety constraints during migration
- Keep current synchronous `tradereal` flow as default for KrakenSpot/KrakenFutures.
- Gate all new behavior behind feature flags (default `off`).
- Enable async workers in shadow mode before any order-affecting cutover.
- Keep immediate fallback switch to synchronous execution.

#### Increment 4.0: Safety scaffolding
- Add feature flags:
	- `CTS_ASYNC_ENGINE_ENABLED`
	- `CTS_ASYNC_SHADOW_MODE`
	- `CTS_WS_MARKETDATA_ENABLED`
	- `CTS_EXCHANGE_BALANCE_CACHE_OWNER`
	- `CTS_OHLCV_GAP_BACKFILL_ON_TRADABLE`
- Add dual-path shadow comparator:
	- sync decision snapshot vs async candidate snapshot
	- compare labels, limit prices, and normalized quantities
	- auto-disable async execution on critical divergence.

#### Increment 4.1: Exchange-owned balance cache sync
- Move balance refresh ownership to exchange adapters.
- Publish exactly one canonical balances snapshot per cycle into `Trade`.
- Keep `Trade` read-only relative to adapter caches.
- Add tests for freshness policy, staleness fallback, and sync parity.

#### Increment 4.2: Async worker topology in shadow mode
- Introduce bounded-channel workers:
	- marketdata, strategy, order-intent, order-execution, order-reconcile, balance-sync
- Keep order submission authoritative in synchronous path while shadow mode records async parity.
- Add heartbeat/watchdog metrics per worker.

#### Increment 4.3: WebSocket ingestion with HTTP fallback
- Prefer websocket trades/klines where supported.
- Maintain HTTP fallback per symbol/venue when stream quality drops.
- Add freshness SLA checks and automatic fallback switching.

#### Increment 4.4: Tradable-base OHLCV persistence + gap closure
- Introduce base state machine:
	- `discovered -> backfill_required -> backfill_in_progress -> data_ready -> tradable`
- On tradable candidate transition:
	- detect persisted OHLCV range
	- close gaps up to cycle boundary
	- gate strategy/order path until `data_ready`.
- Add restart-safe tests for gap closure and no-order-before-data-ready behavior.

#### Increment 4.5: Controlled activation
- Phase A: shadow mode only.
- Phase B: canary execution on selected bases.
- Phase C: full async primary with synchronous emergency fallback retained.
- Acceptance metrics:
	- no increase in reject rate due to normalization
	- reduced REST request volume
	- improved or unchanged reaction latency
	- no OHLCV gaps on active tradable bases.

### Objective 4 increment mapping (legacy notes merged)
- Worker topology originally listed as old Increment 4.1 is now covered by Increment 4.2 (async worker topology in shadow mode).
- Event-driven orchestration originally listed as old Increment 4.2 is now covered by Increment 4.2 and Increment 4.5 (controlled activation gates).
- Safety/resilience originally listed as old Increment 4.3 is now split across Increment 4.0 (rollback and divergence guard), Increment 4.2 (watchdog/heartbeat), and Increment 4.5 (canary rollout and fallback).
- Test scope originally listed as old Increment 4.4 is now distributed across:
	- Increment 4.0 comparator/parity tests
	- Increment 4.1 balance-cache sync tests
	- Increment 4.3 websocket fallback/freshness tests
	- Increment 4.4 OHLCV gap closure and readiness-gate tests
	- Increment 4.5 canary acceptance metrics and regression checks
- The authoritative Objective 4 sequence is now Increment 4.0 through Increment 4.5 above; the old duplicate numbering is retired.

### Exit criteria
- Data refresh and order handling proceed independently.
- Dashboard-facing data feeds are not blocked by execution tasks.
- Controlled shutdown leaves consistent state.

### Deliverables
- Async orchestration layer integrated into `Trade`
- New tests for concurrency behavior

### Decision Record: Adaptive Repricing Cadence (2026-05-17)

Decision scope:
- Choose when adaptive maker orders (`limitprice=nothing`, `maker=true`) should be repriced while still open.

Options:
- Option A: Tick-driven repricing in `Trade._tradestep!` (current implementation).
- Option B: Dedicated websocket/event-driven repricing worker in Objective 4 async topology.

Selection criteria:
- Repricing latency: how quickly order price follows spread changes.
- API pressure: amend/cancel rate versus exchange limits.
- Queue churn: probability of losing queue priority due to unnecessary reprices.
- Complexity and resilience: operational complexity and failure-surface area.
- Testability: deterministic backtest and unit-test reproducibility.

Current decision:
- Default to Option A (tick-driven) as baseline behavior.
- Escalate to Option B only if measured latency/quality targets are not met under realistic load.

Escalation triggers:
- Median reprice lag exceeds configured target for sustained periods.
- Fill quality degradation is attributable to stale maker limits.
- Tick cadence must be increased solely to improve repricing responsiveness.

Guardrails (both options):
- Reprice only when target price changed by at least one tick.
- Keep adaptive intent sticky across order-id replacement.
- Preserve deterministic behavior in simulation/backtest paths.

---

## Objective 5 (Fifth): Interactive non-blocking dashboard for history and open orders

### Design intent
Build on existing cockpit script and feed it with live state and TradeLog logs without blocking the trading engine.

### Increment 5.1: Dashboard data contract
- Define read model for:
	- open orders
	- recent fills
	- order lifecycle history
	- portfolio and exposure snapshots
	- strategy decisions vs executed actions
- Provide read model via cached tables refreshed by background tasks.

### Increment 5.2: UI integration (existing Dash stack)
- Extend `scripts/cryptocockpit.jl` to add dedicated pages/panels for:
	- live open orders
	- historical trades (filter by exchange/account/symbol/date)
	- order timeline view
	- TradeLog export trigger
- Ensure polling/subscription updates are asynchronous.

### Increment 5.3: Performance and UX hardening
- Add pagination/virtualization for large history tables.
- Add last-update indicators and data staleness warnings.
- Add role/operation safety controls for live actions.

### Increment 5.4: Tests
- Smoke tests for dashboard data endpoints/providers.
- Validation that UI remains responsive under active trading load.

### Exit criteria
- Dashboard shows accurate open orders and history across exchanges.
- UI remains responsive while trading loop is active.

### Deliverables
- Updated `scripts/cryptocockpit.jl`
- Optional helper module for dashboard data shaping

---

## Objective 6 (Sixth): Add Interactive Brokers interface via `CryptoXch`

### Design intent
Introduce IB as another adapter under the same abstraction/routing model.

### Increment 6.1: New adapter package skeleton
- Add `InteractiveBrokers/` package (same workspace package structure conventions).
- Implement minimal interface parity with existing exchange adapters:
	- exchange info/symbol metadata
	- balances/portfolio
	- open orders/order query
	- create/cancel/amend order
	- market data retrieval where applicable

### Increment 6.2: `CryptoXch` integration
- Register IB exchange constants and adapter routing.
- Add config/auth mapping and capability flags (spot, derivatives, shorting policy).

### Increment 6.3: Safety gates
- Add paper-trading mode first.
- Add explicit unsupported-operation errors where IB semantics differ.

### Increment 6.4: Tests
- Mocked adapter contract tests.
- Routing tests with mixed exchange deployment.

### Exit criteria
- IB can be selected via `CryptoXch` without changing `Trade` logic.
- Paper trading path is validated before live enablement.

### Deliverables
- New `InteractiveBrokers` package
- `CryptoXch` routing updates and tests

---

## Cross-Phase Test Strategy
- Unit tests for each new pure function and mapper.
- Contract tests for exchange adapters (shared test matrix).
- Integration tests for end-to-end loop in simulation mode.
- Regression backtests comparing previous vs new strategy behavior.

## Cross-Phase Risk Controls
- Feature flags per major capability.
- Dry-run mode for order execution.
- Kill switch for all open positions (`quickexit` path) retained.
- Mandatory structured logging before enabling live routing changes.

## Interruption-Safe Progress Ledger
Update this section after each work session.

### Status Summary
- Objective 1: IN PROGRESS (Increment 1.1 started)
- Objective 2: COMPLETED (Increments 2.1-2.5 done and validated) + MAINTENANCE UPDATES (BybitSim timestamp-aware `get24h` pricing, adapter review, and explicit screening vs valuation USDT market intents)
- Objective 3: IN PROGRESS (3.1 schema + integration slice + TradeLog chain linkage + OCO bracket helper + symbol-info cache done; performance metrics and drawdown tracking remain)
- Objective 4: IN PROGRESS (adaptive-maker repricing implemented; incremental async/socket migration plan defined with shadow-mode rollout)
- Objective 5: NOT STARTED
- Objective 6: NOT STARTED
- Objective 7: IN PROGRESS (runtime API compatibility adapter added in TradingStrategy; Trade now has optional runtime-backed advice collection, runtime history sizing, tradeselection readiness delegation, and test-only default gating with env override)
- Objective 8: DEFINED (resilient trade loop — exchange outage and intermittent response handling; partial fix for base_ohlcv_unavailable crash applied 2026-05-31; full recovery mode not yet implemented)

### Session Log Template
- Date:
- Objective/Increment:
- Completed:
- Files changed:
- Tests run and result:
- Open issues/blockers:
- Next immediate step:

### Session Log
- Date: 2026-05-18
- Objective/Increment: Objective 2 maintenance / BybitSim screening with timestamp-aware cached OHLCV symbols
- Completed:
	- Added selective symbol filtering in BybitSim `_sim_get24h(symbol=nothing)` to iterate `syminfodf` but skip symbols lacking cached OHLCV at simulation reference time; aborts per-symbol failure instead of aborting entire snapshot.
	- Updated `CryptoXch.screeningUSDTmarket` and `CryptoXch.valuationUSDTmarket` to call `setcurrenttime!(xc, dt)` before fetching pricing to align ticker snapshot with trade-selection timestamp.
	- Result: trade-selection screening in simulation uses only viable cached-data-backed symbols and remains deterministic at the requested simulation time.
	- Added focused CryptoXch test coverage for selective symbol filtering and timestamp-aware pricing queries.
- Files changed:
	- `Bybit/src/Bybit.jl`
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/runtests.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `Bybit/test/runtests.jl` passed (17/17)
	- `CryptoXch/test/runtests.jl` passed (26/26)
	- Trade backtest integration: BybitSim selective symbol filtering confirmed with reduced universe filtered by OHLCV availability
- Open issues/blockers:
	- Selective symbol filtering works for broad screening; valuationUSDTmarket with specific requested bases still queries only requested symbols (correct path, no issue).
	- Timestamp-aware pricing now works in screening path; further refinements depend on next simulation scenario.
- Next immediate step:
	- Verify multi-asset screening and valuationUSDTmarket consistency across live and simulation trade runs.

- Date: 2026-05-18
- Objective/Increment: Objective 2 maintenance / explicit screening vs valuation USDT market intents
- Completed:
	- Added explicit `CryptoXch.screeningUSDTmarket` (broad universe) and `CryptoXch.valuationUSDTmarket` (coin-scoped) APIs.
	- Kept `CryptoXch.getUSDTmarket` backward-compatible and routed internal ticker fetch through a shared helper.
	- Updated `CryptoXch.portfolio!` default valuation path to use `valuationUSDTmarket` with balance-derived requested base coins.
	- Updated Trade selection/live wait paths to call `screeningUSDTmarket` explicitly.
	- Added focused CryptoXch tests for intent separation and unrelated-symbol safety (`AAPLXUSDT` present in symbol universe but excluded from valuation when not held).
- Files changed:
	- `CryptoXch/src/CryptoXch.jl`
	- `Trade/src/Trade.jl`
	- `CryptoXch/test/usdtmarket_intent_test.jl`
	- `CryptoXch/test/runtests.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- Root-project include: `Trade/test/backtest_integration_test.jl` passed (21/21)
	- Root-project repro: BybitSim with injected `AAPLXUSDT` symbol and BTC-only holdings passed; `portfolio!` valuation succeeded without unrelated-symbol cache failure.
- Open issues/blockers:
	- Remaining broad market users outside Trade selection are still allowed to call `getUSDTmarket`; intent-specific APIs should be preferred for new code.
- Next immediate step:
	- Add adapter-level contract tests asserting `valuationUSDTmarket` only queries requested symbols for each supported exchange adapter.

- Date: 2026-05-17
- Objective/Increment: Objective 4 / adaptive maker steady-loop repricing slice
- Completed:
	- Added adaptive maker order intent registry in `CryptoXch` (`registeradaptiveorder!`, `unregisteradaptiveorder!`, `isadaptiveorder`, `pruneadaptiveorders!`).
	- Wired registry lifecycle to order create/cancel/amend/getopenorders so adaptive intent survives order-id replacements and is cleaned when orders close.
	- Updated `Trade._tradestep!` to amend adaptive maker orders with `limitprice=nothing` instead of cancelling all open orders.
	- Added no-op short-circuit in KrakenSpot/KrakenFutures `amendorder` when neither effective price nor quantity changed.
	- Added focused Trade test coverage for adaptive order registry behavior.
- Files changed:
	- `CryptoXch/src/CryptoXch.jl`
	- `Trade/src/Trade.jl`
	- `KrakenSpot/src/KrakenSpot.jl`
	- `KrakenFutures/src/KrakenFutures.jl`
	- `Trade/test/backtest_integration_test.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- Root-project includes: `Trade/test/backtest_integration_test.jl` passed (21/21)
	- Root-project includes: `KrakenSpot/test/KrakenSpot_test.jl` passed (23/23)
	- Root-project includes: `KrakenFutures/test/KrakenFutures_test.jl` passed (23/23)
- Open issues/blockers:
	- Repricing currently runs at trade-loop tick cadence; there is no dedicated websocket-driven reprice worker yet.
- Next immediate step:
	- Decide whether to keep tick-driven repricing only or add a dedicated async repricing worker as part of Objective 4.

- Date: 2026-05-17
- Objective/Increment: Objective 2 / BybitSim timestamp-aware maker snapshot pricing
- Completed:
	- Added `simtime` field to `Bybit.BybitCache`.
	- Updated BybitSim `_sim_lastprice` to derive from OHLCV at `floor(simtime, 1m) - 1m` (previous-minute close semantics), including test-base and cached-base paths.
	- Updated `CryptoXch.setcurrenttime!` to propagate `currentdt` into adapters exposing `simtime`.
	- Updated routing test fixture for `BybitCache` constructor shape change.
	- Verified omitted-limit maker order fill price now matches previous-minute close at simulation timestamp.
- Files changed:
	- `Bybit/src/Bybit.jl`
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/routingtests.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `Bybit/test/runtests.jl` passed (17/17)
	- `CryptoXch/test/routingtests.jl` passed (26/26)
	- Manual timestamped BTC repro: `get24h_last == previous-minute close` confirmed
- Open issues/blockers:
	- Price correctness still depends on quality of persisted OHLCV cache data.
- Next immediate step:
	- Add cache sanity checks for critical symbols (BTC/ETH) before simulation runs.

- Date: 2026-05-17
- Objective/Increment: Objective 2 / KrakenSpot + KrakenFutures `get24h` adaptation review
- Completed:
	- Reviewed KrakenSpot/KrakenFutures `get24h`, `createorder`, and `amendorder` paths for synthetic-price behavior.
	- Confirmed both adapters use live REST ticker snapshots (with documented futures spot-fallback), not synthetic symbol-seeded pricing.
	- Confirmed no equivalent adaptation to BybitSim timestamp pricing is currently required for Kraken adapters.
- Files changed:
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- Review-only step; no code change required.
- Open issues/blockers:
	- If simulated Kraken adapters are introduced later, they will need explicit timestamp-aware ticker semantics similar to BybitSim.
- Next immediate step:
	- Document this live-only assumption near Kraken adapter `get24h` implementations when touching those modules again.

- Date: 2026-05-10
- Objective/Increment: Objective 3 / OCO bracket helper + local symbol-info cache for simulation
- Completed:
	- Implemented `createocoorder` in `CryptoXch` as a high-level bracket helper that creates three linked orders (entry, take-profit, stop-loss) in a single call.
	- Shared `leg_group_id` (UUID) is generated for all three legs; each leg carries its `leg_label` (`entry`, `take_profit`, `stop_loss`) and the TP/SL legs carry `parent_order_id = entry_order_id` to wire the causal audit chain.
	- Signal metadata (`signal_label`, `signal_score`, `strategy_engine`, `strategy_config_ref`) is forwarded to all three legs through the `xc.mc[:audit_event_context]` mechanism using an internal `_setlegctx!` helper; context is cleaned up in `try/finally` to ensure no leakage.
	- Implemented a local symbol-info cache (`xc.mc[:syminfo_cache]` as `Dict{String, NamedTuple}`) to fix the `symbolinfo(::Nothing, ::String)` crash that occurred in simulation mode when `_routedbc` returns `nothing` (no live exchange connection).
	- Added `_syminfocache(xc)`, `setsymbolinfocache!(xc, symbol, info)` for test injection, and updated `_exchangesymbolinfo` to populate the cache from live exchange responses and fall back to it in sim mode.
	- When a live exchange connection is available, the cache is populated transparently; when in `cryptoxchsim` mode the cache must be pre-seeded (via `setsymbolinfocache!`) or previously populated in a live session.
	- Added `CryptoXch/test/multileg_order_test.jl` with 28 assertions covering long bracket (buy entry → sell TP + sell SL) and short bracket (sell entry → buy TP + buy SL): leg_group_id consistency, leg_label values, parent_order_id chaining, signal context forwarding, correlation_id chain semantics.
	- Root order `correlation_id` defaults to its own order id (self-referencing root); child legs point to the root id. Test was updated to assert this expected semantic rather than the incorrect inverse.
- Files changed:
	- `CryptoXch/src/CryptoXch.jl` (symbol-info cache helpers + `createocoorder`)
	- `CryptoXch/test/multileg_order_test.jl` (new file)
	- `CryptoXch/test/runtests.jl` (added multileg include)
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd CryptoXch && julia --project=. -e 'include("test/audit_integration_test.jl"); include("test/multileg_order_test.jl")'` → 18/18 + 28/28 passed
- Open issues/blockers:
	- `_exchangevalidsymbol` still calls `validsymbol(nothing, sym)` in sim mode; not exercised by current tests and not triggered by `createocoorder`. Pre-existing gap.
	- Performance metrics (Sharpe/Sortino) and drawdown/recovery tracking are not yet implemented.
- Next immediate step:
	- Consider adding a `STRATEGY_ENGINE` audit event type for periodic performance metric snapshots (Sharpe, Sortino, drawdown) as the next Objective 3 sub-task.

- Date: 2026-05-10
- Objective/Increment: Objective 3 / causal order-chain linkage + fee/commission logging
- Completed:
	- Added audit chain state in `CryptoXch` to persist `correlation_id` and `parent_event_id` for order lifecycle events.
	- Added parent linkage helper for amended/replaced orders so child-order events can point to prior chain events.
	- Extended `changeorder` and `cancelorder` audit behavior to emit lifecycle transitions with causal context.
	- Added fee and commission capture in order audit rows (`fee_amount`, `fee_currency`) with fallback fee estimation for fill events when exchanges do not expose explicit fee fields.
	- Extended focused `CryptoXch` audit tests to validate filled-event fee capture and parent/correlation linkage fields.
- Files changed:
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/audit_integration_test.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd CryptoXch && julia --project=. -e 'include("test/audit_integration_test.jl")'` → passed
	- `cd Trade && julia --project=. -e 'include("test/audit_snapshot_test.jl")'` → passed
- Open issues/blockers:
	- Exchange adapters still differ in whether they expose native execution-level fee fields; current logic uses canonical extraction plus deterministic fallback for fills.
	- Dedicated execution-trade-id enrichment remains pending where adapters expose per-fill trade IDs.
- Next immediate step:
	- Add adapter-level normalization for explicit fee/trade-id fields in `openorders`/`order` payloads to reduce reliance on fallback fee estimation.

- Date: 2026-05-10
- Objective/Increment: Objective 3 / getorder-getopenorders lifecycle reconciliation + per-asset snapshots
- Completed:
	- Added transition-aware audit reconciliation in `CryptoXch` so `getorder`/`getopenorders` persist status changes as `ORDER_ACK`, `ORDER_PARTIAL_FILL`, `ORDER_FILLED`, `ORDER_CANCELED`, or `ORDER_REJECTED`.
	- Added lightweight in-memory order audit state cache keyed by order id to emit events only on status/fill deltas.
	- Extended `getorder` with `auditevent` control to avoid duplicate events for internal lookup usage during order creation.
	- Upgraded `Trade` portfolio snapshots from one aggregate row to per-asset snapshot rows with asset symbol, position quantity, and portfolio totals for replay-grade holdings history.
	- Extended focused tests to validate transition event persistence and per-asset snapshot output.
- Files changed:
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/audit_integration_test.jl`
	- `Trade/src/Trade.jl`
	- `Trade/test/audit_snapshot_test.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd CryptoXch && julia --project=. -e 'include("test/audit_integration_test.jl")'` → passed
	- `cd Trade && julia --project=. -e 'include("test/audit_snapshot_test.jl")'` → passed
- Open issues/blockers:
	- Event reconciliation currently uses order snapshots from polling APIs and does not yet persist exchange-native trade ids when not present on order payloads.
	- Position snapshots are represented as per-asset portfolio rows; dedicated leveraged-position snapshots remain a future enhancement.
- Next immediate step:
	- Add cancel/amend lifecycle correlation (`parent_event_id`/`correlation_id`) and trade-id enrichment where adapters expose execution-level identifiers.

- Date: 2026-05-10
- Objective/Increment: Objective 3 / Integration slice across CryptoXch and Trade
- Completed:
	- Added `TradeAudit` as a dependency of `CryptoXch` and `Trade`.
	- Integrated `CryptoXch` order audit emission for submitted and rejected order events at the public create-order boundary.
	- Added explicit `run_mode` and `run_id` stamping on emitted order and portfolio audit rows.
	- Added per-run-mode partitioning (`run_mode=<live|simulation>`) in the audit folder layout to prevent simulation/live mixing.
	- Added an environment override for the audit root to support deterministic package-local tests without writing into the shared `\$HOME/crypto/audit` tree.
	- Integrated `Trade` portfolio snapshot emission once per trading tick.
	- Added focused tests for CryptoXch order audit persistence and Trade portfolio snapshot persistence.
- Files changed:
	- `TradeAudit/src/TradeAudit.jl`
	- `TradeAudit/test/runtests.jl`
	- `CryptoXch/Project.toml`
	- `CryptoXch/Manifest.toml`
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/runtests.jl`
	- `CryptoXch/test/audit_integration_test.jl`
	- `Trade/Project.toml`
	- `Trade/Manifest.toml`
	- `Trade/src/Trade.jl`
	- `Trade/test/runtests.jl`
	- `Trade/test/audit_snapshot_test.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd CryptoXch && julia --project=. -e 'using Pkg; Pkg.develop(path="../TradeAudit"); Pkg.resolve(); include("test/audit_integration_test.jl")'` → passed
	- `cd Trade && julia --project=. -e 'using Pkg; Pkg.develop(path="../TradeAudit"); Pkg.resolve(); include("test/audit_snapshot_test.jl")'` → passed
- Open issues/blockers:
	- Order status polling, fills, and cancel/amend lifecycle events are not emitted yet.
	- Portfolio snapshots are currently aggregate rows; per-asset or per-position snapshots remain for a later Objective 3 increment.
	- Hash-chain manifest generation and Arrow companion output remain for later Objective 3 increments.
- Next immediate step:
	- Extend Objective 3 lifecycle coverage to `getorder` / `getopenorders` reconciliation so fills, cancels, and status transitions are appended as audit events.

- Date: 2026-05-10
- Objective/Increment: Objective 3 / Initial Increment 3.1 slice
- Completed:
	- Added new `TradeAudit` workspace package as the first Objective 3 implementation surface.
	- Implemented canonical audit enums for event type, asset class, instrument type, market type, and routing role.
	- Implemented flat `AuditEventRow` schema aligned with the Objective 3 canonical event proposal.
	- Implemented partitioned audit path helpers rooted at `\$HOME/crypto/audit`.
	- Implemented append-only JSONL event writing for canonical audit rows.
	- Added isolated unit tests covering schema defaults, payload serialization, partition layout, and append-only file writes.
- Files changed:
	- `TradeAudit/Project.toml`
	- `TradeAudit/Manifest.toml`
	- `TradeAudit/setup.jl`
	- `TradeAudit/src/TradeAudit.jl`
	- `TradeAudit/test/runtests.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd TradeAudit && julia --project=. -e 'using Pkg; Pkg.develop(path="../EnvConfig"); Pkg.resolve(); Pkg.test()'` → passed
- Open issues/blockers:
	- `TradeAudit` is not integrated into `Trade` or `CryptoXch` yet.
	- No lifecycle events are emitted yet; only schema and writer infrastructure exist.
	- Hash-chain manifest generation and Arrow companion output remain for later Objective 3 increments.
- Next immediate step:
	- Wire `TradeAudit` into the first order lifecycle boundary, starting with order submission and rejection events in `CryptoXch`.

- Date: 2026-05-10
- Objective/Increment: Objective 2 / Increments 2.1-2.5
- Completed:
	- Added explicit exchange-role routing in `CryptoXch` for `data_exchange`, `trade_exchange_spot`, and `trade_exchange_futures`.
	- Routed market-data operations through the data role and trading operations through spot/futures trade roles.
	- Added Bybit data-only guardrails that block order placement when Bybit is configured only as the market-data source.
	- Preserved Bybit OHLCV continuity for `cryptodownload`, `_gethistoryohlcv`, and `downloadupdate!` in mixed-mode routing.
	- Added pair-aware `(basecoin, quotecoin)` symbol resolution and validation across `CryptoXch`, `KrakenSpot`, `KrakenFutures`, and `Bybit`.
	- Added deterministic routing and symbol-mapping tests covering BTC/USDT on KrakenSpot and Bybit, plus BTC/USD on KrakenFutures.
	- Added Trade-level guardrail regression coverage and revalidated the Trade package suite.
- Files changed:
	- `CryptoXch/src/CryptoXch.jl`
	- `CryptoXch/test/routingtests.jl`
	- `CryptoXch/test/simruntests.jl`
	- `Trade/test/bybit_guardrail_test.jl`
	- `Trade/test/runtests.jl`
	- `KrakenSpot/src/KrakenSpot.jl`
	- `KrakenFutures/src/KrakenFutures.jl`
	- `Bybit/src/Bybit.jl`
	- `docs/trading-loop-integration-plan-2026-05-10.md`
- Tests run and result:
	- `cd CryptoXch && julia --project=. -e 'include("test/routingtests.jl")'` → passed
	- `cd Trade && julia --project=. -e 'include("test/bybit_guardrail_test.jl")'` → passed
	- `cd Trade && julia --project=. -e 'using Pkg; Pkg.test()'` → passed
- Open issues/blockers:
	- Objective 3 audit logging design not started yet.
	- Pair-aware public APIs in `Trade` still primarily default to `EnvConfig.cryptoquote`; later work may widen caller-side quote selection further.
- Next immediate step:
	- Start Objective 3 by defining the canonical audit event schema and storage layout, including exchange/account and asset-type classification fields.

- Date: 2026-05-10
- Objective/Increment: Objective 1 / Increment 1.1
- Completed:
	- Added strategy feature flag in `TradeCache` (`:strategy_engine` with `:classifier` and `:algorithm03`).
	- Added minimal per-base `algorithm03!` adapter in `Trade`:
		- rolling history capture (`opentime/high/low/close` + classifier label/score)
		- per-base `TradingStrategy.GainSegment` state
		- mapping from algorithm action state to executable trade labels
	- Wired adapter into `tradeloop` so classifier advice is transformed by algorithm03 when enabled.
	- Added focused unit test for adapter label mapping.
- Files changed:
	- `Trade/src/Trade.jl`
	- `Trade/test/runtests.jl`
	- `Trade/test/algorithm03_adapter_test.jl`
- Tests run and result:
	- `Trade` isolated test: `algorithm03_adapter_test.jl` PASSED (5/5).
	- Full `Pkg.test()` for `Trade` still fails due pre-existing external Bybit HTTP issue in `storage_format_test.jl` (not introduced by this increment).
- Open issues/blockers:
	- Existing network-coupled test instability in `storage_format_test.jl` blocks clean full-suite pass.
	- Increment 1.1 currently uses classifier labels as upstream signals; dedicated `tradestep!` extraction still pending.
- Next immediate step:
	- Implement Objective 1 / Increment 1.2 runtime config adapter from TrendDetector config references and parameter validation.

- Date: 2026-05-10
- Objective/Increment: Objective 1 / Legacy compatibility + migration cleanup support
- Completed:
	- Removed temporary `TradingStrategy.TradeAction` compatibility shims and completed full workspace migration to the canonical fields (`label`, `openprice`, `openix`, `closeprice`).
	- Simplified package setup scripts by removing `Pkg.update()` side effects in `EnvConfig/setup.jl`, `Features/setup.jl`, and `Ohlcv/setup.jl` while keeping `Pkg.instantiate()` + `Pkg.resolve()` flow.
	- Updated remaining legacy manifest header in `Assets/Manifest.toml` from Julia `1.10.3` to `1.12.6`.
- Files changed:
	- `TradingStrategy/src/TradingStrategy.jl`
	- `EnvConfig/setup.jl`
	- `Features/setup.jl`
	- `Ohlcv/setup.jl`
	- `Assets/Manifest.toml`
- Tests run and result:
	- `TradingStrategy` package test suite PASSED.
	- `Trade` isolated `algorithm03_adapter_test.jl` PASSED (5/5).
	- Manifest header scan for `1.10` returned no remaining matches.
- Open issues/blockers:
	- Full `Trade` package suite still includes network-coupled paths (Bybit HTTP) outside this cleanup scope.
- Next immediate step:
	- Continue Objective 1 / Increment 1.2 by binding strategy runtime config to TrendDetector references and expanding deterministic loop-step tests.

- Date: 2026-05-10
- Objective/Increment: Objective 1 / Increment 1.2 runtime config bridge
- Completed:
	- Added Trade runtime strategy bridge functions:
		- `apply_tradingstrategy!` to copy `TradingStrategy.GainSegment` parameters into `TradeCache` runtime settings.
		- `apply_trenddetector_strategy!` to accept TrendDetector-style references (`configname`, `tradingstrategy`).
		- `_validatestrategyconfig!` assertions for threshold/gain/range bounds and max window validation.
	- Added dictionary-based bridge methods for deterministic tests without exchange/network initialization.
	- Wired `strategy_maxwindow` into `_strategystate!` so live strategy state uses configured window size.
	- Added focused tests covering parameter copy, source tagging, cache reset behavior, and invalid-range rejection.
- Files changed:
	- `Trade/src/Trade.jl`
	- `Trade/test/strategy_runtime_config_test.jl`
	- `Trade/test/runtests.jl`
- Tests run and result:
	- `Trade` isolated `algorithm03_adapter_test.jl` PASSED (5/5).
	- `Trade` isolated `strategy_runtime_config_test.jl` PASSED (11/11).
- Open issues/blockers:
	- Full `Trade` package suite still includes network-coupled paths (Bybit HTTP) outside this increment scope.
- Next immediate step:
	- Continue Objective 1 with `tradestep!` extraction and deterministic loop-step tests.

## First Execution Slice (Recommended Next Coding Step)
Start Objective 4 with Increment 4.0 safety scaffolding (feature flags + shadow comparator + rollback rule), then implement Increment 4.1 exchange-owned balance cache sync before enabling async order execution.

---

## Objective 4A Workstream: WebSocket-based Market Data for Klines/Trades

### Design intent
Reduce REST API rate-limit pressure and improve latency by migrating kline (OHLCV) and trade data acquisition from HTTP polling to WebSocket streaming where supported by the exchange. This will allow for more efficient, real-time updates and lower the risk of hitting REST rate limits.

### Increments
- 4.1: Audit current exchange adapters for WebSocket support and identify gaps (Bybit, KrakenSpot, KrakenFutures).
- 4.2: Implement or enable WebSocket clients for klines and trades in each adapter.
- 4.3: Refactor polling logic in CryptoXch and Trade to consume and cache data from WebSocket streams instead of periodic HTTP fetches.
- 4.4: Add fallback to HTTP polling if WebSocket is unavailable or unreliable for a given symbol/exchange.
- 4.5: Add diagnostics and tests to verify data freshness, latency, and rate-limit impact.

### Exit criteria
- All supported exchanges use WebSocket for klines/trades where available.
- REST API usage for klines/trades is minimized, reducing rate-limit risk.
- Data freshness and latency are improved or at least equivalent to HTTP polling.
- Fallback to HTTP polling is robust and well-documented.

---

## Objective 7 (New): Strategy-Execution Contract Refactor for Independent Long/Short Action Lanes

### Design intent
Align `TradingStrategy` outputs with real execution behavior where order amount is owned by `Trade`, partial fills are normal, and long/short exposures can overlap (not net out) on KrakenSpot and KrakenFutures. This objective supersedes the earlier proposal statement that implied `buyta` carried amount and that touching `buyta.limitprice` implied complete execution.

### 2026-05-30 Architecture refinement
- `Trade` shall consume strategy output from `TradingStrategy` only and shall not call `Classify` directly at runtime.
- Required-history/readiness shall be propagated as: `Features -> Classifier -> TradingStrategy -> Trade`.
- OHLCV update propagation only shall migrate to a subscription model with `CryptoXch` as the source of new OHLCV samples.
- `Features`, `Targets`, and `Ohlcv` persistence/cache updaters subscribe to OHLCV updates and maintain derived state incrementally.
- `TradingStrategy` and `Trade` integration shall use explicit call interfaces (no pub/sub contract between these layers):
	- `TradingStrategy` pulls required classifier outputs via a clear strategy-runtime API.
	- `Trade` pulls strategy snapshots from `TradingStrategy` via a clear execution-facing API.
- `Trade` remains owner of execution sizing/risk/order lifecycle and receives strategy state snapshots plus exchange reconciliation context.

### Contract corrections
- `TradingStrategy` provides price and direction intent only; it does not provide execution amount.
- `Trade` remains the single owner of order quantity based on allocation constraints, `minimumqty`, and current inventory/borrowed state.
- Strategy state must not assume full fills when a price is touched; fill state must come from exchange order status/reconciliation.
- Strategy emits exactly one shared `label` per sample; long/short lane states remain in parallel for open/close guidance and partial-fill continuity.
- Close orders shall be adapted in each loop cycle to the actual open position/asset amount

### Target action model
- Replace `buyta`/`sellta` semantics with one shared sample label plus independent directional lane state:
	- shared `label` per sample (single source of directional intent)
	- `longta` lane state (`openprice`, `closeprice`, `openix`)
	- `shortta` lane state (`openprice`, `closeprice`, `openix`)
- Shared `label` domain:
	- `longstrongbuy`, `longbuy`, `ignore`, `longstrongclose`, `shortstrongbuy`, `shortbuy`, `shortstrongclose`
- `longta.openprice`:
	- Used as long entry limit price when shared `label == longbuy`.
- `shortta.openprice`:
	- Used as short entry limit price when shared `label == shortbuy`.
- `longta.closeprice`:
	- Always represents the close target price for open long exposure and is updated incrementally each sample.
- `shortta.closeprice`:
	- Always represents the close target price for open short exposure and is updated incrementally each sample.
- Strong labels execution semantics:
	- `longstrongbuy`: ignore `longta.openprice`; submit entry as aggressive maker intent.
	- `longstrongclose`: ignore `longta.closeprice`; close long as aggressive maker intent.
	- `shortstrongbuy`: ignore `shortta.openprice`; submit entry as aggressive maker intent.
	- `shortstrongclose`: ignore `shortta.closeprice`; close short as aggressive maker intent.
- Cancellation semantics:
	- If shared `label` is not in `{longbuy, longstrongbuy}`, cancel existing open long-entry buy orders.
	- If shared `label` is not in `{shortbuy, shortstrongbuy}`, cancel existing open short-entry sell orders.
- Consistency rule
	- Exactly one shared label is emitted per sample.

### Required `gain_limit_reversal!` behavior changes
- Remove the current implicit full-fill handoff behavior where `buyta` is swapped into `sellta` solely because `buyta.limitprice` was touched.
- Keep directional lane state independent (`longta`, `shortta`) and update each lane from observed exchange fill/reconcile state while emitting one shared sample label.
- For open exposure with missing active close guidance, synthesize close advice in-lane:
	- Long exposure: ensure `longta.closeprice > 0` with close direction and entry reference populated.
	- Short exposure: ensure `shortta.closeprice > 0` with close direction and entry reference populated.
- Synthesis inputs must use exchange-observed position/asset context (including average entry price), not simulated full-fill assumptions.

### Trade integration rules
- `Trade` consumes one shared sample label plus `longta`/`shortta` lane state in one tick.
- Open-order prices follow strategy lane entry intent (`openprice` or strong-open signal).
- Close-order prices follow strategy lane close intent (`closeprice` or strong-close signal).
- Open/close order amounts are always computed in `Trade` from:
	- allocation constraints,
	- minimum quantity constraints,
	- available free/borrowed inventory,
	- active risk controls.

### Increments
- 7.1: Introduce new lane structs/fields in `TradingStrategy` and migrate all callers to canonical lane fields.
- 7.2: Refactor `gain_limit_reversal!` to remove full-fill-by-touch assumption and emit one shared label plus lane-state intents each sample.
- 7.3: Add reconciliation hook inputs for partial fills and average entry price per lane.
- 7.4: Update `Trade` adapter path to consume TradingStrategy-only snapshots (no direct `Classify` runtime calls) and keep amount sizing in `Trade` only.
- 7.5: Add deterministic tests for overlapping long/short partial-fill scenarios and cancel/keep rules per lane.
- 7.6: Introduce CryptoXch-driven OHLCV subscription flow and migrate Features/Targets/Ohlcv cache updates off direct Trade-managed classifier refresh.
- 7.7: Add explicit `TradingStrategy` runtime call interfaces for classifier-output ingestion and `Trade` snapshot consumption (no strategy/trade pub-sub coupling).

### Draft Runtime API Signatures (Objective 7.7)

```julia
# TradingStrategy/src/runtime.jl (proposed)

abstract type AbstractStrategyRuntime end

Base.@kwdef mutable struct StrategySnapshot
	base::String
	datetime::DateTime
	label::Targets.TradeLabel = Targets.ignore  # exactly one shared label per sample
	long_openprice::Float32 = 0f0
	long_closeprice::Float32 = 0f0
	long_openix::Int = 0
	short_openprice::Float32 = 0f0
	short_closeprice::Float32 = 0f0
	short_openix::Int = 0
	probability::Float32 = 0f0
	configid::Int = 0
	source::Symbol = :tradingstrategy
end

Base.@kwdef struct StrategyReconciliationInput
	has_long_open::Bool = false
	long_avg_entry::Float32 = 0f0
	long_open_ix::Int = 0
	has_short_open::Bool = false
	short_avg_entry::Float32 = 0f0
	short_open_ix::Int = 0
end

"Required history for readiness checks (Features -> Classifier -> TradingStrategy)."
requiredhistoryminutes(rt::AbstractStrategyRuntime)::Int

"Synchronize runtime readiness state for candidate bases after OHLCV-derived updates are available."
preparebases!(rt::AbstractStrategyRuntime, xc::CryptoXch.XchCache, bases::AbstractVector{<:AbstractString};
	history_startdt::DateTime,
	datetime::DateTime,
	updatecache::Bool=false,
)::Nothing

"Return accepted/ready bases after preparebases! in canonical uppercase base symbols."
acceptedbases(rt::AbstractStrategyRuntime)::Set{String}

"Drop one base from runtime readiness/signal state."
dropbase!(rt::AbstractStrategyRuntime, base::AbstractString)::Nothing

"Reset all per-base runtime state (used on strategy config changes)."
reset!(rt::AbstractStrategyRuntime)::Nothing

"Apply gain-segment configuration to runtime."
apply_strategy!(rt::AbstractStrategyRuntime, gs::TradingStrategy.GainSegment; source::AbstractString="manual")::Nothing

"Produce one strategy snapshot for one base and sample time."
getsnapshot!(rt::AbstractStrategyRuntime, xc::CryptoXch.XchCache, base::AbstractString, datetime::DateTime;
	reconciliation::StrategyReconciliationInput,
)::Union{Nothing, StrategySnapshot}

"Batch variant for one tick."
getsnapshots!(rt::AbstractStrategyRuntime, xc::CryptoXch.XchCache, bases::AbstractVector{<:AbstractString}, datetime::DateTime;
	reconciliation_by_base::AbstractDict{String, StrategyReconciliationInput}=Dict{String, StrategyReconciliationInput}(),
)::Vector{StrategySnapshot}
```

Notes:
- `LaneState` remains an internal TradingStrategy implementation detail if needed for code reuse.
- The public runtime API contract exposes only `StrategySnapshot` flattened fields to avoid redundant public typing.
- Baseline `Trade -> TradingStrategy` reconciliation feedback is intentionally minimal: open-presence flags (`has_long_open`, `has_short_open`), average-entry hints, and optional `open_ix` markers for continuity/diagnostics.

### Trade Function Migration Map (Strategy/Classifier coupling scope)

| Current function in Trade | Decision | Target/notes |
|---|---|---|
| `_tradeselection_history_minutes(tc::TradeCache)` | Move | Replace with `TradingStrategy.requiredhistoryminutes(cache.sr)`.
| `_disablerestrictedbase!(cache, base, reason)` | Keep (edit) | Keep restricted-base behavior in Trade; remove direct `Classify.removebase!` call and call `TradingStrategy.dropbase!(cache.sr, base)`.
| `tradeselection!(tc, assetbases; ...)` | Keep (edit) | Keep portfolio/universe policy in Trade; replace classifier base add/supplement/remove/readiness with `preparebases!` + `acceptedbases` on runtime.
| `_strategyadvice(ta::Classify.TradeAdvice; ...)` | Delete | Removed once Trade no longer receives classifier advice payloads.
| `policyenforcement(cache, tavec::Vector{Classify.TradeAdvice}, ...)` | Delete | Legacy path; replace with policy on `Vector{TradingStrategy.StrategySnapshot}` where needed.
| `tradeamount(cache, tavec::Vector{Classify.TradeAdvice}, ...)` | Delete | Legacy path; sizing remains in Trade but inputs become snapshots/advice rows from runtime.
| `trade!(cache, basecfg, ta::Classify.TradeAdvice, assets)` | Delete | Compatibility overload removed; Trade executes from strategy snapshots only.
| `apply_tradingstrategy!(mc, gs; ...)` | Keep (edit) | Keep as Trade config entrypoint; delegate runtime application to `TradingStrategy.apply_strategy!(cache.sr, gs; source=...)`.
| `apply_tradingstrategy!(cache, gs; ...)` | Keep (edit) | Keep public API for scripts/tests; same delegation as above.
| `_strategyhistory!(cache, base)` | Move | Move into `TradingStrategy` runtime internals.
| `_strategystate!(cache, base)` | Move | Move into `TradingStrategy` runtime internals.
| `_upsert_getgainsalgo_sample!(history, ohlcv, label, score)` | Move | Move into `TradingStrategy` runtime internals.
| `_set_gainsalgo_reconciliation!(gs, base, assets)` | Keep (edit) | Keep in Trade as translation from portfolio/open orders to `StrategyReconciliationInput`; pass to runtime API.
| `_getgainsalgo_lane_advices(gs, ta)` | Delete | Replaced by `TradingStrategy.getsnapshot!` output contract.
| `_getgainsalgo_advices!(cache, base, ta, assets)` | Delete | Replaced by `TradingStrategy.getsnapshot!`/`getsnapshots!`.
| `_getgainsalgo_advice!(cache, base, ta, assets)` | Delete | Replaced by snapshot retrieval API.
| `_collect_strategy_advices(cache, assets)` | Keep (rewrite) | Rewrite to call `TradingStrategy.getsnapshots!` and convert snapshots to execution actions in Trade without `Classify` calls.

### Structural type changes in TradeCache
- `TradeCache.cl::Classify.AbstractClassifier` -> replace with `TradeCache.sr::TradingStrategy.AbstractStrategyRuntime`.
- Keep strategy config mirrors in `mc` during transition, but runtime-owned rolling state moves out of `Trade.mc[:strategy_state]` and `Trade.mc[:strategy_history]`.
- Update script constructors (`tradereal.jl`, `tradesim.jl`) to build strategy runtime and pass it into `TradeCache`.

### Progress update (2026-05-29)
- [x] Completed lane field rename in `TradingStrategy.TradeAction`: `orderlabel` -> `label`, `buyprice` -> `openprice`, `buyix` -> `openix`, `orderlimit` -> `closeprice`.
- [x] Completed workspace migration of call sites in `Trade` and relevant tests to canonical lane field names.
- [x] Removed temporary compatibility shims and legacy field bridging from `TradingStrategy.TradeAction`.
- [x] Updated integration-plan documentation and test descriptions to canonical field names.
- [x] Validation rerun passed after cleanup: `Trade/test/runtests.jl` = `118/118` passing (TradeAudit-dependent tests skipped in current environment).

### Progress update (2026-05-31)
- [x] Added `TradingStrategy` runtime API compatibility layer (`AbstractStrategyRuntime`, `GainSegmentRuntime`, `StrategySnapshot`, `StrategyReconciliationInput`, `preparebases!`, `acceptedbases`, `dropbase!`, `getsnapshot!`, `getsnapshots!`, and reconciliation helpers) with docstrings for the new public surface.
- [x] Moved `Trade` toward the runtime API by making strategy advice collection optionally source snapshots from `TradingStrategy` and by delegating tradeselection readiness/accepted-base bookkeeping through the runtime when enabled.
- [x] Kept `Trade` compatibility behavior intact with a test-only default gate plus explicit `CTS_USE_STRATEGY_RUNTIME_API` override, and documented the env/config inventory in a dedicated reference note.
- [x] Added and validated focused Trade runtime-path tests covering default gating, env override, advice conversion, and restricted-base runtime drop propagation.

### Exit criteria
- Strategy emits independent long-lane and short-lane intents every sample.
- No strategy path assumes complete execution from limit-price touch alone.
- `Trade` remains sole owner of amount sizing and enforces exchange constraints.
- For any open long/short exposure, corresponding close guidance is always present (explicit or synthesized) and consumed by `Trade`.

---

## Objective 8 (New): Resilient Trade Loop — Exchange Outage and Intermittent Response Handling

### Motivation (2026-05-31)
Live tradereal sessions already experience transient exchange API failures. The existing `_servertime_retry_1m` helper in `CryptoXch` handles the case where the server time endpoint is unreachable, but it does not extend to the full set of exchange interactions required per tick (open orders, portfolio, OHLCV klines). When a response fails inside `setcurrenttime!` the affected base is silently removed from `xc.bases`, which caused the `KeyError("TRX")` crash observed in the 2026-05-31 KrakenSpot tradereal run. A broader recovery strategy is needed that:
- classifies exchange errors as transient or persistent,
- suspends execution for the minimum time needed to re-establish connectivity,
- re-synchronizes all state (OHLCV, open orders, positions/portfolio) from the exchange before resuming, and
- leaves open orders and existing positions in place during the outage window rather than attempting to cancel or amend in a degraded state.

### Design intent
Give the trading loop a first-class recovery mode that activates automatically when any exchange API call fails with a connectivity-class error, polls for connectivity on a one-minute cadence, and resumes trading from a clean, fully-synchronized state once the exchange is responsive again.

### Error classification taxonomy

| Class | Examples | Loop action |
|---|---|---|
| `transient_rate_limit` | HTTP 429, cooldown-until error | Already handled via private-read cooldown path; keep as-is. |
| `transient_connectivity` | `HTTP 503`, timeout, connection refused, DNS failure | Activate outage recovery mode. |
| `persistent_auth` | Authentication failure, invalid API key | Log, alert, and abort loop (not recoverable without operator action). |
| `persistent_symbol` | Symbol not found, symbol not tradable | Log and disable the affected base for the current session only. |
| `base_ohlcv_unavailable` | `KeyError` for base in `xc.bases` after `setcurrenttime!` strips it | Skip strategy advice for that base this tick; no crash. |

The `base_ohlcv_unavailable` class is the immediate fix already applied (2026-05-31). The `transient_connectivity` class is the primary target of this objective.

### Recovery mode state machine

```
RUNNING
  │
  │ transient_connectivity error anywhere in _tradestep!
  ▼
OUTAGE_DETECTED
  │  log warning with timestamp; do not cancel/amend open orders
  ▼
POLLING_CONNECTIVITY  (once per minute)
  │  probe: exchange server time endpoint
  │  if still unreachable → sleep 60s, retry
  │  if reachable → proceed to RESYNC
  ▼
RESYNC
  │  1. re-fetch open orders  (getopenorders)
  │  2. re-fetch portfolio    (portfolio!)
  │  3. call setcurrenttime!(xc, datetime) to refresh all base OHLCV
  │     (skip and warn on individual base failures, do not removebase)
  │  if resync succeeds → RUNNING
  │  if resync itself fails → back to POLLING_CONNECTIVITY
  ▼
RUNNING  (resume normal tick cadence from current wall-clock minute)
```

### Policy constraints during outage
- No order cancellations or amendments are attempted while in `OUTAGE_DETECTED` or `POLLING_CONNECTIVITY`.
- The loop does not advance the per-minute tick counter during the outage; it resumes from the current wall-clock minute after resync.
- Maximum outage tolerated before escalating to operator alert: configurable via `CTS_OUTAGE_MAX_MINUTES` (default `60`); after this the loop sets `loop_stopping` state and logs a final alert.
- Individual base OHLCV re-fetch failures during resync produce a warning and skip that base for the tick; they do not block resync completion.
- The existing `stale_openorders_snapshot` fallback path remains active for single-tick private-read cooldowns and does not interact with outage recovery mode.

### OHLCV gap handling after resync
When the loop resumes after an outage of duration `D` minutes, the cached OHLCV for active bases has a gap of up to `D` minutes. Recovery action:
- Call `cryptoupdate!(xc, ohlcv, last_known_dt, current_dt)` for each active base to close the gap.
- Only bases that are fully caught up (OHLCV gap closed) proceed to strategy advice in the resumed tick.
- Bases with persistent OHLCV fetch failures are logged and disabled for the rest of the session (same `persistent_symbol` handling).

### Implementation surface

| Component | Change |
|---|---|
| `CryptoXch.jl` | Add `_isconnectivityerror(err)::Bool` classifier. Keep `_servertime_retry_1m` for server-time-only retry; add separate `_probeconnectivity(xc)::Bool` for lightweight reachability check. Add `_resynccache!(xc, bases)` to re-fetch OHLCV for a given set of bases. Fix `setcurrenttime!(xc, datetime)` to not call `removebase!` on transient failures; instead warn and leave the base in place (missing kline data is tolerable for one tick). |
| `Trade.jl` | Add `loop_outage` to the `LoopState` enum. Add `_activate_outage_recovery!(cache)` to transition state, log, and suppress order actions. Add `_poll_until_responsive!(cache)` with 60s sleep between probes and `CTS_OUTAGE_MAX_MINUTES` escalation. Add `_resync_after_outage!(cache)` to refresh orders, portfolio, and OHLCV. Wire these into `_run_tradeloop!` catch block: connectivity errors → outage recovery path; all other non-interrupt errors → current log-and-stop behavior. |
| `EnvConfig.jl` / env docs | Document `CTS_OUTAGE_MAX_MINUTES` in `docs/config-and-env-options-2026-05-31.md`. |

### Increments
- [ ] **8.0** Fix `setcurrenttime!(xc, datetime)` in `CryptoXch` to not call `removebase!` on transient fetch errors. *(immediate — blocks the current crash)*
- [ ] **8.1** Add `_isconnectivityerror(err)::Bool` classifier and `_probeconnectivity(xc)::Bool` in `CryptoXch`.
- [ ] **8.2** Add `loop_outage` state, `_activate_outage_recovery!`, `_poll_until_responsive!`, and `_resync_after_outage!` in `Trade`. Wire into `_run_tradeloop!`.
- [ ] **8.3** Add OHLCV gap-close pass in `_resync_after_outage!` using `cryptoupdate!` for each active base, gated behind the base-readiness check before strategy advice is collected.
- [ ] **8.4** Add `CTS_OUTAGE_MAX_MINUTES` env flag with default of 60. Log structured alert and set `loop_stopping` when threshold is exceeded.
- [ ] **8.5** Add focused tests: mock connectivity failure mid-loop, assert no cancel/amend attempted during outage, assert OHLCV gap is closed on resync, assert loop resumes.

### Exit criteria
- A connectivity-class exchange error anywhere in `_tradestep!` transitions the loop into recovery mode without raising an exception or losing base state.
- Open orders are preserved as-is during outage; no cancellation or amendment is attempted until resync confirms connectivity.
- After resync, OHLCV gaps for all active bases are closed before the first strategy advice is generated.
- The loop automatically resumes normal minute-cadence operation after a successful resync.
- If the outage exceeds `CTS_OUTAGE_MAX_MINUTES`, the loop stops cleanly and logs a structured alert for operator intervention.

### Note on immediate partial fix (2026-05-31)
Increment 8.0 is partially addressed by the `haskey(cache.xc.bases, base)` guards added to `_getgainsalgo_advices!` (Trade) and `getsnapshot!` (TradingStrategy) in the 2026-05-31 KeyError fix. The `removebase!` call in `setcurrenttime!` itself is the root cause and should be removed/scoped in Increment 8.0 for a complete fix.

---
