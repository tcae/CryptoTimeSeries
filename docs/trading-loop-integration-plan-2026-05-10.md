# Trading Loop Integration Plan (Created 2026-05-10)

## Goal
Integrate `algorithm03!` into a production-ready trading loop with exchange abstraction, audit-grade logging, asynchronous orchestration, and a non-blocking dashboard, then extend the exchange layer with Interactive Brokers.

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

---

## Objective 2 (Second): KrakenSpot + KrakenFutures as active trading exchanges through `CryptoXch`; Bybit only for OHLCV updates

### Design intent
Switch active order routing to Kraken via the abstraction layer while preserving Bybit market data ingestion for continuity.

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

### Increment 2.4: Tests
- Adapter routing tests in `CryptoXch/test/`.
- End-to-end simulation test in `Trade/test/` verifying Bybit no-trade enforcement.

### Exit criteria
- Spot and futures orders are Kraken-routed via `CryptoXch`.
- Any Bybit trade attempt is rejected by policy.
- OHLCV update jobs can still use Bybit.

### Deliverables
- Updated `CryptoXch/src/CryptoXch.jl`
- Potential updates in `Trade/src/Trade.jl`
- New routing/policy tests

---

## Objective 3 (Third): Multi-exchange audit-grade order/trade logging

### Design intent
Create immutable, complete, and queryable records sufficient for audit and tax workflows, designed for multiple exchanges in parallel.

### Increment 3.1: Canonical audit schema
- Define canonical event model with required fields:
	- event metadata: timestamp (UTC), source module, environment, correlation IDs
	- exchange metadata: exchange name, account/auth alias, market type
	- order metadata: client order id, exchange order id, symbol, side, type, tif, requested qty/price
	- execution metadata: fill qty/price, fees, fee currency, status transitions
	- position/portfolio snapshot deltas before and after action
	- strategy context: config ref, signal label/score, algorithm version
- Separate event types: `ORDER_SUBMITTED`, `ORDER_ACK`, `ORDER_PARTIAL_FILL`, `ORDER_FILLED`, `ORDER_CANCELED`, `ORDER_REJECTED`, `POSITION_SNAPSHOT`, `PORTFOLIO_SNAPSHOT`.

### Increment 3.2: Append-only log writer
- Add write-once append log sink (Arrow/CSV/JSONL based on EnvConfig format policy).
- Add partitioning by `exchange/account/date` under `$HOME/crypto/logs`.
- Add tamper-evidence option via per-file hash chain (stored in companion manifest).

### Increment 3.3: Logging integration points
- Instrument `Trade` and `CryptoXch` around all order lifecycle calls and polling updates.
- Ensure failed/rejected/cancelled paths are logged with reason codes.
- Add periodic snapshot events for portfolio state.

### Increment 3.4: Tests and replay utility
- Schema validation tests.
- Replay test that reconstructs order lifecycle from logs.
- Utility script to export tax/audit friendly reports.

### Exit criteria
- Every order status transition is persisted with traceable identifiers.
- Logs are partitioned for multiple exchanges/accounts.
- Replay utility can rebuild per-order lifecycle and daily PnL-relevant events.

### Deliverables
- New audit logging module in `Trade` or dedicated package (preferred if reused across packages)
- Tests plus report/export script

---

## Objective 4 (Fourth): Asynchronous orchestration for trading loop and exchange interactions

### Design intent
Prevent blocking between data updates, signal evaluation, order management, and portfolio refresh.

### Increment 4.1: Task topology
- Introduce asynchronous workers (Julia `Task` + `Channel`) for:
	- market data updater
	- strategy evaluator
	- order executor
	- order status reconciler
	- portfolio synchronizer
- Introduce bounded channels and backpressure strategy.

### Increment 4.2: Event-driven orchestration
- Define typed events and command messages for inter-task communication.
- Use idempotent handlers keyed by correlation/order IDs to avoid double execution.

### Increment 4.3: Safety and resilience
- Add timeout/retry policy by operation type.
- Add watchdog heartbeat and circuit breaker for exchange/API failures.
- Add graceful shutdown to flush in-flight events and persist checkpoints.

### Increment 4.4: Tests
- Concurrency tests for no deadlocks and no lost events.
- Fault-injection tests (slow API, temporary HTTP failure).

### Exit criteria
- Data refresh and order handling proceed independently.
- Dashboard-facing data feeds are not blocked by execution tasks.
- Controlled shutdown leaves consistent state.

### Deliverables
- Async orchestration layer integrated into `Trade`
- New tests for concurrency behavior

---

## Objective 5 (Fifth): Interactive non-blocking dashboard for history and open orders

### Design intent
Build on existing cockpit script and feed it with live state and audit logs without blocking the trading engine.

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
	- audit export trigger
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
- Objective 2: NOT STARTED
- Objective 3: NOT STARTED
- Objective 4: NOT STARTED
- Objective 5: NOT STARTED
- Objective 6: NOT STARTED

### Session Log Template
- Date:
- Objective/Increment:
- Completed:
- Files changed:
- Tests run and result:
- Open issues/blockers:
- Next immediate step:

### Session Log
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
	- Added legacy compatibility shims for `TradingStrategy.TradeAction` field access (`cancelrunningorder`, `amountfactor`) and legacy `orderlimit = nothing` assignment handling.
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

## First Execution Slice (Recommended Next Coding Step)
Start with Objective 1, Increment 1.1 by introducing a minimal `tradestep!` extraction and an `algorithm03` strategy adapter behind a feature flag, with one focused integration test over cached data.
