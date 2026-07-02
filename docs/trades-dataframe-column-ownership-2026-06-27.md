# Trades DataFrame column ownership note (2026-06-27)

## Purpose

This note defines which module owns which columns of the Trades DataFrame and which modules may read or mutate them.

Scope:
- Applies to Trades v1 schema enforced by Xch.
- Applies to runtime usage in TradingStrategy, Trade, and TrendDetector.

Primary references in code:
- Xch schema contract: `Xch/src/XchCore.jl` (`TRADES_V1_REQUIRED_COLUMNS`, `_ensuretradesv1schema`, `_asserttradesv1schema`)
- TradingStrategy row-state logic: `TradingStrategy/src/TradingStrategy.jl` (`gettradesrow!`, `reachgainuntilreversal!`)
- Xch execution feedback logic: `Xch/src/XchCore.jl` (`process_order_request`, `order_status`)

## Ownership principles

1. Xch owns DataFrame lifecycle and schema
- Xch owns the mutable per-pair DataFrame instances in `XchCache.pairstates`.
- Any DataFrame written into Xch via `settrades!` is normalized by `_ensuretradesv1schema`.
- Runtime execution entry points assert schema via `_asserttradesv1schema`.

2. Xch owns row creation and identity metadata
- Xch is the authoritative producer for row identity columns (`opentime`, `pair`) because it owns market-sample ingestion from exchange/ohlcv sources.
- TradingStrategy must consume existing rows and write advice columns only.

3. TradingStrategy owns strategy advice state
- TradingStrategy is the authoritative producer of strategy advice columns used for gain materialization and order intent.
- TradingStrategy does not own row creation.

4. Trade owns request sizing and leverage intent
- Trade is the intended owner of request amount/leverage columns.
- These values are consumed by Xch in `process_order_request`.
- If not set, Xch applies fallback behavior (for example close amount from balances).

5. Xch owns execution/account feedback
- Xch is the authoritative writer for exchange ids, status, fills, average prices, message ids, position summary, and account snapshot fields.

6. TrendDetector is a consumer of persisted outputs
- TrendDetector can pass metadata columns through the DataFrame for diagnostics, but does not own v1 contract columns.

## Column ownership matrix (Trades v1 contract)

| Column(s) | Primary owner | Secondary writer(s) | eltype | default | Main readers | Docstring |
|---|---|---|---|---|---|---|
| `opentime` | Xch | None | `DateTime` | `DateTime[]` for an empty trades frame | TradingStrategy, Xch, TrendDetector | Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required identity column derived from sample data. |
| `lastopentrade` | Xch | TradingStrategy (replay/simulation path only) | `Union{Missing, DateTime}` | `missing` | TradingStrategy, TrendDetector | Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: DateTime when the last open trade was executed (filled with executedqty > 0), not when an order was submitted. Carry forward only while position amount is open (`abs(longamount)>0` or `abs(shortamount)>0`); reset to `missing` when no position is open. |
| `pair` | Xch | None | `CategoricalVector{String}` | `"none"` | Xch, TrendDetector, TradingStrategy | Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Identity/routing column used by Xch to derive the request pair; materialized rows must always fill it. |
| `tradelabel` | TradingStrategy | Trade (reserved override only if explicitly designed) | `TradeLabel` | `ignore` | Xch, TrendDetector | Ensure Trades column `tradelabel` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: TradingStrategy writes enum labels; Xch consumes them to map open/close actions. |
| `labelscore` | TradingStrategy | None | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades column `labelscore` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: Strategy confidence/score for the active label. |
| `longopenlimit`, `longcloselimit`, `shortopenlimit`, `shortcloselimit` | TradingStrategy | Trade (reserved override before request processing) | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades column `longopenlimit`/`longcloselimit`/`shortopenlimit`/`shortcloselimit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance consumed by Xch as requested limit per action. |
| `longamount`, `shortamount` | Trade | None | `Float32` | `0f0` | Xch | Ensure Trades column `longamount`/`shortamount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request sizing column consumed by Xch order processing. |
| `longleverage`, `shortleverage` | Trade | None | `UInt8` | `1` | Xch | Ensure Trades column `longleverage`/`shortleverage` exists. Owner: Trade. Eltype: `UInt8` with `1` as the default. Note: Request leverage column consumed by Xch order processing. |
| `longid`, `shortid` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade, Xch | Ensure Trades column `longid`/`shortid` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order ids written after submit/amend/close request. |
| `longstatus`, `shortstatus` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade, Xch | Ensure Trades column `longstatus`/`shortstatus` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Status transitions (`Submitted`, `Rejected`, `Missing`, `Error`, exchange statuses). |
| `longunfilled`, `shortunfilled` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades column `longunfilled`/`shortunfilled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Remaining base quantity from order status reconciliation. |
| `longpriceavg`, `shortpriceavg` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades column `longpriceavg`/`shortpriceavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. |
| `longmsg`, `shortmsg` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade, diagnostics | Ensure Trades column `longmsg`/`shortmsg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical). |
| `postype` | Xch | None | `Targets.TrendPhase` | `Targets.flat` | Trade, diagnostics | Ensure Trades column `postype` exists. Owner: Xch. Eltype: `Targets.TrendPhase`. Note: Position-side/accounting snapshot for the current action row. |
| `posamount` | Xch | None | `Float32` | `0f0` | Trade, diagnostics | Ensure Trades column `posamount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Position-side/accounting snapshot values for the current action row. |
| `quoteprice`, `maintmargin`, `equity`, `balance`, `freemargin`, `freequote` | Xch | None | `Float32` | `0f0` | Trade, diagnostics | Ensure Trades column `quoteprice`/`maintmargin`/`equity`/`balance`/`freemargin`/`freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Account snapshot field applied during request processing/status refresh. |

## Runtime helper columns (non-v1 contract)

These can appear during strategy execution but are not part of `TRADES_V1_REQUIRED_COLUMNS`:
- `label`, `score`, `high`, `low`, `close`

Ownership:
- Primary writer: TradingStrategy
- Primary readers: TradingStrategy and TrendDetector diagnostics
- Rule: treat as runtime helper columns; do not depend on them as persistence contract.

## Allowed mutation rules by module

### Xch
- May create/normalize any missing v1 columns.
- Must be the only module mutating execution and account feedback columns.
- Must not reinterpret strategy columns except to consume them for order processing.

### TradingStrategy
- May mutate only strategy advice/state columns.
- Must not mutate Xch-owned execution feedback columns (`*id`, `*status`, `*unfilled`, `*priceavg`, `*msg`), position snapshot columns (`postype`, `posamount`), or account snapshot columns (`quoteprice`, `maintmargin`, `equity`, `balance`, `freemargin`, `freequote`).

## Implementation status note

- The architecture target is strict ownership as defined above (Xch owns row identity and row creation).
- If any current helper path still writes identity metadata from TradingStrategy, treat that as transitional behavior and migrate it to an Xch row-provisioning helper.

### Trade
- May mutate request sizing/leverage columns (`longamount`, `shortamount`, `longleverage`, `shortleverage`).
- May request controlled override of strategy limits only as an explicit design decision.
- Must not mutate Xch-owned execution/account feedback columns.

### TrendDetector
- May append diagnostics metadata columns for analysis output.
- Must treat v1 contract columns as data input/output, not ownership targets.

## Conflict resolution policy

If multiple modules attempt to write the same column, ownership precedence is:
1. Xch for execution/account feedback columns
2. TradingStrategy for strategy advice columns
3. Trade for request sizing/leverage columns

Any deviation should be implemented as an explicit API-level exception and documented in this note.

## Practical checklist for contributors

Before adding or changing Trades columns:
1. Add/adjust schema in Xch (`TRADES_V1_REQUIRED_COLUMNS` and `_ensuretradesv1schema`).
2. Assign one primary owner module in this note.
3. Add/adjust tests in owner module and consumer module(s).
4. Ensure no cross-owner overwrite is introduced.
5. If adding non-contract helper columns, mark them explicitly as non-v1 contract.
