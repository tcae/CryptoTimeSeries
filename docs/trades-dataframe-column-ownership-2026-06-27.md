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
- Xch is the authoritative producer for row identity columns (`opentime`, `pair`, `coin`) because it owns market-sample ingestion from exchange/ohlcv sources.
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

| Column(s) | Primary owner | Secondary writer(s) | Main readers | Notes |
|---|---|---|---|---|
| `opentime` | Xch | None | TradingStrategy, Xch, TrendDetector | Required identity column derived from sample data. |
| `lastopentrade` | TradingStrategy | None | TradingStrategy, TrendDetector | Updated during strategy/gain materialization state tracking. |
| `pair`, `coin` | Xch | None | Xch, TrendDetector, TradingStrategy | Identity/routing columns used by Xch to derive request pair. |
| `tradelabel`, `labelscore` | TradingStrategy | Trade (reserved override only if explicitly designed) | Xch, TrendDetector | `tradelabel` drives open/close action mapping in Xch. |
| `longopenlimit`, `longcloselimit`, `shortopenlimit`, `shortcloselimit` | TradingStrategy | Trade (reserved override before request processing) | Xch, TrendDetector | Strategy guidance; also consumed by Xch as requested limit per action. |
| `longamount`, `shortamount` | Trade | None | Xch | Request sizing columns consumed by Xch order processing. |
| `longleverage`, `shortleverage` | Trade | None | Xch | Request leverage columns consumed by Xch order processing. I don't find them back and they are also not required.|
| `longid`, `shortid` | Xch | None | Trade, Xch | Exchange order ids written after submit/amend/close request. |
| `longstatus`, `shortstatus` | Xch | None | Trade, Xch | Status transitions (`Submitted`, `Rejected`, `Missing`, `Error`, exchange statuses). |
| `longunfilled`, `shortunfilled` | Xch | None | Trade, Xch | Remaining base quantity from order status reconciliation. |
| `longpriceavg`, `shortpriceavg` | Xch | None | Trade, Xch | Average fill price from exchange order status. |
| `longmsgid`, `shortmsgid` | Xch | None | Trade, diagnostics | Message catalog id for rejection/errors. |
| `postype`, `posleverage`, `posamount` | Xch | None | Trade, diagnostics | Position-side/accounting snapshot for current action row. |
| `quoteprice`, `maintmargin`, `equity`, `balance`, `freemargin`, `freequote` | Xch | None | Trade, diagnostics | Account snapshot fields applied during request processing/status refresh. |

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
- Must not mutate execution feedback columns (`*id`, `*status`, `*unfilled`, `*priceavg`, `*msgid`) or account snapshot columns.

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
