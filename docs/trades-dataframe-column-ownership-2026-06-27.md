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
| `opentime` | Xch | None | `DateTime` | `DateTime[]` for an empty trades frame | TradingStrategy, Xch, TrendDetector | Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data. |
| `lastopentrade` | Xch | TradingStrategy (replay/simulation path only) | `Union{Missing, DateTime}` | `missing` | TradingStrategy, TrendDetector | Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open-trade event for the pair while `lp_amount > 0f0` or `sp_amount > 0f0`; otherwise `missing`. |
| `pair` | Xch | None | `CategoricalVector{String}` | `"none"` | Xch, TrendDetector, TradingStrategy | Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Required identity/routing column of the trading pair used by Xch. |
| `label` | TradingStrategy | Trade (reserved override only if explicitly designed) | `TradeLabel` | `ignore` | Xch, TrendDetector | Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: TradingStrategy writes enum labels; Xch consumes them to map open/close actions. |
| `score` | TradingStrategy | None | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: Strategy confidence/score for the active label. |
| `lo_limit`, `lc_limit`, `so_limit`, `sc_limit` | TradingStrategy | Trade (reserved override before request processing) | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades column `lo_limit`/`lc_limit`/`so_limit`/`sc_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: Strategy guidance consumed by Xch as requested limit per action. |
| `lo_amount`, `lc_amount`, `so_amount`, `sc_amount` | Trade | None | `Float32` | `0f0` | Xch | Ensure Trades column `lo_amount`/`lc_amount`/`so_amount`/`sc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size consumed by Xch order processing. |
| `longleverage`, `shortleverage` | Trade | None | `UInt8` | `1` | Xch | Ensure Trades column `longleverage`/`shortleverage` exists. Owner: Trade. Eltype: `UInt8` with `1` as the default. Note: Request leverage column consumed by Xch order processing. |
| `lo_id`, `lc_id`, `so_id`, `sc_id` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade, Xch | Ensure Trades column `lo_id`/`lc_id`/`so_id`/`sc_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Exchange order id of a submit/amend/close request. |
| `lo_status`, `lc_status`, `so_status`, `sc_status` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade, Xch | Ensure Trades column `lo_status`/`lc_status`/`so_status`/`sc_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Order status states (mapping via normalize_order_status): none, submitted, closed, canceled, rejected. |
| `lo_filled`, `lc_filled`, `so_filled`, `sc_filled` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades column `lo_filled`/`lc_filled`/`so_filled`/`sc_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Remaining base quantity from order status reconciliation. |
| `lo_pavg`, `lc_pavg`, `so_pavg`, `sc_pavg` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades column `lo_pavg`/`lc_pavg`/`so_pavg`/`sc_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Average fill price from exchange order status. Will not be reset at order close time but at order creation time, so that the average price of a closed order can be stored for later analysis. |
| `lo_msg`, `lc_msg`, `so_msg`, `sc_msg` | Xch | None | `CategoricalVector{String}` | `"none"` | Trade | Ensure Trades column `lo_msg`/`lc_msg`/`so_msg`/`sc_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Direct rejection/error message text (categorical). |
| `lp_amount` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount snapshot for the trading pair. |
| `sp_amount` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount snapshot for the trading pair. |
| `close`, `high`, `low` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `close`/`high`/`low` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: OHLCV sample prices for the trading pair. |
| `maintmargin`, `equity`, `balance`, `freemargin`, `freequote` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `maintmargin`/`equity`/`balance`/`freemargin`/`freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: `maintmargin`: maintenance margin of position; `equity`: account equity amount of trading pair base; `balance`: account balance amount of trading pair base; `freemargin`: free margin amount of trading pair base; `freequote`: free quote amount of trading pair base. |
| `set`, `rangeid` | TrendDetector | None | `String`, `Int` | `""`, `0` | TrendDetector | Ensure Trades column `set`/`rangeid` exists. Owner: TrendDetector. Eltype: `String`/`Int`. Note: Replay metadata contributed by `TrendDetector.tradesdf_contributors()` and not part of the Xch v1 contract. |

order status mapping via XchCore.normalize_order_status():

| Xch | Bybit | Kraken Spot | Kraken Futures |
|---|---|---|---|
| none | no order | no order | no order |
| submitted | Created | — | — |
| submitted | New, Untriggered | open | open |
| submitted | Triggered | — | — |
| submitted | PartiallyFilled | open (with partial fill) | open (with partial fill) |
| closed | Filled | closed | filled |
| canceled | Cancelled, Deactivated | canceled | canceled |
| rejected | — | expired | — |
| rejected | Rejected | — | rejected |

## Runtime helper columns (non-v1 contract)

These can appear during strategy execution but are not part of `TRADES_V1_REQUIRED_COLUMNS`:
- `set`, `rangeid`

`predicted`, `openthreshold`, and `closethreshold` are gaindf metadata columns written by TrendDetector gain post-processing (`addgainadmin!`), not Trades helper columns.

Ownership:
- Primary writer: TrendDetector
- Primary readers: TrendDetector
- Rule: treat as runtime helper columns; do not depend on them as persistence contract.

## Allowed mutation rules by module

### Xch
- May create/normalize any missing v1 columns.
- Must be the only module mutating execution and account feedback columns.
- Must not reinterpret strategy columns except to consume them for order processing.

### TradingStrategy
- May mutate only strategy advice/state columns.
- Must not mutate Xch-owned execution feedback columns (`*id`, `*status`, `*filled`, `*pavg`, `*msg`), position snapshot columns (`lp_amount`, `sp_amount`), or account snapshot columns (`maintmargin`, `equity`, `balance`, `freemargin`, `freequote`).

## Implementation status note

- The architecture target is strict ownership as defined above (Xch owns row identity and row creation).
- If any current helper path still writes identity metadata from TradingStrategy, treat that as transitional behavior and migrate it to an Xch row-provisioning helper.

### Trade
- May mutate request sizing/leverage columns (`lo_amount`, `lc_amount`, `so_amount`, `sc_amount`, `longleverage`, `shortleverage`).
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
