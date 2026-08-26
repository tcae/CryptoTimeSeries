# Trades DataFrame column ownership note (2026-06-27)

Update 2026-07-28:
- Canonical status spelling is `cancelled`.
- `score` is one global per-row strategy confidence field, not per trade lane.
- Account snapshot fields remain in Trades v1 and should be reused when possible to avoid redundant exchange calls.
- `set` and `rangeid` are promoted into Trades v1 and owned by TSM.

Update 2026-08-26:
- The column ownership matrix below is regenerated directly from the `Ensure Trades column ...` docstrings in `TSM/src/TSM.jl` (the ensure-function docstrings are the source of truth; keep them in sync first, this table second).
- `*_id`/`*l_id` order-id lane columns (`lo_id`, `lc_id`, `so_id`, `sc_id`, `lol_id`, `lcl_id`, `sol_id`, `scl_id`, `lcsl_id`, `scsl_id`) are uncompressed `CategoricalVector{String}` (`compress=false`): unlike `status`/`pair`/`set`/`config`/`tsmstate` (small fixed vocabularies, kept compressed), order ids have unbounded cardinality and would risk overflowing a compressed `UInt8` pool on long runs.
- `tsmstate` is now actually written at runtime (previously documented but unused): `TSM.ensuretradesrow!` sets `sync`, `Trade.trade!` sets `request` before running strategy/sizing logic, and `Trade.trade!` sets `xch` right before handing the row to `Xch.process_order_request` (terminal for that row).

## Purpose

This note defines which module owns which columns of the Trades DataFrame and which modules may read or mutate them.

Scope:
- Applies to Trades v1 schema enforced by TSM and consumed by Xch.
- Applies to runtime usage in TradingStrategy, Trade, and TrendDetector.

Primary references in code:
- TSM schema contract: `TSM/src/TSM.jl` (`tradesdf_all_contributors`, `ensuretradeschema!`, contributor helpers)
- TradingStrategy row-state logic: `TradingStrategy/src/TradingStrategy.jl` (`gettradesrow!`, `reachgainuntilreversal!`)
- Xch execution feedback logic: `Xch/src/XchCore.jl` (`process_order_request`, `order_status`)

## Ownership principles

1. TSM owns DataFrame lifecycle and schema contract
- TSM owns the mutable per-pair DataFrame instances in `TsmCache.pairstates`.
- Any DataFrame provisioned through TSM pair-state APIs is normalized via `ensuretradeschema!` and contributor helpers.
- Access to row fields shall only be done via TSM get set functions that include consistency checks
- Xch consumes this contract and mutates only its owned columns.

2. Xch owns row creation and identity metadata
- Xch is the authoritative producer for row identity columns (`opentime`, `pair`) because it owns market-sample ingestion from exchange/ohlcv sources.
- TradingStrategy must consume existing rows and write advice columns only.

3. TradingStrategy owns strategy advice state
- TradingStrategy is the authoritative producer of strategy advice columns used for gain materialization and order intent.
- TradingStrategy does not own row creation.

4. Trade owns request sizing intent
- Trade is the intended owner of request amount columns.
- These values are consumed by Xch in `process_order_request`.
- If not set, Xch applies fallback behavior (for example close amount from balances).

5. Xch owns execution/account feedback
- Xch is the authoritative writer for exchange ids, status, fills, average prices, message ids, position summary, and account snapshot fields.

6. TrendDetector is a consumer of persisted outputs
- TrendDetector can pass metadata columns through the DataFrame for diagnostics, but does not own v1 contract columns.

## Column ownership matrix (Trades v1 contract)

Docstring text below is copied verbatim from the `Ensure Trades column ...` functions in `TSM/src/TSM.jl`; `<lane>` is one of `lo`/`lc`/`so`/`sc`, `<lane>l` its last-minute counterpart, and `<lcsl|scsl>` the close-bracket stop-loss leg of a long/short close lane.

| Column(s) | Owner | Docstring (`TSM/src/TSM.jl`) |
|---|---|---|
| `set` | TSM | Ensure Trades column `set` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Denotes the logical run set (for example train/test/eval/production). |
| `rangeid` | TSM | Ensure Trades column `rangeid` exists. Owner: TSM. Eltype: `Int32`. Denotes one consecutive liquidity range identifier within one pair data set. |
| `config` | TSM | Ensure Trades column `config` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Identifies the Trade configuration id. Any change in config, e.g. different openthresholds, shall result in a different config marker. |
| `tsmstate` | TSM | Ensure Trades column `tsmstate` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Per-row progression (each row is visited once per minute, `TSM_NO_STATE` default until visited): *sync*: the row becomes the active row for its minute; price/execution fields are synced; next is *request*. *request*: TradingStrategy and `Trade.trade!` evaluate the row before handing it to Xch; next is *xch*. *xch*: the row is handed to Xch for order request processing; terminal state, the row does not revisit *sync*. |
| `opentime` | Xch | Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data. Represents the time stamp of the most recent fully closed minute as UTC. |
| `lastopentrade` | Xch | Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open position trade, i.e. lp_amount or sp_amount increased; otherwise `missing`. |
| `pair` | Xch | Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Identifier of the trading pair. |
| `<lane>_id` | Xch | Ensure Trades lane column `<lane>_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}` (uncompressed). Note: exchange provided id of currently active order; otherwise TSM_NO_ORDER_ID. |
| `<lane>_status` | Xch | Ensure Trades lane column `<lane>_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of currently active order as one of the following: TSM_NO_STATE, `submitted`, `closed`, `cancelled`, `rejected`. |
| `<lane>_msg` | Xch | Ensure Trades lane column `<lane>_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the currently active order. |
| `<lane>l_id` | Xch | Ensure Trades last-lane column `<lane>l_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}` (uncompressed). Note: exchange provided id of last minute active order; otherwise TSM_NO_ORDER_ID. |
| `<lane>l_status` | Xch | Ensure Trades last-lane column `<lane>l_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of the last minute active order as one of the following: TSM_NO_STATE, `submitted`, `closed`, `cancelled`, `rejected`. |
| `<lane>l_msg` | Xch | Ensure Trades last-lane column `<lane>l_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the last minute active order. |
| `<lane>l_filled` | Xch | Ensure Trades last-lane column `<lane>l_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: Filled/executed base quantity of the last minute active order. |
| `<lane>l_pavg` | Xch | Ensure Trades last-lane column `<lane>l_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: Average fill price in quote units of the last minute active order. |
| `lp_amount` | Xch | Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount of trading pair holdings. |
| `sp_amount` | Xch | Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount of trading pair holdings. |
| `<lcsl\|scsl>_id` | Xch | Ensure Trades close-bracket stop column `<lcsl\|scsl>_id` exists. Owner: Xch. Eltype: `CategoricalVector{String}` (uncompressed). Note: exchange provided id of the resting stop-loss leg of the close bracket; otherwise TSM_NO_ORDER_ID. |
| `<lcsl\|scsl>_status` | Xch | Ensure Trades close-bracket stop column `<lcsl\|scsl>_status` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of the resting stop-loss leg of the close bracket. |
| `<lcsl\|scsl>_msg` | Xch | Ensure Trades close-bracket stop column `<lcsl\|scsl>_msg` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the stop-loss leg of the close bracket. |
| `<lcsl\|scsl>_limit` | TradingStrategy | Ensure Trades close-bracket stop column `<lcsl\|scsl>_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: stop-loss limit price of the close bracket; `0f0` means no stop-loss leg is requested. Both bracket legs cover the same quantity, tracked by `<lc\|sc>_amount`. |
| `close` | Xch | Ensure Trades column `close` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute close price of the trading pair. |
| `high` | Xch | Ensure Trades column `high` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute high price of trading pair. |
| `low` | Xch | Ensure Trades column `low` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute low price of trading pair. |
| `equity` | Xch | Ensure Trades column `equity` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Most recent equity in quote units as constraint for maximum relative allocation of a trading pair. |
| `freemargin` | Xch | Ensure Trades column `freemargin` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free account margin amount in quote units. Currently equal to freequote. |
| `freequote` | Xch | Ensure Trades column `freequote` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Free account amount for orders in quote units. |
| `label` | TradingStrategy | Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: label represents the TradingStrategy trading advice. |
| `score` | TradingStrategy | Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: likelihood of the label to be correct from TradingStrategy. |
| `<lane>_limit` | TradingStrategy | Ensure Trades lane column `<lane>_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: order limit in case of a currently active order for that trade lane. |
| `<lane>_amount` | Trade | Ensure Trades lane column `<lane>_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: if order amount > 0 then order shall be placed, otherwise not. If a close order amount > 0 then a close order shall be placed. |

order status mapping via XchCore.normalize_order_status():

| Xch | Bybit | Kraken Spot | Kraken Futures |
|---|---|---|---|
| none | no order | no order | no order |
| submitted | Created | — | — |
| submitted | New, Untriggered | open | open |
| submitted | Triggered | — | — |
| submitted | PartiallyFilled | open (with partial fill) | open (with partial fill) |
| closed | Filled | closed | filled |
| cancelled | Cancelled, Deactivated, Canceled | cancelled, canceled | cancelled, canceled |
| rejected | — | expired | — |
| rejected | Rejected | — | rejected |

Normalization policy:
- Canonical internal status vocabulary is: `none`, `submitted`, `closed`, `cancelled`, `rejected`.
- Adapter-specific spellings (`canceled` vs `cancelled`) are normalized at adapter boundaries before writing Trades rows.

## Runtime helper columns (outside Trades v1)

`predicted`, `openthreshold`, and `closethreshold` are gaindf metadata columns written by TrendDetector gain post-processing (`addgainadmin!`), not Trades v1 columns.

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
- Must not mutate Xch-owned execution feedback columns (`*id`, `*status`, `*filled`, `*pavg`, `*msg`), position snapshot columns (`lp_amount`, `sp_amount`), or account snapshot columns (`equity`, `freemargin`, `freequote`).

## Implementation status note

- The architecture target is strict ownership as defined above (Xch owns row identity and row creation).
- If any current helper path still writes identity metadata from TradingStrategy, treat that as transitional behavior and migrate it to an Xch row-provisioning helper.

### Trade
- May mutate request sizing columns (`lo_amount`, `lc_amount`, `so_amount`, `sc_amount`).
- May request controlled override of strategy limits only as an explicit design decision.
- Must not mutate Xch-owned execution/account feedback columns.

### TrendDetector
- May append diagnostics metadata columns for analysis output.
- Must treat v1 contract columns as data input/output, not ownership targets.
- Must not redefine ownership semantics of `set` and `rangeid`; those are TSM-owned v1 fields.

## Conflict resolution policy

If multiple modules attempt to write the same column, ownership precedence is:
1. TSM for TSM-owned contract columns (`set`, `rangeid`, `config`, `tsmstate`)
2. Xch for execution/account feedback columns
3. TradingStrategy for strategy advice columns
4. Trade for request sizing columns

Any deviation should be implemented as an explicit API-level exception and documented in this note.

## Practical checklist for contributors

Before adding or changing Trades columns:
1. Add/adjust schema in TSM (`tradesdf_all_contributors`, `ensuretradeschema!`, and relevant contributor helpers).
2. Assign one primary owner module in this note.
3. Add/adjust tests in owner module and consumer module(s).
4. Ensure no cross-owner overwrite is introduced.
5. If adding non-contract helper columns, mark them explicitly as non-v1 contract.
