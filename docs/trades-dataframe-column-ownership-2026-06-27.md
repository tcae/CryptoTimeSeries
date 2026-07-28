# Trades DataFrame column ownership note (2026-06-27)

Update 2026-07-28:
- Canonical status spelling is `cancelled`.
- `score` is one global per-row strategy confidence field, not per trade lane.
- Account snapshot fields remain in Trades v1 and should be reused when possible to avoid redundant exchange calls.
- `set` and `rangeid` are promoted into Trades v1 and owned by TSM.

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

| Column(s) | Primary owner | Secondary writer(s) | eltype | default | Main readers | Docstring |
|---|---|---|---|---|---|---|
| `opentime` | Xch | None | `DateTime` | `DateTime[]` for an empty trades frame | TradingStrategy, Xch, TrendDetector | Ensure Trades column `opentime` exists. Owner: Xch. Eltype: `DateTime`. Note: Required unique and sorted timestamp derived from sample data. Represents the time stamp of the most recent fully closed minute as UTC. |
| `lastopentrade` | Xch | TradingStrategy (replay/simulation path only) | `Union{Missing, DateTime}` | `missing` | TradingStrategy, TrendDetector | Ensure Trades column `lastopentrade` exists. Owner: Xch. Eltype: `Union{Missing,DateTime}`. Note: Timestamp of the last open position trade, i.e. lp_amount or sp_amount increased; otherwise `missing`. |
| `pair` | Xch | None | `CategoricalVector{String}` | `"none"` | Xch, TrendDetector, TradingStrategy | Ensure Trades column `pair` exists. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: Identifier of the trading pair. |
| `set` | TSM | None | `CategoricalVector{String}` | `TSM_NO_SET` | TradingStrategy, Xch, TrendDetector | Ensure Trades column `set` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Denotes the logical run set (for example train/test/eval/production). |
| `rangeid` | TSM | None | `Int32` | `0` | TradingStrategy, Xch, TrendDetector | Ensure Trades column `rangeid` exists. Owner: TSM. Eltype: `Int32`. Denotes one consecutive liquidity range identifier within one pair data set. |
| `label` | TradingStrategy | Trade (reserved override only if explicitly designed) | `TradeLabel` | `ignore` | Xch, TrendDetector | Ensure Trades column `label` exists. Owner: TradingStrategy. Eltype: `TradeLabel` with `ignore` as the default. Note: label represents the TradingStrategy trading advice. |
| `score` | TradingStrategy | None | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades column `score` exists. Owner: TradingStrategy. Eltype: `Float32`. Note: likelihood of the label to be correct from TradingStrategy. |
| `lo_limit`, `lc_limit`, `so_limit`, `sc_limit` | TradingStrategy | Trade (reserved override before request processing) | `Float32` | `0f0` | Xch, TrendDetector | Ensure Trades lane column `<lane>_limit` exists. Owner: TradingStrategy. Eltype: `Float32` with `0f0` as the default. Note: order limit in case of a currently active order for that trade lane. |
| `lo_amount`, `lc_amount`, `so_amount`, `sc_amount` | Trade | None | `Float32` | `0f0` | Xch | Ensure Trades lane column `<lane>_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: if order amount > 0 then order shall be placed, otherwise not. |
| `lo_id`, `lc_id`, `so_id`, `sc_id`, `lol_id`, `lcl_id`, `sol_id`, `scl_id` | Xch | None | `CategoricalVector{String}` | `TSM_NO_ORDER_ID` | Trade, Xch | Ensure Trades lane column `<lane>_id` and last-lane column `<lane>l_id` exist. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: exchange provided id of currently active order and last minute active order; otherwise TSM_NO_ORDER_ID. |
| `lo_status`, `lc_status`, `so_status`, `sc_status`, `lol_status`, `lcl_status`, `sol_status`, `scl_status` | Xch | None | `CategoricalVector{String}` | `TSM_NO_STATE` | Trade, Xch | Ensure Trades lane column `<lane>_status` and last-lane column `<lane>l_status` exist. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: order status of currently active order and last minute active order as one of the following: TSM_NO_STATE, `submitted`, `closed`, `cancelled`, `rejected`. |
| `lol_filled`, `lcl_filled`, `sol_filled`, `scl_filled` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades last-lane column `<lane>l_filled` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: filled/executed base quantity of the last minute active order. |
| `lol_pavg`, `lcl_pavg`, `sol_pavg`, `scl_pavg` | Xch | None | `Float32` | `0f0` | Trade, Xch | Ensure Trades last-lane column `<lane>l_pavg` exists. Owner: Xch. Eltype: `Float32` with `0f0` as default. Note: average fill price in quote units of the last minute active order. |
| `lo_msg`, `lc_msg`, `so_msg`, `sc_msg`, `lol_msg`, `lcl_msg`, `sol_msg`, `scl_msg` | Xch | None | `CategoricalVector{String}` | `TSM_NO_ORDER_MSG` | Trade | Ensure Trades lane column `<lane>_msg` and last-lane column `<lane>l_msg` exist. Owner: Xch. Eltype: `CategoricalVector{String}`. Note: rejection/error message text for the currently active order and last minute active order. |
| `lp_amount` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `lp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Long position amount of trading pair holdings. |
| `sp_amount` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades column `sp_amount` exists. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Short position amount of trading pair holdings. |
| `close`, `high`, `low` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades columns `close`, `high`, `low` exist. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Note: Last completed minute close/high/low price of the trading pair. |
| `equity`, `freemargin`, `freequote` | Xch | None | `Float32` | `0f0` | Trade | Ensure Trades columns `equity`, `freemargin`, `freequote` exist. Owner: Xch. Eltype: `Float32` with `0f0` as the default. Notes: `equity` is the most recent equity in quote units as constraint for maximum relative allocation of a trading pair; `freemargin` is free account margin amount in quote units and currently equal to `freequote`; `freequote` is free account amount for orders in quote units. |
| `config` | TSM | None | `CategoricalVector{String}` | `TSM_NO_CONFIG` | TSM, Xch, TradingStrategy | Ensure Trades column `config` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. Identifies the Trade configuration id. Any change in config, e.g. different openthresholds, shall result in a different config marker. |
| `tsmstate` | TSM | None | `CategoricalVector{String}` | `TSM_NO_STATE` | TSM, Xch, TradingStrategy | Ensure Trades column `tsmstate` exists. Owner: TSM. Eltype: `CategoricalVector{String}`. `sync`: execution and price changes of the most recent minute are updated in the current row fields; next is `request`. `request`: based on data of the previous minute order requests are defined; next is `xch`. `xch`: order requests are submitted to the exchange; next is `sync`. |

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
