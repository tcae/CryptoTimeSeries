# Minimum exchange interactions to implement the trade loop

## Problem Statement

Too many exchange interactions reduce runtime performance of the trading loop. The following groups of interaction are required:

- Klines for relevant trading pairs. Relevant are all holdings (assets and positions) and all tradable (open and close enabled) trading pairs.
- Order creation, change, cancellation order info (including status). 
- Account status: requires some thoughts what is minimal required because of assets, positions, balances, margin, ...

While every exchange provides functionality for Klines and orders for REST and websockets, there are only a few hints to consider and no need to discuss this in detail.

## Account status

- The maximum amount of available quote budget is required. While the volume of a position can exceed the asset by leveraging, a safe side approach is chosen to only invest in spot as well as futures and margin only the amount of free quote budget. This is a safe side approach, especially if some percentage margin is held back as quote value. Conclusion: free asset quote value is required.
- The amount of a position shall not exceed a certain percentage of the total equity. Therefore the amount of holdings of every asset and position is required. 
- For assets the separation of free and lock amounts per asset shall be available.
- The sum of every sum(holding * current price) is the current equity and serves as basis to determine the amount a holding should not exceed, which is checked before opening a position and not something which is actively managed back because equity goes down temporarily. With the amount of all holdings and the corresponding kline information it is possible to calculate the equity.

### Design Approach

- Per trading pair (i.e. openenabled or closeenabled pairs) and per minute an update of last minute data is required that provides the basis to decide the trading strategy and the asset allocation for the current minute.
- Trades v1 contract is authoritative and explicitly defined in `docs/trades-dataframe-column-ownership-2026-06-27.md` and `TSM/src/TSM.jl`.
- TSM maintains for that purpose a Dict of trading pairs and an associated dataframe per trading pair.
- Administrative and state columns of the v1 contract include:
  - `pair`, `set`, `rangeid`, `config`, `tsmstate`, `opentime`, `lastopentrade`
- Xch writes exchange feedback and account snapshot columns of the v1 contract, including:
  - lane order ids/status/messages for currently active orders (`lo/lc/so/sc`) and last minute execution status (`lol/lcl/sol/scl`)
  - last minute execution metrics (`lol_filled`, `lcl_filled`, `sol_filled`, `scl_filled`, `lol_pavg`, `lcl_pavg`, `sol_pavg`, `scl_pavg`)
  - market and position snapshot (`close`, `high`, `low`, `lp_amount`, `sp_amount`)
  - account snapshot in quote units (`equity`, `freequote`, `freemargin`)
- TradingStrategy writes strategy advice columns of the v1 contract:
  - `label`, `score`, and lane limits (`lo_limit`, `lc_limit`, `so_limit`, `sc_limit`)
- Trade writes lane request amount columns of the v1 contract:
  - `lo_amount`, `lc_amount`, `so_amount`, `sc_amount` and may revise label and limits
- Status vocabulary shall use canonical spelling:
  - `none`, `submitted`, `closed`, `cancelled`, `rejected`


#### Holdings: Assets and Positions

A function per exchange is required that provides a portfolio dataframe with each row representing a holding and the following columns:
symbol, short position amount, long position amount

#### Equity and Free Amount

A function is required that provides in units of quote currency:
equity amount, free amount 

### Kraken Futures

#### Equity and Free Amount

- the REST endpoint accounts.flex (or the corresponding websocket) provides 
  - portfolioValue as equity amount in quote value
  - availableMargin as free amount in quote value

#### Holdings: Assets and Positions

- the REST endpoint openpositions (or the corresponding websocket) provides
  - symbol as pair
  - side as short/long indication
  - size as amount of the short or long position

#### Order remarks

- Kraken Futures does not provide an average fill price per order. Therefore this needs to be approximated in the exchange adapter by using the limit price and weighted by the filled amount with each lp_amount / sp_amount change, i.e. per trade.

### Kraken Spot

#### Equity and Free Amount

- the REST endpoint TradeBalance (or corresponding websocket) provides
  - equity in USD as result.e 
  - the free quote (USD) value as result.mf



#### Holdings: Assets and Positions

- the REST endpoint BalanceEx (or the corresponding websocket) provides
  - a result list of assets with the asset symbol for each asset and the following info in its substructure
    - balance as the overall asset amount, which still needs to be multiplied to have it in quote value
    - to get the free asset amount the following formula of elements from the substructure applies: 
      - free = balance - hold_trade − hold_funding − withheld − staked − collateral
      - hold_trade, hold_funding, withheld, staked, collateral are optional elements and may be not present in the substructure
- the REST endpoint OpenPositions (or the corresponding websocket) provides a result list with position IDs each containing a substructure
  - pair identifying the trading pair
  - vol as the amount of the position in base units

#### Order remarks

- OpenOrders does provide a result list of open orders each order identified by their order id (txid) and a substructure with
  - price as average fill price but for closed orders this needs to be requested via QueryOrders
  - vol_exec provides the filled info
  - status
  - vol as amount in base units
  - limitprice as set limit

### BybitSim

BybitSim shall implement a simplified high performance exchange simulation because it will be used to test with large amounts of cached OHLCV data.

- XchCache shall provide a named parameter simbudget that defines an initial simulation quote amount and is only used by simulations.
- BybitSim shall 
  - maintain a simulation budget for the quote and each traded base and considers the following amounts: free, locked
  - shall not use leverages, i.e. leverage == 1 assumed
  - a submitted order allocates free amount to locked, which applies equally for quote and base amounts
  - a filled order reduces the corresponding locked amount and allocates a corresponding free amounts of the executed side
    - an order to open a position reduces the locked quote amount and creates corresponding free base amount
    - an order to close a position reduces the locked base amount and creates corresponding free quote amount
  - an order cancellation assigns the not yet filled locked amount from locked to free
  - an order amendment that changes the amount
    - results in a reject of the amendment if the amount is lower than the already filled amount
    - releases a reduced and not yet filled amount from locked to free
    - moves the additional amount from free to locked in case of an increased order amount  

#### BybitSim tradesim>>tradeloop calls

The tradeloop does not call BybitSim directly; it goes through Xch, which then dispatches into the Bybit adapter when the exchange is BybitSim.

Setup before the loop:

tradesim.jl:321 creates Bybit.BybitCache(), which in test mode runs Bybit.jl:179.
The script then seeds cash with Bybit.jl:222.
Per tick, the main path is:

Trade.jl:780 calls Xch.balancessnapshot, then Xch.account_status, then XchCore.jl:788.
In BybitSim, Bybit.jl:1742 is the key adapter entry. It processes pending simulation orders via Bybit.jl:1258, which in turn uses Bybit.jl:839 and Bybit.jl:611.
Xch.account_status then routes to Bybit.accountcapacity for the BybitSim capacity view, and Bybit.positionsnapshot is the holdings-style view if requested by Xch helpers.
When the strategy emits an order request, the BybitSim order path is:

Xch.process_order_request calls Bybit.jl:1494, Bybit.jl:1517, Bybit.jl:1684, or the lower-level Bybit.jl:1335, Bybit.jl:1570, and Bybit.jl:1050.
Those mutate simulation state through Bybit.jl:1149, Bybit.jl:1173, and Bybit.jl:1209.
So the practical BybitSim function set from a tradesim loop is:

- BybitCache / _init_simulation!
- seedportfolio!
- balances
- accountcapacity
- positionsnapshot
- get24h
- _simprocesspendingorders!, _sim_klines, _sim_lastprice
- createorder, amendorder, cancelorder, upsertopenorder!, upsertcloseorder!, directsequence!
- _simreserveorder!, _simreleaseorder!, _simapplypendingfill!

## Copilot assessment

### Scope and baseline

This assessment is based on the current Trades v1 contract and ownership note:
- `TSM/src/TSM.jl`
- `docs/trades-dataframe-column-ownership-2026-06-27.md`

### Inconsistencies, gaps, and unclarities

- No critical contract inconsistency is currently open in this note.
- Endpoint naming in prose is intentionally high level and may differ from exact adapter payload field names; this is understood and acceptable for this architecture document.

### What is already available

- Trades v1 contract is explicit and promoted in code, including `set` and `rangeid` as TSM-owned columns.
- Lane naming is explicit and consistent with the contract (`lo/lc/so/sc` for active lanes and `lol/lcl/sol/scl` for last-lane execution fields).
- Status vocabulary is canonicalized to `none`, `submitted`, `closed`, `cancelled`, `rejected`.
- Last-lane execution tracking is available with explicit columns: `lol_filled`, `lcl_filled`, `sol_filled`, `scl_filled`, `lol_pavg`, `lcl_pavg`, `sol_pavg`, `scl_pavg`.
- Account snapshot columns are present in v1 and can be used for optimization decisions (`equity`, `freemargin`, `freequote`).
- The ownership table now mirrors the function docstrings and the promoted v1 contract.
- Kraken Futures pavg fallback policy is explicit in the Order remarks section: trade-based averaging per `lp_amount` / `sp_amount` change.

### Required changes

- Documentation hardening:
  - Keep this note architecture-level and treat `docs/trades-dataframe-column-ownership-2026-06-27.md` as authoritative for exact column semantics.
- Code alignment checks:
  - Ensure all adapters normalize incoming status strings to canonical `cancelled` before writing Trades rows.
  - Preserve your optimization intent: avoid extra exchange calls for non-contract fields unless a strategy explicitly requires them.

### Open decisions

- Optional optimization policy:
  - Recommended default: keep `maintmargin` and `balance` outside Trades v1 and use adapter/account APIs directly only when a strategy explicitly requires them.

## Symbol normalization

Symbol normalization for trading pairs shall be done by base-quote, e.g. BTC-USDT .
That enables easy separation of base and quote while holding it in a single String that can be efficiently stored as categorical vector.

Prior to that normalization, there are Kraken-specific normalizations required to remove prefixes and suffixes and map XBT to/from BTC and XDG to/from DOGE, which is described by the Julia code below.

### Kraken symbol normalization

```julia
module KrakenSymbolNormalization

export normalize_symbol, normalize_asset, normalize_pair

# ---------------------------------------------------------------------
# 1. Asset-level normalization
# ---------------------------------------------------------------------

# Kraken-specific asset mappings
const ASSET_MAP = Dict(
    # Major differences
    "BTC" => "XBT",
    "XBT" => "XBT",
    "DOGE" => "XDG",
    "XDG" => "XDG",

    # Standard crypto assets
    "ETH" => "ETH",
    "LTC" => "LTC",
    "XRP" => "XRP",
    "ADA" => "ADA",
    "DOT" => "DOT",
    "SOL" => "SOL",
    "FIL" => "FIL",
    "UNI" => "UNI",
    "KSM" => "KSM",
    "TRX" => "TRX",
    "ETC" => "ETC",
    "EOS" => "EOS",
    "XLM" => "XLM",
    "XTZ" => "XTZ",

    # Fiat
    "USD" => "USD",
    "EUR" => "EUR",
    "GBP" => "GBP",
    "CAD" => "CAD",
    "JPY" => "JPY",
    "CHF" => "CHF"
)

"""
    normalize_asset(asset::String) -> String

Normalize a Kraken Spot/Futures/external asset symbol.
Handles:
- BTC ↔ XBT
- DOGE ↔ XDG
- Kraken internal prefixes (X, Z)
"""
function normalize_asset(asset::String)
    a = uppercase(asset)

    # Strip Kraken internal prefixes (XXBT, XETH, ZUSD, etc.)
    a = replace(a, r"^[XZ]" => "")

    # Apply Kraken-specific mapping
    return get(ASSET_MAP, a, a)
end

# ---------------------------------------------------------------------
# 2. Pair normalization (Spot-style BASE-QUOTE)
# ---------------------------------------------------------------------

"""
    normalize_pair(pair::String) -> String

Normalize Spot-style pairs:
- "XXBTZUSD" → "XBT-USD"
- "XETHZUSD" → "ETH-USD"
- "BTCUSD" → "XBT-USD"
- "BTC-USD" → "XBT-USD"
"""
function normalize_pair(pair::String)
    p = replace(pair, "-" => "")
    p = uppercase(p)

    # Kraken internal pairs are always 6 characters: XXBTZUSD → XXBT ZUSD
    if length(p) == 6
        base = normalize_asset(p[1:3])
        quote = normalize_asset(p[4:6])
        return "$base-$quote"
    end

    # Generic fallback: split into base-quote by heuristics
    # Try 3+3 split
    if length(p) == 6
        base = normalize_asset(p[1:3])
        quote = normalize_asset(p[4:6])
        return "$base-$quote"
    end

    # Try slash format
    if occursin("-", pair)
        base, quote = split(pair, "-")
        return "$(normalize_asset(base))-$(normalize_asset(quote))"
    end

    error("Cannot normalize pair: $pair")
end

# ---------------------------------------------------------------------
# 3. Futures normalization (PI_XBTUSD, FI_ETHUSD_240628)
# ---------------------------------------------------------------------

"""
    normalize_symbol(sym::String) -> String

Normalize ANY Kraken symbol:
- Spot internal: "XXBTZUSD"
- Spot external: "BTC-USD"
- Futures: "PI_XBTUSD", "FI_ETHUSD_240628"
- Underlying: "XBTUSD"
"""
function normalize_symbol(sym::String)
    s = uppercase(sym)

    # Futures: PI_XBTUSD or FI_ETHUSD_240628
    if startswith(s, "PI_") || startswith(s, "FI_")
        parts = split(s, "_")
        underlying = parts[2]  # XBTUSD or ETHUSD
        base = normalize_asset(underlying[1:3])
        quote = normalize_asset(underlying[4:end])
        return "$base-$quote"
    end

    # Underlying format: XBTUSD
    if length(s) == 6
        base = normalize_asset(s[1:3])
        quote = normalize_asset(s[4:6])
        return "$base-$quote"
    end

    # Spot pair normalization
    return normalize_pair(s)
end

end # module

```markdown


