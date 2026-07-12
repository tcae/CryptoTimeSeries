# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module Xch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging, InlineStrings, UUIDs
using CategoricalArrays: CategoricalVector
using Bybit, EnvConfig, KrakenFutures, KrakenSpot, Ohlcv, Targets
using XchAdapter: XchAdapterCache
import XchAdapter: rawcache, exchangeid, symbolinfo, validsymbol, getklines, get24h, balances, emptyorders, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, marginlimits, marginpermitted, marketdataheartbeats, marketdataheartbeat, wsorderssnapshot, wsordersheartbeat, wsbalancessnapshot, wsbalancesheartbeat, ws_orders, ws_balances, accountcapacity, closeorder, upsertcloseorder!, upsertopenorder!, directsequence!
import XchAdapter: normalize_order_status
import Ohlcv: intervalperiod

const authorization = Ref{Any}(nothing)

Authentication(name::Union{Nothing, AbstractString}=nothing; exchange::Union{Nothing, AbstractString}=nothing) = EnvConfig.Authentication(name; exchange=exchange)

function setauthorization!(name::Union{Nothing, AbstractString}=nothing; exchange::Union{Nothing, AbstractString}=nothing)
    authorization[] = Authentication(name; exchange=exchange)
    return authorization[]
end

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

@enum Sidefactor buy=1 sell=-1 invaid = 0

const EXCHANGE_BYBIT::String = "Bybit"
const EXCHANGE_BYBITSIM::String = "BybitSim"
const EXCHANGE_KRAKENFUTURES::String = "KrakenFutures"
const EXCHANGE_KRAKENSPOT::String = "KrakenSpot"

"Return the default quote coin for one canonical exchange."
function _defaultquote(exchange::AbstractString)::String
    ex = String(exchange)
    if ex == EXCHANGE_KRAKENFUTURES
        return "USD"
    elseif ex == EXCHANGE_KRAKENSPOT
        return "USDC"
    end
    return "USDT"
end

"""
Emit one final private-call diagnostics summary for active Kraken adapters.
Safe to call during shutdown; exchanges without private-call counters are skipped.
"""
function log_private_call_summary!(xc)
    exchanges = Set{String}()
    push!(exchanges, exchange(xc))
    for ex in exchanges
        if ex == EXCHANGE_KRAKENSPOT
            KrakenSpot.log_private_call_summary!()
        elseif ex == EXCHANGE_KRAKENFUTURES
            KrakenFutures.log_private_call_summary!()
        end
    end
    return nothing
end

mutable struct XchCache
    bases  # ::Dict{String, Ohlcv.OhlcvData}
    pairstates::Dict{String, DataFrame}  # keyed by canonical trading pair, stores phase-2 Trades DataFrames
    bc::XchAdapterCache  # typed adapter cache wrapper
    startdt::Dates.DateTime
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    mc::Dict # MC = module constants
    tradesrowtemplate::DataFrame  # single-row default template used when appending new Trades rows
    function XchCache(bc::XchAdapterCache; startdt::DateTime=Dates.now(UTC), enddt=nothing)
        startdt = floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        exchange = exchangeid(bc)
        xc = new(Dict(), Dict{String, DataFrame}(), bc, startdt, nothing, enddt, Dict(), DataFrame())
        xc.tradesrowtemplate = _buildtradesrowtemplate(xc)
        syminfodf = if hasproperty(rawcache(xc.bc), :syminfodf)
            getproperty(rawcache(xc.bc), :syminfodf)
        else
            nothing
        end
        if !isnothing(syminfodf)
            for row in eachrow(syminfodf)
                setsymbolinfocache!(xc, row.symbol, (
                    symbol=String(row.symbol),
                    status=String(row.status),
                    basecoin=String(row.basecoin),
                    quotecoin=String(row.quotecoin),
                    ticksize=(row.ticksize),
                    baseprecision=(row.baseprecision),
                    quoteprecision=(row.quoteprecision),
                    minbaseqty=(row.minbaseqty),
                    minquoteqty=(row.minquoteqty),
                ))
            end
        end
        return xc
    end
end

function _adaptercache(exchange::AbstractString)::XchAdapterCache
    if exchange == EXCHANGE_BYBITSIM
        return Bybit.BybitSimCache()
    elseif exchange == EXCHANGE_BYBIT
        return Bybit.BybitCache()
    elseif exchange == EXCHANGE_KRAKENSPOT
        return KrakenSpot.KrakenSpotCache()
    elseif exchange == EXCHANGE_KRAKENFUTURES
        return KrakenFutures.KrakenFuturesCache()
    end
    throw(ArgumentError("unsupported exchange=$(exchange), expected one of $(EXCHANGE_BYBIT), $(EXCHANGE_BYBITSIM), $(EXCHANGE_KRAKENSPOT), $(EXCHANGE_KRAKENFUTURES)"))
end

function XchCache(;startdt::DateTime=Dates.now(UTC), enddt=nothing, exchange::AbstractString=EXCHANGE_KRAKENSPOT)::XchCache
    return XchCache(_adaptercache(exchange); startdt=startdt, enddt=enddt)
end

exchange(xc::XchCache)::String = exchangeid(xc.bc)

"""
    tradingpairkey(base, quotecoin)

Return the canonical in-memory key for one trading pair state table.
Phase 2 stores Trades DataFrames by uppercase concatenated base and quote.
"""
function tradingpairkey(base::AbstractString, quotecoin::AbstractString)::String
    return uppercase(String(base)) * uppercase(String(quotecoin))
end

"Log a trading issue and return the normalized message text for direct storage in Trades columns."
function log_trading_issue(xc::XchCache, issuer::AbstractString, message::AbstractString)::String
    issuerstr = String(issuer)
    messagestr = String(message)
    @warn "Xch.$(ttstr(xc)) $(issuerstr): $(messagestr)"
    return _normalized_order_msg(messagestr)
end

log_trading_issue(issuer::AbstractString, message::AbstractString) = error("log_trading_issue requires an XchCache; call log_trading_issue(xc, issuer, message)")

"""
    hastrades(xc, pair)

Return `true` when a Phase 2 Trades DataFrame is already stored for `pair`.
`pair` can be a concatenated symbol like `"BTCUSDT"`.
"""
function hastrades(xc::XchCache, pair::AbstractString)::Bool
    return haskey(xc.pairstates, uppercase(String(pair)))
end

"""
    hastrades(xc, base, quotecoin)

Return `true` when a Phase 2 Trades DataFrame is already stored for `(base, quotecoin)`.
"""
function hastrades(xc::XchCache, base::AbstractString, quotecoin::AbstractString)::Bool
    return hastrades(xc, tradingpairkey(base, quotecoin))
end

const NO_ORDER_ID = "none"
const NO_ORDER_MSG = "none"

@inline _normalized_order_msg(v)::String = begin
    s = ismissing(v) ? "" : strip(String(v))
    return (isempty(s) || lowercase(s) == "none") ? NO_ORDER_MSG : s
end

"""
    trades(xc, pair)

Return the stored Phase 2 Trades DataFrame for `pair`, creating an empty one when missing.
The returned dataframe is the cache-owned object so callers can mutate it in place.
"""
function trades(xc::XchCache, pair::AbstractString)::DataFrame
    key = uppercase(String(pair))
    return get!(xc.pairstates, key) do
        _applytradescontributors!(xc, _emptytradesv1df())
    end
end

"""
    trades(xc, base, quotecoin)

Return the stored Phase 2 Trades DataFrame for one `(base, quotecoin)` pair.
"""
function trades(xc::XchCache, base::AbstractString, quotecoin::AbstractString)::DataFrame
    return trades(xc, tradingpairkey(base, quotecoin))
end

"""
    settrades!(xc, pair, df)

Store `df` as the Phase 2 Trades DataFrame for `pair` and return the cache.
"""
function settrades!(xc::XchCache, pair::AbstractString, df::AbstractDataFrame)
    _applytradescontributors!(xc, normalized)
    pairkey = uppercase(String(pair))
    basekey = try
        bq = basequote(pairkey)
        isnothing(bq) ? nothing : uppercase(String(bq.basecoin))
    catch
        nothing
    end
    _ensuretradesidentity!(normalized, pairkey; basekey=basekey)
    xc.pairstates[pairkey] = normalized
    return xc
end

"""
    settrades!(xc, base, quotecoin, df)

Store `df` as the Phase 2 Trades DataFrame for one `(base, quotecoin)` pair.
"""
function settrades!(xc::XchCache, base::AbstractString, quotecoin::AbstractString, df::AbstractDataFrame)
    pairkey = tradingpairkey(base, quotecoin)
    basekey = uppercase(String(base))
    _applytradescontributors!(xc, df)
    _ensuretradesidentity!(df, pairkey; basekey=basekey)
    xc.pairstates[pairkey] = df
    return xc
end

"""Ensure per-row Trades identity metadata (`pair`) is populated."""
function _ensuretradesidentity!(df::DataFrame, pairkey::AbstractString; basekey::Union{Nothing, AbstractString}=nothing)::DataFrame
    pkey = uppercase(String(pairkey))

    if :pair ∉ propertynames(df)
        df[!, :pair] = fill(pkey, nrow(df))
    else
        df[!, :pair] = [
            (ismissing(v) || isempty(strip(String(v))) || (uppercase(strip(String(v))) == "NONE")) ? pkey : String(v)
            for v in df[!, :pair]
        ]
    end

    return df
end

"""
    ensuretradesrow!(xc, base, quotecoin, opentime)

Return a writable `(tradesdf, rowix)` for one sample row, creating the row when
missing and materializing Xch-owned identity metadata.
"""
function ensuretradesrow!(xc::XchCache, base::AbstractString, quotecoin::AbstractString, opentime::DateTime)
    basekey = uppercase(String(base))
    pairkey = tradingpairkey(basekey, quotecoin)
    tdf = trades(xc, pairkey)

    rowix = nothing
    n = nrow(tdf)
    if n > 0
        last_open = tdf[n, :opentime]
        if last_open == opentime
            rowix = n
        elseif last_open < opentime
            rowix = _appendtradesrow!(xc, tdf, pairkey, opentime)
        end
    end

    if isnothing(rowix)
        rowix = findlast(==(opentime), tdf[!, :opentime])
    end
    if isnothing(rowix)
        rowix = _appendtradesrow!(xc, tdf, pairkey, opentime)
    end

    tdf[rowix, :opentime] = opentime
    tdf[rowix, :pair] = pairkey
    return (tradesdf=tdf, rowix=Int(rowix))
end

"""
    tradingpairs(xc)

Return stored Phase 2 trading-pair keys in sorted order.
"""
function tradingpairs(xc::XchCache)::Vector{String}
    return sort!(collect(keys(xc.pairstates)))
end

"Store one canonical websocket marketdata heartbeat timestamp in `xc.mc`."
function setmarketdataheartbeat!(xc::XchCache, dt::DateTime)
    xc.mc[:marketdata_ws_last_update_dt] = dt
    return dt
end

"Store one canonical websocket marketdata heartbeat timestamp for one symbol in `xc.mc`."
function setmarketdataheartbeat!(xc::XchCache, symbol::AbstractString, dt::DateTime)
    key = uppercase(String(symbol))
    if !haskey(xc.mc, :marketdata_ws_last_update_by_symbol)
        xc.mc[:marketdata_ws_last_update_by_symbol] = Dict{String, DateTime}()
    end
    xc.mc[:marketdata_ws_last_update_by_symbol][key] = dt
    localdt = get(xc.mc, :marketdata_ws_last_update_dt, nothing)
    if isnothing(localdt) || (dt > DateTime(localdt))
        xc.mc[:marketdata_ws_last_update_dt] = dt
    end
    return dt
end

"Return canonical per-symbol websocket marketdata heartbeat map, merging latest adapter values when available."
function marketdataheartbeats(xc::XchCache)
    if !haskey(xc.mc, :marketdata_ws_last_update_by_symbol)
        xc.mc[:marketdata_ws_last_update_by_symbol] = Dict{String, DateTime}()
    end
    localmap = xc.mc[:marketdata_ws_last_update_by_symbol]

    moduledict = marketdataheartbeats(xc.bc)
    for (sym, dt) in moduledict
        key = uppercase(String(sym))
        moddt = DateTime(dt)
        prev = get(localmap, key, nothing)
        if isnothing(prev) || (moddt > DateTime(prev))
            localmap[key] = moddt
        end
    end
    return copy(localmap)
end

"Return the latest websocket marketdata heartbeat timestamp from canonical state or routed data adapter when available."
function marketdataheartbeat(xc::XchCache; symbol::Union{Nothing, AbstractString}=nothing)
    if !isnothing(symbol)
        key = uppercase(String(symbol))
        localmap = marketdataheartbeats(xc)
        localdt = get(localmap, key, nothing)
        moduledt = marketdataheartbeat(xc.bc; symbol=key)
        if isnothing(localdt)
            if !isnothing(moduledt)
                setmarketdataheartbeat!(xc, key, DateTime(moduledt))
                return DateTime(moduledt)
            end
            return nothing
        end
        if isnothing(moduledt)
            return localdt
        end
        latest = DateTime(moduledt) > DateTime(localdt) ? DateTime(moduledt) : DateTime(localdt)
        setmarketdataheartbeat!(xc, key, latest)
        return latest
    end

    localdt = get(xc.mc, :marketdata_ws_last_update_dt, nothing)
    moduledt = marketdataheartbeat(xc.bc)

    if isnothing(localdt)
        if !isnothing(moduledt)
            xc.mc[:marketdata_ws_last_update_dt] = DateTime(moduledt)
            return xc.mc[:marketdata_ws_last_update_dt]
        end
        return nothing
    end
    if isnothing(moduledt)
        return localdt
    end
    latest = DateTime(moduledt) > DateTime(localdt) ? DateTime(moduledt) : DateTime(localdt)
    xc.mc[:marketdata_ws_last_update_dt] = latest
    return latest
end

function _wsenabled(xc::XchCache, key::Symbol, default::Bool=false)::Bool
    return Bool(get(xc.mc, key, default))
end


function _drainwschannel!(ch; max_items::Int=256)
    isnothing(ch) && return 0
    drained = 0
    while (drained < max_items) && isready(ch)
        take!(ch)
        drained += 1
    end
    return drained
end

# Stub implementations for removed routing layer WebSocket functions
_ensurewschannel!(xc::XchCache, args...; kwargs...) = nothing
wsdfsnapshot(xc::XchCache, args...; kwargs...) = DataFrame()
wsheartbeat(xc::XchCache, args...; kwargs...) = nothing

function _refreshwsstreams!(xc::XchCache)
    # Stub: WebSocket streams no longer managed via routing layer
    return nothing
end


function _ensurewsorders!(xc::XchCache)
    _ = ws_orders(xc.bc)
    return nothing
end

function _ensurewsbalances!(xc::XchCache)
    _ = ws_balances(xc.bc)
    return nothing
end



"Return latest adapter websocket order snapshot (canonical normalized open-order rows when available)."
function wsordersnapshot(xc::XchCache)::DataFrame
    snapshot = wsorderssnapshot(xc.bc)
    return isnothing(snapshot) ? DataFrame() : DataFrame(snapshot; copycols=true)
end

"Return latest adapter websocket balances snapshot (canonical normalized balance rows when available)."
function wsbalancessnapshot(xc::XchCache)::DataFrame
    snapshot = wsbalancessnapshot(xc.bc)
    return isnothing(snapshot) ? DataFrame() : DataFrame(snapshot; copycols=true)
end

"Return latest adapter websocket order heartbeat timestamp when available."
function wsordersheartbeat(xc::XchCache)
    return wsordersheartbeat(xc.bc)
end

"Return latest adapter websocket balances heartbeat timestamp when available."
function wsbalancesheartbeat(xc::XchCache)
    return wsbalancesheartbeat(xc.bc)
end

"""
Return the adapter cache for the given `role`, using the routing config when available.
Falls back to `xc.bc` (the primary adapter) when no role override is configured.
"""

"Return the exchange module for the given adapter instance."

"Return the syminfo cache dict, creating it lazily."
_syminfocache(xc::XchCache) = get!(xc.mc, :syminfo_cache, Dict{String, NamedTuple}())

"""
    setsymbolinfocache!(xc, symbol, info)

Manually seed the local symbol-info cache entry for `symbol` (e.g. `"BTCUSDT"`).
`info` must be a `NamedTuple` with at least the fields required by simulation:
`minbaseqty`, `minquoteqty`, `ticksize`, `baseprecision`, `quoteprecision`,
`status`, `quotecoin`, `basecoin`.
This is primarily useful for tests and offline simulation where no live exchange
connection is available.
"""
function setsymbolinfocache!(xc::XchCache, symbol::AbstractString, info::NamedTuple)
    _syminfocache(xc)[uppercase(symbol)] = info
    return xc
end

"""
Fetch symbol info from the exchange and cache the result locally.
Falls back to the local cache when no live connection is available (sim mode).
"""
function _exchangesymbolinfo(xc::XchCache, symbol)
    symbol = uppercase(string(symbol))
    bc = rawcache(xc.bc)
    if !isnothing(bc)
        row = symbolinfo(xc.bc, symbol)
        if !isnothing(row)
            # Populate / refresh local cache from live data
            nt = (
                symbol        = symbol,
                status        = string(row.status),
                basecoin      = string(row.basecoin),
                quotecoin     = string(row.quotecoin),
                ticksize      = (row.ticksize),
                baseprecision = (row.baseprecision),
                quoteprecision = (row.quoteprecision),
                minbaseqty    = (row.minbaseqty),
                minquoteqty   = (row.minquoteqty),
            )
            _syminfocache(xc)[symbol] = nt
            return row  # keep returning the original DataFrameRow for backward compat
        end
        return nothing  # symbol not found on exchange
    end
    # No live connection (bybitsim mode) — use cached info
    return get(_syminfocache(xc), symbol, nothing)
end

function _orderfield(orderinfo, field::Symbol)
    if isnothing(orderinfo) || !hasproperty(orderinfo, field)
        return missing
    end
    return getproperty(orderinfo, field)
end


"""
Return the set of order ids that were created as adaptive maker orders with `limitprice=nothing`.
"""
function _adaptiveordercache!(xc::XchCache)
    if !haskey(xc.mc, :adaptive_maker_orders)
        xc.mc[:adaptive_maker_orders] = Set{String}()
    end
    return xc.mc[:adaptive_maker_orders]
end

"""
Register `orderid` as an adaptive maker order.
"""
function registeradaptiveorder!(xc::XchCache, orderid)
    push!(_adaptiveordercache!(xc), String(orderid))
    return xc
end

"""
Remove `orderid` from the adaptive maker order registry.
"""
function unregisteradaptiveorder!(xc::XchCache, orderid)
    delete!(_adaptiveordercache!(xc), String(orderid))
    return xc
end

"""
Return true when `orderid` is tracked as an adaptive maker order.
"""
function isadaptiveorder(xc::XchCache, orderid)::Bool
    return String(orderid) in _adaptiveordercache!(xc)
end

"""
Drop adaptive order ids that are no longer present in `openorderids`.
"""
function pruneadaptiveorders!(xc::XchCache, openorderids)
    active = Set(String.(collect(openorderids)))
    adaptive = _adaptiveordercache!(xc)
    for orderid in collect(adaptive)
        orderid in active || delete!(adaptive, orderid)
    end
    return xc
end


function _orderfieldfirst(orderinfo, fields::Vector{Symbol})
    for field in fields
        value = _orderfield(orderinfo, field)
        if !ismissing(value) && !isnothing(value)
            return value
        end
    end
    return missing
end











"""
Emit cancellation events for orders that were previously observed as open but are
missing from the latest full `getopenorders` response.
"""



"Normalize adapter order-create response into `(orderid, orderinfo)` where possible."
function _normalizecreatedorder(xc::XchCache, created)
    if isnothing(created)
        return (nothing, nothing)
    end
    if created isa AbstractString
        oid = String(created)
        info = getorder(xc, oid; auditevent=false)
        if isnothing(info)
            info = (orderid=oid, status="Unknown", rejectreason="NO ERROR", executedqty=missing, avgprice=missing, timeinforce=missing)
        end
        return (oid, info)
    end
    if hasproperty(created, :orderid)
        return (String(getproperty(created, :orderid)), created)
    end
    return (nothing, created)
end

"Normalize adapter amend response into `(orderid, orderinfo)` where possible."
function _normalizeamendedorder(xc::XchCache, amended)
    if isnothing(amended)
        return (nothing, nothing)
    end
    if amended isa AbstractString
        oid = String(amended)
        return (oid, getorder(xc, oid; auditevent=false))
    end
    if hasproperty(amended, :orderid)
        return (String(getproperty(amended, :orderid)), amended)
    end
    return (nothing, amended)
end

"""
Create a close order for one existing position side.

- `positionside=:long` closes long exposure via a Sell order.
- `positionside=:short` closes short exposure via a Buy order.

Adapters may specialize this by implementing `closeorder(bc, symbol, positionside, basequantity, limitprice, maker; reduceonly=...)`.
If no adapter specialization exists, this function falls back to existing `createbuyorder`/`createsellorder` behavior.
"""
function closeorder(xc::XchCache, base::AbstractString; positionside::Symbol, limitprice, basequantity, maker::Bool=true, reduceonly::Bool=true, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    side = Symbol(lowercase(String(positionside)))
    @assert side in (:long, :short) "closeorder positionside=$(positionside) must be :long or :short"

    baseup = uppercase(String(base))
    symbol = symboltoken(xc, baseup, EnvConfig.pairquote)
    created = closeorder(xc.bc, symbol, side, basequantity, limitprice, maker; reduceonly=reduceonly)
    if !isnothing(created)
        oid, oocreate = _normalizecreatedorder(xc, created)
        orderside = side == :long ? "Sell" : "Buy"
        if isnothing(limitprice) && maker && !isnothing(oid)
            registeradaptiveorder!(xc, oid)
        end
        return oid
    end

    if side == :long
        return createsellorder(xc, baseup; limitprice=limitprice, basequantity=basequantity, maker=maker, reduceonly=reduceonly, parent_order_id=parent_order_id, leg_group_id=leg_group_id, leg_label=leg_label)
    end
    return createbuyorder(xc, baseup; limitprice=limitprice, basequantity=basequantity, maker=maker, reduceonly=reduceonly, parent_order_id=parent_order_id, leg_group_id=leg_group_id, leg_label=leg_label)
end

setstartdt(xc::XchCache, dt::DateTime) = (xc.startdt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
setenddt(xc::XchCache, dt::DateTime) = (xc.enddt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
bases(xc::XchCache) = keys(xc.bases)
ohlcv(xc::XchCache) = values(xc.bases)
ohlcv(xc::XchCache, base::AbstractString) = xc.bases[base]
baseohlcvdict(xc::XchCache) = xc.bases

"Return the OhlcvData for `base`. Alias for `ohlcv(xc, base)`."
getohlcv(xc::XchCache, base::AbstractString) = ohlcv(xc, base)

"Return the current close price for an OhlcvData at its current index."
currentprice(o::Ohlcv.OhlcvData) = Ohlcv.dataframe(o)[o.ix, :close]

basenottradable = ["MATIC", "FTM", "KFEE"]  # KFEE = Kraken proprietary fee credit, never tradeable
basestablecoin = ["USD", "USD1", "USDT", "TUSD", "BUSD", "USDC", "USDE", "EUR", "DAI"]
quotecoins = ["USDT"]  # , "USDC"]
baseignore = uppercase.(union(basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT

MAXLIMITDELTA = 0.1

_isleveraged(token) = !isnothing(token) && (length(token) > 2) && (token[end] in ['S', 'L']) && isdigit(token[end-1])

#region support

validbase(xc::XchCache, base::AbstractString) =
    (uppercase(base) != uppercase(EnvConfig.pairquote)) &&
    validsymbol(xc, symboltoken(xc, base, EnvConfig.pairquote))

removebase!(xc::XchCache, base) = delete!(xc.bases, base)
removeallbases(xc::XchCache) = xc.bases = Dict()

function addbase!(xc::XchCache, ohlcv::Ohlcv.OhlcvData)
    xc.bases[ohlcv.base] = ohlcv
    setcurrenttime!(xc, ohlcv.base, isnothing(xc.currentdt) ? xc.startdt : xc.currentdt)
end

function addbase!(xc::XchCache, base, startdt, enddt)
    base = String(base)
    enddt = isnothing(enddt) ? floor(Dates.now(UTC), Minute(1)) : floor(enddt, Minute(1))
    startdt = isnothing(startdt) ? enddt : floor(startdt, Minute(1))
    ohlcv = cryptodownload(xc, base, "1m", startdt, enddt)
    ohlcv.ix = firstindex(ohlcv.df, 1)
    xc.bases[base] = ohlcv
    setcurrenttime!(xc, base, startdt)
end

function addbases!(xc::XchCache, bases, startdt, enddt)
    for base in bases
        addbase!(xc, base, startdt, enddt)
    end
end

assetbases(xc::XchCache) = filter(!=(uppercase(EnvConfig.pairquote)), uppercase.(Xch.balances(xc)[!, :coin]))

symboltoken(basecoin, quotecoin=EnvConfig.pairquote) = isnothing(basecoin) ? nothing : uppercase(basecoin * quotecoin)

"""
Resolve the exchange-specific symbol token for a pair on the primary exchange.
Falls back to a concatenated symbol if the adapter cannot map the pair yet.
"""
function symboltoken(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.pairquote)
    bc = rawcache(xc.bc)
    if isnothing(bc)
        return symboltoken(basecoin, quotecoin)
    end
    return symboltoken(xc.bc, basecoin, quotecoin)
end

"Return side-specific margin leverage caps for one symbol when supported by the primary exchange."
function marginlimits(xc::XchCache, symbol::AbstractString)
    bc = rawcache(xc.bc)
    isnothing(bc) && return (maxleveragebuy=0, maxleveragesell=0)
    return marginlimits(xc.bc, symbol)
end

"Return true when primary exchange metadata permits side/leverage for one symbol."
function marginpermitted(xc::XchCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed)::Bool
    marginleverage <= 0 && return true
    bc = rawcache(xc.bc)
    isnothing(bc) && return false
    return marginpermitted(xc.bc, symbol, orderside, marginleverage)
end

ceilbase(base, qty) = base == "usdt" ? ceil(qty, digits=3) : ceil(qty, digits=5)
floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.pairquote)) &&
    !(uppercase(symbol[1:end-length(EnvConfig.pairquote)]) in baseignore)

"Returns pair of basecoin and quotecoin if quotecoin in `quotecoins` or equals `EnvConfig.pairquote` else `nothing` is returned"
function basequote(symbol)
    symbol = uppercase(symbol)
    candidates = union(quotecoins, [uppercase(EnvConfig.pairquote)])
    range = nothing
    for qc in candidates
        range = findfirst(qc, symbol)
        if !isnothing(range)
            break
        end
    end
    return isnothing(range) ? nothing : (basecoin = symbol[begin:range[1]-1], quotecoin = symbol[range])
end

"""
Return minimum quantities for a `(basecoin, quotecoin)` pair.
"""
function minimumqty(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString)
    return minimumqty(xc, symboltoken(xc, basecoin, quotecoin))
end

"""
Return precision information for a `(basecoin, quotecoin)` pair.
"""
function precision(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString)
    return precision(xc, symboltoken(xc, basecoin, quotecoin))
end

#endregion support

#region time

"""
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(xc::XchCache, startdt, enddt)
    for ohlcv in Xch.ohlcv(xc)
        (verbosity >= 3) && println("before Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
        Ohlcv.timerangecut!(ohlcv, startdt, enddt)
        (verbosity >= 3) && println("after Ohlcv.timerangecut!($ohlcv, $startdt, $enddt)")
    end
end

function Base.iterate(xc::XchCache, currentdt=nothing)
    currentdt = isnothing(currentdt) ? xc.startdt : currentdt + Minute(1)
    _sleepuntil(xc, currentdt)

    (verbosity >= 3) && println("iterate: startdt=$(xc.startdt), currentdt=$(xc.currentdt), enddt=$(xc.enddt) local currentdt=$currentdt")
    # println("\rcurrentdt=$(string(currentdt)) xc.enddt=$(string(xc.enddt)) ")
    if !isnothing(xc.enddt) && (currentdt > xc.enddt)
        xc.currentdt = nothing
        return nothing
    else
        Xch.setcurrenttime!(xc, currentdt)  # also updates bases if current time is > last time of xc
    end
    (verbosity >= 3) && println("iterate: utcnow=$(Dates.now(UTC)) startdt=$(xc.startdt), currentdt=$(xc.currentdt), enddt=$(xc.enddt)")
    return xc, currentdt
end

timesimulation(xc::XchCache)::Bool = !isnothing(xc.currentdt) && !isnothing(xc.enddt)
tradetime(xc::XchCache) = isnothing(xc.currentdt) ? floor(servertime(xc.bc), Minute(1)) : xc.currentdt
# tradetime(xc::XchCache) = (xc.mc[:simmode] != bybitsim) ? servertime(xc.bc) : Dates.now(UTC)
ttstr(dt::DateTime) = "LT" * EnvConfig.now() * "/TT" * Dates.format(dt, EnvConfig.datetimeformat)
ttstr(xc::XchCache) = ttstr(tradetime(xc))

"""
Return exchange server time and keep retrying every 60 seconds on connectivity/API failures.

Used by the live loop so transient or prolonged exchange/network outages do not terminate
the session. Backtest paths are unaffected because they do not call this helper.
"""
function _servertime_retry_1m(xc::XchCache)::DateTime
	while true
		try
            return servertime(xc.bc)
		catch err
			(verbosity >= 1) && @warn "exchange server time unavailable; retrying in 60 seconds" retry_seconds=60 exception=sprint(showerror, err)
			sleep(60)
		end
	end
end

function _sleepuntil(xc::XchCache, dt::DateTime)
    if timesimulation(xc)
        return
    end
    sleepperiod = (dt + Second(2)) - _servertime_retry_1m(xc)
    if sleepperiod <= Dates.Second(0)
        return
    end
    if sleepperiod > Minute(1)
        (verbosity >= 2) && println("TT=$(tradetime(xc)) waiting until $dt resulting in long sleep $(floor(sleepperiod, Minute))")
    end
    # println("sleeping $(floor(sleepperiod, Second))")
    sleep(sleepperiod)
end

function _upsert_closed_wscandle!(ohlcv, candle)
    isnothing(candle) && return nothing
    df = Ohlcv.dataframe(ohlcv)
    cdt = floor(DateTime(candle.opentime), Minute(1))
    copen = (candle.open)
    chigh = (candle.high)
    clow = (candle.low)
    cclose = (candle.close)
    cvol = (candle.basevolume)

    rowix = size(df, 1) == 0 ? nothing : findfirst(==(cdt), df[!, :opentime])
    if isnothing(rowix)
        if :pivot in names(df)
            push!(df, (opentime=cdt, open=copen, high=chigh, low=clow, close=cclose, basevolume=cvol, pivot=cclose); promote=true)
        else
            push!(df, (opentime=cdt, open=copen, high=chigh, low=clow, close=cclose, basevolume=cvol); promote=true)
        end
        sort!(df, :opentime)
    else
        df[rowix, :opentime] = cdt
        df[rowix, :open] = copen
        df[rowix, :high] = chigh
        df[rowix, :low] = clow
        df[rowix, :close] = cclose
        df[rowix, :basevolume] = cvol
        (:pivot in names(df)) && (df[rowix, :pivot] = cclose)
    end
    Ohlcv.setdataframe!(ohlcv, df)
    return nothing
end

"Sleeps until `datetime` if reached if `datetime` is in the future, set the *current* time and updates ohlcv if required"
function setcurrenttime!(xc::XchCache, base, datetime::DateTime)
    dt = floor(datetime, Minute(1))
    ot = []
    if base in keys(xc.bases)
        ohlcv = xc.bases[base]
        ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
        if (length(ot) == 0) || (dt > ot[end])
            xc.bases[base] = cryptoupdate!(xc, ohlcv, (length(ot) == 0 ? dt : ot[begin]), dt)
        end
    else
        xc.bases[base] = ohlcv = cryptodownload(xc, base, "1m", dt, dt)
        ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    end
    Ohlcv.setix!(ohlcv, Ohlcv.rowix(ohlcv, dt))
    if (length(ot) > 0) && (ot[begin] <= dt <= ot[end]) && (ot[Ohlcv.ix(ohlcv)] != dt)
        if (verbosity >= 1) && (EnvConfig.configmode == production)
            @warn "setcurrenttime!($base, $dt) failed, opentime[ix]=$(Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)])"
        end
    end
    return ohlcv
end

"Set xc.currentdt and all cached base ohlcv.ix to the provided datetime. If isnothing(datetime) the only xc.currentdt is set to nothing"
function setcurrenttime!(xc::XchCache, datetime::Union{DateTime, Nothing})
    function _setsimtime!(bc, dt)
        if !isnothing(bc) && hasproperty(bc, :simtime)
            setproperty!(bc, :simtime, dt)
        end
        return nothing
    end

    xc.currentdt = datetime
    _setsimtime!(rawcache(xc.bc), datetime)
    if !isnothing(datetime)
        for base in keys(xc.bases)
            try
                setcurrenttime!(xc, base, datetime)
            catch err
                (verbosity >= 2) && @warn "setcurrenttime!($base, $datetime) failed; skipping base" exception=sprint(showerror, err)
                removebase!(xc, base)
            end
        end
    end
end

#endregion time

#region klines

"""
Requests base/USDT from start until end (both including) in interval frequency but will return a maximum of 1000 entries.
Subsequent calls are required to get > 1000 entries.
Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function _ohlcfromexchange(xc::XchCache, base::AbstractString, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", quotecoin=EnvConfig.pairquote)
    symbol = uppercase(base*quotecoin)
    df = getklines(xc.bc, symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
    Ohlcv.addpivot!(df)
    return df
end

"""
Requests base/USDT from start until end (both including) in interval frequency. If required Bybit is internally called several times to fill the request.

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

time gaps will not be filled
"""
function _gethistoryohlcv(xc::XchCache, base::AbstractString, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    fetches = 0
    # println("requesting from $startdt until $enddt $(ceil(enddt - startdt, intervalperiod(interval)) + intervalperiod(interval)) $base OHLCV from binance")

    notreachedstartdt = true
    df = Ohlcv.defaultohlcvdataframe()
    lastdt = enddt + Dates.Minute(1)  # make sure lastdt break condition is not true
    (verbosity >= 3) && @info "request from $startdt until $enddt at entry"
    while notreachedstartdt
        # fills from newest to oldest using Bybit
        fetches =+ 1
        if startdt > enddt
            (verbosity >= 3) && @warn "fetch $fetches: startdt $startdt > enddt $enddt at entry - exchanging"
            dt = startdt
            startdt = enddt
            enddt = dt
        end
        res = _ohlcfromexchange(xc, base, startdt, enddt, interval)
        if size(res, 1) == 0
            # will be the case for the timerange before the first coin data is available
            # Logging.@warn "no $base $interval data returned by last ohlcv read from $startdt until $enddt"
            break
        end
        notreachedstartdt = (res[begin, :opentime] > startdt) # Bybit loads newest first
        if res[begin, :opentime] >= lastdt
            # no progress since last ohlcv read - will be the case for all coins that have no cached data because startdt is likely before the first coin data
            (verbosity >= 3) && @warn "fetch $fetches: no progress since last ohlcv read: requested from $startdt until $enddt - received from $(res[begin, :opentime]) until $(res[end, :opentime]), lastdt=$lastdt - returning df from $(df[begin, :opentime]) until $(df[end, :opentime])"
            break
        end
        lastdt = res[begin, :opentime]
        # println("$(Dates.now()) read $(nrow(res)) $base from $enddt backwards until $lastdt")
        enddt = floor(lastdt, intervalperiod(interval))
        while (size(df,1) > 0) && (size(res,1) > 0) && (res[end, :opentime] >= df[begin, :opentime])  # replace last row with updated data
            deleteat!(res, size(res, 1))
        end
        @assert all(names(df) .== names(res)) "names(df)=$(names(df)) .== names(res)=$(names(res))"
        if size(res, 1) > 0
            if size(df, 1) > 0
                df = vcat(res, df)
            else
                df = res
            end
        end
    end
    return df
end

"""
Returns the OHLCV data of the requested time range by first checking the given (`ohlcv` parameter) cache data and if unsuccessful requesting it from the exchange.

- ohlcv containes the requested base identifier and interval - the result will be stored in the data frame of this structure
- startdt and enddt are DateTime stamps that specify the requested time range

"""
function cryptoupdate!(xc::XchCache, ohlcv, startdt, enddt)
    base = ohlcv.base
    interval = ohlcv.interval
    # println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt < startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    olddf = Ohlcv.dataframe(ohlcv)
    if (size(olddf, 1) > 0) && (startdt < olddf[end, :opentime]) && (enddt > olddf[begin, :opentime]) # there is already data available and overlapping
        if (startdt < olddf[begin, :opentime])
            # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
            tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
            # get data of a timerange before the already available data
            newdf = _gethistoryohlcv(xc, base, startdt, tmpdt, interval)
            if size(newdf, 1) > 0
                if names(olddf) == names(newdf)
                    olddf = vcat(newdf, olddf)
                else
                    (verbosity >= 1) && @error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
        if (enddt > olddf[end, :opentime])
            tmpdt = olddf[end, :opentime]  # update last data row
            newdf = _gethistoryohlcv(xc, base, tmpdt, enddt, interval)
            if size(newdf, 1) > 0
                while (size(olddf, 1) > 0) && (newdf[begin, :opentime] <= olddf[end, :opentime])  # replace last row with updated data
                    deleteat!(olddf, size(olddf, 1))
                end
                if names(olddf) == names(newdf)
                    olddf = vcat(olddf, newdf)
                else
                    (verbosity >= 1) && @error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end

    else # size(olddf, 1) == 0
        newdf = _gethistoryohlcv(xc, base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    xc.bases[ohlcv.base] = ohlcv
    return ohlcv
end

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from the Exchange.

    - *base* identifier and interval specify what data is requested - the result will be returned as OhlcvData structure
    - startdt and enddt are DateTime stamps that specify the requested time range
    - any gap to chached data will be closed when asking for missing data from Bybit
"""
function cryptodownload(xc::XchCache, base, interval, startdt, enddt)::OhlcvData
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    if validbase(xc, base)
        if Ohlcv.file(ohlcv).existing
            Ohlcv.read!(ohlcv)
        end
        cryptoupdate!(xc, ohlcv, startdt, enddt)
        ohlcv.ix = firstindex(ohlcv.df, 1)
    else
        (verbosity >= 3) && @warn "base=$base is unknown or invalid"
    end
    return ohlcv
end

"downloads missing data and merges with canned data then saves it as supplemented canned data"
function downloadupdate!(xc::XchCache, bases, enddt, period=Dates.Year(10))
    count = length(bases)
    enddt = floor(enddt, Dates.Minute)
    startdt = floor(enddt - period, Dates.Minute)
    for (ix, base) in enumerate(bases)
        # break
        (verbosity >= 2) && println("\n$(EnvConfig.now()) start updating $base ($ix of $count) request from $startdt until $enddt")
        ohlcv = Xch.cryptodownload(xc, base, "1m", startdt, enddt)
        Ohlcv.write(ohlcv)
    end
end

"Downloads all basecoins with USDT quote that shows a minimumdayquotevolume and saves it as canned data"
function downloadallUSDT(xc::XchCache, enddt, period=Dates.Year(10), minimumdayquotevolume = 10000000)
    df = getUSDTmarket(xc)
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    bases = sort!(setdiff(df[!, :basecoin], baseignore))
    (verbosity >= 2) && println("$(EnvConfig.now())downloading the following bases bases with $(EnvConfig.pairquote) quote: $bases")
    downloadupdate!(xc, bases, enddt, period)
    return df
end

#endregion klines

#region public

function validsymbol(xc::XchCache, symbol)
    sym = _exchangesymbolinfo(xc, symbol)
    if isnothing(sym)
        return false
    end
    exch_valid = validsymbol(xc.bc, sym)
    r = !isnothing(sym) &&
        exch_valid &&
        !(sym.basecoin in baseignore) &&
        !_isleveraged(sym.basecoin)
    return r
end

function validsymbol(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString)
    return validsymbol(xc, symboltoken(xc, basecoin, quotecoin))
end

"Returns a tuple of (minimum base quantity, minimum quote quantity)"
function minimumqty(xc::XchCache, sym::AbstractString)
    syminfo = _exchangesymbolinfo(xc, sym)
    if isnothing(syminfo)
        validsymbol(xc, sym) && (verbosity >= 1) && @error "cannot find symbol $sym in $(exchange(xc)) exchange info"
        return nothing
    end
    return (minbaseqty=syminfo.minbaseqty, minquoteqty=syminfo.minquoteqty)
end

function minimumbasequantity(xc::XchCache, base::AbstractString, price=(base in bases(xc) ? Ohlcv.dataframe(ohlcv(xc, base))[Ohlcv.ix(ohlcv(xc, base)), :close] : nothing))
    if isnothing(price)
        return nothing
    end
    sym = Xch.symboltoken(base)
    syminfo = Xch.minimumqty(xc, sym)
    return isnothing(syminfo) ? nothing : 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
end

function minimumquotequantity(xc::XchCache, base::AbstractString, price=(base in bases(xc) ? Ohlcv.dataframe(ohlcv(xc, base))[Ohlcv.ix(ohlcv(xc, base)), :close] : nothing))
    if isnothing(price)
        return nothing
    end
    sym = Xch.symboltoken(base)
    syminfo = Xch.minimumqty(xc, sym)
    return isnothing(syminfo) ? nothing : 1.01 * max(syminfo.minbaseqty * price, syminfo.minquoteqty) # 1% more to avoid issues by rounding errors
end

function precision(xc::XchCache, sym::AbstractString)
    syminfo = _exchangesymbolinfo(xc, sym)
    if isnothing(syminfo)
        (verbosity >= 1) && @error "cannot find symbol $sym in $(exchange(xc)) exchange info"
        return nothing
    end
    return (baseprecision=syminfo.baseprecision, quoteprecision=syminfo.quoteprecision)
end

_emptymarkets()::DataFrame = DataFrame(basecoin=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])

function _usdtmarkettickers(xc::XchCache; requestedbases=nothing)
    if isnothing(requestedbases)
        return get24h(xc.bc)
    end

    rows = DataFrame(askprice=Float32[], bidprice=Float32[], lastprice=Float32[], quotevolume24h=Float32[], pricechangepercent=Float32[], symbol=String[])
    quotetoken = uppercase(String(EnvConfig.pairquote))
    wanted = unique([uppercase(String(base)) for base in requestedbases if !isnothing(base) && (uppercase(String(base)) != quotetoken)])
    for base in wanted
        symbol = symboltoken(xc, base, quotetoken)
        row = _tickerrow(get24h(xc.bc, symbol))
        isnothing(row) && continue
        push!(rows, row)
    end
    return rows
end

function _tickerrow(data)
    if isnothing(data)
        return nothing
    end
    row = if data isa DataFrames.DataFrameRow
        data
    elseif data isa AbstractDataFrame
        size(data, 1) > 0 ? data[1, :] : nothing
    else
        data
    end
    isnothing(row) && return nothing

    return (
        symbol=String(row.symbol),
        askprice=(row.askprice),
        bidprice=(row.bidprice),
        lastprice=(row.lastprice),
        quotevolume24h=(row.quotevolume24h),
        pricechangepercent=(row.pricechangepercent),
    )
end

"""
Returns a dataframe with 24h values of all USDT quotecoin bases that are not in baseignore list with the following columns:

- basecoin
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

getUSDTmarket: 512×6 DataFrame
 Row │ askprice       bidprice       lastprice      quotevolume24h  pricechangepercent  basecoin
     │ Float32        Float32        Float32        Float32         Float32             String
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │    0.65           0.6499         0.6499           6.51727e6             -0.0536  OP
"""
function getUSDTmarket(xc::XchCache; dt::DateTime=tradetime(xc), requestedbases=nothing)
    usdtdf = _usdtmarkettickers(xc; requestedbases=requestedbases)
    if isnothing(usdtdf) || (size(usdtdf, 1) == 0)
        return _emptymarkets()
    end

    bq = [basequote(s) for s in usdtdf.symbol]  # create vector of pairs (basecoin, quotecoin)
    @assert length(bq) == size(usdtdf, 1)
    
    # Normalize base coins using the exchange adapter's normalization (e.g., XBT→BTC for KrakenSpot/KrakenFutures)
    function normalize_basecoin(xc_inner, basecoin_raw)
        if xc_inner.bc isa KrakenSpot.KrakenSpotCache
            return KrakenSpot._normalizeasset(basecoin_raw)
        elseif xc_inner.bc isa KrakenFutures.KrakenFuturesCache
            return KrakenFutures._normalizeasset(basecoin_raw)
        end
        return basecoin_raw
    end
    
    normalized_bases = [isnothing(bqe) ? missing : normalize_basecoin(xc, bqe.basecoin) for bqe in bq]
    usdtdf[!, :basecoin] = normalized_bases
    nbq = [!ismissing(bc) && validbase(xc, bc) && (bqe.quotecoin == EnvConfig.pairquote) for (bc, bqe) in zip(normalized_bases, bq)]
    usdtdf = usdtdf[nbq, Not(:symbol)]
    return usdtdf
end

"""
Returns the broad USDT market snapshot used for selection/screening logic.
"""
function screeningUSDTmarket(xc::XchCache; dt::DateTime=tradetime(xc))
    setcurrenttime!(xc, dt)
    return getUSDTmarket(xc; dt=dt)
end

"""
Returns a coin-scoped USDT market snapshot used for portfolio valuation.
Only the requested base coins are queried from the exchange adapter.
"""
function valuationUSDTmarket(xc::XchCache, requestedbases; dt::DateTime=tradetime(xc))
    setcurrenttime!(xc, dt)
    return getUSDTmarket(xc; dt=dt, requestedbases=requestedbases)
end

#endregion public

#region account

function _asfloat64(value, default::Float64=0.0)::Float64
    if ismissing(value) || isnothing(value)
        return default
    elseif value isa AbstractFloat
        return (value)
    elseif value isa Real
        return (value)
    elseif value isa AbstractString
        stripped = strip(String(value))
        isempty(stripped) && return default
        parsed = try
            parse(Float64, stripped)
        catch
            default
        end
        return isfinite(parsed) ? parsed : default
    end
    return default
end

function _normalizeaccountcapacity(snapshot)
    return (
        equity_quote=max(0.0, _asfloat64(get(snapshot, :equity_quote, 0.0), 0.0)),
        available_opening_quote=max(0.0, _asfloat64(get(snapshot, :available_opening_quote, 0.0), 0.0)),
        available_long_quote=max(0.0, _asfloat64(get(snapshot, :available_long_quote, get(snapshot, :available_opening_quote, 0.0)), 0.0)),
        available_short_quote=max(0.0, _asfloat64(get(snapshot, :available_short_quote, get(snapshot, :available_opening_quote, 0.0)), 0.0)),
        initial_margin_quote=max(0.0, _asfloat64(get(snapshot, :initial_margin_quote, 0.0), 0.0)),
        maintenance_margin_quote=max(0.0, _asfloat64(get(snapshot, :maintenance_margin_quote, 0.0), 0.0)),
        source=String(get(snapshot, :source, "unknown")),
    )
end

function _fallbackaccountcapacity(xc::XchCache)
    balancesdf = balances(xc; ignoresmallvolume=false)
    assets = portfolio!(xc, balancesdf; ignoresmallvolume=false)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    quotefree = 0.0
    if (:coin in names(assets)) && (:free in names(assets))
        for row in eachrow(assets)
            if uppercase(String(row.coin)) == quotecoin
                quotefree += max(0.0, (row.free))
            end
        end
    end
    equity = (:usdtvalue in names(assets)) ? (sum(assets[!, :usdtvalue])) : quotefree
    return (
        equity_quote=max(0.0, equity),
        available_opening_quote=max(0.0, quotefree),
        available_long_quote=max(0.0, quotefree),
        available_short_quote=max(0.0, quotefree),
        initial_margin_quote=0.0,
        maintenance_margin_quote=0.0,
        source="Xch:portfolio_fallback",
    )
end

"""
Return exchange-concept account capacity snapshot in quote currency.

Fields:
- `equity_quote`: exchange-equity style net worth in quote terms
- `available_opening_quote`: side-agnostic conservative opening capacity
- `available_long_quote`: opening capacity for long/spot buy side
- `available_short_quote`: opening capacity for short/margin sell side
"""
function accountcapacity(xc::XchCache; force_refresh::Bool=false, ttl_seconds::Int=5)
    if !force_refresh && !timesimulation(xc)
        if haskey(xc.mc, :account_capacity_snapshot) && haskey(xc.mc, :account_capacity_snapshot_dt)
            dt = xc.mc[:account_capacity_snapshot_dt]
            if (dt isa DateTime) && ((Dates.now(UTC) - dt) < Dates.Second(max(1, ttl_seconds)))
                return xc.mc[:account_capacity_snapshot]
            end
        end
    end

    snapshot = try
        accountcapacity(xc.bc)
    catch err
        (verbosity >= 1) && @warn "accountcapacity: exchange snapshot failed, using fallback" exchange=exchange(xc) error=sprint(showerror, err)
        nothing
    end
    if isnothing(snapshot)
        snapshot = _fallbackaccountcapacity(xc)
    end
    normalized = _normalizeaccountcapacity(snapshot)
    xc.mc[:account_capacity_snapshot] = normalized
    xc.mc[:account_capacity_snapshot_dt] = Dates.now(UTC)
    return normalized
end

"Return the current account snapshot used by Trade loop orchestration."
function account_status(xc::XchCache; force_refresh::Bool=false, ttl_seconds::Int=5)
    balancesdf = balances(xc; ignoresmallvolume=false)
    assetsdf = portfolio!(xc, balancesdf; ignoresmallvolume=false)
    capacity = accountcapacity(xc; force_refresh=force_refresh, ttl_seconds=ttl_seconds)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    freequote = 0.0
    if (:coin in names(assetsdf)) && (:free in names(assetsdf))
        for row in eachrow(assetsdf)
            if uppercase(String(row.coin)) == quotecoin
                freequote += max(0.0, (row.free))
            end
        end
    end
    return (
        balances=balancesdf,
        assets=assetsdf,
        capacity=capacity,
        equity_quote=capacity.equity_quote,
        free_quote=freequote,
        free_margin_quote=capacity.available_opening_quote,
        maintenance_margin_quote=capacity.maintenance_margin_quote,
    )
end

"Return the current order state for one order id."
order_status(xc::XchCache, orderid; auditevent::Bool=true) = getorder(xc, orderid; auditevent=auditevent)

_hascol(df::DataFrame, col::Symbol) = col in propertynames(df)

function _pairfromtradesrow(tradesdf::DataFrame, ix::Integer)
    pair = String(tradesdf[ix, :pair])
    bq = basequote(pair)
    @assert !isnothing(bq) "trades row pair=$(pair) is not a valid base-quote symbol"
    return bq
end

function _ordersidefromaction(action::Symbol)::String
    if action in [:long_open, :short_close]
        return "Buy"
    end
    return "Sell"
end

function _openorderremaining(orow)
    baseqty = hasproperty(orow, :baseqty) ? (orow.baseqty) : 0.0
    executed = hasproperty(orow, :executedqty) ? (orow.executedqty) : 0.0
    return max(0.0, baseqty - executed)
end

function _floatcell(tradesdf::DataFrame, ix::Integer, col::Symbol, default::Float64=0.0)::Float64
    if !_hascol(tradesdf, col)
        return default
    end
    value = tradesdf[ix, col]
    if ismissing(value) || isnothing(value)
        return default
    elseif value isa Real
        return (value)
    end
    return default
end

" tradesdf limit price == 0f0 means adaptive maker price that follows the market price "
_rowlimitprice(value)::Union{Nothing, Real} = value == 0 ? nothing : value

function _implicitflipplan(tradesdf::DataFrame, ix::Integer, action::Symbol, open_limitprice)
    if action == :long_open
        closeqty = tradesdf[ix, :sp_amount]
        closelimit = (tradesdf[ix, :sc_limit] == 0f0) || (open_limitprice == 0f0) ? nothing : min(tradesdf[ix, :sc_limit], open_limitprice)
        # closelimit = 0f0 means adaptive maker price that follows the market price
        return (needed=closeqty > 0.0, positionside=:short, closeqty=closeqty, closelimit=closelimit, close_id_col=:sc_id, close_status_col=:sc_status, close_filled_col=:sc_filled, close_pavg_col=:sc_pavg)
    elseif action == :short_open
        closeqty = tradesdf[ix, :lp_amount]
        closelimit = (tradesdf[ix, :sc_limit] == 0f0) || (open_limitprice == 0f0) ? nothing : max(tradesdf[ix, :sc_limit], open_limitprice)
        # closelimit = 0f0 means adaptive maker price that follows the market price
        return (needed=closeqty > 0.0, positionside=:long, closeqty=closeqty, closelimit=closelimit, close_id_col=:lc_id, close_status_col=:lc_status, close_filled_col=:lc_filled, close_pavg_col=:lc_pavg)
    end
    return (needed=false, positionside=:long, closeqty=0.0, closelimit=open_limitprice, close_id_col=:lc_id, close_status_col=:lc_status, close_filled_col=:lc_filled, close_pavg_col=:lc_pavg)
end

function _apply_accountsnapshot!(tradesdf::DataFrame, ix::Integer, acct)
    tradesdf[ix, :equity] = (acct.equity_quote)
    tradesdf[ix, :balance] = (acct.free_quote)
    tradesdf[ix, :freemargin] = (acct.free_margin_quote)
    tradesdf[ix, :freequote] = (acct.free_quote)
    return nothing
end

function _rejectedrequest!(xc::XchCache, tradesdf::DataFrame, ix::Integer, action::Symbol, message::AbstractString)
    logged = log_trading_issue(xc, "Trading", message)
    if action == :long_open
        tradesdf[ix, :lo_status] = "rejected"
        tradesdf[ix, :lo_msg] = logged
    elseif action == :long_close
        tradesdf[ix, :lc_status] = "rejected"
        tradesdf[ix, :lc_msg] = logged
    elseif action == :short_open
        tradesdf[ix, :so_status] = "rejected"
        tradesdf[ix, :so_msg] = logged
    else
        tradesdf[ix, :sc_status] = "rejected"
        tradesdf[ix, :sc_msg] = logged
    end
    return logged
end

@inline function _is_open_label(label)::Bool
    return label in (longopen, longstrongopen, shortopen, shortstrongopen)
end

function _row_has_position_amount(tradesdf::DataFrame, ix::Integer)::Bool
    has_lo = _hascol(tradesdf, :lo_amount) && !ismissing(tradesdf[ix, :lo_amount]) && (abs((tradesdf[ix, :lo_amount])) > 0f0)
    has_lc = _hascol(tradesdf, :lc_amount) && !ismissing(tradesdf[ix, :lc_amount]) && (abs((tradesdf[ix, :lc_amount])) > 0f0)
    has_so = _hascol(tradesdf, :so_amount) && !ismissing(tradesdf[ix, :so_amount]) && (abs((tradesdf[ix, :so_amount])) > 0f0)
    has_sc = _hascol(tradesdf, :sc_amount) && !ismissing(tradesdf[ix, :sc_amount]) && (abs((tradesdf[ix, :sc_amount])) > 0f0)
    return has_lo || has_lc || has_so || has_sc
end

function _carry_lastopentrade_from_previous!(tradesdf::DataFrame, ix::Integer)
    if !_row_has_position_amount(tradesdf, ix)
        tradesdf[ix, :lastopentrade] = missing
        return nothing
    end
    if !ismissing(tradesdf[ix, :lastopentrade])
        return nothing
    end
    for j in (ix - 1):-1:1
        prev = tradesdf[j, :lastopentrade]
        if !ismissing(prev)
            tradesdf[ix, :lastopentrade] = prev
            break
        end
    end
    return nothing
end

"Synchronize one trades row's exchange feedback columns from current order ids."
function order_status(xc::XchCache, tradesdf::DataFrame, ix::Integer; auditevent::Bool=true)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    _carry_lastopentrade_from_previous!(tradesdf, ix)
    row_is_open_intent = _is_open_label(tradesdf[ix, :label])

    for (idcol, stcol, filledcol, avgcol, msgcol) in [
        (:lo_id, :lo_status, :lo_filled, :lo_pavg, :lo_msg),
        (:lc_id, :lc_status, :lc_filled, :lc_pavg, :lc_msg),
        (:so_id, :so_status, :so_filled, :so_pavg, :so_msg),
        (:sc_id, :sc_status, :sc_filled, :sc_pavg, :sc_msg),
    ]
        oid = tradesdf[ix, idcol]
        ismissing(oid) && continue  # no order id assigned to this lane
        oids = String(oid)
        (lowercase(strip(oids)) == NO_ORDER_ID || isempty(strip(oids))) && continue
        info = getorder(xc, String(oid); auditevent=auditevent)
        if isnothing(info)
            tradesdf[ix, stcol] = "none"
            continue
        end
        rawstatus = hasproperty(info, :status) ? String(info.status) : "unknown"
        status = normalize_order_status(xc.bc, rawstatus)
        tradesdf[ix, stcol] = status
        if hasproperty(info, :baseqty) && hasproperty(info, :executedqty)
            executed = (info.executedqty)
            tradesdf[ix, filledcol] = (max(0.0, executed))
            if row_is_open_intent && (executed > 0.0)
                tradesdf[ix, :lastopentrade] = tradesdf[ix, :opentime]
            end
        end
        if hasproperty(info, :avgprice) && !ismissing(info.avgprice)
            tradesdf[ix, avgcol] = (info.avgprice)
        end
        if hasproperty(info, :rejectreason)
            rr = String(info.rejectreason)
            if !isempty(strip(rr)) && (uppercase(rr) != "NO ERROR")
                tradesdf[ix, msgcol] = log_trading_issue(xc, exchange(xc), rr)
            end
        end
    end
    return tradesdf
end

"""
    sync_latest_trades_rows!(xc, syncpairs=nothing)

Materialize or advance Trades rows to the current OHLCV timestamp for each active
base, applying the latest order status, portfolio positions, and account snapshot.

When `syncpairs` is provided (e.g. `["BTCUSDT"]`), only those pairs are synced
and missing pair entries are created. When `syncpairs` is `nothing`, all bases
currently in `xc.bases` are synced.

Returns `Dict{String, NamedTuple{(:tradesdf, :rowix)}}` keyed by uppercase base.
"""
function sync_latest_trades_rows!(xc::XchCache, syncpairs=nothing)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    acct = account_status(xc; force_refresh=true, ttl_seconds=0)
    balancesdf = acct.balances

    bases_to_sync = String[]
    if isnothing(syncpairs)
        for base in keys(xc.bases)
            uppercase(String(base)) == quotecoin && continue
            push!(bases_to_sync, uppercase(String(base)))
        end
    else
        for pair in syncpairs
            bq = basequote(String(pair))
            isnothing(bq) && continue
            base = uppercase(String(bq.basecoin))
            base == quotecoin && continue
            base in bases_to_sync || push!(bases_to_sync, base)
        end
    end

    rowsbybase = Dict{String, NamedTuple}()

    for base in bases_to_sync
        pairkey = tradingpairkey(base, quotecoin)
        currentdt = if base in keys(xc.bases)
            o = ohlcv(xc, base)
            odf = Ohlcv.dataframe(o)
            size(odf, 1) > 0 ? odf[Ohlcv.ix(o), :opentime] : (isnothing(xc.currentdt) ? xc.startdt : xc.currentdt)
        else
            isnothing(xc.currentdt) ? xc.startdt : xc.currentdt
        end

        tdf_rowix = ensuretradesrow!(xc, base, quotecoin, currentdt)
        tdf = tdf_rowix.tradesdf
        rowix = tdf_rowix.rowix

        # OHLCV columns
        if base in keys(xc.bases)
            o = ohlcv(xc, base)
            odf = Ohlcv.dataframe(o)
            oix = Ohlcv.ix(o)
            if size(odf, 1) > 0 && 1 <= oix <= size(odf, 1)
                :close ∈ propertynames(tdf) && (tdf[rowix, :close] = (odf[oix, :close]))
                :high  ∈ propertynames(tdf) && (tdf[rowix, :high]  = (odf[oix, :high]))
                :low   ∈ propertynames(tdf) && (tdf[rowix, :low]   = (odf[oix, :low]))
            end
        end

        # Sync order statuses for all lanes
        order_status(xc, tdf, rowix; auditevent=false)

        # Position amounts from portfolio snapshot
        bix = _hascol(balancesdf, :coin) ? findfirst(==(base), uppercase.(String.(balancesdf[!, :coin]))) : nothing
        if !isnothing(bix)
            free_val  = _hascol(balancesdf, :free)     ? (balancesdf[bix, :free])     : 0f0
            borr_val  = _hascol(balancesdf, :borrowed) ? (balancesdf[bix, :borrowed]) : 0f0
            :lp_amount ∈ propertynames(tdf) && (tdf[rowix, :lp_amount] = max(0f0, free_val))
            :sp_amount ∈ propertynames(tdf) && (tdf[rowix, :sp_amount] = max(0f0, borr_val))
        end

        # lastopentrade: set to current time on open-order fills, else propagate or clear
        lo_filled = (:lo_filled ∈ propertynames(tdf) && !ismissing(tdf[rowix, :lo_filled])) ? (tdf[rowix, :lo_filled]) : 0f0
        so_filled = (:so_filled ∈ propertynames(tdf) && !ismissing(tdf[rowix, :so_filled])) ? (tdf[rowix, :so_filled]) : 0f0
        lp_amount = (:lp_amount ∈ propertynames(tdf) && !ismissing(tdf[rowix, :lp_amount])) ? (tdf[rowix, :lp_amount]) : 0f0
        sp_amount = (:sp_amount ∈ propertynames(tdf) && !ismissing(tdf[rowix, :sp_amount])) ? (tdf[rowix, :sp_amount]) : 0f0
        if :lastopentrade ∈ propertynames(tdf)
            if lo_filled > 0f0 || so_filled > 0f0
                tdf[rowix, :lastopentrade] = currentdt
            elseif lp_amount > 0f0 || sp_amount > 0f0
                if ismissing(tdf[rowix, :lastopentrade])
                    for j in (rowix - 1):-1:1
                        prev = tdf[j, :lastopentrade]
                        if !ismissing(prev)
                            tdf[rowix, :lastopentrade] = prev
                            break
                        end
                    end
                end
            else
                tdf[rowix, :lastopentrade] = missing
            end
        end

        # Account snapshot columns
        _apply_accountsnapshot!(tdf, rowix, acct)
        :maintmargin ∈ propertynames(tdf) && (tdf[rowix, :maintmargin] = (acct.capacity.maintenance_margin_quote))

        rowsbybase[base] = (tradesdf=tdf, rowix=rowix)
    end

    return rowsbybase
end

"Evaluate and execute one row-level order request from the Trades DataFrame."
function process_order_request(xc::XchCache, tradesdf::DataFrame, ix::Integer)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    pair = _pairfromtradesrow(tradesdf, ix)
    base = pair.basecoin
    quotecoin = pair.quotecoin
    labelval = tradesdf[ix, :label]
    action = if labelval in (longopen, longstrongopen)
        :long_open
    elseif (labelval in (longclose, longstrongclose)) && (tradesdf[ix, :lp_amount] > 0f0)
        :long_close
    elseif labelval in (shortopen, shortstrongopen)
        :short_open
    elseif (labelval in (shortclose, shortstrongclose)) && (tradesdf[ix, :sp_amount] > 0f0)
        :short_close
    else
        :none
    end
    action == :none && return (accepted=false, action=:none, reason="no actionable label")

    limitcol = if action == :long_open
        :lo_limit
    elseif action == :long_close
        :lc_limit
    elseif action == :short_open
        :so_limit
    else  # :short_close
        :sc_limit
    end
    orderamountcol = if action in [:long_open, :long_close]
        action == :long_open ? :lo_amount : :lc_amount
    else
        action == :short_open ? :so_amount : :sc_amount
    end
    idcol = if action == :long_open
        :lo_id
    elseif action == :long_close
        :lc_id
    elseif action == :short_open
        :so_id
    else  # :short_close
        :sc_id
    end
    stcol = if action == :long_open
        :lo_status
    elseif action == :long_close
        :lc_status
    elseif action == :short_open
        :so_status
    else  # :short_close
        :sc_status
    end
    filledcol = if action == :long_open
        :lo_filled
    elseif action == :long_close
        :lc_filled
    elseif action == :short_open
        :so_filled
    else  # :short_close
        :sc_filled
    end
    avgcol = if action == :long_open
        :lo_pavg
    elseif action == :long_close
        :lc_pavg
    elseif action == :short_open
        :so_pavg
    else  # :short_close
        :sc_pavg
    end

    limitprice = _rowlimitprice(tradesdf[ix, limitcol])
    orderamount = tradesdf[ix, orderamountcol]

    if !(orderamount > 0f0)
        _rejectedrequest!(xc, tradesdf, ix, action, "amount=$(orderamount) is not tradable for action=$(action) pair=$(base)-$(quotecoin)")
        return (accepted=false, action=action, reason="amount_not_positive")
    end

    minqty = minimumbasequantity(xc, base, tradesdf[ix, :close])
    if orderamount < minqty
        _rejectedrequest!(xc, tradesdf, ix, action, "base amount=$(orderamount) below minimum base qty $(minqty) for pair=$(base)-$(quotecoin)")
        return (accepted=false, action=action, reason="below_minimum_qty")
    end

    side = _ordersidefromaction(action)
    oid = nothing
    try
        if action in [:long_open, :short_open]
            flip = _implicitflipplan(tradesdf, ix, action, limitprice)
            closeoid = nothing
            if flip.needed
                existing_closeid = (tradesdf[ix, flip.close_id_col] == NO_ORDER_ID) || (tradesdf[ix, flip.close_id_col] == "") ? nothing : tradesdf[ix, flip.close_id_col]
                closeoid = upsertcloseorder!(xc.bc, pair, flip.positionside, flip.closeqty, flip.closelimit; existing_orderid=existing_closeid, maker=true, reduceonly=true)

                if isnothing(closeoid)
                    _rejectedrequest!(xc, tradesdf, ix, action, "exchange returned no close order id for action=$(action) pair=$(base)-$(quotecoin)")
                    # return (accepted=false, action=action, reason="missing_close_orderid") # not applicable because the order can be executed in the meanwhile, the open order still need tobe placed 
                else
                    closeoid = String(closeoid)
                    tradesdf[ix, flip.close_id_col] = closeoid
                    tradesdf[ix, flip.close_status_col] = "submitted"
                end
            end

            existing_openid = (tradesdf[ix, idcol] == NO_ORDER_ID) || (tradesdf[ix, idcol] == "") ? nothing : tradesdf[ix, idcol]
            openside = action == :long_open ? :long : :short
            oid = upsertopenorder!(xc.bc, pair, openside, orderamount, limitprice; existing_orderid=existing_openid, maker=true, reduceonly=false)
            if isnothing(oid)
                _rejectedrequest!(xc, tradesdf, ix, action, "exchange returned no open order id for action=$(action) pair=$(base)-$(quotecoin)")
                return (accepted=false, action=action, reason="missing_open_orderid")
            end
            oid = String(oid)
            tradesdf[ix, idcol] = oid
            tradesdf[ix, stcol] = "submitted"
            if flip.needed
                _ = directsequence!(xc.bc, closeoid, oid)
            end
        elseif action == :long_close
            existing_closeid = (tradesdf[ix, idcol] == NO_ORDER_ID) || (tradesdf[ix, idcol] == "") ? nothing : tradesdf[ix, idcol]
            oid = upsertcloseorder!(xc.bc, pair, :long, orderamount, limitprice; existing_orderid=existing_closeid, maker=true, reduceonly=true)
            if isnothing(oid)
                _rejectedrequest!(xc, tradesdf, ix, action, "exchange returned no order id for action=$(action) pair=$(base)-$(quotecoin)")
                return (accepted=false, action=action, reason="missing_orderid")
            end
            tradesdf[ix, idcol] = String(oid)
            tradesdf[ix, stcol] = "submitted"
        elseif action == :short_close
            existing_closeid = (tradesdf[ix, idcol] == NO_ORDER_ID) || (tradesdf[ix, idcol] == "") ? nothing : tradesdf[ix, idcol]
            oid = upsertcloseorder!(xc.bc, pair, :short, orderamount, limitprice; existing_orderid=existing_closeid, maker=true, reduceonly=true)
            if isnothing(oid)
                _rejectedrequest!(xc, tradesdf, ix, action, "exchange returned no order id for action=$(action) pair=$(base)-$(quotecoin)")
                return (accepted=false, action=action, reason="missing_orderid")
            end
            tradesdf[ix, idcol] = String(oid)
            tradesdf[ix, stcol] = "submitted"
        else # no action
            return (accepted=true, action=action, reason="no_action")
        end
    catch err
        logged = log_trading_issue(xc, exchange(xc), sprint(showerror, err))
        if action == :long_open
            tradesdf[ix, :lo_msg] = logged
            tradesdf[ix, :lo_status] = "Error"
        elseif action == :long_close
            tradesdf[ix, :lc_msg] = logged
            tradesdf[ix, :lc_status] = "Error"
        elseif action == :short_open
            tradesdf[ix, :so_msg] = logged
            tradesdf[ix, :so_status] = "Error"
        else  # :short_close
            tradesdf[ix, :sc_msg] = logged
            tradesdf[ix, :sc_status] = "Error"
        end
        return (accepted=false, action=action, reason="exchange_error", error=sprint(showerror, err))
    end

    return (accepted=true, action=action, orderid=oid)
end

function _adapterbalances(xc::XchCache)::DataFrame
    bdf = balances(xc.bc)
    return isnothing(bdf) ? DataFrame() : DataFrame(bdf; copycols=true)
end

function _filterbalances!(xc::XchCache, bdf::DataFrame; ignoresmallvolume::Bool=true)::DataFrame
    if (size(bdf, 1) > 0) && ignoresmallvolume
        delrows = []
        for ix in eachindex(bdf[!, :coin])
            if bdf[ix, :coin] != EnvConfig.pairquote
                sym = symboltoken(bdf[ix, :coin])
                syminfo = minimumqty(xc, sym)
                if !validsymbol(xc, sym) || ((abs(bdf[ix, :free]) + abs(bdf[ix, :locked]) + abs(bdf[ix, :borrowed])) < 1.01 * syminfo.minbaseqty) # 1% more to avoid issues by rounding errors
                    push!(delrows, ix)
                end
            end
        end
        deleteat!(bdf, delrows)
    end
    return bdf
end

"Returns a DataFrame[:coin, :locked, :free, :borrowed, :accruedinterest] of wallet/portfolio balances"
function balances(xc::XchCache; ignoresmallvolume=true, prefer_websocket::Bool=true)
    use_ws_primary = prefer_websocket && _wsenabled(xc, :ws_primary_mode, false) && _wsenabled(xc, :ws_balances_enabled, false)
    bdf = if use_ws_primary
        refreshbalancessnapshot!(xc; ignoresmallvolume=false).snapshot
    else
        _adapterbalances(xc)
    end
    return _filterbalances!(xc, DataFrame(bdf; copycols=true); ignoresmallvolume=ignoresmallvolume)
end

"Capture one canonical exchange-owned balances snapshot and store it in `xc.mc`."
function refreshbalancessnapshot!(xc::XchCache; ignoresmallvolume::Bool=false)
    use_ws_primary = _wsenabled(xc, :ws_primary_mode, false) && _wsenabled(xc, :ws_balances_enabled, false)
    snapshot = if use_ws_primary
        _ensurewsbalances!(xc)
        wsb = wsbalancessnapshot(xc)
        wsdt = wsbalancesheartbeat(xc)
        if (size(wsb, 1) > 0) || !isnothing(wsdt)
            wsb
        else
            (verbosity >= 1) && @warn "ws balance snapshot unavailable; falling back to REST balances"
            _adapterbalances(xc)
        end
    else
        _adapterbalances(xc)
    end
    snapshotdf = isnothing(snapshot) ? DataFrame() : DataFrame(snapshot; copycols=true)
    _filterbalances!(xc, snapshotdf; ignoresmallvolume=ignoresmallvolume)
    xc.mc[:exchange_balances_snapshot] = deepcopy(snapshotdf)
    xc.mc[:exchange_balances_snapshot_dt] = isnothing(xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : xc.currentdt
    return (snapshot=xc.mc[:exchange_balances_snapshot], datetime=xc.mc[:exchange_balances_snapshot_dt], fresh=true)
end

"Return the canonical exchange-owned balances snapshot from `xc.mc`, refreshing on demand when requested or missing."
function balancessnapshot(xc::XchCache; force_refresh::Bool=false, max_age::Dates.Period=Minute(2), ignoresmallvolume::Bool=false)
    has_snapshot = haskey(xc.mc, :exchange_balances_snapshot) && haskey(xc.mc, :exchange_balances_snapshot_dt)
    if force_refresh || !has_snapshot
        return refreshbalancessnapshot!(xc; ignoresmallvolume=ignoresmallvolume)
    end

    snapshot = xc.mc[:exchange_balances_snapshot]
    snapdt = xc.mc[:exchange_balances_snapshot_dt]
    nowdt = isnothing(xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : xc.currentdt
    if isnothing(snapdt)
        return refreshbalancessnapshot!(xc; ignoresmallvolume=ignoresmallvolume)
    end
    fresh = (nowdt - DateTime(snapdt)) <= max_age
    return (snapshot=snapshot, datetime=snapdt, fresh=fresh)
end

"""
Appends a balances DataFrame with the USDT value of the coin asset using usdtdf[:lastprice] and returns it as DataFrame[:coin, :locked, :free, :usdtprice, :usdtvalue].
"""
function portfolio!(xc::XchCache, balancesdf=balances(xc, ignoresmallvolume=false), usdtdf=nothing; ignoresmallvolume=true)
    if isnothing(xc.currentdt)
        if isnothing(usdtdf)
            quotetoken = uppercase(String(EnvConfig.pairquote))
            requestedbases = [uppercase(String(c)) for c in balancesdf[!, :coin] if uppercase(String(c)) != quotetoken]
            usdtdf = valuationUSDTmarket(xc, requestedbases)
        end
        portfoliodf = leftjoin(balancesdf, usdtdf[!, [:basecoin, :lastprice]], on = :coin => :basecoin)
        portfoliodf.lastprice = coalesce.(portfoliodf.lastprice, 1.0f0)
        rename!(portfoliodf, :lastprice => "usdtprice")
    else
        usdtprice = Float32[]
        portfoliodf = balancesdf[:, :]
        for bix in eachindex(portfoliodf[!, :coin])
            if portfoliodf[bix, :coin] == EnvConfig.pairquote
                push!(usdtprice, 1f0)
            else
                if !validbase(xc, portfoliodf[bix, :coin])
                    (verbosity >= 2) && @warn "portfolio!: skipping invalid/non-tradeable base $(portfoliodf[bix, :coin])"
                    push!(usdtprice, 0f0)
                    continue
                end
                ohlcv = try
                    setcurrenttime!(xc, portfoliodf[bix, :coin], xc.currentdt)
                catch err
                    (verbosity >= 3) && @warn "portfolio!: skipping price fetch for $(portfoliodf[bix, :coin]) — unknown or unsupported pair" exception=sprint(showerror, err)
                    push!(usdtprice, 0f0)
                    continue
                end
                if size(ohlcv.df, 1) > 0
                    push!(usdtprice, ohlcv.df[ohlcv.ix, :close])
                else
                    (verbosity >= 3) && @warn "found no data at $(xc.currentdt) for asset $ohlcv"
                    push!(usdtprice, 0f0)
                end
            end
        end
        portfoliodf.usdtprice = usdtprice
    end
    # Value is net base exposure in USDT (free + locked - borrowed).
    # This keeps pure shorts (free=0, borrowed>0) negative instead of incorrectly zero.
    portfoliodf.usdtvalue = (portfoliodf.free .+ portfoliodf.locked .- portfoliodf.borrowed) .* portfoliodf.usdtprice
    if ignoresmallvolume
        delrows = []
        for ix in eachindex(portfoliodf[!, :coin])
            coin = String(portfoliodf[ix, :coin])
            minbasequant = minimumbasequantity(xc, coin, portfoliodf[ix, :usdtprice])
            is_quotecoin = (uppercase(coin) == uppercase(EnvConfig.pairquote)) || (coin in quotecoins)
            if !is_quotecoin && (isnothing(minbasequant) || ((abs(portfoliodf[ix, :free]) + abs(portfoliodf[ix, :locked]) + abs(portfoliodf[ix, :borrowed])) < minbasequant))
                push!(delrows, ix)
            end
        end
        deleteat!(portfoliodf, delrows)
    end
    return portfoliodf
end

openstatus(st::AbstractString)::Bool = st in ["New", "PartiallyFilled", "Untriggered", "Open"]
openstatus(stvec::AbstractVector{String})::Vector{Bool} = [openstatus(st) for st in stvec]

"""
Returns an AbstractDataFrame of open **spot** orders with columns:

- orderid ::String
- symbol ::String
- side ::String (`Buy` or `Sell`)
- baseqty ::Float32
- ordertype ::String  `Market`, `Limit`
- timeinforce ::String      `GTC` GoodTillCancel, `IOC` ImmediateOrCancel, `FOK` FillOrKill, `PostOnly`
- limitprice ::Float32
- executedqty ::Float32  (to be executed qty = baseqty - executedqty)
- status ::String      `New`, `PartiallyFilled`, `Untriggered`, `Rejected`, `PartiallyFilledCanceled`, `Filled`, `Cancelled`, `Triggered`, `Deactivated`
- created ::DateTime
- updated ::DateTime
- rejectreason ::String
"""
function getopenorders(xc::XchCache, base=nothing)::AbstractDataFrame
    use_ws_primary = isnothing(base) && _wsenabled(xc, :ws_primary_mode, false) && _wsenabled(xc, :ws_orders_enabled, false)
    oo = if use_ws_primary
        _ensurewsorders!(xc)
        wsdf = wsordersnapshot(xc)
        wsdt = wsordersheartbeat(xc)
        if (size(wsdf, 1) > 0) || !isnothing(wsdt)
            wsdf
        else
            (verbosity >= 1) && @warn "ws order snapshot unavailable; falling back to REST openorders"
            openorders(xc.bc, symbol=symboltoken(base))
        end
    else
        openorders(xc.bc, symbol=symboltoken(base))
    end
    openordersdf = size(oo) == (0, 0) ? emptyorders(xc) : oo
    if isnothing(base) && "orderid" in names(openordersdf)
        pruneadaptiveorders!(xc, openordersdf[!, :orderid])
    end
    return openordersdf
end

"Returns a named tuple with elements equal to columns of getopenorders() dataframe of the identified order or `nothing` if order is not found"
function getorder(xc::XchCache, orderid; auditevent::Bool=true)
    orderinfo = order(xc.bc, orderid)
    return orderinfo
end

"Returns orderid in case of a successful cancellation"
function cancelorder(xc::XchCache, base, orderid; leg_group_id=nothing, leg_label=nothing)
    unregisteradaptiveorder!(xc, orderid)
    cancelsymbol = symboltoken(xc, base, EnvConfig.pairquote)
    cancelled = cancelorder(xc.bc, cancelsymbol, orderid)
    return cancelled
end


"""
Create an open position order with explicit configside intent.
- `configside=:long` submits a buy order.
- `configside=:short` submits a sell order.
Returns `nothing` when `basequantity` is below the symbol minimum quantity.
Throws `ArgumentError` for invalid (negative) `basequantity`.
"""

function createopenorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=true, configside::Symbol, reduceonly::Bool=false, kwargs...)
    basequantity < 0 && throw(ArgumentError("basequantity=$(basequantity) must be non-negative for createopenorder"))
    @assert configside in (:long, :short) "createopenorder configside=$(configside) must be :long or :short"
    refprice = isnothing(limitprice) ? nothing : (limitprice)
    if isnothing(refprice) && uppercase(String(base)) in keys(xc.bases)
        refprice = (currentprice(ohlcv(xc, uppercase(String(base)))))
    end
    minqty = isnothing(refprice) || (refprice <= 0f0) ? nothing : minimumbasequantity(xc, base, refprice)
    if !isnothing(minqty) && (basequantity) < (minqty)
        return nothing
    end
    if configside == :long
        return createbuyorder(xc, base; limitprice=limitprice, basequantity=basequantity, maker=maker, reduceonly=reduceonly)
    else
        return createsellorder(xc, base; limitprice=limitprice, basequantity=basequantity, maker=maker, reduceonly=reduceonly)
    end
end

function createbuyorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=false, reduceonly::Bool=false, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    base = uppercase(base)
    symbol = symboltoken(xc, base, EnvConfig.pairquote)
    try
        # Adapter-backed path for both live and simulation exchanges.
        created = createorder(xc.bc, symbol, "Buy", basequantity, limitprice, maker, reduceonly=reduceonly)
        oid, oocreate = _normalizecreatedorder(xc, created)
        if isnothing(limitprice) && maker && !isnothing(oid)
            registeradaptiveorder!(xc, oid)
        end
        (verbosity >= 3) && @info "$(tradetime(xc)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    catch err
        rethrow()
    end
end

"""
Places an order using the adapter-defined execution configuration.
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.

Pass `limitprice=nothing` together with `maker=true` to ask the adapter to choose
a limit price as close as possible to the current spread while remaining post-only,
so the order can qualify for maker fees.

Order is rejected (but order created) if the resulting price crosses the spread in
order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createsellorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=true, reduceonly::Bool=false, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    base = uppercase(base)
    symbol = symboltoken(xc, base, EnvConfig.pairquote)
    try
        # Adapter-backed path for both live and simulation exchanges.
        created = createorder(xc.bc, symbol, "Sell", basequantity, limitprice, maker, reduceonly=reduceonly)
        oid, oocreate = _normalizecreatedorder(xc, created)
        if isnothing(limitprice) && maker && !isnothing(oid)
            registeradaptiveorder!(xc, oid)
        end
        (verbosity >= 3) && @info "$(tradetime(xc)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    catch err
        rethrow()
    end
end

"""
Amend an existing order.

If the order is post-only and `limitprice=nothing`, the routed adapter will
re-snapshot the current spread and keep the maker intent adaptive instead of
freezing the previous limit.
"""
function changeorder(xc::XchCache, symbol::AbstractString, orderid; limitprice=nothing, basequantity=nothing, leg_group_id=nothing, leg_label=nothing)
    amended = amendorder(xc.bc, String(symbol), String(orderid); basequantity=basequantity, limitprice=limitprice)
    new_orderid, ooamend = _normalizeamendedorder(xc, amended)
    if isnothing(new_orderid)
        return nothing
    end
    old_orderid = String(orderid)
    if new_orderid != old_orderid
        if isadaptiveorder(xc, old_orderid)
            unregisteradaptiveorder!(xc, old_orderid)
            registeradaptiveorder!(xc, new_orderid)
        end
    end
    return new_orderid
end

function changeorder(xc::XchCache, orderid; limitprice=nothing, basequantity=nothing, leg_group_id=nothing, leg_label=nothing)
    amended = amendorder(xc.bc, String(orderid); basequantity=basequantity, limitprice=limitprice)
    new_orderid, ooamend = _normalizeamendedorder(xc, amended)
    if isnothing(new_orderid)
        return nothing
    end
    old_orderid = String(orderid)
    if new_orderid != old_orderid
        if isadaptiveorder(xc, old_orderid)
            unregisteradaptiveorder!(xc, old_orderid)
            registeradaptiveorder!(xc, new_orderid)
        end
    end
    return new_orderid
end

"""
    createocoorder(xc, base; entry_side, entry_price, take_profit_price, stop_loss_price,
                   basequantity, maker=false, signal_label=nothing,
                   signal_score=nothing, strategy_engine=nothing, strategy_config_ref=nothing) -> NamedTuple

Places a three-leg bracket (OCO) order group:
- **entry**: the initial buy or sell (`entry_side ∈ (:buy, :sell)`)
- **take_profit**: limit order on the opposite side at `take_profit_price`
- **stop_loss**: limit order on the opposite side at `stop_loss_price`

All three legs share the same `leg_group_id` (a new UUID) and the take-profit/stop-loss
legs record the entry order id in the trades dataframe .

Returns a `NamedTuple` `(; leg_group_id, entry_order_id, take_profit_order_id, stop_loss_order_id)`.
Any leg that fails to submit will have `nothing` as its order id.
"""
function createocoorder(xc::XchCache, base::AbstractString;
                        entry_side::Symbol,
                        entry_price::Real,
                        take_profit_price::Real,
                        stop_loss_price::Real,
                        basequantity::Real,
                        maker::Bool=false,
                        signal_label=nothing,
                        signal_score=nothing,
                        strategy_engine=nothing,
                        strategy_config_ref=nothing)
    @assert entry_side in (:buy, :sell) "entry_side must be :buy or :sell, got $entry_side"
    leg_group_id = string(UUIDs.uuid4())
    exit_buy = entry_side == :sell

    # Helper: set full context (signal info + leg metadata) and return it to the caller so
    # we can manage the clear ourselves rather than relying on createXorder's finally block.
    leg_group_id = string(UUIDs.uuid4())
    exit_buy = entry_side == :sell

    # --- entry leg ---
    entry_order_id = if entry_side == :buy
        createbuyorder(xc, base;
            limitprice=(entry_price),
            basequantity=(basequantity),
            maker=maker,
        )
    else
        createsellorder(xc, base;
            limitprice=(entry_price),
            basequantity=(basequantity),
            maker=maker,
        )
    end

    # --- take-profit leg ---
    take_profit_order_id = if exit_buy
        createbuyorder(xc, base;
            limitprice=(take_profit_price),
            basequantity=(basequantity),
            maker=maker,
            parent_order_id=entry_order_id,
        )
    else
        createsellorder(xc, base;
            limitprice=(take_profit_price),
            basequantity=(basequantity),
            maker=maker,
            parent_order_id=entry_order_id,
        )
    end

    # --- stop-loss leg ---
    stop_loss_order_id = if exit_buy
        createbuyorder(xc, base;
            limitprice=(stop_loss_price),
            basequantity=(basequantity),
            maker=maker,
            parent_order_id=entry_order_id,
        )
    else
        createsellorder(xc, base;
            limitprice=(stop_loss_price),
            basequantity=(basequantity),
            maker=maker,
            parent_order_id=entry_order_id,
        )
    end

    return (; leg_group_id, entry_order_id, take_profit_order_id, stop_loss_order_id)
end

#endregion account

#region bookkeeping


"Finds or creates an asset order row in an asset dataframe and returns it. "
function _assetrow!(adf::DataFrame, coin)
    aorow = nothing
    adfix = size(adf, 1) > 0 ? findfirst(x -> x == coin, adf[!, :coin]) : nothing
    if isnothing(adfix)
        push!(adf, (coin = coin, free = 0f0, locked = 0f0, marginfree = 0f0, marginlocked = 0f0, assetborrowed = 0f0, orderborrowed = 0f0, accruedinterest = 0f0))
        aorow = last(adf)
    else
        aorow = adf[adfix, :]
    end
    return aorow
end

"Set a fixed asset amount for coin in adapter-backed bookkeeping and return the asset row."
function _updateasset!(xc::XchCache, coin, amount)
    if !(xc.bc isa Bybit.BybitSimCache)
        throw(ArgumentError("_updateasset! requires BybitSim adapter cache for adapter-backed seeding, got $(typeof(xc.bc))"))
    end
    bc = rawcache(xc.bc)
    Bybit.seedportfolio!(bc, coin, amount)
    ix = findfirst(==(uppercase(String(coin))), bc.assets[!, :coin])
    return isnothing(ix) ? nothing : bc.assets[ix, :]
end


_emptyassets()::DataFrame = DataFrame(coin=String31[], free=Float32[], locked=Float32[], marginfree=Float32[], marginlocked=Float32[], assetborrowed=Float32[], orderborrowed=Float32[], accruedinterest=Float32[])

"Return an empty order dataframe with Xch bookkeeping columns added."
function emptyorders(xc::XchCache)::DataFrame
    df = emptyorders(xc.bc)
    if !hasproperty(df, :marginleverage)
        insertcols!(df, :marginleverage => Vector{Int32}(undef, 0))
    end
    return df
end

function _ordersfilestem(xc::XchCache)
    ORDERPREFIX = "Orders"
    fnvec = [ORDERPREFIX]
    push!(fnvec, string(EnvConfig.configmode))
    bases = sort(collect(keys(xc.bases)))
    fnvec = vcat(fnvec, bases)
    push!(fnvec, Dates.format(xc.startdt, "yy-mm-dd"))
    enddt = xc.enddt
    push!(fnvec, Dates.format(enddt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_ordersfilename(xc::XchCache; format::Symbol=:arrow) = EnvConfig.tablepath(_ordersfilestem(xc); folderpath=EnvConfig.logfolder(), format=format)

function writeorders(xc::XchCache)
    # Orders field removed - orders are now managed externally
    return
end

function _assetsfilestem(xc::XchCache, dt)
    ASSETPREFIX = "Assets"
    fnvec = [ASSETPREFIX]
    push!(fnvec, string(EnvConfig.configmode))
    push!(fnvec, Dates.format(dt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_assetsfilename(xc::XchCache, dt; format::Symbol=:arrow) = EnvConfig.tablepath(_assetsfilestem(xc, dt); folderpath=EnvConfig.logfolder(), format=format)

function writeassets(xc::XchCache, dt::DateTime)
    # Assets field removed - asset snapshots are now managed externally
    return
end

include("XchTrades.jl")

#endregion bookkeeping

end  # of module
