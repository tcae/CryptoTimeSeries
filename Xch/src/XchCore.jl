# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "CSV"])


module Xch

using Dates, DataFrames, DataAPI, CSV, Logging, InlineStrings, UUIDs
using CategoricalArrays: CategoricalVector
using Bybit, EnvConfig, KrakenFutures, KrakenSpot, Ohlcv, Targets, TSM
using XchAdapter: XchAdapterCache, TradingPairRef
import XchAdapter: rawcache, exchangeid, symbolinfo, validsymbol, getklines, get24h, balances, positionsnapshot, accountsnapshot, emptyorders, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, marginlimits, marginpermitted, marketdataheartbeats, marketdataheartbeat, wsorderssnapshot, wsordersheartbeat, wsbalancessnapshot, wsbalancesheartbeat, ws_orders, ws_balances, accountcapacity, closeorder, upsertcloseorder!, upsertopenorder!, directsequence!, drainliquidations!, preparetradingpairs!
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
    bases::Dict{String, Ohlcv.OhlcvData}
    tsm::TSM.TsmCache  # owns the pair-state Trades DataFrames and template cache
    bc::XchAdapterCache  # typed adapter cache wrapper
    startdt::Dates.DateTime
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    mc::Dict # MC = module constants
    lastsyncedopentime::Dict{String, Dates.DateTime}  # per-base opentime last processed by sync_latest_trades_rows!
    tradingpairepoch::UInt
    tradingpairrefs::Vector{TradingPairRef}
    tradingpairinfo::Vector{NamedTuple}
    function XchCache(bc::XchAdapterCache; startdt::DateTime=Dates.now(UTC), enddt=nothing)
        startdt = floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        xc = new(Dict{String, Ohlcv.OhlcvData}(), TSM.TsmCache(), bc, startdt, nothing, enddt, Dict(), Dict{String, Dates.DateTime}(), UInt(0), TradingPairRef[], NamedTuple[])
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

"Prepare canonical pair metadata for one Trade configuration epoch."
function preparetradingpairs!(xc::XchCache, pairs::Vector{String})::Vector{TradingPairRef}
    epoch = xc.tradingpairepoch + UInt(1)
    refs = TradingPairRef[]
    info = NamedTuple[]
    sizehint!(refs, length(pairs))
    sizehint!(info, length(pairs))
    quote_coin = String(EnvConfig.pairquote)
    for (ix, pair) in enumerate(pairs)
        @assert pair == uppercase(pair) "trading pair must be canonical uppercase: pair=$(pair)"
        @assert endswith(pair, quote_coin) && (length(pair) > length(quote_coin)) "trading pair must end with configured quote=$(quote_coin): pair=$(pair)"
        base = pair[begin:end - length(quote_coin)]
        symbol = symboltoken(xc.bc, base, quote_coin)
        syminfo = symbolinfo(xc.bc, symbol)
        @assert !isnothing(syminfo) "exchange metadata missing trading pair=$(pair) adapter_symbol=$(symbol)"
        ref = TradingPairRef(pair, UInt(ix), epoch)
        push!(refs, ref)
        push!(info, (pair=pair, basecoin=base, quotecoin=quote_coin, symbol=symbol, minbaseqty=syminfo.minbaseqty, minquoteqty=syminfo.minquoteqty))
    end
    xc.tradingpairepoch = epoch
    xc.tradingpairrefs = refs
    xc.tradingpairinfo = info
    preparetradingpairs!(xc.bc, refs)
    return refs
end

"Return the current epoch pair reference for one one-based Trade cfg index."
function tradingpairref(xc::XchCache, cfgindex::UInt)::TradingPairRef
    @assert xc.tradingpairepoch > 0 "trading pair epoch is not initialized"
    @assert cfgindex > 0 && cfgindex <= length(xc.tradingpairrefs) "cfgindex=$(cfgindex) is outside active pair references=$(length(xc.tradingpairrefs))"
    return xc.tradingpairrefs[cfgindex]
end

"Return the active reference for one cfg pair, or the zero-sentinel fallback before epoch preparation."
function tradingpairref(xc::XchCache, pair::String, cfgindex::UInt)::TradingPairRef
    if xc.tradingpairepoch == 0
        return TradingPairRef(pair, UInt(0), UInt(0))
    end
    ref = tradingpairref(xc, cfgindex)
    @assert ref.pair == pair "Trade cfg pair/index mismatch: pair=$(pair) cfgindex=$(cfgindex) indexed.pair=$(ref.pair)"
    return ref
end

"Return the current Xch pair record for a prepared reference, or `nothing` for its zero sentinel."
function _preparedtradingpairinfo(xc::XchCache, pairref::TradingPairRef)
    pairref.epoch == 0 && return nothing
    @assert pairref.epoch == xc.tradingpairepoch "Xch pair epoch mismatch: pair=$(pairref.pair) ref.epoch=$(pairref.epoch) xch.epoch=$(xc.tradingpairepoch)"
    @assert pairref.cfgindex > 0 "Xch prepared pair reference requires cfgindex > 0: pair=$(pairref.pair) epoch=$(pairref.epoch)"
    @assert pairref.cfgindex <= length(xc.tradingpairinfo) "Xch pair cfgindex=$(pairref.cfgindex) exceeds prepared pairs=$(length(xc.tradingpairinfo))"
    info = xc.tradingpairinfo[pairref.cfgindex]
    @assert info.pair == pairref.pair "Xch pair index mismatch: ref.pair=$(pairref.pair) cfgindex=$(pairref.cfgindex) indexed.pair=$(info.pair)"
    return info
end

"Return the prepared metadata for one active trading pair reference."
function tradingpairinfo(xc::XchCache, pairref::TradingPairRef)
    info = _preparedtradingpairinfo(xc, pairref)
    @assert !isnothing(info) "tradingpairinfo requires an active prepared pair reference: pair=$(pairref.pair) cfgindex=$(pairref.cfgindex) epoch=$(pairref.epoch)"
    return info
end

"Log a trading issue and return the normalized message text for direct storage in Trades columns."
function log_trading_issue(xc::XchCache, issuer::AbstractString, message::AbstractString)::String
    issuerstr = String(issuer)
    messagestr = String(message)
    # @warn "Xch.$(ttstr(xc)) $(issuerstr): $(messagestr)"
    return _normalized_order_msg(messagestr)
end

log_trading_issue(issuer::AbstractString, message::AbstractString) = error("log_trading_issue requires an XchCache; call log_trading_issue(xc, issuer, message)")

const NO_ORDER_ID = "none"
const NO_ORDER_MSG = "none"

@inline _normalized_order_msg(v)::String = begin
    s = ismissing(v) ? "" : strip(String(v))
    return (isempty(s) || lowercase(s) == "none") ? NO_ORDER_MSG : s
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
Return true when `orderid` is still open, preferring a fresh websocket order
snapshot and falling back to a direct (HTTP) order lookup when the websocket
snapshot carries no data for this order (adapter has no websocket support, is
disconnected, or the order predates the current snapshot).
"""
function _orderstillopen(xc::XchCache, orderid::AbstractString)::Bool
    if !isnothing(wsordersheartbeat(xc))
        wsdf = wsordersnapshot(xc)
        if "orderid" in names(wsdf)
            wsix = findfirst(==(String(orderid)), String.(wsdf[!, :orderid]))
            if !isnothing(wsix)
                return openstatus(String(wsdf[wsix, :status]))
            end
        end
    end
    info = order(xc.bc, orderid)
    return !isnothing(info) && hasproperty(info, :status) && openstatus(String(info.status))
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
removeallbases(xc::XchCache) = xc.bases = Dict{String, Ohlcv.OhlcvData}()

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
tradetime(xc::XchCache) = isnothing(xc.currentdt) ? (isnothing(xc.enddt) ? floor(servertime(xc.bc), Minute(1)) : xc.enddt) : xc.currentdt
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

    # Share xc.bases by reference (not a copy) with adapters that simulate order matching
    # against OHLCV candles, so they read the same already-loaded data instead of holding
    # their own duplicate cache. Adapters without this concept (e.g. KrakenSpot/Futures) are
    # left untouched.
    function _setohlcvcache!(bc, bases)
        if !isnothing(bc) && hasproperty(bc, :ohlcvcache)
            setproperty!(bc, :ohlcvcache, bases)
        end
        return nothing
    end

    xc.currentdt = datetime
    _setsimtime!(rawcache(xc.bc), datetime)
    _setohlcvcache!(rawcache(xc.bc), xc.bases)
    if !isnothing(datetime)
        for base in keys(xc.bases)
            try
                setcurrenttime!(xc, base, datetime)
            catch err
                err isa InterruptException && rethrow(err)
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

"Return a conservative quote price for one base from the in-memory OHLCV cache."
function _pricefrombases(xc::XchCache, coin::AbstractString)::Union{Nothing, Float64}
    base = uppercase(String(coin))
    if haskey(xc.bases, base)
        o = ohlcv(xc, base)
        odf = Ohlcv.dataframe(o)
        oix = Ohlcv.ix(o)
        if (size(odf, 1) > 0) && (1 <= oix <= size(odf, 1))
            px = odf[oix, :close]
            return px > 0 ? (px) : nothing
        end
    end

    return nothing
end

"Merge explicit adapter position amounts into balances and compute quote valuation in Xch."
function _assetssnapshot_from_balances_positions(xc::XchCache, balancesdf::AbstractDataFrame; positionsdf=nothing, resolve_missing_prices::Bool=true)::DataFrame
    assets = DataFrame(balancesdf; copycols=true)
    cols = propertynames(assets)
    if !(:coin in cols)
        return DataFrame(coin=String[], free=Float32[], locked=Float32[], usdtprice=Float32[], usdtvalue=Float32[])
    end
    !(:free in cols) && (assets[!, :free] = fill(0f0, nrow(assets)))
    !(:locked in cols) && (assets[!, :locked] = fill(0f0, nrow(assets)))
    !(:short in cols) && (assets[!, :short] = fill(0f0, nrow(assets)))

    # Some adapters report position exposure separate from wallet balances.
    # We merge those quantities here so valuation can be done centrally in Xch.
    posdf = if isnothing(positionsdf)
        try
            positionsnapshot(xc.bc)
        catch err
            err isa InterruptException && rethrow(err)
            (verbosity >= 2) && @warn "positionsnapshot unavailable; falling back to balances-only valuation" exchange=exchange(xc) exception=sprint(showerror, err)
            DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
        end
    else
        positionsdf
    end

    if (:coin in propertynames(posdf)) && (:long_qty in propertynames(posdf)) && (:short_qty in propertynames(posdf))
        for prow in eachrow(posdf)
            coin = uppercase(String(prow.coin))
            qix = findfirst(==(coin), uppercase.(String.(assets[!, :coin])))
            if isnothing(qix)
                push!(assets, (coin=coin, free=0f0, locked=0f0))
                qix = nrow(assets)
            end
            lqty = max(0f0, (prow.long_qty))
            sqty = max(0f0, (prow.short_qty))
            # positionsnapshot reports the TOTAL exposure (free+locked combined, e.g. once
            # part of a position is reserved for a pending reduce-only order). Only top up
            # the shortfall against the existing free+locked total; comparing against
            # :free alone double-counted the already-reserved (:locked) portion.
            existingtotal = assets[qix, :free] + assets[qix, :locked]
            if lqty > existingtotal
                assets[qix, :free] += lqty - existingtotal
            end
            if sqty > assets[qix, :short]
                assets[qix, :short] = sqty
            end
        end
    end

    quotecoin = uppercase(String(EnvConfig.pairquote))
    usdtprice = Float32[]
    for row in eachrow(assets)
        coin = uppercase(String(row.coin))
        if coin == quotecoin
            push!(usdtprice, 1f0)
            continue
        end
        px = _pricefrombases(xc, coin)
        push!(usdtprice, isnothing(px) ? 0f0 : (px))
    end

    # For live modes, resolve missing quote prices via coin-scoped market snapshots.
    if resolve_missing_prices && !timesimulation(xc)
        missingbases = String[]
        for ix in eachindex(usdtprice)
            coin = uppercase(String(assets[ix, :coin]))
            if (coin != quotecoin) && (usdtprice[ix] <= 0f0)
                push!(missingbases, coin)
            end
        end
        if !isempty(missingbases)
            qdf = valuationUSDTmarket(xc, unique(missingbases))
            if (:basecoin in propertynames(qdf)) && (:lastprice in propertynames(qdf))
                pxbycoin = Dict{String, Float64}()
                for row in eachrow(qdf)
                    pxbycoin[uppercase(String(row.basecoin))] = (row.lastprice)
                end
                for ix in eachindex(usdtprice)
                    if usdtprice[ix] <= 0f0
                        coin = uppercase(String(assets[ix, :coin]))
                        if haskey(pxbycoin, coin)
                            usdtprice[ix] = (pxbycoin[coin])
                        end
                    end
                end
            end
        end
    end

    assets[!, :usdtprice] = usdtprice
    assets[!, :usdtvalue] = (assets[!, :free] .+ assets[!, :locked] .- assets[!, :short]) .* assets[!, :usdtprice]
    return assets
end

"Merge Xch-valued capacity with exchange aggregate capacity when available."
function _mergecapacity(assetcap, exchcap)
    _capvalue(cap, field::Symbol, default::Float64=0.0) = hasproperty(cap, field) ? max(0.0, _asfloat64(getproperty(cap, field), default)) : default
    exch_equity = _capvalue(exchcap, :equity_quote)
    exch_opening = _capvalue(exchcap, :available_opening_quote)
    exch_initial_margin = _capvalue(exchcap, :initial_margin_quote)
    exch_maintenance_margin = _capvalue(exchcap, :maintenance_margin_quote)

    equity = exchcap.equity_quote > 0.0 ? exchcap.equity_quote : assetcap.equity_quote
    opening = assetcap.available_opening_quote
    if exch_opening > 0.0
        opening = min(opening, exch_opening)
    end
    opening = min(max(0.0, opening), equity)
    return (
        equity_quote=max(0.0, equity),
        available_opening_quote=opening,
        available_long_quote=opening,
        available_short_quote=opening,
        initial_margin_quote=exch_initial_margin,
        maintenance_margin_quote=exch_maintenance_margin,
        source=exch_equity > 0.0 ? string("Xch+", exchcap.source) : assetcap.source,
    )
end

function _fallbackaccountcapacity(xc::XchCache)
    balancesdf = balances(xc; ignoresmallvolume=false)
    assets = _assetssnapshot_from_balances_positions(xc, balancesdf; resolve_missing_prices=true)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    quotefree = 0.0
    if (:coin in propertynames(assets)) && (:free in propertynames(assets))
        for row in eachrow(assets)
            if uppercase(String(row.coin)) == quotecoin
                quotefree += max(0.0, (row.free))
            end
        end
    end
    equity = (:usdtvalue in propertynames(assets)) ? (sum(assets[!, :usdtvalue])) : quotefree
    equityc = max(0.0, equity)
    openingc = min(max(0.0, quotefree), equityc)
    return (
        equity_quote=equityc,
        available_opening_quote=openingc,
        source="Xch:portfolio_fallback",
    )
end

"""
Return exchange-concept account capacity snapshot in quote currency.

Fields:
- `equity_quote`: exchange-equity style net worth in quote terms
- `available_opening_quote`: side-agnostic conservative opening capacity
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

function _capacityfromassets(assetsdf::AbstractDataFrame)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    freequote = 0.0
    if (:coin in propertynames(assetsdf)) && (:free in propertynames(assetsdf))
        for row in eachrow(assetsdf)
            if uppercase(String(row.coin)) == quotecoin
                freequote += max(0.0, (row.free))
            end
        end
    end
    equity = (:usdtvalue in propertynames(assetsdf)) ? max(0.0, (sum(assetsdf[!, :usdtvalue]))) : freequote
    opening = min(max(0.0, freequote), equity)
    return (
        equity_quote=equity,
        available_opening_quote=opening,
        source="Xch:assets_snapshot",
    )
end

"Return true when the adapter cache is running simulation-owned bookkeeping."
function _issimulationassetcache(xc::XchCache)::Bool
    if timesimulation(xc) || (exchange(xc) == EXCHANGE_BYBITSIM)
        return true
    end
    return (xc.bc isa Bybit.BybitCache) && !isnothing(xc.bc.assets)
end

"Return the current account snapshot used by Trade loop orchestration."
function account_status(xc::XchCache; force_refresh::Bool=false, ttl_seconds::Int=5, balancesdf=nothing, positionsdf=nothing, assetsdf=nothing, require_holding_valuation::Bool=false)
    balancesdf = isnothing(balancesdf) ? balances(xc; ignoresmallvolume=false) : balancesdf
    assetsdf = if isnothing(assetsdf)
        _assetssnapshot_from_balances_positions(xc, balancesdf; positionsdf=positionsdf, resolve_missing_prices=require_holding_valuation || timesimulation(xc))
    else
        DataFrame(assetsdf; copycols=true)
    end
    assetcap = _capacityfromassets(assetsdf)
    exchcap = if _issimulationassetcache(xc)
        assetcap
    else
        accountcapacity(xc; force_refresh=force_refresh, ttl_seconds=ttl_seconds)
    end
    capacity = _mergecapacity(assetcap, exchcap)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    freequote = 0.0
    if (:coin in propertynames(assetsdf)) && (:free in propertynames(assetsdf))
        for row in eachrow(assetsdf)
            if uppercase(String(row.coin)) == quotecoin
                freequote += max(0.0, (row.free))
            end
        end
    end
    freequote = max(0.0, freequote)
    # bc.assets is Float32-backed; tolerate Float32 rounding noise (~1e-5 relative) that
    # accumulates across many fills/rebalances, while still catching genuine invariant
    # violations (which are orders of magnitude larger in practice).
    tolerance = max(1e-6, 1e-4 * abs(capacity.equity_quote))
    @assert freequote <= capacity.equity_quote + tolerance "account_status freequote=$(freequote) exceeds equity_quote=$(capacity.equity_quote) source=$(capacity.source)"
    freemargin = min(max(0.0, capacity.available_opening_quote), capacity.equity_quote)
    return (
        balances=balancesdf,
        assets=assetsdf,
        capacity=capacity,
        equity_quote=capacity.equity_quote,
        free_quote=freequote,
        free_margin_quote=freemargin,
        maintenance_margin_quote=capacity.maintenance_margin_quote,
    )
end

"Capture one coherent adapter account state and derive the Xch account status for the current tick."
function refreshaccountstatus!(xc::XchCache; ignoresmallvolume::Bool=false, require_holding_valuation::Bool=false)
    adaptersnapshot = accountsnapshot(xc.bc)
    balancesdf = isnothing(adaptersnapshot) ? _adapterbalances(xc) : _normalizeadapterbalances(adaptersnapshot.balances)
    _filterbalances!(xc, balancesdf; ignoresmallvolume=ignoresmallvolume)
    positionsdf = isnothing(adaptersnapshot) ? positionsnapshot(xc.bc) : adaptersnapshot.positions
    snapshotdt = isnothing(xc.currentdt) ? floor(Dates.now(Dates.UTC), Minute(1)) : xc.currentdt
    xc.mc[:exchange_balances_snapshot] = balancesdf
    xc.mc[:exchange_balances_snapshot_dt] = snapshotdt
    status = account_status(xc; force_refresh=true, ttl_seconds=0, balancesdf=balancesdf, positionsdf=positionsdf, require_holding_valuation=require_holding_valuation)
    return (status..., positions=positionsdf, datetime=snapshotdt)
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

"""Return the opposite position side's close lane info for `action`, or `needed=false` if none is held."""
function _oppositeclosestate(tradesdf::DataFrame, ix::Integer, action::Symbol)
    if action == :long_open
        return (needed=(tradesdf[ix, :sp_amount] > 0f0), close_id_col=:sc_id)
    elseif action == :short_open
        return (needed=(tradesdf[ix, :lp_amount] > 0f0), close_id_col=:lc_id)
    end
    return (needed=false, close_id_col=:lc_id)
end

function _rejectedrequest!(xc::XchCache, tradesdf::DataFrame, ix::Integer, action::Symbol, message::AbstractString)
    logged = log_trading_issue(xc, "Xch", message)
    if action == :long_open
        TSM.settrades_status!(tradesdf, ix, longopen, "rejected")
        TSM.settrades_msg!(tradesdf, ix, longopen, logged)
    elseif action == :long_close
        TSM.settrades_status!(tradesdf, ix, longclose, "rejected")
        TSM.settrades_msg!(tradesdf, ix, longclose, logged)
    elseif action == :short_open
        TSM.settrades_status!(tradesdf, ix, shortopen, "rejected")
        TSM.settrades_msg!(tradesdf, ix, shortopen, logged)
    else
        TSM.settrades_status!(tradesdf, ix, shortclose, "rejected")
        TSM.settrades_msg!(tradesdf, ix, shortclose, logged)
    end
    return logged
end

function _row_has_position_amount(tradesdf::DataFrame, ix::Integer)::Bool
    return (tradesdf[ix, :lp_amount] > 0f0) || (tradesdf[ix, :sp_amount] > 0f0)
end

function _row_position_increased(tradesdf::DataFrame, ix::Integer)::Bool
    if ix < lastindex(tradesdf, 1)
        return (tradesdf[ix, :lp_amount] < tradesdf[ix + 1, :lp_amount]) || (tradesdf[ix, :sp_amount] < tradesdf[ix + 1, :sp_amount])
    else
        return false
    end
end

function _carry_lastopentrade_from_previous!(tradesdf::DataFrame, ix::Integer)
    if !_row_has_position_amount(tradesdf, ix)
        TSM.settrades_lastopentrade!(tradesdf, ix, missing)
        return 
    end
    if !ismissing(tradesdf[ix, :lastopentrade])
        return 
    end
    for j in (ix - 1):-1:firstindex(tradesdf, 1)
        if _row_position_increased(tradesdf, j)
            TSM.settrades_lastopentrade!(tradesdf, ix, tradesdf[j + 1, :opentime])
            break
        end
        prev = tradesdf[j, :lastopentrade]
        if !ismissing(prev)
            TSM.settrades_lastopentrade!(tradesdf, ix, prev)
            break
        end
    end
    return 
end

"""
Carry `lol_pavg`/`sol_pavg` forward from the previous row while a position stays open.
Unlike `lcl_pavg`/`scl_pavg` (last close, valid only for the tick a close fills), the open
price must remain readable for as long as `lp_amount`/`sp_amount` stays positive - closes
and gains compilation both read it as the position's entry price, not just this tick's fill.
"""
function _carry_openpavg_from_previous!(tradesdf::DataFrame, ix::Integer)
    if ix <= 1
        return nothing
    end
    if (tradesdf[ix, :lp_amount] > 0f0) && (tradesdf[ix, :lol_pavg] <= 0f0)
        prevpavg = tradesdf[ix - 1, :lol_pavg]
        (prevpavg > 0f0) && TSM.settrades_last_pavg!(tradesdf, ix, longopen, prevpavg)
    end
    if (tradesdf[ix, :sp_amount] > 0f0) && (tradesdf[ix, :sol_pavg] <= 0f0)
        prevpavg = tradesdf[ix - 1, :sol_pavg]
        (prevpavg > 0f0) && TSM.settrades_last_pavg!(tradesdf, ix, shortopen, prevpavg)
    end
    return nothing
end

"Synchronize one trades row's exchange feedback columns from current order ids."
function order_status(xc::XchCache, tradesdf::DataFrame, ix::Integer; auditevent::Bool=true)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    _lane_orderid(v) = begin
        if ismissing(v) || isnothing(v)
            return nothing
        end
        s = strip(String(v))
        return (isempty(s) || (lowercase(s) == NO_ORDER_ID)) ? nothing : s
    end

    _isopenstatuslabel(s::AbstractString) = lowercase(strip(String(s))) in ("submitted", "new", "partiallyfilled", "untriggered", "open")

    # Both legs of a close bracket report their fill into the shared last-close columns, so
    # `lastidcol` records which leg produced it.
    for (idcol, stcol, filledcol, avgcol, msgcol, amountcol, poscol, lastidcol, laststcol) in [
        (:lo_id, :lo_status, :lol_filled, :lol_pavg, :lo_msg, :lo_amount, :lp_amount, :lol_id, :lol_status),
        (:lc_id, :lc_status, :lcl_filled, :lcl_pavg, :lc_msg, :lc_amount, :lp_amount, :lcl_id, :lcl_status),
        (:lcsl_id, :lcsl_status, :lcl_filled, :lcl_pavg, :lcsl_msg, :lc_amount, :lp_amount, :lcl_id, :lcl_status),
        (:so_id, :so_status, :sol_filled, :sol_pavg, :so_msg, :so_amount, :sp_amount, :sol_id, :sol_status),
        (:sc_id, :sc_status, :scl_filled, :scl_pavg, :sc_msg, :sc_amount, :sp_amount, :scl_id, :scl_status),
        (:scsl_id, :scsl_status, :scl_filled, :scl_pavg, :scsl_msg, :sc_amount, :sp_amount, :scl_id, :scl_status),
    ]
        oid = _lane_orderid(tradesdf[ix, idcol])

        # For a new row, lane id can be defaulted to `none`; reconcile from previous open lane state.
        # Skip this once this row already recorded its own close (normal fill or forced
        # liquidation): a row can be revisited many times while OHLCV data for this pair is
        # gapped (ensuretradesrow! reuses the same row until the next candle arrives), and
        # `ix-1` then reflects the state *before* this row's close, not the previous tick -
        # resurrecting that stale open id would clobber the close just recorded on this row.
        if isnothing(oid) && (ix > 1) && (String(tradesdf[ix, laststcol]) != "closed")
            previd = _lane_orderid(tradesdf[ix - 1, idcol])
            prevstatus_raw = tradesdf[ix - 1, stcol]
            @assert !ismissing(prevstatus_raw) && !isnothing(prevstatus_raw) "Schema violation: $(stcol) must be non-missing at ix=$(ix-1), pair=$(tradesdf[ix - 1, :pair]), opentime=$(tradesdf[ix - 1, :opentime])"
            prevstatus = String(prevstatus_raw)
            if !isnothing(previd) && _isopenstatuslabel(prevstatus)
                oid = previd
                TSM.settradesfield!(tradesdf, ix, idcol, previd)
                TSM.settradesfield!(tradesdf, ix, stcol, prevstatus)
                TSM.settradesfield!(tradesdf, ix, filledcol, tradesdf[ix - 1, filledcol])
                TSM.settradesfield!(tradesdf, ix, avgcol, tradesdf[ix - 1, avgcol])
                # no msg take over from previous row
                TSM.settradesfield!(tradesdf, ix, amountcol, tradesdf[ix - 1, amountcol])
                TSM.settradesfield!(tradesdf, ix, poscol, tradesdf[ix - 1, poscol])
            end
        end

        isnothing(oid) && continue

        info = getorder(xc, oid; auditevent=auditevent)
        if isnothing(info)
            TSM.settradesfield!(tradesdf, ix, stcol, "none")
            TSM.settradesfield!(tradesdf, ix, idcol, NO_ORDER_ID)
            continue
        end
        rawstatus = if hasproperty(info, :status)
            statusraw = info.status
            @assert !ismissing(statusraw) && !isnothing(statusraw) "Schema violation: adapter order status is missing for orderid=$(oid), lane=$(idcol), ix=$(ix), pair=$(tradesdf[ix, :pair])"
            String(statusraw)
        else
            "unknown"
        end
        status = normalize_order_status(xc.bc, rawstatus)
        TSM.settradesfield!(tradesdf, ix, stcol, status)
        if hasproperty(info, :baseqty) && hasproperty(info, :executedqty)
            executed = (info.executedqty)
            @assert !ismissing(executed) && !isnothing(executed) "Schema violation: adapter executedqty is missing for orderid=$(oid), lane=$(idcol), ix=$(ix), pair=$(tradesdf[ix, :pair])"
            TSM.settradesfield!(tradesdf, ix, filledcol, max(0.0, executed))
            # An open lane fill is what dates the position, independent of this tick's label.
            if (idcol in (:lo_id, :so_id)) && (executed > 0.0)
                TSM.settrades_lastopentrade!(tradesdf, ix, tradesdf[ix, :opentime])
            end

            # If the lane order closed this tick, materialize into position amounts immediately.
            # Portfolio snapshot reconciliation later in sync_latest_trades_rows! remains authoritative.
            if status == "closed"
                if idcol == :lo_id
                    TSM.settrades_lp_amount!(tradesdf, ix, max(tradesdf[ix, :lp_amount], (executed)))
                elseif idcol == :so_id
                    TSM.settrades_sp_amount!(tradesdf, ix, max(tradesdf[ix, :sp_amount], (executed)))
                end
            end
        end
        if hasproperty(info, :avgprice) && !ismissing(info.avgprice)
            TSM.settradesfield!(tradesdf, ix, avgcol, info.avgprice)
        end
        if hasproperty(info, :rejectreason)
            rrraw = info.rejectreason
            @assert !ismissing(rrraw) && !isnothing(rrraw) "Schema violation: adapter rejectreason is missing for orderid=$(oid), lane=$(idcol), ix=$(ix), pair=$(tradesdf[ix, :pair])"
            rr = String(rrraw)
            if !isempty(strip(rr)) && (uppercase(rr) != "NO ERROR")
                TSM.settradesfield!(tradesdf, ix, msgcol, log_trading_issue(xc, exchange(xc), rr))
            end
        end

        if status in ("closed", "rejected")
            TSM.settradesfield!(tradesdf, ix, lastidcol, oid)
            TSM.settradesfield!(tradesdf, ix, laststcol, status)
        end
        if status in ("closed", "cancelled", "rejected", "none")
            TSM.settradesfield!(tradesdf, ix, idcol, NO_ORDER_ID)
            if amountcol in propertynames(tradesdf)
                TSM.settradesfield!(tradesdf, ix, amountcol, 0f0)
            end
        end
    end
    return tradesdf
end

"""
Reflect one adapter-reported forced-close event (`drainliquidations!`) into the close lane
columns of one Trades row: the superseded resting close/stop order (if any) is cancelled
(`lc_id`/`sc_id`, `lc_status`/`sc_status`, `lc_amount`/`sc_amount`), while the last-lane
columns record the actual execution (`lcl_status`/`scl_status`, `lcl_filled`/`scl_filled`,
`lcl_pavg`/`scl_pavg`, `lcl_msg`/`scl_msg`).
"""
function _applyliquidationevent!(tradesdf::DataFrame, ix::Integer, ev::NamedTuple)::Nothing
    long = ev.positionside == :long
    if ev.hadpendingorder
        TSM.settradesfield!(tradesdf, ix, long ? :lc_id : :sc_id, NO_ORDER_ID)
        TSM.settradesfield!(tradesdf, ix, long ? :lc_status : :sc_status, "cancelled")
        TSM.settradesfield!(tradesdf, ix, long ? :lc_amount : :sc_amount, 0f0)
    end
    TSM.settradesfield!(tradesdf, ix, long ? :lcl_status : :scl_status, "closed")
    TSM.settradesfield!(tradesdf, ix, long ? :lcl_filled : :scl_filled, Float32(ev.qty))
    TSM.settradesfield!(tradesdf, ix, long ? :lcl_pavg : :scl_pavg, Float32(ev.price))
    TSM.settradesfield!(tradesdf, ix, long ? :lcl_msg : :scl_msg, String(ev.reason))
    return nothing
end

function _sync_basekeys(syncpairs, quotecoin::AbstractString)::Vector{String}
    if isnothing(syncpairs)
        return String[]
    end
    seen = Set{String}()
    bases = String[]
    for token in syncpairs
        candidate = uppercase(strip(String(token)))
        isempty(candidate) && continue
        if candidate in (quotecoin, "QUOTE")
            continue
        end
        pair = if occursin('/', candidate)
            candidate
        elseif endswith(candidate, quotecoin) && (length(candidate) > length(quotecoin))
            candidate
        else
            uppercase(String(candidate)) * quotecoin
        end
        bq = basequote(pair)
        isnothing(bq) && continue
        base = uppercase(String(bq.basecoin))
        base == quotecoin && continue
        if base in seen
            continue
        end
        push!(bases, base)
        push!(seen, base)
    end
    return bases
end

"Return deduplicated base keys from active prepared pair references."
function _sync_basekeys(xc::XchCache, pairrefs::AbstractVector{<:TradingPairRef})::Vector{String}
    seen = Set{String}()
    bases = String[]
    sizehint!(bases, length(pairrefs))
    for pairref in pairrefs
        base = tradingpairinfo(xc, pairref).basecoin
        if !(base in seen)
            push!(bases, base)
            push!(seen, base)
        end
    end
    return bases
end

function _sync_positions_by_coin(positionsdf)
    pos_by_coin = Dict{String, Tuple{Float32, Float32}}()
    if (:coin in propertynames(positionsdf)) && (:long_qty in propertynames(positionsdf)) && (:short_qty in propertynames(positionsdf))
        for row in eachrow(positionsdf)
            coin = uppercase(String(row.coin))
            pos_by_coin[coin] = (max(0f0, (row.long_qty)), max(0f0, (row.short_qty)))
        end
    end
    return pos_by_coin
end

"Index available long-only balance inventory once for the current account snapshot."
function _sync_balances_by_coin(balancesdf)
    balances_by_coin = Dict{String, Float32}()
    if _hascol(balancesdf, :coin) && _hascol(balancesdf, :free)
        for row in eachrow(balancesdf)
            balances_by_coin[uppercase(String(row.coin))] = max(0f0, row.free)
        end
    end
    return balances_by_coin
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
function sync_latest_trades_rows!(xc::XchCache, syncpairs=nothing; acct=nothing, positionsdf=nothing)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    bases_to_sync = if isnothing(syncpairs)
        String[uppercase(String(base)) for base in keys(xc.bases) if uppercase(String(base)) != quotecoin]
    elseif syncpairs isa AbstractVector{<:TradingPairRef}
        _sync_basekeys(xc, syncpairs)
    else
        _sync_basekeys(syncpairs, quotecoin)
    end

    for base in bases_to_sync
        @assert base in keys(xc.bases) "sync_latest_trades_rows! missing base=$(base) in xc.bases; addbase! and iterator-driven setcurrenttime! must prepare all synced bases"
    end

    acct = isnothing(acct) ? refreshaccountstatus!(xc; ignoresmallvolume=false, require_holding_valuation=false) : acct
    balancesdf = acct.assets
    posdf = if isnothing(positionsdf)
        acct.positions
    else
        positionsdf
    end
    pos_by_coin = _sync_positions_by_coin(posdf)
    balances_by_coin = _sync_balances_by_coin(balancesdf)

    pairkeys_by_base = Dict{String, String}()
    if syncpairs isa AbstractVector{<:TradingPairRef}
        for pairref in syncpairs
            pairinfo = tradingpairinfo(xc, pairref)
            pairkeys_by_base[pairinfo.basecoin] = pairinfo.pair
        end
    end

    liquidations_by_base = Dict{String, Vector{NamedTuple}}()
    for ev in drainliquidations!(xc.bc)
        bq = basequote(String(ev.symbol))
        isnothing(bq) && continue
        push!(get!(() -> NamedTuple[], liquidations_by_base, uppercase(String(bq.basecoin))), ev)
    end

    rowsbybase = Dict{String, NamedTuple}()

    for base in bases_to_sync
        pairkey = get!(pairkeys_by_base, base) do
            tradingpairkey(base, quotecoin)
        end
        currentdt = if base in keys(xc.bases)
            o = ohlcv(xc, base)
            odf = Ohlcv.dataframe(o)
            size(odf, 1) > 0 ? odf[Ohlcv.ix(o), :opentime] : (isnothing(xc.currentdt) ? xc.startdt : xc.currentdt)
        else
            isnothing(xc.currentdt) ? xc.startdt : xc.currentdt
        end

        # No candle advanced for this pair since the last time this function synced it (a
        # genuine data gap, e.g. a classifier-partition boundary or exchange downtime): treat
        # this tick as "no contact" for this pair and leave its resting orders/position
        # completely untouched rather than re-processing the same stale row - repeated
        # reprocessing can otherwise resurrect stale order state and corrupt the eventual
        # close price once data resumes.
        # A drained liquidation event must still be applied even while frozen, or it is lost -
        # drainliquidations! above already removed it from the shared adapter-side queue.
        if (get(xc.lastsyncedopentime, base, nothing) == currentdt) && !haskey(liquidations_by_base, base)
            continue
        end

        existing_rowix = TSM.tradesrowindex(xc.tsm, base, quotecoin, currentdt)
        if isnothing(existing_rowix)
            tdf_rowix = TSM.ensuretradesrow!(xc.tsm, base, quotecoin, currentdt)
            tdf = tdf_rowix.tradesdf
            rowix = tdf_rowix.rowix
        else
            tdf = TSM.trades(xc.tsm, tradingpairkey(base, quotecoin))
            rowix = existing_rowix
        end

        # OHLCV columns
        if base in keys(xc.bases)
            o = ohlcv(xc, base)
            odf = Ohlcv.dataframe(o)
            oix = Ohlcv.ix(o)
            if size(odf, 1) > 0 && 1 <= oix <= size(odf, 1)
                TSM.settrades_close!(tdf, rowix, odf[oix, :close])
                TSM.settrades_high!(tdf, rowix, odf[oix, :high])
                TSM.settrades_low!(tdf, rowix, odf[oix, :low])
            end
        end

        # Sync order statuses for all lanes
        order_status(xc, tdf, rowix; auditevent=false)

        # Reflect any forced closes (margin-call liquidation, exchange-side stop) since a
        # forced close never fills our own tracked resting order, so order_status alone
        # cannot report its execution.
        explicitliquidationsides = Set{Symbol}()
        for ev in get(liquidations_by_base, base, NamedTuple[])
            _applyliquidationevent!(tdf, rowix, ev)
            push!(explicitliquidationsides, ev.positionside)
        end

        prevlqty = rowix > 1 ? tdf[rowix - 1, :lp_amount] : 0f0
        prevsqty = rowix > 1 ? tdf[rowix - 1, :sp_amount] : 0f0

        # Position amounts from portfolio snapshot
        newlqty, newsqty = if haskey(pos_by_coin, base)
            pos_by_coin[base]
        elseif haskey(balances_by_coin, base)
            # Fallback when adapter positionsnapshot is unavailable.
            # No-liability policy: treat base inventory as long-only.
            (balances_by_coin[base], 0f0)
        else
            (tdf[rowix, :lp_amount], tdf[rowix, :sp_amount])
        end
        TSM.settrades_lp_amount!(tdf, rowix, newlqty)
        TSM.settrades_sp_amount!(tdf, rowix, newsqty)

        # A position that vanished without our own tracked close order having closed it
        # (and without an adapter-reported liquidation event) was force-closed by the
        # exchange; approximate its execution with the current mark price.
        markprice = tdf[rowix, :close]
        if !(:long in explicitliquidationsides) && (prevlqty > 0f0) && (newlqty == 0f0) && (String(tdf[rowix, :lc_status]) != "closed") && (markprice > 0f0)
            _applyliquidationevent!(tdf, rowix, (positionside=:long, qty=prevlqty, price=markprice, hadpendingorder=String(tdf[rowix, :lc_id]) != NO_ORDER_ID, reason="liquidation"))
        end
        if !(:short in explicitliquidationsides) && (prevsqty > 0f0) && (newsqty == 0f0) && (String(tdf[rowix, :sc_status]) != "closed") && (markprice > 0f0)
            _applyliquidationevent!(tdf, rowix, (positionside=:short, qty=prevsqty, price=markprice, hadpendingorder=String(tdf[rowix, :sc_id]) != NO_ORDER_ID, reason="liquidation"))
        end

        _carry_lastopentrade_from_previous!(tdf, rowix)
        _carry_openpavg_from_previous!(tdf, rowix)

        # Account snapshot columns
        TSM.settrades_equity!(tdf, rowix, acct.equity_quote)
        TSM.settrades_freemargin!(tdf, rowix, acct.free_margin_quote)
        TSM.settrades_freequote!(tdf, rowix, acct.free_quote)

        rowsbybase[base] = (tradesdf=tdf, rowix=rowix)
        xc.lastsyncedopentime[base] = currentdt
    end

    return rowsbybase
end

"""
Maintain the resting close bracket of one position side from the Trades row.

Xch does not read the trade label for closes: `<lane>_amount > 0` is the request to hold a
resting close, and a positive `<lane>sl_limit` upgrades it into a bracket whose second leg
is the protective stop. Both legs cover the same quantity and share one exchange-side
reservation, so the stop leg carries no amount of its own. `<lane>_limit == 0` keeps the
close leg at an adaptive maker price, while `<lane>sl_limit == 0` means no stop is wanted.
"""
function _ensureclosebracketside!(xc::XchCache, tradesdf::DataFrame, ix::Integer, symbol::AbstractString, positionside::Symbol; pairref::Union{Nothing, TradingPairRef}=nothing, pairinfo=nothing)
    long = positionside == :long
    posamount = tradesdf[ix, long ? :lp_amount : :sp_amount]
    closeamount = tradesdf[ix, long ? :lc_amount : :sc_amount]
    ((posamount > 0f0) && (closeamount > 0f0)) || return nothing
    qty = min(closeamount, posamount)
    qty > 0f0 || return nothing

    minqty = if isnothing(pairinfo)
        bq = _pairfromtradesrow(tradesdf, ix)
        minimumbasequantity(xc, bq.basecoin, tradesdf[ix, :close])
    else
        1.01 * max(pairinfo.minbaseqty, pairinfo.minquoteqty / tradesdf[ix, :close])
    end
    closeaction = long ? :long_close : :short_close
    if isnothing(minqty)
        _rejectedrequest!(xc, tradesdf, ix, closeaction, "minimum base quantity unavailable")
        return nothing
    end
    if qty < minqty
        _rejectedrequest!(xc, tradesdf, ix, closeaction, "base amount below minimum quantity")
        return nothing
    end

    _laneorderid(raw) = let sid = strip(String(raw))
        (isempty(sid) || (lowercase(sid) == NO_ORDER_ID)) ? nothing : sid
    end
    _resolvedid(raw) = raw isa AbstractString ? String(raw) : String(getproperty(raw, :orderid))

    closeidcol = long ? :lc_id : :sc_id
    closestcol = long ? :lc_status : :sc_status
    closelane = long ? "lc" : "sc"
    closelimit = _rowlimitprice(tradesdf[ix, long ? :lc_limit : :sc_limit])
    oid = upsertcloseorder!(xc.bc, symbol, positionside, qty, closelimit; existing_orderid=_laneorderid(tradesdf[ix, closeidcol]), maker=true, reduceonly=true, lane=closelane, pairref=pairref)
    if isnothing(oid)
        _rejectedrequest!(xc, tradesdf, ix, long ? :long_close : :short_close, "exchange returned no close order id")
    else
        tradesdf[ix, closeidcol] = _resolvedid(oid)
        tradesdf[ix, closestcol] = "submitted"
    end

    stopidcol = long ? :lcsl_id : :scsl_id
    stopstcol = long ? :lcsl_status : :scsl_status
    stoplimit = tradesdf[ix, long ? :lcsl_limit : :scsl_limit]
    stoplimit > 0f0 || return nothing
    soid = upsertcloseorder!(xc.bc, symbol, positionside, qty, Float32(stoplimit); existing_orderid=_laneorderid(tradesdf[ix, stopidcol]), maker=true, reduceonly=true, lane=long ? "lcsl" : "scsl", pairref=pairref)
    if isnothing(soid)
        (verbosity >= 1) && @warn "stop-loss bracket leg rejected" symbol positionside
        return nothing
    end
    tradesdf[ix, stopidcol] = _resolvedid(soid)
    tradesdf[ix, stopstcol] = "submitted"
    return nothing
end

"Maintain the resting close bracket of both position sides from the Trades row."
function _ensureclosebracket!(xc::XchCache, tradesdf::DataFrame, ix::Integer, symbol::AbstractString; pairref::Union{Nothing, TradingPairRef}=nothing, pairinfo=nothing)
    _ensureclosebracketside!(xc, tradesdf, ix, symbol, :long; pairref=pairref, pairinfo=pairinfo)
    _ensureclosebracketside!(xc, tradesdf, ix, symbol, :short; pairref=pairref, pairinfo=pairinfo)
    return nothing
end

"Evaluate and execute one row-level order request from the Trades DataFrame."
function process_order_request(xc::XchCache, tradesdf::DataFrame, ix::Integer; pairref::Union{Nothing, TradingPairRef}=nothing)
    @assert 1 <= ix <= nrow(tradesdf) "ix=$(ix) out of bounds for trades rows=$(nrow(tradesdf))"

    pairinfo = isnothing(pairref) ? nothing : _preparedtradingpairinfo(xc, pairref)
    if isnothing(pairinfo)
        pair = _pairfromtradesrow(tradesdf, ix)
        base = pair.basecoin
        quotecoin = pair.quotecoin
        symbol = symboltoken(xc, base, quotecoin)
    else
        base = pairinfo.basecoin
        quotecoin = pairinfo.quotecoin
        symbol = pairinfo.symbol
    end
    # Xch reads amounts and limits only; trade labels stay Trade/TradingStrategy vocabulary.
    _ensureclosebracket!(xc, tradesdf, ix, symbol; pairref=pairref, pairinfo=pairinfo)
    longopenamount = tradesdf[ix, :lo_amount]
    shortopenamount = tradesdf[ix, :so_amount]
    @assert !((longopenamount > 0f0) && (shortopenamount > 0f0)) "opposite open requests in one row: lo_amount=$(longopenamount), so_amount=$(shortopenamount), pair=$(tradesdf[ix, :pair]), opentime=$(tradesdf[ix, :opentime])"
    action = if longopenamount > 0f0
        :long_open
    elseif shortopenamount > 0f0
        :short_open
    else
        :none
    end
    action == :none && return (accepted=false, action=:none, reason="no open amount requested")

    limitcol = action == :long_open ? :lo_limit : :so_limit
    orderamountcol = action == :long_open ? :lo_amount : :so_amount
    idcol = action == :long_open ? :lo_id : :so_id
    stcol = action == :long_open ? :lo_status : :so_status
    filledcol = action == :long_open ? :lol_filled : :sol_filled
    avgcol = action == :long_open ? :lol_pavg : :sol_pavg

    limitprice = _rowlimitprice(tradesdf[ix, limitcol])
    orderamount = tradesdf[ix, orderamountcol]

    if !(orderamount > 0f0)
        _rejectedrequest!(xc, tradesdf, ix, action, "amount is not positive")
        return (accepted=false, action=action, reason="amount_not_positive")
    end

    minqty = isnothing(pairinfo) ? minimumbasequantity(xc, base, tradesdf[ix, :close]) : 1.01 * max(pairinfo.minbaseqty, pairinfo.minquoteqty / tradesdf[ix, :close])
    if isnothing(minqty)
        _rejectedrequest!(xc, tradesdf, ix, action, "minimum base quantity unavailable")
        return (accepted=false, action=action, reason="minimum_qty_unavailable")
    end
    if orderamount < minqty
        _rejectedrequest!(xc, tradesdf, ix, action, "base amount below minimum quantity")
        # _rejectedrequest!(xc, tradesdf, ix, action, "base amount=$(orderamount) below minimum base qty $(minqty) for pair=$(base)-$(quotecoin)")
        return (accepted=false, action=action, reason="below_minimum_qty")
    end

    side = _ordersidefromaction(action)

    _orderid(raw)::Union{Nothing, String} = begin
        if isnothing(raw)
            return nothing
        elseif raw isa AbstractString
            return String(raw)
        elseif hasproperty(raw, :orderid)
            return String(getproperty(raw, :orderid))
        else
            return nothing
        end
    end

    _existing_orderid(raw)::Union{Nothing, String} = begin
        sid = strip(String(raw))
        return (isempty(sid) || (lowercase(sid) == NO_ORDER_ID)) ? nothing : sid
    end

    oid = nothing
    if action in [:long_open, :short_open]
        # The one-position-at-a-time rule: an opposite position must be fully closed before
        # this side opens. `_ensureclosebracket!` above already keeps that close resting;
        # just gate on the lane id it maintains instead of resubmitting a second close here.
        opposite = _oppositeclosestate(tradesdf, ix, action)
        closeoid = opposite.needed ? _existing_orderid(tradesdf[ix, opposite.close_id_col]) : nothing
        if !isnothing(closeoid) && _orderstillopen(xc, closeoid)
            TSM.settrades_msg!(tradesdf, ix, action == :long_open ? longopen : shortopen, "awaiting opposite close before open")
            return (accepted=false, action=action, reason="awaiting_close")
        end

        existing_openid = _existing_orderid(tradesdf[ix, idcol])
        openside = action == :long_open ? :long : :short
        oid = upsertopenorder!(xc.bc, symbol, openside, orderamount, limitprice; existing_orderid=existing_openid, maker=true, reduceonly=false)
        if isnothing(oid)
            _rejectedrequest!(xc, tradesdf, ix, action, "exchange returned no open order id")
            return (accepted=false, action=action, reason="missing_open_orderid")
        end
        oid = _orderid(oid)
        @assert !isnothing(oid) "open order result must provide orderid for action=$(action), pair=$(base)-$(quotecoin)"
        tradesdf[ix, idcol] = oid
        tradesdf[ix, stcol] = "submitted"
        if !isnothing(closeoid)
            _ = directsequence!(xc.bc, closeoid, oid)
        end
    else # no action
        return (accepted=true, action=action, reason="no_action")
    end

    return (accepted=true, action=action, orderid=oid)
end

function _adapterbalances(xc::XchCache)::DataFrame
    return _normalizeadapterbalances(balances(xc.bc))
end

"Normalize one adapter balances response into Xch's canonical balance schema."
function _normalizeadapterbalances(bdf)::DataFrame
    if isnothing(bdf)
        return DataFrame()
    end
    cols = propertynames(bdf)

    # Normalize side-lane adapter balances (coin/side/free/locked) to the
    # canonical Xch balances schema with explicit short exposure. Reads `bdf` directly
    # (no upfront copy) since only `Xch.balances` calls this, and it already copies the
    # result again before any in-place filtering.
    if (:side in cols) && (:coin in cols) && (:free in cols) && (:locked in cols)
        bycoin = Dict{String, Tuple{Float32, Float32, Float32}}()
        for row in eachrow(bdf)
            coin = uppercase(String(row.coin))
            side = lowercase(String(row.side))
            freev = max(0f0, (row.free))
            lockedv = max(0f0, (row.locked))
            prev = get(bycoin, coin, (0f0, 0f0, 0f0))
            if side == "short"
                bycoin[coin] = (prev[1], prev[2], prev[3] + freev + lockedv)
            else
                bycoin[coin] = (prev[1] + freev, prev[2] + lockedv, prev[3])
            end
        end
        coins = collect(keys(bycoin))
        return DataFrame(
            coin=coins,
            free=Float32[bycoin[c][1] for c in coins],
            locked=Float32[bycoin[c][2] for c in coins],
            short=Float32[bycoin[c][3] for c in coins],
        )
    end

    return bdf
end

function _filterbalances!(xc::XchCache, bdf::DataFrame; ignoresmallvolume::Bool=true)::DataFrame
    if (size(bdf, 1) > 0) && ignoresmallvolume
        delrows = []
        for ix in eachindex(bdf[!, :coin])
            if bdf[ix, :coin] != EnvConfig.pairquote
                sym = symboltoken(bdf[ix, :coin])
                syminfo = minimumqty(xc, sym)
                shortv = _hascol(bdf, :short) ? bdf[ix, :short] : 0f0
                if !validsymbol(xc, sym) || ((abs(bdf[ix, :free]) + abs(bdf[ix, :locked]) + abs(shortv)) < 1.01 * syminfo.minbaseqty) # 1% more to avoid issues by rounding errors
                    push!(delrows, ix)
                end
            end
        end
        deleteat!(bdf, delrows)
    end
    return bdf
end

"Returns a DataFrame[:coin, :locked, :free, :short] of wallet/portfolio balances"
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
                    err isa InterruptException && rethrow(err)
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
    !(:short in propertynames(portfoliodf)) && (portfoliodf[!, :short] = fill(0f0, nrow(portfoliodf)))
    # Value is net exposure in USDT with explicit short quantity subtraction.
    portfoliodf.usdtvalue = (portfoliodf.free .+ portfoliodf.locked .- portfoliodf.short) .* portfoliodf.usdtprice
    if ignoresmallvolume
        delrows = []
        for ix in eachindex(portfoliodf[!, :coin])
            coin = String(portfoliodf[ix, :coin])
            minbasequant = minimumbasequantity(xc, coin, portfoliodf[ix, :usdtprice])
            is_quotecoin = (uppercase(coin) == uppercase(EnvConfig.pairquote)) || (coin in quotecoins)
            if !is_quotecoin && (isnothing(minbasequant) || ((abs(portfoliodf[ix, :free]) + abs(portfoliodf[ix, :locked]) + abs(portfoliodf[ix, :short])) < minbasequant))
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
        push!(adf, (coin = coin, free = 0f0, locked = 0f0, marginfree = 0f0, marginlocked = 0f0, accruedinterest = 0f0))
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


_emptyassets()::DataFrame = DataFrame(coin=String31[], free=Float32[], locked=Float32[], marginfree=Float32[], marginlocked=Float32[], accruedinterest=Float32[])

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

#endregion bookkeeping

end  # of module
