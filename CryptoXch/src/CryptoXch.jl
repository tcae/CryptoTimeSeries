# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch

using Dates, DataFrames, DataAPI, JDF, CSV, Logging, InlineStrings, UUIDs
using Bybit, EnvConfig, KrakenFutures, KrakenSpot, Ohlcv, TradeLog
import Ohlcv: intervalperiod

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

@enum Sidefactor buy=1 sell=-1 invaid = 0
@enum SimMode nosimulation bybitsim

"""
Exchange operation roles used for routing.
- `data_exchange`: source of OHLCV/market data (e.g. Bybit)
- `trade_exchange_spot`: target for spot order placement (e.g. KrakenSpot)
- `trade_exchange_futures`: target for futures order placement (e.g. KrakenFutures)
"""
@enum ExchangeRole data_exchange trade_exchange_spot trade_exchange_futures

"""
Holds the exchange name and authentication alias for one exchange role.
"""
struct ExchangeRouteEntry
    exchange::String
    authname::Union{Nothing, String}
end

"""
Maps each `ExchangeRole` to an `ExchangeRouteEntry`.
Roles that are not explicitly configured fall back to the `XchCache.exchange`.
"""
struct ExchangeRouting
    routes::Dict{ExchangeRole, ExchangeRouteEntry}
    ExchangeRouting() = new(Dict{ExchangeRole, ExchangeRouteEntry}())
end

"Set a role mapping in an `ExchangeRouting`."
function setrole!(routing::ExchangeRouting, role::ExchangeRole, exchange::AbstractString, authname::Union{Nothing, AbstractString}=nothing)
    if !isnothing(authname)
        throw(ArgumentError("setrole! authname is deprecated and no longer needed. Configure exactly one auth tuple per exchange in auth.json."))
    end
    routing.routes[role] = ExchangeRouteEntry(_normalizeexchange(String(exchange)), isnothing(authname) ? nothing : String(authname))
    return routing
end

"Return the exchange name for a role, falling back to `default_exchange` if not configured."
function _routeexchange(routing::ExchangeRouting, role::ExchangeRole, default_exchange::AbstractString)::String
    haskey(routing.routes, role) ? routing.routes[role].exchange : String(default_exchange)
end

const EXCHANGE_BYBIT::String = "Bybit"
const EXCHANGE_BYBITSIM::String = "BybitSim"
const EXCHANGE_KRAKENFUTURES::String = "KrakenFutures"
const EXCHANGE_KRAKENSPOT::String = "KrakenSpot"

function _normalizeexchange(exchange::AbstractString)::String
    ex = lowercase(strip(exchange))
    if ex == lowercase(EXCHANGE_BYBIT)
        return EXCHANGE_BYBIT
    elseif ex == lowercase(EXCHANGE_BYBITSIM)
        return EXCHANGE_BYBITSIM
    elseif ex == lowercase(EXCHANGE_KRAKENFUTURES)
        return EXCHANGE_KRAKENFUTURES
    elseif ex == lowercase(EXCHANGE_KRAKENSPOT)
        return EXCHANGE_KRAKENSPOT
    end
    throw(ArgumentError("unsupported exchange=$(exchange), supported=[$(EXCHANGE_BYBIT), $(EXCHANGE_BYBITSIM), $(EXCHANGE_KRAKENFUTURES), $(EXCHANGE_KRAKENSPOT)]"))
end

"Return the EnvConfig coin-folder token for one normalized exchange name."
function _coinsfoldertoken(exchange::AbstractString)::String
    ex = _normalizeexchange(exchange)
    if ex == EXCHANGE_BYBITSIM
        return "bybit"
    end
    return lowercase(replace(ex, r"[^A-Za-z0-9]+" => ""))
end

"Update EnvConfig coin root to coins_<exchange> based on active data exchange routing."
function _setexchangecoinspath!(xc)::String
    data_ex = _routeexchange(xc.routing, data_exchange, xc.exchange)
    return EnvConfig.setcoinspath!("coins_" * _coinsfoldertoken(data_ex))
end

function _exchangeModule(exchange::AbstractString)
    ex = _normalizeexchange(exchange)
    if ex == EXCHANGE_BYBIT
        return Bybit
    elseif ex == EXCHANGE_BYBITSIM
        return Bybit
    elseif ex == EXCHANGE_KRAKENFUTURES
        return KrakenFutures
    end
    return KrakenSpot
end

_exchangeemptyorders(exchange::AbstractString)::DataFrame = _exchangeModule(exchange).emptyorders()

function _authfromname(exchange::AbstractString)
    ex = _normalizeexchange(exchange)
    # Kraken adapters require exchange-specific credentials and should always
    # resolve auth tuples constrained by exchange.
    if ex == EXCHANGE_KRAKENSPOT || ex == EXCHANGE_KRAKENFUTURES
        return EnvConfig.Authentication(nothing; exchange=ex)
    elseif ex == EXCHANGE_BYBIT
        # Bybit keeps legacy global authorization behavior.
        return EnvConfig.authorization
    end
    return nothing
end

"""
Emit one final private-call diagnostics summary for active Kraken adapters.
Safe to call during shutdown; exchanges without private-call counters are skipped.
"""
function log_private_call_summary!(xc)
    exchanges = Set{String}()
    push!(exchanges, _normalizeexchange(xc.exchange))
    push!(exchanges, _routeexchange(xc.routing, trade_exchange_spot, xc.exchange))
    push!(exchanges, _routeexchange(xc.routing, trade_exchange_futures, xc.exchange))
    for ex in exchanges
        if ex == EXCHANGE_KRAKENSPOT
            KrakenSpot.log_private_call_summary!()
        elseif ex == EXCHANGE_KRAKENFUTURES
            KrakenFutures.log_private_call_summary!()
        end
    end
    return nothing
end

function _assertsimmodesupported(exchange::AbstractString, simmode::SimMode)
    ex = _normalizeexchange(exchange)
    if simmode == bybitsim && !(ex == EXCHANGE_BYBIT || ex == EXCHANGE_BYBITSIM)
        throw(ArgumentError("simmode=$(simmode) is only supported for $(EXCHANGE_BYBIT)/$(EXCHANGE_BYBITSIM), got exchange=$(ex)"))
    end
end

function _exchangecache(exchange::AbstractString, simmode::SimMode)
    ex = _normalizeexchange(exchange)
    _assertsimmodesupported(ex, simmode)
    auth = _authfromname(ex)
    publickey = isnothing(auth) ? "" : String(auth.key)
    secretkey = isnothing(auth) ? "" : String(auth.secret)
    if ex == EXCHANGE_BYBIT
        bc = Bybit.BybitCache(simmode == bybitsim, publickey, secretkey)
        (simmode == bybitsim) && Bybit._init_simulation!(bc)
        return bc
    elseif ex == EXCHANGE_BYBITSIM
        bc = Bybit.BybitCache(false, publickey, secretkey)
        Bybit._init_simulation!(bc)
        return bc
    elseif ex == EXCHANGE_KRAKENFUTURES
        return KrakenFutures.KrakenFuturesCache(publickey=publickey, secretkey=secretkey)
    end
    return KrakenSpot.KrakenSpotCache(publickey=publickey, secretkey=secretkey)
end

mutable struct XchCache
    orders  # ::DataFrame
    closedorders  # ::DataFrame
    assets  # :: DataFrame
    bases  # ::Dict{String, Ohlcv.OhlcvData}
    bc  # exchange specific cache, e.g. Bybit.BybitCache or KrakenSpot.KrakenSpotCache
    exchange::String
    authname::Union{Nothing, String}
    feerate  # 0.001 = 0.1% maker/taker fee by default  #TODO store exchange info and account fee rate and use it in offline backtest simulation
    startdt::Dates.DateTime
    currentdt::Union{Nothing, Dates.DateTime}  # current back testing time
    enddt::Union{Nothing, Dates.DateTime}  # end time back testing; nothing == request life data without defined termination
    mnemonic  # String or nothing
    mc::Dict # MC = module constants
    routing::ExchangeRouting   # per-role exchange/auth overrides; empty = all ops go to `exchange`/`bc`
    routecaches::Dict{String, Any}  # keyed by exchange name; lazily populated adapter caches for routing
    function XchCache(;startdt::DateTime=Dates.now(UTC), enddt=nothing, mnemonic=nothing, exchange::String=EXCHANGE_BYBIT, authname::Union{Nothing, AbstractString}=nothing)
        startdt = floor(startdt, Minute(1))
        enddt = isnothing(enddt) ? nothing : floor(enddt, Minute(1))
        # simmode = bybitsim # simulation mode with adapter-backed trading + deterministic offline market data
        # simmode = nosimulation # uses production mode of Bybit without any exchange simulation
        if !isnothing(authname)
            throw(ArgumentError("XchCache authname is deprecated and no longer needed. Configure exactly one auth tuple per exchange in auth.json and pass only exchange."))
        end
        exchange = _normalizeexchange(exchange)
        authname = nothing
        simmode = if EnvConfig.configmode == production
            nosimulation
        else
            bybitsim
        end
        xc = new(_emptyorders(exchange), _emptyorders(exchange), _emptyassets(), Dict(), _exchangecache(exchange, simmode), exchange, authname, 0.001, startdt, nothing, enddt, mnemonic, Dict(), ExchangeRouting(), Dict{String, Any}())
        xc.mc[:simmode] = simmode
        _setexchangecoinspath!(xc)
        if hasproperty(xc.bc, :syminfodf) && !isnothing(xc.bc.syminfodf)
            for row in eachrow(xc.bc.syminfodf)
                setsymbolinfocache!(xc, row.symbol, (
                    symbol=String(row.symbol),
                    status=String(row.status),
                    basecoin=String(row.basecoin),
                    quotecoin=String(row.quotecoin),
                    ticksize=Float32(row.ticksize),
                    baseprecision=Float32(row.baseprecision),
                    quoteprecision=Float32(row.quoteprecision),
                    minbaseqty=Float32(row.minbaseqty),
                    minquoteqty=Float32(row.minquoteqty),
                ))
            end
        end
        xc.mc[:tradelog_run_id] = get(ENV, "CTS_RUN_ID", string(uuid4()))
        return xc
    end
end

function tradelogrunmode(xc::XchCache)::String
    return xc.mc[:simmode] == nosimulation ? "live" : "simulation"
end

function tradelogrunid(xc::XchCache)::String
    if !haskey(xc.mc, :tradelog_run_id)
        # Migrate lazily from old key when available.
        if haskey(xc.mc, :audit_run_id)
            xc.mc[:tradelog_run_id] = String(xc.mc[:audit_run_id])
        else
            xc.mc[:tradelog_run_id] = get(ENV, "CTS_RUN_ID", string(uuid4()))
        end
    end
    return String(xc.mc[:tradelog_run_id])
end

# Backward-compatible aliases for legacy callers.
auditrunmode(xc::XchCache)::String = tradelogrunmode(xc)
auditrunid(xc::XchCache)::String = tradelogrunid(xc)

exchange(xc::XchCache)::String = xc.exchange
authname(xc::XchCache) = xc.authname
_exchangeModule(xc::XchCache) = _exchangeModule(xc.exchange)

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

    mod = _routedModule(xc, data_exchange)
    if isdefined(mod, :marketdataheartbeats)
        moduledict = try
            getproperty(mod, :marketdataheartbeats)(_routedbc(xc, data_exchange))
        catch
            try
                getproperty(mod, :marketdataheartbeats)()
            catch
                Dict{String, DateTime}()
            end
        end
        for (sym, dt) in moduledict
            key = uppercase(String(sym))
            moddt = DateTime(dt)
            prev = get(localmap, key, nothing)
            if isnothing(prev) || (moddt > DateTime(prev))
                localmap[key] = moddt
            end
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
        moduledt = nothing
        mod = _routedModule(xc, data_exchange)
        if isdefined(mod, :marketdataheartbeat)
            try
                moduledt = getproperty(mod, :marketdataheartbeat)(_routedbc(xc, data_exchange), key)
            catch
                try
                    moduledt = getproperty(mod, :marketdataheartbeat)(key)
                catch
                    moduledt = nothing
                end
            end
        end
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
    moduledt = nothing
    mod = _routedModule(xc, data_exchange)
    if isdefined(mod, :marketdataheartbeat)
        try
            moduledt = getproperty(mod, :marketdataheartbeat)(_routedbc(xc, data_exchange))
        catch
            try
                moduledt = getproperty(mod, :marketdataheartbeat)()
            catch
                moduledt = nothing
            end
        end
    end

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

function _ensurewschannel!(xc::XchCache, channel_key::Symbol, role::ExchangeRole, fn_name::Symbol)
    haskey(xc.mc, channel_key) && return xc.mc[channel_key]
    mod = _routedModule(xc, role)
    if !isdefined(mod, fn_name)
        xc.mc[channel_key] = nothing
        return nothing
    end
    fn = getproperty(mod, fn_name)
    bc = _routedbc(xc, role)
    ch = try
        fn(bc)
    catch err
        if (err isa MethodError) && (getproperty(err, :f) === fn)
            try
                fn()
            catch err_fallback
                (verbosity >= 1) && @warn "failed to start websocket channel" fn=String(fn_name) role=Symbol(role) error=sprint(showerror, err_fallback)
                nothing
            end
        else
            (verbosity >= 1) && @warn "failed to start websocket channel" fn=String(fn_name) role=Symbol(role) error=sprint(showerror, err)
            nothing
        end
    end
    xc.mc[channel_key] = ch
    return ch
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

function _refreshwsstreams!(xc::XchCache)
    if _wsenabled(xc, :ws_orders_enabled, false)
        ch = _ensurewschannel!(xc, :ws_orders_channel, trade_exchange_spot, :ws_orders)
        _drainwschannel!(ch)
    end
    if _wsenabled(xc, :ws_balances_enabled, false)
        ch = _ensurewschannel!(xc, :ws_balances_channel, trade_exchange_spot, :ws_balances)
        _drainwschannel!(ch)
    end
    return nothing
end

function _adapterwsdfsnapshot(xc::XchCache, role::ExchangeRole, fn_name::Symbol)
    _refreshwsstreams!(xc)
    mod = _routedModule(xc, role)
    if !isdefined(mod, fn_name)
        return DataFrame()
    end
    return try
        getproperty(mod, fn_name)(_routedbc(xc, role))
    catch
        try
            getproperty(mod, fn_name)()
        catch
            DataFrame()
        end
    end
end

function _adapterwsheartbeat(xc::XchCache, role::ExchangeRole, fn_name::Symbol)
    _refreshwsstreams!(xc)
    mod = _routedModule(xc, role)
    if !isdefined(mod, fn_name)
        return nothing
    end
    return try
        getproperty(mod, fn_name)(_routedbc(xc, role))
    catch
        try
            getproperty(mod, fn_name)()
        catch
            nothing
        end
    end
end

"Return latest adapter websocket order snapshot (canonical normalized open-order rows when available)."
function wsordersnapshot(xc::XchCache)::DataFrame
    return _adapterwsdfsnapshot(xc, trade_exchange_spot, :wsordersnapshot)
end

"Return latest adapter websocket balances snapshot (canonical normalized balance rows when available)."
function wsbalancessnapshot(xc::XchCache)::DataFrame
    return _adapterwsdfsnapshot(xc, trade_exchange_spot, :wsbalancessnapshot)
end

"Return latest adapter websocket order heartbeat timestamp when available."
function wsordersheartbeat(xc::XchCache)
    return _adapterwsheartbeat(xc, trade_exchange_spot, :wsordersheartbeat)
end

"Return latest adapter websocket balances heartbeat timestamp when available."
function wsbalancesheartbeat(xc::XchCache)
    return _adapterwsheartbeat(xc, trade_exchange_spot, :wsbalancesheartbeat)
end

"""
Return the adapter cache for the given `role`, using the routing config when available.
Falls back to `xc.bc` (the primary adapter) when no role override is configured.
"""
function _routedbc(xc::XchCache, role::ExchangeRole)
    if isempty(xc.routing.routes) || !haskey(xc.routing.routes, role)
        return xc.bc  # no routing configured for this role — use primary adapter
    end
    entry = xc.routing.routes[role]
    # lazily build and cache the adapter for this exchange
    if !haskey(xc.routecaches, entry.exchange)
        if !isnothing(entry.authname)
            throw(ArgumentError("route-level authname is deprecated and no longer needed. Configure exactly one auth tuple per exchange in auth.json."))
        end
        xc.routecaches[entry.exchange] = _exchangecache(entry.exchange, xc.mc[:simmode])
    end
    return xc.routecaches[entry.exchange]
end

"Return the exchange module for the given `role`, with routing fallback."
function _routedModule(xc::XchCache, role::ExchangeRole)
    if isempty(xc.routing.routes) || !haskey(xc.routing.routes, role)
        return _exchangeModule(xc.exchange)
    end
    return _exchangeModule(xc.routing.routes[role].exchange)
end

"""
Configure exchange role routing on an `XchCache`.

Example — Bybit for data, KrakenSpot for spot trading, KrakenFutures for futures:
```julia
setrole!(xc, data_exchange, CryptoXch.EXCHANGE_BYBIT)
setrole!(xc, trade_exchange_spot, CryptoXch.EXCHANGE_KRAKENSPOT)
setrole!(xc, trade_exchange_futures, CryptoXch.EXCHANGE_KRAKENFUTURES)
```
"""
function setrole!(xc::XchCache, role::ExchangeRole, exchange::AbstractString, authname::Union{Nothing, AbstractString}=nothing)
    if !isnothing(authname)
        throw(ArgumentError("setrole! authname is deprecated and no longer needed. Configure exactly one auth tuple per exchange in auth.json."))
    end
    setrole!(xc.routing, role, exchange, authname)
    # Evict stale cached adapter so it is rebuilt with the new auth on next use.
    delete!(xc.routecaches, _normalizeexchange(String(exchange)))
    _setexchangecoinspath!(xc)
    return xc
end

_exchangeservertime(xc::XchCache) = _exchangeModule(xc).servertime(xc.bc)

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
    bc = _routedbc(xc, data_exchange)
    if !isnothing(bc)
        row = _routedModule(xc, data_exchange).symbolinfo(bc, symbol)
        if !isnothing(row)
            # Populate / refresh local cache from live data
            nt = (
                symbol        = symbol,
                status        = string(row.status),
                basecoin      = string(row.basecoin),
                quotecoin     = string(row.quotecoin),
                ticksize      = Float32(row.ticksize),
                baseprecision = Float32(row.baseprecision),
                quoteprecision = Float32(row.quoteprecision),
                minbaseqty    = Float32(row.minbaseqty),
                minquoteqty   = Float32(row.minquoteqty),
            )
            _syminfocache(xc)[symbol] = nt
            return row  # keep returning the original DataFrameRow for backward compat
        end
        return nothing  # symbol not found on exchange
    end
    # No live connection (bybitsim mode) — use cached info
    return get(_syminfocache(xc), symbol, nothing)
end

_exchangevalidsymbol(xc::XchCache, sym) = _routedModule(xc, data_exchange).validsymbol(_routedbc(xc, data_exchange), sym)
_exchangegetklines(xc::XchCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m") = _routedModule(xc, data_exchange).getklines(_routedbc(xc, data_exchange), symbol; startDateTime=startDateTime, endDateTime=endDateTime, interval=interval)
_exchangeget24h(xc::XchCache) = _routedModule(xc, data_exchange).get24h(_routedbc(xc, data_exchange))
_exchangeget24h(xc::XchCache, symbol) = _routedModule(xc, data_exchange).get24h(_routedbc(xc, data_exchange), symbol)
_exchangebalances(xc::XchCache) = _routedModule(xc, trade_exchange_spot).balances(_routedbc(xc, trade_exchange_spot))
function _exchangeaccountcapacity(xc::XchCache)
    mod = _routedModule(xc, trade_exchange_spot)
    if isdefined(mod, :accountcapacity)
        return getproperty(mod, :accountcapacity)(_routedbc(xc, trade_exchange_spot))
    end
    return nothing
end
_exchangeopenorders(xc::XchCache; symbol=nothing, orderid=nothing, orderLinkId=nothing) = _routedModule(xc, trade_exchange_spot).openorders(_routedbc(xc, trade_exchange_spot); symbol=symbol, orderid=orderid, orderLinkId=orderLinkId)
_exchangeorder(xc::XchCache, orderid) = _routedModule(xc, trade_exchange_spot).order(_routedbc(xc, trade_exchange_spot), orderid)
_exchangecancelorder(xc::XchCache, symbol, orderid) = _routedModule(xc, trade_exchange_spot).cancelorder(_routedbc(xc, trade_exchange_spot), symbol, orderid)

"""
Guard: raise an error if the `trade_exchange_spot` role is explicitly configured to a data-only exchange (Bybit)
while a different exchange is set as the data source.  This prevents accidental live Bybit order placement
once Kraken routing is active.
"""
function _asserttradeallowed(xc::XchCache)
    spot_exchange = _routeexchange(xc.routing, trade_exchange_spot, xc.exchange)
    data_ex = _routeexchange(xc.routing, data_exchange, xc.exchange)
    if spot_exchange == EXCHANGE_BYBIT && data_ex != EXCHANGE_BYBIT && !isempty(xc.routing.routes)
        error("Order placement blocked: trade_exchange_spot is routed to $(EXCHANGE_BYBIT) which is configured as data-only (data_exchange=$(data_ex)). Use setrole! to configure a Kraken trade exchange.")
    end
end

_tradelogstring(value) = ismissing(value) || isnothing(value) ? missing : String(value)
_tradelogstring(value::Enum) = String(Symbol(value))
_tradelogfloat(value) = ismissing(value) || isnothing(value) ? missing : Float64(value)

function _orderfield(orderinfo, field::Symbol)
    if isnothing(orderinfo) || !hasproperty(orderinfo, field)
        return missing
    end
    return getproperty(orderinfo, field)
end

function _tradelogroutingrole(role::ExchangeRole)::TradeLog.AuditRoutingRole
    if role == data_exchange
        return TradeLog.routing_data_exchange
    elseif role == trade_exchange_spot
        return TradeLog.routing_trade_exchange_spot
    else
        return TradeLog.routing_trade_exchange_futures
    end
end

function _tradelogmarkettype(role::ExchangeRole)::TradeLog.AuditMarketType
    if role == trade_exchange_spot
        return TradeLog.market_spot
    elseif role == trade_exchange_futures
        return TradeLog.market_futures
    end
    return TradeLog.market_unknown
end

function _tradeloginstrumenttype(role::ExchangeRole)::TradeLog.AuditInstrumentType
    if role == trade_exchange_spot
        return TradeLog.spot_pair
    elseif role == trade_exchange_futures
        return TradeLog.perpetual_future
    end
    return TradeLog.instrument_unknown
end

function _tradelogeventcontext!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_event_context)
        xc.mc[:tradelog_event_context] = Dict{Symbol, Any}()
    end
    return xc.mc[:tradelog_event_context]
end

"Set temporary TradeLog context fields used for subsequent order events."
function settradelogcontext!(xc::XchCache; strategy_engine=missing, strategy_config_ref=missing, signal_label=missing, signal_score=missing, notes=missing, leg_group_id=missing, leg_label=missing)
    ctx = _tradelogeventcontext!(xc)
    for (k, v) in [
        (:strategy_engine, strategy_engine),
        (:strategy_config_ref, strategy_config_ref),
        (:signal_label, signal_label),
        (:signal_score, signal_score),
        (:notes, notes),
        (:leg_group_id, leg_group_id),
        (:leg_label, leg_label),
    ]
        if ismissing(v) || isnothing(v)
            haskey(ctx, k) && delete!(ctx, k)
        else
            ctx[k] = v
        end
    end
    return xc
end

"Clear all temporary TradeLog context fields."
function cleartradelogcontext!(xc::XchCache)
    haskey(xc.mc, :tradelog_event_context) && empty!(xc.mc[:tradelog_event_context])
    return xc
end

# Backward-compatible aliases for legacy callers.
setauditcontext!(xc::XchCache; kwargs...) = settradelogcontext!(xc; kwargs...)
clearauditcontext!(xc::XchCache) = cleartradelogcontext!(xc)

function _tradelogslippagebps(limitprice, fill_price::Union{Missing, Float64}, orderside::AbstractString, event_type::TradeLog.AuditEventType)
    if !(event_type in (TradeLog.ORDER_PARTIAL_FILL, TradeLog.ORDER_FILLED))
        return missing
    end
    if isnothing(limitprice) || ismissing(fill_price)
        return missing
    end
    req = Float64(limitprice)
    req <= 0.0 && return missing
    fill = Float64(fill_price)
    side = lowercase(String(orderside))
    # Positive values mean worse-than-requested execution.
    if side == "buy"
        return ((fill - req) / req) * 10000.0
    elseif side == "sell"
        return ((req - fill) / req) * 10000.0
    end
    return ((fill - req) / req) * 10000.0
end

"Return TradeLog event timestamp in UTC; use simulated time in sim mode when available."
function _tradelogeventtimeutc(xc::XchCache)::DateTime
    if !isnothing(xc.currentdt)
        return DateTime(xc.currentdt)
    end
    return Dates.now(Dates.UTC)
end

function _tradelogorderevent!(xc::XchCache, event_type::TradeLog.AuditEventType, role::ExchangeRole, symbol::AbstractString, orderside::AbstractString, basequantity::Real, limitprice, marginleverage::Signed; orderinfo=nothing, status_reason=nothing)
    pair = basequote(symbol)
    simmode = String(Symbol(xc.mc[:simmode]))
    exchange_order_id = _tradelogstring(_orderfield(orderinfo, :orderid))
    orderid_key = ismissing(exchange_order_id) ? nothing : String(exchange_order_id)
    chains = _tradelogchaincache!(xc)
    correlation_id = isnothing(orderid_key) ? missing : get(chains, orderid_key, orderid_key)
    lastevents = _tradeloglasteventcache!(xc)
    pendingparents = _tradelogpendingparentcache!(xc)
    parent_event_id = if isnothing(orderid_key)
        missing
    elseif haskey(lastevents, orderid_key)
        lastevents[orderid_key]
    elseif haskey(pendingparents, orderid_key)
        pendingparents[orderid_key]
    else
        missing
    end
    fill_base_qty = _tradelogfloat(_orderfield(orderinfo, :executedqty))
    fill_price = _tradelogfloat(_orderfield(orderinfo, :avgprice))
    slippage_bps = _tradelogslippagebps(limitprice, fill_price, orderside, event_type)
    fee_amount = _tradelogfeeamount(xc, event_type, orderinfo, fill_base_qty, fill_price)
    fee_currency = _tradelogfeecurrency(orderinfo, isnothing(pair) ? nothing : pair.quotecoin)
    ctx = _tradelogeventcontext!(xc)
    strategy_engine = haskey(ctx, :strategy_engine) ? String(ctx[:strategy_engine]) : missing
    strategy_config_ref = haskey(ctx, :strategy_config_ref) ? String(ctx[:strategy_config_ref]) : missing
    signal_label = haskey(ctx, :signal_label) ? String(ctx[:signal_label]) : missing
    signal_score = haskey(ctx, :signal_score) ? Float64(ctx[:signal_score]) : missing
    sim_notes = (xc.mc[:simmode] == nosimulation ? missing : "simulation_mode=$(simmode)")
    ctx_notes = haskey(ctx, :notes) ? String(ctx[:notes]) : missing
    leg_group_id = haskey(ctx, :leg_group_id) ? String(ctx[:leg_group_id]) : missing
    leg_label = haskey(ctx, :leg_label) ? String(ctx[:leg_label]) : missing
    notes_parts = String[]
    !ismissing(sim_notes) && push!(notes_parts, sim_notes)
    !ismissing(ctx_notes) && push!(notes_parts, ctx_notes)
    !ismissing(leg_group_id) && push!(notes_parts, "leg_group_id=$(leg_group_id)")
    !ismissing(leg_label) && push!(notes_parts, "leg_label=$(leg_label)")
    notes = isempty(notes_parts) ? missing : join(notes_parts, ";")
    event_time = _tradelogeventtimeutc(xc)
    created_at = Dates.now(Dates.UTC)
    event = TradeLog.AuditEventRow(
        event_type=event_type,
        event_time_utc=event_time,
        created_at_utc=created_at,
        source_module="CryptoXch",
        environment=string(Symbol(EnvConfig.configmode)),
        run_mode=tradelogrunmode(xc),
        run_id=tradelogrunid(xc),
        correlation_id=correlation_id,
        parent_event_id=parent_event_id,
        exchange=_routeexchange(xc.routing, role, xc.exchange),
        account_alias=_routeexchange(xc.routing, role, xc.exchange),
        routing_role=_tradelogroutingrole(role),
        market_type=_tradelogmarkettype(role),
        asset_class=TradeLog.crypto,
        instrument_type=_tradeloginstrumenttype(role),
        venue_instrument_type=(role == trade_exchange_futures ? "futures" : role == trade_exchange_spot ? "spot" : missing),
        symbol=String(symbol),
        baseasset=isnothing(pair) ? missing : pair.basecoin,
        quoteasset=isnothing(pair) ? missing : pair.quotecoin,
        settlement_asset=isnothing(pair) ? missing : pair.quotecoin,
        exchange_order_id=exchange_order_id,
        side=String(orderside),
        order_type=isnothing(limitprice) ? "Market" : "Limit",
        time_in_force=_tradelogstring(_orderfield(orderinfo, :timeinforce)),
        status=_tradelogstring(_orderfield(orderinfo, :status)),
        status_reason=ismissing(status_reason) || isnothing(status_reason) ? _tradelogstring(_orderfield(orderinfo, :rejectreason)) : String(status_reason),
        requested_base_qty=Float64(basequantity),
        requested_quote_qty=isnothing(limitprice) ? missing : Float64(basequantity) * Float64(limitprice),
        requested_limit_price=isnothing(limitprice) ? missing : Float64(limitprice),
        requested_notional=isnothing(limitprice) ? missing : Float64(basequantity) * Float64(limitprice),
        leverage=marginleverage > 0 ? Float64(marginleverage) : missing,
        fill_base_qty=fill_base_qty,
        fill_price=fill_price,
        fee_amount=fee_amount,
        fee_currency=fee_currency,
        slippage_bps=slippage_bps,
        strategy_engine=strategy_engine,
        strategy_config_ref=strategy_config_ref,
        signal_label=signal_label,
        signal_score=signal_score,
        notes=notes,
    )
    try
        TradeLog.writeeventwithhash(event)
        if !isnothing(orderid_key)
            chains[orderid_key] = get(chains, orderid_key, orderid_key)
            lastevents[orderid_key] = event.event_id
            haskey(pendingparents, orderid_key) && delete!(pendingparents, orderid_key)
        end
    catch tradelog_error
        (verbosity >= 1) && @warn "failed to persist tradelog event" event_type symbol exception=(tradelog_error, catch_backtrace())
    end
    return event
end

function _tradelogcreatedorder!(xc::XchCache, role::ExchangeRole, symbol::AbstractString, orderside::AbstractString, basequantity::Real, limitprice, marginleverage::Signed, orderinfo)
    if isnothing(orderinfo)
        _tradelogorderevent!(xc, TradeLog.ORDER_REJECTED, role, symbol, orderside, basequantity, limitprice, marginleverage; status_reason="createorder returned nothing")
        return nothing
    end
    _tradelogorderevent!(xc, TradeLog.ORDER_SUBMITTED, role, symbol, orderside, basequantity, limitprice, marginleverage; orderinfo=orderinfo)
    orderstatus = _tradelogstring(_orderfield(orderinfo, :status))
    rejectreason = _tradelogstring(_orderfield(orderinfo, :rejectreason))
    if orderstatus == "Rejected" || (!ismissing(rejectreason) && rejectreason != "NO ERROR")
        _tradelogorderevent!(xc, TradeLog.ORDER_REJECTED, role, symbol, orderside, basequantity, limitprice, marginleverage; orderinfo=orderinfo, status_reason=ismissing(rejectreason) ? orderstatus : rejectreason)
    end
    return nothing
end

function _tradelogordererror!(xc::XchCache, role::ExchangeRole, symbol::AbstractString, orderside::AbstractString, basequantity::Real, limitprice, marginleverage::Signed, err)
    _tradelogorderevent!(xc, TradeLog.ORDER_REJECTED, role, symbol, orderside, basequantity, limitprice, marginleverage; status_reason=sprint(showerror, err))
    return nothing
end

function _tradelogorderstatecache!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_order_state)
        xc.mc[:tradelog_order_state] = Dict{String, NamedTuple{(:status, :executedqty), Tuple{String, Float64}}}()
    end
    return xc.mc[:tradelog_order_state]
end

function _tradelogordersnapshotcache!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_order_snapshot)
        xc.mc[:tradelog_order_snapshot] = Dict{String, NamedTuple{(:symbol, :side, :baseqty, :limitprice, :marginleverage), Tuple{String, String, Float64, Union{Nothing, Float64}, Int}}}()
    end
    return xc.mc[:tradelog_order_snapshot]
end

function _tradeloglasteventcache!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_order_last_event)
        xc.mc[:tradelog_order_last_event] = Dict{String, String}()
    end
    return xc.mc[:tradelog_order_last_event]
end

function _tradelogchaincache!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_order_chain)
        xc.mc[:tradelog_order_chain] = Dict{String, String}()
    end
    return xc.mc[:tradelog_order_chain]
end

function _tradelogpendingparentcache!(xc::XchCache)
    if !haskey(xc.mc, :tradelog_pending_parent_event)
        xc.mc[:tradelog_pending_parent_event] = Dict{String, String}()
    end
    return xc.mc[:tradelog_pending_parent_event]
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

function _tradelogsetorderparent!(xc::XchCache, new_orderid::AbstractString, old_orderid::AbstractString)
    new_id = String(new_orderid)
    old_id = String(old_orderid)
    chains = _tradelogchaincache!(xc)
    chains[new_id] = get(chains, old_id, old_id)
    lastevents = _tradeloglasteventcache!(xc)
    if haskey(lastevents, old_id)
        _tradelogpendingparentcache!(xc)[new_id] = lastevents[old_id]
    end
    return nothing
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

function _tradelogordernumber(orderinfo, field::Symbol; default::Float64=0.0)
    value = _orderfield(orderinfo, field)
    if ismissing(value) || isnothing(value)
        return default
    end
    try
        return Float64(value)
    catch
        return default
    end
end

function _tradelogordernumber(orderinfo, fields::Vector{Symbol}; default::Float64=0.0)
    value = _orderfieldfirst(orderinfo, fields)
    if ismissing(value) || isnothing(value)
        return default
    end
    try
        return Float64(value)
    catch
        return default
    end
end

function _tradelogordermaybeprice(orderinfo)
    value = _orderfield(orderinfo, :limitprice)
    if ismissing(value) || isnothing(value)
        return nothing
    end
    try
        return Float64(value)
    catch
        return nothing
    end
end

function _tradelogordermaybeprice(orderinfo, fields::Vector{Symbol})
    value = _orderfieldfirst(orderinfo, fields)
    if ismissing(value) || isnothing(value)
        return nothing
    end
    try
        return Float64(value)
    catch
        return nothing
    end
end

function _tradelogfillsnapshotenabled(xc::XchCache)::Bool
    return Bool(get(xc.mc, :tradelog_migration_fill_balance_enabled, false))
end

function _tradelogsnapshotnetbalance(snapshot, asset::AbstractString)::Float64
    snapshot isa AbstractDataFrame || return 0.0
    hascoin = "coin" in names(snapshot)
    hascoin || return 0.0
    size(snapshot, 1) == 0 && return 0.0
    target = uppercase(String(asset))
    hasfree = "free" in names(snapshot)
    haslocked = "locked" in names(snapshot)
    hasborrowed = "borrowed" in names(snapshot)
    hasaccrued = "accruedinterest" in names(snapshot)
    for row in eachrow(snapshot)
        uppercase(String(row.coin)) == target || continue
        freeqty = hasfree ? Float64(row.free) : 0.0
        lockedqty = haslocked ? Float64(row.locked) : 0.0
        borrowedqty = hasborrowed ? Float64(row.borrowed) : 0.0
        accruedqty = hasaccrued ? Float64(row.accruedinterest) : 0.0
        return freeqty + lockedqty - borrowedqty - accruedqty
    end
    return 0.0
end

function _tradelogfillquoteqty(fill_base_qty::Union{Missing, Float64}, fill_price::Union{Missing, Float64})
    if ismissing(fill_base_qty) || ismissing(fill_price)
        return missing
    end
    return Float64(fill_base_qty) * Float64(fill_price)
end

function _tradelogwritefillbalancesnapshot!(xc::XchCache, trigger_event::TradeLog.AuditEventRow, role::ExchangeRole, symbol::AbstractString)
    _tradelogfillsnapshotenabled(xc) || return nothing
    trigger_event.event_type == TradeLog.ORDER_FILLED || return nothing
    pair = basequote(symbol)
    isnothing(pair) && return nothing

    snapshot_before = if haskey(xc.mc, :exchange_balances_snapshot)
        deepcopy(xc.mc[:exchange_balances_snapshot])
    else
        DataFrame()
    end
    snapshot_before_dt = get(xc.mc, :exchange_balances_snapshot_dt, nothing)

    snapshot_after = try
        balancessnapshot(xc; force_refresh=true, ignoresmallvolume=false)
    catch err
        (verbosity >= 1) && @warn "failed to persist fill balance migration snapshot" symbol error=sprint(showerror, err)
        return nothing
    end

    base_before = _tradelogsnapshotnetbalance(snapshot_before, pair.basecoin)
    base_after = _tradelogsnapshotnetbalance(snapshot_after.snapshot, pair.basecoin)
    quote_before = _tradelogsnapshotnetbalance(snapshot_before, pair.quotecoin)
    quote_after = _tradelogsnapshotnetbalance(snapshot_after.snapshot, pair.quotecoin)
    snapshot_notes = [
        "migration_probe=fill_balance_check_v1",
        "snapshot_before_dt=$(isnothing(snapshot_before_dt) ? "missing" : snapshot_before_dt)",
        "snapshot_after_dt=$(snapshot_after.datetime)",
        "base_delta=$(base_after - base_before)",
        "quote_delta=$(quote_after - quote_before)",
    ]
    event_time = _tradelogeventtimeutc(xc)
    created_at = Dates.now(Dates.UTC)
    event = TradeLog.AuditEventRow(
        event_type=TradeLog.POSITION_SNAPSHOT,
        event_time_utc=event_time,
        created_at_utc=created_at,
        source_module="CryptoXch",
        environment=string(Symbol(EnvConfig.configmode)),
        run_mode=tradelogrunmode(xc),
        run_id=tradelogrunid(xc),
        correlation_id=trigger_event.correlation_id,
        parent_event_id=trigger_event.event_id,
        exchange=_routeexchange(xc.routing, role, xc.exchange),
        account_alias=_routeexchange(xc.routing, role, xc.exchange),
        routing_role=_tradelogroutingrole(role),
        market_type=_tradelogmarkettype(role),
        asset_class=TradeLog.crypto,
        instrument_type=_tradeloginstrumenttype(role),
        venue_instrument_type=(role == trade_exchange_futures ? "futures" : role == trade_exchange_spot ? "spot" : missing),
        symbol=String(symbol),
        baseasset=pair.basecoin,
        quoteasset=pair.quotecoin,
        settlement_asset=pair.quotecoin,
        exchange_order_id=trigger_event.exchange_order_id,
        side=trigger_event.side,
        order_type=trigger_event.order_type,
        status="balance_observed_after_fill",
        status_reason="source=fill_balance_probe",
        fill_base_qty=trigger_event.fill_base_qty,
        fill_quote_qty=_tradelogfillquoteqty(trigger_event.fill_base_qty, trigger_event.fill_price),
        fill_price=trigger_event.fill_price,
        fee_amount=trigger_event.fee_amount,
        fee_currency=trigger_event.fee_currency,
        position_qty_before=base_before,
        position_qty_after=base_after,
        cash_before=quote_before,
        cash_after=quote_after,
        strategy_engine=trigger_event.strategy_engine,
        strategy_config_ref=trigger_event.strategy_config_ref,
        signal_label=trigger_event.signal_label,
        signal_score=trigger_event.signal_score,
        notes=join(snapshot_notes, ";"),
    )
    try
        TradeLog.writeeventwithhash(event)
    catch err
        (verbosity >= 1) && @warn "failed to persist fill balance migration snapshot event" symbol error=sprint(showerror, err)
    end
    return nothing
end

function _tradelogeventtypeforstatus(status::AbstractString, previous_status::Union{Nothing, String}, executedqty::Float64, previous_executedqty::Union{Nothing, Float64}, baseqty::Float64, source::AbstractString)::TradeLog.AuditEventType
    st = lowercase(String(status))
    if st == "rejected"
        return TradeLog.ORDER_REJECTED
    elseif st in ["cancelled", "canceled", "partiallyfilledcanceled", "deactivated"]
        return TradeLog.ORDER_CANCELED
    elseif st == "replaced" || source == "changeorder"
        return TradeLog.ORDER_AMENDED
    elseif (st == "filled") || ((baseqty > 0.0) && (executedqty >= baseqty - 1e-9))
        return TradeLog.ORDER_FILLED
    elseif (st == "partiallyfilled") || (!isnothing(previous_executedqty) && (executedqty > previous_executedqty + 1e-9))
        return TradeLog.ORDER_PARTIAL_FILL
    elseif isnothing(previous_status)
        return TradeLog.ORDER_OBSERVED
    end
    return TradeLog.ORDER_ACK
end

function _tradelogreconcileorderstate!(xc::XchCache, orderinfo; role::ExchangeRole=trade_exchange_spot, source::AbstractString="orderpoll")
    orderid = _tradelogstring(_orderfield(orderinfo, :orderid))
    symbol = _tradelogstring(_orderfield(orderinfo, :symbol))
    if ismissing(orderid) || ismissing(symbol)
        return nothing
    end

    status_raw = _tradelogstring(_orderfield(orderinfo, :status))
    status = ismissing(status_raw) ? "Unknown" : String(status_raw)
    executedqty = _tradelogordernumber(orderinfo, :executedqty; default=0.0)
    baseqty = _tradelogordernumber(orderinfo, :baseqty; default=executedqty)
    side_raw = _tradelogstring(_orderfield(orderinfo, :side))
    side = ismissing(side_raw) ? "Unknown" : String(side_raw)
    limitprice = _tradelogordermaybeprice(orderinfo)
    marginleverage = Int(round(_tradelogordernumber(orderinfo, :marginleverage; default=0.0)))

    states = _tradelogorderstatecache!(xc)
    previous = get(states, String(orderid), nothing)
    previous_status = isnothing(previous) ? nothing : previous.status
    previous_executedqty = isnothing(previous) ? nothing : previous.executedqty
    changed = isnothing(previous) || (status != previous_status) || (isnothing(previous_executedqty) || (abs(executedqty - previous_executedqty) > 1e-9))
    if !changed
        return nothing
    end

    event_type = _tradelogeventtypeforstatus(status, previous_status, executedqty, previous_executedqty, baseqty, source)
    status_reason = _tradelogstring(_orderfield(orderinfo, :rejectreason))
    if event_type in (TradeLog.ORDER_ACK, TradeLog.ORDER_OBSERVED, TradeLog.ORDER_AMENDED)
        status_reason = "source=$(source)"
    elseif ismissing(status_reason)
        status_reason = "source=$(source)"
    end
    event = _tradelogorderevent!(xc, event_type, role, String(symbol), side, baseqty, limitprice, marginleverage; orderinfo=orderinfo, status_reason=status_reason)
    _tradelogwritefillbalancesnapshot!(xc, event, role, String(symbol))
    states[String(orderid)] = (status=status, executedqty=executedqty)
    _tradelogordersnapshotcache!(xc)[String(orderid)] = (symbol=String(symbol), side=side, baseqty=baseqty, limitprice=limitprice, marginleverage=marginleverage)
    return nothing
end

"""
Emit cancellation events for orders that were previously observed as open but are
missing from the latest full `getopenorders` response.
"""
function _tradeloglogmissingopenorders!(xc::XchCache, openorderids)
    active = Set(String.(collect(openorderids)))
    states = _tradelogorderstatecache!(xc)
    snapshots = _tradelogordersnapshotcache!(xc)
    for (orderid, state) in collect(states)
        openstatus(state.status) || continue
        orderid in active && continue
        # Try resolving a terminal order state first. This captures fast fills
        # that disappear from openorders between polling cycles.
        resolved = false
        try
            order = getorder(xc, orderid; auditevent=true)
            resolved = !isnothing(order)
        catch err
            (verbosity >= 1) && @warn "failed to resolve missing open order via getorder" orderid error=sprint(showerror, err)
        end
        resolved && continue
        if haskey(snapshots, orderid)
            snap = snapshots[orderid]
            missingorder = (
                orderid=orderid,
                symbol=snap.symbol,
                side=snap.side,
                baseqty=Float32(snap.baseqty),
                executedqty=Float32(state.executedqty),
                limitprice=isnothing(snap.limitprice) ? missing : Float32(snap.limitprice),
                marginleverage=snap.marginleverage,
                status="Cancelled",
                updated=Dates.now(Dates.UTC),
                rejectreason="source=getopenorders_missing",
            )
            _tradelogreconcileorderstate!(xc, missingorder; source="getopenorders_missing")
        else
            states[orderid] = (status="Cancelled", executedqty=state.executedqty)
        end
    end
    return nothing
end

function _tradelogfeeamount(xc::XchCache, event_type::TradeLog.AuditEventType, orderinfo, fill_base_qty::Union{Missing, Float64}, fill_price::Union{Missing, Float64})
    explicit_fee = _tradelogordernumber(orderinfo, [:fee_amount, :feeamount, :fee, :commission, :cumexecfee, :execfee, :fees]; default=NaN)
    if isfinite(explicit_fee)
        return explicit_fee
    end
    if (event_type in (TradeLog.ORDER_PARTIAL_FILL, TradeLog.ORDER_FILLED)) && !ismissing(fill_base_qty) && !ismissing(fill_price)
        return Float64(fill_base_qty) * Float64(fill_price) * Float64(xc.feerate)
    end
    return missing
end

function _tradelogfeecurrency(orderinfo, quotecoin)
    fee_currency = _tradelogstring(_orderfieldfirst(orderinfo, [:fee_currency, :feecurrency, :commissionasset]))
    if !ismissing(fee_currency)
        return fee_currency
    end
    return isnothing(quotecoin) ? missing : quotecoin
end

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

_exchangecreateorder(xc::XchCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0, reduceonly::Bool=false) = (_asserttradeallowed(xc); _routedModule(xc, trade_exchange_spot).createorder(_routedbc(xc, trade_exchange_spot), symbol, orderside, basequantity, price, maker, marginleverage=marginleverage, reduceonly=reduceonly))
_exchangeamendorder(xc::XchCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = (_asserttradeallowed(xc); _routedModule(xc, trade_exchange_spot).amendorder(_routedbc(xc, trade_exchange_spot), symbol, orderid; basequantity=basequantity, limitprice=limitprice))
_exchangeamendorder(xc::XchCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing) = (_asserttradeallowed(xc); _routedModule(xc, trade_exchange_spot).amendorder(_routedbc(xc, trade_exchange_spot), orderid; basequantity=basequantity, limitprice=limitprice))

"""
Create a close order for one existing position side.

- `positionside=:long` closes long exposure via a Sell order.
- `positionside=:short` closes short exposure via a Buy order.

Adapters may specialize this by implementing `closeorder(bc, symbol, positionside, basequantity, limitprice, maker; marginleverage=..., reduceonly=...)`.
If no adapter specialization exists, this function falls back to existing `createbuyorder`/`createsellorder` behavior.
"""
function closeorder(xc::XchCache, base::AbstractString; positionside::Symbol, limitprice, basequantity, maker::Bool=true, marginleverage::Signed=0, reduceonly::Bool=true, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    side = Symbol(lowercase(String(positionside)))
    @assert side in (:long, :short) "closeorder positionside=$(positionside) must be :long or :short"

    baseup = uppercase(String(base))
    symbol = symboltoken(xc, baseup, EnvConfig.cryptoquote; role=trade_exchange_spot)
    mod = _routedModule(xc, trade_exchange_spot)
    bc = _routedbc(xc, trade_exchange_spot)

    if isdefined(mod, :closeorder)
        fn = getfield(mod, :closeorder)
        if applicable(fn, bc, symbol, side, basequantity, limitprice, maker; marginleverage=marginleverage, reduceonly=reduceonly)
            _asserttradeallowed(xc)
            if !isnothing(leg_group_id) || !isnothing(leg_label)
                settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
            end
            try
                created = fn(bc, symbol, side, basequantity, limitprice, maker; marginleverage=marginleverage, reduceonly=reduceonly)
                oid, oocreate = _normalizecreatedorder(xc, created)
                orderside = side == :long ? "Sell" : "Buy"
                if !isnothing(parent_order_id) && !isnothing(oid)
                    _tradelogsetorderparent!(xc, String(oid), String(parent_order_id))
                end
                _tradelogcreatedorder!(xc, trade_exchange_spot, symbol, orderside, basequantity, limitprice, marginleverage, oocreate)
                if isnothing(limitprice) && maker && !isnothing(oid)
                    registeradaptiveorder!(xc, oid)
                end
                return oid
            catch err
                orderside = side == :long ? "Sell" : "Buy"
                _tradelogordererror!(xc, trade_exchange_spot, symbol, orderside, basequantity, limitprice, marginleverage, err)
                rethrow()
            finally
                if !isnothing(leg_group_id) || !isnothing(leg_label)
                    cleartradelogcontext!(xc)
                end
            end
        end
    end

    if side == :long
        return createsellorder(xc, baseup; limitprice=limitprice, basequantity=basequantity, maker=maker, marginleverage=marginleverage, reduceonly=reduceonly, parent_order_id=parent_order_id, leg_group_id=leg_group_id, leg_label=leg_label)
    end
    return createbuyorder(xc, baseup; limitprice=limitprice, basequantity=basequantity, maker=maker, marginleverage=marginleverage, reduceonly=reduceonly, parent_order_id=parent_order_id, leg_group_id=leg_group_id, leg_label=leg_label)
end

setstartdt(xc::XchCache, dt::DateTime) = (xc.startdt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
setenddt(xc::XchCache, dt::DateTime) = (xc.enddt = isnothing(dt) ? nothing : floor(dt, Minute(1)))
bases(xc::XchCache) = keys(xc.bases)
ohlcv(xc::XchCache) = values(xc.bases)
ohlcv(xc::XchCache, base::AbstractString) = xc.bases[base]
baseohlcvdict(xc::XchCache) = xc.bases

basenottradable = ["MATIC", "FTM", "KFEE"]  # KFEE = Kraken proprietary fee credit, never tradeable
basestablecoin = ["USD", "USD1", "USDT", "TUSD", "BUSD", "USDC", "USDE", "EUR", "DAI"]
quotecoins = ["USDT"]  # , "USDC"]
baseignore = uppercase.(union(basestablecoin, basenottradable))
minimumquotevolume = 10  # USDT

MAXLIMITDELTA = 0.1

_isleveraged(token) = !isnothing(token) && (length(token) > 2) && (token[end] in ['S', 'L']) && isdigit(token[end-1])

#region support

validbase(xc::XchCache, base::AbstractString) =
    (uppercase(base) != uppercase(EnvConfig.cryptoquote)) && validsymbol(xc, symboltoken(base))

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

assetbases(xc::XchCache) = filter(!=(uppercase(EnvConfig.cryptoquote)), uppercase.(CryptoXch.balances(xc)[!, :coin]))

symboltoken(basecoin, quotecoin=EnvConfig.cryptoquote) = isnothing(basecoin) ? nothing : uppercase(basecoin * quotecoin)

"""
Resolve the exchange-specific symbol token for a pair on the routed exchange.
Falls back to a concatenated symbol if the adapter cannot map the pair yet.
"""
function symboltoken(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.cryptoquote; role::ExchangeRole=trade_exchange_spot)
    bc = _routedbc(xc, role)
    if isnothing(bc)
        return symboltoken(basecoin, quotecoin)
    end
    return _routedModule(xc, role).symboltoken(bc, basecoin, quotecoin)
end

"Return side-specific margin leverage caps for one symbol when supported by the routed exchange."
function marginlimits(xc::XchCache, symbol::AbstractString; role::ExchangeRole=trade_exchange_spot)
    bc = _routedbc(xc, role)
    isnothing(bc) && return (maxleveragebuy=0, maxleveragesell=0)
    mod = _routedModule(xc, role)
    if isdefined(mod, :marginlimits) && applicable(getfield(mod, :marginlimits), bc, symbol)
        return mod.marginlimits(bc, symbol)
    end
    return (maxleveragebuy=0, maxleveragesell=0)
end

"Return true when routed exchange metadata permits side/leverage for one symbol."
function marginpermitted(xc::XchCache, symbol::AbstractString, orderside::AbstractString, marginleverage::Signed; role::ExchangeRole=trade_exchange_spot)::Bool
    marginleverage <= 0 && return true
    bc = _routedbc(xc, role)
    isnothing(bc) && return false
    mod = _routedModule(xc, role)
    if isdefined(mod, :marginpermitted) && applicable(getfield(mod, :marginpermitted), bc, symbol, orderside, marginleverage)
        return mod.marginpermitted(bc, symbol, orderside, marginleverage)
    end
    return true
end

ceilbase(base, qty) = base == "usdt" ? ceil(qty, digits=3) : ceil(qty, digits=5)
floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.cryptoquote)) &&
    !(uppercase(symbol[1:end-length(EnvConfig.cryptoquote)]) in baseignore)

"Returns pair of basecoin and quotecoin if quotecoin in `quotecoins` or equals `EnvConfig.cryptoquote` else `nothing` is returned"
function basequote(symbol)
    symbol = uppercase(symbol)
    candidates = union(quotecoins, [uppercase(EnvConfig.cryptoquote)])
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
    return minimumqty(xc, symboltoken(xc, basecoin, quotecoin; role=trade_exchange_spot))
end

"""
Return precision information for a `(basecoin, quotecoin)` pair.
"""
function precision(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString)
    return precision(xc, symboltoken(xc, basecoin, quotecoin; role=trade_exchange_spot))
end

#endregion support

#region time

"""
Removes ohlcv data rows that are outside the date boundaries (nothing= no boundary) and adjusts ohlcv.ix to stay within the new data range.
"""
function timerangecut!(xc::XchCache, startdt, enddt)
    for ohlcv in CryptoXch.ohlcv(xc)
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
        CryptoXch.setcurrenttime!(xc, currentdt)  # also updates bases if current time is > last time of xc
    end
    (verbosity >= 3) && println("iterate: utcnow=$(Dates.now(UTC)) startdt=$(xc.startdt), currentdt=$(xc.currentdt), enddt=$(xc.enddt)")
    return xc, currentdt
end

timesimulation(xc::XchCache)::Bool = !isnothing(xc.currentdt) && !isnothing(xc.enddt)
tradetime(xc::XchCache) = isnothing(xc.currentdt) ? (isnothing(xc.bc) ? xc.startdt : floor(_exchangeservertime(xc), Minute(1))) : xc.currentdt
# tradetime(xc::XchCache) = (xc.mc[:simmode] != bybitsim) ? _exchangeservertime(xc) : Dates.now(UTC)
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
			return _exchangeservertime(xc)
		catch err
			(verbosity >= 1) && @warn "exchange server time unavailable; retrying in 60 seconds" retry_seconds=60 exception=sprint(showerror, err)
			sleep(60)
		end
	end
end

function _sleepuntil(xc::XchCache, dt::DateTime)
    if !isnothing(xc.enddt) || (xc.mc[:simmode] != nosimulation)
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

function _fetchwsclosedkline(xc::XchCache, symbol::AbstractString, interval::AbstractString)
    mod = _routedModule(xc, data_exchange)
    if !isdefined(mod, :wsclosedkline)
        return nothing
    end
    try
        return getproperty(mod, :wsclosedkline)(_routedbc(xc, data_exchange), symbol, String(interval))
    catch
        try
            return getproperty(mod, :wsclosedkline)(symbol, String(interval))
        catch
            return nothing
        end
    end
end

function _upsert_closed_wscandle!(ohlcv, candle)
    isnothing(candle) && return nothing
    df = Ohlcv.dataframe(ohlcv)
    cdt = floor(DateTime(candle.opentime), Minute(1))
    copen = Float32(candle.open)
    chigh = Float32(candle.high)
    clow = Float32(candle.low)
    cclose = Float32(candle.close)
    cvol = Float32(candle.basevolume)

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

function _applywsclosedcandle!(xc::XchCache, ohlcv, dt::DateTime)
    sym = symboltoken(xc, ohlcv.base, ohlcv.quotecoin; role=data_exchange)
    candle = _fetchwsclosedkline(xc, sym, ohlcv.interval)
    isnothing(candle) && return nothing
    cutoff = dt - intervalperiod(ohlcv.interval)
    if floor(DateTime(candle.opentime), Minute(1)) <= cutoff
        _upsert_closed_wscandle!(ohlcv, candle)
    end
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
    _applywsclosedcandle!(xc, ohlcv, dt)
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
    _setsimtime!(xc.bc, datetime)
    for bc in values(xc.routecaches)
        _setsimtime!(bc, datetime)
    end
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
function _ohlcfromexchange(xc::XchCache, base::AbstractString, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    symbol = uppercase(base*cryptoquote)
    df = _exchangegetklines(xc, symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
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
        ohlcv = CryptoXch.cryptodownload(xc, base, "1m", startdt, enddt)
        Ohlcv.write(ohlcv)
    end
end

"Downloads all basecoins with USDT quote that shows a minimumdayquotevolume and saves it as canned data"
function downloadallUSDT(xc::XchCache, enddt, period=Dates.Year(10), minimumdayquotevolume = 10000000)
    df = getUSDTmarket(xc)
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    bases = sort!(setdiff(df[!, :basecoin], baseignore))
    (verbosity >= 2) && println("$(EnvConfig.now())downloading the following bases bases with $(EnvConfig.cryptoquote) quote: $bases")
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
    exch_valid = _exchangevalidsymbol(xc, sym)
    r = !isnothing(sym) &&
        exch_valid &&
        !(sym.basecoin in baseignore) &&
        !_isleveraged(sym.basecoin)
    return r
end

function validsymbol(xc::XchCache, basecoin::AbstractString, quotecoin::AbstractString)
    bc = _routedbc(xc, data_exchange)
    return !isnothing(bc) && validsymbol(bc, basecoin, quotecoin)
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
    sym = CryptoXch.symboltoken(base)
    syminfo = CryptoXch.minimumqty(xc, sym)
    return isnothing(syminfo) ? nothing : 1.01 * max(syminfo.minbaseqty, syminfo.minquoteqty/price) # 1% more to avoid issues by rounding errors
end

function minimumquotequantity(xc::XchCache, base::AbstractString, price=(base in bases(xc) ? Ohlcv.dataframe(ohlcv(xc, base))[Ohlcv.ix(ohlcv(xc, base)), :close] : nothing))
    if isnothing(price)
        return nothing
    end
    sym = CryptoXch.symboltoken(base)
    syminfo = CryptoXch.minimumqty(xc, sym)
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
        return _exchangeget24h(xc)
    end

    rows = DataFrame(askprice=Float32[], bidprice=Float32[], lastprice=Float32[], quotevolume24h=Float32[], pricechangepercent=Float32[], symbol=String[])
    quotetoken = uppercase(String(EnvConfig.cryptoquote))
    wanted = unique([uppercase(String(base)) for base in requestedbases if !isnothing(base) && (uppercase(String(base)) != quotetoken)])
    for base in wanted
        symbol = symboltoken(xc, base, quotetoken; role=data_exchange)
        row = _tickerrow(_exchangeget24h(xc, symbol))
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
        askprice=Float32(row.askprice),
        bidprice=Float32(row.bidprice),
        lastprice=Float32(row.lastprice),
        quotevolume24h=Float32(row.quotevolume24h),
        pricechangepercent=Float32(row.pricechangepercent),
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
    usdtdf[!, :basecoin] = [isnothing(bqe) ? missing : bqe.basecoin for bqe in bq]
    nbq = [!isnothing(bqe) && validbase(xc, bqe.basecoin) && (bqe.quotecoin == EnvConfig.cryptoquote) for bqe in bq]  # create binary vector as DataFrame filter
    usdtdf = usdtdf[nbq, Not(:symbol)]
    return usdtdf
end

"""
Returns the broad USDT market snapshot used for selection/screening logic.
"""
function screeningUSDTmarket(xc::XchCache; dt::DateTime=tradetime(xc))
    if get(xc.mc, :simmode, nosimulation) == bybitsim
        setcurrenttime!(xc, dt)
    end
    return getUSDTmarket(xc; dt=dt)
end

"""
Returns a coin-scoped USDT market snapshot used for portfolio valuation.
Only the requested base coins are queried from the exchange adapter.
"""
function valuationUSDTmarket(xc::XchCache, requestedbases; dt::DateTime=tradetime(xc))
    if get(xc.mc, :simmode, nosimulation) == bybitsim
        setcurrenttime!(xc, dt)
    end
    return getUSDTmarket(xc; dt=dt, requestedbases=requestedbases)
end

#endregion public

#region account

function _asfloat64(value, default::Float64=0.0)::Float64
    if ismissing(value) || isnothing(value)
        return default
    elseif value isa AbstractFloat
        return Float64(value)
    elseif value isa Real
        return Float64(value)
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
    quotecoin = uppercase(String(EnvConfig.cryptoquote))
    quotefree = 0.0
    if (:coin in names(assets)) && (:free in names(assets))
        for row in eachrow(assets)
            if uppercase(String(row.coin)) == quotecoin
                quotefree += max(0.0, Float64(row.free))
            end
        end
    end
    equity = (:usdtvalue in names(assets)) ? Float64(sum(assets[!, :usdtvalue])) : quotefree
    return (
        equity_quote=max(0.0, equity),
        available_opening_quote=max(0.0, quotefree),
        available_long_quote=max(0.0, quotefree),
        available_short_quote=max(0.0, quotefree),
        initial_margin_quote=0.0,
        maintenance_margin_quote=0.0,
        source="CryptoXch:portfolio_fallback",
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
    simmode = get(xc.mc, :simmode, nosimulation)
    if !force_refresh && (simmode == nosimulation)
        if haskey(xc.mc, :account_capacity_snapshot) && haskey(xc.mc, :account_capacity_snapshot_dt)
            dt = xc.mc[:account_capacity_snapshot_dt]
            if (dt isa DateTime) && ((Dates.now(UTC) - dt) < Dates.Second(max(1, ttl_seconds)))
                return xc.mc[:account_capacity_snapshot]
            end
        end
    end

    snapshot = try
        _exchangeaccountcapacity(xc)
    catch err
        (verbosity >= 1) && @warn "accountcapacity: exchange snapshot failed, using fallback" exchange=_routeexchange(xc.routing, trade_exchange_spot, xc.exchange) error=sprint(showerror, err)
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

"Returns a DataFrame[:coin, :locked, :free, :borrowed, :accruedinterest] of wallet/portfolio balances"
function balances(xc::XchCache; ignoresmallvolume=true)
    bdf = _exchangebalances(xc)
    if !isnothing(bdf)
        select = [!(coin in baseignore) || (coin == EnvConfig.cryptoquote) for coin in bdf[!, :coin]]
        bdf = bdf[select, :]
    end
    if !isnothing(bdf) && (size(bdf, 1) > 0) && ignoresmallvolume
        delrows = []
        for ix in eachindex(bdf[!, :coin])
            if bdf[ix, :coin] != EnvConfig.cryptoquote
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

"Capture one canonical exchange-owned balances snapshot and store it in `xc.mc`."
function refreshbalancessnapshot!(xc::XchCache; ignoresmallvolume::Bool=false)
    use_ws_primary = _wsenabled(xc, :ws_primary_mode, false) && _wsenabled(xc, :ws_balances_enabled, false)
    snapshot = if use_ws_primary
        wsb = wsbalancessnapshot(xc)
        wsdt = wsbalancesheartbeat(xc)
        if (size(wsb, 1) > 0) || !isnothing(wsdt)
            wsb
        else
            (verbosity >= 1) && @warn "ws balance snapshot unavailable; falling back to REST balances"
            balances(xc; ignoresmallvolume=ignoresmallvolume)
        end
    else
        balances(xc; ignoresmallvolume=ignoresmallvolume)
    end
    if isnothing(snapshot)
        snapshot = DataFrame()
    end
    xc.mc[:exchange_balances_snapshot] = deepcopy(snapshot)
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
            quotetoken = uppercase(String(EnvConfig.cryptoquote))
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
            if portfoliodf[bix, :coin] == EnvConfig.cryptoquote
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
            is_quotecoin = (uppercase(coin) == uppercase(EnvConfig.cryptoquote)) || (coin in quotecoins)
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
        wsdf = wsordersnapshot(xc)
        wsdt = wsordersheartbeat(xc)
        if (size(wsdf, 1) > 0) || !isnothing(wsdt)
            wsdf
        else
            (verbosity >= 1) && @warn "ws order snapshot unavailable; falling back to REST openorders"
            _exchangeopenorders(xc, symbol=symboltoken(base))
        end
    else
        _exchangeopenorders(xc, symbol=symboltoken(base))
    end
    openordersdf = size(oo) == (0, 0) ? _emptyorders(exchange(xc)) : oo
    for row in eachrow(openordersdf)
        _tradelogreconcileorderstate!(xc, NamedTuple(row); source="getopenorders")
    end
    if isnothing(base) && "orderid" in names(openordersdf)
        _tradeloglogmissingopenorders!(xc, openordersdf[!, :orderid])
    end
    if "orderid" in names(openordersdf)
        pruneadaptiveorders!(xc, openordersdf[!, :orderid])
    end
    return openordersdf
end

"Returns a named tuple with elements equal to columns of getopenorders() dataframe of the identified order or `nothing` if order is not found"
function getorder(xc::XchCache, orderid; auditevent::Bool=true)
    order = _exchangeorder(xc, orderid)
    if auditevent && !isnothing(order)
        _tradelogreconcileorderstate!(xc, order; source="getorder")
    end
    return order
end

"Returns orderid in case of a successful cancellation"
function cancelorder(xc::XchCache, base, orderid; leg_group_id=nothing, leg_label=nothing)
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
    end
    unregisteradaptiveorder!(xc, orderid)
    cancelsymbol = symboltoken(xc, base, EnvConfig.cryptoquote; role=trade_exchange_spot)
    cancelled = _exchangecancelorder(xc, cancelsymbol, orderid)
    if !isnothing(cancelled)
        # Assume exchange-side cancel success when Kraken confirms CancelOrder.
        # If reality diverges, the next OpenOrders loop will re-discover the order.
        current = (
            orderid=String(orderid),
            symbol=String(cancelsymbol),
            side=missing,
            baseqty=0f0,
            executedqty=0f0,
            limitprice=missing,
            marginleverage=0,
            status="Cancelled",
            updated=Dates.now(Dates.UTC),
            rejectreason="cancelled_by_user",
        )
        _tradelogreconcileorderstate!(xc, current; source="cancelorder_assumed_success")
    end
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        cleartradelogcontext!(xc)
    end
    return cancelled
end

"""
Places an order: spot order by default or margin order if 2 <= marginleverage <= 10
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.

Pass `limitprice=nothing` together with `maker=true` to ask the adapter to choose
a limit price as close as possible to the current spread while remaining post-only,
so the order can qualify for maker fees.

Order is rejected (but order created) if the resulting price crosses the spread in
order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createbuyorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=false, marginleverage::Signed=0, reduceonly::Bool=false, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    base = uppercase(base)
    _asserttradeallowed(xc)
    symbol = symboltoken(xc, base, EnvConfig.cryptoquote; role=trade_exchange_spot)
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
    end
    try
        # Adapter-backed path for both live and simulation exchanges.
        created = _exchangecreateorder(xc, symbol, "Buy", basequantity, limitprice, maker, marginleverage=marginleverage, reduceonly=reduceonly)
        oid, oocreate = _normalizecreatedorder(xc, created)
        if !isnothing(parent_order_id) && !isnothing(oid)
            _tradelogsetorderparent!(xc, String(oid), String(parent_order_id))
        end
        _tradelogcreatedorder!(xc, trade_exchange_spot, symbol, "Buy", basequantity, limitprice, marginleverage, oocreate)
        if isnothing(limitprice) && maker && !isnothing(oid)
            registeradaptiveorder!(xc, oid)
        end
        (verbosity >= 3) && @info "$(tradetime(xc)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    catch err
        _tradelogordererror!(xc, trade_exchange_spot, symbol, "Buy", basequantity, limitprice, marginleverage, err)
        rethrow()
    finally
        if !isnothing(leg_group_id) || !isnothing(leg_label)
            cleartradelogcontext!(xc)
        end
    end
end

"""
Places an order: spot order by default or margin order if 2 <= marginleverage <= 10
Adapts `limitprice` and `basequantity` according to symbol rules and executes order.

Pass `limitprice=nothing` together with `maker=true` to ask the adapter to choose
a limit price as close as possible to the current spread while remaining post-only,
so the order can qualify for maker fees.

Order is rejected (but order created) if the resulting price crosses the spread in
order to secure maker price fees.
Returns `nothing` in case order execution fails.
"""
function createsellorder(xc::XchCache, base::AbstractString; limitprice, basequantity, maker::Bool=true, marginleverage::Signed=0, reduceonly::Bool=false, parent_order_id=nothing, leg_group_id=nothing, leg_label=nothing)
    base = uppercase(base)
    _asserttradeallowed(xc)
    symbol = symboltoken(xc, base, EnvConfig.cryptoquote; role=trade_exchange_spot)
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
    end
    try
        # Adapter-backed path for both live and simulation exchanges.
        created = _exchangecreateorder(xc, symbol, "Sell", basequantity, limitprice, maker, marginleverage=marginleverage, reduceonly=reduceonly)
        oid, oocreate = _normalizecreatedorder(xc, created)
        if !isnothing(parent_order_id) && !isnothing(oid)
            _tradelogsetorderparent!(xc, String(oid), String(parent_order_id))
        end
        _tradelogcreatedorder!(xc, trade_exchange_spot, symbol, "Sell", basequantity, limitprice, marginleverage, oocreate)
        if isnothing(limitprice) && maker && !isnothing(oid)
            registeradaptiveorder!(xc, oid)
        end
        (verbosity >= 3) && @info "$(tradetime(xc)) $base: $(isnothing(oocreate) ? "no order info" : oocreate)"
        return oid
    catch err
        _tradelogordererror!(xc, trade_exchange_spot, symbol, "Sell", basequantity, limitprice, marginleverage, err)
        rethrow()
    finally
        if !isnothing(leg_group_id) || !isnothing(leg_label)
            cleartradelogcontext!(xc)
        end
    end
end

"""
Amend an existing order.

If the order is post-only and `limitprice=nothing`, the routed adapter will
re-snapshot the current spread and keep the maker intent adaptive instead of
freezing the previous limit.
"""
function changeorder(xc::XchCache, symbol::AbstractString, orderid; limitprice=nothing, basequantity=nothing, leg_group_id=nothing, leg_label=nothing)
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
    end
    amended = _exchangeamendorder(xc, String(symbol), String(orderid); basequantity=basequantity, limitprice=limitprice)
    new_orderid, ooamend = _normalizeamendedorder(xc, amended)
    if isnothing(new_orderid)
        if !isnothing(leg_group_id) || !isnothing(leg_label)
            cleartradelogcontext!(xc)
        end
        return nothing
    end
    old_orderid = String(orderid)
    if new_orderid != old_orderid
        if isadaptiveorder(xc, old_orderid)
            unregisteradaptiveorder!(xc, old_orderid)
            registeradaptiveorder!(xc, new_orderid)
        end
        if !isnothing(ooamend) && hasproperty(ooamend, :symbol)
            cancelled = (
                orderid=old_orderid,
                symbol=String(getproperty(ooamend, :symbol)),
                side=hasproperty(ooamend, :side) ? getproperty(ooamend, :side) : missing,
                baseqty=hasproperty(ooamend, :baseqty) ? getproperty(ooamend, :baseqty) : 0f0,
                executedqty=0f0,
                limitprice=missing,
                marginleverage=0,
                status="Cancelled",
                updated=Dates.now(Dates.UTC),
                rejectreason="amended_to=$(new_orderid)",
            )
            _tradelogreconcileorderstate!(xc, cancelled; source="changeorder")
        end
        _tradelogsetorderparent!(xc, new_orderid, old_orderid)
    end
    if !isnothing(ooamend)
        _tradelogreconcileorderstate!(xc, ooamend; source="changeorder")
    end
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        cleartradelogcontext!(xc)
    end
    return new_orderid
end

function changeorder(xc::XchCache, orderid; limitprice=nothing, basequantity=nothing, leg_group_id=nothing, leg_label=nothing)
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        settradelogcontext!(xc; leg_group_id=leg_group_id, leg_label=leg_label)
    end
    amended = _exchangeamendorder(xc, String(orderid); basequantity=basequantity, limitprice=limitprice)
    new_orderid, ooamend = _normalizeamendedorder(xc, amended)
    if isnothing(new_orderid)
        if !isnothing(leg_group_id) || !isnothing(leg_label)
            cleartradelogcontext!(xc)
        end
        return nothing
    end
    old_orderid = String(orderid)
    if new_orderid != old_orderid
        if isadaptiveorder(xc, old_orderid)
            unregisteradaptiveorder!(xc, old_orderid)
            registeradaptiveorder!(xc, new_orderid)
        end
        if !isnothing(ooamend) && hasproperty(ooamend, :symbol)
            cancelled = (
                orderid=old_orderid,
                symbol=String(getproperty(ooamend, :symbol)),
                side=hasproperty(ooamend, :side) ? getproperty(ooamend, :side) : missing,
                baseqty=hasproperty(ooamend, :baseqty) ? getproperty(ooamend, :baseqty) : 0f0,
                executedqty=0f0,
                limitprice=missing,
                marginleverage=0,
                status="Cancelled",
                updated=Dates.now(Dates.UTC),
                rejectreason="amended_to=$(new_orderid)",
            )
            _tradelogreconcileorderstate!(xc, cancelled; source="changeorder")
        end
        _tradelogsetorderparent!(xc, new_orderid, old_orderid)
    end
    if !isnothing(ooamend)
        _tradelogreconcileorderstate!(xc, ooamend; source="changeorder")
    end
    if !isnothing(leg_group_id) || !isnothing(leg_label)
        cleartradelogcontext!(xc)
    end
    return new_orderid
end

"""
    createocoorder(xc, base; entry_side, entry_price, take_profit_price, stop_loss_price,
                   basequantity, maker=false, marginleverage=0, signal_label=nothing,
                   signal_score=nothing, strategy_engine=nothing, strategy_config_ref=nothing) -> NamedTuple

Places a three-leg bracket (OCO) order group:
- **entry**: the initial buy or sell (`entry_side ∈ (:buy, :sell)`)
- **take_profit**: limit order on the opposite side at `take_profit_price`
- **stop_loss**: limit order on the opposite side at `stop_loss_price`

All three legs share the same `leg_group_id` (a new UUID) and the take-profit/stop-loss
legs record the entry order id as their `parent_order_id` in the TradeLog trail.

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
                        marginleverage::Signed=0,
                        signal_label=nothing,
                        signal_score=nothing,
                        strategy_engine=nothing,
                        strategy_config_ref=nothing)
    @assert entry_side in (:buy, :sell) "entry_side must be :buy or :sell, got $entry_side"
    leg_group_id = string(UUIDs.uuid4())
    exit_buy = entry_side == :sell

    # Helper: set full context (signal info + leg metadata) and return it to the caller so
    # we can manage the clear ourselves rather than relying on createXorder's finally block.
    _setlegctx!(leg_label_str) = settradelogcontext!(xc;
        strategy_engine=something(strategy_engine, missing),
        strategy_config_ref=something(strategy_config_ref, missing),
        signal_label=something(signal_label, missing),
        signal_score=something(signal_score, missing),
        leg_group_id=leg_group_id,
        leg_label=leg_label_str,
    )

    # We call createXorder without leg_group_id/leg_label so it does NOT touch the context
    # (createXorder only calls settradelogcontext!/cleartradelogcontext! when those kwargs are
    # non-nothing).  We manage context ourselves here.

    # --- entry leg ---
    _setlegctx!("entry")
    entry_order_id = try
        if entry_side == :buy
            createbuyorder(xc, base;
                limitprice=Float32(entry_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
            )
        else
            createsellorder(xc, base;
                limitprice=Float32(entry_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
            )
        end
    finally
        cleartradelogcontext!(xc)
    end

    # --- take-profit leg ---
    _setlegctx!("take_profit")
    take_profit_order_id = try
        if exit_buy
            createbuyorder(xc, base;
                limitprice=Float32(take_profit_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
                parent_order_id=entry_order_id,
            )
        else
            createsellorder(xc, base;
                limitprice=Float32(take_profit_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
                parent_order_id=entry_order_id,
            )
        end
    finally
        cleartradelogcontext!(xc)
    end

    # --- stop-loss leg ---
    _setlegctx!("stop_loss")
    stop_loss_order_id = try
        if exit_buy
            createbuyorder(xc, base;
                limitprice=Float32(stop_loss_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
                parent_order_id=entry_order_id,
            )
        else
            createsellorder(xc, base;
                limitprice=Float32(stop_loss_price),
                basequantity=Float32(basequantity),
                maker=maker,
                marginleverage=marginleverage,
                parent_order_id=entry_order_id,
            )
        end
    finally
        cleartradelogcontext!(xc)
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
    bc = _routedbc(xc, trade_exchange_spot)
    if !(bc isa Bybit.BybitCache)
        throw(ArgumentError("_updateasset! requires Bybit cache for adapter-backed seeding, got $(typeof(bc))"))
    end
    Bybit.seedportfolio!(bc, coin, amount)
    ix = findfirst(==(uppercase(String(coin))), bc.assets[!, :coin])
    return isnothing(ix) ? nothing : bc.assets[ix, :]
end


_emptyassets()::DataFrame = DataFrame(coin=String31[], free=Float32[], locked=Float32[], marginfree=Float32[], marginlocked=Float32[], assetborrowed=Float32[], orderborrowed=Float32[], accruedinterest=Float32[])

"Return an empty order dataframe with CryptoXch bookkeeping columns added."
function _emptyorders(exchange::AbstractString=EXCHANGE_BYBIT)::DataFrame
    df = _exchangeemptyorders(exchange)
    if !hasproperty(df, :marginleverage)
        insertcols!(df, :marginleverage => Vector{Int32}(undef, 0))
    end
    return df
end

function _ordersfilestem(xc::XchCache)
    ORDERPREFIX = "Orders"
    fnvec = [ORDERPREFIX]
    if !isnothing(xc.mnemonic)
        push!(fnvec, xc.mnemonic)
    end
    push!(fnvec, string(EnvConfig.configmode))
    bases = sort(collect(keys(xc.bases)))
    fnvec = vcat(fnvec, bases)
    push!(fnvec, Dates.format(xc.startdt, "yy-mm-dd"))
    enddt = isnothing(xc.enddt) ? (size(xc.orders, 1) > 0 ? xc.orders[end, :created] : (size(xc.closedorders, 1) > 0 ? xc.closedorders[end, :created] : xc.startdt)) : xc.enddt
    push!(fnvec, Dates.format(enddt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_ordersfilename(xc::XchCache; format::Symbol=:arrow) = EnvConfig.tablepath(_ordersfilestem(xc); folderpath=EnvConfig.logfolder(), format=format)

function writeorders(xc::XchCache)
    fn = _ordersfilename(xc; format=:arrow)
    (verbosity >=0) && println("saving order log in filename=$fn")
    df = nothing
    if size(xc.closedorders, 1) > 0
        df = xc.closedorders
        if size(xc.orders, 1) > 0
            df = vcat(df, xc.orders)
        end
    elseif size(xc.orders, 1) > 0
        df = xc.orders
    else
        @warn "no orders to save in $fn"
        return
    end
    EnvConfig.savedf(df, _ordersfilestem(xc); folderpath=EnvConfig.logfolder(), format=:arrow)
    legacyfile = _ordersfilename(xc; format=:jdf)
    if isdir(legacyfile) || isfile(legacyfile)
        rm(legacyfile; force=true, recursive=true)
    end
end

function _assetsfilestem(xc::XchCache, dt)
    ASSETPREFIX = "Assets"
    fnvec = [ASSETPREFIX]
    if !isnothing(xc.mnemonic)
        push!(fnvec, xc.mnemonic)
    end
    push!(fnvec, string(EnvConfig.configmode))
    push!(fnvec, Dates.format(dt, "yy-mm-dd"))
    return join(fnvec, "_")
end

_assetsfilename(xc::XchCache, dt; format::Symbol=:arrow) = EnvConfig.tablepath(_assetsfilestem(xc, dt); folderpath=EnvConfig.logfolder(), format=format)

function writeassets(xc::XchCache, dt::DateTime)
    fn = _assetsfilename(xc, dt; format=:arrow)
    (verbosity >=3) && println("saving asset snapshot in filename=$fn")
    EnvConfig.savedf(xc.assets, _assetsfilestem(xc, dt); folderpath=EnvConfig.logfolder(), format=:arrow)
    legacyfile = _assetsfilename(xc, dt; format=:jdf)
    if isdir(legacyfile) || isfile(legacyfile)
        rm(legacyfile; force=true, recursive=true)
    end
end

#endregion bookkeeping

end  # of module
