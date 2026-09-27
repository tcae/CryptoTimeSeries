"""
`IbkrSpot` is an `XchAdapter` implementation for Interactive Brokers cash equities.

Unlike the crypto adapters it does not talk to an HTTP/websocket venue API but to a
locally running Trader Workstation (TWS) or IB Gateway over the TWS socket API.
`Jib` provides the wire protocol; this module adds the request/response correlation
layer that turns the asynchronous callback model into the synchronous adapter contract.
"""
module IbkrSpot

using CSV, DataFrames, Dates, EnvConfig, JSON3, Logging, Sockets, TimeZones
using XchAdapter
import XchAdapter: rawcache, exchangeid, symbolinfo, validsymbol, getklines, get24h, balances, positionsnapshot, accountsnapshot, emptyorders, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, executionorderspec, marginlimits, marginpermitted, marketdataheartbeats, marketdataheartbeat, wsorderssnapshot, wsordersheartbeat, wsbalancessnapshot, wsbalancesheartbeat, ws_orders, ws_balances, accountcapacity, closeorder, upsertcloseorder!, upsertopenorder!, directsequence!, wsclosedkline, preparetradingpairs!
import XchAdapter: normalize_order_status
import Jib

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

const EXECUTION_CONFIG_PATH = joinpath(@__DIR__, "..", "data", "execution_config.json")

"Load the IBKR execution configuration for connection settings and side-specific order limits."
function executionconfig()
    isfile(EXECUTION_CONFIG_PATH) || error("missing IbkrSpot execution config: $(EXECUTION_CONFIG_PATH)")
    return JSON3.read(read(EXECUTION_CONFIG_PATH, String))
end

"Return side-specific execution config owned by the IbkrSpot adapter."
function _executionorderspec(side::Symbol)
    side in (:long, :short) || error("IbkrSpot executionorderspec side=$(side) must be :long or :short")
    cfg = executionconfig()
    haskey(cfg, "orders") || error("missing IbkrSpot execution config orders section")
    orders = cfg["orders"]
    haskey(orders, String(side)) || error("missing IbkrSpot execution config orders.$(side) section")
    sidecfg = orders[String(side)]
    instrument = haskey(sidecfg, "instrument") ? lowercase(String(sidecfg["instrument"])) : ""
    leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
    max_quote = haskey(sidecfg, "max_quote") ? sidecfg["max_quote"] : nothing
    return (side=side, instrument=instrument, leverage=leverage, max_quote=max_quote)
end

"Return the connection section of the execution config."
function _connectionconfig()
    cfg = executionconfig()
    haskey(cfg, "connection") || error("missing IbkrSpot execution config connection section")
    conn = cfg["connection"]
    configured_port = Int(get(conn, "port", 7497))
    port_override = get(ENV, "IBKR_PORT", nothing)
    return (
        host=String(get(conn, "host", "127.0.0.1")),
        port=isnothing(port_override) ? configured_port : parse(Int, port_override),
        clientid=Int(get(conn, "clientid", 17)),
        exchange_route=String(get(conn, "exchange_route", "SMART")),
        primary_exchange=String(get(conn, "primary_exchange", "")),
        currency=String(get(conn, "currency", "USD")),
        account=String(get(conn, "account", "")),
    )
end

#region request correlation

# TWS delivers every response asynchronously, tagged with the request id that the client
# chose. `_collector` maps that id to the buffer a waiting caller is blocked on.
const _reqid_lock = ReentrantLock()
const _reqid_counter = Ref{Int}(1000)
const _collector_lock = ReentrantLock()
const _collectors = Dict{Int, Vector{Any}}()
const _completed = Dict{Int, Bool}()
const _failed = Dict{Int, String}()

const REQUEST_TIMEOUT = Dates.Second(20)
const REQUEST_POLL = 0.02

"Allocate a process-unique TWS request id."
function _nextreqid()::Int
    lock(_reqid_lock) do
        _reqid_counter[] += 1
        return _reqid_counter[]
    end
end

"Register a response buffer for `reqid` before the request is sent."
function _opencollector!(reqid::Int)
    lock(_collector_lock) do
        _collectors[reqid] = Any[]
        _completed[reqid] = false
        delete!(_failed, reqid)
    end
    return reqid
end

"Append one response element to the buffer of `reqid`, ignoring unsolicited ids."
function _push!(reqid::Int, value)
    lock(_collector_lock) do
        haskey(_collectors, reqid) && push!(_collectors[reqid], value)
    end
    return nothing
end

"Mark `reqid` complete so a waiting caller stops polling."
function _complete!(reqid::Int)
    lock(_collector_lock) do
        haskey(_completed, reqid) && (_completed[reqid] = true)
    end
    return nothing
end

"Mark `reqid` failed so the waiting caller throws instead of timing out."
function _fail!(reqid::Int, message::AbstractString)
    lock(_collector_lock) do
        if haskey(_completed, reqid)
            _failed[reqid] = String(message)
            _completed[reqid] = true
        end
    end
    return nothing
end

"Discard the buffer of `reqid`."
function _closecollector!(reqid::Int)
    lock(_collector_lock) do
        delete!(_collectors, reqid)
        delete!(_completed, reqid)
        delete!(_failed, reqid)
    end
    return nothing
end

"""
Block until `reqid` is completed by a callback and return the collected elements.

Throws on TWS-reported request errors and on timeout, because a missing response means
the adapter cannot honour the synchronous contract its callers rely on.
"""
function _await(reqid::Int, what::AbstractString; timeout::Dates.Period=REQUEST_TIMEOUT)
    deadline = Dates.now(UTC) + timeout
    while Dates.now(UTC) < deadline
        done, err, values = lock(_collector_lock) do
            (get(_completed, reqid, false), get(_failed, reqid, nothing), get(_collectors, reqid, Any[]))
        end
        if done
            _closecollector!(reqid)
            isnothing(err) || error("IbkrSpot $(what) failed reqid=$(reqid): $(err)")
            return values
        end
        sleep(REQUEST_POLL)
    end
    _closecollector!(reqid)
    error("IbkrSpot $(what) timed out reqid=$(reqid) after $(timeout)")
end

#endregion request correlation

#region streamed account and order state

const _state_lock = ReentrantLock()
const _accountvalues = Dict{String, Dict{String, Float64}}()  # account => tag => value
const _portfolio = Dict{String, NamedTuple}()                 # symbol => position snapshot
const _orderstate = Dict{Int, Dict{Symbol, Any}}()            # orderId => merged openOrder/orderStatus fields
const _nextvalidid = Ref{Int}(0)
const _managedaccounts = Ref{String}("")
const _currenttime_reqid = Ref{Int}(0)
const _position_reqid = Ref{Int}(0)
const _openorders_reqid = Ref{Int}(0)
const _lasttick = Dict{String, Dict{Int, Float64}}()          # symbol => tickType => price
const _marketdata_heartbeat_by_symbol = Dict{String, DateTime}()
const _marketdata_heartbeat = Ref{Union{Nothing, DateTime}}(nothing)
const _orders_heartbeat = Ref{Union{Nothing, DateTime}}(nothing)
const _balances_heartbeat = Ref{Union{Nothing, DateTime}}(nothing)
const _liquidations = Vector{NamedTuple}()

# TWS reports market-data farm connection state through the error channel; these are
# status notices rather than request failures and must not abort a pending request.
const _INFO_ERROR_CODES = Set([1100, 1101, 1102, 2103, 2104, 2105, 2106, 2107, 2108, 2119, 2158])

"Reserve the next client-side order id handed out by TWS at connect time."
function _takeorderid()::Int
    lock(_state_lock) do
        @assert _nextvalidid[] > 0 "IbkrSpot has no valid order id yet; connection not established"
        id = _nextvalidid[]
        _nextvalidid[] = id + 1
        return id
    end
end

#endregion streamed account and order state

include("adapter.jl")
include("watchlists.jl")

end  # module
