module Bybit

using HTTP, SHA, JSON3, Dates, Printf, Logging, DataFrames, InlineStrings, Format, Downloads
using EnvConfig
using Ohlcv
using TestOhlcv
using XchAdapter
import XchAdapter: rawcache, exchangeid, symbolinfo, validsymbol, getklines, get24h, balances, positionsnapshot, accountsnapshot, emptyorders, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, executionorderspec, accountcapacity, closeorder, upsertcloseorder!, upsertopenorder!, directsequence!, drainliquidations!, preparetradingpairs!
import XchAdapter: normalize_order_status

# base URL of the ByBit API
# BYBIT_API_REST = "https://api.bybit.com"
# BYBIT_API_WS = "to be defined for Bybit"  # "wss://stream.binance.com:9443/ws/"
# BYBIT_API_USER_DATA_STREAM ="to be defined for Bybit"

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

const EXECUTION_CONFIG_PATH = joinpath(@__DIR__, "..", "data", "execution_config.json")

"Load the Bybit-side execution configuration for side-specific order limits and instruments."
function executionconfig()
    isfile(EXECUTION_CONFIG_PATH) || error("missing Bybit execution config: $(EXECUTION_CONFIG_PATH)")
    return JSON3.read(read(EXECUTION_CONFIG_PATH, String))
end

function _executionconfigside(configside::Union{Nothing, Symbol}, orderside::AbstractString)::Symbol
    if isnothing(configside)
        return lowercase(String(orderside)) == "buy" ? :long : :short
    end
    side = Symbol(lowercase(String(configside)))
    @assert side in (:long, :short) "invalid Bybit configside=$(configside)"
    return side
end

function _executionorderspec(configside::Union{Nothing, Symbol}, orderside::AbstractString)
    side = _executionconfigside(configside, orderside)
    cfg = executionconfig()
    orders = cfg["orders"]
    sidecfg = orders[String(side)]
    instrument = lowercase(String(sidecfg["instrument"]))
    max_quote = haskey(sidecfg, "max_quote") ? sidecfg["max_quote"] : nothing
    leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
    return (side=side, instrument=instrument, max_quote=max_quote, leverage=leverage)
end

"Compatibility overload for legacy call sites still passing margin leverage explicitly."
function _executionorderspec(configside::Union{Nothing, Symbol}, orderside::AbstractString, marginleverage::Signed)
    spec = _executionorderspec(configside, orderside)
    leverage = (spec.leverage == 0 && marginleverage > 0) ? marginleverage : spec.leverage
    return (side=spec.side, instrument=spec.instrument, max_quote=spec.max_quote, leverage=leverage)
end

"Return side-specific execution config owned by the Bybit adapter."
function _executionorderspec(side::Symbol)
    side in (:long, :short) || error("Bybit executionorderspec side=$(side) must be :long or :short")
    cfg = executionconfig()
    haskey(cfg, "orders") || error("missing Bybit execution config orders section")
    orders = cfg["orders"]
    haskey(orders, String(side)) || error("missing Bybit execution config orders.$(side) section")
    sidecfg = orders[String(side)]
    instrument = haskey(sidecfg, "instrument") ? lowercase(String(sidecfg["instrument"])) : ""
    leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
    max_quote = haskey(sidecfg, "max_quote") ? (sidecfg["max_quote"]) : nothing
    return (side=side, instrument=instrument, leverage=leverage, max_quote=max_quote)
end

function _enforce_maxquote_policy(spec, symbol::AbstractString, basequantity::Real, price::Union{Real, Nothing}, reduceonly::Bool)
    if isnothing(spec.max_quote) || isnothing(price)
        return nothing
    end
    notional = (basequantity) * (price)
    if notional <= spec.max_quote + 1e-9
        return nothing
    end
    if reduceonly
        throw(ArgumentError("Bybit oversized reduce-only order is not yet supported on the current spot/spot-margin path; symbol=$(symbol) configside=$(spec.side) notional=$(notional) max_quote=$(spec.max_quote)"))
    end
    throw(ArgumentError("Bybit oversized opening order requires adapter-side websocket sequencing; symbol=$(symbol) configside=$(spec.side) notional=$(notional) max_quote=$(spec.max_quote)"))
end

const _recvwindow = "5000000"  # "5000" extended by factor 1000 due to nanoseconds in julia
const _sim_order_counter = IdDict{Any, Int64}()
# Per-cache successor orderid => predecessor orderid registered via directsequence!.
const _sim_sequencing = IdDict{Any, Dict{String, String}}()
"Resolved trading pair symbol per cache, keyed by the caller supplied base*quote spelling."
const _symboltoken_memo = IdDict{Any, Dict{String, String}}()
"Queued forced-liquidation events per simulation cache, drained by Xch into TSM trades rows."
const _sim_liquidations = IdDict{Any, Vector{NamedTuple}}()
"Persisted OHLCV per (base, interval), loaded once from disk and reused by every BybitSim kline/price lookup instead of re-reading on every tick."
const _sim_ohlcv_cache = Dict{Tuple{String,String}, Ohlcv.OhlcvData}()

const _bybitsim_test_basecoins = ("SINE", "DOUBLESINE")
const _klineinterval = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W"]
const interval2bybitinterval = Dict(
    "1m" => "1",
    "3m" => "3",
    "5m" => "5",
    "15m" => "15",
    "30m" => "30",
    "1h" => "60",
    "2h" => "120",
    "4h" => "240",
    "6h" => "360",
    "12h" => "720",
    "1d" => "D",
    "1w" => "W"
)

# Balance caching to avoid Bybit API rate limits
# Cache TTL: 5 seconds. At 1 balance call/minute, this is well under rate limits.
const _balance_cache_lock = ReentrantLock()
const _balance_cache = Ref{Union{Nothing, DataFrame}}(nothing)
const _balance_cache_time = Ref{Union{Nothing, DateTime}}(nothing)
const BALANCE_CACHE_TTL = Dates.Second(5)

"Bybit exchange cache supporting both production API and simulation mode (BybitSim).
When used in BybitSim mode, assets and orderbook track simulated bookkeeping."
mutable struct BybitCache <: XchAdapter.XchAdapterCache
    syminfodf::Union{Nothing, DataFrame}
    apirest::String
    publickey
    secretkey
    simtime::Union{Nothing, DateTime}
    lastpendingdecisiondt::Union{Nothing, DateTime}
    # Simulation state (populated only in BybitSim mode, nil in production).
    # Rows represent holdings lanes with side in {quote,long,short}.
    assets::Union{Nothing, DataFrame}
    orderbook::Union{Nothing, DataFrame}
    orderindex::Dict{String, Int}
    openorderindex::Vector{Int}
    tradingpairepoch::UInt
    tradingpairinfo::Vector{NamedTuple}
    # Shared reference to the owning XchCache's per-base OHLCV cache (duck-typed wiring via
    # Xch.setcurrenttime!, mirrors `simtime`). Same Dict object, no per-simulation copy.
    ohlcvcache::Union{Nothing, Dict{String, Ohlcv.OhlcvData}}
end

executionorderspec(bc::BybitCache, side::Symbol) = _executionorderspec(side)
const BybitSimCache = BybitCache
const BybitsimCache = BybitCache
exchangeid(bc::BybitCache)::String = "Bybit"

"Prepare direct Bybit exchange-info access for one Xch trading-pair epoch."
function preparetradingpairs!(bc::BybitCache, pairrefs::Vector{XchAdapter.TradingPairRef})
    isempty(pairrefs) && (bc.tradingpairepoch = UInt(0); empty!(bc.tradingpairinfo); return nothing)
    epoch = pairrefs[1].epoch
    info = NamedTuple[]
    sizehint!(info, length(pairrefs))
    for ref in pairrefs
        @assert ref.epoch == epoch && ref.cfgindex > 0 "Bybit pair reference must have matching nonzero epoch/index: pair=$(ref.pair) cfgindex=$(ref.cfgindex) epoch=$(ref.epoch)"
        @assert ref.cfgindex == length(info) + 1 "Bybit pair references must be ordered by cfgindex: pair=$(ref.pair) cfgindex=$(ref.cfgindex) expected=$(length(info) + 1)"
        ix = findfirst(==(ref.pair), bc.syminfodf[!, :symbol])
        @assert !isnothing(ix) "Bybit exchange info missing canonical trading pair=$(ref.pair)"
        row = bc.syminfodf[ix, :]
        push!(info, (pair=ref.pair, symbol=String(row.symbol), syminfoix=UInt(ix), ticksize=row.ticksize, baseprecision=row.baseprecision, quoteprecision=row.quoteprecision, minbaseqty=row.minbaseqty, minquoteqty=row.minquoteqty))
    end
    bc.tradingpairepoch = epoch
    bc.tradingpairinfo = info
    return nothing
end

"Return prepared Bybit metadata for a current epoch pair reference."
function _preparedpairinfo(bc::BybitCache, pairref::XchAdapter.TradingPairRef)
    if pairref.epoch == 0
        return nothing
    end
    @assert pairref.epoch == bc.tradingpairepoch "Bybit pair epoch mismatch: pair=$(pairref.pair) ref.epoch=$(pairref.epoch) adapter.epoch=$(bc.tradingpairepoch)"
    @assert pairref.cfgindex > 0 "Bybit prepared pair reference requires cfgindex > 0: pair=$(pairref.pair) epoch=$(pairref.epoch)"
    @assert pairref.cfgindex <= length(bc.tradingpairinfo) "Bybit pair cfgindex=$(pairref.cfgindex) exceeds prepared pairs=$(length(bc.tradingpairinfo))"
    info = bc.tradingpairinfo[pairref.cfgindex]
    @assert info.pair == pairref.pair "Bybit pair index mismatch: ref.pair=$(pairref.pair) cfgindex=$(pairref.cfgindex) indexed.pair=$(info.pair)"
    return info
end

"Return (creating if needed) the successor=>predecessor sequencing map for one simulation cache."
function _sim_sequencing_for(bc::BybitCache)::Dict{String, String}
    return get!(() -> Dict{String, String}(), _sim_sequencing, bc)
end

"Return the incrementally maintained orderid => row-index map for the simulation orderbook."
function _sim_orderindex_for(bc::BybitCache)::Dict{String, Int}
    isnothing(bc.orderbook) && (empty!(bc.orderindex); return bc.orderindex)
    @assert length(bc.orderindex) == nrow(bc.orderbook) "BybitSim order index length=$(length(bc.orderindex)) must match orderbook rows=$(nrow(bc.orderbook))"
    return bc.orderindex
end

"Return the incrementally maintained open-order row indices for a simulation cache."
function _sim_openorderindex_for(bc::BybitCache)::Vector{Int}
    isnothing(bc.orderbook) && (empty!(bc.openorderindex); return bc.openorderindex)
    return bc.openorderindex
end

"Rebuild simulation indexes after initialization or restoring a persisted orderbook."
function _simrebuildorderindexes!(bc::BybitCache)
    book = bc.orderbook
    @assert !isnothing(book) "BybitSim orderbook must exist when rebuilding indexes"
    orderidx = bc.orderindex
    openix = bc.openorderindex
    empty!(orderidx)
    empty!(openix)
    for ix in 1:nrow(book)
        orderid = String(book[ix, :orderid])
        @assert !haskey(orderidx, orderid) "BybitSim orderbook contains duplicate orderid=$(orderid)"
        orderidx[orderid] = ix
        _isopenstatus(String(book[ix, :status])) && push!(openix, ix)
    end
    return nothing
end

"Append one simulation orderbook row and update its indexes without rescanning the ledger."
function _simappendorder!(bc::BybitCache, row)
    book = bc.orderbook
    @assert !isnothing(book) "BybitSim orderbook must exist when appending order"
    orderidx = _sim_orderindex_for(bc)
    openix = _sim_openorderindex_for(bc)
    orderid = String(row.orderid)
    @assert !haskey(orderidx, orderid) "BybitSim orderbook already contains orderid=$(orderid)"
    push!(book, row)
    ix = nrow(book)
    orderidx[orderid] = ix
    if _isopenstatus(String(row.status))
        isempty(openix) || @assert last(openix) < ix "BybitSim open-order indices must be ascending: last=$(last(openix)) appended=$(ix)"
        push!(openix, ix)
    end
    return book[ix, :]
end

"Mark one orderbook row terminal while retaining it for order-id lookup and replay audit."
function _simfinalizeorder!(bc::BybitCache, ix::Int, status::String, updated::DateTime; rejectreason::String="NO ERROR", avgprice=nothing, executedqty=nothing)
    book = bc.orderbook
    @assert !isnothing(book) "BybitSim orderbook must exist when finalizing order index=$(ix)"
    openix = _sim_openorderindex_for(bc)
    openpos = findfirst(==(ix), openix)
    @assert !isnothing(openpos) "BybitSim terminal order index=$(ix) is missing from open-order index=$(openix)"
    deleteat!(openix, openpos)
    book[ix, :status] = status
    book[ix, :updated] = updated
    book[ix, :lastcheck] = updated
    book[ix, :rejectreason] = rejectreason
    isnothing(avgprice) || (book[ix, :avgprice] = avgprice)
    isnothing(executedqty) || (book[ix, :executedqty] = executedqty)
    return book[ix, :]
end

"Return (creating if needed) the queued liquidation events for one simulation cache."
function _sim_liquidations_for(bc::BybitCache)::Vector{NamedTuple}
    return get!(() -> NamedTuple[], _sim_liquidations, bc)
end

"Return and clear the queued liquidation events for one simulation cache."
function drainliquidations!(bc::BybitCache)::Vector{NamedTuple}
    events = _sim_liquidations_for(bc)
    drained = copy(events)
    empty!(events)
    return drained
end

"""Normalize Bybit raw order status into Xch status vocabulary."""
function normalize_order_status(bc::BybitCache, rawstatus::AbstractString)::String
    st = lowercase(String(rawstatus))
    if st in ["created", "new", "untriggered", "triggered", "partiallyfilled"]
        return "submitted"
    elseif st in ["filled"]
        return "closed"
    elseif st in ["cancelled", "deactivated"]
        return "cancelled"
    elseif st in ["rejected"]
        return "rejected"
    end
    return st
end

BYBIT_APIREST = "https://api.bybit.com"
BYBIT_TESTNET_APIREST = "https://api-testnet.bybit.com"

"Initializes Bybit if testnet==true then the Bybit Testnet is used"
function BybitCache(testnet::Bool=EnvConfig.configmode == EnvConfig.test, publickey::Union{Nothing, AbstractString}=nothing, secretkey::Union{Nothing, AbstractString}=nothing)::BybitCache
    apirest = testnet ? BYBIT_TESTNET_APIREST : BYBIT_APIREST
    if isnothing(publickey) || isnothing(secretkey)
        if isnothing(EnvConfig.authorization)
            pk = ""
            sk = ""
        else
            pk = String(EnvConfig.authorization.key)
            sk = String(EnvConfig.authorization.secret)
        end
    else
        pk = String(publickey)
        sk = String(secretkey)
    end
    bc = BybitCache(nothing, apirest, pk, sk, nothing, nothing, nothing, nothing, Dict{String, Int}(), Int[], UInt(0), NamedTuple[], nothing)
    xchinfo = _exchangeinfo(bc)
    xchinfo = sort!(xchinfo[xchinfo.quotecoin .== EnvConfig.pairquote, :], :basecoin)
    @assert (!isnothing(xchinfo)) && (size(xchinfo, 1) > 0) "missing exchangeinfo isnothing(xchinfo)=$(isnothing(xchinfo)) size(xchinfo, 1)=$(size(xchinfo, 1))"
    bc = BybitCache(xchinfo, apirest, pk, sk, nothing, nothing, nothing, nothing, Dict{String, Int}(), Int[], UInt(0), NamedTuple[], nothing)
    EnvConfig.setcoinspath!(exchangeid(bc))
	EnvConfig.setpairquote!("USDT")
    if EnvConfig.configmode == EnvConfig.test
        _init_simulation!(bc)
    end
    return bc
end

"Initialize simulation state (assets and orderbook) for BybitSim mode"
function _init_simulation!(bc::BybitCache)
    _ensure_sim_symboluniverse!(bc)
    bc.lastpendingdecisiondt = nothing
    if isnothing(bc.assets)
        bc.assets = DataFrame(coin=String31[], side=String7[], free=Float32[], locked=Float32[], collateral=Float32[], proceeds=Float32[])
        bc.orderbook = DataFrame(orderid=String[], symbol=String[], side=String[], positionside=String[], lane=String[], baseqty=Float32[], ordertype=String[], isLeverage=Bool[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[], marginleverage=Int32[], reduceonly=Bool[])
        _simrebuildorderindexes!(bc)
    end
    haskey(_sim_order_counter, bc) || (_sim_order_counter[bc] = 0)
    return bc
end

function _ensure_sim_symboluniverse!(bc::BybitCache)
    isnothing(bc.syminfodf) && return bc
    for base in _bybitsim_test_basecoins
        symbol = uppercase(string(base, EnvConfig.pairquote))
        ix = findfirst(==(symbol), bc.syminfodf[!, :symbol])
        if isnothing(ix)
            push!(bc.syminfodf, (
                symbol=symbol,
                status="Trading",
                basecoin=String(base),
                quotecoin=String(EnvConfig.pairquote),
                ticksize=1f-6,
                baseprecision=1f-5,
                quoteprecision=1f-6,
                minbaseqty=1f-5,
                minquoteqty=1f0,
                innovation=0,
            ))
        end
    end
    return bc
end

"Return the next per-cache simulation order sequence number."
function _nextsimorderseq!(bc::BybitCache)::Int64
    seq = get(_sim_order_counter, bc, 0) + 1
    _sim_order_counter[bc] = seq
    return seq
end

"Seed simulation portfolio with an initial balance"
function seedportfolio!(bc::BybitCache, coin::AbstractString, free::Real; locked::Real=0, side::Union{Nothing, Symbol}=nothing)
    isnothing(bc.assets) && _init_simulation!(bc)
    coinup = uppercase(String(coin))
    qcoin = uppercase(String(EnvConfig.pairquote))
    laneside = isnothing(side) ? (coinup == qcoin ? :quote : :long) : Symbol(lowercase(String(side)))
    @assert laneside in (:quote, :long, :short) "seedportfolio! side=$(laneside) must be :quote, :long, or :short"

    ix = findfirst(((bc.assets[!, :coin] .== coinup) .& (bc.assets[!, :side] .== String(laneside))))
    if isnothing(ix)
        row = (coin=coinup, side=String(laneside), free=(free), locked=(locked), collateral=0f0, proceeds=0f0)
        push!(bc.assets, row; cols=:intersect)
    else
        bc.assets[ix, :free] = (free)
        bc.assets[ix, :locked] = (locked)
    end
    return bc
end

function apiKS()
    apiPublicKey = get(ENV, "BYBIT_APIKEY", "")
    apiSecretKey = get(ENV, "BYBIT_SECRET", "")

    @assert apiPublicKey != "" || apiSecretKey != "" "BYBIT_APIKEY/BYBIT_APISECRET should be present in the environment dictionary ENV"

    apiPublicKey, apiSecretKey
end

function _dict2paramsget(dict::Union{Dict, Nothing})
    params = ""
    if isnothing(dict)
        return params
    else
        for kv in dict
            params = string(params, "&$(kv[1])=$(kv[2])")
        end
        return params[2:end]
    end
end

_dict2paramspost(dict::Union{Dict, Nothing}) = isnothing(dict) ? "" : JSON3.write(dict)

function timestamp()
    Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000))
    # if Sys.isapple()
    #     Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000))
    # else
    #     Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC))))
    #     # Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC))))
    # end
    # if Sys.islinux()
    #     # rootpath = joinpath(@__DIR__, "..")
    #     println("Linux, rootpath: $rootpath, homepath: $(homedir())")
    # elseif Sys.isapple()
    #     # rootpath = joinpath(@__DIR__, "..")
    #     println("Apple, rootpath: $rootpath, homepath: $(homedir())")
    # elseif Sys.iswindows()
    #     # rootpath = joinpath(@__DIR__, "..")
    #     println("Windows, rootpath: $rootpath, homepath: $(homedir())")
    # else
    #     # rootpath = joinpath(@__DIR__, "..")
    #     println("unknown OS, rootpath: $rootpath, homepath: $(homedir())")
    # end
end

function _hmac(key::Vector{UInt8}, msg::Vector{UInt8}, hash, blocksize::Int=64)
    if length(key) > blocksize
        key = hash(key)
    end

    pad = blocksize - length(key)

    if pad > 0
        resize!(key, blocksize)
        key[end - pad + 1:end] .= 0
    end

    o_key_pad = key .⊻ 0x5c
    i_key_pad = key .⊻ 0x36

    hash([o_key_pad; hash([i_key_pad; msg])])
end

function _dosign(queryString, apiSecret)
    bytes2hex(_hmac(Vector{UInt8}(apiSecret), Vector{UInt8}(queryString), SHA.sha256))
end

function _gensignature(time_stamp, payload, public_key, secret_key)
    param_str = time_stamp * public_key * _recvwindow * payload
    hash = _dosign(param_str, secret_key)
    return hash
end

function _checkresponse(response)
    if response.status != 200  # response.status::Int16
        @warn "HTTP response=$response"
    end
    for header in response.headers  # response.headers::Vector{pair}
        if (header[1] == "X-Bapi-Limit-Status") && (parse(Int, header[2]) == 1)
            @warn "h1=$(header[1]) h2=$(header[2]) fullheader=$(header) waiting for 1s"
            sleep(1)
        end
        # if (header[1] == "X-Bapi-Limit-Status")
        #     remaining = parse(Int, header[2])
        #     println("remaining=$remaining")
        # end
    end
end

function HttpPrivateRequest(bc::BybitCache, method, endPoint, params, info)
    methodpost = method == "POST"
    url = headers = payload = returnbody = body = nothing
    nextrequestrequired = true
    requestcount = 0
    try
        while nextrequestrequired
            payload = methodpost ? _dict2paramspost(params) : _dict2paramsget(params)
            time_stamp = string(timestamp())
            signature = _gensignature(time_stamp, payload, bc.publickey, bc.secretkey)
            headers = Dict(
                "X-BAPI-API-KEY" => bc.publickey,
                "X-BAPI-SIGN" => signature,
                "X-BAPI-SIGN-TYPE" => "2",
                "X-BAPI-TIMESTAMP" => time_stamp,
                "X-BAPI-RECV-WINDOW" => _recvwindow,
                "Content-Type" => "application/json"  # ; charset=utf-8"
            )
            response = url = ""
            httptry = 1
            while httptry > 0
                try
                    (verbosity >= 4) && print("\n$(EnvConfig.now()) HttpPrivateRequest httptry=$httptry $info #$requestcount $method response=$body url=$url headers=$headers payload=$payload")
                    if methodpost
                        # headers["Content-Type"] = "application/json; charset=utf-8"
                        url = bc.apirest * endPoint
                        response = HTTP.request(method, url, headers, payload; retry_non_idempotent = true, retries = 10, readtimeout = 60)
                    else
                        url = bc.apirest * endPoint * "?" * payload
                        response = HTTP.request(method, url, headers; retry = true, retries = 10, readtimeout = 60)
                    end
                    (verbosity >= 4) && println(" $(EnvConfig.now()) HttpPrivateRequest response=$response  done")
                    httptry -= 1
                    #TODO check ratelimit overrun
                    _checkresponse(response)
                    body = String(response.body)
                    body = JSON3.read(body, Dict)
                    body = _dictstring2values!(body)
                    if occursin("Too many visits!", body["retMsg"])
                        @warn "Too many visits! - waiting 5 seconds"
                        sleep(5) # wait 5 seconds
                    end
                catch httperr
                    if (occursin("DNSError", string(httperr)) || occursin("ReadTimeoutError", string(httperr))) && (5 >= httptry > 0)
                        (verbosity >= 1) && @info "HttpPrivateRequest httptry=$httptry $info #$requestcount $method response=$body \nurl=$url \nheaders=$headers \npayload=$payload \nexception=$httperr"
                        sleep(5 * httptry) # sleep (5 seconds x number of retry) then retry = sleep with every retry longer
                        httptry += 1
                        continue
                    end
                    (verbosity >= 1) && @info "exception=$httperr within core HttpPrivateRequest: httptry=$httptry info=$info #$requestcount $method response=$body \nurl=$url \nheaders=$headers \npayload=$payload"
                    rethrow()
                end
            end
            requestcount += 1
            if (body["retCode"] != 0) && (body["retCode"] != 170213)  # 170213 == cancelorder: Order does not exist.
                @warn "HttpPrivateRequest $info #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body"
                println("server time $(servertime(bc)) X-BAPI-TIMESTAMP $(Dates.unix2datetime(parse(Int, time_stamp)))")
                # println("public_key=$public_key, secret_key=$secret_key")
                # "retCode" => 170193, "retMsg" => "Buy order price cannot be higher than 43183.1929USDT."
            end
            # @info "$(Dates.now()) HttpPrivateRequest #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body \nreturnbody=$(string(returnbody))"
            # println("$(EnvConfig.now()) body=$body \nreturnbody=$(string(returnbody))")
            nextrequestrequired = ("result" in keys(body)) && ("nextPageCursor" in keys(body["result"])) && (length(body["result"]["nextPageCursor"]) > 0) && ("list" in keys(body["result"]))
            # nextrequestrequired = (requestcount <=3) && ("result" in keys(body)) && ("nextPageCursor" in keys(body["result"])) && (length(body["result"]["nextPageCursor"]) > 0) && ("list" in keys(body["result"]))
            if nextrequestrequired
                params["cursor"] = body["result"]["nextPageCursor"]
                if !isnothing(returnbody) && (length(returnbody["result"]["list"]) > 0)
                    returnbody["result"]["list"] = vcat(returnbody["result"]["list"], body["result"]["list"])
                end
                delete!(body["result"], "nextPageCursor")
            end
            returnbody = isnothing(returnbody) ? body : returnbody  # 1st time in the loop returnbody=body, in following loops body is appended
        end
    catch err
        if !isa(err, InterruptException)
            @error "HttpPrivateRequest $info #$requestcount $method response=$body \nurl=$url \nheaders=$headers \npayload=$payload \nexception=$err"
        end
        rethrow()
    end
    return returnbody
end

function HttpPublicRequest(bc::BybitCache, method, endPoint, params::Union{Dict, Nothing}, info)
    methodpost = method == "POST"
    payload = isnothing(params) ? "" : (methodpost ? _dict2paramspost(params) : _dict2paramsget(params))
    url = bc.apirest * endPoint
    if !methodpost && !isempty(payload)
        url *= "?" * payload
    end

    body = Dict()
    try
        io = IOBuffer()
        if methodpost
            Downloads.request(url; method=method, headers=["Content-Type" => "application/json"], input=IOBuffer(payload), output=io)
        else
            Downloads.request(url; method=method, output=io)
        end
        body = JSON3.read(String(take!(io)), Dict)
        body = _dictstring2values!(body)
        if body["retCode"] != 0
            @warn "HttpPublicRequest $method, url=$url, payload=$payload, response=$body"
        end
        return body
    catch err
        @error "HttpPublicRequest $method failed, url=$url, payload=$payload, response=$body, exception=$err"
        rethrow()
    end
end

function HttpPrivateRequest(method, endPoint, params, info, public_key=EnvConfig.authorization.key, secret_key=EnvConfig.authorization.secret)
    @assert !isnothing(BYBIT_APIREST) "Bybit.init() not yet done resulting in missing URL"
    methodpost = method == "POST"
    url = headers = payload = returnbody = body = nothing
    nextrequestrequired = true
    requestcount = 0
    try
        while nextrequestrequired
            payload = methodpost ? _dict2paramspost(params) : _dict2paramsget(params)
            time_stamp = string(timestamp())
            signature = _gensignature(time_stamp, payload, public_key, secret_key)
            headers = Dict(
                "X-BAPI-API-KEY" => public_key,
                "X-BAPI-SIGN" => signature,
                "X-BAPI-SIGN-TYPE" => "2",
                "X-BAPI-TIMESTAMP" => time_stamp,
                "X-BAPI-RECV-WINDOW" => _recvwindow,
                "Content-Type" => "application/json"  # ; charset=utf-8"
            )
            response = url = ""
            if methodpost
                # headers["Content-Type"] = "application/json; charset=utf-8"
                url = BYBIT_APIREST * endPoint
                response = HTTP.request(method, url, headers, payload)
            else
                url = BYBIT_APIREST * endPoint * "?" * payload
                response = HTTP.request(method, url, headers)
            end
            requestcount += 1
            _checkresponse(response)
            body = String(response.body)
            body = JSON3.read(body, Dict)
            body = _dictstring2values!(body)
            if body["retCode"] != 0
                @warn "HttpPrivateRequest $info #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body"
                # println("public_key=$public_key, secret_key=$secret_key")
                # "retCode" => 170193, "retMsg" => "Buy order price cannot be higher than 43183.1929USDT."
            end
            # @info "$(Dates.now()) HttpPrivateRequest #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body \nreturnbody=$(string(returnbody))"
            # println("$(EnvConfig.now()) body=$body \nreturnbody=$(string(returnbody))")
            nextrequestrequired = (requestcount <=3) && ("result" in keys(body)) && ("nextPageCursor" in keys(body["result"])) && (length(body["result"]["nextPageCursor"]) > 0) && ("list" in keys(body["result"]))
            if nextrequestrequired
                params["cursor"] = body["result"]["nextPageCursor"]
                if !isnothing(returnbody) && (length(returnbody["result"]["list"]) > 0)
                    returnbody["result"]["list"] = vcat(returnbody["result"]["list"], body["result"]["list"])
                end
                delete!(body["result"], "nextPageCursor")
            end
            returnbody = isnothing(returnbody) ? body : returnbody
        end
    catch err
        @error "HttpPrivateRequest $info #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body"
        # println("public_key=$public_key, secret_key=$secret_key")
        # println(err)
        # rethrow()
    end
    return returnbody
end

function HttpPublicRequest(method, endPoint, params::Union{Dict, Nothing}, info)
    return HttpPrivateRequest(method, endPoint, params, info)

    methodpost = method == "POST"
    payload = methodpost ? _dict2paramspost(params) : _dict2paramsget(params)
    response = url = ""
    body = Dict()
    try
        if methodpost
            url = BYBIT_APIREST * endPoint
            response = HTTP.request(method, url, payload)
        elseif isnothing(params)
            url = BYBIT_APIREST * endPoint
            response = HTTP.request(method, url)
        else
            url = BYBIT_APIREST * endPoint * "?" * payload
            response = HTTP.request(method, url)
        end
        _checkresponse(response)
        body = String(response.body)
        body = JSON3.read(body, Dict)
        body = _dictstring2values!(body)
        # println("conv: $body")
        if body["retCode"] != 0
            println("HttpPublicRequest $method, url=$url, payload=$payload, response=$body")
        end
        return body
    catch err
        println("HttpPublicRequest $method, url=$url, payload=$payload, response=$body")
        println(err)
        rethrow()
    end
end

# function HTTP response 2 JSON
function _r2j(response)
    JSON3.read(String(response), Dict)
end

function _dictstring2values!(bybitdict::T) where T <: AbstractDict
    f32keys = [
        "price", "qty", "avgPrice", "leavesQty", "leavesValue", "cumExecQty",
        "cumExecValue", "cumExecFee", "orderIv", "triggerPrice", "takeProfit",
        "stopLoss", "tpLimitPrice", "slLimitPrice", "lastPriceOnCreated",
        "ask1Price", "usdIndexPrice", "indexPrice", "markPrice", "lastPrice", "prevPrice24h", "ask1Size",
        "highPrice24h", "turnover24h", "bid1Size", "price24hPcnt", "volume24h",
        "lowPrice24h", "bid1Price", "prevPrice1h", "openInterest", "openInterestValue",
        "turnover24h", "fundingRate", "predictedDeliveryPrice", "basisRate", "deliveryFeeRate",
        "maxLeverage", "minLeverage", "leverageStep", "minPrice", "maxPrice", "tickSize",
        "maxTradingQty", "minTradingQty", "qtyStep", "postOnlyMaxOrderQty", "maxOrderQty",
        "minOrderQty", "minTradeQty", "basePrecision", "quotePrecision", "minTradeAmt",
        "maxTradeQty", "maxTradeAmt", "minPricePrecision", "minOrderAmt", "o", "h", "l", "c", "v",
        "price", "qty", "avgPrice", "leavesQty", "leavesValue", "cumExecQty", "cumExecValue",
        "cumExecFee", "triggerPrice", "takeProfit", "stopLoss", "maxOrderAmt",
        "walletBalance", "locked", "borrowAmount", "accruedInterest"]
    datetimekeys = ["timeSecond"]
    nostringdatetimemillikeys = ["time", "t"]
    datetimemillikeys = ["createdTime", "updatedTime", "nextFundingTime", "deliveryTime", "launchTime"]
    datetimenanokeys = ["timeNano"]
    boolkeys = ["isLeverage"]
    intkeys = ["showStatus", "innovation"]
    for entry in keys(bybitdict)
        if entry in f32keys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : parse(Float32, bybitdict[entry])
        elseif entry in intkeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : parse(Int, bybitdict[entry])
        elseif entry in datetimekeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : Dates.unix2datetime(parse(Int, bybitdict[entry]))
        elseif entry in nostringdatetimemillikeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : Dates.unix2datetime(bybitdict[entry] / 1000)
        elseif entry in datetimemillikeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : Dates.unix2datetime(parse(Int, bybitdict[entry]) / 1000)
        elseif entry in datetimenanokeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : Dates.unix2datetime(parse(Int, bybitdict[entry]) / 1000000000)
        elseif entry in boolkeys
            bybitdict[entry] = bybitdict[entry] == "" ? nothing : parse(Bool, bybitdict[entry])
        elseif entry == "s"
            bybitdict["base"] = uppercase(replace(bybitdict["s"], uppercase(EnvConfig.pairquote) => ""))
            #TODO assumption that only USDT quotecoin is traded is containment - requires a more general approach
        elseif (typeof(bybitdict[entry]) <: AbstractDict) || (typeof(bybitdict[entry]) <: AbstractVector)
            bybitdict[entry] = _dictstring2values!(bybitdict[entry])
        end
    end
    # println("dict conv: $bybitdict")
    return bybitdict
end

function _dictstring2values!(bybitarray::T) where T <:AbstractVector
    for bybitelem in bybitarray
        if (typeof(bybitelem) <: AbstractDict) || (typeof(bybitelem) <: AbstractVector)
            _dictstring2values!(bybitelem)
        end
    end
    # println("array conv: $bybitarray")
    return bybitarray
end

##################### PUBLIC CALL's #####################

"""Returns the DateTime of the Bybit server time as UTC"""
function servertime(bc::BybitCache)
    # ret = HttpPublicRequest(bc, "GET", "/v3/public/time", nothing, "server time")
    ret = HttpPublicRequest(bc, "GET", "/v5/market/time", nothing, "server time")
    return ret["time"]
end

function _simreferencedt(bc::BybitCache, atdt::Union{Nothing, DateTime}=nothing)::DateTime
    dt = isnothing(atdt) ? bc.simtime : atdt
    dt = isnothing(dt) ? floor(Dates.now(Dates.UTC), Minute(1)) : floor(dt, Minute(1))
    return dt - Minute(1)
end

"""
Return simulated last price for one symbol using the closest known 1-minute close
at (or before) the previous minute of the simulation timestamp.
"""
function _sim_lastprice(bc::BybitCache, symbol::AbstractString; atdt::Union{Nothing, DateTime}=nothing)::Float32
    sym = uppercase(String(symbol))
    base = _basefromsymbol(sym)
    refdt = _simreferencedt(bc, atdt)

    if base in _bybitsim_test_basecoins
        testdf = TestOhlcv.testdataframe(base, refdt - Minute(32), refdt, "1m", EnvConfig.pairquote)
        size(testdf, 1) > 0 || error("BybitSim missing test OHLCV for base=$(base) at refdt=$(refdt).")
        ix = Ohlcv.rowix(testdf[!, :opentime], refdt, Minute(1))
        ix > 0 || error("BybitSim test OHLCV row lookup failed for base=$(base) at refdt=$(refdt).")
        return (testdf[ix, :close])
    end

    cached = _sim_cached_ohlcv(bc, base, "1m")
    size(cached.df, 1) > 0 || error("BybitSim missing cached OHLCV for base=$(base), symbol=$(sym).")
    ix = Ohlcv.rowix(cached, refdt)
    ix > 0 || error("BybitSim OHLCV row lookup failed for base=$(base), symbol=$(sym), refdt=$(refdt).")
    return (cached.df[ix, :close])
end

function _sim_get24h(bc::BybitCache, symbol=nothing)
    isempty = DataFrame(symbol=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])
    if isnothing(bc.syminfodf) || (size(bc.syminfodf, 1) == 0)
        return isnothing(symbol) ? isempty : nothing
    end

    quotecoin = uppercase(String(EnvConfig.pairquote))
    rowok(row) = (uppercase(String(row.quotecoin)) == quotecoin) && (String(row.status) == "Trading") && (Int(row.innovation) == 0)

    if !isnothing(symbol) && (symbol != "")
        sym = uppercase(String(symbol))
        ix = findfirst(row -> (uppercase(String(row.symbol)) == sym) && rowok(row), eachrow(bc.syminfodf))
        if isnothing(ix)
            return nothing
        end
        sp = _sim_lastprice(bc, sym)
        return (symbol=sym, quotevolume24h=50_000_000f0, pricechangepercent=0f0, lastprice=sp, askprice=sp * 1.0001f0, bidprice=sp * 0.9999f0)
    end

    df = DataFrame(symbol=String[], quotevolume24h=Float32[], pricechangepercent=Float32[], lastprice=Float32[], askprice=Float32[], bidprice=Float32[])
    pricecache = Dict{String, Float32}()
    missingbases = Set{String}()
    for row in eachrow(bc.syminfodf)
        rowok(row) || continue
        sym = uppercase(String(row.symbol))
        base = _basefromsymbol(sym)
        if base in missingbases
            continue
        end
        sp = if haskey(pricecache, base)
            pricecache[base]
        else
            try
                px = _sim_lastprice(bc, sym)
                pricecache[base] = px
                px
            catch err
                if err isa ErrorException
                    push!(missingbases, base)
                    (verbosity >= 2) && @warn "BybitSim skipping symbol without cached OHLCV at sim reference time" symbol=sym message=err.msg
                    continue
                end
                rethrow(err)
            end
        end
        push!(df, (symbol=sym, quotevolume24h=50_000_000f0, pricechangepercent=0f0, lastprice=sp, askprice=sp * 1.0001f0, bidprice=sp * 0.9999f0))
    end
    return df
end

function get24h(bc::BybitCache, symbol=nothing)
    # BybitSim/offline mode: synthesize stable market snapshot from symbol universe.
    if !isnothing(bc.orderbook)
        return _sim_get24h(bc, symbol)
    end

    if isnothing(symbol) || (symbol == "")
        response = HttpPublicRequest(bc, "GET", "/v5/market/tickers", Dict("category" => "spot"), "ticker/24h")
    else
        response = HttpPublicRequest(bc, "GET", "/v5/market/tickers", Dict("category" => "spot", "symbol" => symbol), "ticker/24h for symbol=$symbol")
    end
    # println(response["result"]["list"])
    df = DataFrame()
    if length(response["result"]["list"]) > 0
        for col in keys(response["result"]["list"][1])
            df[:, col] = [col in keys(entry) ? entry[col] : "" for entry in response["result"]["list"]]
        end
        # 485×12 DataFrame
        # Row   │ ask1Price       lastPrice       prevPrice24h    ask1Size         highPrice24h    turnover24h     symbol        bid1Size        price24hPcnt  volume24h        lowPrice24h      bid1Price
        #       │ Float32         Float32         Float32         Float32          Float32         Float32         String        Float32         Float32       Float32          Float32          Float32
        # ──-───┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        #     1 │ 0.02003         0.02            0.02008      17494.4             0.02069         32411.5         RVNUSDT       52797.6         -0.004        1.59298e6        0.01995          0.02
        # df = df[!, [:ask1Price, :bid1Price, :lastPrice, :turnover24h, :price24hPcnt, :symbol]]
        df = select(df, :ask1Price => "askprice", :bid1Price => "bidprice", :lastPrice => "lastprice", :turnover24h => "quotevolume24h", :price24hPcnt => "pricechangepercent", :symbol)
    end
    validvec = [!isnothing(symbolinfo(bc, df[ix, :symbol])) && (symbolinfo(bc, df[ix, :symbol]).innovation == 0) for ix in eachindex(df[!, :symbol])]
    df = df[validvec, :]
    if !isnothing(symbol) && (size(df, 1)> 0)
        (size(df, 1)> 1) && @error "unexpected multiple entries for $(symbol)"
        return df[1, :]  # should be a DataFrameRow
    else
        return df
    end
end

"Return one simulated ticker using prepared epoch pair metadata."
function get24h(bc::BybitCache, pairref::XchAdapter.TradingPairRef)
    info = _preparedpairinfo(bc, pairref)
    isnothing(info) && return get24h(bc, pairref.pair)
    price = _sim_lastprice(bc, info.symbol)
    return (symbol=info.symbol, quotevolume24h=50_000_000f0, pricechangepercent=0f0, lastprice=price, askprice=price * 1.0001f0, bidprice=price * 0.9999f0)
end

"""
Returns a DataFrame with trading constraints one row per symbol. If symbol is provided the returned DataFrame is limited to that symbol.

- symbol
- status
- basecoin
- quotecoin
- ticksize
- baseprecision
- quoteprecision
- minbaseqty
- minquoteqty
"""
exchangeinfo(bc::BybitCache, symbol=nothing) = isnothing(symbol) ? bc.syminfodf : bc.syminfodf[:symbol .== symbol, :]

function _exchangeinfo(bc::BybitCache, symbol=nothing)
    params = Dict("category" => "spot")
    isnothing(symbol) ? nothing : params["symbol"] = uppercase(symbol)
    response = HttpPublicRequest(bc, "GET", "/v5/market/instruments-info", params, "instruments-info")
    # response = HttpPublicRequest("GET", "/v5/market/instruments-info", params, "instruments-info")
    df = DataFrame()
    if length(response["result"]["list"]) > 0
        for col in keys(response["result"]["list"][1])
            if typeof(response["result"]["list"][1][col]) <: AbstractDict
                for subcol in keys(response["result"]["list"][1][col])
                    df[:, subcol] = [entry[col][subcol] for entry in response["result"]["list"]]
                end
            else
                df[:, col] = [entry[col] for entry in response["result"]["list"]]
            end
        end

        # 1×13 DataFrame
        # Row  │ quoteCoin  status   innovation  marginTrading  symbol   tickSize  baseCoin  maxOrderAmt  quotePrecision  maxOrderQty  minOrderQty  basePrecision  minOrderAmt
        #      │ String     String   Int64       String         String   Float32   String    Float32      Float32         Float32      Float32      Float32        Float32
        # ─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        #    1 │ USDT       Trading           0  both           BTCUSDT      0.01  BTC             2.0e6          1.0e-8      71.7396       4.8e-5         1.0e-6          1.0

        # rename!(df, Dict(:quoteCoin => "quotecoin", :baseCoin => "base", :tickSize => "ticksize", :quotePrecision => "quoteprecision", :basePrecision => "baseprecision", :minOrderQty => "minbaseqty", :minOrderAmt => "minquoteqty"))
        df = select(df, :symbol, :status, :baseCoin => :basecoin, :quoteCoin => :quotecoin, :tickSize => :ticksize, :basePrecision => :baseprecision, :quotePrecision => :quoteprecision, :minOrderQty => :minbaseqty, :minOrderAmt => :minquoteqty, :innovation)
    end
    return df
end

"""
Returns a DataFrameRow with trading constraints. If symbol is not found then `nothing` is returned.

- symbol
- status
- basecoin
- quotecoin
- ticksize
- baseprecision
- quoteprecision
- minbaseqty
- minquoteqty
"""
function symbolinfo(bc::BybitCache, symbol::AbstractString)::Union{Nothing, DataFrameRow}
    symbol = uppercase(symbol)
    symix = findfirst(x -> x == symbol, bc.syminfodf[!, :symbol])
    return isnothing(symix) ? nothing : bc.syminfodf[symix, :]
end

"""
Resolve the normalized internal symbol for a `(basecoin, quotecoin)` pair.
"""
function symboltoken(bc::BybitCache, basecoin::AbstractString, quotecoin::AbstractString=EnvConfig.pairquote)::String
    # The symbol of a pair is fixed for the run, but resolving it scans syminfodf row by row
    # and uppercases each candidate, so it is memoized per cache. The raw argument is cached
    # too, which keeps the hot path free of the (always allocating) uppercase call.
    memo = get!(() -> Dict{String, String}(), _symboltoken_memo, bc)
    rawkey = String(basecoin) * String(quotecoin)
    cached = get(memo, rawkey, nothing)
    isnothing(cached) || return cached

    base = uppercase(basecoin)
    qtoken = uppercase(quotecoin)
    resolved = uppercase(base * qtoken)
    if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
        matchix = findfirst(row -> (uppercase(String(row.basecoin)) == base) && (uppercase(String(row.quotecoin)) == qtoken), eachrow(bc.syminfodf))
        if !isnothing(matchix)
            resolved = uppercase(String(bc.syminfodf[matchix, :symbol]))
        end
    end
    memo[rawkey] = resolved
    memo[base * qtoken] = resolved
    return resolved
end

validsymbol(bc::BybitCache, sym::Union{Nothing, DataFrameRow}) = !isnothing(sym) && (sym.quotecoin == EnvConfig.pairquote) && (sym.innovation == 0) && (sym.status == "Trading") # no Bybit innovation coins
validsymbol(bc::BybitCache, symbol::AbstractString) = validsymbol(bc, symbolinfo(bc, symbol))
function validsymbol(bc::BybitCache, basecoin::AbstractString, quotecoin::AbstractString)
    sym = symbolinfo(bc, symboltoken(bc, basecoin, quotecoin))
    return !isnothing(sym) && (uppercase(String(sym.quotecoin)) == uppercase(quotecoin)) && (Int(sym.innovation) == 0) && (String(sym.status) == "Trading")
end


"Returns a Ohlcv row compatible row data (and skips intentionally turnover)"
_convertkline(kline) = [ix == firstindex(kline) ? Dates.unix2datetime(parse(Int, kline[ix]) / 1000) : parse(Float32, kline[ix]) for ix in eachindex(kline) if ix != lastindex(kline)]

"Returns an Ohlcv compatible klines DataFrame from a Bybit klines structure"
function _convertklines(klines)
    df = DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])  # , quotevolume=Float32[])
    for kix in eachindex(klines)
        push!(df, _convertkline(klines[reverseind(klines, kix)]))  # reverseind() ensures oldest first row sequence
    end
    return df
end

function _intervalperiod(interval::AbstractString)
    m = match(r"^(\d+)([mhdw])$"i, strip(String(interval)))
    isnothing(m) && throw(ArgumentError("unsupported interval=$(interval), expected like 1m,5m,1h,1d,1w"))
    n = parse(Int, m.captures[1])
    unit = lowercase(m.captures[2])
    if unit == "m"
        return Minute(n)
    elseif unit == "h"
        return Hour(n)
    elseif unit == "d"
        return Day(n)
    end
    return Week(n)
end

"""
Return the cached OHLCV data for `base`/`interval`. Prefers `bc.ohlcvcache` (the owning
XchCache's own `xc.bases`, wired in by `Xch.setcurrenttime!`, shared by reference so no
duplicate copy is held) when it already has the base loaded; otherwise falls back to
Bybit's own disk-loaded cache, e.g. for standalone `BybitCache()` use without Xch, or for
bases not (yet) added to the driving `XchCache` (market-wide screening).
"""
function _sim_cached_ohlcv(bc::BybitCache, base::AbstractString, interval::AbstractString)::Ohlcv.OhlcvData
    key = (uppercase(String(base)), interval)
    if !isnothing(bc.ohlcvcache) && (interval == "1m") && haskey(bc.ohlcvcache, key[1])
        return bc.ohlcvcache[key[1]]
    end
    return get!(_sim_ohlcv_cache, key) do
        ohlcv = Ohlcv.defaultohlcv(key[1], interval)
        Ohlcv.read!(ohlcv)
        ohlcv
    end
end

function _sim_klines(bc::BybitCache, symbol::AbstractString; startDateTime=nothing, endDateTime=nothing, interval::AbstractString="1m")
    p = _intervalperiod(interval)
    enddt = isnothing(endDateTime) ? floor(Dates.now(Dates.UTC), p) : floor(endDateTime, p)
    startdt = isnothing(startDateTime) ? floor(enddt - (999 * p), p) : floor(startDateTime, p)
    if enddt < startdt
        return DataFrame(opentime=DateTime[], open=Float32[], high=Float32[], low=Float32[], close=Float32[], basevolume=Float32[])
    end

    base = _basefromsymbol(symbol)
    if base in _bybitsim_test_basecoins
        tdf = TestOhlcv.testdataframe(base, startdt, enddt, interval, EnvConfig.pairquote)
        if size(tdf, 1) == 0
            error("BybitSim missing test OHLCV for base=$(base), interval=$(interval), range=$(startdt) to $(enddt).")
        end
        return select(tdf, :opentime, :open, :high, :low, :close, :basevolume)
    end

    # Prefer persisted OHLCV cache for normal symbols to keep BybitSim prices realistic
    # (e.g., BTC around market magnitude instead of synthetic fallback waves). The cache is
    # loaded once and never mutated in place here, so timerangecut! must not be applied to it.
    cached = _sim_cached_ohlcv(bc, base, interval)
    if size(cached.df, 1) > 0
        startix = Ohlcv.rowix(cached, startdt)
        endix = Ohlcv.rowix(cached, enddt)
        if (startix > 0) && (endix > 0) && (startix <= endix)
            return select(cached.df[startix:endix, :], :opentime, :open, :high, :low, :close, :basevolume)
        end
    end

    error("BybitSim missing cached OHLCV for base=$(base), interval=$(interval), range=$(startdt) to $(enddt). Synthetic fallback is disabled for non-test symbols.")
end

"""
Returns ohlcv/klines data as DataFrame with oldest first rows (which is compatible to Ohlcv but in **contrast to the Bybit default!**)
```
1000×6 DataFrame
  Row │ opentime             open     high     low      close    basevolume
      │ DateTime             Float32  Float32  Float32  Float32  Float32
──────┼─────────────────────────────────────────────────────────────────────
    1 │ 2024-01-14T12:59:00  42758.0  42758.0  42735.1  42744.0  2.71146
"""
function getklines(bc::BybitCache, symbol; startDateTime=nothing, endDateTime=nothing, interval="1m")
    if !isnothing(bc.orderbook)
        return _sim_klines(bc, symbol; startDateTime=startDateTime, endDateTime=endDateTime, interval=interval)
    end

    @assert interval in keys(interval2bybitinterval) "$interval is unknown Bybit interval"
    @assert !isnothing(symbol) && (symbol != "") "missing symbol for Bybit klines"
    params = Dict("category" => "spot", "symbol" => symbol, "interval" => interval2bybitinterval[interval], "limit" => 1000)
    if !isnothing(startDateTime) && !isnothing(endDateTime)
        params["start"] = Printf.@sprintf("%.0d",Dates.datetime2unix(startDateTime) * 1000)
        params["end"] = Printf.@sprintf("%.0d",Dates.datetime2unix(endDateTime) * 1000)
    end
    response = HttpPublicRequest(bc, "GET", "/v5/market/kline", params, "kline")
    response["result"]["list"] = length(response["result"]) == 0 ? _convertklines(Dict()) : _convertklines(response["result"]["list"])
    return response["result"]["list"]
end

##################### SECURED CALL's NEEDS apiKey / apiSecret #####################

"""
Returns accout information, e.g.
acc=Dict{String, Any}("unifiedMarginStatus" => 4, "marginMode" => "REGULAR_MARGIN", "timeWindow" => 10, "smpGroup" => 0, "dcpStatus" => "OFF", "updatedTime" => DateTime("2023-08-13T21:19:17"), "isMasterTrader" => false, "spotHedgingStatus" => "OFF")
"""
function account(bc::BybitCache)
    ret = HttpPrivateRequest(bc, "GET", "/v5/account/info", nothing, "AccountInfo")
    return ret["result"]
end

emptyorders()::DataFrame = EnvConfig.configmode == production ? DataFrame() : DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], isLeverage=Bool[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[], marginleverage=Int32[], reduceonly=Bool[])
emptyorders(::BybitCache)::DataFrame = emptyorders()

"""
Returns a DataFrame of open **spot** orders with columns:

- orderid ::String
- symbol ::String
- side ::String (`Buy` or `Sell`)
- baseqty ::Float32
- ordertype ::String  `Market`, `Limit`
- timeinforce ::String      `GTC` GoodTillCancel, `IOC` ImmediateOrCancel, `FOK` FillOrKill, `PostOnly`
- limitprice ::Float32
- avgprice ::Float32
- executedqty ::Float32  (to be executed qty = baseqty - executedqty)
- status ::String      raw exchange status (normalized by Xch.normalize_order_status)
- created ::DateTime
- updated ::DateTime
- rejectreason ::String
"""
function openorders(bc::BybitCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
    # Check if in simulation mode
    if !isnothing(bc.orderbook)
        _simprocesspendingorders!(bc)
        book = bc.orderbook
        openix = _sim_openorderindex_for(bc)
        if isnothing(orderid) && isnothing(symbol) && isnothing(orderLinkId)
            isempty(openix) && return DataFrame()
            return book[openix, :]
        end

        if !isnothing(orderid)
            idx = get(_sim_orderindex_for(bc), String(orderid), nothing)
            isnothing(idx) && return DataFrame()
            row = book[idx, :]
            if !isnothing(symbol) && uppercase(String(row.symbol)) != uppercase(String(symbol))
                return DataFrame()
            end
            _isopenstatus(String(row.status)) || return DataFrame()
            return row[1:1, :]
        end

        df = book[openix, :]
        if !isnothing(symbol)
            df = df[df[!, :symbol] .== uppercase(String(symbol)), :]
        end
        return df
    end
    
    # Production mode: call Bybit API
    params = Dict("category" => "spot")
    isnothing(symbol) ? nothing : params["symbol"] = symbol
    isnothing(orderid) ? nothing : params["orderId"] = orderid
    isnothing(orderLinkId) ? nothing : params["orderLinkId"] = orderLinkId
    httpresponse = HttpPrivateRequest(bc, "GET", "/v5/order/realtime", params, "openorders")
    df = DataFrame()
    if ("list" in keys(httpresponse["result"])) && (length(httpresponse["result"]["list"]) > 0)
        for col in keys(httpresponse["result"]["list"][1])
            df[:, col] = [entry[col] for entry in httpresponse["result"]["list"]]
        end
        df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :isLeverage => "isLeverage", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason", :reduceOnly => "reduceonly")
    end
#     41×3 DataFrame
#     Row │ variable            min                      eltype
#         │ Symbol              Any                      DataType
#    ─────┼───────────────────────────────────────────────────────
#       1 │ blockTradeId                                 String
#       2 │ price               39900.0                  Float32
#       3 │ timeInForce         PostOnly                 String
#       4 │ leavesQty           0.000116                 Float32
#       5 │ triggerBy                                    String
#       6 │ lastPriceOnCreated                           Nothing
#       7 │ tpTriggerBy                                  String
#       8 │ orderId             1598068305732831744      String
#       9 │ qty                 0.000116                 Float32
#      10 │ leavesValue         4.6284                   Float32
#      11 │ positionIdx         0                        Int64
#      12 │ triggerPrice        0.0                      Float32
#      13 │ cancelType          UNKNOWN                  String
#      14 │ cumExecFee          0.0                      Float32
#      15 │ takeProfit          0.0                      Float32
#      16 │ isLeverage          false                    Bool
#      17 │ cumExecQty          0.0                      Float32
#      18 │ smpOrderId                                   String
#      19 │ slTriggerBy                                  String
#      20 │ orderIv                                      Nothing
#      21 │ avgPrice            0.0                      Float32
#      22 │ smpType             None                     String
#      23 │ stopLoss            0.0                      Float32
#      24 │ marketUnit                                   String
#      25 │ cumExecValue        0.0                      Float32
#      26 │ smpGroup            0                        Int64
#      27 │ reduceOnly          false                    Bool
#      28 │ stopOrderType                                String
#      29 │ symbol              BTCUSDT                  String
#      30 │ orderType           Limit                    String
#      31 │ closeOnTrigger      false                    Bool
#      32 │ orderLinkId         1705240585143            String
#      33 │ orderStatus         New                      String
#      34 │ createdTime         2024-01-14T13:56:27.380  DateTime
#      35 │ side                Buy                      String
#      36 │ slLimitPrice        0.0                      Float32
#      37 │ updatedTime         2024-01-14T13:56:27.382  DateTime
#      38 │ placeType                                    String
#      39 │ tpLimitPrice        0.0                      Float32
#      40 │ rejectReason        EC_NoError               String
#      41 │ triggerDirection    0                        Int64
    return df
end

function allorders(bc::BybitCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
    params = Dict("category" => "spot")
    isnothing(symbol) ? nothing : params["symbol"] = symbol
    isnothing(orderid) ? nothing : params["orderId"] = orderid
    isnothing(orderLinkId) ? nothing : params["orderLinkId"] = orderLinkId
    httpresponse = HttpPrivateRequest(bc, "GET", "/v5/order/history", params, "allorders")
    df = DataFrame()
    if ("list" in keys(httpresponse["result"])) && (length(httpresponse["result"]["list"]) > 0)
        # return httpresponse["result"]["list"]
        for col in keys(httpresponse["result"]["list"][1])
            df[:, col] = [entry[col] for entry in httpresponse["result"]["list"]]
        end
        # df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason")
    end
    return df
end

function alltransactions(bc::BybitCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
    params = Dict("category" => "spot")
    isnothing(symbol) ? nothing : params["symbol"] = symbol
    isnothing(orderid) ? nothing : params["orderId"] = orderid
    isnothing(orderLinkId) ? nothing : params["orderLinkId"] = orderLinkId
    httpresponse = HttpPrivateRequest(bc, "GET", "/v5/execution/list", params, "alltransactions")
    df = DataFrame()
    if length(httpresponse["result"]["list"]) > 0
        # return httpresponse["result"]["list"]
        for col in keys(httpresponse["result"]["list"][1])
            df[:, col] = [entry[col] for entry in httpresponse["result"]["list"]]
        end
        # df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason")
    end
    return df
end

"Return the matching simulation orderbook row by `orderid` without the extra public wrapper indirection."
function _simorderrow(bc::BybitCache, orderid)
    oid = String(orderid)
    if !isnothing(bc.orderbook)
        _simprocesspendingorders!(bc)
        idx = get(_sim_orderindex_for(bc), oid, nothing)
        isnothing(idx) && return nothing
        return bc.orderbook[idx, :]
    end
    return nothing
end

"Returns a named tuple of the identified order or `nothing` if order is not found"
function order(bc::BybitCache, orderid)
    if isnothing(orderid)
        return nothing
    end
    row = _simorderrow(bc, orderid)
    if !isnothing(row)
        return row
    end
    return nothing
end

"""Cancels an open spot order and returns the cancelled orderid"""
function cancelorder(bc::BybitCache, symbol, orderid)
    # Check if in simulation mode
    if !isnothing(bc.orderbook)
        _simprocesspendingorders!(bc)
        ix = get(_sim_orderindex_for(bc), String(orderid), nothing)
        if !isnothing(ix)
            row = bc.orderbook[ix, :]
            _isopenstatus(String(row.status)) || return nothing
            _simreleaseorder!(bc, row.symbol, row.side, Symbol(lowercase(String(row.positionside))), Bool(row.reduceonly), row.baseqty, row.limitprice; lane=String(row.lane))
            dt = isnothing(bc.simtime) ? Dates.now(Dates.UTC) : DateTime(bc.simtime)
            _simfinalizeorder!(bc, ix, "Cancelled", dt)
            delete!(_sim_sequencing_for(bc), String(orderid))
            return String(orderid)
        end
        return nothing
    end
    
    # Production mode: call Bybit API
    params = Dict("category" => "spot", "symbol" => symbol, "orderId" => orderid)
    httpresponse = HttpPrivateRequest(bc, "POST", "/v5/order/cancel", params, "cancelorder")
    # if !("orderId" in keys(httpresponse["result"])) || (httpresponse["result"]["orderId"] != orderid)
    #     @warn "cancel order not confirmed by ByBit via returned orderid: posted=$orderid returned=$(!("orderId" in keys(httpresponse["result"])) ? nothing : httpresponse["result"]["orderId"]) "
    # end
    return !("orderId" in keys(httpresponse["result"])) ? nothing : httpresponse["result"]["orderId"]
end

function _positionlane(side::Symbol)::String
    @assert side in (:long, :short) "position lane side=$(side) must be :long or :short"
    return String(side)
end

"""
Trades-lane identity of one simulated order, matching the TSM lane vocabulary
(`lo`/`lc`/`lcsl`/`so`/`sc`/`scsl`). Symbol plus lane pairs the two legs of a close
bracket, so no separate group id is needed.
"""
const SIM_ORDER_LANES = ("lo", "lc", "lcsl", "so", "sc", "scsl")

"Return the default (non-bracket) lane for one position side and close intent."
function _orderlane(positionside::Symbol, reduceonly::Bool)::String
    @assert positionside in (:long, :short) "_orderlane positionside=$(positionside) must be :long or :short"
    return positionside == :long ? (reduceonly ? "lc" : "lo") : (reduceonly ? "sc" : "so")
end

"Return the stop-loss bracket leg lane for one position side."
_stoporderlane(positionside::Symbol)::String = positionside == :long ? "lcsl" : "scsl"

"Return the other leg's lane of the same close bracket, or `nothing` outside a bracket."
function _bracketsiblinglane(lane::AbstractString)::Union{Nothing, String}
    l = String(lane)
    l == "lc" && return "lcsl"
    l == "lcsl" && return "lc"
    l == "sc" && return "scsl"
    l == "scsl" && return "sc"
    return nothing
end

"Return true when the other leg of this close bracket is still resting for the same symbol."
function _sim_hasopenbracketsibling(bc::BybitCache, symbol::AbstractString, lane::AbstractString)::Bool
    sibling = _bracketsiblinglane(lane)
    isnothing(sibling) && return false
    (isnothing(bc.orderbook) || (nrow(bc.orderbook) == 0)) && return false
    sym = uppercase(String(symbol))
    for ix in _sim_openorderindex_for(bc)
        row = bc.orderbook[ix, :]
        (uppercase(String(row.symbol)) == sym) || continue
        (String(row.lane) == sibling) || continue
        return true
    end
    return false
end

function _iscloseintent(positionside::Symbol, orderside::AbstractString)::Bool
    os = lowercase(String(orderside))
    @assert os in ("buy", "sell") "orderside=$(orderside) must be Buy or Sell"
    return (positionside == :long && os == "sell") || (positionside == :short && os == "buy")
end

"Ensure one holdings row exists in simulation balances and return its row index."
function _ensureholdingrow!(bc::BybitCache, coin::AbstractString, side::AbstractString)
    coinup = uppercase(String(coin))
    sideraw = String(side)
    ix = findfirst(((bc.assets[!, :coin] .== coinup) .& (bc.assets[!, :side] .== sideraw)))
    if isnothing(ix)
        row = (coin=coinup, side=sideraw, free=0f0, locked=0f0, collateral=0f0, proceeds=0f0)
        push!(bc.assets, row; cols=:intersect)
        return lastindex(bc.assets[!, :coin])
    end
    return ix
end

"""Reserve balances for one pending BybitSim order.

Returns `true` once reserved. For an opening order, returns `false` (no state changed)
when free quote is insufficient - insufficient buying power is a normal, recoverable
exchange rejection, not a bookkeeping bug, so callers must reject the order/amend instead
of crashing the trading loop."""
function _simreserveorder!(bc::BybitCache, symbol::AbstractString, side::AbstractString, positionside::Symbol, reduceonly::Bool, basequantity::Real, limitprice::Real; lane::AbstractString)::Bool
    base = _basefromsymbol(symbol)
    quote_coin = uppercase(EnvConfig.pairquote)
    qix = _ensureholdingrow!(bc, quote_coin, "quote")

    is_close = reduceonly || _iscloseintent(positionside, side)
    if is_close
        # Both bracket legs cover the same position, so only the first one reserves it.
        _sim_hasopenbracketsibling(bc, symbol, lane) && return true
        pix = _ensureholdingrow!(bc, base, _positionlane(positionside))
        @assert bc.assets[pix, :free] >= basequantity "BybitSim reserve close requires free position >= quantity; free=$(bc.assets[pix, :free]) quantity=$(basequantity) symbol=$(symbol) positionside=$(positionside)"
        bc.assets[pix, :free] -= basequantity
        bc.assets[pix, :locked] += basequantity
        return true
    end

    cost = basequantity * limitprice
    avail_free = bc.assets[qix, :free]
    # Existing qix.locked is already earmarked (per-position collateral for other open
    # shorts, or reservations for other pending orders); it is not spare capacity. A new
    # open order must be backed by genuinely free quote, or the reservation would silently
    # overcommit the shared pool beyond actual capital.
    avail_free >= cost || return false
    bc.assets[qix, :free] -= cost
    bc.assets[qix, :locked] += cost
    return true
end

"Release the reservation of one pending BybitSim order without filling it."
function _simreleaseorder!(bc::BybitCache, symbol::AbstractString, side::AbstractString, positionside::Symbol, reduceonly::Bool, basequantity::Real, limitprice::Real; lane::AbstractString)
    base = _basefromsymbol(symbol)
    quote_coin = uppercase(EnvConfig.pairquote)
    qix = _ensureholdingrow!(bc, quote_coin, "quote")

    is_close = reduceonly || _iscloseintent(positionside, side)
    if is_close
        # The surviving bracket leg keeps the shared reservation.
        _sim_hasopenbracketsibling(bc, symbol, lane) && return nothing
        pix = _ensureholdingrow!(bc, base, _positionlane(positionside))
        release = min(bc.assets[pix, :locked], basequantity)
        bc.assets[pix, :locked] -= release
        bc.assets[pix, :free] += release
        return nothing
    end

    cost = basequantity * limitprice
    release = min(bc.assets[qix, :locked], cost)
    bc.assets[qix, :locked] -= release
    bc.assets[qix, :free] += release
    return nothing
end

"Apply one pending-order fill while consuming reserved balances."
function _simapplypendingfill!(bc::BybitCache, orderrow, fillprice::Real)
    symbol = String(orderrow.symbol)
    side = String(orderrow.side)
    baseqty = (orderrow.baseqty)
    positionside = Symbol(lowercase(String(orderrow.positionside)))
    reduceonly = Bool(orderrow.reduceonly)
    base = _basefromsymbol(symbol)
    quote_coin = uppercase(EnvConfig.pairquote)
    qix = _ensureholdingrow!(bc, quote_coin, "quote")
    pix = _ensureholdingrow!(bc, base, _positionlane(positionside))

    is_close = reduceonly || _iscloseintent(positionside, side)
    if is_close
        # Close: consume locked position quantity.
        # Long close realizes quote proceeds, short close pays quote to buy back.
        haslane_qty = bc.assets[pix, :free] + bc.assets[pix, :locked]
        release = min(bc.assets[pix, :locked], baseqty)
        bc.assets[pix, :locked] -= release
        quote_flow = release * fillprice
        if positionside == :short
            # Release this position's own share of the pooled quote lock - both its
            # mark-to-market margin and the sale proceeds credited at open - back to free,
            # then pay the buyback out of free. Net wallet change is the realized P&L
            # (proceeds - buyback notional). Other positions' reservations stay untouched.
            fraction = haslane_qty > 0f0 ? (release / haslane_qty) : 1f0
            released_collateral = hasproperty(bc.assets, :collateral) ? bc.assets[pix, :collateral] * fraction : quote_flow
            hasproperty(bc.assets, :collateral) && (bc.assets[pix, :collateral] -= released_collateral)
            released_proceeds = hasproperty(bc.assets, :proceeds) ? bc.assets[pix, :proceeds] * fraction : 0f0
            hasproperty(bc.assets, :proceeds) && (bc.assets[pix, :proceeds] -= released_proceeds)
            unlock = released_collateral + released_proceeds
            locked_use = min(bc.assets[qix, :locked], unlock)
            bc.assets[qix, :locked] -= locked_use
            bc.assets[qix, :free] += locked_use
            bc.assets[qix, :free] -= quote_flow
        else
            bc.assets[qix, :free] += quote_flow
        end
        return nothing
    end

    # Open: add executed quantity to the position lane.
    cost = baseqty * fillprice
    if positionside == :short
        # Short open releases the reservation made at order placement (which may differ from
        # the fill-price notional), then locks the fill notional twice over: once as the
        # mark-to-market margin (`:collateral`, kept current by _simrebalancecollateral!) and
        # once as the sale proceeds (`:proceeds`) that the account receives for the borrowed
        # base. Crediting the proceeds is what makes the position's value conserved: without
        # it the buyback at close consumed cash that was never received, so every short
        # permanently destroyed its own notional.
        release = min(bc.assets[qix, :locked], cost)
        bc.assets[qix, :locked] -= release
        bc.assets[pix, :free] += baseqty
        bc.assets[qix, :locked] += cost
        hasproperty(bc.assets, :collateral) && (bc.assets[pix, :collateral] += cost)
        if hasproperty(bc.assets, :proceeds)
            bc.assets[qix, :locked] += cost
            bc.assets[pix, :proceeds] += cost
        end
        return nothing
    end

    # Long open consumes locked quote budget.
    release = min(bc.assets[qix, :locked], cost)
    bc.assets[qix, :locked] -= release
    residual = cost - release
    if residual > 0f0
        @assert bc.assets[qix, :free] >= residual "BybitSim pending open fill requires free quote >= residual; free=$(bc.assets[qix, :free]) residual=$(residual) symbol=$(symbol)"
        bc.assets[qix, :free] -= residual
    end
    bc.assets[pix, :free] += baseqty
    return nothing
end

"""
Return true when one candle reaches the pending order's trigger price.

Direction follows the order side and whether the order is the stop-loss leg of a close
bracket (lane `lcsl`/`scsl`). A normal limit fills when price comes to it (buy on a fall,
sell on a rise); a protective stop is priced on the adverse side and must wait for price
to move through it, so its direction is inverted.
"""
function _simordertriggered(orderrow, candle)::Bool
    os = lowercase(String(orderrow.side))
    isstop = String(orderrow.lane) in ("lcsl", "scsl")
    limitprice = (orderrow.limitprice)

    if os == "buy"
        return isstop ? ((candle.high) >= limitprice) : ((candle.low) <= limitprice)
    end
    return isstop ? ((candle.low) <= limitprice) : ((candle.high) >= limitprice)
end

"Process pending BybitSim orders using candles from each order's `lastcheck` until current reference time."
function _simprocesspendingorders!(bc::BybitCache; atdt::Union{Nothing, DateTime}=nothing)
    isnothing(bc.orderbook) && return nothing

    decisiondt = isnothing(atdt) ? bc.simtime : atdt
    decisiondt = isnothing(decisiondt) ? floor(Dates.now(Dates.UTC), Minute(1)) : floor(decisiondt, Minute(1))
    bc.lastpendingdecisiondt == decisiondt && return nothing
    size(bc.orderbook, 1) == 0 && (bc.lastpendingdecisiondt = decisiondt; return nothing)
    refdt = decisiondt

    seq = _sim_sequencing_for(bc)
    openix = _sim_openorderindex_for(bc)
    pending_ids = Set(String(bc.orderbook[ix, :orderid]) for ix in openix)

    # Detect fill candidates for every still-open order without applying yet, so a
    # close->open flip pair (registered via directsequence!) can be sequenced below.
    filldt_by_ix = Dict{Int, DateTime}()
    for ix in openix
        row = bc.orderbook[ix, :]
        !_isopenstatus(String(row.status)) && continue

        lastcheck = DateTime(row.lastcheck)
        if decisiondt <= lastcheck
            continue
        end

        startdt = floor(lastcheck, Minute(1))
        candles = _sim_klines(bc, String(row.symbol); startDateTime=startdt, endDateTime=refdt, interval="1m")
        filldt = nothing
        for candle in eachrow(candles)
            candledt = DateTime(candle.opentime) + Minute(1)
            candledt <= lastcheck && continue
            if _simordertriggered(row, candle)
                filldt = candledt
                break
            end
        end

        if isnothing(filldt)
            bc.orderbook[ix, :lastcheck] = decisiondt
            bc.orderbook[ix, :updated] = decisiondt
            continue
        end
        filldt_by_ix[ix] = filldt
    end

    # Both legs of a close bracket can trigger within the same pass. The earlier candle wins;
    # on a tie the protective stop wins, so replay stays deterministic and conservative.
    for ix in sort!(collect(keys(filldt_by_ix)))
        haskey(filldt_by_ix, ix) || continue
        lane = String(bc.orderbook[ix, :lane])
        sibling = _bracketsiblinglane(lane)
        isnothing(sibling) && continue
        sym = uppercase(String(bc.orderbook[ix, :symbol]))
        for jx in sort!(collect(keys(filldt_by_ix)))
            (jx == ix) && continue
            haskey(filldt_by_ix, jx) || continue
            (uppercase(String(bc.orderbook[jx, :symbol])) == sym) || continue
            (String(bc.orderbook[jx, :lane]) == sibling) || continue
            loser = if filldt_by_ix[ix] < filldt_by_ix[jx]
                jx
            elseif filldt_by_ix[jx] < filldt_by_ix[ix]
                ix
            else
                (lane in ("lcsl", "scsl")) ? jx : ix
            end
            delete!(filldt_by_ix, loser)
            break
        end
    end

    triggered_ids = Set(String(bc.orderbook[ix, :orderid]) for ix in keys(filldt_by_ix))

    # A registered successor never fires ahead of its predecessor: if the
    # predecessor is still open and did not resolve in this same pass, the
    # successor stays pending and is retried on the next call untouched.
    immediateix = Int[]
    deferredix = Int[]
    for ix in keys(filldt_by_ix)
        orderid = String(bc.orderbook[ix, :orderid])
        predecessor = get(seq, orderid, nothing)
        if isnothing(predecessor) || !(predecessor in pending_ids)
            push!(immediateix, ix)
        elseif predecessor in triggered_ids
            push!(deferredix, ix)  # predecessor resolves in round A below; apply this after
        end
        # else: predecessor still open and unresolved this pass -> leave pending, retry later
    end

    closerows = NamedTuple[]
    _applysimfill!(ix) = begin
        row = bc.orderbook[ix, :]
        _simapplypendingfill!(bc, row, row.limitprice)
        push!(closerows, (
            orderid=String(row.orderid),
            symbol=String(row.symbol),
            side=String(row.side),
            positionside=String(row.positionside),
            lane=String(row.lane),
            baseqty=(row.baseqty),
            ordertype=String(row.ordertype),
            isLeverage=Bool(row.isLeverage),
            timeinforce=String(row.timeinforce),
            limitprice=(row.limitprice),
            avgprice=(row.limitprice),
            executedqty=(row.baseqty),
            status="Filled",
            created=DateTime(row.created),
            updated=filldt_by_ix[ix],
            rejectreason="NO ERROR",
            lastcheck=filldt_by_ix[ix],
            marginleverage=Int32(row.marginleverage),
            reduceonly=Bool(row.reduceonly),
        ))
        _simfinalizeorder!(bc, ix, "Filled", filldt_by_ix[ix]; avgprice=row.limitprice, executedqty=row.baseqty)
        delete!(seq, String(row.orderid))
        return nothing
    end

    # Round A: independent orders and predecessors resolve first.
    for ix in immediateix
        _applysimfill!(ix)
    end
    # Round B: successors whose predecessor just resolved above.
    for ix in deferredix
        _applysimfill!(ix)
    end

    _simcancelbracketsiblings!(bc, closerows, decisiondt)
    price_by_symbol = Dict{String, Union{Nothing, Float32}}()
    _simrebalancecollateral!(bc; atdt=decisiondt, price_by_symbol=price_by_symbol)
    _simliquidatemargincall!(bc; atdt=decisiondt, price_by_symbol=price_by_symbol)
    bc.lastpendingdecisiondt = decisiondt
    return nothing
end

"""
Cancel the surviving leg of every close bracket whose other leg just filled. The two legs
share one position reservation, which the fill already consumed, so no release is due.
"""
function _simcancelbracketsiblings!(bc::BybitCache, filledrows, decisiondt::DateTime)
    isempty(filledrows) && return nothing
    (isnothing(bc.orderbook) || (nrow(bc.orderbook) == 0)) && return nothing

    cancelix = Int[]
    for frow in filledrows
        sibling = _bracketsiblinglane(String(frow.lane))
        isnothing(sibling) && continue
        sym = uppercase(String(frow.symbol))
        for ix in _sim_openorderindex_for(bc)
            row = bc.orderbook[ix, :]
            (uppercase(String(row.symbol)) == sym) || continue
            (String(row.lane) == sibling) || continue
            (ix in cancelix) || push!(cancelix, ix)
        end
    end
    for ix in cancelix
        row = bc.orderbook[ix, :]
        _simfinalizeorder!(bc, ix, "Cancelled", decisiondt; rejectreason="bracket sibling filled")
        delete!(_sim_sequencing_for(bc), String(row.orderid))
    end
    return nothing
end

"""
Recompute quote collateral for each open short position at the current mark price, once per
tick, moving the shortfall from free into locked (or releasing excess back to free) as price
moves against or in favor of the trade. `:collateral` tracks per-position what is currently
earmarked, so this only touches that position's own share of the pooled quote locked/free
split and leaves other positions' or pending orders' reservations untouched.
"""
function _simcurrentprice!(bc::BybitCache, price_by_symbol::Dict{String, Union{Nothing, Float32}}, symbol::AbstractString, atdt::DateTime)::Union{Nothing, Float32}
    key = uppercase(String(symbol))
    return get!(() -> _simcurrentprice(bc, key, atdt), price_by_symbol, key)
end

function _simrebalancecollateral!(bc::BybitCache; atdt::Union{Nothing, DateTime}=nothing, price_by_symbol::Dict{String, Union{Nothing, Float32}}=Dict{String, Union{Nothing, Float32}}())
    isnothing(bc.assets) && return nothing
    !hasproperty(bc.assets, :collateral) && return nothing
    decisiondt = isnothing(atdt) ? bc.simtime : atdt
    isnothing(decisiondt) && return nothing

    quotecoin = uppercase(EnvConfig.pairquote)
    qix = findfirst(==(quotecoin), uppercase.(String.(bc.assets[!, :coin])))
    isnothing(qix) && return nothing

    for ix in 1:nrow(bc.assets)
        (ix == qix) && continue
        String(bc.assets[ix, :side]) != "short" && continue
        qty = (bc.assets[ix, :free]) + (bc.assets[ix, :locked])
        qty <= 0f0 && continue
        coin = uppercase(String(bc.assets[ix, :coin]))
        symbol = uppercase(string(coin, quotecoin))
        price = _simcurrentprice!(bc, price_by_symbol, symbol, decisiondt)
        isnothing(price) && continue

        required = qty * price
        delta = required - bc.assets[ix, :collateral]
        if delta > 0f0
            moved = min(delta, bc.assets[qix, :free])
            bc.assets[qix, :free] -= moved
            bc.assets[qix, :locked] += moved
            bc.assets[ix, :collateral] += moved
        elseif delta < 0f0
            moved = min(-delta, bc.assets[qix, :locked])
            bc.assets[qix, :locked] -= moved
            bc.assets[qix, :free] += moved
            bc.assets[ix, :collateral] -= moved
        end
    end
    return nothing
end
"Return the latest simulated close price at or before `atdt` for one symbol, or `nothing` if unavailable."
function _simcurrentprice(bc::BybitCache, symbol::AbstractString, atdt::DateTime)::Union{Nothing, Float32}
    candles = _sim_klines(bc, symbol; startDateTime=atdt - Minute(5), endDateTime=atdt, interval="1m")
    size(candles, 1) == 0 && return nothing
    return Float32(candles[end, :close])
end

"""
Force-close all open BybitSim positions at current market price when total account
equity has dropped to zero or below (maintenance-margin breach), mirroring an
exchange margin call. Without this, an unmonitored short/long can accumulate an
unbounded unrealized loss since simulated fills never trigger a liquidation.
"""
function _simliquidatemargincall!(bc::BybitCache; atdt::Union{Nothing, DateTime}=nothing, price_by_symbol::Dict{String, Union{Nothing, Float32}}=Dict{String, Union{Nothing, Float32}}())
    isnothing(bc.assets) && return nothing
    decisiondt = isnothing(atdt) ? bc.simtime : atdt
    isnothing(decisiondt) && return nothing

    quotecoin = uppercase(EnvConfig.pairquote)
    equity = 0.0
    pricebyix = Dict{Int, Float32}()
    for ix in 1:nrow(bc.assets)
        coin = uppercase(String(bc.assets[ix, :coin]))
        free = (bc.assets[ix, :free])
        locked = (bc.assets[ix, :locked])
        if coin == quotecoin
            equity += free + locked
            continue
        end
        qty = free + locked
        qty <= 0f0 && continue
        side = String(bc.assets[ix, :side])
        symbol = uppercase(string(coin, quotecoin))
        price = _simcurrentprice!(bc, price_by_symbol, symbol, decisiondt)
        isnothing(price) && continue
        pricebyix[ix] = price
        signedqty = side == "short" ? -qty : qty
        equity += signedqty * price
    end

    equity > 0.0 && return nothing

    liquidated = String[]
    for (ix, price) in pricebyix
        coin = String(bc.assets[ix, :coin])
        side = String(bc.assets[ix, :side])
        symbol = uppercase(string(uppercase(coin), quotecoin))
        _simforceliquidateposition!(bc, symbol, Symbol(side), price, decisiondt) && push!(liquidated, "$(coin)/$(side)")
    end
    !isempty(liquidated) && (verbosity >= 1) && @warn "BybitSim margin call: liquidated underwater positions" liquidated equity at=decisiondt
    return nothing
end

"""
Force-close one underwater position at the liquidation mark price. Cancels any pending
reduce-only close/stop-loss order for the position (it never filled at its own price) and
queues a structured event for Xch to reconcile into TSM trades columns: `lcl_pavg`/
`scl_pavg`, `lcl_filled`/`scl_filled`, `lcl_status`/`scl_status`, `lcl_msg`/`scl_msg`, plus
resetting the cancelled order's own lane (`lc_id`/`sc_id`, `lc_status`/`sc_status`,
`lc_amount`/`sc_amount`). Returns `true` when a position was actually liquidated.
"""
function _simforceliquidateposition!(bc::BybitCache, symbol::AbstractString, positionside::Symbol, price::Real, decisiondt::DateTime; reason::AbstractString="liquidation")::Bool
    base = _basefromsymbol(symbol)
    pix = _ensureholdingrow!(bc, base, _positionlane(positionside))
    qty = bc.assets[pix, :free] + bc.assets[pix, :locked]
    qty <= 0f0 && return false

    # The pending reduce-only close/stop-loss orders never filled at their own price; cancel
    # them like any other cancellation instead of pretending they filled. A close bracket has
    # two legs, so drain them one at a time - deleting each before the next lookup keeps the
    # shared-reservation check accurate, and the last one releases back to :free.
    hadpendingorder = false
    if !isnothing(bc.orderbook) && (nrow(bc.orderbook) > 0)
        while true
            orderix = nothing
            for ix in _sim_openorderindex_for(bc)
                row = bc.orderbook[ix, :]
                if (uppercase(String(row.symbol)) == uppercase(symbol)) && (Symbol(lowercase(String(row.positionside))) == positionside) && Bool(row.reduceonly)
                    orderix = ix
                    break
                end
            end
            isnothing(orderix) && break
            hadpendingorder = true
            row = bc.orderbook[orderix, :]
            cancelledorderid = String(row.orderid)
            _simreleaseorder!(bc, row.symbol, row.side, positionside, true, row.baseqty, row.limitprice; lane=String(row.lane))
            _simfinalizeorder!(bc, orderix, "Cancelled", decisiondt; rejectreason=String(reason))
            delete!(_sim_sequencing_for(bc), cancelledorderid)
        end
    end

    # Reserve the full (now fully free) position into :locked so the close-fill logic
    # below, which only ever consumes from :locked, covers the entire position.
    freeportion = bc.assets[pix, :free]
    if freeportion > 0f0
        bc.assets[pix, :free] -= freeportion
        bc.assets[pix, :locked] += freeportion
    end

    side = positionside == :short ? "Buy" : "Sell"
    fillprice = Float32(price)
    orderid = "SIM-Liquidate-$(symbol)-$(Int(round(Dates.datetime2unix(decisiondt))))"
    _simapplypendingfill!(bc, (symbol=String(symbol), side=side, baseqty=qty, positionside=String(positionside), reduceonly=true), fillprice)
    _simappendorder!(bc, (
        orderid=orderid, symbol=String(symbol), side=side, positionside=String(positionside), lane=_orderlane(positionside, true), baseqty=qty,
        ordertype="Market", isLeverage=true, timeinforce="IOC", limitprice=fillprice, avgprice=fillprice,
        executedqty=qty, status="Filled", created=decisiondt, updated=decisiondt,
        rejectreason=reason, lastcheck=decisiondt, marginleverage=Int32(2), reduceonly=true,
    ))
    push!(_sim_liquidations_for(bc), (symbol=String(symbol), positionside=positionside, qty=qty, price=fillprice, decisiondt=decisiondt, hadpendingorder=hadpendingorder, reason=String(reason)))
    return true
end

"""
Create one spot order and return an order row compatible named tuple.

If `price` is omitted and `maker=true`, the simulation and live adapters will
choose a limit price as close as possible to the current spread while staying
post-only so the order can qualify for maker fees.
"""
function createorder(bc::BybitCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; configside::Union{Nothing, Symbol}=nothing, execution_spec=nothing, reduceonly::Bool=false, lane::Union{Nothing, AbstractString}=nothing)
    spec = isnothing(execution_spec) ? _executionorderspec(configside, orderside) : execution_spec
    effective_marginleverage = spec.instrument == "spot_margin" ? spec.leverage : 0
    if spec.instrument == "spot_margin"
        2 <= effective_marginleverage <= 10 || error("invalid Bybit spot-margin leverage $(effective_marginleverage) for symbol=$(symbol) configside=$(spec.side)")
    elseif spec.instrument != "spot"
        error("unsupported Bybit execution instrument $(spec.instrument) for symbol=$(symbol) configside=$(spec.side)")
    end
    orderlane = isnothing(lane) ? _orderlane(spec.side, reduceonly) : String(lane)
    @assert orderlane in SIM_ORDER_LANES "createorder lane=$(orderlane) must be one of $(SIM_ORDER_LANES)"
    # Check if in simulation mode
    if !isnothing(bc.orderbook)
        _simprocesspendingorders!(bc)
        syminfo = symbolinfo(bc, symbol)
        if isnothing(syminfo)
            return nothing
        end
        limitprice = isnothing(price) ? (get24h(bc, symbol).lastprice) : (price)
        dt = isnothing(bc.simtime) ? Dates.now(Dates.UTC) : DateTime(bc.simtime)
        orderid = string("SIM-", uppercasefirst(lowercase(orderside)), "-", uppercase(symbol), "-", _nextsimorderseq!(bc))
        if maker
            # `created=dt` is the order timestamp and `lastcheck=dt` is the last
            # decision row already processed for this order. The next simulation row
            # can then evaluate the first newly visible candle whose `opentime`
            # became observable after `dt`.
            row = (orderid=orderid, symbol=symbol, side=uppercasefirst(lowercase(orderside)), positionside=String(spec.side), lane=orderlane, baseqty=(basequantity), ordertype="Limit", isLeverage=(effective_marginleverage > 0), timeinforce="PostOnly", limitprice=limitprice, avgprice=0f0, executedqty=0f0, status="New", created=dt, updated=dt, rejectreason="NO ERROR", lastcheck=dt, marginleverage=Int32(effective_marginleverage), reduceonly=reduceonly)
            # Insufficient buying power is a normal exchange rejection (e.g. account
            # already committed elsewhere), not a bookkeeping bug - reject like any other
            # order-creation failure instead of crashing the trading loop.
            _simreserveorder!(bc, symbol, orderside, spec.side, reduceonly, basequantity, limitprice; lane=orderlane) || return nothing
            _simappendorder!(bc, row)
        else # taker
            reserved = if reduceonly
                _simreserveorder!(bc, symbol, orderside, spec.side, true, basequantity, limitprice; lane=orderlane)
            else
                _simreserveorder!(bc, symbol, orderside, spec.side, false, basequantity, limitprice; lane=orderlane)
            end
            reserved || return nothing
            row = (orderid=orderid, symbol=symbol, side=uppercasefirst(lowercase(orderside)), positionside=String(spec.side), lane=orderlane, baseqty=(basequantity), ordertype="Limit", isLeverage=(effective_marginleverage > 0), timeinforce="GTC", limitprice=limitprice, avgprice=limitprice, executedqty=(basequantity), status="Filled", created=dt, updated=dt, rejectreason="NO ERROR", lastcheck=dt, marginleverage=Int32(effective_marginleverage), reduceonly=reduceonly)
            _simappendorder!(bc, row)
            _simapplypendingfill!(bc, row, limitprice)
        end
        return row
    end
    
    # Production mode: original API implementation
    @assert basequantity > 0.0 "createorder $symbol basequantity of $basequantity cannot be <=0 for order type Limit"
    @assert isnothing(price) || price > 0.0 "createorder $symbol price of $price cannot be <=0 for order type Limit"
    @assert orderside in ["Buy", "Sell"] "createorder $symbol orderside=$orderside no in [Buy, Sell]"

    syminfo = symbolinfo(bc, symbol)
    if isnothing(syminfo)
        @warn "no instrument info for $symbol"
        return nothing
    end
    if syminfo.status != "Trading"
        @warn "$symbol status=$(syminfo.status) != Trading"
        return nothing
    end
    if 2 <= effective_marginleverage <= 10
        HttpPrivateRequest(bc, "POST", "/v5/spot-margin-trade/set-leverage", Dict("leverage" => string(effective_marginleverage)), "set margin leverage")
    end
    attempts = 5
    httpresponse = orderid = nothing
    limitprice = 0f0
    pricedigits = (round(Int, log(10, 1/syminfo.ticksize)))
    params = Dict(
        "category" => "spot",
        "symbol" => symbol,
        "side" => orderside,
        "orderType" => "Limit",
        "qty" => "undefined",
        "price" => "undefined",
        "isLeverage" => (effective_marginleverage == 0 ? 0 : 1),
        "timeInForce" => "undefined")  # "PostOnly" "GTC
    while attempts > 0
        if isnothing(price) # == market order
            now = get24h(bc, symbol)
            # devratio = round(abs(now.lastprice - price) / price * 100)
            # if devratio > 0.01
            #     @warn "limitprice=$price deviates $(devratio)% > 1% of currentprice=$(now.lastprice)"
            #     return nothing
            # end
            # println("pricedigits=$pricedigits, ticksize=$(syminfo.ticksize)")
            if maker
                # The ask price is typically higher than the bid price.
                # The bid price is the price at which a buyer is willing to purchase a security.
                # The ask price is the price at which a seller is willing to longclose a security.
                limitprice = orderside == "Buy" ? now.askprice - syminfo.ticksize : now.bidprice + syminfo.ticksize
                params["timeinforce"] = "PostOnly"
            else # taker
                limitprice = orderside == "Buy" ? now.askprice : now.bidprice
                params["timeinforce"] = "GTC"
            end
        else
            limitprice = round(price, digits=pricedigits)
            attempts = 0
            params["timeinforce"] = maker ? "PostOnly" : "GTC"
        end
        basequantity = (basequantity * limitprice) < syminfo.minquoteqty ? syminfo.minquoteqty / limitprice : basequantity
        basequantity = basequantity < syminfo.minbaseqty ? syminfo.minbaseqty : basequantity
        qtydigits = (round(Int, log(10, 1/syminfo.baseprecision)))
        basequantity = floor(basequantity, digits=qtydigits)
        _enforce_maxquote_policy(spec, symbol, basequantity, limitprice, reduceonly)
        params["qty"] = Format.format(basequantity, precision=qtydigits)
        params["price"] = Format.format(limitprice, precision=pricedigits)
        httpresponse = HttpPrivateRequest(bc, "POST", "/v5/order/create", params, "create order")
        attempts = httpresponse["retCode"] != 0 ? 0 : attempts  # leave loop in case of errors
        if "orderId" in keys(httpresponse["result"])
            orderid = httpresponse["result"]["orderId"]
            if maker
                order = order(bc, httpresponse["result"]["orderId"])
                if !isnothing(order) && (order.status == "Rejected")
                    (verbosity >= 3) && println("$(attempts) PostOnly order for $symbol is rejected")
                    attempts = attempts - 1
                    if attempts == 0
                        (verbosity >= 3) && @warn "exhausted retry attempts for PostOnly order $httpresponse with input price=$(isnothing(price) ? "marketprice" : price)"
                        orderid = nothing
                    end
                else
                    attempts = 0
                end
            else
                attempts = 0
            end
        else
            attempts = 0
        end
    end
    """
    Returns a DataFrame of open **spot** orders with columns:

    - orderid ::String
    - symbol ::String
    - side ::String (`Buy` or `Sell`)
    - baseqty ::Float32
    - ordertype ::String  `Market`, `Limit`
    - timeinforce ::String      `GTC` GoodTillCancel, `IOC` ImmediateOrCancel, `FOK` FillOrKill, `PostOnly`
    - limitprice ::Float32
    - avgprice ::Float32
    - executedqty ::Float32  (to be executed qty = baseqty - executedqty)
    - status ::String      `New`, `PartiallyFilled`, `Untriggered`, `Rejected`, `PartiallyFilledCanceled`, `Filled`, `Cancelled`, `Triggered`, `Deactivated`
    - created ::DateTime
    - updated ::DateTime
    - rejectreason ::String
    """
    if !isnothing(orderid)
        dt = servertime(bc)
        order = (orderid=orderid, symbol=symbol, side=orderside, baseqty=(basequantity), ordertype=params["orderType"], timeinforce=params["timeinforce"], limitprice=limitprice, avgprice=0f0, executedqty=0f0, status="New", created=dt, updated=dt, rejectreason="SIM_NoError")
        return order
    end
    return orderid  # == nothing
end

"""
Create one close order for an existing position side.

- `positionside=:long` maps to a Sell close.
- `positionside=:short` maps to a Buy close.
"""
function closeorder(bc::BybitCache, symbol::String, positionside::Symbol, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; execution_spec=nothing, reduceonly::Bool=true, lane::Union{Nothing, AbstractString}=nothing)
    side = Symbol(lowercase(String(positionside)))
    @assert side in [:long, :short] "closeorder positionside=$(positionside) must be :long or :short"
    orderside = side == :long ? "Sell" : "Buy"
    return createorder(bc, symbol, orderside, basequantity, price, maker; configside=side, execution_spec=execution_spec, reduceonly=reduceonly, lane=lane)
end

_isopenstatus(status::AbstractString)::Bool = lowercase(strip(String(status))) in ("new", "partiallyfilled", "untriggered", "open")

"Upsert one close leg independent from any open leg handling."
function upsertcloseorder!(bc::BybitCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=true, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, XchAdapter.TradingPairRef}=nothing)
    existing = nothing
    if !isnothing(existing_orderid)
        probe = _simorderrow(bc, String(existing_orderid))
        if !isnothing(probe) && hasproperty(probe, :status) && _isopenstatus(String(probe.status))
            existing = probe
        end
    end
    if isnothing(existing)
        return closeorder(bc, symbol, positionside, basequantity, limitprice, maker; reduceonly=reduceonly, lane=lane)
    end

    remaining = max(0.0, (existing.baseqty) - (existing.executedqty))
    currentlimit = hasproperty(existing, :limitprice) ? existing.limitprice : nothing
    qtychanged = remaining != basequantity
    limitchanged = (isnothing(currentlimit) && !isnothing(limitprice)) || (!isnothing(currentlimit) && isnothing(limitprice)) || (!isnothing(currentlimit) && !isnothing(limitprice) && (currentlimit != limitprice))
    if qtychanged || limitchanged
        return amendorder(bc, String(existing.symbol), String(existing.orderid); basequantity=basequantity, limitprice=limitprice, pairref=pairref)
    end
    return String(existing.orderid)
end

"Upsert one open leg independent from any close leg handling."
function upsertopenorder!(bc::BybitCache, symbol::String, positionside::Symbol, basequantity::Real, limitprice::Union{Real, Nothing}; existing_orderid::Union{Nothing, AbstractString}=nothing, maker::Bool=true, reduceonly::Bool=false, lane::Union{Nothing, AbstractString}=nothing, pairref::Union{Nothing, XchAdapter.TradingPairRef}=nothing)
    side = Symbol(lowercase(String(positionside)))
    @assert side in [:long, :short] "upsertopenorder! positionside=$(positionside) must be :long or :short"
    orderside = side == :long ? "Buy" : "Sell"
    existing = nothing
    if !isnothing(existing_orderid)
        probe = _simorderrow(bc, String(existing_orderid))
        if !isnothing(probe) && hasproperty(probe, :status) && _isopenstatus(String(probe.status))
            existing = probe
        end
    end
    if isnothing(existing)
        return createorder(bc, symbol, orderside, basequantity, limitprice, maker; configside=side, reduceonly=reduceonly, lane=lane)
    end

    remaining = max(0.0, (existing.baseqty) - (existing.executedqty))
    currentlimit = hasproperty(existing, :limitprice) ? existing.limitprice : nothing
    qtychanged = remaining != basequantity
    limitchanged = (isnothing(currentlimit) && !isnothing(limitprice)) || (!isnothing(currentlimit) && isnothing(limitprice)) || (!isnothing(currentlimit) && !isnothing(limitprice) && (currentlimit != limitprice))
    if qtychanged || limitchanged
        return amendorder(bc, String(existing.symbol), String(existing.orderid); basequantity=basequantity, limitprice=limitprice, pairref=pairref)
    end
    return String(existing.orderid)
end

"Register direct predecessor/successor sequencing at adapter layer."
function directsequence!(bc::BybitCache, predecessor_orderid::AbstractString, successor_orderid::AbstractString)
    predecessor = order(bc, String(predecessor_orderid))
    successor = order(bc, String(successor_orderid))
    @assert !isnothing(predecessor) "directsequence! predecessor order missing predecessor_orderid=$(predecessor_orderid)"
    @assert !isnothing(successor) "directsequence! successor order missing successor_orderid=$(successor_orderid)"
    @assert String(predecessor.symbol) == String(successor.symbol) "directsequence! symbol mismatch predecessor_symbol=$(String(predecessor.symbol)) successor_symbol=$(String(successor.symbol)) predecessor_orderid=$(predecessor_orderid) successor_orderid=$(successor_orderid)"
    _sim_sequencing_for(bc)[String(successor_orderid)] = String(predecessor_orderid)
    return (predecessor_orderid=String(predecessor_orderid), successor_orderid=String(successor_orderid), symbol=String(predecessor.symbol), acknowledged=true)
end

"Sequence a close order before an opening order using the Bybit adapter's own execution path."
function closebeforeopenflip!(bc::BybitCache, symbol::String, positionside::Symbol, close_basequantity::Real, close_limitprice::Union{Real, Nothing}, close_maker::Bool=true, open_maker::Bool=true; open_limitprice::Union{Real, Nothing}=nothing, open_basequantity::Union{Nothing, Real}=nothing, close_marginleverage::Signed=0, open_marginleverage::Signed=0, close_reduceonly::Bool=true, open_reduceonly::Bool=false)
    side = Symbol(lowercase(String(positionside)))
    @assert side in (:long, :short) "closebeforeopenflip! positionside=$(positionside) must be :long or :short"
    openqty = isnothing(open_basequantity) ? close_basequantity : open_basequantity
    closeoid = closeorder(bc, symbol, side, close_basequantity, close_limitprice, close_maker; reduceonly=close_reduceonly)
    isnothing(closeoid) && return (closeorderid=nothing, openorderid=nothing)
    openoid = side == :long ? createorder(bc, symbol, "Sell", openqty, open_limitprice, open_maker; configside=:short, reduceonly=open_reduceonly) : createorder(bc, symbol, "Buy", openqty, open_limitprice, open_maker; configside=:long, reduceonly=open_reduceonly)
    return (closeorderid=closeoid, openorderid=openoid)
end

"""
Amend one open order.

Only provide `basequantity` or `limitprice` if they have changed values. For a
post-only order, omitting `limitprice` keeps the order adaptive by
re-snapshotting the current spread instead of freezing the previous limit.
"""
function amendorder(bc::BybitCache, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
    orderatentry = order(bc, orderid)
    if isnothing(orderatentry)
        @warn "cannot amend order because orderid $orderid not found"
        return nothing
    end
    return amendorder(bc, String(orderatentry.symbol), orderid; basequantity=basequantity, limitprice=limitprice)
end

function amendorder(bc::BybitCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing, pairref::Union{Nothing, XchAdapter.TradingPairRef}=nothing)
    @assert isnothing(basequantity) ? true : basequantity > 0.0 "amendorder $symbol basequantity of $basequantity cannot be <=0 for order type Limit"
    @assert isnothing(limitprice) ? true : limitprice > 0.0 "amendorder $symbol limitprice of $limitprice cannot be <=0 for order type Limit"
    if !isnothing(bc.orderbook)
        _simprocesspendingorders!(bc)
        ix = get(_sim_orderindex_for(bc), String(orderid), nothing)
        if isnothing(ix)
            @warn "cannot amend order because orderid $orderid not found"
            return nothing
        end

        orderatentry = bc.orderbook[ix, :]
        if !_isopenstatus(String(orderatentry.status))
            @warn "cannot amend terminal order because orderid $orderid status=$(orderatentry.status)"
            return nothing
        end
        syminfo = symbolinfo(bc, symbol)
        if isnothing(syminfo)
            @warn "no instrument info for $symbol"
            return nothing
        end

        maker = orderatentry.timeinforce == "PostOnly"
        now = isnothing(pairref) ? Bybit.get24h(bc, symbol) : Bybit.get24h(bc, pairref)
        changedprice = if maker && isnothing(limitprice)
            orderatentry.side == "Buy" ? now.askprice - syminfo.ticksize : now.bidprice + syminfo.ticksize
        elseif !isnothing(limitprice)
            limitprice
        else
            orderatentry.limitprice
        end
        changedqty = isnothing(basequantity) ? orderatentry.baseqty : basequantity

        oldsymbol = String(orderatentry.symbol)
        oldside = String(orderatentry.side)
        oldqty = (orderatentry.baseqty)
        oldlimit = (orderatentry.limitprice)
        oldsidepos = Symbol(lowercase(String(orderatentry.positionside)))
        oldreduceonly = Bool(orderatentry.reduceonly)
        oldlane = String(orderatentry.lane)

        assetsbackup = copy(bc.assets)
        _simreleaseorder!(bc, oldsymbol, oldside, oldsidepos, oldreduceonly, oldqty, oldlimit; lane=oldlane)
        # Insufficient buying power for the larger/repriced amend is a normal exchange
        # rejection, not a bookkeeping bug - restore the released reservation and leave
        # the order resting unchanged instead of crashing the trading loop.
        if !_simreserveorder!(bc, oldsymbol, oldside, oldsidepos, oldreduceonly, changedqty, changedprice; lane=oldlane)
            bc.assets = assetsbackup
            return nothing
        end

        bc.orderbook[ix, :baseqty] = changedqty
        bc.orderbook[ix, :limitprice] = changedprice
        dt = isnothing(bc.simtime) ? Dates.now(Dates.UTC) : DateTime(bc.simtime)
        bc.orderbook[ix, :updated] = dt
        bc.orderbook[ix, :lastcheck] = dt
        return bc.orderbook[ix, :]
    end

    syminfo = symbolinfo(bc, symbol)
    if isnothing(syminfo)
        @warn "no instrument info for $symbol"
        return nothing
    end
    if syminfo.status != "Trading"
        @warn "$symbol status=$(syminfo.status) != Trading"
        return nothing
    end
    orderatentry = order(bc, orderid)
    if isnothing(orderatentry)
        @warn "cannot amend order because orderid $orderid not found"
        return nothing
    end
    maker = orderatentry.timeinforce == "PostOnly"
    params = Dict(
        "category" => "spot",
        "symbol" => orderatentry.symbol,
        "orderId" => orderid
    )
    attempts = 1
    changedprice = httpresponse = orderid = orderafterattempt = orderpreviousattempt = nothing
    while attempts > 0
        #TODO retry loop in amend fails because the order - once rejected - cannot be changed and is therefore not found anymore
        now = Bybit.get24h(bc, symbol)
        limitchanged = quantitychanged = false
        pricedigits = (round(Int, log(10, 1/syminfo.ticksize)))
        if maker
            # Keep post-only orders adaptive by refreshing against the current spread.
            changedprice = orderatentry.side == "Buy" ? now.askprice - syminfo.ticksize : now.bidprice + syminfo.ticksize
            attempts = 10
        elseif !isnothing(limitprice)
            changedprice = limitprice
        else
            changedprice = orderatentry.limitprice
        end
        changedprice = (round(changedprice, digits=pricedigits))
        if changedprice != orderatentry.limitprice
            limitchanged = true
            params["price"] = Format.format(changedprice, precision=pricedigits)
        end
        if !isnothing(basequantity)
            basequantity = basequantity * changedprice < syminfo.minquoteqty ? syminfo.minquoteqty / changedprice : basequantity
            basequantity = basequantity < syminfo.minbaseqty ? syminfo.minbaseqty : basequantity
            qtydigits = (round(Int, log(10, 1/syminfo.baseprecision)))
            basequantity = (round(basequantity, digits=qtydigits))
            if basequantity != orderatentry.baseqty
                quantitychanged = true
                params["qty"] = Format.format(basequantity, precision=qtydigits)
            end
        end

        if limitchanged || quantitychanged
            httpresponse = HttpPrivateRequest(bc, "POST", "/v5/order/amend", params, "amend order")
            orderafterattempt = Bybit.order(bc, orderid)
            # if httpresponse["retCode"] == 10001
            #     println("previous order values: $orderatentry")
            #     println("changed order values $params")
            #     println("input: limitchanged=$limitchanged, limitprice=$limitprice, changedprice=$changedprice, quantitychanged=$quantitychanged, basequantity=$basequantity")
            # end
            # if httpresponse["retCode"] == 170213
            # end
            if "orderId" in keys(httpresponse["result"])
                orderid = httpresponse["result"]["orderId"]
                if maker
                    if !isnothing(orderafterattempt) && (orderafterattempt.status == "Rejected")
                        (verbosity >= 3) && println("PostOnly order for $symbol is rejected")
                        if attempts == 1
                            @warn "exhausted retry attempts for PostOnly order $orderafterattempt"
                            orderid = nothing
                        end
                    end
                end
            end
            if (httpresponse["retCode"] != 0)
                if (httpresponse["retCode"] == 10001)  # ignore 10001
                    break
                end
                println("entry order: $orderatentry")
                println("changed order values $params")
                println("HTTP response: $httpresponse")
                println("order after attempt: $orderafterattempt")
                println("order previous attempt: $orderpreviousattempt")
                println("attempts=$attempts")
                println("leaving amendorder due to returned error code $(httpresponse["retCode"]), attempts=$attempts")
                break
            end
        end
        attempts -= 1
        orderpreviousattempt = orderafterattempt
    end
    if !isnothing(orderid)
        dt = servertime(bc)
        amendorder = (orderatentry..., baseqty=(isnothing(basequantity) ? orderatentry.baseqty : basequantity),  limitprice=changedprice, updated=dt)
        return amendorder
    end
    return orderid  # == nothing
end

"""
Returns DataFrame with 5 columns of wallet positions of Unified Trade Account
```
   18×2 DataFrame
  ─────┼───────────────────────────────────────────
     1 │ coin                 BTC
     2 │ locked               0
     3 │ free                 0.00011588
     4 │ borrowed             0
     5 │ accruedinterest      0
````
     """
function balances(bc::BybitCache)
    # Check if in simulation mode (BybitSim with simulation state initialized)
    if !isnothing(bc.assets)
        _simprocesspendingorders!(bc)
        return _balancescolumnsdf(bc.assets)
    end
    
    # Production mode: check balance cache (5s TTL to avoid Bybit API rate limits)
    lock(_balance_cache_lock) do
        now = Dates.now(UTC)
        if !isnothing(_balance_cache[]) && !isnothing(_balance_cache_time[])
            if (now - _balance_cache_time[]) < BALANCE_CACHE_TTL
                (verbosity >= 3) && println("balances: returning cached result (age=$(now - _balance_cache_time[]))")
                return copy(_balance_cache[])
            end
        end
    end
    
    # Production mode: call Bybit API
    response = HttpPrivateRequest(bc, "GET", "/v5/account/wallet-balance", Dict("accountType" => "UNIFIED"), "wallet balance")
    df = DataFrame(coin=AbstractString[], locked=Float32[], free=Float32[], borrowed=Float32[], accruedinterest=Float32[])
    if "list" in keys(response["result"])
        for account in response["result"]["list"]
            if account["accountType"] != "UNIFIED"
                @warn "unexpected account type $(account["accountType"])"
            end
            if "coin" in keys(account)
                for coin in account["coin"]
                    walletbalance = isnothing(coin["walletBalance"]) ? 0f0 : coin["walletBalance"]
                    locked = isnothing(coin["locked"]) ? 0f0 : coin["locked"]
                    borrowed = isnothing(coin["borrowAmount"]) ? 0f0 : coin["borrowAmount"]
                    free = abs(walletbalance) - locked - borrowed
                    accruedinterest = isnothing(coin["accruedInterest"]) ? 0f0 : coin["accruedInterest"]
                    push!(df, (coin=coin["coin"], locked=locked, free=free, borrowed=borrowed, accruedinterest=accruedinterest))
                end
            end
        end
    else
        @warn "unexpected missing Bybit balance info: $response"
    end

    # Cache the result for 5 seconds
    lock(_balance_cache_lock) do
        _balance_cache[] = copy(df)
        _balance_cache_time[] = Dates.now(UTC)
    end

    return df
end

"Return one coherent BybitSim balance and position snapshot after pending-order processing."
function accountsnapshot(bc::BybitCache)
    isnothing(bc.assets) && return nothing
    _simprocesspendingorders!(bc)
    return (balances=_balancescolumnsdf(bc.assets), positions=positionsnapshot(bc))
end

"""
Return account capacity in quote currency for Bybit and BybitSim.

The returned tuple aligns with `Xch.accountcapacity` fields and reports
quote-currency-conservative opening capacity while exposing full account equity:
- `available_opening_quote`, `available_long_quote`, `available_short_quote`
  are based on free quote balance.
- `equity_quote` is net marked-to-market account equity in quote terms
    (quote cash + long value - short liability).

For BybitSim, each non-quote held asset is priced individually via `_sim_lastprice`
(which reads cached OHLCV) rather than via a bulk `get24h` join. This avoids the
join key mismatch (`"AAVE"` vs `"AAVEUSDT"`) and the cost of loading hundreds of
OHLCV files for symbols not in the portfolio.
"""
function accountcapacity(bc::BybitCache)
    bdf = balances(bc)
    quotecoin = uppercase(String(EnvConfig.pairquote))
    cols = propertynames(bdf)
    quotefree = 0.0
    wallet_quote = 0.0
    if (:coin in cols) && (:free in cols)
        for row in eachrow(bdf)
            coin = uppercase(String(row.coin))
            free = max(0.0, (row.free))
            locked = (:locked in cols) ? max(0.0, (row.locked)) : 0.0
            if coin == quotecoin
                quotefree += free
                wallet_quote += free + locked
            end
        end
    end

    position_quote = 0.0
    posdf = positionsnapshot(bc)
    if (:coin in propertynames(posdf)) && (:long_qty in propertynames(posdf)) && (:short_qty in propertynames(posdf))
        for prow in eachrow(posdf)
            coin = uppercase(String(prow.coin))
            symbol = string(coin, quotecoin)
            price = try
                (_sim_lastprice(bc, symbol))
            catch
                0.0
            end
            longqty = max(0.0, (prow.long_qty))
            shortqty = max(0.0, (prow.short_qty))
            position_quote += (longqty - shortqty) * price
        end
    end

    # Equity is wallet cash plus the current position mark-to-market. A short's sale proceeds
    # are credited into the wallet at fill (see _simapplypendingfill!), so the borrowed
    # quantity is subtracted here as the offsetting liability.
    equity_quote = wallet_quote + position_quote
    # Keep equity conservative but not below immediately available quote cash.
    equity_quote = max(equity_quote, quotefree)

    equityc = max(0.0, equity_quote)
    openingc = min(max(0.0, quotefree), equityc)
    source = isnothing(bc.assets) ? "Bybit:wallet_balance" : "Bybit:sim_wallet"
    return (
        equity_quote=equityc,
        available_opening_quote=openingc,
        available_long_quote=openingc,
        available_short_quote=openingc,
        initial_margin_quote=0.0,
        maintenance_margin_quote=0.0,
        source=source,
    )
end

"""
Return explicit per-base position quantities from Bybit balances.

`short_qty` is sourced from borrowed balance to represent margin short exposure.
`long_qty` uses free base quantity.
"""
function positionsnapshot(bc::BybitCache)::DataFrame
    if isnothing(bc.assets)
        bdf = balances(bc)
        cols = propertynames(bdf)
        if !((:coin in cols) && (:free in cols))
            return DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
        end
        quotecoin = uppercase(String(EnvConfig.pairquote))
        out = DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
        hasborrowed = :borrowed in cols
        for row in eachrow(bdf)
            coin = uppercase(String(row.coin))
            coin == quotecoin && continue
            longqty = max(0f0, (row.free))
            shortqty = hasborrowed ? max(0f0, (row.borrowed)) : 0f0
            (longqty == 0f0 && shortqty == 0f0) && continue
            push!(out, (coin=coin, long_qty=longqty, short_qty=shortqty))
        end
        return out
    end

    if !((:coin in propertynames(bc.assets)) && (:side in propertynames(bc.assets)) && (:free in propertynames(bc.assets)) && (:locked in propertynames(bc.assets)))
        return DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
    end

    quotecoin = uppercase(String(EnvConfig.pairquote))
    coins = unique(String.(bc.assets[!, :coin]))
    out = DataFrame(coin=String[], long_qty=Float32[], short_qty=Float32[])
    for coinraw in coins
        coin = uppercase(String(coinraw))
        coin == quotecoin && continue
        lmask = (bc.assets[!, :coin] .== coin) .& (bc.assets[!, :side] .== "long")
        smask = (bc.assets[!, :coin] .== coin) .& (bc.assets[!, :side] .== "short")
        lqty = any(lmask) ? sum(bc.assets[lmask, :free]) + sum(bc.assets[lmask, :locked]) : 0f0
        sqty = any(smask) ? sum(bc.assets[smask, :free]) + sum(bc.assets[smask, :locked]) : 0f0
        (lqty == 0f0 && sqty == 0f0) && continue
        push!(out, (coin=coin, long_qty=max(0f0, lqty), short_qty=max(0f0, sqty)))
    end
    return out
end

"Helper function to format balances DataFrame for both production and simulation"
function _balancescolumnsdf(df::DataFrame)::DataFrame
    if (:side in propertynames(df)) && (:free in propertynames(df)) && (:locked in propertynames(df))
        return select(df, :coin, :side, :locked, :free)
    end
    cols = Symbol[]
    for c in (:coin, :locked, :free, :borrowed)
        c in propertynames(df) && push!(cols, c)
    end
    return select(df, cols)
end

"Helper function to extract base coin from symbol (e.g., 'BTCUSDT' -> 'BTC')"
function _basefromsymbol(symbol::AbstractString)
    # Try to extract base from symbol using quote coin
    quote_up = uppercase(EnvConfig.pairquote)
    sym = uppercase(String(symbol))
    if endswith(sym, quote_up)
        return sym[1:(end-length(quote_up))]
    end
    # Fallback for non-standard symbols
    return sym[1:end-4]  # Assume 4-char quote (USDT)
end


# # Websockets functions

# function wsFunction(bc::BybitCache, channel::Channel, ws::String, symbol::String)
#     @assert false "not implemented for Bybit"
#     HTTP.WebSockets.open(string(BYBIT_API_WS, uppercase(symbol), ws); verbose=false) do io
#       while !eof(io);
#         put!(channel, _r2j(readavailable(io)))
#     end
#   end
# end

# function wsTradeAgg(bc::BybitCache, channel::Channel, symbol::String)
#     @assert false "not implemented for Bybit"
#     wsFunction(channel, "@aggTrade", symbol)
# end

# function wsTradeRaw(bc::BybitCache, channel::Channel, symbol::String)
#     @assert false "not implemented for Bybit"
#     wsFunction(channel, "@trade", symbol)
# end

# function wsDepth(bc::BybitCache, channel::Channel, symbol::String; level=5)
#     @assert false "not implemented for Bybit"
#     wsFunction(channel, string("@depth", level), symbol)
# end

# function wsDepthDiff(bc::BybitCache, channel::Channel, symbol::String)
#     @assert false "not implemented for Bybit"
#     wsFunction(channel, "@depth", symbol)
# end

# function wsTicker(bc::BybitCache, channel::Channel, symbol::String)
#     @assert false "not implemented for Bybit"
#     wsFunction(channel, "@ticker", symbol)
# end

# function wsTicker24Hr(bc::BybitCache, channel::Channel)
#     @assert false "not implemented for Bybit"
#     HTTP.WebSockets.open(string(BYBIT_API_WS, "!ticker@arr"); verbose=false) do io
#       while !eof(io);
#         put!(channel, _r2j(readavailable(io)))
#     end
#   end
# end

# function wsKline(bc::BybitCache, channel::Channel, symbol::String; interval="1m")
#   #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
#   @assert false "not implemented for Bybit"
#   wsFunction(channel, string("@kline_", interval), symbol)
# end

# function wsKlineStreams(bc::BybitCache, channel::Channel, symbols::Array, interval="1m")
#   #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
#   @assert false "not implemented for Bybit"
#   allStreams = map(s -> string(uppercase(s), "@kline_", interval), symbols)
#     error = false;
#     while !error
#         try
#             HTTP.WebSockets.open(string(BYBIT_API_WS,join(allStreams, "/")); verbose=false) do io
#             while !eof(io);
#                 put!(channel, String(readavailable(io)))
#             end
#       end
#         catch e
#             println(e)
#             error=true;
#             println("error occured bailing wsklinestreams !")
#         end
#     end
# end

# function wsKlineStreams(bc::BybitCache, callback::Function, symbols::Array; interval="1m")
#     #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
#     @assert false "not implemented for Bybit"
#       allStreams = map(s -> string(uppercase(s), "@kline_", interval), symbols)
#       @async begin
#         HTTP.WebSockets.open(string("wss://stream.binance.com:9443/ws/",join(allStreams, "/")); verbose=false) do io
#             while !eof(io)
#                     data = String(readavailable(io))
#                     callback(data)
#             end
#         end
#     end
# end

# function openUserData(bc::BybitCache, apiKey)
#     @assert false "not implemented for Bybit"
#     headers = Dict("X-BAPI-API-KEY" => apiKey)
#     r = HTTP.request("POST", BYBIT_API_USER_DATA_STREAM, headers)
#     return _r2j(r.body)["listenKey"]
# end

# function keepAlive(bc::BybitCache, apiKey, listenKey)
#     @assert false "not implemented for Bybit"
#     if length(listenKey) == 0
#         return false
#     end

#     headers = Dict("X-BAPI-API-KEY" => apiKey)
#     body = string("listenKey=", listenKey)
#     r = HTTP.request("PUT", BYBIT_API_USER_DATA_STREAM, headers, body)
#     return true
# end

# function closeUserData(bc::BybitCache, apiKey, listenKey)
#     @assert false "not implemented for Bybit"
#     if length(listenKey) == 0
#         return false
#     end
#     headers = Dict("X-BAPI-API-KEY" => apiKey)
#     body = string("listenKey=", listenKey)
#     r = HTTP.request("DELETE", BYBIT_API_USER_DATA_STREAM, headers, body)
#    return true
# end

# function wsUserData(bc::BybitCache, channel::Channel, apiKey, listenKey; reconnect=true)
#     @assert false "not implemented for Bybit"

#     function mykeepAlive()
#         return keepAlive(apiKey, listenKey)
#     end

#     Timer(mykeepAlive, 1800; interval = 1800)

#     error = false;
#     while !error
#         try
#             HTTP.WebSockets.open(string(BYBIT_API_WS, listenKey); verbose=false) do io
#                 while !eof(io);
#                     put!(channel, _r2j(readavailable(io)))
#                 end
#             end
#         catch x
#             println(x)
#             error = true;
#         end
#     end

#     if reconnect
#         wsUserData(channel, apiKey, openUserData(apiKey))
#     end

# end

# helper
filterOnRegex(matcher, withDictArr; withKey="symbol") = filter(x -> match(Regex(matcher), !isnothing(x[withKey])), withDictArr);


end


