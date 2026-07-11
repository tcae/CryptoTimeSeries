module Bybit

using HTTP, SHA, JSON3, Dates, Printf, Logging, DataFrames, InlineStrings, Format, Downloads
using EnvConfig
using Ohlcv
using TestOhlcv
using XchAdapter
import XchAdapter: rawcache, symbolinfo, validsymbol, getklines, get24h, balances, openorders, order, cancelorder, createorder, amendorder, servertime, symboltoken, accountcapacity, closeorder

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

function _executionorderspec(configside::Union{Nothing, Symbol}, orderside::AbstractString, marginleverage::Signed)
    side = _executionconfigside(configside, orderside)
    cfg = executionconfig()
    orders = cfg["orders"]
    sidecfg = orders[String(side)]
    instrument = lowercase(String(sidecfg["instrument"]))
    max_quote = haskey(sidecfg, "max_quote") ? sidecfg["max_quote"] : nothing
    leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : Int(marginleverage)
    return (side=side, instrument=instrument, max_quote=max_quote, leverage=leverage)
end

"Return side-specific execution config owned by the Bybit adapter."
function executionorderspec(side::Symbol)
    side in (:long, :short) || error("Bybit executionorderspec side=$(side) must be :long or :short")
    cfg = executionconfig()
    haskey(cfg, "orders") || error("missing Bybit execution config orders section")
    orders = cfg["orders"]
    haskey(orders, String(side)) || error("missing Bybit execution config orders.$(side) section")
    sidecfg = orders[String(side)]
    instrument = haskey(sidecfg, "instrument") ? lowercase(String(sidecfg["instrument"])) : ""
    leverage = haskey(sidecfg, "leverage") ? Int(sidecfg["leverage"]) : 0
    max_quote = haskey(sidecfg, "max_quote") ? Float64(sidecfg["max_quote"]) : nothing
    return (side=side, instrument=instrument, leverage=leverage, max_quote=max_quote)
end

function _enforce_maxquote_policy(spec, symbol::AbstractString, basequantity::Real, price::Union{Real, Nothing}, reduceonly::Bool)
    if isnothing(spec.max_quote) || isnothing(price)
        return nothing
    end
    notional = Float64(basequantity) * Float64(price)
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
When used in BybitSim mode, assets/orders/closedorders track simulated bookkeeping."
mutable struct BybitCache <: XchAdapter.XchAdapterCache
    syminfodf::Union{Nothing, DataFrame}
    apirest::String
    publickey
    secretkey
    simtime::Union{Nothing, DateTime}
    # Simulation state (populated only in BybitSim mode, nil in production)
    assets::Union{Nothing, DataFrame}
    orders::Union{Nothing, DataFrame}
    closedorders::Union{Nothing, DataFrame}
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
    bc = BybitCache(nothing, apirest, pk, sk, nothing, nothing, nothing, nothing)
    xchinfo = _exchangeinfo(bc)
    xchinfo = sort!(xchinfo[xchinfo.quotecoin .== EnvConfig.pairquote, :], :basecoin)
    @assert (!isnothing(xchinfo)) && (size(xchinfo, 1) > 0) "missing exchangeinfo isnothing(xchinfo)=$(isnothing(xchinfo)) size(xchinfo, 1)=$(size(xchinfo, 1))"
    return BybitCache(xchinfo, apirest, pk, sk, nothing, nothing, nothing, nothing)
end

"Initialize simulation state (assets, orders, closedorders) for BybitSim mode"
function _init_simulation!(bc::BybitCache)
    _ensure_sim_symboluniverse!(bc)
    if isnothing(bc.assets)
        bc.assets = DataFrame(coin=String31[], free=Float32[], locked=Float32[], borrowed=Float32[], accruedinterest=Float32[])
        bc.orders = DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], isLeverage=Bool[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[], marginleverage=Int32[], reduceonly=Bool[])
        bc.closedorders = similar(bc.orders)
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
function seedportfolio!(bc::BybitCache, coin::AbstractString, free::Real; locked::Real=0, borrowed::Real=0)
    isnothing(bc.assets) && _init_simulation!(bc)
    coin = uppercase(String(coin))
    ix = findfirst(==(coin), bc.assets[!, :coin])
    if isnothing(ix)
        push!(bc.assets, (coin=coin, free=Float32(free), locked=Float32(locked), borrowed=Float32(borrowed), accruedinterest=0f0))
    else
        bc.assets[ix, :free] = Float32(free)
        bc.assets[ix, :locked] = Float32(locked)
        bc.assets[ix, :borrowed] = Float32(borrowed)
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

"""
Returns a DataFrame with trading information of the last 24h one row per symbol. If symbol is provided the returned DataFrame is limited to that symbol.

In BybitSim mode this snapshot is used to support maker orders created with
`price=nothing`: the backend chooses the closest post-only limit price it can
derive from the simulated ticker snapshot.

- symbol
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

"""
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
        return Float32(testdf[ix, :close])
    end

    cached = Ohlcv.defaultohlcv(base, "1m")
    Ohlcv.read!(cached)
    size(cached.df, 1) > 0 || error("BybitSim missing cached OHLCV for base=$(base), symbol=$(sym).")
    ix = Ohlcv.rowix(cached, refdt)
    ix > 0 || error("BybitSim OHLCV row lookup failed for base=$(base), symbol=$(sym), refdt=$(refdt).")
    return Float32(cached.df[ix, :close])
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
    if !isnothing(bc.orders)
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
    base = uppercase(basecoin)
    qtoken = uppercase(quotecoin)
    if !isnothing(bc.syminfodf) && (size(bc.syminfodf, 1) > 0)
        matchix = findfirst(row -> (uppercase(String(row.basecoin)) == base) && (uppercase(String(row.quotecoin)) == qtoken), eachrow(bc.syminfodf))
        if !isnothing(matchix)
            return uppercase(String(bc.syminfodf[matchix, :symbol]))
        end
    end
    return uppercase(base * qtoken)
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

function _sim_klines(symbol::AbstractString; startDateTime=nothing, endDateTime=nothing, interval::AbstractString="1m")
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
    # (e.g., BTC around market magnitude instead of synthetic fallback waves).
    cached = Ohlcv.defaultohlcv(base, interval)
    Ohlcv.read!(cached)
    if size(cached.df, 1) > 0
        Ohlcv.timerangecut!(cached, startdt, enddt)
        if size(cached.df, 1) > 0
            return select(cached.df, :opentime, :open, :high, :low, :close, :basevolume)
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
    if !isnothing(bc.orders)
        return _sim_klines(symbol; startDateTime=startDateTime, endDateTime=endDateTime, interval=interval)
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
    if !isnothing(bc.orders)
        df = copy(bc.orders)
        if !isnothing(symbol)
            df = df[df[!, :symbol] .== uppercase(String(symbol)), :]
        end
        if !isnothing(orderid)
            df = df[df[!, :orderid] .== String(orderid), :]
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

"Returns a named tuple of the identified order or `nothing` if order is not found"
function order(bc::BybitCache, orderid)
    if isnothing(orderid)
        return nothing
    end

    oo = openorders(bc, orderid=orderid)
    if size(oo, 1) > 0
        return oo[1, :]
    end

    # BybitSim keeps immediately-filled/cancelled orders in `closedorders`.
    # Expose them via order(id) so callers can still reconcile status after close.
    if !isnothing(bc.closedorders)
        co = bc.closedorders[bc.closedorders[!, :orderid] .== String(orderid), :]
        if size(co, 1) > 0
            return co[end, :]
        end
    end
    return nothing
end

"""Cancels an open spot order and returns the cancelled orderid"""
function cancelorder(bc::BybitCache, symbol, orderid)
    # Check if in simulation mode
    if !isnothing(bc.orders)
        ix = findfirst(==(String(orderid)), bc.orders[!, :orderid])
        if !isnothing(ix)
            row = bc.orders[ix, :]
            push!(bc.closedorders, row)
            deleteat!(bc.orders, ix)
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

"Helper function to apply order fill to simulation balances"
function _applyfill!(bc::BybitCache, symbol::AbstractString, side::AbstractString, basequantity::Real, price::Real, marginleverage::Signed=0)
    base = _basefromsymbol(symbol)
    quote_coin = uppercase(EnvConfig.pairquote)
    bix = findfirst(==(base), bc.assets[!, :coin])
    qix = findfirst(==(quote_coin), bc.assets[!, :coin])
    if isnothing(bix)
        push!(bc.assets, (coin=base, free=0f0, locked=0f0, borrowed=0f0, accruedinterest=0f0))
        bix = lastindex(bc.assets[!, :coin])
    end
    if isnothing(qix)
        push!(bc.assets, (coin=quote_coin, free=0f0, locked=0f0, borrowed=0f0, accruedinterest=0f0))
        qix = lastindex(bc.assets[!, :coin])
    end
    
    is_short_margin = marginleverage > 0  # All current margin trades are shorts; distinguishes from potential future is_long_margin
    is_long_margin = false  # Reserved for future long margin trades
    
    if lowercase(String(side)) == "buy"
        bc.assets[qix, :free] -= Float32(basequantity * price)
        if is_short_margin
            # Short margin close: reduce borrowed amount (covering short), don't add to free
            bc.assets[bix, :borrowed] -= Float32(basequantity)
        elseif is_long_margin
            # Long margin open/maintain: add to free base (or track via borrowed in margin account)
            bc.assets[bix, :free] += Float32(basequantity)
        else
            # Spot buy: add to free base
            bc.assets[bix, :free] += Float32(basequantity)
        end
    else
        bc.assets[qix, :free] += Float32(basequantity * price)
        if is_short_margin
            # Short margin open: increase borrowed base (short position), don't decrease free
            bc.assets[bix, :borrowed] += Float32(basequantity)
        elseif is_long_margin
            # Long margin close/reduce: decrease free base (or track via borrowed)
            bc.assets[bix, :free] -= Float32(basequantity)
        else
            # Spot sell: decrease free base
            bc.assets[bix, :free] -= Float32(basequantity)
        end
    end
end

"""
Create one spot order and return an order row compatible named tuple.

If `price` is omitted and `maker=true`, the simulation and live adapters will
choose a limit price as close as possible to the current spread while staying
post-only so the order can qualify for maker fees.
"""
function createorder(bc::BybitCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; configside::Union{Nothing, Symbol}=nothing, execution_spec=nothing, marginleverage::Signed=0, reduceonly::Bool=false)
    spec = isnothing(execution_spec) ? _executionorderspec(configside, orderside, marginleverage) : execution_spec
    effective_marginleverage = spec.instrument == "spot_margin" ? spec.leverage : 0
    if spec.instrument == "spot_margin"
        2 <= effective_marginleverage <= 10 || error("invalid Bybit spot-margin leverage $(effective_marginleverage) for symbol=$(symbol) configside=$(spec.side)")
    elseif spec.instrument != "spot"
        error("unsupported Bybit execution instrument $(spec.instrument) for symbol=$(symbol) configside=$(spec.side)")
    end
    # Check if in simulation mode
    if !isnothing(bc.orders)
        syminfo = symbolinfo(bc, symbol)
        if isnothing(syminfo)
            return nothing
        end
        limitprice = isnothing(price) ? Float32(get24h(bc, symbol).lastprice) : Float32(price)
        dt = Dates.now(Dates.UTC)
        orderid = string("SIM-", uppercasefirst(lowercase(orderside)), "-", uppercase(symbol), "-", _nextsimorderseq!(bc))
        row = (orderid=orderid, symbol=symbol, side=uppercasefirst(lowercase(orderside)), baseqty=Float32(basequantity), ordertype="Limit", isLeverage=(effective_marginleverage > 0), timeinforce=maker ? "PostOnly" : "GTC", limitprice=limitprice, avgprice=limitprice, executedqty=Float32(basequantity), status="Filled", created=dt, updated=dt, rejectreason="NO ERROR", lastcheck=dt, marginleverage=Int32(effective_marginleverage), reduceonly=reduceonly)
        push!(bc.closedorders, row)
        _applyfill!(bc, symbol, orderside, basequantity, limitprice, effective_marginleverage)
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
        Bybit.HttpPrivateRequest(bc, "POST", "/v5/spot-margin-trade/set-leverage", Dict("leverage" => string(effective_marginleverage)), "set margin leverage")
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
            now = Bybit.get24h(bc, symbol)
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
                order = Bybit.order(bc, httpresponse["result"]["orderId"])
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
        order = (orderid=orderid, symbol=symbol, side=orderside, baseqty=Float32(basequantity), ordertype=params["orderType"], timeinforce=params["timeinforce"], limitprice=limitprice, avgprice=0f0, executedqty=0f0, status="New", created=dt, updated=dt, rejectreason="SIM_NoError")
        return order
    end
    return orderid  # == nothing
end

"""
Create one close order for an existing position side.

- `positionside=:long` maps to a Sell close.
- `positionside=:short` maps to a Buy close.
"""
function closeorder(bc::BybitCache, symbol::String, positionside::Symbol, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; execution_spec=nothing, marginleverage::Signed=0, reduceonly::Bool=true)
    side = Symbol(lowercase(String(positionside)))
    @assert side in [:long, :short] "closeorder positionside=$(positionside) must be :long or :short"
    orderside = side == :long ? "Sell" : "Buy"
    return createorder(bc, symbol, orderside, basequantity, price, maker; configside=side, execution_spec=execution_spec, marginleverage=marginleverage, reduceonly=reduceonly)
end

"Sequence a close order before an opening order using the Bybit adapter's own execution path."
function closebeforeopenflip!(bc::BybitCache, symbol::String, positionside::Symbol, close_basequantity::Real, close_limitprice::Union{Real, Nothing}, close_maker::Bool=true, open_maker::Bool=true; open_limitprice::Union{Real, Nothing}=nothing, open_basequantity::Union{Nothing, Real}=nothing, close_marginleverage::Signed=0, open_marginleverage::Signed=0, close_reduceonly::Bool=true, open_reduceonly::Bool=false)
    side = Symbol(lowercase(String(positionside)))
    @assert side in (:long, :short) "closebeforeopenflip! positionside=$(positionside) must be :long or :short"
    openqty = isnothing(open_basequantity) ? close_basequantity : open_basequantity
    closeoid = closeorder(bc, symbol, side, close_basequantity, close_limitprice, close_maker; marginleverage=close_marginleverage, reduceonly=close_reduceonly)
    isnothing(closeoid) && return (closeorderid=nothing, openorderid=nothing)
    openoid = side == :long ? createorder(bc, symbol, "Sell", openqty, open_limitprice, open_maker; configside=:short, marginleverage=open_marginleverage, reduceonly=open_reduceonly) : createorder(bc, symbol, "Buy", openqty, open_limitprice, open_maker; configside=:long, marginleverage=open_marginleverage, reduceonly=open_reduceonly)
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

function amendorder(bc::BybitCache, symbol::String, orderid::String; basequantity::Union{Nothing, Real}=nothing, limitprice::Union{Nothing, Real}=nothing)
    @assert isnothing(basequantity) ? true : basequantity > 0.0 "amendorder $symbol basequantity of $basequantity cannot be <=0 for order type Limit"
    @assert isnothing(limitprice) ? true : limitprice > 0.0 "amendorder $symbol limitprice of $limitprice cannot be <=0 for order type Limit"
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
        changedprice = Float32(round(changedprice, digits=pricedigits))
        if changedprice != orderatentry.limitprice
            limitchanged = true
            params["price"] = Format.format(changedprice, precision=pricedigits)
        end
        if !isnothing(basequantity)
            basequantity = basequantity * changedprice < syminfo.minquoteqty ? syminfo.minquoteqty / changedprice : basequantity
            basequantity = basequantity < syminfo.minbaseqty ? syminfo.minbaseqty : basequantity
            qtydigits = (round(Int, log(10, 1/syminfo.baseprecision)))
            basequantity = Float32(round(basequantity, digits=qtydigits))
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
        amendorder = (orderatentry..., baseqty=Float32(isnothing(basequantity) ? orderatentry.baseqty : basequantity),  limitprice=changedprice, updated=dt)
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
        return _emptybalances(bc.assets)
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

"""
Return account capacity in quote currency for Bybit and BybitSim.

The returned tuple aligns with `Xch.accountcapacity` fields and reports
quote-currency-conservative opening capacity while exposing full account equity:
- `available_opening_quote`, `available_long_quote`, `available_short_quote`
  are based on free quote balance.
- `equity_quote` is full marked-to-market account equity in quote terms.

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
    equity_quote = 0.0
    if (:coin in cols) && (:free in cols)
        for row in eachrow(bdf)
            coin = uppercase(String(row.coin))
            free  = max(0.0, Float64(row.free))
            locked   = (:locked   in cols) ? max(0.0, Float64(row.locked))   : 0.0
            borrowed = (:borrowed in cols) ? max(0.0, Float64(row.borrowed)) : 0.0
            net = free + locked - borrowed
            if coin == quotecoin
                quotefree    += free
                equity_quote += net  # quote coin priced at 1.0
            elseif !isnothing(bc.assets) && net != 0.0
                # BybitSim: price non-quote asset at current sim time.
                # Any pricing failure is treated as price=0 (conservative; does not
                # deduct the liability but also does not inflate equity).
                symbol = string(coin, quotecoin)
                price = try
                    Float64(_sim_lastprice(bc, symbol))
                catch
                    0.0
                end
                equity_quote += net * price
            end
            # Live mode: only quote-wallet balance contributes to equity (conservative).
        end
    end
    source = isnothing(bc.assets) ? "Bybit:wallet_balance" : "Bybit:sim_wallet"
    return (
        equity_quote=max(0.0, equity_quote),
        available_opening_quote=max(0.0, quotefree),
        available_long_quote=max(0.0, quotefree),
        available_short_quote=max(0.0, quotefree),
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
        longqty = max(0f0, Float32(row.free))
        shortqty = hasborrowed ? max(0f0, Float32(row.borrowed)) : 0f0
        (longqty == 0f0 && shortqty == 0f0) && continue
        push!(out, (coin=coin, long_qty=longqty, short_qty=shortqty))
    end
    return out
end

"Helper function to format balances DataFrame for both production and simulation"
function _emptybalances(df::DataFrame)
    return select(df, :coin, :locked, :free, :borrowed, :accruedinterest)
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


