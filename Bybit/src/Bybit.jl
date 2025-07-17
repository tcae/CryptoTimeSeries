module Bybit

using HTTP, SHA, JSON3, Dates, Printf, Logging, DataFrames, Format
using EnvConfig

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

const _recvwindow = "5000"
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

struct BybitCache
    syminfodf::Union{Nothing, DataFrame}
    apirest::String
    publickey
    secretkey
end

BYBIT_APIREST = "https://api.bybit.com"
BYBIT_TESTNET_APIREST = "https://api-testnet.bybit.com"

"Initializes Bybit if testnet==true then the Bybit Testnet is used"
function BybitCache(testnet::Bool=EnvConfig.configmode == EnvConfig.test)::BybitCache
    apirest = testnet ? BYBIT_TESTNET_APIREST : BYBIT_APIREST
    bc = BybitCache(nothing, apirest, EnvConfig.authorization.key, EnvConfig.authorization.secret)
    xchinfo = _exchangeinfo(bc)
    xchinfo = sort!(xchinfo[xchinfo.quotecoin .== EnvConfig.cryptoquote, :], :basecoin)
    @assert (!isnothing(xchinfo)) && (size(xchinfo, 1) > 0) "missing exchangeinfo isnothing(xchinfo)=$(isnothing(xchinfo)) size(xchinfo, 1)=$(size(xchinfo, 1))"
    return BybitCache(xchinfo, apirest, EnvConfig.authorization.key, EnvConfig.authorization.secret)
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
    if Sys.isapple()
        Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000))
    else
        Int64(floor(Dates.datetime2unix(Dates.now())))
        # Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC))))
    end
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

HttpPublicRequest(bc::BybitCache, method, endPoint, params::Union{Dict, Nothing}, info) = HttpPrivateRequest(bc, method, endPoint, params, info)

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
            bybitdict["base"] = uppercase(replace(bybitdict["s"], uppercase(EnvConfig.cryptoquote) => ""))
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

- symbol
- quotevolume24h
- pricechangepercent
- lastprice
- askprice
- bidprice

"""
function get24h(bc::BybitCache, symbol=nothing)
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

validsymbol(bc::BybitCache, sym::Union{Nothing, DataFrameRow}) = !isnothing(sym) && (sym.quotecoin == EnvConfig.cryptoquote) && (sym.innovation == 0) && (sym.status == "Trading") # no Bybit innovation coins
validsymbol(bc::BybitCache, symbol::AbstractString) = validsymbol(bc, symbolinfo(bc, symbol))


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

emptyorders()::DataFrame = EnvConfig.configmode == production ? DataFrame() : DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], isLeverage=Bool[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[])

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
function openorders(bc::BybitCache; symbol=nothing, orderid=nothing, orderLinkId=nothing)
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
        df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :isLeverage => "isLeverage", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason")
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
    if !isnothing(orderid)
        oo = openorders(bc, orderid=orderid)
        return size(oo, 1) > 0 ? oo[1, :] : nothing
    else
        return nothing
    end
end

"""Cancels an open spot order and returns the cancelled orderid"""
function cancelorder(bc::BybitCache, symbol, orderid)
    params = Dict("category" => "spot", "symbol" => symbol, "orderId" => orderid)
    httpresponse = HttpPrivateRequest(bc, "POST", "/v5/order/cancel", params, "cancelorder")
    # if !("orderId" in keys(httpresponse["result"])) || (httpresponse["result"]["orderId"] != orderid)
    #     @warn "cancel order not confirmed by ByBit via returned orderid: posted=$orderid returned=$(!("orderId" in keys(httpresponse["result"])) ? nothing : httpresponse["result"]["orderId"]) "
    # end
    return !("orderId" in keys(httpresponse["result"])) ? nothing : httpresponse["result"]["orderId"]
end

"Places an order: spot order by default or margin order if 2 <= marginleverage <= 10"
function createorder(bc::BybitCache, symbol::String, orderside::String, basequantity::Real, price::Union{Real, Nothing}, maker::Bool=true; marginleverage::Signed=0)
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
    if 2 <= marginleverage <= 10
        Bybit.HttpPrivateRequest(bc, "POST", "/v5/spot-margin-trade/set-leverage", Dict("leverage" => string(marginleverage)), "set margin leverage")
    elseif marginleverage != 0
        @error "invalid Bybit margin leverage $marginleverage != [0,2-10]"
        return nothing
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
        "isLeverage" => (marginleverage == 0 ? 0 : 1),
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

"Only provide *basequantity* or *limitprice* if they have changed values."
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
        if !isnothing(limitprice)
            if maker
                # use changedprice instead of changing limitprice because original value of limitprice is also checked in successive loop rounds
                changedprice =  orderatentry.side == "Buy" ? now.askprice - syminfo.ticksize : now.bidprice + syminfo.ticksize
                attempts = 10
            else # take input limitprice
                changedprice = limitprice
            end
            changedprice = Float32(round(changedprice, digits=pricedigits))
            if changedprice != orderatentry.limitprice
                limitchanged = true
                params["price"] = Format.format(changedprice, precision=pricedigits)
            end
        else
            changedprice = orderatentry.limitprice
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
    response = HttpPrivateRequest(bc, "GET", "/v5/account/wallet-balance", Dict("accountType" => "UNIFIED"), "wallet balance")
    # println(response["result"]["list"][1]["coin"][1])
    if length(response["result"]["list"]) > 1
        @warn "unexpected more than 1 account type Bybit balance info: $response"
    end

    # example of BTC margin sell balance result:
    # ("locked" => 0.0f0, "accruedInterest" => 0.0f0, "usdValue" => "-9.08460318", 
    # "spotHedgingQty" => "0", "cumRealisedPnl" => "-0.0044106", "availableToBorrow" => "", 
    # "availableToWithdraw" => 0.0f0, "bonus" => "0", "unrealisedPnl" => "0", "coin" => "BTC", 
    # "borrowAmount" => 9.235487f-5, "walletBalance" => "-0.00009235", "collateralSwitch" => true, 
    # "marginCollateral" => true, "equity" => "-0.00009235", "totalPositionMM" => 0.0f0, 
    # "totalOrderIM" => "0", "totalPositionIM" => 0.0f0)

    # println(response)
    # Dict:
    #    ─────┼───────────────────────────────────────────
    #       1 │ locked               0
    #       2 │ accruedInterest      0
    #       3 │ usdValue             2087.78289118
    #       4 │ spotHedgingQty       0
    #       5 │ cumRealisedPnl       -0.00000011
    #       6 │ availableToBorrow
    #       7 │ availableToWithdraw  0.00011588
    #       8 │ bonus                0
    #       9 │ unrealisedPnl        0
    #      10 │ coin                 BTC
    #      11 │ borrowAmount         0.000000000000000000
    #      12 │ walletBalance        0.00011588
    #      13 │ collateralSwitch     true
    #      14 │ marginCollateral     true
    #      15 │ equity               0.00011588
    #      16 │ totalPositionMM      0
    #      17 │ totalOrderIM         0
    #      18 │ totalPositionIM      0
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
    return df
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


