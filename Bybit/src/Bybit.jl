module Bybit

using HTTP, SHA, JSON3, Dates, Printf, Logging, DataFrames, Formatting
using EnvConfig

# base URL of the ByBit API
# BYBIT_API_REST = "https://api.bybit.com"
# BYBIT_API_WS = "to be defined for Bybit"  # "wss://stream.binance.com:9443/ws/"
# BYBIT_API_USER_DATA_STREAM ="to be defined for Bybit"

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

function BybitCache(testnet::Bool=EnvConfig.configmode == EnvConfig.test)::BybitCache
    apirest = testnet ? "https://api-testnet.bybit.com" : "https://api.bybit.com"
    bc = BybitCache(nothing, apirest, EnvConfig.authorization.key, EnvConfig.authorization.secret)
    xchinfo = exchangeinfo(bc)
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

# signing with apiKey and apiSecret
function timestamp()
    Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000))
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
            @info "h1=$(header[1]) h2=$(header[2]) fullheader=$(header) waiting for 1s"
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
            if methodpost
                # headers["Content-Type"] = "application/json; charset=utf-8"
                url = bc.apirest * endPoint
                response = HTTP.request(method, url, headers, payload)
            else
                url = bc.apirest * endPoint * "?" * payload
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
        @error "HttpPrivateRequest $info #$requestcount $method response=$body \nurl=$url \nheaders=$headers \npayload=$payload"
        # @error "HttpPrivateRequest $info #$requestcount $method return code == $(body["retCode"]) \nurl=$url \nheaders=$headers \npayload=$payload \nresponse=$body"
        # println("public_key=$public_key, secret_key=$secret_key")
        # println(err)
        # rethrow()
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
        "availableToWithdraw", "locked"]
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
    ret = HttpPublicRequest(bc, "GET", "/v3/public/time", nothing, "server time")
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
    response["result"]["list"] = df
    return response["result"]["list"]
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
function exchangeinfo(bc::BybitCache, symbol=nothing)
    params = Dict("category" => "spot")
    isnothing(symbol) ? nothing : params["symbol"] = uppercase(symbol)
    response = HttpPublicRequest(bc, "GET", "/v5/market/instruments-info", params, "instruments-info")
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
        df = select(df, :symbol, :status, :baseCoin => :basecoin, :quoteCoin => :quotecoin, :tickSize => :ticksize, :basePrecision => :baseprecision, :quotePrecision => :quoteprecision, :minOrderQty => :minbaseqty, :minOrderAmt => :minquoteqty)
    end
    return df
end

"""
Returns a NamedTuple with trading constraints. If symbol is not found then `nothing` is returned.

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
function symbolinfo(bc::BybitCache, symbol)
    symbol = uppercase(symbol)
    df = bc.syminfodf[bc.syminfodf.symbol .== symbol, :]
    if size(df, 1) == 0
        # @warn "symbol $symbol not found"
        return nothing
    end
    if size(df, 1) > 1
        @warn "more than one entry found for $symbol => using first\n$df"
    end
    return NamedTuple(df[1,:])
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

emptyorders()::DataFrame = EnvConfig.configmode == production ? DataFrame() : DataFrame(orderid=String[], symbol=String[], side=String[], baseqty=Float32[], ordertype=String[], timeinforce=String[], limitprice=Float32[], avgprice=Float32[], executedqty=Float32[], status=String[], created=DateTime[], updated=DateTime[], rejectreason=String[], lastcheck=DateTime[])

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
    oo = HttpPrivateRequest(bc, "GET", "/v5/order/realtime", params, "openorders")
    df = DataFrame()
    if length(oo["result"]["list"]) > 0
        for col in keys(oo["result"]["list"][1])
            df[:, col] = [entry[col] for entry in oo["result"]["list"]]
        end
        df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason")
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
    oo = HttpPrivateRequest(bc, "GET", "/v5/order/history", params, "allorders")
    df = DataFrame()
    if length(oo["result"]["list"]) > 0
        # return oo["result"]["list"]
        for col in keys(oo["result"]["list"][1])
            df[:, col] = [entry[col] for entry in oo["result"]["list"]]
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
    oo = HttpPrivateRequest(bc, "GET", "/v5/execution/list", params, "alltransactions")
    df = DataFrame()
    if length(oo["result"]["list"]) > 0
        # return oo["result"]["list"]
        for col in keys(oo["result"]["list"][1])
            df[:, col] = [entry[col] for entry in oo["result"]["list"]]
        end
        # df = select(df, :orderId => "orderid", :symbol, :side, [:leavesQty, :cumExecQty] => ((leavesQty, cumExecQty) -> leavesQty + cumExecQty) => "baseqty", :orderType => "ordertype", :timeInForce => "timeinforce", :price => "limitprice", :avgPrice => "avgprice", :cumExecQty => "executedqty", :orderStatus => "status", :createdTime => "created", :updatedTime => "updated", :rejectReason => "rejectreason")
    end
    return df
end

"Returns a named tuple of the identified order or `nothing` if order is not found"
function order(bc::BybitCache, orderid)
    if !isnothing(orderid)
        oo = openorders(bc, orderid=orderid)
        return size(oo, 1) > 0 ? NamedTuple(oo[1, :]) : nothing
    else
        return nothing
    end
end

"""Cancels an open spot order and returns the cancelled orderid"""
function cancelorder(bc::BybitCache, symbol, orderid)
    params = Dict("category" => "spot", "symbol" => symbol, "orderId" => orderid)
    oo = HttpPrivateRequest(bc, "POST", "/v5/order/cancel", params, "cancelorder")
    if !(oo["result"]["orderId"] == orderid)
        @warn "cancel order not confirmed by ByBit via returned orderid: posted=$orderid returned=$(oo["orderId"]) "
    end
    return oo["result"]["orderId"]
end

function createorder(bc::BybitCache, symbol::String, orderside::String, quantity::Real, price::Real)
    @assert quantity > 0.0 "createorder $symbol quantity of $quantity cannot be <=0 for order type Limit"
    @assert price > 0.0 "createorder $symbol price of $price cannot be <=0 for order type Limit"
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
    pricedigits = (round(Int, log(10, 1/syminfo.ticksize)))
    price = round(price, digits=pricedigits)
    quantity = quantity * price < syminfo.minquoteqty ? syminfo.minquoteqty / price : quantity
    quantity = quantity < syminfo.minbaseqty ? syminfo.minbaseqty : quantity
    qtydigits = (round(Int, log(10, 1/syminfo.baseprecision)))
    quantity = floor(quantity, digits=qtydigits)

    params = Dict(
        "category" => "spot",
        "symbol" => symbol,
        "side" => orderside,
        "orderType" => "Limit",
        "qty" => Formatting.format(quantity, precision=qtydigits),
        "price" => Formatting.format(price, precision=pricedigits),
        "timeInForce" => "GTC")  # "PostOnly" does not help as long as not VIP status because maker fee = taker fee 0.1%
    oo = HttpPrivateRequest(bc, "POST", "/v5/order/create", params, "create order")
    if "orderId" in keys(oo["result"])
        return oo["result"]["orderId"]
    else
        return nothing
    end
end

"Only provide *quantity* or *limitprice* if they have changed values."
function amendorder(bc::BybitCache, symbol::String, orderid::String; quantity=nothing::Union{Nothing, Real}, limitprice=nothing::Union{Nothing, Real})
    @assert isnothing(quantity) ? true : quantity > 0.0 "amendorder $symbol quantity of $quantity cannot be <=0 for order type Limit"
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
    ont = order(bc, orderid)
    if isnothing(ont)
        @warn "cannot amend order because orderid $orderid not found"
        return nothing
    end
    params = Dict(
        "category" => "spot",
        "symbol" => symbol,
        "orderId" => orderid
    )
    if isnothing(limitprice)
        changeprice =  ont.limitprice
    else
        changeprice =  limitprice
        pricedigits = (round(Int, log(10, 1/syminfo.ticksize)))
        limitprice = round(limitprice, digits=pricedigits)
        params["price"] = Formatting.format(limitprice, precision=pricedigits)
    end
    if !isnothing(quantity)
        quantity = quantity * changeprice < syminfo.minquoteqty ? syminfo.minquoteqty / changeprice : quantity
        quantity = quantity < syminfo.minbaseqty ? syminfo.minbaseqty : quantity
        qtydigits = (round(Int, log(10, 1/syminfo.baseprecision)))
        quantity = round(quantity, digits=qtydigits)
        params["qty"] = Formatting.format(quantity, precision=qtydigits)
    end

    oo = HttpPrivateRequest(bc, "POST", "/v5/order/amend", params, "amend order")
    if "orderId" in keys(oo["result"])
        return oo["result"]["orderId"]
    else
        return nothing
    end
end

"""
Returns DataFrame with 3 columns of wallet positions of Unified Trade Account
```
   18×2 DataFrame
  ─────┼───────────────────────────────────────────
     1 │ coin                 BTC
     2 │ locked               0
     3 │ free                 0.00011588
     """
function balances(bc::BybitCache)
    response = HttpPrivateRequest(bc, "GET", "/v5/account/wallet-balance", Dict("accountType" => "UNIFIED"), "wallet balance")
    if length(response["result"]["list"]) > 1
        @warn "unexpected more than 1 account type Bybit balance info: $response"
    end
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
     df = DataFrame(coin=String[], locked=Float32[], free=Float32[])
    if "list" in keys(response["result"])
        for account in response["result"]["list"]
            if account["accountType"] != "UNIFIED"
                @warn "unexpected account type $(account["accountType"])"
            end
            if "coin" in keys(account)
                for coin in account["coin"]
                    push!(df, (coin=coin["coin"], locked=coin["locked"], free=coin["availableToWithdraw"]))
                end
            end
        end
    else
        @warn "unexpected missing Bybit balance info: $response"
    end
    return df
end


# Websockets functions

function wsFunction(bc::BybitCache, channel::Channel, ws::String, symbol::String)
    @assert false "not implemented for Bybit"
    HTTP.WebSockets.open(string(BYBIT_API_WS, uppercase(symbol), ws); verbose=false) do io
      while !eof(io);
        put!(channel, _r2j(readavailable(io)))
    end
  end
end

function wsTradeAgg(bc::BybitCache, channel::Channel, symbol::String)
    @assert false "not implemented for Bybit"
    wsFunction(channel, "@aggTrade", symbol)
end

function wsTradeRaw(bc::BybitCache, channel::Channel, symbol::String)
    @assert false "not implemented for Bybit"
    wsFunction(channel, "@trade", symbol)
end

function wsDepth(bc::BybitCache, channel::Channel, symbol::String; level=5)
    @assert false "not implemented for Bybit"
    wsFunction(channel, string("@depth", level), symbol)
end

function wsDepthDiff(bc::BybitCache, channel::Channel, symbol::String)
    @assert false "not implemented for Bybit"
    wsFunction(channel, "@depth", symbol)
end

function wsTicker(bc::BybitCache, channel::Channel, symbol::String)
    @assert false "not implemented for Bybit"
    wsFunction(channel, "@ticker", symbol)
end

function wsTicker24Hr(bc::BybitCache, channel::Channel)
    @assert false "not implemented for Bybit"
    HTTP.WebSockets.open(string(BYBIT_API_WS, "!ticker@arr"); verbose=false) do io
      while !eof(io);
        put!(channel, _r2j(readavailable(io)))
    end
  end
end

function wsKline(bc::BybitCache, channel::Channel, symbol::String; interval="1m")
  #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
  @assert false "not implemented for Bybit"
  wsFunction(channel, string("@kline_", interval), symbol)
end

function wsKlineStreams(bc::BybitCache, channel::Channel, symbols::Array, interval="1m")
  #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
  @assert false "not implemented for Bybit"
  allStreams = map(s -> string(uppercase(s), "@kline_", interval), symbols)
    error = false;
    while !error
        try
            HTTP.WebSockets.open(string(BYBIT_API_WS,join(allStreams, "/")); verbose=false) do io
            while !eof(io);
                put!(channel, String(readavailable(io)))
            end
      end
        catch e
            println(e)
            error=true;
            println("error occured bailing wsklinestreams !")
        end
    end
end

function wsKlineStreams(bc::BybitCache, callback::Function, symbols::Array; interval="1m")
    #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    @assert false "not implemented for Bybit"
      allStreams = map(s -> string(uppercase(s), "@kline_", interval), symbols)
      @async begin
        HTTP.WebSockets.open(string("wss://stream.binance.com:9443/ws/",join(allStreams, "/")); verbose=false) do io
            while !eof(io)
                    data = String(readavailable(io))
                    callback(data)
            end
        end
    end
end

function openUserData(bc::BybitCache, apiKey)
    @assert false "not implemented for Bybit"
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    r = HTTP.request("POST", BYBIT_API_USER_DATA_STREAM, headers)
    return _r2j(r.body)["listenKey"]
end

function keepAlive(bc::BybitCache, apiKey, listenKey)
    @assert false "not implemented for Bybit"
    if length(listenKey) == 0
        return false
    end

    headers = Dict("X-BAPI-API-KEY" => apiKey)
    body = string("listenKey=", listenKey)
    r = HTTP.request("PUT", BYBIT_API_USER_DATA_STREAM, headers, body)
    return true
end

function closeUserData(bc::BybitCache, apiKey, listenKey)
    @assert false "not implemented for Bybit"
    if length(listenKey) == 0
        return false
    end
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    body = string("listenKey=", listenKey)
    r = HTTP.request("DELETE", BYBIT_API_USER_DATA_STREAM, headers, body)
   return true
end

function wsUserData(bc::BybitCache, channel::Channel, apiKey, listenKey; reconnect=true)
    @assert false "not implemented for Bybit"

    function mykeepAlive()
        return keepAlive(apiKey, listenKey)
    end

    Timer(mykeepAlive, 1800; interval = 1800)

    error = false;
    while !error
        try
            HTTP.WebSockets.open(string(BYBIT_API_WS, listenKey); verbose=false) do io
                while !eof(io);
                    put!(channel, _r2j(readavailable(io)))
                end
            end
        catch x
            println(x)
            error = true;
        end
    end

    if reconnect
        wsUserData(channel, apiKey, openUserData(apiKey))
    end

end

# helper
filterOnRegex(matcher, withDictArr; withKey="symbol") = filter(x -> match(Regex(matcher), !isnothing(x[withKey])), withDictArr);


end


