# using Pkg
# Pkg.add(["SHA", "JSON", "Dates", "Printf", "HTTP"])
module Bybit

import HTTP, SHA, JSON, Dates, Printf, Logging

# base URL of the ByBit API
BYBIT_API_REST = "https://api.bybit.com"
BYBIT_API_WS = "wss://stream.binance.com:9443/ws/"

const recv_window = "5000"


function apiKS()
    apiKey = get(ENV, "BYBIT_APIKEY", "")
    apiSecret = get(ENV, "BYBIT_SECRET", "")

    @assert apiKey != "" || apiSecret != "" "BYBIT_APIKEY/BYBIT_APISECRET should be present in the environment dictionary ENV"

    apiKey, apiSecret
end

function dict2Params(dict::Union{Dict, Nothing})
    params = ""
    if isnothing(dict)
        return params
    else
        for kv in dict
            params = string(params, "&$(kv[1])=$(kv[2])")
        end
        params[2:end]
    end
end

# signing with apiKey and apiSecret
function timestamp()
    Int64(floor(Dates.datetime2unix(Dates.now(Dates.UTC)) * 1000))
end

function hmac(key::Vector{UInt8}, msg::Vector{UInt8}, hash, blocksize::Int=64)
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

function doSign(queryString, apiSecret)
    bytes2hex(hmac(Vector{UInt8}(apiSecret), Vector{UInt8}(queryString), SHA.sha256))
end

function genSignature(time_stamp, payload, public_key, secret_key)
    param_str = time_stamp * public_key * recv_window * payload
    hash = doSign(param_str, secret_key)
    return hash
end

function HttpPrivateRequest(method, endPoint, payload, Info, public_key, secret_key)
    time_stamp = string(timestamp())
    signature = genSignature(time_stamp, payload, public_key, secret_key)
    headers = Dict(
        "X-BAPI-API-KEY" => public_key,
        "X-BAPI-SIGN" => signature,
        "X-BAPI-SIGN-TYPE" => "2",
        "X-BAPI-TIMESTAMP" => time_stamp,
        "X-BAPI-RECV-WINDOW" => recv_window,
        "Content-Type" => "application/json"
    )
    response = url = ""
    try
        if method == "POST"
            url = BYBIT_API_REST * endPoint
            response = HTTP.request(method, url, headers=headers, data=payload)
        else
            url = BYBIT_API_REST * endPoint * "?" * payload
            response = HTTP.request(method, url, headers=headers)
        end
        body = String(response.body)
        # println(body)
        parsed_body = JSON.parse(body)
        if parsed_body["retCode"] != 0
            println("HttpPrivateRequest $method, url=$url, headers=$headers, payload=$payload")
            println("public_key=$public_key, secret_key=$secret_key")
            println(body)
        end
        result = parsed_body["result"]
        # println(Info * " Elapsed Time : " * string(response.time))
        return result
    catch err
        println("HttpPrivateRequest $method, url=$url, headers=$headers, payload=$payload")
        println("public_key=$public_key, secret_key=$secret_key")
        println(err)
        rethrow()
    end
end

function HttpPublicRequest(method, endPoint, params::Union{Dict, Nothing}, Info)
    payload = dict2Params(params)
    response = url = ""
    try
        if method == "POST"
            url = BYBIT_API_REST * endPoint
            response = HTTP.request(method, url, data=payload)
        else
            url = BYBIT_API_REST * endPoint * "?" * payload
            response = HTTP.request(method, url)
        end
        # println("response status (typeof: $(typeof(response.status))): $(response.status)")
        # println("response headers (typeof: $(typeof(response.headers))): $(response.headers)")
        body = String(response.body)
        # println("response body (typeof: $(typeof(response.body))): $(body)")
        # println(body)
        parsed_body = JSON.parse(body)
        result = parsed_body["result"]
        # println(Info * " Elapsed Time : " * string(response.time))
        return result
    catch err
        println("HttpPublicRequest $method, url=$url, payload=$payload")
        println(err)
        rethrow()
    end
end

# function HTTP response 2 JSON
function r2j(response)
    JSON.parse(String(response))
end

##################### PUBLIC CALL's #####################

# Simple test if ByBit API is online
function ping()
    # received response: 200
    r = HTTP.request("GET", string(BYBIT_API_REST, "api/v1/ping"))
    r.status
end

# ByBit servertime
function serverTime() # Bybit tested
    # at 2022-01-01 17:14 local MET received 2022-01-01T16:14:09.849
    r = HTTP.request("GET", string(BYBIT_API_REST, "/v3/public/time"))
    result = r2j(r.body)
    Dates.unix2datetime(result["time"] / 1000)
end

function get24HR() # Bybit tested
    # 1869-element Vector{Any}:
    # Dict{String, Any}("weightedAvgPrice" => "0.07925733", "askQty" => "1.90000000", "quoteVolume" => "3444.47417461", "priceChangePercent" => "0.077", "count" => 98593, "lastPrice" => "0.07887600", "openPrice" => "0.07881500", "firstId" => 317932287, "lastQty" => "0.06160000", "openTime" => 1640966508178…)
    # return HttpPublicRequest("GET", "/spot/v3/public/quote/ticker/24hr", nothing, "ticker/24h")["list"]
    return HttpPublicRequest("GET", "/v5/market/tickers", Dict("category" => "spot"), "ticker/24h")["list"]
end

function getDepth(symbol::String; limit=100) # 500(5), 1000(10)
    # getDepth("BTCUSDT"; limit=5)
    # Dict{String, Any} with 3 entries:
    # "lastUpdateId" => 16005065018
    # "asks"         => Any[Any["47402.00000000", "0.98372000"], Any["47402.28000000", "0.00851000"], Any["47402.29000000", "0.02434000"], Any["47407.71000000", "0.14470000"], Any["47407.72000000", "0.18910000"]]
    # "bids"         => Any[Any["47401.99000000", "0.05273000"], Any["47401.98000000", "0.10546000"], Any["47401.06000000", "0.00098000"], Any["47400.00000000", "0.00023000"], Any["47399.55000000", "0.01055000"]]
    r = HTTP.request("GET", string(BYBIT_API_DEPTH, "?symbol=", symbol,"&limit=",limit))
    r2j(r.body)
end

function get24HR(symbol::String) # Bybit tested
    # "category": "spot",
    # "list": [
    #     {
    #         "symbol": "BTCUSDT",
    #         "bid1Price": "20517.96",
    #         "bid1Size": "2",
    #         "ask1Price": "20527.77",
    #         "ask1Size": "1.862172",
    #         "lastPrice": "20533.13",
    #         "prevPrice24h": "20393.48",
    #         "price24hPcnt": "0.0068",
    #         "highPrice24h": "21128.12",
    #         "lowPrice24h": "20318.89",
    #         "turnover24h": "243765620.65899866",
    #         "volume24h": "11801.27771",
    #         "usdIndexPrice": "20784.12009279"
    #     }
    try
        response = HttpPublicRequest("GET", "/v5/market/tickers", Dict("category" => "spot", "symbol" => symbol), "ticker/24h")
        return response["list"][1]
    catch err
        rethrow()
    end
    # return HttpPublicRequest("GET", "/spot/v3/public/quote/ticker/24hr", Dict("symbol" => symbol), "ticker/24h")
end

function getExchangeInfo(symbol=nothing)
    # Dict{String, Any} with 5 entries:
    # "symbols"         => Any[Dict{String, Any}("orderTypes"=>Any["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"], "ocoAllowed"=>true, "isSpotTradingAllowed"=>true, "baseAssetPrecision"=>8, "quoteAsset"=>"BTC", "status"=>"TRADING", "icebergAllowed…
    # "rateLimits"      => Any[Dict{String, Any}("intervalNum"=>1, "interval"=>"MINUTE", "rateLimitType"=>"REQUEST_WEIGHT", "limit"=>1200), Dict{String, Any}("intervalNum"=>10, "interval"=>"SECOND", "rateLimitType"=>"ORDERS", "limit"=>50), Dict{String, Any}("intervalNum"=>1…
    # "exchangeFilters" => Any[]
    # "serverTime"      => 1641054370495
    # "timezone"        => "UTC"


    # GET /v5/market/instruments-info?category=spot&symbol=BTCUSDT HTTP/1.1    # "category": "spot",
    # "list": [
    #     {
    #         "symbol": "BTCUSDT",
    #         "baseCoin": "BTC",
    #         "quoteCoin": "USDT",
    #         "innovation": "0",
    #         "status": "Trading",
    #         "marginTrading": "both",
    #         "lotSizeFilter": {
    #TODO         "basePrecision": "0.000001",
    #TODO         "quotePrecision": "0.00000001",
    #TODO         "minOrderQty": "0.000048",
    #         "maxOrderQty": "71.73956243",
    #TODO         "minOrderAmt": "1",
    #         "maxOrderAmt": "2000000"
    #         },
    #         "priceFilter": {
    #TODO         "tickSize": "0.01"
    #     }
    # ]
    try
        if isnothing(symbol) || (symbol == "")
            response = HttpPublicRequest("GET", "/v5/market/instruments-info", Dict("category" => "spot"), "instruments-info")
        else
            response = HttpPublicRequest("GET", "/v5/market/instruments-info", Dict("category" => "spot", "symbol" => symbol), "instruments-info")
        end
        return response["list"]
    catch err
        rethrow()
    end
end

function getMarket()
    # error - not found
    r = HTTP.request("GET", "https://www.binance.com/exchange/public/product")
    r2j(r.body)["data"]
end

function getMarket(symbol::String)
    # error - not found
    r = HTTP.request("GET", string("https://www.binance.com/exchange/public/product?symbol=", symbol))
    r2j(r.body)["data"]
end

# ByBit get candlesticks/klines data
function getKlines(symbol; startDateTime=nothing, endDateTime=nothing, interval="1m")
    # getKlines("BTCUSDT")
    # 500-element Vector{Any}:
    # Any[1641024600000, "47092.03000000", "47107.42000000", "47085.31000000", "47098.98000000", "7.44682000", 1641024659999, "350714.74503740", 319, "2.75790000", "129875.86140450", "0"]
    query = string("?symbol=", symbol, "&interval=", interval)

    if !isnothing(startDateTime) && !isnothing(endDateTime)
        startTime = Printf.@sprintf("%.0d",Dates.datetime2unix(startDateTime) * 1000)
        endTime = Printf.@sprintf("%.0d",Dates.datetime2unix(endDateTime) * 1000)
        query = string(query, "&startTime=", startTime, "&endTime=", endTime)
    end
    r = HTTP.request("GET", string(BYBIT_API_KLINES, query))

    #! TCAE: HTTP response log inserted to understand errors - especially rate limit errors - headers may be good enough
    # julia> dump(HTTP.Response)
    # HTTP.Response <: Any
    # status::Int32
    # major::Int16
    # minor::Int16
    # cookies::Array{HTTP.Cookies.Cookie,1}
    # headers::Dict{String,String}
    # body::HTTP.FIFOBuffers.FIFOBuffer
    # request::Nullable{HTTP.Request}
    # history::Array{HTTP.Response,1}
    if r.status != 200
        filename = pwd() * "/$(Dates.now())HTTP-log.json"
        Logging.@warn "HTTP binanace klines request NOT OK returning status $stat - log file response: $filename"
        open(filename,"a") do io
            # JSON.print(io, r2j(r), 4)  # ERROR: LoadError: Unexpected character Line: 0 Around: ...HTTP/1.1 200 OK  Conte...
            println(io, r)
        end
    end

    return r.status, r2j(r.body)
end

##################### SECURED CALL's NEEDS apiKey / apiSecret #####################
function openOrders(symbol, apiKey::String, apiSecret::String)
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    if (symbol === nothing) || (length(symbol) == 0)
        query = string("recvWindow=50000&timestamp=", timestamp())
    else
        query = string("&symbol=", symbol, "&recvWindow=50000&timestamp=", timestamp())
    end
    r = HTTP.request("GET", string(BYBIT_API_REST, "api/v3/openOrders?", query, "&signature=", doSign(query, apiSecret)), headers)
    if r.status != 200
        println(r)
        return r.status
    end

    r2j(r.body)
end

function order(symbol, orderid, apiKey::String, apiSecret::String)
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    if !(symbol === nothing) && !(length(symbol) == 0) && (orderid > 0)
        query = string("&symbol=", symbol, "&orderId=", orderid, "&recvWindow=50000&timestamp=", timestamp())
    end
    r = HTTP.request("GET", string(BYBIT_API_REST, "api/v3/order?", query, "&signature=", doSign(query, apiSecret)), headers)
    if r.status != 200
        println(r)
        return r.status
    end

    r2j(r.body)
end

function cancelOrder(symbol, orderid, apiKey::String, apiSecret::String)
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    if !(symbol === nothing) && !(length(symbol) == 0) && (orderid > 0)
        query = string("&symbol=", symbol, "&orderId=", orderid, "&recvWindow=50000&timestamp=", timestamp())
    end
    r = HTTP.request("DELETE", string(BYBIT_API_REST, "api/v3/order?", query, "&signature=", doSign(query, apiSecret)), headers)
    if r.status != 200
        println(r)
        return r.status
    end

    r2j(r.body)
end

# function cancelOrder(symbol,origClientOrderId)
#     query = string("recvWindow=5000&timestamp=", timestamp(),"&symbol=", symbol,"&origClientOrderId=", origClientOrderId)
#     r = HTTP.request("DELETE", string(BYBIT_API_REST, "api/v3/order?", query, "&signature=", doSign(query)), headers)
#     r2j(r.body)
# end

function createOrder(symbol::String, orderSide::String;
    quantity::Float64=0.0, orderType::String = "LIMIT",
    price::Float64=0.0, stopPrice::Float64=0.0,
    icebergQty::Float64=0.0, newClientOrderId::String="")

      if quantity <= 0.0
          error("Quantity cannot be <=0 for order type.")
      end

      println("$orderSide => $symbol q: $quantity, p: $price ")

      order = Dict("symbol"           => symbol,
                      "side"             => orderSide,
                      "type"             => orderType,
                      "quantity"         => Printf.@sprintf("%.8f", quantity),
                      "newOrderRespType" => "FULL",
                      "recvWindow"       => 10000)

      if newClientOrderId != ""
          order["newClientOrderId"] = newClientOrderId;
      end

      if orderType == "LIMIT" || orderType == "LIMIT_MAKER"
          if price <= 0.0
              error("Price cannot be <= 0 for order type.")
          end
          order["price"] =  Printf.@sprintf("%.8f", price)
      end

      if orderType == "STOP_LOSS" || orderType == "TAKE_PROFIT"
          if stopPrice <= 0.0
              error("StopPrice cannot be <= 0 for order type.")
          end
          order["stopPrice"] = Printf.@sprintf("%.8f", stopPrice)
      end

      if orderType == "STOP_LOSS_LIMIT" || orderType == "TAKE_PROFIT_LIMIT"
          if price <= 0.0 || stopPrice <= 0.0
              error("Price / StopPrice cannot be <= 0 for order type.")
          end
          order["price"] =  Printf.@sprintf("%.8f", price)
          order["stopPrice"] =  Printf.@sprintf("%.8f", stopPrice)
      end

      if orderType == "TAKE_PROFIT"
          if price <= 0.0 || stopPrice <= 0.0
              error("Price / StopPrice cannot be <= 0 for STOP_LOSS_LIMIT order type.")
          end
          order["price"] =  Printf.@sprintf("%.8f", price)
          order["stopPrice"] =  Printf.@sprintf("%.8f", stopPrice)
      end

      if orderType == "LIMIT"  || orderType == "STOP_LOSS_LIMIT" || orderType == "TAKE_PROFIT_LIMIT"
          order["timeInForce"] = "GTC"
      end

      order
  end

# account call contains balances
function account(apiKey::String, apiSecret::String)
    endpoint = "/v5/account/info"
    method = "GET"
    params = ""
    myresult = HttpPrivateRequest(method, endpoint, params, "AccountInfo", apiKey, apiSecret)
end

function executeOrder(order::Dict, apiKey, apiSecret; execute=false)
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    query = string(dict2Params(order), "&timestamp=", timestamp())
    body = string(query, "&signature=", doSign(query, apiSecret))
    println(body)

    uri = "api/v3/order/test"
    if execute
        uri = "api/v3/order"
    end

    r = HTTP.request("POST", string(BYBIT_API_REST, uri), headers, body)
    r2j(r.body)
end

# returns default balances with amounts > 0
function balances(apiKey::String, apiSecret::String; balanceFilter = x -> parse(Float64, x["free"]) > 0.0 || parse(Float64, x["locked"]) > 0.0)
    acc = account(apiKey,apiSecret)
    balances = filter(balanceFilter, acc["balances"])
end


# Websockets functions

function wsFunction(channel::Channel, ws::String, symbol::String)
    HTTP.WebSockets.open(string(BYBIT_API_WS, lowercase(symbol), ws); verbose=false) do io
      while !eof(io);
        put!(channel, r2j(readavailable(io)))
    end
  end
end

function wsTradeAgg(channel::Channel, symbol::String)
    wsFunction(channel, "@aggTrade", symbol)
end

function wsTradeRaw(channel::Channel, symbol::String)
    wsFunction(channel, "@trade", symbol)
end

function wsDepth(channel::Channel, symbol::String; level=5)
    wsFunction(channel, string("@depth", level), symbol)
end

function wsDepthDiff(channel::Channel, symbol::String)
    wsFunction(channel, "@depth", symbol)
end

function wsTicker(channel::Channel, symbol::String)
    wsFunction(channel, "@ticker", symbol)
end

function wsTicker24Hr(channel::Channel)
    HTTP.WebSockets.open(string(BYBIT_API_WS, "!ticker@arr"); verbose=false) do io
      while !eof(io);
        put!(channel, r2j(readavailable(io)))
    end
  end
end

function wsKline(channel::Channel, symbol::String; interval="1m")
  #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    wsFunction(channel, string("@kline_", interval), symbol)
end

function wsKlineStreams(channel::Channel, symbols::Array, interval="1m")
  #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    allStreams = map(s -> string(lowercase(s), "@kline_", interval), symbols)
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

function wsKlineStreams(callback::Function, symbols::Array; interval="1m")
    #interval => 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
      allStreams = map(s -> string(lowercase(s), "@kline_", interval), symbols)
      @async begin
        HTTP.WebSockets.open(string("wss://stream.binance.com:9443/ws/",join(allStreams, "/")); verbose=false) do io
            while !eof(io)
                    data = String(readavailable(io))
                    callback(data)
            end
        end
    end
end

function openUserData(apiKey)
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    r = HTTP.request("POST", BYBIT_API_USER_DATA_STREAM, headers)
    return r2j(r.body)["listenKey"]
end

function keepAlive(apiKey, listenKey)
    if length(listenKey) == 0
        return false
    end

    headers = Dict("X-BAPI-API-KEY" => apiKey)
    body = string("listenKey=", listenKey)
    r = HTTP.request("PUT", BYBIT_API_USER_DATA_STREAM, headers, body)
    return true
end

function closeUserData(apiKey, listenKey)
    if length(listenKey) == 0
        return false
    end
    headers = Dict("X-BAPI-API-KEY" => apiKey)
    body = string("listenKey=", listenKey)
    r = HTTP.request("DELETE", BYBIT_API_USER_DATA_STREAM, headers, body)
   return true
end

function wsUserData(channel::Channel, apiKey, listenKey; reconnect=true)

    function mykeepAlive()
        return keepAlive(apiKey, listenKey)
    end

    Timer(mykeepAlive, 1800; interval = 1800)

    error = false;
    while !error
        try
            HTTP.WebSockets.open(string(BYBIT_API_WS, listenKey); verbose=false) do io
                while !eof(io);
                    put!(channel, r2j(readavailable(io)))
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


