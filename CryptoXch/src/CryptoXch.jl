# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])


module CryptoXch
using Dates, DataFrames, DataAPI, JDF, CSV, Logging
using MyBinance, EnvConfig, Ohlcv, TestOhlcv
import Ohlcv: intervalperiod


basenottradable = ["boba",
    "btt", "bcc", "ven", "pax", "bchabc", "bchsv", "usds", "nano", "usdsb", "erd", "npxs", "storm", "hc", "mco",
    "bull", "bear", "ethbull", "ethbear", "eosbull", "eosbear", "xrpbull", "xrpbear", "strat", "bnbbull", "bnbbear",
    "xzc", "gxs", "lend", "bkrw", "dai", "xtzup", "xtzdown", "bzrx", "eosup", "eosdown", "ltcup", "ltcdown"]
basestablecoin = ["usdt", "tusd", "busd", "usdc", "eur"]
baseignore = []
baseignore = append!(baseignore, basestablecoin, basenottradable)
minimumquotevolume = 10  # USDT

function klines2jdict(jsonkline)
    Dict(
        :opentime => Dates.unix2datetime(jsonkline[1]/1000),
        :open => parse(Float32, jsonkline[2]),
        :high => parse(Float32, jsonkline[3]),
        :low => parse(Float32, jsonkline[4]),
        :close => parse(Float32, jsonkline[5]),
        :basevolume => parse(Float32, jsonkline[6]),
        :closetime => Dates.unix2datetime(jsonkline[7]/1000),
        :quotevolume => parse(Float32, jsonkline[8]),
        :nbrtrades => Int32(jsonkline[9]),
        :takerbuybasevolume => parse(Float32, jsonkline[10]),
        :takerbuyquotevolume => parse(Float32, jsonkline[11]),
        :ignore => jsonkline[12]
    )
end

function klines2jdf(jsonkline)
    df = DataFrames.DataFrame()
    if ismissing(jsonkline)
        df = DataFrame(
            opentime=Float32[],
            open=Float32[],
            high=Float32[],
            low=Float32[],
            close=Float32[],
            basevolume=Float32[]
            # quotevolume=Float32[]
            )
    else
        len = length(jsonkline)
        df[:, :opentime] = [Dates.unix2datetime(jsonkline[ix][1]/1000) for ix in 1:len]
        df[:, :open] = [parse(Float32, jsonkline[ix][2]) for ix in 1:len]
        df[:, :high] = [parse(Float32, jsonkline[ix][3]) for ix in 1:len]
        df[:, :low] = [parse(Float32, jsonkline[ix][4]) for ix in 1:len]
        df[:, :close] = [parse(Float32, jsonkline[ix][5]) for ix in 1:len]
        df[:, :basevolume] = [parse(Float32, jsonkline[ix][6]) for ix in 1:len]
        Ohlcv.addpivot!(df)
        # df.quotevolume = [parse(Float32, jsonkline[ix][8]) for ix in 1:len]
    end
    return df
end

defaultcryptoexchange = "binance"

"""
Requests base/USDT from start until end (both including) in interval frequency but maximum 1000 entries

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function ohlcfromexchange(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    rstatus = 0
    df = nothing
    if EnvConfig.configmode == EnvConfig.test
        println("ohlcfromexchange test mode: $base, $startdt, $enddt, $interval, $cryptoquote")
        # if base in EnvConfig.bases
            rstatus, df = TestOhlcv.testdataframe(base, startdt, enddt, interval, cryptoquote)
        # else
        #     df = Ohlcv.defaultohlcvdataframe()
        #     rstatus = 112
        #     @warn "$base is an unknown base for EnvConfig.test mode"
        # end
    end
    if (EnvConfig.configmode != EnvConfig.test) || (rstatus == 111)
        try
            symbol = uppercase(base*cryptoquote)
            # println("symbol=$symbol start=$startdt end=$enddt")
            rstatus, arr = MyBinance.getKlines(symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
            # println(typeof(r))
            # show(r)
            # arr = MyBinance.r2j(r.body)
            df = klines2jdf(arr)
            # return Dict(:status => r.status, :headers => r.headers, :body => df, :version => r.version, :request => r.request)
        catch e
            Logging.@warn "exception $e detected"
            df = klines2jdf(missing)
        end
    end
    return rstatus, df
end

function getlastminutesdata()
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Minute(7)
    stat, res = ohlcfromexchange("BTCUSDT", startdt, enddt)
    # display(nrow(res))
    println("getlastminutesdata $stat")
    display(last(res, 3))
    # display(first(res, 3))
    enddt = Dates.now(Dates.UTC)
    stat, res2 = ohlcfromexchange("BTCUSDT", enddt - Dates.Second(1), enddt)
    println("getlastminutesdata $stat")
    # display(res)
    display(nrow(res2))
    display(last(res2, 3))
    println("dates equal? $(res[end, :opentime]==res2[end, :opentime])")
    # display(first(res2, 3))
    # display(res[:body][1:3, :])
    # display(res[:body][end-3:end, :])
end

"""
Requests base/USDT from start until end (both including) in interval frequency

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

time gaps will not be filled
"""
function gethistoryohlcv(base::String, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    # println("requesting from $startdt until $enddt $(ceil(enddt - startdt, intervalperiod(interval)) + intervalperiod(interval)) $base OHLCV from binance")

    notreachedenddate = true
    df = Ohlcv.defaultohlcvdataframe()
    lastdt = startdt - Dates.Minute(1)  # make sure lastdt break condition is not true
    while notreachedenddate
        stat, res = ohlcfromexchange(base, startdt, enddt, interval)
        # display(nrow(res))
        # display(first(res, 3))
        # display(last(res, 3))
        if stat != 200  # == NOT OK
            Logging.@warn "HTTP binance klines request NOT OK returning status $stat"
            break
        end
        if size(res, 1) == 0
            # Logging.@warn "no $base $interval data returned by last ohlcv read from $startdt until $enddt"
            break
        end
        notreachedenddate = (res[end, :opentime] < enddt)
        if res[end, :opentime] <= lastdt
            # no progress since last ohlcv  read
            Logging.@warn "no progress since last ohlcv read"
            break
        end
        lastdt = res[end, :opentime]
        # println("$(Dates.now()) read $(nrow(res)) $base from $startdt until $lastdt")
        startdt = floor(lastdt, intervalperiod(interval))
        while (size(df,1) > 0) && (res[begin, :opentime] <= df[end, :opentime])  # replace last row with updated data
            deleteat!(df, size(df, 1))
        end
        if (size(res, 1) > 0) && (names(df) == names(res))
            df = vcat(df, res)
        else
            Logging.@error "vcat data frames names not matching df: $(names(df)) - res: $(names(res))"
            break
        end
    end
    # display(nrow(df))
    # display(first(df, 3))
    # display(last(df, 3))
    return df
end

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from Binance.

- ohlcv containes the requested base identifier and interval - the result will be stored in the data frame of this structure
- startdt and enddt are DateTime stamps that specify the requested time range
- if closecachegap==true then any gap to chached data will be closed when asking for missing data from Binance

This function reduces the returned ohlcv to the requested time range (different to cryptodownload).
Therefore, don't use the ohlcv to write to stored cache because the stored history will be overridden and is lost.
"""
function cryptoupdate!(ohlcv, startdt, enddt, closecachegap=false)
    base = ohlcv.base
    interval = ohlcv.interval
    println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt <= startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    olddf = Ohlcv.dataframe(ohlcv)
    if size(olddf, 1) > 0  # there is already data available
        if (startdt < olddf[begin, :opentime])
            if (enddt >= olddf[begin, :opentime]) || closecachegap
                # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
                tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
                # get data of a timerange before the already available data
                newdf = gethistoryohlcv(base, startdt, tmpdt, interval)
                if size(newdf, 1) > 0
                    if names(olddf) == names(newdf)
                        olddf = vcat(newdf, olddf)
                    else
                        Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                    end
                end
                Ohlcv.setdataframe!(ohlcv, olddf)
            end
        end
        if (enddt > olddf[end, :opentime])
            if (startdt <= olddf[end, :opentime]) || closecachegap
                tmpdt = olddf[end, :opentime]  # update last data row
                newdf = gethistoryohlcv(base, tmpdt, enddt, interval)
                if size(newdf, 1) > 0
                    while (size(olddf, 1) > 0) && (newdf[begin, :opentime] <= olddf[end, :opentime])  # replace last row with updated data
                        deleteat!(olddf, size(olddf, 1))
                    end
                    if names(olddf) == names(newdf)
                        olddf = vcat(olddf, newdf)
                    else
                        Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                    end
                end
                Ohlcv.setdataframe!(ohlcv, olddf)
            end
        end
        subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
    end
    if size(olddf, 1) == 0
        newdf = gethistoryohlcv(base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    return ohlcv
end

"""
Returns the OHLCV data of the requested time range by first checking the stored cache data and if unsuccessful requesting it from Binance.

    - *base* identifier and interval specify what data is requested - the result will be returned as OhlcvData structure
    - startdt and enddt are DateTime stamps that specify the requested time range
    - any gap to chached data will be closed when asking for missing data from Binance
    - beside returning the ohlcv data to the caller, it is also written to stored cache to reduce slow data request to Binance
"""
function cryptodownload(base, interval, startdt, enddt)::OhlcvData
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt <= startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    Ohlcv.read!(ohlcv)
    olddf = Ohlcv.dataframe(ohlcv)
    if size(olddf, 1) > 0  # there is already data available
        if startdt < olddf[begin, :opentime]
            # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
            tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
            # get data of a timerange before the already available data
            newdf = gethistoryohlcv(base, startdt, tmpdt, interval)
            if size(newdf, 1) > 0
                if names(olddf) == names(newdf)
                    olddf = vcat(newdf, olddf)
                else
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
        if enddt > olddf[end, :opentime]
            tmpdt = olddf[end, :opentime]  # update last data row
            newdf = gethistoryohlcv(base, tmpdt, enddt, interval)
            if size(newdf, 1) > 0
                while (size(olddf, 1) > 0) && (newdf[begin, :opentime] <= olddf[end, :opentime])  # replace last row with updated data
                    deleteat!(olddf, size(olddf, 1))
                end
                if names(olddf) == names(newdf)
                    olddf = vcat(olddf, newdf)
                else
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
    else
        newdf = gethistoryohlcv(base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    if size(Ohlcv.dataframe(ohlcv), 1) > 0
        Ohlcv.write(ohlcv)
    else
        @warn "cryptodownload: No $base OHLCV data written due to empty dataframe"
    end
    return ohlcv
end

function downloadupdate!(bases, enddt, period=Dates.Year(4))
    count = length(bases)
    for (ix, base) in enumerate(bases)
        # break
        println()
        println("$(EnvConfig.now()) updating $base ($ix of $count)")
        startdt = enddt - period
        ohlcv = CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        ohlcv = Ohlcv.accumulate!(ohlcv, "1d")
        Ohlcv.write(ohlcv)
    end
end

floorbase(base, qty) = base == "usdt" ? floor(qty, digits=3) : floor(qty, digits=5)
roundbase(base, qty) = base == "usdt" ? round(qty, digits=3) : round(qty, digits=5)
# TODO read base specific digits from binance and use them base specific

onlyconfiguredsymbols(symbol) =
    endswith(symbol, uppercase(EnvConfig.cryptoquote)) &&
    !(lowercase(symbol[1:end-length(EnvConfig.cryptoquote)]) in baseignore)

"""
Returns a dataframe with 24h values of all USDT quote bases with the following columns:

- base
- quotevolume24h
- pricechangepercent
- lastprice

"""
function getUSDTmarket()
    df = DataFrame(
        base=String[],
        qte=String[],
        # weightedAvgPrice=Float32[],
        # askQty=Float32[],
        quotevolume24h=Float32[],
        pricechangepercent=Float32[],
        # count=Int64[],
        lastprice=Float32[],
        # openPrice=Float32[],
        # firstId=Int64[],
        # lastQty=Float32[],
        # openTime=Dates.DateTime[],
        # closeTime=Dates.DateTime[],
        # askPrice=Float32[],
        # priceChange=Float32[],
        # highprice=Float32[],
        # prevClosePrice=Float32[],
        # bidQty=Float32[],
        # volume=Float32[],
        # bidPrice=Float32[],
        # lastId=Int64[],
        # lowprice=Float32[]
    )
    if EnvConfig.configmode == EnvConfig.production
        p24dictarray = MyBinance.get24HR()
        for (index, p24dict) in enumerate(p24dictarray)
            if onlyconfiguredsymbols(p24dict["symbol"])
                # printorderinfo(index, oodict)
                push!(df, (
                    lowercase(replace(p24dict["symbol"], uppercase(EnvConfig.cryptoquote) => "")),
                    lowercase(EnvConfig.cryptoquote),
                    # parse(Float32, p24dict["weightedAvgPrice"]),
                    # parse(Float32, p24dict["askQty"]),
                    parse(Float32, p24dict["quoteVolume"]),
                    parse(Float32, p24dict["priceChangePercent"]),
                    # p24dict["count"],
                    parse(Float32, p24dict["lastPrice"]),
                    # parse(Float32, p24dict["openPrice"]),
                    # p24dict["firstId"],
                    # parse(Float32, p24dict["lastQty"]),
                    # Dates.unix2datetime(p24dict["openTime"] / 1000),
                    # Dates.unix2datetime(p24dict["closeTime"] / 1000),
                    # parse(Float32, p24dict["askPrice"]),
                    # parse(Float32, p24dict["priceChange"]),
                    # parse(Float32, p24dict["highPrice"]),
                    # parse(Float32, p24dict["prevClosePrice"]),
                    # parse(Float32, p24dict["bidQty"]),
                    # parse(Float32, p24dict["volume"]),
                    # parse(Float32, p24dict["bidPrice"]),
                    # p24dict["lastId"],
                    # parse(Float32, p24dict["lowPrice"])
                ))
            end
        end
    else  # test or training
        for base in EnvConfig.bases
            push!(df, (
                base,
                lowercase(EnvConfig.cryptoquote),
                # 0.0,        # "weightedAvgPrice"
                # 0.0,        # "askQty"
                15000000.0, # "quoteVolume"
                5.0,        # "priceChangePercent"
                # 2,          # "count"
                100.0,      # "lastPrice"
                # 100.0,      # "openPrice"
                # 20,         # "firstId"
                # 3,          # "lastQty"
                # DateTime("2019-01-02 01:11:58:121", "y-m-d H:M:S:s"),  # "openTime"
                # DateTime("2019-01-02 01:12:59:121", "y-m-d H:M:S:s"),  # "closeTime"
                # 100.0,      # "askPrice"
                # 0.0,        # "priceChange"
                # 100.0,      # "highPrice"
                # 100.0,      # "prevClosePrice"
                # 4.0,        # "bidQty"
                # 2.0,        # "volume"
                # 99.0,       # "bidPrice"
                # 21,         # "lastId"
                # 99.5        # "lowPrice"
            ))

        end
    end
    return df
end

function balances()
    portfolio = MyBinance.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
    # println(portfolio)
    return portfolio
end

function portfolio(usdtdf)
    df = DataFrame(
        base=String[],
        locked=Float32[],
        free=Float32[],
        usdt=Float32[]
        )

    if EnvConfig.configmode == EnvConfig.production
        portfolioarray = MyBinance.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
        for pdict in portfolioarray
            base = lowercase(pdict["asset"])
            freebase = parse(Float32, pdict["free"])
            lockedbase = parse(Float32, pdict["locked"])
            if (base in usdtdf.base) && !(base in basenottradable)
                lastprices = usdtdf[usdtdf.base .== base, :lastprice]
                usdtvolumebase = (freebase + lockedbase) * lastprices[begin]
                if usdtvolumebase >= minimumquotevolume
                    push!(df, (base, lockedbase, freebase, usdtvolumebase))
                end
            elseif base in basestablecoin
                push!(df, (base, lockedbase, freebase, 1.0))
            end
        end
    else  # test or training
        initialusdt = 10000.0
        push!(df, ("usdt", 0.0, initialusdt, initialusdt))
    end
    return df
end

function downloadallUSDT(enddt, period=Dates.Year(4), minimumdayquotevolume = 10000000)
    df = getUSDTmarket()
    df = df[df.quotevolume24h .> minimumdayquotevolume , :]
    count = size(df, 1)
    for (ix, base) in enumerate(df[!, :base])
        break
        println()
        println("$(EnvConfig.now()) updating $base ($ix of $count)")
        startdt = enddt - period
        CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        # ohlcv = Ohlcv.defaultohlcv(base)
        # Ohlcv.setinterval!(ohlcv, "1m")
        # Ohlcv.read!(ohlcv)
        # olddf = Ohlcv.dataframe(ohlcv)
        # if size(olddf, 1) > 0
        #     startdt = olddf[end, :opentime]
        #     cryptoupdate!(ohlcv, floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        #     Ohlcv.write(ohlcv)
        # else
        #     startdt = enddt - period
        #     CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        # end
    end
    return df
end

function printorderinfo(orderix, oodict)
    println("order #$orderix")
    for (key, value) in oodict
        if key in ["updateTime", "time"]
            println("key: $key value: $(Dates.unix2datetime(value/1000)) of type: $(typeof(value))")
        elseif  key in ["price", "origQty"]
            val32 = parse(Float32, value)
            val64 = parse(Float64, value)
            println("key: $key value: $val32 of type: $(typeof(val32)) value: $val64 of type: $(typeof(val64))")
        else
            println("key: $key value: $value of type: $(typeof(value))")
        end
    end
end

"""
Returns a data frame with filled with the binance response as provided in `orderdictarray` with a `logtimeutc` timestamp per row.
If `orderdictarray` is empty then an empty dataframe is returned.
"""
function orderdataframe(orderdictarray)
    df = DataFrame(
        base=String[],
        orderId=Int64[],
        # clientOrderId=String[],
        price=Float32[],
        origQty=Float32[],
        executedQty=Float32[],
        # cummulativeQuoteQty=Float32[],
        status=String[],
        timeInForce=String[],
        type=String[],
        side=String[]
        # stopPrice=Float32[],
        # icebergQty=Float32[],
        # time=Dates.DateTime[],
        # updateTime=Dates.DateTime[],
        # isWorking=Bool[],
        # origQuoteOrderQty=Float32[],
        )

    for oodict in orderdictarray
        if onlyconfiguredsymbols(oodict["symbol"])
            push!(df, (
                lowercase(replace(oodict["symbol"], uppercase(EnvConfig.cryptoquote) => "")),
                oodict["orderId"],
                # oodict["clientOrderId"],
                parse(Float32, oodict["price"]),
                parse(Float32, oodict["origQty"]),
                parse(Float32, oodict["executedQty"]),
                # parse(Float32, oodict["cummulativeQuoteQty"]),
                oodict["status"],
                oodict["timeInForce"],
                oodict["type"],
                oodict["side"]
                # parse(Float32, oodict["stopPrice"]),
                # parse(Float32, oodict["icebergQty"]),
                # Dates.unix2datetime(oodict["time"] / 1000),
                # Dates.unix2datetime(oodict["updateTime"] / 1000),
                # oodict["isWorking"],
                # parse(Float32, oodict["origQuoteOrderQty"]),
                ))
                # println(df)
            if "fills" in keys(oodict)
                println("FILLS on $(oodict["status"]) $(oodict["side"]) order $(oodict["orderId"]) of $(oodict["symbol"]): $(oodict["fills"])")
            end
        else
            @warn "$(EnvConfig.now()) getopenorders: ignoring $(oodict["symbol"]) as not configured symbol"
            # printorderinfo(index, oodict)
        end
    end
    # "symbol": "LTCBTC",
    # "orderId": 1,
    # "orderListId": -1, //Unless OCO, the value will always be -1
    # "clientOrderId": "myOrder1",
    # "price": "0.1",
    # "origQty": "1.0",
    # "executedQty": "0.0",
    # "cummulativeQuoteQty": "0.0",
    # "status": "NEW",
    # "timeInForce": "GTC",
    # "type": "LIMIT",
    # "side": "BUY",
    # "stopPrice": "0.0",
    # "icebergQty": "0.0",
    # "time": 1499827319559,
    # "updateTime": 1499827319559,
    # "isWorking": true,
    # "origQuoteOrderQty": "0.000000"
    return df
end

function getopenorders(base)
    symbol = isnothing(base) ? nothing : uppercase(base * EnvConfig.cryptoquote)
    if EnvConfig.configmode == EnvConfig.production
        ooarray = MyBinance.openOrders(symbol, EnvConfig.authorization.key, EnvConfig.authorization.secret)
        df = orderdataframe(ooarray)
        return df
    else
    end
end

function getorder(base, orderid)
    symbol = uppercase(base * EnvConfig.cryptoquote)
    if EnvConfig.configmode == EnvConfig.production
        oo = MyBinance.order(symbol, orderid, EnvConfig.authorization.key, EnvConfig.authorization.secret)
        df = orderdataframe([oo])
        return df
    else
    end
end

function cancelorder(base, orderid)
    symbol = uppercase(base * EnvConfig.cryptoquote)
    if EnvConfig.configmode == EnvConfig.production
        oo = MyBinance.cancelOrder(symbol, orderid, EnvConfig.authorization.key, EnvConfig.authorization.secret)
        df = orderdataframe([oo])
        return df
        # "symbol": "LTCBTC",
        # "origClientOrderId": "myOrder1",
        # "orderId": 4,
        # "orderListId": -1, //Unless part of an OCO, the value will always be -1.
        # "clientOrderId": "cancelMyOrder1",
        # "price": "2.00000000",
        # "origQty": "1.00000000",
        # "executedQty": "0.00000000",
        # "cummulativeQuoteQty": "0.00000000",
        # "status": "CANCELED",
        # "timeInForce": "GTC",
        # "type": "LIMIT",
        # "side": "BUY"
    else
    end
end

function createorder(base::String, orderside::String, limitprice, usdtquantity)
    symbol = uppercase(base * EnvConfig.cryptoquote)
    qty = floor(usdtquantity / limitprice; digits=4)  #* round due to LOT_FILTER minimum granularity constraint
    order = MyBinance.createOrder(symbol, orderside; quantity=qty, orderType="LIMIT", price=limitprice)
    oo = MyBinance.executeOrder(order, EnvConfig.authorization.key, EnvConfig.authorization.secret; execute=true)
    println(oo)
    df = orderdataframe([oo])
    return df
    # "symbol": "BTCUSDT",
    # "orderId": 28,
    # "orderListId": -1, //Unless OCO, value will be -1
    # "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
    # "transactTime": 1507725176595,
    # "price": "0.00000000",
    # "origQty": "10.00000000",
    # "executedQty": "10.00000000",
    # "cummulativeQuoteQty": "10.00000000",
    # "status": "FILLED",
    # "timeInForce": "GTC",
    # "type": "MARKET",
    # "side": "SELL",
    # "fills": [...

end

EnvConfig.init(EnvConfig.production)
# EnvConfig.init(EnvConfig.test)
# oo = createorder("btc", "BUY", 18850.0, 90.0)
# oo = getopenorders(nothing)
# oo = getopenorders("btc")
# println(oo)
# oo = getorder("btc", 10759633755)
# oo = cancelorder("btc", 10767791414)
# println(oo)
# getbalances()
# println(getUSDTmarket())

# ap = MyBinance.getAllPrices()
# p24 = MyBinance.get24HR()
# println("len(all prices)=$(length(ap)) len(24HR)=$(length(p24))")
# println(p24[1])
# for (key, value) in p24[1]
#     println("key: $key  value: $value  $(typeof(value))")
# end
# for p in ap
#     println(p)
# end

# usdtdf = getUSDTmarket()
# println("getUSDTmarket #$(size(usdtdf,1)) rows cols: $(names(usdtdf))")
# pdf = portfolio(usdtdf)
# println(pdf)
# usdtdf = usdtdf[usdtdf.quotevolume24h .> 10000000, :]
# println("getUSDTmarket #$(size(usdtdf,1)) rows cols: $(names(usdtdf))")
# println(df)
end  # of module
