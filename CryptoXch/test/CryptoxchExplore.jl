
module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

function balances_test()
    result = CryptoXch.balances()
    display(result)
    display(EnvConfig.bases)
    display(EnvConfig.trainingbases)
    display(EnvConfig.datapath)
end

# EnvConfig.init(test)
EnvConfig.init(production)
# balances_test()

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")
# res = Binance.getKlines("BTCUSDT"; startDateTime=startdt, endDateTime=enddt, interval="1m")
# display(res)
# display(last(res[:body], 3))
# display(first(res[:body], 3))
# display(res[:body][1:3, :])
# display(res[:body][end-3:end, :])

# Binance.wsKlineStreams(cb, ["BTCUSDT", "XRPUSDT"])


function initialbtcdownload()
    startdt = DateTime("2022-01-02T22:45:03")
    enddt = DateTime("2022-01-02T22:49:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end


# oo1 = CryptoXch.createorder("btc", "BUY", 19001.0, 20.0)
# println("createorder: $oo1")
# oo2 = CryptoXch.getorder("btc", oo1["orderId"])
# println("getorder: $oo2")
# ooarray = CryptoXch.getopenorders(nothing)
# println("getopenorders(nothing): $ooarray")
# ooarray = CryptoXch.getopenorders("btc")
# println("getopenorders(\"btc\"): $ooarray")
# oo2 = CryptoXch.cancelorder("btc", oo1["orderId"])
# println("cancelorder: $oo2")

function showxchinfo()
    exchangeinfo = Bybit.getExchangeInfo()

    println("Binance server time: $(Dates.unix2datetime(exchangeinfo["serverTime"]/1000)) $(exchangeinfo["timezone"]) - entries: $(keys(exchangeinfo))")
    println("Rate limits ($(length(exchangeinfo["rateLimits"]))):")
    for ratelimit in exchangeinfo["rateLimits"]
        println("\t---")
        for (rl, rlvalue) in ratelimit
            println("\t\t $rl : $rlvalue")
        end
    end
    println("Exchange Filters ($(length(exchangeinfo["exchangeFilters"]))):")
    for xchfilter in exchangeinfo["exchangeFilters"]
        println("\t---")
        for (xf, xfvalue) in xchfilter
            println("\t\t $xf : $xfvalue")
        end
    end
    println("Symbols ($(length(exchangeinfo["symbols"]))):")
    count = 0
    for sym in exchangeinfo["symbols"]
        if (sym["quoteAsset"] == "USDT") && sym["isSpotTradingAllowed"] && ("LIMIT" in sym["orderTypes"]) && ("TRADING" == sym["status"])
            println("\t$(sym["symbol"]) base: $(sym["baseAsset"]) with precision $(sym["baseAssetPrecision"]) quote: $(sym["quoteAsset"]) with precision $(sym["quoteAssetPrecision"])")
            if sym["baseAssetPrecision"] != sym["baseCommissionPrecision"]
                @warn "baseAssetPrecision of $(sym["baseAssetPrecision"]) != baseCommissionPrecision of $(sym["baseCommissionPrecision"])"
            end
            if sym["quoteAssetPrecision"] != sym["quoteCommissionPrecision"]
                @warn "quoteAssetPrecision of $(sym["quoteAssetPrecision"]) != quoteCommissionPrecision of $(sym["quoteCommissionPrecision"])"
            end
            count += 1
            for filter in sym["filters"]
                println("\t\t filterType $(filter["filterType"])")
                for (fek, fev) in filter
                    if fek != "filterType"
                        println("\t\t\t $fek : $fev")
                    end
                end
            end
            for (sk, sv) in sym
                println("\t\t $sk : $sv ($(typeof(sv)))")
            end
            if count > 3
                break
            end
        end
    end
    println("listed $count symbols")
end

function testorder(price, usdtvol)
    oo = nothing
    try
        oo = CryptoXch.createorder("btc", "BUY", price, usdtvol)
    catch err
        @error err
    end
    !isnothing(oo) && @info oo
end

# showxchinfo()

testorder(19001.0, 20.0)
testorder(19001.0001, 5.0)
testorder(19001.00000000002, 20.00008)

end  # module
