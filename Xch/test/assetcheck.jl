using Dates, DataFrames
using Ohlcv, EnvConfig, Xch, Bybit, Trade

# EnvConfig.init(training)
# xc = Xch.XchCache()
# Xch.updateasset!(xc, "BTC", 0.001, 0.002)
# Xch.updateasset!(xc, "XRP", 0.001, 0.002)
# Xch.updateasset!(xc, "USDT", 0.001, 0.002)

EnvConfig.init(production)
xc = Xch.XchCache()

function assetcheck()
    cache = Trade.TradeCache(xc=xc)
    tcdf, assets = Trade.assetsconfig!(cache)
    println("portfolio: $assets")
    println("trading strategy: tc=$(tcdf)")
    println("openenabled coins=$(count(tcdf[!, :openenabled]))")
    println("coins to trade without assets: $(setdiff(tcdf[!, :basecoin], assets[!, :coin]))")
    println("assets that are not listed as tradable coins: $(setdiff(assets[!, :coin], tcdf[!, :basecoin]))")
    println("$(Xch.ttstr(xc)): $(Trade.USDTmsg(cache, assets))")
    return tcdf, assets
end

tcdf, assets = assetcheck()
# btcprice = tcdf[tcdf.basecoin .== "BTC", :lastprice][1,1]
# oid = Xch.createsellorder(xc, "btc", limitprice=btcprice, basequantity=10f0/btcprice, maker=false, marginleverage=2) # limitprice out of allowed range
# println("oid=$oid\n")
# oid = Xch.createbuyorder(xc, "btc", limitprice=btcprice, basequantity=10f0/btcprice, maker=false) # limitprice out of allowed range
# println("oid=$oid\n")
# tcdf, assets = assetcheck()
print("done")
