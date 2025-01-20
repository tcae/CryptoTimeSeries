using Dates, DataFrames
using Ohlcv, EnvConfig, CryptoXch, Bybit, Trade

# EnvConfig.init(training)
# xc = CryptoXch.XchCache()
# CryptoXch.updateasset!(xc, "BTC", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "XRP", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "USDT", 0.001, 0.002)

EnvConfig.init(production)
xc = CryptoXch.XchCache()

function assetcheck()
    tcdf, assets = Trade.assetsconfig!(Trade.TradeCache(xc=xc))
    println("portfolio: $assets")
    println("trading strategy: tc=$(tcdf)")
    println("buyenabled coins=$(count(tcdf[!, :buyenabled]))")
    println("coins to trade without assets: $(setdiff(tcdf[!, :basecoin], assets[!, :coin]))")
    println("assets that are not listed as tradable coins: $(setdiff(assets[!, :coin], tcdf[!, :basecoin]))")
    println("$(CryptoXch.ttstr(xc)): $(Trade.USDTmsg(assets))")
    return tcdf, assets
end

tcdf, assets = assetcheck()
# btcprice = tcdf[tcdf.basecoin .== "BTC", :lastprice][1,1]
# oid = CryptoXch.createsellorder(xc, "btc", limitprice=btcprice, basequantity=10f0/btcprice, maker=false, marginleverage=2) # limitprice out of allowed range
# println("oid=$oid\n")
# oid = CryptoXch.createbuyorder(xc, "btc", limitprice=btcprice, basequantity=10f0/btcprice, maker=false) # limitprice out of allowed range
# println("oid=$oid\n")
# tcdf, assets = assetcheck()
print("done")
