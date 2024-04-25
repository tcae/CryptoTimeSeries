module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit, TradingStrategy, Trade

# EnvConfig.init(training)
# xc = CryptoXch.XchCache(true)
# CryptoXch.updateasset!(xc, "BTC", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "XRP", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "USDT", 0.001, 0.002)

EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

tcdf, assets = TradingStrategy.assetsconfig!(TradingStrategy.TradeConfig(xc))
println("portfolio: $assets")
println("trading strategy: tc=$(tcdf)")
println("buysell coins=$(count(tcdf[!, :buysell]))")
println("coins to trade without assets: $(setdiff(tcdf[!, :basecoin], assets[!, :coin]))")
println("assets that are not listed as tradable coins: $(setdiff(assets[!, :coin], tcdf[!, :basecoin]))")
println("$(CryptoXch.ttstr(xc)): $(Trade.USDTmsg(assets))")
end
