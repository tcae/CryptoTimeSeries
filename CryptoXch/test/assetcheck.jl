module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit, TradingStrategy

# EnvConfig.init(training)
# xc = CryptoXch.XchCache(true)
# CryptoXch.updateasset!(xc, "BTC", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "XRP", 0.001, 0.002)
# CryptoXch.updateasset!(xc, "USDT", 0.001, 0.002)

EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

# assets = CryptoXch.balances(xc,ignoresmallvolume=false)
# println("balances1: $assets")
# assets = CryptoXch.portfolio!(xc, ignoresmallvolume=false)
# println("portfolio1: $assets")
# assets = CryptoXch.balances(xc,ignoresmallvolume=true)
# println("balances2: $assets")
# assets = CryptoXch.portfolio!(xc, ignoresmallvolume=true)
# println("portfolio2: $assets")
# assets = CryptoXch.balances(xc)
# println("balances: $assets")
assets = CryptoXch.portfolio!(xc)
sort!(assets, [:coin])
println("portfolio: $assets")
startdt = Dates.now(UTC)
tc = TradingStrategy.readconfig!(TradingStrategy.TradeConfig(xc), startdt)
sort!(tc.cfg, [:basecoin])
println("trading strategy: tc=$(tc.cfg)")
println("buysell coins=$(count(tc.cfg[!, :buysell]))")
end
