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
startdt = Dates.now(UTC)
tc = TradingStrategy.readconfig!(TradingStrategy.TradeConfig(xc), startdt)

sort!(tc.cfg, [:basecoin])
tc.cfg = tc.cfg[!, Not([:startdt, :enddt, :totalgain, :mediangain, :meangain, :cumgain, :maxcumgain, :mincumgain])]
tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
println("portfolio: $assets")
println("trading strategy: tc=$(tc.cfg)")
println("buysell coins=$(count(tc.cfg[!, :buysell]))")
println("coins to trade without assets: $(setdiff(tc.cfg[!, :basecoin], assets[!, :coin]))")
println("assets that are not listed as tradable coins: $(setdiff(assets[!, :coin], tc.cfg[!, :basecoin]))")
end
