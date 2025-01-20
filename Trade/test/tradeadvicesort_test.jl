using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Targets, Ohlcv, CryptoXch

println("$(EnvConfig.now()): started")

Classify.verbosity = 2
Ohlcv.verbosity = 1
Features.verbosity = 1
Trade.verbosity = 3
# EnvConfig.init(training)
EnvConfig.init(test)
xc = CryptoXch.XchCache()
tc = Trade.TradeCache(xc=xc)
tav = [
    Classify.TradeAdvice(tc.cl, 0, longbuy, 1f0, "BTC", 123f0, nothing, 1.2, 1f0, 0)
    Classify.TradeAdvice(tc.cl, 0, longclose, 1f0, "BTC", 123f0, nothing, 1.2, 1f0, 0)
    Classify.TradeAdvice(tc.cl, 0, shortbuy, 1f0, "BTC", 123f0, nothing, 1.1, 1f0, 0)
]
println("before sort!:\n$tav")
sort!(tav, lt=Trade.tradeadvicelessthan)
println("after sort!:\n$tav")
