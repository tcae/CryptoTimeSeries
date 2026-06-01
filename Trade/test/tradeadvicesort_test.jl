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
testdt = DateTime(2026, 1, 1)
tav = [
    Trade.StrategyAdvice(classifier=tc.cl, configid=0, tradelabel=longbuy, relativeamount=1f0, base="BTC", price=123f0, datetime=testdt, hourlygain=1.2f0, probability=1f0, investmentid=0)
    Trade.StrategyAdvice(classifier=tc.cl, configid=0, tradelabel=longclose, relativeamount=1f0, base="BTC", price=123f0, datetime=testdt, hourlygain=1.2f0, probability=1f0, investmentid=0)
    Trade.StrategyAdvice(classifier=tc.cl, configid=0, tradelabel=shortbuy, relativeamount=1f0, base="BTC", price=123f0, datetime=testdt, hourlygain=1.1f0, probability=1f0, investmentid=0)
]
println("before sort!:\n$tav")
sort!(tav, lt=Trade.tradeadvicelessthan)
println("after sort!:\n$tav")
