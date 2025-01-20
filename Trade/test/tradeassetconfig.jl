using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Features, Ohlcv, CryptoXch

println("$(EnvConfig.now()): started")

Classify.verbosity = 2
Ohlcv.verbosity = 1
Features.verbosity = 1
Trade.verbosity = 3
# EnvConfig.init(training)
EnvConfig.init(production)
xc = CryptoXch.XchCache()

assets = CryptoXch.portfolio!(xc)
# tc = Trade.tradeselection!(Trade.TradeCache(xc=xc), assets[!, :coin]; datetime=Dates.now(UTC), updatecache=true) # revised config is saved

assetsconfig, assets = Trade.assetsconfig!(Trade.TradeCache(xc=xc)) # read latest from file merged with assets in correct format
println("trade assets config: tc=$(assetsconfig)\nassets=$assets")
