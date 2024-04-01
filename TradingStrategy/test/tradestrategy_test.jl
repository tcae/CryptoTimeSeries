module TradingStrategyTest

using Test, Dates, Logging, LoggingExtras
using EnvConfig, TradingStrategy, Classify, Features, Ohlcv, CryptoXch

println("$(EnvConfig.now()): started")

Classify.verbosity = 2
Ohlcv.verbosity = 1
Features.verbosity = 1
# EnvConfig.init(training)
EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

startdt = DateTime("2024-03-19T00:00:00")
enddt = DateTime("2024-03-30T10:03:00")
assets = CryptoXch.portfolio!(xc)
tc = TradingStrategy.train!(TradingStrategy.TradeConfig(xc), assets[!, :coin]; enddt=enddt)
println("trading startegy: tc=$(tc.cfg)")
# startdt = Dates.now(UTC)
# enddt = nothing

end  # module