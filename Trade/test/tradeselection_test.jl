module TradingStrategyTest

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Features, Ohlcv, CryptoXch

println("TradingStrategyTest tradestrategy_test")
println("$(EnvConfig.now()): started")

Classify.verbosity = 2
Ohlcv.verbosity = 1
Features.verbosity = 1
Trade.verbosity = 3
# EnvConfig.init(training)
EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

dummy = DateTime("2000-01-01T00:00:00")
# startdt = DateTime("2024-03-19T00:00:00")
startdt = Dates.now(UTC) # - Hour(1)
# startdt = DateTime("2024-04-15T06:00:00")
enddt = nothing  # DateTime("2024-03-30T10:03:00")
assets = CryptoXch.portfolio!(xc)
# tc = TradingStrategy.read!(TradingStrategy.TradeConfig(xc), startdt)
tc = Trade.tradeselection!(Trade.TradeCache(xc=xc), assets[!, :coin]; datetime=startdt, minimumdayquotevolume=2*1000000, updatecache=true)
# tc = TradingStrategy.tradeselection!(TradingStrategy.TradeConfig(xc), ["BTC", "MATIC"]; datetime=startdt, assetonly=true)


# df = DataFrame()
# for ohlcv in CryptoXch.ohlcv(xc)
#     size(ohlcv.df, 1) > 0 ? push!(df, (base=ohlcv.base, len=length(ohlcv.df[!, :opentime]), startdt=ohlcv.df[begin, :opentime], enddt=ohlcv.df[end, :opentime])) : push!(df, (base=ohlcv.base, len=0, startdt=dummy, enddt=dummy))
# end
# println("ohlcv data before timecut: $df")
# TradingStrategy.timerangecut!(tc, startdt, startdt)
# df = DataFrame()
# for ohlcv in CryptoXch.ohlcv(xc)
#     size(ohlcv.df, 1) > 0 ? push!(df, (base=ohlcv.base, len=length(ohlcv.df[!, :opentime]), startdt=ohlcv.df[begin, :opentime], enddt=ohlcv.df[end, :opentime])) : push!(df, (base=ohlcv.base, len=0, startdt=dummy, enddt=dummy))
# end
# println("ohlcv data after timecut: $df")
println("trading strategy: tc=$(tc.cfg)")
# startdt = Dates.now(UTC)
# enddt = nothing

end  # module