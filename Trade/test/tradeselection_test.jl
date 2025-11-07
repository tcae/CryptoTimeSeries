module TradingStrategyTest

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Features, Ohlcv, CryptoXch

println("TradingStrategyTest tradestrategy_test")
println("$(EnvConfig.now()): started")

Classify.verbosity = 1
Ohlcv.verbosity = 1
Features.verbosity = 1
Trade.verbosity = 2
CryptoXch.verbosity = 1
# EnvConfig.init(training)
EnvConfig.init(production)
# EnvConfig.init(test)
xc = CryptoXch.XchCache()

# startdt = DateTime("2024-03-19T00:00:00")
startdt = Dates.now(UTC) # - Hour(1)
# startdt = DateTime("2024-04-15T06:00:00")
enddt = nothing  # DateTime("2024-03-30T10:03:00")

# assets = CryptoXch.portfolio!(xc)
# tc = Trade.tradeselection!(Trade.TradeCache(xc=xc), assets[!, :coin]; datetime=startdt, updatecache=true)

tc = Trade.tradeselection!(Trade.TradeCache(xc=xc), []; datetime=startdt, updatecache=true)


# dummy = DateTime("2000-01-01T00:00:00")
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

# println("the second update should be much faster")
# tc = Trade.tradeselection!(tc, assets[!, :coin]; datetime=Dates.now(UTC), updatecache=true)
# println("trading strategy: tc=$(tc.cfg)")

end  # module