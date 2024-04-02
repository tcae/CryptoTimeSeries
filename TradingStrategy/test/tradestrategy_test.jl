module TradingStrategyTest

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, TradingStrategy, Classify, Features, Ohlcv, CryptoXch

println("$(EnvConfig.now()): started")

Classify.verbosity = 2
Ohlcv.verbosity = 1
Features.verbosity = 1
# EnvConfig.init(training)
EnvConfig.init(production)
xc = CryptoXch.XchCache(true)

dummy = DateTime("2000-01-01T00:00:00")
startdt = DateTime("2024-03-19T00:00:00")
enddt = DateTime("2024-03-30T10:03:00")
assets = CryptoXch.portfolio!(xc)
tc = TradingStrategy.train!(TradingStrategy.TradeConfig(xc), assets[!, :coin]; enddt=startdt)
df = DataFrame()
for ohlcv in CryptoXch.ohlcv(xc)
    size(ohlcv.df, 1) > 0 ? push!(df, (base=ohlcv.base, len=length(ohlcv.df[!, :opentime]), startdt=ohlcv.df[begin, :opentime], enddt=ohlcv.df[end, :opentime])) : push!(df, (base=ohlcv.base, len=0, startdt=dummy, enddt=dummy))
end
println("ohlcv data before timecut: $df")
TradingStrategy.timerangecut!(tc, startdt, startdt)
df = DataFrame()
for ohlcv in CryptoXch.ohlcv(xc)
    size(ohlcv.df, 1) > 0 ? push!(df, (base=ohlcv.base, len=length(ohlcv.df[!, :opentime]), startdt=ohlcv.df[begin, :opentime], enddt=ohlcv.df[end, :opentime])) : push!(df, (base=ohlcv.base, len=0, startdt=dummy, enddt=dummy))
end
println("ohlcv data after timecut: $df")
println("trading strategy: tc=$(tc.cfg)")
# startdt = Dates.now(UTC)
# enddt = nothing

end  # module