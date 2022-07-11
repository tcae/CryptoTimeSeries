using Pkg


module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade, Classify

# EnvConfig.init(test)
# backtestchunk = 100; enddt = DateTime("2022-03-28T10:00:00"); period = Dates.Day(1)  # short backtest
# # @info "\n\ntraderules002 bestgain, selected and 1h grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down and dependent on regr gradient, std instead of medianstd; backtest chunk=$backtestchunk period=$period enddt=$enddt" Classify.tr001default
# @info "\n\ntraderules002 bestgain, selected and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down and dependent on regr gradient, std instead of medianstd; backtest chunk=$backtestchunk period=$period enddt=$enddt" Classify.tr001default
# cache = Trade.Cache(backtestchunk, period, enddt, ["sine"])
# @time Trade.tradeloop(cache)
# @info "traderules002 bestgain, selected and 1h grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down and dependent on regr gradient, std instead of medianstd; backtest chunk=$backtestchunk period=$period enddt=$enddt" Classify.tr001default



# EnvConfig.init(production)
EnvConfig.init(training)
# println(EnvConfig.trainingbases)
# Threads.@threads for base in ["eos", "xrp"]

backtestchunk = 100; enddt = DateTime("2022-04-02T01:00:00"); period = Dates.Month(6)  # fix enddt and backtestperiod to get reproducible results for backtest
# backtestchunk = 0; enddt = floor(Dates.now(Dates.UTC), Dates.Minute); period = Dates.Minute(1)  # live production

# infostr = "traderules001 bestgain, selected and 1h grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down and dependent on regr gradient, std instead of medianstd; backtest chunk=$backtestchunk period=$period enddt=$enddt"
infostr = "traderules002 bestgain, current grad > regr specific minimum, sell when regr grad <= 0; backtest chunk=$backtestchunk period=$period enddt=$enddt"
for base in ["btc"]
    @info infostr Classify.tr002default
    # @info "\n\ntraderules001 bestgain, selected and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down and dependent on regr gradient, std instead of medianstd; backtest chunk=$backtestchunk period=$period enddt=$enddt" Classify.tr001default
    cache = Trade.Cache(backtestchunk, period, enddt, [base])
    @time Trade.tradeloop(cache)
    @info infostr Classify.tr002default
end



# tradecaches = Trade.preparetradecache(false)
# for (key, tc) in tradecaches
#     println("tradecache base=$key")
#     print(tc.features)
# end
# @testset "Trade tests" begin

# @test true

# end

end  # module