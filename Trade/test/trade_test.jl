using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade, Classify

EnvConfig.init(test)
# EnvConfig.init(production)
# EnvConfig.init(training)
# println(EnvConfig.trainingbases)
# Threads.@threads for base in ["eos", "xrp"]
for base in ["btc"]
    @info "\n\ntraderules001 bestgain, selected and 1h grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
    # cache = Trade.Cache(100, Dates.Day(1), DateTime("2022-03-28T10:00:00"), [base])  # fix enddt and backtestperiod to get reproducible results for backtest
    cache = Trade.Cache(100, Dates.Day(1), DateTime("2022-03-28T10:00:00"), nothing)  # fix enddt and backtestperiod to get reproducible results for backtest
    @time Trade.tradeloop(cache)
    @info "traderules001 bestgain, selected and 1h grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
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