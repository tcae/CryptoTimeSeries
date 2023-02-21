using Pkg


module TradeTest

using Test, Dates
using EnvConfig
using Trade

# EnvConfig.init(test)
# backtestchunk = 100; enddt = DateTime("2022-03-28T10:00:00"); period = Dates.Day(1)  # short backtest
# cache = Trade.Cache(backtestchunk, period, enddt, ["sine"])
# @time Trade.tradeloop(cache)



# EnvConfig.init(production)
EnvConfig.init(training)
# println(EnvConfig.trainingbases)
# Threads.@threads for base in ["eos", "xrp"]

# backtestchunk = 100; enddt = DateTime("2022-04-02T01:00:00"); period = Dates.Month(6)  # fix enddt and backtestperiod to get reproducible results for backtest
# backtestchunk = 0; enddt = floor(Dates.now(Dates.UTC), Dates.Minute); period = Dates.Minute(1)  # live production
# backtestchunk = 100; enddt = floor(Dates.now(Dates.UTC), Dates.Minute); period = Dates.Month(1)
backtestchunk = 100; enddt = DateTime("2022-08-05T20:00:00"); period = Dates.Month(3)  # fix enddt and backtestperiod to get reproducible results for backtest

for base in ["btc"]  # , "atom", "doge", "sol", "mana", "near", "axs", "matic", "link", "waves", "ada", "bnb", "eos", "vet"]
    cache = Trade.Cache(backtestchunk, period, enddt, [base])
    Trade.tradeloop(cache)
    # @time Trade.tradeloop(cache)
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