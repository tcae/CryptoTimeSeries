using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade, Classify

# EnvConfig.init(test)
# EnvConfig.init(production)
EnvConfig.init(training)
println(EnvConfig.trainingbases)
@info "traderules001 bestgain, selected and 15min grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
@time Trade.tradeloop(100)
@info "traderules001 bestgain, selected and 15min grad > min and 24h grad > min gain and current grad > window/4 pastgrad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
# tradecaches = Trade.preparetradecache(false)
# for (key, tc) in tradecaches
#     println("tradecache base=$key")
#     print(tc.features)
# end
# @testset "Trade tests" begin

# @test true

# end

end  # module