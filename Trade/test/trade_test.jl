module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade, Classify

# EnvConfig.init(test)
EnvConfig.init(training)
println(EnvConfig.trainingbases)
@info "bestgain, 24h grad positiv, 4hgrad > 2 * min grad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
@time Trade.tradeloop(100)
@info "bestgain, 24h grad positiv, 4hgrad > 2 * min grad, sellprice adaptation up & down, std instead of medianstd" Classify.tr001default
# tradecaches = Trade.preparetradecache(false)
# for (key, tc) in tradecaches
#     println("tradecache base=$key")
#     print(tc.features)
# end
# @testset "Trade tests" begin

# @test true

# end

end  # module