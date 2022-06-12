module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade

# EnvConfig.init(test)
EnvConfig.init(training)
println(EnvConfig.trainingbases)
@time Trade.tradeloop(100)
# tradecaches = Trade.preparetradecache(false)
# for (key, tc) in tradecaches
#     println("tradecache base=$key")
#     print(tc.features)
# end
# @testset "Trade tests" begin

# @test true

# end

end  # module