module TradeTest

using Test, Dates
using EnvConfig, Ohlcv, Features
using Trade

EnvConfig.init(test)
println(EnvConfig.trainingbases)
Trade.tradeloop(true)
# tradecaches = Trade.preparetradecache(false)
# for (key, tc) in tradecaches
#     println("tradecache base=$key")
#     print(tc.features)
# end
# @testset "Trade tests" begin

# @test true

# end

end  # module