module TradeTest

using Test, Dates
using EnvConfig, Ohlcv
using Trade

EnvConfig.init(test)
println(EnvConfig.trainingbases)
tc = Trade.preparetradecache(false)
println(tc)
@testset "Trade tests" begin

@test true

end

end  # module