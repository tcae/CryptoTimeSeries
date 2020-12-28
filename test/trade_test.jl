using DrWatson
@quickactivate "CryptoTimeSeries"


include(srcdir("trade.jl"))

module TradeTest

using Test
using ..Trade

@testset "Trade tests" begin

@test true

end

end  # module