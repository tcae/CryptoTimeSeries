include(srcdir("env_config.jl"))
include(srcdir("ohlcv.jl"))
include(srcdir("trade.jl"))

module TradeTest

using Test, Dates
using ..EnvConfig, ..Ohlcv
using ..Trade

Config.init(production)
Trade.gettrainingohlcv(["btc"])

@testset "Trade tests" begin

@test true

end

end  # module