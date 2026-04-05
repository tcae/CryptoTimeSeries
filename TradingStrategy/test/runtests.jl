module TradingStrategyTest

using Test
using Targets
using TradingStrategy

include("lstm_trade_decider_test.jl")
include("limit_trade_simulation_test.jl")

end # module
