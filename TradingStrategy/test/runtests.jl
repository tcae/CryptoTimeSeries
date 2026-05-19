module TradingStrategyTest

using Test
using Targets
using TradingStrategy

include("lstm_trade_decider_test.jl")
include("limit_trade_simulation_test.jl")
include("gain_limit_reversal_direction_test.jl")
include("trade_storage_test.jl")

end # module
