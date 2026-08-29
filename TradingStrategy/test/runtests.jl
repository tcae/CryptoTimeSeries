module TradingStrategyTest

using Test
using DataFrames
using Targets
using TSM
using TradingStrategy

"""Resolve Trades column handles; the strategy API operates on handles, not on the frame."""
tcols(df::DataFrame) = TSM.TradesColumns(df)

include("gain_limit_reversal_direction_test.jl")
include("rowtakeover_carry_test.jl")
include("runtime_api_test.jl")
include("replay_input_aliasing_test.jl")
include("trade_storage_test.jl")
include("tradesdf_limit_reversal_test.jl")

end # module
