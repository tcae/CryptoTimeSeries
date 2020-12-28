using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/env_config.jl")
include("../src/classify.jl")
# include(srcdir("classify.jl"))

module Trade

using Dates, DataFrames
using ..Config, ..Classify



end  # module