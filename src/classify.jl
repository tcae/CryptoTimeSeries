using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/features.jl")
# include(srcdir("features.jl"))

module Classify

import Pkg; Pkg.add(["JDF", "RollingFunctions"])
using JDF, Dates, CSV, DataFrames
using ..Features


end  # module
