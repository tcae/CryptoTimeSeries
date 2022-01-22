# include("../src/features.jl")

module Classify

import Pkg; Pkg.add(["JDF", "RollingFunctions", "MLJ"])
using JDF, Dates, CSV, DataFrames  # , MLJ
using ..Features


end  # module
