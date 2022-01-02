include("../src/features.jl")
# include(srcdir("features.jl"))

module Classify

import Pkg; Pkg.add(["JDF", "RollingFunctions", "MLJ"])
using JDF, Dates, CSV, DataFrames, MLJ
using ..Features


end  # module
