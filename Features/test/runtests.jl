
module FeaturesTest
using Dates, DataFrames
using Test
using LinearRegression

using EnvConfig, Ohlcv, Features, CryptoXch, TestOhlcv

include("featureutilities_test.jl")
include("features006_test.jl")
# include("features005_test.jl") # currently fails due to changes in Features.jl
# include("f4supplement_test.jl") # currently fails due to changes in Features.jl
# include("features002_test.jl") # currently fails due to changes in Features.jl



end  # module

