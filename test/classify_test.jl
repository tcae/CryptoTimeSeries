
using DrWatson
@quickactivate "CryptoTimeSeries"

include(srcdir("classify.jl"))

module ClassifyTest
using Dates, DataFrames
using Test

using ..Classify

@testset "Classify tests" begin

# @test true

end

end  # module