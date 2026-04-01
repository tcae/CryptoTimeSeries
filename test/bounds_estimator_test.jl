module BoundsEstimatorTests

using Test
using DataFrames

include("../scripts/BoundsEstimator.jl")

@testset "BoundsEstimator ratio targets" begin
    phot = DataFrame(
        high=Float32[110, 220],
        low=Float32[90, 180],
        pivot=Float32[100, 200],
    )

    y = BoundsEstimator.calcboundstargets!(phot)

    @test size(y) == (2, 2)
    @test y[1, :] == Float32[0.0, 0.0]
    @test y[2, :] == Float32[0.2, 0.2]
end

@testset "BoundsEstimator predicted band reconstruction" begin
    pred_center = Float32[0.0, 0.01, -0.02]
    pred_width = Float32[0.2, 0.1, 0.05]
    pivot = Float32[100, 200, 50]

    lower, upper = BoundsEstimator.denormalize_predicted_bounds(pred_center, pred_width, pivot)

    @test lower ≈ Float32[90.0, 192.0, 47.75]
    @test upper ≈ Float32[110.0, 212.0, 50.25]
end

@testset "BoundsEstimator rejects non-positive pivot" begin
    @test_throws AssertionError BoundsEstimator.denormalize_predicted_bounds(Float32[0.0], Float32[0.1], Float32[0.0])
end

end # module
