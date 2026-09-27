using IbkrSpot, Test

@testset "IbkrSpot tests" begin
    include("IbkrSpot_test.jl")
    include("IbkrSpot_online_test.jl")
    include("IbkrSpot_order_test.jl")
end
