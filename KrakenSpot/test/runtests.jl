using KrakenSpot, Test

@testset "KrakenSpot tests" begin
    include("KrakenSpot_test.jl")
    include("KrakenSpot_online_test.jl")
end
