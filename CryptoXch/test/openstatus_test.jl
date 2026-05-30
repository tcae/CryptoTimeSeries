using Test
using CryptoXch

@testset "openstatus includes exchange Open state" begin
    @test CryptoXch.openstatus("Open")
    @test CryptoXch.openstatus("New")
    @test CryptoXch.openstatus("PartiallyFilled")
    @test CryptoXch.openstatus("Untriggered")
    @test !CryptoXch.openstatus("Filled")
    @test !CryptoXch.openstatus("Cancelled")
    @test !CryptoXch.openstatus("Rejected")
end
