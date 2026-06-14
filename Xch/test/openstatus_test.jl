using Test
using Xch

@testset "openstatus includes exchange Open state" begin
    @test Xch.openstatus("Open")
    @test Xch.openstatus("New")
    @test Xch.openstatus("PartiallyFilled")
    @test Xch.openstatus("Untriggered")
    @test !Xch.openstatus("Filled")
    @test !Xch.openstatus("Cancelled")
    @test !Xch.openstatus("Rejected")
end
