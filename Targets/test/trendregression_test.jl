using Test

@testset "TrendRegression interface" begin
    @test_throws AssertionError Targets.TrendRegression(0, 0.02f0, -0.02f0)
    @test_throws AssertionError Targets.TrendRegression(3, -0.02f0, -0.01f0)

    upohlcv = testohlcvfrompivots(Float32[100.0, 105.0, 110.0, 115.0, 120.0])
    upf6 = Features.Features006()
    Features.addgrad!(upf6, window=3, offset=0)
    Features.setbase!(upf6, upohlcv, usecache=false)

    uptrd = Targets.TrendRegression(3, 0.04f0, -0.04f0)
    Targets.setbase!(uptrd, upf6)

    @test Targets.firstrowix(uptrd) == 1
    @test Targets.lastrowix(uptrd) == 5
    @test Targets.uniquelabels(uptrd) == [Targets.longhold, Targets.shorthold, Targets.allclose]
    @test collect(Targets.labels(uptrd)) == [Targets.longhold, Targets.longhold, Targets.longhold, Targets.longhold, Targets.allclose]
    @test all(Targets.relativegain(uptrd, 1, 4) .> 0.04f0)
    @test Targets.relativegain(uptrd, 5, 5) == [0.0f0]
    @test collect(Targets.labelbinarytargets(uptrd, Targets.longhold)) == [true, true, true, true, false]
    @test collect(Targets.labelrelativegain(uptrd, Targets.longhold)) == collect(Targets.relativegain(uptrd)) .* Float32[1, 1, 1, 1, 0]

    downohlcv = testohlcvfrompivots(Float32[120.0, 115.0, 110.0, 105.0, 100.0])
    downf6 = Features.Features006()
    Features.addgrad!(downf6, window=3, offset=0)
    Features.setbase!(downf6, downohlcv, usecache=false)

    downtrd = Targets.TrendRegression(3, 0.04f0, -0.04f0)
    Targets.setbase!(downtrd, downf6)
    @test collect(Targets.labels(downtrd)) == [Targets.shorthold, Targets.shorthold, Targets.shorthold, Targets.shorthold, Targets.allclose]
    @test all(Targets.relativegain(downtrd, 1, 4) .< -0.04f0)

    badf6 = Features.Features006()
    Features.addregry!(badf6, window=3, offset=0)
    Features.setbase!(badf6, upohlcv, usecache=false)
    @test_throws AssertionError Targets.setbase!(Targets.TrendRegression(3, 0.04f0, -0.04f0), badf6)
end