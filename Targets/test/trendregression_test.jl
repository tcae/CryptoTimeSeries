using Test

@testset "TrendRegression interface" begin
    @test_throws AssertionError Targets.TrendRegression(0, 0.02f0, -0.02f0)
    @test_throws AssertionError Targets.TrendRegression(3, -0.02f0, -0.01f0)
    @test_throws AssertionError Targets.TrendRegression(3, 0.04f0, -0.04f0; f6=Features.Features006())

    upohlcv = testohlcvfrompivots(Float32[100.0, 105.0, 110.0, 115.0, 120.0])
    upf6 = Features.Features006()
    Features.addstd!(upf6, window=3, offset=0)
    Features.setbase!(upf6, upohlcv, usecache=false)
    @test Features.issupplementedcurrent(upf6)
    uptrd = Targets.TrendRegression(3, 0.04f0, -0.04f0; f6=upf6)
    Targets.setbase!(uptrd, upf6)

    @test Targets.firstrowix(uptrd) == 1
    @test Targets.lastrowix(uptrd) == 5
    @test Targets.uniquelabels(uptrd) == [Targets.longbuy, Targets.shortbuy, Targets.allclose]
    @test collect(Targets.labels(uptrd)) == [Targets.longbuy, Targets.longbuy, Targets.longbuy, Targets.longbuy, Targets.allclose]
    @test all(Targets.relativegain(uptrd, 1, 4) .> 0.04f0)
    @test Targets.relativegain(uptrd, 5, 5) == [0.0f0]
    @test collect(Targets.labelbinarytargets(uptrd, Targets.longbuy)) == [true, true, true, true, false]
    @test collect(Targets.labelrelativegain(uptrd, Targets.longbuy)) == collect(Targets.relativegain(uptrd)) .* Float32[1, 1, 1, 1, 0]

    downohlcv = testohlcvfrompivots(Float32[120.0, 115.0, 110.0, 105.0, 100.0])
    downf6 = Features.Features006()
    Features.addstd!(downf6, window=3, offset=0)
    Features.setbase!(downf6, downohlcv, usecache=false)
    downtrd = Targets.TrendRegression(3, 0.04f0, -0.04f0; f6=downf6)
    Targets.setbase!(downtrd, downf6)
    @test collect(Targets.labels(downtrd)) == [Targets.shortbuy, Targets.shortbuy, Targets.shortbuy, Targets.shortbuy, Targets.allclose]
    @test all(Targets.relativegain(downtrd, 1, 4) .< -0.04f0)

    piggybackf6 = Features.Features006()
    Features.addstd!(piggybackf6, window=3, offset=0)
    Features.setbase!(piggybackf6, upohlcv, usecache=false)
    requested_before = copy(Features.f6requested(piggybackf6))
    required_before = copy(piggybackf6.required)
    piggybacktrd = Targets.TrendRegression(3, 0.04f0, -0.04f0; f6=piggybackf6)
    Targets.setbase!(piggybacktrd, piggybackf6)
    @test collect(Targets.labels(piggybacktrd)) == collect(Targets.labels(uptrd))
    @test Features.f6requested(piggybackf6) == requested_before
    @test piggybackf6.required == required_before

    tailohlcv = testohlcvfrompivots(Float32[100.0, 102.0, 110.0, 111.0])
    tailf6 = Features.Features006()
    Features.addstd!(tailf6, window=3, offset=0)
    Features.setbase!(tailf6, tailohlcv, usecache=false)
    tailtrd = Targets.TrendRegression(3, 0.005f0, -0.005f0; f6=tailf6)
    Targets.setbase!(tailtrd, tailf6)
    @test tailtrd.df[3, :label] == Targets.longbuy

    extendedohlcv = testohlcvfrompivots(Float32[100.0, 102.0, 110.0, 111.0, 70.0])
    Ohlcv.setdataframe!(tailohlcv, Ohlcv.dataframe(extendedohlcv))
    @test !Features.issupplementedcurrent(tailf6)
    @test_throws AssertionError Targets.supplement!(tailtrd)
    Features.supplement!(tailf6)
    @test Features.issupplementedcurrent(tailf6)
    Targets.supplement!(tailtrd)

    fulltailf6 = Features.Features006()
    Features.addstd!(fulltailf6, window=3, offset=0)
    Features.setbase!(fulltailf6, extendedohlcv, usecache=false)
    fulltailtrd = Targets.TrendRegression(3, 0.005f0, -0.005f0; f6=fulltailf6)
    Targets.setbase!(fulltailtrd, fulltailf6)

    @test Targets.lastrowix(tailtrd) == 5
    @test tailtrd.df[3, :label] == Targets.shortbuy
    @test collect(tailtrd.df[!, :opentime]) == collect(fulltailtrd.df[!, :opentime])
    @test collect(tailtrd.df[!, :label]) == collect(fulltailtrd.df[!, :label])
    @test tailtrd.df[1, :label] == Targets.longbuy
    @test tailtrd.df[!, :relgain] ≈ fulltailtrd.df[!, :relgain]

    emptyf6 = Features.Features006()
    @test_throws AssertionError Targets.setbase!(Targets.TrendRegression(3, 0.04f0, -0.04f0), emptyf6)
    @test_throws ArgumentError Targets.setbase!(Targets.TrendRegression(3, 0.04f0, -0.04f0), upohlcv)

    badf6 = Features.Features006()
    Features.addmindist!(badf6, window=3, offset=0)
    Features.setbase!(badf6, upohlcv, usecache=false)
    @test_throws AssertionError Targets.setbase!(Targets.TrendRegression(3, 0.04f0, -0.04f0), badf6)
end