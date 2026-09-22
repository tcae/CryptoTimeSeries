using Dates
using Test

function trend05withfeatures(ohlcv; leadregr::Int=2, supportregr::Int=3, trendregr::Int=4, triggerdist::Float32=0.02f0, targetgain::Float32=0.05f0)
    trd = Targets.Trend05(; leadregr, supportregr, trendregr, triggerdist, targetgain)
    f6 = Features.Features006()
    for window in (leadregr, supportregr, trendregr, 60)
        Features.addregry!(f6, window=window, offset=0)
        Features.addgrad!(f6, window=window, offset=0)
    end
    Features.setbase!(f6, ohlcv, usecache=false)
    Targets.setbase!(trd, f6)
    return trd
end

@testset "Trend05 interface and directional labels" begin
    rising = testohlcvfrompivots(Float32.(100:1:160))
    emptytrd = Targets.Trend05()
    @test Targets.firstrowix(emptytrd) == 1
    @test Targets.lastrowix(emptytrd) == 0
    @test isnothing(emptytrd.ohlcv)
    @test isnothing(emptytrd.df)
    trd = trend05withfeatures(rising)

    @test trd.targetgain == 0.05f0

    labels = collect(Targets.labels(trd))
    gains = collect(Targets.relativegain(trd))

    @test length(labels) == 61
    @test length(gains) == 61
    @test Targets.firstrowix(trd) == 1
    @test Targets.lastrowix(trd) == 61
    @test labels[1] == Targets.allclose
    @test labels[2] == Targets.longopen
    @test all(==(Targets.longhold), labels[3:7])
    @test gains[2] > 0.05f0
    @test gains[3] > 0f0
    @test all(label -> label in Targets.uniquelabels(trd), labels)

    df = Targets.df(trd)
    @test Tuple(propertynames(df)) == (:opentime, :label, :relgain)
    @test nrow(Targets.labelvalues(trd)) == 61
    @test collect(Targets.labelbinarytargets(trd, Targets.longopen)) == (labels .== Targets.longopen)
    @test collect(Targets.labelrelativegain(trd, Targets.longopen)) == gains .* (labels .== Targets.longopen)

    dates = Ohlcv.dataframe(rising)[!, :opentime]
    @test collect(Targets.labels(trd, dates[2], dates[7])) == labels[2:7]
    @test occursin("leadregr=2", Targets.describe(trd))
    @test occursin("targetgain=0.05", Targets.describe(trd))

    Targets.removebase!(trd)
    @test isnothing(trd.ohlcv)
    @test isnothing(trd.f6)
    @test isnothing(trd.df)
end

@testset "Trend05 short labels" begin
    falling = testohlcvfrompivots(Float32.(160:-1:100))
    trd = trend05withfeatures(falling)
    labels = collect(Targets.labels(trd))
    gains = collect(Targets.relativegain(trd))

    @test labels[1] == Targets.allclose
    @test labels[2] == Targets.shortopen
    @test all(==(Targets.shorthold), labels[3:10])
    @test gains[2] < -0.05f0
    @test all(label -> label in Targets.uniquelabels(trd), labels)
end

@testset "Trend05 invalid configuration" begin
    @test_throws AssertionError Targets.Trend05(leadregr=0)
    @test_throws AssertionError Targets.Trend05(supportregr=0)
    @test_throws AssertionError Targets.Trend05(trendregr=0)
    @test_throws AssertionError Targets.Trend05(triggerdist=-0.01f0)
    @test_throws AssertionError Targets.Trend05(targetgain=0f0)
end

@testset "Trend05 requires a feature superset" begin
    ohlcv = testohlcvfrompivots(Float32.(100:1:30))
    f6 = Features.Features006()
    for window in (2, 3)
        Features.addregry!(f6, window=window, offset=0)
        Features.addgrad!(f6, window=window, offset=0)
    end
    Features.setbase!(f6, ohlcv, usecache=false)
    trd = Targets.Trend05(leadregr=2, supportregr=3, trendregr=4)
    @test_throws AssertionError Targets.setbase!(trd, f6)
    @test_throws ArgumentError Targets.setbase!(trd, ohlcv)
end

@testset "Trend05 entry prerequisites" begin
    trd = Targets.Trend05(triggerdist=0.01f0, targetgain=0.02f0)
    support = (regry=100f0, grad=0f0)

    @test Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=1f0), :long)
    @test Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=0f0), (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=0f0), :long)
    @test Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=0f0), (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 98f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 100f0, nothing, support, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=1f0), nothing, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=1f0), support, nothing, :long)
end

@testset "Trend05 hold prerequisites" begin
    @test Targets._trend05_hold((regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_hold((regry=100f0, grad=0f0), :long)
    @test !Targets._trend05_hold((regry=100f0, grad=-1f0), :long)
    @test Targets._trend05_hold((regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_hold((regry=100f0, grad=0f0), :short)
    @test !Targets._trend05_hold((regry=100f0, grad=1f0), :short)
    @test !Targets._trend05_hold(nothing, :long)
    @test !Targets._trend05_hold(nothing, :short)
end

@testset "Trend05 target reachability" begin
    pivots = Float32[100, 101, 102, 103, 104, 105]
    @test Targets._trend05_target(pivots, 1, :long, 0.05f0) == 6
    @test Targets._trend05_target(pivots, 2, :long, 0.05f0) === nothing
    shortpivots = Float32[106, 105, 104, 103, 102, 101, 100]
    @test Targets._trend05_target(shortpivots, 1, :short, 0.05f0) == 7
    @test Targets._trend05_target(shortpivots, 2, :short, 0.05f0) === nothing

    shortohlcv = testohlcvfrompivots(Float32[100, 101, 102, 103])
    shorttrend = trend05withfeatures(shortohlcv; targetgain=0.05f0)
    @test all(==(Targets.allclose), Targets.labels(shorttrend))
end

@testset "Trend05 regression warmup and supplementation" begin
    shortohlcv = testohlcvfrompivots(Float32.(100:1:110))
    trd = trend05withfeatures(shortohlcv; targetgain=0.05f0)
    oldrows = nrow(trd.df)

    extended = testohlcvfrompivots(Float32.(100:1:130))
    Ohlcv.setdataframe!(shortohlcv, Ohlcv.dataframe(extended))
    Features.supplement!(trd.f6)
    Targets.supplement!(trd)
    @test nrow(trd.df) == 31
    @test nrow(trd.df) > oldrows
    @test length(Targets.labels(trd)) == 31

    tiny = trend05withfeatures(testohlcvfrompivots(Float32[100, 101]); targetgain=0.05f0)
    @test all(==(Targets.allclose), Targets.labels(tiny))
end
