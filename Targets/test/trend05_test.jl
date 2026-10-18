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

@testset "Trend05 accepts offset-backed regression features" begin
    ohlcv = testohlcvfrompivots(Float32.(100:1:30))
    f6 = Features.Features006()
    for window in (2, 3, 4)
        Features.addregry!(f6, window=window, offset=1)
        Features.addgrad!(f6, window=window, offset=1)
    end
    @test all(feature -> feature.o == 1, Features.f6all(f6))
    Features.setbase!(f6, ohlcv, usecache=false)
    Features.supplement!(f6)

    trd = Targets.Trend05(leadregr=2, supportregr=3, trendregr=4)
    required = Targets._trend05_requiredfeatures(f6, trd)
    @test required == [
        Features._grad(f6, window=2, offset=0), Features._regry(f6, window=2, offset=0),
        Features._grad(f6, window=3, offset=0), Features._regry(f6, window=3, offset=0),
        Features._grad(f6, window=4, offset=0), Features._regry(f6, window=4, offset=0),
    ]
end

@testset "Trend05 requires aligned feature timestamps" begin
    firstmismatch = trend05withfeatures(testohlcvfrompivots(Float32.(100:1:160)))
    firstmismatch.f6.fdfno = copy(firstmismatch.f6.fdfno)
    firstmismatch.f6.fdfno[begin, :opentime] += Minute(1)
    @test_throws AssertionError Targets.supplement!(firstmismatch)

    lastmismatch = trend05withfeatures(testohlcvfrompivots(Float32.(100:1:160)))
    lastmismatch.f6.fdfno = copy(lastmismatch.f6.fdfno)
    lastmismatch.f6.fdfno[end, :opentime] += Minute(1)
    @test_throws AssertionError Targets.supplement!(lastmismatch)
end

@testset "Trend05 entry prerequisites" begin
    trd = Targets.Trend05(triggerdist=0.01f0, targetgain=0.02f0)
    support = (regry=100f0, grad=0f0)

    @test Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=1f0), :long)
    @test Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), :long)
    @test Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=0f0), (regry=100f0, grad=1f0), (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=1f0), support, (regry=100f0, grad=0f0), :long)
    @test Targets._trend05_entry(trd, 101f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test Targets._trend05_entry(trd, 102f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), :short)
    @test Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 99f0, (regry=100f0, grad=0f0), (regry=100f0, grad=-1f0), (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 98f0, (regry=100f0, grad=-1f0), support, (regry=100f0, grad=-1f0), :short)
    @test !Targets._trend05_entry(trd, 100f0, nothing, support, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=1f0), nothing, (regry=100f0, grad=1f0), :long)
    @test !Targets._trend05_entry(trd, 100f0, (regry=100f0, grad=1f0), support, nothing, :long)
end

@testset "Trend05 entry requires uninterrupted holds through target" begin
    rising = trend05withfeatures(testohlcvfrompivots(Float32.(100:1:160)))
    risingcolumns = Targets._trend05_regressioncolumns(rising)
    @test Targets.labels(rising)[2] == Targets.longopen
    rising.f6.fdfno[4, risingcolumns.support.grad] = 0f0
    Targets.supplement!(rising)
    @test Targets.labels(rising)[2] != Targets.longopen

    falling = trend05withfeatures(testohlcvfrompivots(Float32.(160:-1:100)))
    fallingcolumns = Targets._trend05_regressioncolumns(falling)
    @test Targets.labels(falling)[2] == Targets.shortopen
    falling.f6.fdfno[4, fallingcolumns.support.grad] = 0f0
    Targets.supplement!(falling)
    @test Targets.labels(falling)[2] != Targets.shortopen
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

@testset "Trend05 requires current supplemented features" begin
    ohlcv = testohlcvfrompivots(Float32.(100:1:110))
    trd = trend05withfeatures(ohlcv)
    previous = copy(trd.df)
    extended = testohlcvfrompivots(Float32.(100:1:130))
    Ohlcv.setdataframe!(ohlcv, Ohlcv.dataframe(extended))
    @test !Features.issupplementedcurrent(trd.f6)
    @test_throws r"requires Features.supplement!\(f6\)" Targets.supplement!(trd)
    @test trd.df == previous
    Features.supplement!(trd.f6)
    @test Features.issupplementedcurrent(trd.f6)
    @test isnothing(Targets.supplement!(trd))
    @test nrow(trd.df) == 31
    trd.f6.fdf = nothing
    @test_throws r"requires Features.supplement!\(f6\)" Targets.supplement!(trd)
end

@testset "Trend05 appended targets equal full rebuilds" begin
    for pivots in (Float32.(100:1:180), Float32.(180:-1:100),
        vcat(Float32.(100:1:140), Float32.(139:-1:100), Float32.(101:1:140)))
        ohlcv = testohlcvfrompivots(pivots[1:11])
        trd = trend05withfeatures(ohlcv)
        targetdf = trd.df
        for rows in (12, 13, 20, 21, 40, length(pivots))
            finalix = trd.lastfinalix
            prefix = copy(trd.df[1:finalix, :])
            extended = testohlcvfrompivots(pivots[1:rows])
            Ohlcv.setdataframe!(ohlcv, Ohlcv.dataframe(extended))
            Features.supplement!(trd.f6)
            Targets.supplement!(trd)
            rebuilt = trend05withfeatures(extended)
            @test trd.df === targetdf
            @test trd.df[1:finalix, :] == prefix
            @test trd.df == rebuilt.df
            @test trd.lastfinalix == rebuilt.lastfinalix
        end
        replacement = trend05withfeatures(testohlcvfrompivots(reverse(pivots)))
        Targets.setbase!(trd, replacement.f6)
        @test trd.df == replacement.df
        @test trd.lastfinalix == replacement.lastfinalix
    end
end

@testset "Trend05 append resumes before provisional open signals" begin
    for (pivots, gradient, openlabel) in (
        (Float32[108, 100, 101, 106, 108, 114], 1f0, Targets.longopen),
        (Float32[108, 116, 115, 110, 108, 102], -1f0, Targets.shortopen))
        ohlcv = testohlcvfrompivots(pivots[1:5])
        trd = trend05withfeatures(ohlcv)
        extended = testohlcvfrompivots(pivots)
        rebuilt = trend05withfeatures(extended)
        for target in (trd, rebuilt)
            columns = Targets._trend05_regressioncolumns(target)
            for regression in values(columns)
                target.f6.fdfno[!, regression.grad] .= gradient
                target.f6.fdfno[!, regression.regry] .= 100f0
            end
            Targets.supplement!(target)
        end
        @test trd.lastfinalix == 0
        @test Targets.labels(trd)[1] == Targets.allclose
        @test Targets.labels(trd)[2] == openlabel
        @test Targets.labels(rebuilt)[1] == openlabel
        targetdf = trd.df
        Ohlcv.setdataframe!(ohlcv, Ohlcv.dataframe(extended))
        Features.supplement!(trd.f6)
        columns = Targets._trend05_regressioncolumns(trd)
        for regression in values(columns)
            trd.f6.fdfno[!, regression.grad] .= gradient
            trd.f6.fdfno[!, regression.regry] .= 100f0
        end
        Targets.supplement!(trd)
        @test trd.df === targetdf
        @test trd.df == rebuilt.df
        @test trd.lastfinalix == rebuilt.lastfinalix
    end
end

@testset "Trend05 final label rows" begin
    reached = trend05withfeatures(testohlcvfrompivots(Float32.(100:1:160)))
    @test reached.lastfinalix < 61
    @test Targets.labels(reached)[2] == Targets.longopen
    @test reached.lastfinalix >= 7

    # rising without reaching the target gain leaves the first entry decision open
    unresolved = trend05withfeatures(testohlcvfrompivots(Float32.(100:1:103)); targetgain=0.05f0)
    @test unresolved.lastfinalix < 4
    Targets.removebase!(unresolved)
    @test unresolved.lastfinalix == 0
end
