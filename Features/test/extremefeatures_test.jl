using Test, Statistics, Dates, DataFrames
using Features, EnvConfig, Ohlcv, TestOhlcv

@testset "contiguous minute sequences" begin
    time = DateTime(2025)
    @test Features.minutesequences(DateTime[]) == UnitRange{Int}[]
    @test Features.minutesequences([time]) == [1:1]
    @test Features.minutesequences(time .+ Minute.([0,1,81,82,100])) == [1:2,3:4,5:5]
    for times in ([time,time], [time,time-Minute(1)], [time,time+Second(90)])
        @test_throws AssertionError Features.minutesequences(times)
    end
end

"""Independent centered least-squares reference on the represented input prices."""
function centeredreference(prices::AbstractVector)
    count = length(prices)
    meanprice = sum(price -> price + 0.0, prices) / count
    center = (count + 1) / 2
    covariance = sum(ix -> (ix - center) * (prices[ix] - meanprice), eachindex(prices))
    slope = count == 1 ? 0.0 : covariance / sum(ix -> (ix - center)^2, eachindex(prices))
    return (; regry=meanprice + slope * (count - 1) / 2, grad=slope, meanprice)
end

"""Independent sequential reference for threshold-qualified, reestablishing anchors."""
function referenceanchors(prices::AbstractVector, bandratio::Real)
    minimumix = maximumix = 1
    active = :seed
    for ix in 2:length(prices)
        price = prices[ix]
        if active == :seed
            if price < prices[minimumix]
                minimumix = ix
            end
            if price > prices[maximumix]
                maximumix = ix
            end
            if (price - prices[minimumix]) / prices[minimumix] > bandratio
                active = :maximum
            elseif (prices[maximumix] - price) / prices[maximumix] > bandratio
                active = :minimum
            end
        elseif active == :minimum
            if price < prices[minimumix]
                minimumix = ix
            else
                if (price - prices[minimumix]) / prices[minimumix] > bandratio
                    active = :maximum
                    maximumix = ix
                end
            end
        else
            if price > prices[maximumix]
                maximumix = ix
            else
                if (prices[maximumix] - price) / prices[maximumix] > bandratio
                    active = :minimum
                    minimumix = ix
                end
            end
        end
    end
    return minimumix, maximumix
end

@testset "centered regression statistics" begin
    empty = Features.RegressionMoments()
    @test empty.count == 0
    @test_throws AssertionError Features.regressionresult(empty)
    single = Features.appendprice(empty, 100f0)
    @test Features.regressionresult(single) == (regry=100.0, grad=0.0)
    @test Features.removeprice(single, 100f0) == empty
    @test Features.mergeregression(empty, single) == single
    @test Features.mergeregression(single, empty) == single
    prices = [100.0, 101.0, 98.0, 104.0, 102.0]
    whole = Features.regressionmoments(prices, 1, 5)
    merged = Features.mergeregression(Features.regressionmoments(prices, 1, 2),
        Features.regressionmoments(prices, 3, 5))
    @test merged.mean ≈ whole.mean
    @test merged.covariance ≈ whole.covariance
    removed = Features.removeprice(whole, prices[1])
    reference = centeredreference(prices[2:end])
    @test Features.regressionresult(removed).grad ≈ reference.grad
    @test Features.regressionresult(removed).regry ≈ reference.regry
end

@testset "confirmed extreme event metadata" begin
    reversal = [110.0, 100, 102, 105, 104.8, 103.9, 104.2, 105.1, 104.9, 104.0]
    @test Features.confirmedextremes(reversal; bandratio=0.01) ==
        [(index=2,direction=-1,confirmedix=3),
         (index=4,direction=1,confirmedix=6),
         (index=6,direction=-1,confirmedix=8),
         (index=8,direction=1,confirmedix=10)]
    for prices in (reversal, 220 .- reversal)
        events = Features.confirmedextremes(prices; bandratio=0.01, sequenceended=true)
        @test all(ix -> events[ix-1].index < events[ix].index, 2:length(events))
        tree = Features.ExtremeSummaryTree(prices)
        state = Features.ExtremeState(1,1,0)
        Features.replayextremes!(state,tree,0.01,2,length(prices))
        @test (state.minimumix,state.maximumix) ==
            referenceanchors(prices,0.01)
    end
    prices = [128.0, 100, 105, 110, 114, 125, 124, 108, 103, 100, 99, 114]
    @test Features.confirmedextremes(prices; bandratio=0.125) ==
        [(index=2, direction=-1, confirmedix=5),
         (index=6, direction=1, confirmedix=8),
         (index=11, direction=-1, confirmedix=12)]
    @test Features.confirmedextremes(prices[1:7]; bandratio=0.125) ==
        [(index=2, direction=-1, confirmedix=5)]
    @test Features.confirmedextremes(prices[1:7]; bandratio=0.125, sequenceended=true) ==
        [(index=2, direction=-1, confirmedix=5), (index=6, direction=1, confirmedix=7)]
    @test Features.confirmedextremes(prices[1:11]; bandratio=0.125, sequenceended=true)[end] ==
        (index=11, direction=-1, confirmedix=11)
    @test isempty(Features.confirmedextremes([128.0, 100, 105, 110]; bandratio=0.125, sequenceended=true))
    @test isempty(Features.confirmedextremes(fill(100.0, 20); bandratio=0.01))
    @test Features.confirmedextremes([110.0, 100, 100.5, 99.8, 101]; bandratio=0.01) ==
        [(index=4, direction=-1, confirmedix=5)]
    for prices in (Float64[], [0.0], [NaN], [-1.0], [Inf])
        @test_throws AssertionError Features.confirmedextremes(prices; bandratio=0.01)
    end
    @test_throws AssertionError Features.confirmedextremes([100.0]; bandratio=0.0)
end

@testset "stable rolling regression accuracy and alignment" begin
    for scale in (0.0001, 100.0, 1e6), sign in (-1, 0, 1)
        prices = Float32[scale * (1 + sign * 1e-5 * (ix - 1) / 14399) for ix in 1:14400]
        endpoint, slope = Features.rollingregression(prices, 14400)
        reference = centeredreference(prices)
        @test eltype(slope) == Float32
        @test isapprox(slope[end], reference.grad; rtol=2e-6, atol=1e-20)
        @test isapprox(endpoint[end], reference.regry; rtol=2e-7)
        @test all(isfinite, slope)
        @test all(isfinite, endpoint)
    end
    prices = [100.0 + 0.001sin(ix) for ix in 1:200]
    for window in (2, 7, 60, 300), startix in (1, 3, 75, 200)
        endpoint, slope = Features.rollingregression(prices, window, startix)
        @test length(endpoint) == length(slope) == length(prices) - startix + 1
        for ix in startix:length(prices)
            reference = centeredreference(prices[max(1, ix - window + 1):ix])
            @test isapprox(slope[ix - startix + 1], reference.grad; rtol=1e-7, atol=1e-12)
            @test isapprox(endpoint[ix - startix + 1], reference.regry; rtol=1e-12)
        end
    end
    endpoint, slope = Features.rollingregression!(nothing, nothing, prices[1:80], 20)
    endpoint, slope = Features.rollingregression!(endpoint, slope, prices, 20)
    fullendpoint, fullslope = Features.rollingregression(prices, 20)
    @test endpoint ≈ fullendpoint
    @test slope ≈ fullslope rtol=1e-7 atol=1e-12
end

@testset "extreme transitions and exact threshold" begin
    prices = [110.0, 100.0, 100.5, 99.8, 100.7, 100.9, 101.2, 100.8, 100.0]
    @test Features.extremeanchors(prices, 1, 3, 0.01) == (2, 1)
    @test Features.extremeanchors(prices, 1, 4, 0.01) == (4, 1)
    @test Features.extremeanchors(prices, 1, 5, 0.01) == (4, 1)
    @test Features.extremeanchors(prices, 1, 6, 0.01) == (4, 6)
    @test Features.extremeanchors(prices, 1, 7, 0.01) == (4, 7)
    @test Features.extremeanchors([100.0, 99.0, 99.5, 98.9], 1, 4, 0.01) == (4, 1)
    # A binary-exact 12.5% band leaves an equal-sized move unqualified.
    @test Features.extremeanchors([128.0, 100.0, 112.5, 99.0], 1, 4, 0.125) == (4, 1)
    @test Features.extremeanchors([128.0, 100.0, 112.6, 99.0], 1, 4, 0.125) == (2, 3)
    result = Features.extremesfeatures(prices; bandratio=0.01, maxwindow=20)
    @test result.minage[4] == 0 && result.maxage[4] == 3
    @test result.minnrg[4] == 0 && result.maxnrg[4] < 0
    @test result.minnrg[5] > 0
    @test result.maxnrg[8] < 0
    @test result.minnrg[6] > 0 && result.maxnrg[9] < 0
    rising = Features.extremesfeatures([100.0,102,105,110,115];bandratio=0.01,maxwindow=20)
    falling = Features.extremesfeatures([115.0,110,105,102,100];bandratio=0.01,maxwindow=20)
    @test rising.minage == Float32[0,1,2,3,4]
    @test rising.maxage == zeros(Float32,5)
    @test all(>(0),rising.minnrg[2:end])
    @test falling.maxage == Float32[0,1,2,3,4]
    @test falling.minage == zeros(Float32,5)
    @test all(<(0),falling.maxnrg[2:end])
end

@testset "bounded tree replay and regression reference" begin
    patterns = [
        fill(100.0, 41),
        collect(100.0:0.3:112.0),
        collect(112.0:-0.3:100.0),
        [100.0 + 4sin(ix * 1.7) + 0.5cos(ix * 0.3) for ix in 1:81],
        [100.0 + 0.1sin(ix) for ix in 1:81],
        [100.0, 110, 105, 100, 100, 101, 102, 103, 103, 103, 102, 100],
    ]
    for prices in patterns, window in (2, 5, 16, 100), band in (0.001, 0.01, 0.125)
        result = Features.extremesfeatures(prices; bandratio=band, maxwindow=window)
        tree = Features.ExtremeSummaryTree(prices)
        for ix in eachindex(prices)
            firstix = max(1, ix - window + 1)
            segment = prices[firstix:ix]
            minlocal, maxlocal = referenceanchors(segment, band)
            minimumix, maximumix = firstix + minlocal - 1, firstix + maxlocal - 1
            @test Features.extremeanchors(prices, firstix, ix, band) == (minimumix, maximumix)
            state = Features.ExtremeState(firstix, firstix, 0)
            @test Features.replayextremes!(state, tree, band, firstix + 1, ix) === state
            @test (state.minimumix, state.maximumix) == (minimumix, maximumix)
            for (anchor, slope, age) in ((minimumix, result.minnrg, result.minage),
                    (maximumix, result.maxnrg, result.maxage))
                reference = centeredreference(prices[anchor:ix])
                moments = Features.rangeregression(tree, anchor, ix)
                @test isapprox(Features.regressionresult(moments).grad, reference.grad; atol=1e-12, rtol=1e-9)
                @test isapprox(slope[ix], reference.grad / reference.meanprice; atol=1e-9, rtol=2e-6)
                @test age[ix] == ix - anchor
                @test 0 <= age[ix] <= window - 1
            end
        end
        prefix = Features.extremesfeatures(prices[1:10]; bandratio=band, maxwindow=window)
        scaled = Features.extremesfeatures(prices .* 16; bandratio=band, maxwindow=window)
        for code in propertynames(result)
            @test getproperty(prefix, code) == getproperty(result, code)[1:10]
            @test getproperty(scaled, code) ≈ getproperty(result, code)
        end
    end
    @test Features.extremesfeatures([100.0]; bandratio=0.01, maxwindow=2) ==
        (minnrg=Float32[0], minage=Float32[0], maxnrg=Float32[0], maxage=Float32[0])
    for prices in (Float64[], [0.0], [-1.0], [NaN], [Inf])
        @test_throws AssertionError Features.extremesfeatures(prices; bandratio=0.01, maxwindow=2)
    end
    @test_throws AssertionError Features.extremesfeatures([100.0]; bandratio=0.0, maxwindow=2)
    @test_throws AssertionError Features.extremesfeatures([100.0]; bandratio=0.01, maxwindow=1)
end

"""Configure the four extreme inputs and optionally an existing offset regression feature."""
function extremeconfiguration(; mixed::Bool=false)
    f6 = Features.Features006()
    Features.addextremes!(f6; bandratio=0.01, maxwindow=16)
    mixed && Features.addnormgrad!(f6; window=5, offset=2)
    return f6
end

@testset "Features006 extreme integration" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime("2025-02-17T13:30:00")
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(80))
    odf = Ohlcv.dataframe(ohlcv)
    odf[!, :pivot] .= [100 + 4sin(ix * 0.4) for ix in 1:nrow(odf)]
    f6 = extremeconfiguration()
    @test Features.featurecount(f6) == 4
    @test Features.requiredminutes(f6) == 16
    @test f6.maxoffset == 0
    @test Features.regressionwindows(f6) == ([], Set{Int}())
    @test Features.requestedcolumns(f6) == ["minnrg+0.01+16", "minage+0.01+16", "maxnrg+0.01+16", "maxage+0.01+16"]
    Features.setbase!(f6, ohlcv)
    @test Features.issupplementedcurrent(f6)
    @test size(Features.features(f6)) == (nrow(odf), 4)
    @test all(col -> eltype(f6.fdf[!, col]) == Float32, Features.requestedcolumns(f6))
    @test_throws AssertionError Features.addextremes!(f6)
    @test_throws AssertionError Features.addextremes!(Features.Features006(); bandratio=NaN)
    @test_throws AssertionError Features.addextremes!(Features.Features006(); maxwindow=1)
    @test_throws AssertionError Features.addextremes!(extremeconfiguration(); bandratio=0.01, maxwindow=16)
    alternative = Features.Features006()
    Features.addextremes!(alternative; bandratio=0.02, maxwindow=16)
    @test Features.configdescription(alternative) != Features.configdescription(f6)
    Features.addextremes!(alternative; bandratio=0.01, maxwindow=16)
    @test Features.featurecount(alternative) == 8
    floatband = Features.Features006()
    Features.addextremes!(floatband; bandratio=0.01f0, maxwindow=16)
    @test Features.configdescription(floatband) != Features.configdescription(f6)
    missingminute = Ohlcv.defaultohlcv("SINE")
    Ohlcv.setdataframe!(missingminute, DataFrame(odf[vcat(1:20, 22:nrow(odf)), :]))
    gapfeatures = extremeconfiguration()
    Features.setbase!(gapfeatures, missingminute)
    @test all(col -> gapfeatures.fdf[21,col] == 0, Features.requestedcolumns(gapfeatures))

    for mixed in (false, true)
        full = extremeconfiguration(; mixed)
        Features.setbase!(full, ohlcv)
        live = Ohlcv.defaultohlcv("SINE")
        Ohlcv.setdataframe!(live, DataFrame(odf[1:20, :]))
        incremental = extremeconfiguration(; mixed)
        Features.setbase!(incremental, live)
        for lastix in (30, 50, nrow(odf))
            Ohlcv.setdataframe!(live, DataFrame(odf[1:lastix, :]))
            Features.supplement!(incremental)
        end

        for col in Features.requestedcolumns(full)
            @test incremental.fdf[!, col] ≈ full.fdf[!, col]
        end
        Ohlcv.setdataframe!(live, DataFrame(odf[20:50, :]))
        Features.supplement!(incremental)
        Ohlcv.setdataframe!(live, DataFrame(odf))
        Features.supplement!(incremental)
        for col in Features.requestedcolumns(full)
            @test incremental.fdf[!, col] ≈ full.fdf[!, col]
        end
        sliced = Ohlcv.defaultohlcv("SINE")
        Ohlcv.setdataframe!(sliced, view(odf, 40:70, :))
        slicefeatures = extremeconfiguration(; mixed)
        Features.setbase!(slicefeatures, sliced)
        for col in Features.requestedcolumns(f6)
            @test slicefeatures.fdf[!, col] == f6.fdf[40:70, col]
        end
    end
end

@testset "extreme feature store persistence and extension" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        oldpath = EnvConfig.coinspath()
        EnvConfig.setcoinspath!(folder)
        try
            startdt = DateTime("2025-02-17T13:30:00")
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(80))
            odf = Ohlcv.dataframe(ohlcv)
            shortohlcv = Ohlcv.defaultohlcv("SINE")
            Ohlcv.setdataframe!(shortohlcv, DataFrame(odf[1:40, :]))
            for mixed in (false, true)
                featuresid = mixed ? "EXTREMEMIXEDTEST" : "EXTREMETEST"
                initial = Features.updatefeatures!(shortohlcv, featuresid, extremeconfiguration(; mixed); confighash="extreme-test")
                @test nrow(initial) == 40 - 16 + 1
                extended = Features.updatefeatures!(ohlcv, featuresid, extremeconfiguration(; mixed); confighash="extreme-test")
                full = extremeconfiguration(; mixed)
                Features.setbase!(full, ohlcv; usecache=false)
                expected = full.fdf[(16 - full.maxoffset):end, :]
                @test extended.opentime == expected.opentime
                @test names(extended) == names(expected)
                for col in Features.requestedcolumns(full)
                    @test extended[!, col] ≈ expected[!, col]
                end
                loaded = Features.readfeatures(ohlcv.base, ohlcv.quotecoin, featuresid; confighash="extreme-test")
                @test loaded == extended
            end
        finally
            EnvConfig.setcoinspath!(oldpath)
        end
    end
end
