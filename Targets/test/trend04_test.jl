# Tests for the Trend04 AbstractTargets implementation covering all public
# interface functions beyond the crosscheck-focused tests in
# trend04_crosscheck_test.jl.
#
# Note: testohlcvfrompivots is defined in trend04_crosscheck_test.jl and is
# available here because that file is included first in runtests.jl.

# ---------------------------------------------------------------------------
# Constructor assertions
# ---------------------------------------------------------------------------

@testset "Trend04 constructor assertions" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)

    # minwindow == maxwindow violates strict inequality
    @test_throws AssertionError Targets.Trend04(10, 10, thres)
    # minwindow > maxwindow
    @test_throws AssertionError Targets.Trend04(20, 10, thres)
    # negative minwindow
    @test_throws AssertionError Targets.Trend04(-1, 10, thres)

    # threshold order violated: longbuy < longhold
    bad_thres1 = Targets.LabelThresholds(longbuy=0.005f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    @test_throws AssertionError Targets.Trend04(2, 10, bad_thres1)
    # threshold order violated: shortbuy > shorthold
    bad_thres2 = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.005f0)
    @test_throws AssertionError Targets.Trend04(2, 10, bad_thres2)

    # valid minimal construction
    trd = Targets.Trend04(0, 1, thres)
    @test trd.minwindow == 0
    @test trd.maxwindow == 1
    @test isnothing(trd.ohlcv)
    @test isnothing(trd.df)
end

# ---------------------------------------------------------------------------
# firstrowix / lastrowix before and after setbase!
# ---------------------------------------------------------------------------

@testset "Trend04 firstrowix/lastrowix" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    @test Targets.firstrowix(trd) == 1
    @test Targets.lastrowix(trd)  == 0

    pivots = Float32[100.0, 104.0, 108.0, 112.0, 116.0]
    ohlcv = testohlcvfrompivots(pivots)
    Targets.setbase!(trd, ohlcv)
    @test Targets.firstrowix(trd) == 1
    @test Targets.lastrowix(trd)  == length(pivots)
end

# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

@testset "Trend04 describe" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    d = Targets.describe(trd)
    @test occursin("Trend04", d)
    @test occursin("maxwindow=10", d)
    @test occursin("minwindow=2", d)
    @test occursin("Base?", d)  # no ohlcv set yet

    pivots = Float32[100.0, 102.0, 105.0, 108.0, 110.0]
    ohlcv = testohlcvfrompivots(pivots)
    Targets.setbase!(trd, ohlcv)
    d2 = Targets.describe(trd)
    @test occursin("TEST", d2)
    @test occursin("maxwindow=10", d2)
end

# ---------------------------------------------------------------------------
# uniquelabels
# ---------------------------------------------------------------------------

@testset "Trend04 uniquelabels" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    ul = Targets.uniquelabels(trd)
    @test longbuy   in ul
    @test longhold  in ul
    @test shortbuy  in ul
    @test shorthold in ul
    @test allclose  in ul
end

# ---------------------------------------------------------------------------
# supplement! without ohlcv (early-return path)
# ---------------------------------------------------------------------------

@testset "Trend04 supplement! without ohlcv" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    Targets.supplement!(trd)  # must not throw
    @test isnothing(trd.df)
end

# ---------------------------------------------------------------------------
# supplement! incremental (append new rows after initial setbase!)
# ---------------------------------------------------------------------------

@testset "Trend04 supplement! incremental" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)

    pivots_short = Float32[100.0, 104.0, 108.0, 112.0]
    pivots_full  = Float32[100.0, 104.0, 108.0, 112.0, 116.0, 120.0]
    ohlcv_short  = testohlcvfrompivots(pivots_short)
    ohlcv_full   = testohlcvfrompivots(pivots_full)

    Targets.setbase!(trd, ohlcv_short)
    @test size(trd.df, 1) == length(pivots_short)

    # Replace ohlcv with a longer one; supplement! must add the missing rows
    trd.ohlcv = ohlcv_full
    Targets.supplement!(trd)
    @test size(trd.df, 1) == length(pivots_full)
    @test trd.df[end, :opentime] == Ohlcv.dataframe(ohlcv_full)[end, :opentime]
end

# ---------------------------------------------------------------------------
# removebase!
# ---------------------------------------------------------------------------

@testset "Trend04 removebase!" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    pivots = Float32[100.0, 102.0, 105.0, 108.0, 110.0]
    ohlcv = testohlcvfrompivots(pivots)
    Targets.setbase!(trd, ohlcv)
    @test !isnothing(trd.df)
    @test !isnothing(trd.ohlcv)
    Targets.removebase!(trd)
    @test isnothing(trd.df)
    @test isnothing(trd.ohlcv)
end

# ---------------------------------------------------------------------------
# timerangecut! without ohlcv (early-return path)
# ---------------------------------------------------------------------------

@testset "Trend04 timerangecut! without ohlcv" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    Targets.timerangecut!(trd)  # must not throw
    @test isnothing(trd.df)
end

# ---------------------------------------------------------------------------
# labels()
# ---------------------------------------------------------------------------

@testset "Trend04 labels()" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)

    # Before setbase!: returns empty
    @test length(Targets.labels(trd)) == 0

    pivots = Float32[100.0, 104.0, 108.0, 112.0, 116.0, 120.0]
    ohlcv = testohlcvfrompivots(pivots)
    n = length(pivots)
    Targets.setbase!(trd, ohlcv)

    # Full range
    lbls = Targets.labels(trd)
    @test length(lbls) == n
    @test all(lbl in Targets.uniquelabels(trd) for lbl in lbls)

    # Index subrange
    sub = Targets.labels(trd, 2, 4)
    @test length(sub) == 3
    @test collect(sub) == collect(lbls[2:4])

    # DateTime subrange
    df = Ohlcv.dataframe(ohlcv)
    dt_sub = Targets.labels(trd, df[2, :opentime], df[4, :opentime])
    @test length(dt_sub) == 3
    @test collect(dt_sub) == collect(sub)
end

# ---------------------------------------------------------------------------
# relativegain()
# ---------------------------------------------------------------------------

@testset "Trend04 relativegain()" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)

    # Before setbase!: returns empty Float32 vector
    rg = Targets.relativegain(trd)
    @test length(rg) == 0

    pivots = Float32[100.0, 104.0, 108.0, 112.0, 116.0, 120.0]
    ohlcv = testohlcvfrompivots(pivots)
    n = length(pivots)
    Targets.setbase!(trd, ohlcv)

    # Full range
    rg = Targets.relativegain(trd)
    @test length(rg) == n
    @test all(isfinite, rg)

    # Index subrange
    sub_rg = Targets.relativegain(trd, 2, 4)
    @test length(sub_rg) == 3

    # DateTime subrange matches index subrange
    df = Ohlcv.dataframe(ohlcv)
    dt_rg = Targets.relativegain(trd, df[2, :opentime], df[4, :opentime])
    @test length(dt_rg) == 3
    @test dt_rg ≈ sub_rg  atol=1e-6
end

# ---------------------------------------------------------------------------
# labelbinarytargets() and labelrelativegain()
# ---------------------------------------------------------------------------

@testset "Trend04 labelbinarytargets and labelrelativegain" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    pivots = Float32[100.0, 104.0, 108.0, 112.0, 116.0, 120.0]
    ohlcv = testohlcvfrompivots(pivots)
    n = length(pivots)
    Targets.setbase!(trd, ohlcv)

    lbls = collect(Targets.labels(trd))
    rg   = Targets.relativegain(trd)

    # labelbinarytargets: true exactly where label matches
    bt = Targets.labelbinarytargets(trd, longbuy)
    @test length(bt) == n
    @test all(bt[i] == (lbls[i] == longbuy) for i in 1:n)

    # labelbinarytargets with DateTime range
    df = Ohlcv.dataframe(ohlcv)
    bt_dt = Targets.labelbinarytargets(trd, longbuy, df[1, :opentime], df[n, :opentime])
    @test collect(bt_dt) == collect(bt)

    # labelrelativegain: zeroes out non-matching positions
    lrg = Targets.labelrelativegain(trd, longbuy)
    @test length(lrg) == n
    for i in 1:n
        if bt[i]
            @test lrg[i] ≈ rg[i]  atol=1e-6
        else
            @test lrg[i] == 0
        end
    end

    # labelrelativegain with index range
    lrg_sub = Targets.labelrelativegain(trd, longbuy, 2, 4)
    @test length(lrg_sub) == 3

    # labelrelativegain with DateTime range
    lrg_dt = Targets.labelrelativegain(trd, longbuy, df[2, :opentime], df[4, :opentime])
    @test collect(lrg_dt) ≈ collect(lrg_sub)  atol=1e-6
end

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@testset "Trend04 diagnostics" begin
    Targets.reset_trend04_diagnostics!()
    @test isempty(Targets.trend04_diagnostics())

    Targets.enable_trend04_diagnostics!(true)
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    pivots = Float32[100.0, 104.0, 108.0, 112.0, 116.0]
    ohlcv = testohlcvfrompivots(pivots)
    Targets.setbase!(trd, ohlcv)

    diags = Targets.trend04_diagnostics()
    @test haskey(diags, "calls.filltrendanchor")
    @test diags["calls.filltrendanchor"] > 0

    # Disabling stops accumulation; reset clears counters
    Targets.enable_trend04_diagnostics!(false)
    Targets.reset_trend04_diagnostics!()
    @test isempty(Targets.trend04_diagnostics())

    # Running setbase! with disabled diagnostics must not populate counters
    trd2 = Targets.Trend04(2, 10, thres)
    Targets.setbase!(trd2, ohlcv)
    @test isempty(Targets.trend04_diagnostics())
end

# ---------------------------------------------------------------------------
# Base.show
# ---------------------------------------------------------------------------

@testset "Trend04 Base.show" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    s = sprint(show, trd)
    @test occursin("Trend04", s)
    @test occursin("maxwindow=10", s)

    pivots = Float32[100.0, 103.0, 106.0]
    ohlcv = testohlcvfrompivots(pivots)
    Targets.setbase!(trd, ohlcv)
    s2 = sprint(show, trd)
    @test occursin("TEST", s2)
end

# ---------------------------------------------------------------------------
# crosscheck(trd, labels, pivots) edge cases
# ---------------------------------------------------------------------------

@testset "Trend04 crosscheck(trd, labels, pivots) edge cases" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)

    # length mismatch
    issues = Targets.crosscheck(trd, TradeLabel[longbuy], Float32[1.0, 2.0])
    @test !isempty(issues)
    @test any(occursin("length mismatch", s) for s in issues)

    # empty inputs
    issues = Targets.crosscheck(trd, TradeLabel[], Float32[])
    @test !isempty(issues)
    @test any(occursin("empty inputs", s) for s in issues)

    # longhold not preceded by longbuy
    issues = Targets.crosscheck(trd, TradeLabel[longhold, longhold], Float32[100.0, 101.0])
    @test !isempty(issues)
    @test any(occursin("must be preceded by a longbuy segment", s) for s in issues)

    # shorthold not preceded by shortbuy
    issues = Targets.crosscheck(trd, TradeLabel[shorthold, shorthold], Float32[100.0, 99.0])
    @test !isempty(issues)
    @test any(occursin("must be preceded by a shortbuy segment", s) for s in issues)

    # pure allclose is always valid
    issues = Targets.crosscheck(trd, TradeLabel[allclose, allclose, allclose], Float32[100.0, 100.5, 101.0])
    @test isempty(issues)

    # well-formed longbuy segment (5 bars, >3% gain, at segment high)
    issues = Targets.crosscheck(trd, TradeLabel[longbuy, longbuy, longbuy, longbuy, longbuy], Float32[100.0, 101.0, 102.0, 103.0, 104.0])
    if !isempty(issues)
        println("Trend04 crosscheck longbuy issues (informational): " * join(issues, "; "))
    end
    # gain = (104-100)/100 = 4% > 3%, span = 5 >= minwindow = 2 → should be valid
    @test isempty(issues)

    # well-formed shortbuy segment (5 bars, >3% loss, at segment low)
    issues = Targets.crosscheck(trd, TradeLabel[shortbuy, shortbuy, shortbuy, shortbuy, shortbuy], Float32[104.0, 103.0, 102.0, 101.0, 100.0])
    if !isempty(issues)
        println("Trend04 crosscheck shortbuy issues (informational): " * join(issues, "; "))
    end
    # gain = (100-104)/104 ≈ -3.85% < -3% → should be valid
    @test isempty(issues)
end

# ---------------------------------------------------------------------------
# crosscheck(trd) convenience overload without setbase!
# ---------------------------------------------------------------------------

@testset "Trend04 crosscheck(trd) without setbase!" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 10, thres)
    issues = Targets.crosscheck(trd)
    @test !isempty(issues)
    @test any(occursin("missing ohlcv", s) for s in issues)
end

# ---------------------------------------------------------------------------
# Full SINE workflow: setbase! + crosscheck(trd)
# ---------------------------------------------------------------------------

@testset "Trend04 full SINE workflow crosscheck" begin
    thres = Targets.LabelThresholds(longbuy=0.03f0, longhold=0.01f0, shorthold=-0.01f0, shortbuy=-0.03f0)
    trd = Targets.Trend04(2, 240, thres)
    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(200)
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
    n = size(Ohlcv.dataframe(ohlcv), 1)
    Targets.setbase!(trd, ohlcv)

    @test size(trd.df, 1) == n
    @test :label   in propertynames(trd.df)
    @test :relix   in propertynames(trd.df)
    @test :opentime in propertynames(trd.df)
    @test all(lbl in Targets.uniquelabels(trd) for lbl in trd.df[!, :label])

    issues = Targets.crosscheck(trd)
    if !isempty(issues)
        println("Trend04 SINE crosscheck issues:\n" * join(issues, "\n"))
    end
    @test isempty(issues)
end
