"""
Unit tests for Bounds01 targets using synthetic SINE OHLCV data.

Tests cover:
- absolute bounds: trd.df columns :hightarget and :lowtarget are trailing rolling
  max/min of the high/low prices over the given window.
- relative bounds via lowhigh(trd; relpricediff=true): bounds are expressed as
  (bound - pivot) / pivot.
- centerwidth absolute and relative.
- supplement! incremental (new rows appended to ohlcv).
- removebase!, timerangecut!, uniquelabels, describe, firstrowix/lastrowix.
- DateTime and index-range overloads of lowhigh, centerwidth, labelvalues.
- standalone helpers centerwidth2lowhigh and lowhigh2centerwidth.
"""

# ---------------------------------------------------------------------------
# Helper: compute expected trailing rolling max/min via plain Julia loops so
# the tests are independent of the RollingFunctions library internals.
# ---------------------------------------------------------------------------

function _trailing_max(v::AbstractVector, window::Int)
    n = length(v)
    result = similar(v)
    for i in 1:n
        lo = max(1, i - window + 1)
        result[i] = maximum(v[lo:i])
    end
    return result
end

function _trailing_min(v::AbstractVector, window::Int)
    n = length(v)
    result = similar(v)
    for i in 1:n
        lo = max(1, i - window + 1)
        result[i] = minimum(v[lo:i])
    end
    return result
end

# ---------------------------------------------------------------------------
# Constructor guard
# ---------------------------------------------------------------------------

@testset "Bounds01 constructor guard" begin
    @test_throws AssertionError Targets.Bounds01(0)
    @test_throws AssertionError Targets.Bounds01(-1)
    trd = Targets.Bounds01(1)
    @test trd.window == 1
    @test isnothing(trd.ohlcv)
    @test isnothing(trd.df)
end

# ---------------------------------------------------------------------------
# firstrowix / lastrowix before setbase!
# ---------------------------------------------------------------------------

@testset "Bounds01 firstrowix/lastrowix before setbase!" begin
    trd = Targets.Bounds01(5)
    @test Targets.firstrowix(trd) == 1
    @test Targets.lastrowix(trd)  == 0
end

# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

@testset "Bounds01 describe" begin
    trd = Targets.Bounds01(7)
    d = Targets.describe(trd)
    @test occursin("Bounds01", d)
    @test occursin("window=7", d)
    @test occursin("Base?", d)   # no ohlcv set yet

    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(50)
    ohlcv   = TestOhlcv.testohlcv("SINE", startdt, enddt)
    Targets.setbase!(trd, ohlcv)
    d2 = Targets.describe(trd)
    @test occursin("SINE", d2)
    @test occursin("window=7", d2)
end

# ---------------------------------------------------------------------------
# supplement! without ohlcv (early-return path)
# ---------------------------------------------------------------------------

@testset "Bounds01 supplement! without ohlcv" begin
    trd = Targets.Bounds01(5)
    Targets.supplement!(trd)   # must not throw
    @test isnothing(trd.df)
end

# ---------------------------------------------------------------------------
# uniquelabels
# ---------------------------------------------------------------------------

@testset "Bounds01 uniquelabels" begin
    trd = Targets.Bounds01(5)
    ul = Targets.uniquelabels(trd)
    @test "lowtarget"  in ul
    @test "hightarget" in ul
end

# ---------------------------------------------------------------------------
# removebase!
# ---------------------------------------------------------------------------

@testset "Bounds01 removebase!" begin
    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(50)
    ohlcv   = TestOhlcv.testohlcv("SINE", startdt, enddt)
    trd = Targets.Bounds01(5)
    Targets.setbase!(trd, ohlcv)
    @test !isnothing(trd.df)
    @test !isnothing(trd.ohlcv)
    Targets.removebase!(trd)
    @test isnothing(trd.df)
    @test isnothing(trd.ohlcv)
end

# ---------------------------------------------------------------------------
# supplement! incremental (add rows after initial setbase!)
# ---------------------------------------------------------------------------

@testset "Bounds01 supplement! incremental" begin
    startdt     = DateTime("2025-01-02T01:11:00")
    enddt_short = startdt + Dates.Minute(100)
    enddt_full  = startdt + Dates.Minute(200)
    window      = 10

    ohlcv_short = TestOhlcv.testohlcv("SINE", startdt, enddt_short)
    ohlcv_full  = TestOhlcv.testohlcv("SINE", startdt, enddt_full)
    n_short = size(Ohlcv.dataframe(ohlcv_short), 1)
    n_full  = size(Ohlcv.dataframe(ohlcv_full),  1)
    @assert n_full > n_short "test assumption: full ohlcv must have more rows than short ohlcv"

    trd = Targets.Bounds01(window)
    Targets.setbase!(trd, ohlcv_short)
    @test size(trd.df, 1) == n_short

    # Swap to the longer ohlcv and call supplement! — should add the missing rows
    trd.ohlcv = ohlcv_full
    Targets.supplement!(trd)
    @test size(trd.df, 1) == n_full
    @test trd.df[begin, :opentime] == Ohlcv.dataframe(ohlcv_full)[begin, :opentime]
    @test trd.df[end,   :opentime] == Ohlcv.dataframe(ohlcv_full)[end,   :opentime]
end

# ---------------------------------------------------------------------------
# Main setbase! + core accessor tests
# ---------------------------------------------------------------------------

@testset "Bounds01 tests" begin
for coin in ["SINE"]
    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(200)
    window  = 10

    ohlcv = TestOhlcv.testohlcv(coin, startdt, enddt)
    df    = Ohlcv.dataframe(ohlcv)
    n     = size(df, 1)
    @assert n > window "need more rows than window for a meaningful test, got n=$n"
    # println("Testing Bounds01 with $coin data... max=$(maximum(df[!, :high])), min=$(minimum(df[!, :low]))")

    # --- absolute target columns stored in trd.df ---
    trd = Targets.Bounds01(window)
    Targets.setbase!(trd, ohlcv)

    @test size(trd.df, 1) == n
    @test :hightarget in propertynames(trd.df)
    @test :lowtarget  in propertynames(trd.df)
    @test :opentime   in propertynames(trd.df)
    @test trd.df[!, :opentime] == df[!, :opentime]

    # absolute bounds must bracket the pivot at every row
    @test all(trd.df[!, :hightarget] .>= df[!, :pivot])
    @test all(trd.df[!, :lowtarget]  .<= df[!, :pivot])
    @test all(trd.df[!, :hightarget] .>= trd.df[!, :lowtarget])

    # values must match independently calculated trailing rolling max/min
    expected_high = _trailing_max(df[!, :high], window)
    expected_low  = _trailing_min(df[!, :low],  window)
    @test trd.df[!, :hightarget] ≈ expected_high  atol=1e-5
    @test trd.df[!, :lowtarget]  ≈ expected_low   atol=1e-5
    # println("\n absolute bounds test for $coin\n $(describe(trd.df, :all))")

    # firstrowix / lastrowix after setbase!
    @test Targets.firstrowix(trd) == 1
    @test Targets.lastrowix(trd)  == n

    # --- lowhigh with relpricediff=false returns absolute values (same as trd.df) ---
    abs_df = Targets.lowhigh(trd; relpricediff=false)
    @test abs_df[!, :hightarget] ≈ trd.df[!, :hightarget]  atol=1e-5
    @test abs_df[!, :lowtarget]  ≈ trd.df[!, :lowtarget]   atol=1e-5

    # --- lowhigh with relpricediff=true returns relative (bound - pivot) / pivot ---
    rel_df = Targets.lowhigh(trd; relpricediff=true)
    @test :hightarget in propertynames(rel_df)
    @test :lowtarget  in propertynames(rel_df)

    # relative high >= 0 (max(high) >= pivot), relative low <= 0 (min(low) <= pivot)
    @test all(rel_df[!, :hightarget] .>= 0)
    @test all(rel_df[!, :lowtarget]  .<= 0)
    @test all(rel_df[!, :hightarget] .>= rel_df[!, :lowtarget])
    # println("\n relative bounds test for $coin\n $(describe(rel_df, :all))")

    piv           = df[!, :pivot]
    max_high      = _trailing_max(df[!, :high], window)
    min_low       = _trailing_min(df[!, :low],  window)
    expected_high = (max_high .- piv) ./ piv
    expected_low  = (min_low  .- piv) ./ piv
    @test rel_df[!, :hightarget] ≈ expected_high  atol=1e-5
    @test rel_df[!, :lowtarget]  ≈ expected_low   atol=1e-5

    # --- lowhigh with explicit index subrange ---
    firstix = 5
    lastix  = 50
    sub_abs = Targets.lowhigh(trd, firstix, lastix; relpricediff=false)
    @test size(sub_abs, 1) == lastix - firstix + 1
    @test sub_abs[!, :hightarget] ≈ trd.df[firstix:lastix, :hightarget]  atol=1e-5
    @test sub_abs[!, :lowtarget]  ≈ trd.df[firstix:lastix, :lowtarget]   atol=1e-5

    # --- lowhigh with DateTime range ---
    sub_startdt = df[firstix, :opentime]
    sub_enddt   = df[lastix,  :opentime]
    sub_dt = Targets.lowhigh(trd, sub_startdt, sub_enddt; relpricediff=false)
    @test size(sub_dt, 1) == lastix - firstix + 1
    @test sub_dt[!, :hightarget] ≈ sub_abs[!, :hightarget]  atol=1e-5

    # --- centerwidth absolute ---
    cw_abs = Targets.centerwidth(trd; relpricediff=false)
    @test :centertarget in propertynames(cw_abs)
    @test :widthtarget  in propertynames(cw_abs)
    @test all(cw_abs[!, :widthtarget] .>= 0)
    expected_center = (trd.df[!, :hightarget] .+ trd.df[!, :lowtarget]) ./ 2
    @test cw_abs[!, :centertarget] ≈ expected_center  atol=1e-5

    # --- centerwidth relative ---
    cw_rel = Targets.centerwidth(trd; relpricediff=true)
    @test :centertarget in propertynames(cw_rel)
    @test :widthtarget  in propertynames(cw_rel)
    @test all(cw_rel[!, :widthtarget] .>= 0)
    # relative center = (abs_center - pivot) / pivot
    abs_center = (trd.df[!, :hightarget] .+ trd.df[!, :lowtarget]) ./ 2
    abs_width  = trd.df[!, :hightarget] .- trd.df[!, :lowtarget]
    @test cw_rel[!, :centertarget] ≈ (abs_center .- piv) ./ piv  atol=1e-5
    @test cw_rel[!, :widthtarget]  ≈ abs_width ./ piv             atol=1e-5

    # --- centerwidth with DateTime range ---
    cw_dt = Targets.centerwidth(trd, sub_startdt, sub_enddt; relpricediff=false)
    @test size(cw_dt, 1) == lastix - firstix + 1
    @test cw_dt[!, :centertarget] ≈ cw_abs[firstix:lastix, :centertarget]  atol=1e-5

    # --- labelvalues with index range ---
    lv = Targets.labelvalues(trd, firstix, lastix)
    @test size(lv, 1) == lastix - firstix + 1
    @test :hightarget in propertynames(lv)
    @test :lowtarget  in propertynames(lv)

    # --- labelvalues with DateTime range ---
    lv_dt = Targets.labelvalues(trd, sub_startdt, sub_enddt)
    @test size(lv_dt, 1) == lastix - firstix + 1
    @test lv_dt[!, :hightarget] ≈ lv[!, :hightarget]  atol=1e-5
end
end

# ---------------------------------------------------------------------------
# timerangecut!
# ---------------------------------------------------------------------------

@testset "Bounds01 timerangecut!" begin
    startdt  = DateTime("2025-01-02T01:11:00")
    enddt    = startdt + Dates.Minute(200)
    mid_end  = startdt + Dates.Minute(100)
    window   = 10

    ohlcv_full  = TestOhlcv.testohlcv("SINE", startdt, enddt)
    ohlcv_short = TestOhlcv.testohlcv("SINE", startdt, mid_end)
    n_full  = size(Ohlcv.dataframe(ohlcv_full),  1)
    n_short = size(Ohlcv.dataframe(ohlcv_short), 1)

    trd = Targets.Bounds01(window)
    Targets.setbase!(trd, ohlcv_full)
    @test size(trd.df, 1) == n_full

    # Replace ohlcv with a shorter one without calling setbase!, then cut
    trd.ohlcv = ohlcv_short
    Targets.timerangecut!(trd)
    @test size(trd.df, 1) == n_short
    @test trd.df[end, :opentime] == Ohlcv.dataframe(ohlcv_short)[end, :opentime]

    # timerangecut! without ohlcv must not throw
    trd2 = Targets.Bounds01(window)
    Targets.timerangecut!(trd2)
end

# ---------------------------------------------------------------------------
# Standalone helpers: lowhigh2centerwidth
# ---------------------------------------------------------------------------

@testset "lowhigh2centerwidth no base" begin
    low  = Float32[90.0, 100.0, 50.0]
    high = Float32[110.0, 120.0, 80.0]
    cw = Targets.lowhigh2centerwidth(low, high)
    @test cw[!, :centertarget] ≈ Float32[100.0, 110.0, 65.0]  atol=1e-5
    @test cw[!, :widthtarget]  ≈ Float32[ 20.0,  20.0, 30.0]  atol=1e-5
    @test all(cw[!, :widthtarget] .>= 0)
end

@testset "lowhigh2centerwidth with base" begin
    base = Float32[100.0, 200.0]
    low  = Float32[ 90.0, 190.0]
    high = Float32[110.0, 210.0]
    # abs_center = [100, 200], so rel_center = (center - base)/base = [0, 0]
    # abs_width  = [20, 20],   so rel_width  = width/base = [0.2, 0.1]
    cw = Targets.lowhigh2centerwidth(low, high, base)
    @test cw[!, :centertarget] ≈ Float32[0.0, 0.0]   atol=1e-5
    @test cw[!, :widthtarget]  ≈ Float32[0.2, 0.1]   atol=1e-5
end

@testset "lowhigh2centerwidth assert violations" begin
    # lowtarget > hightarget
    @test_throws AssertionError Targets.lowhigh2centerwidth(Float32[110.0], Float32[90.0])
    # low < 0
    @test_throws AssertionError Targets.lowhigh2centerwidth(Float32[-1.0], Float32[10.0])
    # length mismatch
    @test_throws AssertionError Targets.lowhigh2centerwidth(Float32[1.0, 2.0], Float32[3.0])
end

# ---------------------------------------------------------------------------
# Standalone helpers: centerwidth2lowhigh
# ---------------------------------------------------------------------------

@testset "centerwidth2lowhigh no base" begin
    center = Float32[100.0, 110.0]
    width  = Float32[ 20.0,  20.0]
    lh = Targets.centerwidth2lowhigh(center, width)
    @test lh[!, :lowtarget]  ≈ Float32[ 90.0, 100.0]  atol=1e-5
    @test lh[!, :hightarget] ≈ Float32[110.0, 120.0]  atol=1e-5
end

@testset "centerwidth2lowhigh with base (relative center/width)" begin
    base           = Float32[100.0, 200.0]
    # rel_center = 0.0 means abs_center = base; rel_width = 0.2 means abs_width = 20/40
    rel_center     = Float32[0.0, 0.0]
    rel_width      = Float32[0.2, 0.1]
    lh = Targets.centerwidth2lowhigh(rel_center, rel_width, base)
    # abs_center = base * (1 + 0) = base; abs_width = base * rel_width
    # lowtarget_rel  = (abs_low - base) / base; abs_low = base - width/2
    expected_low_rel  = Float32[-0.1, -0.05]
    expected_high_rel = Float32[ 0.1,  0.05]
    @test lh[!, :lowtarget]  ≈ expected_low_rel   atol=1e-5
    @test lh[!, :hightarget] ≈ expected_high_rel  atol=1e-5
end

@testset "centerwidth2lowhigh assert violations" begin
    # negative width
    @test_throws BoundsError Targets.centerwidth2lowhigh(Float32[100.0], Float32[-1.0])
    # negative center
    @test_throws BoundsError Targets.centerwidth2lowhigh(Float32[-1.0], Float32[10.0])
    # length mismatch center vs width
    @test_throws AssertionError Targets.centerwidth2lowhigh(Float32[1.0, 2.0], Float32[3.0])
    # length mismatch center vs base
    @test_throws AssertionError Targets.centerwidth2lowhigh(Float32[1.0, 2.0], Float32[1.0, 1.0], Float32[100.0])
end
