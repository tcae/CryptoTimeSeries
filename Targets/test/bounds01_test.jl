"""
Unit tests for Bounds01 targets using synthetic SINE and DOUBLESINE OHLCV data.

Tests cover:
- absolute bounds (relpricediff=false): highbound and lowbound are trailing rolling
  max/min of the high/low prices over the given window.
- relative bounds (relpricediff=true): bounds are expressed as (bound - pivot) / pivot.
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

@testset "Bounds01 tests" begin
for coin in ["SINE"]
    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(200)
    window  = 10

    ohlcv = TestOhlcv.testohlcv(coin, startdt, enddt)
    df    = Ohlcv.dataframe(ohlcv)
    n     = size(df, 1)
    @assert n > window "need more rows than window for a meaningful test, got n=$n"
    println("Testing Bounds01 with $coin data... max=$(maximum(df[!, :high])), min=$(minimum(df[!, :low]))")

    trd = Targets.Bounds01(window; relpricediff=false)
    Targets.setbase!(trd, ohlcv)

    @test size(trd.df, 1) == n
    @test :highbound in propertynames(trd.df)
    @test :lowbound  in propertynames(trd.df)
    @test :opentime  in propertynames(trd.df)
    @test trd.df[!, :opentime] == df[!, :opentime]

    # bounds must bracket the pivot at every row
    @test all(trd.df[!, :highbound] .>= df[!, :pivot])
    @test all(trd.df[!, :lowbound]  .<= df[!, :pivot])
    @test all(trd.df[!, :highbound] .>= trd.df[!, :lowbound])

    # values must match independently calculated trailing rolling max/min
    expected_high = _trailing_max(df[!, :high], window)
    expected_low  = _trailing_min(df[!, :low],  window)
    @test trd.df[!, :highbound] ≈ expected_high  atol=1e-5
    @test trd.df[!, :lowbound]  ≈ expected_low   atol=1e-5
    println("\n absolute bounds test for $coin\n $(describe(trd.df, :all))")

    ohlcv = TestOhlcv.testohlcv(coin, startdt, enddt)
    df    = Ohlcv.dataframe(ohlcv)
    n     = size(df, 1)
    @assert n > window "need more rows than window for a meaningful test, got n=$n"

    trd = Targets.Bounds01(window; relpricediff=false)
    Targets.setbase!(trd, ohlcv)

    # basic structural checks
    @test size(trd.df, 1) == n
    @test :highbound in propertynames(trd.df)
    @test :lowbound  in propertynames(trd.df)
    @test :opentime  in propertynames(trd.df)
    @test trd.df[!, :opentime] == df[!, :opentime]

    # bounds must bracket the pivot at every row
    @test all(trd.df[!, :highbound] .>= df[!, :pivot])
    @test all(trd.df[!, :lowbound]  .<= df[!, :pivot])
    @test all(trd.df[!, :highbound] .>= trd.df[!, :lowbound])

    # values must match independently calculated trailing rolling max/min
    expected_high = _trailing_max(df[!, :high], window)
    expected_low  = _trailing_min(df[!, :low],  window)
    @test trd.df[!, :highbound] ≈ expected_high  atol=1e-5
    @test trd.df[!, :lowbound]  ≈ expected_low   atol=1e-5

    startdt = DateTime("2025-01-02T01:11:00")
    enddt   = startdt + Dates.Minute(200)
    window  = 10

    ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
    df    = Ohlcv.dataframe(ohlcv)
    n     = size(df, 1)

    trd = Targets.Bounds01(window; relpricediff=true)
    Targets.setbase!(trd, ohlcv)

    @test size(trd.df, 1) == n
    @test :highbound in propertynames(trd.df)
    @test :lowbound  in propertynames(trd.df)

    # relative high bound >= 0 (max(high) >= pivot), relative low bound <= 0 (min(low) <= pivot)
    @test all(trd.df[!, :highbound] .>= 0)
    @test all(trd.df[!, :lowbound]  .<= 0)
    @test all(trd.df[!, :highbound] .>= trd.df[!, :lowbound])
    println("\n relative bounds test for $coin\n $(describe(trd.df, :all))")

    # values must match independently calculated relative trailing rolling bounds
    piv           = df[!, :pivot]
    max_high      = _trailing_max(df[!, :high], window)
    min_low       = _trailing_min(df[!, :low],  window)
    expected_high = (max_high .- piv) ./ piv
    expected_low  = (min_low  .- piv) ./ piv
    @test trd.df[!, :highbound] ≈ expected_high  atol=1e-5
    @test trd.df[!, :lowbound]  ≈ expected_low   atol=1e-5
end
end
