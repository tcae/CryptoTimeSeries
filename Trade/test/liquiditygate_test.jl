using Test, Dates, DataFrames
using Trade, Ohlcv

@testset "continuous liquidity gate" begin
    function gatedf(quotevolumes::Vector{Float32})
        rows = length(quotevolumes)
        return DataFrame(opentime=[DateTime(2025, 1, 1) + Minute(ix - 1) for ix in 1:rows],
            pivot=ones(Float32, rows), basevolume=quotevolumes)
    end
    checkperiod = 100
    accumulate = 5
    minquotevol = 50f0
    threshold = 0.05
    gate(df, dt) = Trade._continuous_liquidity_now(df, dt; minquotevol, accumulate, checkperiod, threshold)
    rows = checkperiod + accumulate - 1

    liquid = gatedf(fill(20f0, rows))
    @test gate(liquid, last(liquid.opentime))
    @test !gate(liquid, liquid.opentime[end - 1])  # one row short of the required window

    # misses counts accumulated windows below minquotevol: liquid iff misses < round(100*0.05)=5
    misses(qv) = count(ix -> sum(qv[ix-accumulate+1:ix]) < minquotevol, accumulate:rows)
    sparse = fill(20f0, rows)
    sparse[50:53] .= 0f0
    @test misses(sparse) == 4
    @test gate(gatedf(sparse), last(liquid.opentime))
    sparse[54] = 0f0
    @test misses(sparse) == 5
    @test !gate(gatedf(sparse), last(liquid.opentime))

    # The gate becomes true at the same minute as the Ohlcv.liquiditycheck range start.
    ramp = vcat(fill(0f0, 150), fill(20f0, 300))
    df = gatedf(ramp)
    ranges = Ohlcv.liquiditycheck(df; minquotevol, accumulate, checkperiod, startthreshold=threshold,
        stopthreshold=0.25, minliquidminutes=1, startdistance=0)
    @test length(ranges) == 1
    firstopen = findfirst(ix -> gate(df, df.opentime[ix]), eachindex(df.opentime))
    @test firstopen == first(only(ranges))
    # Ohlcv.ld defaults need checkperiod + accumulate - 1 rows
    @test !Trade._continuous_liquidity_now(df, last(df.opentime))
    long = gatedf(fill(1000f0, Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate - 1))
    @test Trade._continuous_liquidity_now(long, last(long.opentime))
    @test Trade._minimum_daily_quotevolume() ==
        round(Ohlcv.liquiddailyminimumquotevolume() / 1000000, digits=0) * 1000000
end
