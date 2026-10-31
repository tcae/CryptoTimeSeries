using Test, Dates, DataFrames, Features, Statistics, EnvConfig, Arrow, Targets, Ohlcv
include("../scripts/diagnose_market_regimes.jl")

@testset "Normalized minute slope diagnostic" begin
    times = collect(DateTime("2025-01-01T00:00:00"):Minute(1):DateTime("2025-01-01T03:59:00"))
    prices = [100.0 + ix * 0.01 for ix in eachindex(times)]
    df = DataFrame(opentime=times, open=prices, high=prices, low=prices, close=prices, basevolume=ones(length(times)))
    observed = MarketRegimeDiagnostics.coinindicator(df; longminutes=120, shortminutes=60)
    @test observed.opentime == [DateTime("2025-01-01T02:00:00"), DateTime("2025-01-01T03:00:00"), DateTime("2025-01-01T04:00:00")]
    for (rowix, endpoint) in enumerate((120, 180, 240))
        regry, gradient = Features.rollingregression(prices, 120, endpoint)
        expected = gradient[1] / Features.regressionmeanpivot(regry[1], gradient[1], 120) * 1440
        @test observed.slope[rowix] ≈ expected rtol=1e-8
        @test observed.quote_turnover[rowix] ≈ mean(prices[endpoint-119:endpoint])
    end
    liquid = trues(length(times))
    liquid[180] = false
    @test MarketRegimeDiagnostics.coinindicator(df; longminutes=120, shortminutes=60, liquid).opentime ==
        [DateTime("2025-01-01T02:00:00"), DateTime("2025-01-01T04:00:00")]

    @testset "Stored eligible regime sample counts" begin
        EnvConfig.init(EnvConfig.test)
        previous = EnvConfig.activecoinspath
        folder = mktempdir(mkpath(EnvConfig.defaultdebugfilespath))
        try
            EnvConfig.setcoinspath!(relpath(folder, dirname(EnvConfig.defaultlogfilespath)))
            coinfolder = EnvConfig.coinfolderpath("REGIMETEST", EnvConfig.pairquote)
            for directory in ("features", "targets", "sets")
                mkpath(joinpath(coinfolder, directory))
            end
            start = DateTime("2025-01-01")
            sampletimes = [start + Minute(ix) for ix in 0:3]
            Arrow.write(joinpath(coinfolder, "sets", "P02_D01.arrow"),
                DataFrame(opentime=sampletimes, set=["train", "eval", "test", "train"], rangeid=1:4))
            Arrow.write(joinpath(coinfolder, "features", "F19.arrow"), DataFrame(opentime=sampletimes[2:4]))
            Arrow.write(joinpath(coinfolder, "targets", "T24.arrow"),
                DataFrame(opentime=sampletimes, label=[Targets.tradelabelcode(Targets.allclose),
                    Targets.tradelabelcode(Targets.longopen), Targets.tradelabelcode(Targets.ignore),
                    Targets.tradelabelcode(Targets.shortopen)]))
            Arrow.write(joinpath(folder, "slope0.0025-strength0.5-confirm12-hourly.arrow"),
                DataFrame(opentime=[start], retrospective=["up"], online=["flat"]))
            table = MarketRegimeDiagnostics.samplecoverage(folder; coins=["REGIMETEST"])
            @test sum(table.samples) == 2
            @test Set(table.set) == Set(["train", "eval"])
            @test all(==("up"), table.regime)
            @test all(==("2025-01"), table.month)
            @test isfile(joinpath(folder, "sample-store-status.arrow"))
        finally
            EnvConfig.setcoinspath!(relpath(previous, dirname(EnvConfig.defaultlogfilespath)))
            rm(folder; recursive=true)
        end
    end
    df.open .*= 10
    df.high .*= 10
    df.low .*= 10
    df.close .*= 10
    scaled = MarketRegimeDiagnostics.coinindicator(df; longminutes=120, shortminutes=60)
    @test scaled.slope ≈ observed.slope
    @test scaled.quote_turnover ≈ 10 .* observed.quote_turnover
    nonlinear = copy(df)
    nonlinear.open .= [100 + 3sin(ix / 17) for ix in eachindex(times)]
    nonlinear.high .= nonlinear.open
    nonlinear.low .= nonlinear.open
    nonlinear.close .= nonlinear.open
    observed_nonlinear = MarketRegimeDiagnostics.coinindicator(nonlinear; longminutes=120, shortminutes=60)
    for (rowix, endpoint) in enumerate((120, 180, 240))
        windowprices = nonlinear.open[endpoint-119:endpoint]
        regry, gradient = Features.rollingregression(windowprices, 120, 120)
        expected = gradient[1] / Features.regressionmeanpivot(regry[1], gradient[1], 120) * 1440
        @test observed_nonlinear.slope[rowix] ≈ expected rtol=1e-8
    end
    @test_throws AssertionError MarketRegimeDiagnostics.coinindicator(vcat(df, df[end:end, :]); longminutes=120, shortminutes=60)
    df.opentime[121:end] .+= Minute(1)
    gap = MarketRegimeDiagnostics.coinindicator(df; longminutes=120, shortminutes=60)
    @test all(time -> time == observed.opentime[1] || time >= DateTime("2025-01-01T04:00:00"), gap.opentime)
end

@testset "Capped market weights" begin
    weights = MarketRegimeDiagnostics.cappedweights([100.0, 1, 1, 1]; cap=0.4)
    @test weights ≈ [0.4, 0.2, 0.2, 0.2]
    @test sum(weights) ≈ 1
    @test maximum(weights) <= 0.4
    @test MarketRegimeDiagnostics.cappedweights(ones(10)) ≈ fill(0.1, 10)
    @test_throws AssertionError MarketRegimeDiagnostics.cappedweights(ones(9))
    @test_throws AssertionError MarketRegimeDiagnostics.cappedweights([1.0, 0]; cap=0.5)
end

@testset "Liquidity mask matches Ohlcv.liquiditycheck" begin
    rows = 30 * 1440
    times = [DateTime("2025-01-01") + Minute(ix - 1) for ix in 1:rows]
    prices = fill(10f0, rows)
    volumes = fill(1000f0, rows)  # 10_000 quote per minute is liquid
    volumes[20*1440:22*1440] .= 0f0  # two illiquid days end the range
    df = DataFrame(opentime=times, open=prices, high=prices, low=prices, close=prices, basevolume=volumes)
    mask = MarketRegimeDiagnostics.liquidmask(df)
    ranges = Ohlcv.liquiditycheck(DataFrame(pivot=prices, basevolume=volumes))
    @test !isempty(ranges)
    expected = falses(rows)
    for range in ranges
        expected[range] .= true
    end
    @test mask == expected
    @test !any(mask[1:Ohlcv.ld.startdistance])
    @test !mask[21*1440]
end

@testset "Confirmation and retrospective regime boundaries" begin
    times = collect(DateTime("2025-01-01"):Hour(1):DateTime("2025-01-01T11:00:00"))
    slopes = [0.0, 0, 0.01, 0.01, 0.01, 0.006, 0.006, 0, 0, 0, -0.01, -0.01]
    online, historical, segments = MarketRegimeDiagnostics.regimes(times, slopes, slopes, fill(0.01, 12),
        [0.5,0.5,0.8,0.8,0.8,0.8,0.8,0.5,0.5,0.5,0.2,0.2];
        threshold=0.005, strength=0.5, confirmation=3)
    @test online[3:4] == ["flat", "flat"]
    @test online[5] == "up"
    @test historical[3:4] == ["up", "up"]
    @test segments.start[2] == times[3]
    @test segments.confirmed[2] == times[5]
    @test historical[8:10] == fill("flat", 3)
    @test online[11:12] == fill("flat", 2)
    @test all(segments.confirmed .>= segments.start)
    # Prefix invariance of online labels; historical annotations may be revised.
    prefix, _, _ = MarketRegimeDiagnostics.regimes(times[1:4], slopes[1:4], slopes[1:4], fill(0.01, 4),
        [0.5,0.5,0.8,0.8]; threshold=0.005, strength=0.5, confirmation=3)
    @test prefix == online[1:4]
    gap_times = [times[1], times[2], times[6], times[7]]
    _, _, gap_segments = MarketRegimeDiagnostics.regimes(gap_times, fill(0.01, 4), fill(0.01, 4),
        fill(0.01, 4), fill(0.8, 4); threshold=0.005, strength=0.5, confirmation=3)
    @test nrow(gap_segments) == 2
    weak, _, _ = MarketRegimeDiagnostics.regimes(times, fill(0.01, 12), fill(0.01, 12),
        fill(1.0, 12), fill(0.8, 12); threshold=0.005, strength=0.5, confirmation=3)
    @test all(==("flat"), weak)
    narrow, _, _ = MarketRegimeDiagnostics.regimes(times, fill(0.01, 12), fill(0.01, 12),
        fill(0.01, 12), fill(0.5, 12); threshold=0.005, strength=0.5, confirmation=3)
    @test all(==("flat"), narrow)
end
