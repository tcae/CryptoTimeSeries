module TsmCompileGainsDfTest
using Test
using Dates
using DataFrames
using CategoricalArrays

using EnvConfig, TSM

@testset "Continuous output retains processed gap executions, not unused seeds" begin
    tsm = TSM.TsmCache()
    times = collect(DateTime(2025, 3, 7, 11, 29):Minute(1):DateTime(2025, 3, 7, 19, 35))
    rows = DataFrame(
        pair=fill("SOLUSDT", length(times)), opentime=times,
        set=fill("none", length(times)), rangeid=zeros(Int32, length(times)),
        lp_amount=zeros(Float32, length(times)), sp_amount=zeros(Float32, length(times)),
        lol_pavg=zeros(Float32, length(times)), lcl_pavg=zeros(Float32, length(times)),
        sol_pavg=zeros(Float32, length(times)), scl_pavg=zeros(Float32, length(times)),
        tsmstate=fill("xch", length(times)),
    )
    rows[1, :set] = "train"
    rows[1, :rangeid] = 60258
    nextseed = findfirst(==(DateTime(2025, 3, 7, 19, 30)), times)
    rows[nextseed:end, :set] .= "eval"
    rows[nextseed:end, :rangeid] .= 60259
    rows[3:(end - 2), :sp_amount] .= 2f0
    rows[3, :sol_pavg] = 100f0
    rows[end - 1, :scl_pavg] = 90f0
    rows[end, :tsmstate] = TSM.TSM_NO_STATE
    TSM.settrades!(tsm, "SOLUSDT", rows)
    @test nrow(TSM.collecttradesdf(tsm)) == length(times)
    processed = TSM.collecttradesdf(tsm; processedonly=true)
    @test nrow(processed) == length(times) - 1
    @test times[3] in processed.opentime
    @test !(times[end] in processed.opentime)
    gains = TSM.compilegains(processed; setpartitions=false)
    @test nrow(gains) == 1
    @test gains[1, :opentime] == times[2]
    @test gains[1, :closetime] == times[end - 2]
    @test gains[1, :openprice] == 100f0
    @test gains[1, :closeprice] == 90f0
    @test gains[1, :gainquote] == 20f0
end

@testset "TSM compilegainsdf matches FIFO within pair set and range" begin
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()

    try
        EnvConfig.setdfformat!(:arrow)

        tradesdf = DataFrame(
            pair=[
                "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT",
                "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT",
                "BTCUSDT", "BTCUSDT", "BTCUSDT",
            ],
            set=categorical([
                "eval", "eval", "eval", "eval", "eval",
                "eval", "eval", "eval", "eval",
                "test", "test", "test",
            ]),
            rangeid=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            closereason=["none", "none", "none", "stoploss", "trendchange", "none", "none", "stoploss", "takeprofit", "none", "none", "liquidation"],
            opentime=[
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
                DateTime(2024, 1, 1, 0, 3),
                DateTime(2024, 1, 1, 0, 4),
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
                DateTime(2024, 1, 1, 0, 3),
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
            ],
            lp_amount=Float32[0f0, 300f0, 350f0, 150f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 40f0, 0f0],
            sp_amount=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 150f0, 50f0, 0f0, 0f0, 0f0, 0f0],
            lol_pavg=Float32[0f0, 100f0, 105f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 200f0, 0f0],
            lcl_pavg=Float32[0f0, 0f0, 0f0, 110f0, 120f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 210f0],
            sol_pavg=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 90f0, 0f0, 0f0, 0f0, 0f0, 0f0],
            scl_pavg=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 80f0, 70f0, 0f0, 0f0, 0f0],
        )

        tradepath = TSM.savetradesdf(tradesdf; stem="trades-compilegainsdf", folderpath=tmpdir)
        gainsdf = TSM.compilegainsdf(tradesdf; stem="tsmgains", folderpath=tmpdir, setpartitions=true)

        @test isfile(tradepath)
        @test isfile(EnvConfig.tablepath("tsmgains"; folderpath=tmpdir, format=:arrow))
        @test nrow(gainsdf) == 6
        @test gainsdf[!, :pair] == ["BTCUSDT", "BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT", "BTCUSDT"]
        @test gainsdf[!, :set] == ["eval", "eval", "eval", "eval", "eval", "test"]
        @test gainsdf[!, :rangeid] == [1, 1, 1, 1, 1, 2]
        @test gainsdf[!, :side] == ["long", "long", "long", "short", "short", "long"]
        @test gainsdf[!, :opentime] == [
            DateTime(2024, 1, 1, 0, 0),
            DateTime(2024, 1, 1, 0, 0),
            DateTime(2024, 1, 1, 0, 1),
            DateTime(2024, 1, 1, 0, 0),
            DateTime(2024, 1, 1, 0, 0),
            DateTime(2024, 1, 1, 0, 0),
        ]
        @test gainsdf[!, :closetime] == [
            DateTime(2024, 1, 1, 0, 2),
            DateTime(2024, 1, 1, 0, 3),
            DateTime(2024, 1, 1, 0, 3),
            DateTime(2024, 1, 1, 0, 1),
            DateTime(2024, 1, 1, 0, 2),
            DateTime(2024, 1, 1, 0, 1),
        ]
        @test gainsdf[!, :openprice] == Float32[100f0, 100f0, 105f0, 90f0, 90f0, 200f0]
        @test gainsdf[!, :closeprice] == Float32[110f0, 120f0, 120f0, 80f0, 70f0, 210f0]
        @test gainsdf[!, :volume] == Float32[200f0, 100f0, 50f0, 100f0, 50f0, 40f0]
        @test isapprox.(gainsdf[!, :gain], Float32[0.1f0, 0.2f0, 15f0 / 105f0, 10f0 / 90f0, 20f0 / 90f0, 0.05f0]; atol=1f-6) |> all
        @test gainsdf[!, :gainquote] == Float32[2000f0, 2000f0, 750f0, 1000f0, 1000f0, 400f0]
        @test gainsdf[!, :closereason] == ["stoploss", "trendchange", "trendchange", "stoploss", "takeprofit", "liquidation"]

        loaded = EnvConfig.readdf("tsmgains"; folderpath=tmpdir)
        @test !isnothing(loaded)
        @test nrow(loaded) == nrow(gainsdf)
    finally
        EnvConfig.setdfformat!(oldformat)
        rm(tmpdir; force=true, recursive=true)
    end
end

@testset "TSM gainsreport aggregates by set across pairs and ranges" begin
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()

    try
        EnvConfig.setdfformat!(:arrow)

        tradesdf = DataFrame(
            pair=[
                "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT",
                "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT",
                "BTCUSDT", "BTCUSDT", "BTCUSDT",
            ],
            set=[
                "eval", "eval", "eval", "eval", "eval",
                "eval", "eval", "eval", "eval",
                "test", "test", "test",
            ],
            rangeid=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            closereason=["none", "none", "none", "stoploss", "trendchange", "none", "none", "stoploss", "takeprofit", "none", "none", "liquidation"],
            opentime=[
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
                DateTime(2024, 1, 1, 0, 3),
                DateTime(2024, 1, 1, 0, 4),
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
                DateTime(2024, 1, 1, 0, 3),
                DateTime(2024, 1, 1, 0, 0),
                DateTime(2024, 1, 1, 0, 1),
                DateTime(2024, 1, 1, 0, 2),
            ],
            lp_amount=Float32[0f0, 300f0, 350f0, 150f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 40f0, 0f0],
            sp_amount=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 150f0, 50f0, 0f0, 0f0, 0f0, 0f0],
            lol_pavg=Float32[0f0, 100f0, 105f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 200f0, 0f0],
            lcl_pavg=Float32[0f0, 0f0, 0f0, 110f0, 120f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 210f0],
            sol_pavg=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 90f0, 0f0, 0f0, 0f0, 0f0, 0f0],
            scl_pavg=Float32[0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 80f0, 70f0, 0f0, 0f0, 0f0],
        )

        TSM.compilegainsdf(tradesdf; stem="tsmgains", folderpath=tmpdir, setpartitions=true)
        report = TSM.gainsreport(instem="tsmgains", stem="xchgainsreport", folderpath=tmpdir)

        @test isfile(EnvConfig.tablepath("xchgainsreport"; folderpath=tmpdir, format=:arrow))
        @test nrow(report) == 2
        @test report[!, :set] isa CategoricalVector
        @test String.(report[!, :set]) == ["eval", "test"]

        loadedreport = EnvConfig.readdf("xchgainsreport"; folderpath=tmpdir)
        @test !isnothing(loadedreport)
        @test nrow(loadedreport) == nrow(report)

        evalix = findfirst(==("eval"), String.(report[!, :set]))
        testix = findfirst(==("test"), String.(report[!, :set]))
        @test !isnothing(evalix)
        @test !isnothing(testix)

        @test report[evalix, :segments] == 5
        @test report[evalix, :stoplosses] == 2
        @test report[evalix, :trendchanges] == 2
        @test report[evalix, :liquidations] == 0
        @test report[evalix, :takeprofits] == 1
        @test isapprox(report[evalix, :avggain], (0.1f0 + 0.2f0 + (15f0 / 105f0) + (10f0 / 90f0) + (20f0 / 90f0)) / 5f0; atol=1f-6)
        @test report[evalix, :avgminutes] == 3.0
        @test report[evalix, :q75minutes] == 3
        @test report[evalix, :maxminutes] == 4

        @test report[testix, :segments] == 1
        @test report[testix, :stoplosses] == 0
        @test report[testix, :trendchanges] == 0
        @test report[testix, :liquidations] == 1
        @test report[testix, :takeprofits] == 0
        @test report[testix, :avggain] == 0.05f0
        @test report[testix, :avgminutes] == 2.0
        @test report[testix, :q75minutes] == 2
        @test report[testix, :maxminutes] == 2

        pairreport = TSM.gainsreport(TSM.compilegains(tradesdf; setpartitions=true); groupcols=[:pair], totalcols=Symbol[])
        @test String.(pairreport[!, :pair]) == ["BTCUSDT", "ETHUSDT", "total"]
        @test pairreport[end, :segments] == 6

        @testset "Continuous FIFO spans sets, subranges and liquidity ranges" begin
            for side in (:long, :short), nextid in (2, TSM.RANGEID_SUBRANGE_SPAN * 2)
                crossrange = DataFrame(
                    pair=fill("SOLUSDT", 4),
                    set=["train", "train", "eval", "eval"],
                    rangeid=[1, 1, nextid, nextid],
                    opentime=collect(DateTime(2025, 3, 7):Minute(1):DateTime(2025, 3, 7, 0, 3)),
                    lp_amount=side == :long ? Float32[0, 2, 2, 0] : zeros(Float32, 4),
                    sp_amount=side == :short ? Float32[0, 2, 2, 0] : zeros(Float32, 4),
                    lol_pavg=fill(100f0, 4), lcl_pavg=fill(110f0, 4),
                    sol_pavg=fill(100f0, 4), scl_pavg=fill(90f0, 4),
                )
                @test TSM._compilegains_groupcols(crossrange) == [:pair]
                continuous = TSM.compilegains(crossrange; setpartitions=false)
                @test nrow(continuous) == 1
                @test continuous[1, :volume] == 2f0
                @test continuous[1, :gainquote] == 20f0
                @test continuous[1, :side] == String(side)
                @test continuous[1, :set] == "eval"
                @test continuous[1, :rangeid] == nextid
                @test continuous[1, :opentime] == crossrange[1, :opentime]
                @test continuous[1, :closetime] == crossrange[3, :opentime]
                @test_throws "unmatched" TSM.compilegains(crossrange; setpartitions=true)
            end
        end

        fee_gains = TSM.compilegains(tradesdf; setpartitions=true, makerfee=0.01f0, takerfee=0.02f0)
        @test isapprox(fee_gains[1, :gain], 0.079f0; atol=1f-6)
        @test isapprox(fee_gains[4, :gain], (90f0 * 0.99f0 - 80f0 * 1.01f0) / 90f0; atol=1f-6)
        @test isapprox(fee_gains[6, :gain], (210f0 * 0.98f0 - 200f0 * 1.01f0) / 200f0; atol=1f-6)
    finally
        EnvConfig.setdfformat!(oldformat)
        rm(tmpdir; force=true, recursive=true)
    end
end

end