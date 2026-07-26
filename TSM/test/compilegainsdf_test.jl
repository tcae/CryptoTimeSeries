module TsmCompileGainsDfTest
using Test
using Dates
using DataFrames

using EnvConfig, TSM

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
            set=[
                "eval", "eval", "eval", "eval", "eval",
                "eval", "eval", "eval", "eval",
                "test", "test", "test",
            ],
            rangeid=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
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
        gainsdf = TSM.compilegainsdf(tradesdf; stem="xchgains", folderpath=tmpdir)

        @test isfile(tradepath)
        @test isfile(EnvConfig.tablepath("xchgains"; folderpath=tmpdir, format=:arrow))
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

        loaded = EnvConfig.readdf("xchgains"; folderpath=tmpdir)
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

        TSM.compilegainsdf(tradesdf; stem="xchgains", folderpath=tmpdir)
        report = TSM.gainsreport(instem="xchgains", stem="xchgainsreport", folderpath=tmpdir)

        @test isfile(EnvConfig.tablepath("xchgainsreport"; folderpath=tmpdir, format=:arrow))
        @test nrow(report) == 2
        @test report[!, :set] == ["eval", "test"]

        evalix = findfirst(==("eval"), report[!, :set])
        testix = findfirst(==("test"), report[!, :set])
        @test !isnothing(evalix)
        @test !isnothing(testix)

        @test report[evalix, :segments] == 5
        @test isapprox(report[evalix, :avggain], (0.1f0 + 0.2f0 + (15f0 / 105f0) + (10f0 / 90f0) + (20f0 / 90f0)) / 5f0; atol=1f-6)
        @test report[evalix, :avgminutes] == 3.0
        @test report[evalix, :q75minutes] == 3
        @test report[evalix, :maxminutes] == 4

        @test report[testix, :segments] == 1
        @test report[testix, :avggain] == 0.05f0
        @test report[testix, :avgminutes] == 2.0
        @test report[testix, :q75minutes] == 2
        @test report[testix, :maxminutes] == 2
    finally
        EnvConfig.setdfformat!(oldformat)
        rm(tmpdir; force=true, recursive=true)
    end
end

end