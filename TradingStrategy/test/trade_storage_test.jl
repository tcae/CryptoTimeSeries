using Test
using Dates
using DataFrames
using EnvConfig
using TradingStrategy

@testset "Trade storage layout" begin
    oldfolder = EnvConfig.logfolder()
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()

    try
        EnvConfig.setlogpath(tmpdir)
        EnvConfig.setdfformat!(:arrow)

        tradedf = DataFrame(
            coin=["BTC", "BTC", "ETH"],
            set=["eval", "test", "eval"],
            trend=["up", "down", "up"],
            gain=Float32[0.1f0, -0.05f0, 0.2f0],
            startdt=[DateTime(2024, 1, 1), DateTime(2024, 1, 1, 0, 5), DateTime(2024, 1, 1, 0, 10)],
            enddt=[DateTime(2024, 1, 1, 0, 1), DateTime(2024, 1, 1, 0, 6), DateTime(2024, 1, 1, 0, 11)],
        )

        paths = TradingStrategy.savetrades(tradedf; stem="gains")
        @test length(paths) >= 3
        @test isfile(EnvConfig.tablepath(joinpath("trades", "gains_all"); folderpath=EnvConfig.logfolder(), format=:arrow))
        @test isfile(EnvConfig.tablepath(joinpath("trades", "gains", "BTC"); folderpath=EnvConfig.logfolder(), format=:arrow))
        @test isfile(EnvConfig.tablepath(joinpath("trades", "gains", "ETH"); folderpath=EnvConfig.logfolder(), format=:arrow))

        loadedall = TradingStrategy.loadtrades(; stem="gains")
        @test nrow(loadedall) == nrow(tradedf)
        @test sort(loadedall, [:coin, :startdt])[!, :gain] == sort(tradedf, [:coin, :startdt])[!, :gain]

        loadedbtc = TradingStrategy.loadtrades("BTC"; stem="gains")
        @test nrow(loadedbtc) == 2
        @test all(==("BTC"), loadedbtc[!, :coin])
    finally
        EnvConfig.setdfformat!(oldformat)
        EnvConfig.setlogpath(oldfolder)
        rm(tmpdir; force=true, recursive=true)
    end
end
