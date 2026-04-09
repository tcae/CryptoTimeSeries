using Test, Dates, DataFrames
using EnvConfig, Trade, Classify, CryptoXch

@testset "Trade config storage honors Arrow format" begin
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()
    timestamp = DateTime("2025-01-05T11:19:00")

    try
        EnvConfig.init(EnvConfig.test)
        EnvConfig.setdfformat!(:arrow)

        tc = Trade.TradeCache(
            xc=CryptoXch.XchCache(startdt=timestamp, enddt=timestamp),
            cl=Classify.Classifier011(),
        )
        tc.cfg = DataFrame(
            basecoin=["BTC", "ETH"],
            classifieraccepted=[true, false],
            minquotevol=[true, true],
            continuousminvol=[true, false],
            buyenabled=[false, false],
            datetime=[timestamp, timestamp],
        )

        Trade.write(tc, timestamp; folderpath=tmpdir)

        @test isfile(joinpath(tmpdir, "TradeConfig.arrow"))
        @test isfile(joinpath(tmpdir, "TradeConfig_25-01-05.arrow"))

        loaded = Trade.readconfig!(
            Trade.TradeCache(
                xc=CryptoXch.XchCache(startdt=timestamp, enddt=timestamp),
                cl=Classify.Classifier011(),
            ),
            timestamp;
            folderpath=tmpdir,
        )

        @test !isnothing(loaded)
        @test loaded.cfg[!, :basecoin] == ["BTC", "ETH"]
        @test all(loaded.cfg[!, :whitelisted])
        @test loaded.cfg[1, :buyenabled]
        @test !loaded.cfg[2, :buyenabled]
    finally
        EnvConfig.setdfformat!(oldformat)
        rm(tmpdir; force=true, recursive=true)
    end
end
