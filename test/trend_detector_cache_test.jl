module TrendDetectorCacheTests

using Test
using DataFrames
using Dates
using EnvConfig
using Classify
using Targets

include("../scripts/TrendDetector.jl")

@testset "TrendDetector skips empty coin results" begin
    mktempdir() do tmpdir
        @test !TrendDetector._persist_coin_featuretarget_cache("SINE", nothing, nothing; folderpath=tmpdir)

        emptyresults = DataFrame(target=Targets.TradeLabel[])
        emptyfeatures = DataFrame(dummy=Float32[])
        @test !TrendDetector._persist_coin_featuretarget_cache("SINE", emptyresults, emptyfeatures; folderpath=tmpdir)

        savedresults = DataFrame(target=[Targets.allclose], coin=["SINE"], rangeid=Int16[1], set=["train"])
        savedfeatures = DataFrame(dummy=Float32[1.0])
        @test TrendDetector._persist_coin_featuretarget_cache("SINE", savedresults, savedfeatures; folderpath=tmpdir)
        @test EnvConfig.tableexists(TrendDetector.resultsfilename("SINE"); folderpath=tmpdir, format=:auto)
        @test EnvConfig.tableexists(TrendDetector.featuresfilename("SINE"); folderpath=tmpdir, format=:auto)
    end
end

@testset "TrendDetector reads coin-specific Arrow caches without merged cache" begin
    oldformat = EnvConfig.dfformat()
    EnvConfig.init(test)
    EnvConfig.setdfformat!(:arrow)

    cfg = TrendDetector.TrendDetectorConfig(
        configname="ut-trend-coin-arrow-caches",
        featconfig=TrendDetector.trendf6config01(),
        targetconfig=TrendDetector.targetconfig01(),
        classifiermodel=Classify.model002,
        tradingstrategy=TrendDetector.tradingstrategy02(),
        startdt=DateTime("2025-01-01T00:00:00"),
        enddt=DateTime("2025-01-01T00:10:00"),
        coins=["SINE"],
        classbalancing=false,
    )

    try
        cached_results = DataFrame(
            target=Int8[0, 4],
            pivot=Float32[100.0, 101.0],
            high=Float32[102.0, 103.0],
            low=Float32[98.0, 99.0],
            close=Float32[100.0, 100.5],
            set=["train", "eval"],
            coin=["SINE", "SINE"],
            rangeid=Int16[1, 1],
            opentime=[DateTime("2025-01-01T00:00:00"), DateTime("2025-01-01T00:01:00")],
        )
        cached_features = DataFrame(dummy=Float32[1.0, 2.0])

        EnvConfig.deletefolder(TrendDetector.resultsfilename())
        EnvConfig.deletefolder(TrendDetector.featuresfilename())
        EnvConfig.savedf(cached_results, TrendDetector.resultsfilename("SINE"); format=:arrow)
        EnvConfig.savedf(cached_features, TrendDetector.featuresfilename("SINE"); format=:arrow)

        @eval TrendDetector begin
            function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
                error("unexpected trend cache rebuild")
            end
        end

        resultsdf, featuresdf = TrendDetector.getfeaturestargetsdf(cfg)
        @test size(resultsdf, 1) == 2
        @test size(featuresdf, 1) == 2
        @test resultsdf[!, :coin] == cached_results[!, :coin]
        @test resultsdf[!, :target] == [Targets.allclose, Targets.longbuy]
        @test featuresdf[!, :dummy] == cached_features[!, :dummy]
    finally
        EnvConfig.setdfformat!(oldformat)
    end
end

end # module
