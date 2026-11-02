module ScriptHelpTests

using Test, Dates, Statistics
using DataFrames
using Classify, Targets, Distributions, EnvConfig, Xch, Features, Ohlcv, TestOhlcv, TradingStrategy

include("../scripts/BoundsEstimator.jl")
include("../scripts/TrendDetector.jl")
include("../scripts/RegimeTraining.jl")
include("../scripts/refresh.jl")

@testset "script help text includes defaults" begin
    bhelp = BoundsEstimator.boundsestimatorhelp()
    @test occursin("help, --help, -h", bhelp)
    @test occursin("config=<configname>", bhelp)
    @test occursin("test startdt: `2025-01-17T20:56:00`", bhelp)

    thelp = TrendDetector.trenddetectorhelp()
    @test occursin("TS_CONFIGS", thelp)
    @test occursin("stage=<stores|train|predict|evaluate|gain>", thelp)
    @test occursin("help, --help, -h", thelp)

end

@testset "RegimeTraining slices complete shared stores" begin
    startdt = DateTime("2025-02-01T00:00:00")
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(1800))
    history = DataFrame(Ohlcv.dataframe(ohlcv)[101:1400,:];copycols=true)
    featurefactory = RegimeTraining.syntheticfeatures
    targetfactory = RegimeTraining.synthetictarget

    features = featurefactory()
    Features.setbase!(features,ohlcv;usecache=false)
    required = Features.requiredminutes(features)
    featurestore = DataFrame(features.fdf[(required-features.maxoffset):end,
        vcat(["opentime"],Features.requestedcolumns(features))];copycols=true)

    target = targetfactory()
    regression = Targets.regressionfeatures(target)
    Features.setbase!(regression,ohlcv;usecache=false)
    Targets.setbase!(target,regression)
    codes = Targets.tradelabelcode.(Targets.labels(target))
    codes[(Targets.finallabelcount(target,length(codes))+1):end] .= Targets.tradelabelcode(Targets.ignore)
    targetstore = DataFrame(opentime=copy(Ohlcv.dataframe(ohlcv).opentime),label=codes)

    selectedfeatures, selectedtargets = RegimeTraining.sharedstorewindow(history,
        featurestore,targetstore,featurefactory,targetfactory,"SINE")
    @test selectedfeatures.opentime == history.opentime[Features.requiredminutes(featurefactory()):end]
    @test selectedtargets.opentime == history.opentime
    lastfinal = nrow(history) - Targets.lookaheadminutes(targetfactory())
    @test all(selectedtargets[(lastfinal+1):end,:label] .== Targets.tradelabelcode(Targets.ignore))
    mktempdir() do folder
        shared = Dict("SINE"=>(selectedfeatures,selectedtargets))
        stores = TradingStrategy.experimentstores!(Dict("SINE"=>history),featurefactory,
            targetfactory,folder;sharedstores=shared)
        @test stores.featurestores[1] == selectedfeatures
        @test stores.targetstores[1] == selectedtargets
        reused = TradingStrategy.experimentstores!(Dict("SINE"=>history),featurefactory,
            targetfactory,folder;sharedstores=shared)
        @test reused.featurestores[1] == selectedfeatures
        @test reused.targetstores[1] == selectedtargets
    end
end

@testset "Refresh config parsing" begin
    @test Refresh.parseoptions(["features=F19,F20", "targets=T24", "pairs=BTC-USDT,ETH"]) ==
        Dict("features"=>"F19,F20", "targets"=>"T24", "pairs"=>"BTC-USDT,ETH")
    @test_throws AssertionError Refresh.parseoptions(["pair=BTC-USDT"])
    @test_throws AssertionError Refresh.parseoptions(["unknown=value"])
    @test_throws AssertionError Refresh.parseoptions(["features=F19", "features=F20"])
    @test Refresh.configids("f19,F20", "features") == ["F19", "F20"]
    @test_throws AssertionError Refresh.configids("F19,F19", "features")
    @test Refresh.exchangecfg("krakenspot") == (Xch.EXCHANGE_KRAKENSPOT, "USDC")
    @test Refresh.resolvepairs("btc-usdt,eth", ["BTC-USDT", "ETH-USDT"], "USDT") ==
        ["BTC-USDT", "ETH-USDT"]
    @test_throws AssertionError Refresh.resolvepairs("BTC", ["ETH-USDT"], "USDT")
    @test Refresh.main(["features=F19", "help"]) === nothing
    @test Refresh.main(["--help"]) === nothing
end

@testset "Refresh writes complete shared stores" begin
    EnvConfig.init(EnvConfig.test)
    oldcoinspath = EnvConfig.activecoinspath
    oldquote = EnvConfig.pairquote
    tempfolder = mktempdir()
    try
        EnvConfig.setcoinspath!(tempfolder)
        EnvConfig.setpairquote!("USDT")
        startdt = DateTime("2025-03-01T00:00:00")
        ohlcv = TestOhlcv.testohlcv("SINE",startdt,startdt+Minute(1200))
        Ohlcv.write(ohlcv)
        failures = Pair{String,String}[]
        Refresh.refreshpair!(failures, "SINE-USDT", ["F08", "F19"], ["T24"])
        @test first.(failures) == ["SINE-USDT F08"]
        @test occursin("insufficient OHLCV history", last(only(failures)))
        Refresh.refreshpair!(failures, "MISSING-USDT", ["F19"], ["T24"])
        @test first.(failures) == ["SINE-USDT F08", "MISSING-USDT OHLCV"]
        @test Refresh.reportfailures(failures) === nothing
        features = Features.readfeatures("SINE","USDT","F19";
            confighash=TradingStrategy.featureshash("F19"))
        targets = Targets.readtargets("SINE","USDT","T24";
            confighash=TradingStrategy.targetshash("T24"))
        @test nrow(features) == nrow(Ohlcv.dataframe(ohlcv))-Features.requiredminutes(TradingStrategy.featureconfig("F19"))+1
        @test features.opentime[end] == Ohlcv.dataframe(ohlcv).opentime[end]
        @test nrow(targets) == nrow(Ohlcv.dataframe(ohlcv))
        @test targets.opentime[end] == Ohlcv.dataframe(ohlcv).opentime[end]
    finally
        EnvConfig.setcoinspath!(relpath(oldcoinspath,dirname(EnvConfig.defaultlogfilespath)))
        EnvConfig.setpairquote!(oldquote)
        rm(tempfolder;force=true,recursive=true)
    end
end

@testset "script help flag detection" begin
    @test BoundsEstimator._wants_help(["help"])
    @test BoundsEstimator._wants_help(["--help"])
    @test BoundsEstimator._wants_help(["help=true"])
    @test !BoundsEstimator._wants_help(["help=false"])

    @test TrendDetector._wants_help(["-h"])
    @test !TrendDetector._wants_help(["train"])

    @test isnothing(TrendDetector._assertknownargs(["train", "ts=050", "stage=train", "coins=SINE", "runid=x"]))
    @test_throws AssertionError TrendDetector._assertknownargs(["config=050"])
    @test_throws AssertionError TrendDetector._assertknownargs(["special"])
    @test_throws AssertionError TrendDetector._assertknownargs(["TRAIN"])

end

@testset "script config selection by configname" begin
    startdt = DateTime("2025-01-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    EnvConfig.init(test)
    rc = TrendDetector.buildrunconfig(["ts=029", "stage=train", "coins=SINE"])
    @test rc.tscfg.configname == "029"
    @test rc.clcfg.classbalancing == false
    @test rc.coinset == "CS00"
    @test rc.coins == ["SINE"]
    @test TrendDetector.classifiername(rc) == "F01-T01-P02-M002-B0-CS00-D00"
    @test TrendDetector.runsstage(rc, "stores") && !TrendDetector.runsstage(rc, "predict")
    @test_throws AssertionError TrendDetector.buildrunconfig(["stage=unknown"])
    @test_throws AssertionError TrendDetector.buildrunconfig(["startdt=2025-07-01T01:00:00"])

    boundscfg = BoundsEstimator.buildcfg(["config=001"], ["SINE"], startdt, enddt)
    @test boundscfg.configname == "001"

end

@testset "class weights use inverse class frequency" begin
    weights = Classify.classweights([2, 1, 0])
    @test weights[1] < weights[2]
    @test weights[3] == 0f0
    @test isapprox((2 * weights[1] + weights[2]) / 3, 1.0; atol=1e-6)
end

@testset "weighted classifier adaptation accepts sample weights" begin
    features = Float32[0 0 0 1; 0 1 2 3]
    targets = [longopen, longopen, longopen, allclose]
    nn = Classify.model006(size(features, 1), [longopen, allclose], "ut_weighted_trend")
    weights = Classify.classweights([3, 1])[[1, 1, 1, 2]]
    mktempdir() do tmpdir
        adapted = Classify.adaptnn!(nn, Classify.MatrixSampleSource(nn, features, targets, weights); maxepochs=3, folderpath=tmpdir)
        @test length(adapted.losses) == 3
    end
end

end
