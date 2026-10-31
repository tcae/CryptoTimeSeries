using Test, Dates, DataFrames
import EnvConfig, TradingStrategy, Classify, Targets
include("../scripts/RegimeTraining.jl")

@testset "Synthetic regime training pipeline" begin
    @test RegimeTraining.main(["--help"]) === nothing
    @test_throws AssertionError RegimeTraining.options(["runid=x","runid=y"])
    @test_throws AssertionError RegimeTraining.options(["runid=../invalid"])
    @test_throws AssertionError RegimeTraining.options(["runid=x","unknown=y"])
    EnvConfig.init(EnvConfig.test)
    input = RegimeTraining.inputs(Dict("mode"=>"synthetic"))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        prepared = RegimeTraining.prepare!(input,folder)
        sources = RegimeTraining.trainingsources(prepared,prepared.sets,
            joinpath(folder,"model");trainweights="balanced")
        @test all(!=("unknown"),sources.trainindex.regime)
        @test all(!=("unknown"),sources.evalindex.regime)
        trainkeys = Set(zip(sources.trainindex.coin,sources.trainindex.opentime))
        evalkeys = Set(zip(sources.evalindex.coin,sources.evalindex.opentime))
        @test isempty(intersect(trainkeys,evalkeys))
        @test sources.train isa TradingStrategy.WeightedExperimentSource
        best = RegimeTraining.trainreport!(prepared,sources;epochs=1,resume=false)
        metrics = EnvConfig.readtable("eval-signals";folderpath=sources.folder,format=:arrow)
        @test Set(metrics.grouping)==Set(["overall","regime","fold","regime_fold"])
        @test all(in(["longopen","longhold","shortopen","shorthold","allclose",
            "long","short"]),metrics.signal)
        @test :precision_lower in propertynames(metrics)
        @test any(!ismissing,metrics.coverage_lower)
        @test nrow(EnvConfig.readtable("eval-macro";folderpath=sources.folder,format=:arrow))>0
        @test isfile(joinpath(sources.folder,"training-report.txt"))
        @test occursin("stage=gain",read(joinpath(sources.folder,"next-steps.txt"),String))
        @test isfile(joinpath(sources.folder,"eval-curves.svg"))
        svg = read(joinpath(sources.folder,"eval-curves.svg"),String)
        @test occursin(".point:hover .tooltip",svg)
        @test occursin("tabindex=\"0\"",svg)
        @test occursin("coverage (0-1)",svg)
        @test isfile(joinpath(sources.folder,"eval-candidates.arrow"))
        @test isfile(joinpath(sources.folder,"eval-macro.arrow"))
        @test !isfile(joinpath(sources.folder,"test-predictions.arrow"))
        tested = RegimeTraining.testreport!(prepared,sources,prepared.sets,best,0.5)
        @test all(==("test"),tested.set)
        @test all(!=("unknown"),tested.regime)
        testkeys = Set(zip(tested.coin,tested.opentime))
        @test isempty(intersect(trainkeys,testkeys))
        @test isempty(intersect(evalkeys,testkeys))
        @test_logs (:warn,r"test metric threshold or checkpoint changed") RegimeTraining.testreport!(prepared,sources,prepared.sets,best,0.6)
        frozen = EnvConfig.readtable("test-signals";folderpath=sources.folder,format=:arrow)
        @test unique(frozen.threshold)==[0.6]
        thresholdhistory = EnvConfig.readtable("test-threshold-history";folderpath=sources.folder,format=:arrow)
        @test thresholdhistory.threshold==[0.5,0.6]
        RegimeTraining.gainreport!(prepared,sources,best)
        gainfolder = joinpath(prepared.folder,"gain",input.rc.folder)
        gainidentity = EnvConfig.readtable("gain-identity";folderpath=gainfolder,format=:arrow)
        @test only(gainidentity.population)=="test"
        gainpredictions = EnvConfig.readtable("test-predictions";folderpath=sources.folder,format=:arrow)
        @test all(==("test"),gainpredictions.set)
        @test Set(zip(gainpredictions.coin,gainpredictions.opentime))==testkeys
        strategy = TradingStrategy.strategytemplate(input.rc.tscfg.strategy)
        index = EnvConfig.readtable("test-index";folderpath=sources.folder,format=:arrow)
        @test RegimeTraining.recordteststrategy!(sources,index,strategy)===nothing
        fields = NamedTuple{fieldnames(TradingStrategy.StrategyConfig)}(
            map(field->getfield(strategy,field),fieldnames(TradingStrategy.StrategyConfig)))
        changedstrategy = TradingStrategy.StrategyConfig(;merge(fields,(maxbudgetquote=201f0,))...)
        @test_logs (:warn,r"test replay strategy or checkpoint changed") RegimeTraining.recordteststrategy!(sources,index,changedstrategy)
        recorded = EnvConfig.readtable("test-strategy";folderpath=sources.folder,format=:arrow)
        @test only(recorded.strategy)==repr(changedstrategy)
        history = EnvConfig.readtable("test-strategy-history";folderpath=sources.folder,format=:arrow)
        @test nrow(history)==3
        @test history.strategy[1]==repr(strategy)
        @test history.strategy[end]==repr(changedstrategy)
        @test RegimeTraining.recordteststrategy!(sources,index,changedstrategy)===nothing
        @test_logs (:warn,r"adapting a classifier previously used for test metric reporting") (:warn,r"adapting a classifier previously used for test gain replay") match_mode=:any begin
            best = RegimeTraining.trainreport!(prepared,sources;epochs=2,resume=true,reinforce=1)
        end
        losses = EnvConfig.readtable("epoch-losses";folderpath=sources.folder,format=:arrow)
        @test losses.epoch==[1,2]
        refreshes = EnvConfig.readtable("reinforcement-refreshes";folderpath=sources.folder,format=:arrow)
        @test refreshes.epoch==[1,2]
        @test all(==(1),refreshes.interval)
        @test_logs (:warn,r"test metric threshold or checkpoint changed") RegimeTraining.testreport!(prepared,sources,prepared.sets,best,0.4)
        thresholdhistory = EnvConfig.readtable("test-threshold-history";folderpath=sources.folder,format=:arrow)
        @test thresholdhistory.threshold==[0.5,0.6,0.4]
        @test thresholdhistory.checkpoint[end]!=thresholdhistory.checkpoint[1]
        schedule = TradingStrategy.walkforwardschedule(input.startdt,input.enddt;
            trainminutes=6000,evalminutes=3000,testminutes=3000,gapminutes=60)
        window = first(eachrow(schedule))
        sets = TradingStrategy.chronologicalmemberships(prepared,window)
        for table in values(sets)
            @test all(table.opentime[table.set .== "train"] .<= window.trainstop-Minute(60))
            @test all(table.opentime[table.set .== "eval"] .<= window.evalstop-Minute(60))
            @test all(table.opentime[table.set .== "test"] .<= window.teststop-Minute(60))
            @test all(!=("unknown"),table.regime)
        end

        let input = nothing
        @testset "Registered RegimeTraining CLI configuration" begin
            EnvConfig.init(EnvConfig.test)
            rc = RegimeTraining.buildrunconfig(RegimeTraining.options(["test","ts=063"]))
            @test rc.coinset=="CS00"
            @test rc.daterange=="D00"
            @test rc.clcfg.features=="F20"
            @test rc.clcfg.targets=="T25"
            @test rc.clcfg.model=="M011"
            @test rc.epochs==10
            @test rc.coins==TradingStrategy.coinsetconfig("CS00").coins
            @test rc.startdt==TradingStrategy.daterangeconfig("D00").startdt
            @test rc.runid==RegimeTraining.buildrunconfig(RegimeTraining.options(["ts=063","test","stage=gain"])).runid
            reinforced = RegimeTraining.buildrunconfig(Dict("ts"=>"063","test"=>"true","reinforce"=>"3"))
            @test reinforced.reinforce==3
            @test reinforced.runid==rc.runid
            @test reinforced.description==rc.description
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("reinforce"=>"-1"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("stage"=>"gain","reinforce"=>"3"))
            @test_throws AssertionError RegimeTraining.options(["test","train"])
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"060"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("startdt"=>"2025-01-01"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("coins"=>"SINE,SINE"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("mincoins"=>"5"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("stage"=>"gain","retrain"=>"true"))
            TradingStrategy.TS_CONFIGS["regime-test-context"] = merge(rc.tscfg,
                (configname="regime-test-context",strategy="S07"))
            try
                other = RegimeTraining.buildrunconfig(Dict("ts"=>"regime-test-context"))
                @test other.clcfg==rc.clcfg
                @test other.description==rc.description
                @test other.runid==rc.runid
            finally
                delete!(TradingStrategy.TS_CONFIGS,"regime-test-context")
            end
            input = RegimeTraining.inputs(Dict{String,String}(),rc)
            @test Set(keys(input.histories))==Set(rc.coins)
            @test input.startdt==rc.startdt
            @test input.enddt==rc.enddt
            @test input.featurefactory===TradingStrategy.featurefactory("F20")
            @test Targets.configdescription(input.targetfactory())==Targets.configdescription(TradingStrategy.targetconfig("T25"))
            @test input.modelid=="M011"
            @test all(df->last(df.opentime)==rc.enddt,values(input.histories))
            @test all(df->first(df.opentime)<rc.startdt,values(input.histories))
            EnvConfig.init(EnvConfig.training)
            market = RegimeTraining.buildrunconfig(Dict("train"=>"true","coins"=>"AAVE,ADA",
                "startdt"=>"2026-08-01","enddt"=>"2026-10-31","mincoins"=>"5","weightcap"=>"0.2"))
            @test market.coinset=="CS01"
            @test market.daterange=="D01"
            @test market.epochs==200
            @test market.mode=="market"
            @test market.coins==["AAVE","ADA"]
        end
        end
        future = deepcopy(input.histories)
        for history in values(future)
            history.basevolume[history.opentime .> window.teststop] .= 0
        end
        changedinput = merge(input,(histories=future,))
        changed = merge(prepared,(input=changedinput,))
        @test isequal(sets,TradingStrategy.chronologicalmemberships(changed,window))
        cutoff = input.startdt+Minute(12000)
        partial = TradingStrategy.foldcalendar(prepared.market,prepared.eligible,input.folds;
            horizonminutes=60,enddt=cutoff)
        @test all(partial.stop .<= cutoff-Minute(60))
    end
end
