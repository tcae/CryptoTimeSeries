using Test, Dates, DataFrames
import EnvConfig, TradingStrategy, Classify, Targets
include("../scripts/RegimeTraining.jl")

@testset "Readable gain reports separate predicted and reference trades" begin
    gains = DataFrame(set=fill("test<&",3),predicted=[true,true,false],
        opentime=fill(DateTime(2025),3),closetime=fill(DateTime(2025)+Minute(10),3),
        gain=Float32[0.01,0.03,-0.02],closereason=fill("trendchange",3))
    original = copy(gains)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        @test RegimeTraining.gainreadablereport!(gains,folder) === nothing
        text = read(joinpath(folder,"gain-report.txt"),String)
        svg = read(joinpath(folder,"gain-report.svg"),String)
        @test occursin("predicted=true: classifier predictions",text)
        @test occursin("predicted=false: target-label reference",text)
        @test occursin("test&lt;&amp;: Predicted: 2.0%; completed trades=2",svg)
        @test occursin("test&lt;&amp;: Target reference: -2.0%; completed trades=1",svg)
        @test occursin("Not portfolio return",svg)
        @test !occursin("NaN",svg) && !occursin("Inf",svg)
        @test isequal(gains,original)
        gains.gain .= 0f0
        @test RegimeTraining.gainreadablereport!(gains,folder) === nothing
        svg = read(joinpath(folder,"gain-report.svg"),String)
        @test occursin("0.0%",svg)
        @test !occursin("NaN",svg) && !occursin("Inf",svg)
        @test RegimeTraining.gainreadablereport!(gains[1:0,:],folder) === nothing
        @test occursin("No completed trades.",read(joinpath(folder,"gain-report.txt"),String))
        @test occursin("No completed trades.",read(joinpath(folder,"gain-report.svg"),String))
        @test !occursin("Target reference:",read(joinpath(folder,"gain-report.svg"),String))
        gains.gain[1] = NaN32
        @test_throws AssertionError RegimeTraining.gainreadablereport!(gains,folder)
    end
end

@testset "Timestamped progress and mode-specific artifact roots" begin
    io = IOBuffer()
    started = time_ns()
    @test RegimeTraining.progressmessage("test phase";started,io)===nothing
    output = String(take!(io))
    @test occursin(r"^\d{6} \d{2}:\d{2} RegimeTraining test phase elapsed=\d+\.\d+s\n$",output)
    @test RegimeTraining.progressmessage("test phase";io)===nothing
    @test !occursin("elapsed=",String(take!(io)))
    for mode in ("market","test","synthetic")
        root = mode=="market" ? EnvConfig.defaultlogfilespath : EnvConfig.defaultdebugfilespath
        @test RegimeTraining.experimentfolder((;mode,runid="test-run"))==
            joinpath(root,"regime-training","test-run")
    end
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        metrics = DataFrame(regime=[missing,"up"],foldid=[missing,1],
            precision=[missing,missing],recall=[0.5,0.5])
        @test RegimeTraining.persistreportmetrics!(metrics,folder,"metrics")===nothing
        @test eltype(metrics.precision)==Union{Missing,Float64}
        @test eltype(metrics.regime)==Union{Missing,String}
        persisted = EnvConfig.readtable("metrics";folderpath=folder,format=:arrow)
        @test isequal(persisted,metrics)
        invalid = DataFrame(coin=["SINE"],opentime=[DateTime(2025)],set=["eval"],
            regime=["up"],foldid=[1],blockid=[1],target=["longopen"],
            label=["longopen"],score=[2.0])
        @test_throws AssertionError RegimeTraining.signalreport!(invalid,folder,"invalid";
            thresholds=[0.5])
        @test isfile(joinpath(folder,"invalid-predictions.arrow"))
        @test !isfile(joinpath(folder,"invalid-signals.arrow"))
        valid = DataFrame(coin=["AAA","BBB","AAA","BBB"],
            opentime=DateTime(2026).+Minute.([0,0,1,1]),set=fill("eval",4),
            regime=fill("up",4),foldid=ones(Int,4),blockid=ones(Int,4),
            target=fill("longopen",4),label=fill("longopen",4),score=fill(0.9f0,4))
        EnvConfig.writetable(valid,"mapped-input";folderpath=folder,format=:arrow)
        mapped = EnvConfig.readtable("mapped-input";folderpath=folder,format=:arrow)
        metrics = RegimeTraining.signalreport!(mapped,folder,"mapped";thresholds=[0.5])
        @test isequal(metrics,Classify.signalmetrics(valid;thresholds=[0.5],bootstrap=200))
        @test isfile(joinpath(folder,"coin-reports","mapped-coin-AAA.arrow"))
        @test isfile(joinpath(folder,"coin-reports","mapped-coin-BBB.arrow"))
        @test isfile(joinpath(folder,"mapped-report.txt"))
    end
end

@testset "Best-eval classifier publication precedes reporting" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        coinset = "regime-publication-"*basename(root)
        TradingStrategy.COINSET_CONFIGS[coinset] = TradingStrategy.coinsetconfig("CS00")
        rc = RegimeTraining.buildrunconfig(Dict("ts"=>"062","coinset"=>coinset))
        name = TradingStrategy.classifiername(rc.clcfg;coinset,daterange=rc.daterange)
        folder = EnvConfig.classifierfolder(name)
        try
            labels = TradingStrategy.classifierlabels(rc.clcfg)
            cols = TradingStrategy.classifierfeaturecols(rc.clcfg)
            nn = TradingStrategy.modelconfig(rc.clcfg.model)(length(cols),labels,"publication-test")
            targets = repeat(labels,2)
            features = zeros(Float32,length(cols),length(targets))
            train = Classify.MatrixSampleSource(nn,features,targets)
            eval = Classify.MatrixSampleSource(nn,features,targets)
            # Missing prediction metadata deliberately fails reporting after adaptation.
            evalindex = DataFrame(regime=fill("up",length(targets)),foldid=ones(Int,length(targets)))
            evalweights = Classify.fixedevalweights(evalindex.regime,evalindex.foldid)
            reportindex = copy(evalindex)
            membershiprefs = [WeakRef(evalindex.regime), WeakRef(evalindex.foldid)]
            evalindexloader = () -> begin
                GC.gc()
                @test all(reference -> isnothing(reference.value),membershiprefs)
                copy(reportindex)
            end
            input = (;rc,mode="test",modelid=rc.clcfg.model,coins=rc.coins,
                startdt=rc.startdt,enddt=rc.enddt)
            prepared = (;input,identity="publication-dataset")
            sources = (;train,eval,trainindex=DataFrame(),testindex=DataFrame(),
                evalindex,evalweights,evalindexloader,cols,labels,identity="publication-training",
                folder=joinpath(root,"balanced"))
            @test_throws ArgumentError RegimeTraining.trainreport!(prepared,sources;epochs=1,resume=false)
            @test isempty(sources.evalindex)
            @test isfile(joinpath(folder,"nn.bson"))
            @test !isfile(joinpath(folder,"nn-lastepoch.bson"))
            @test isfile(joinpath(sources.folder,"eval-last.bson"))
            @test isfile(joinpath(sources.folder,"eval-best.bson"))
            @test !isfile(joinpath(folder,"fixed-threshold.arrow"))
            loaded = TradingStrategy.loadtrendclassifier(rc.clcfg;coinset,daterange=rc.daterange)
            best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
            @test loaded.nn.fileprefix==Classify.NN_FILEPREFIX
            @test loaded.nn.model.parameters==best.model.parameters
            @test Classify.maxpredict(loaded.nn,features)==Classify.maxpredict(best,features)
            meta = Classify.readmeta(name)
            @test meta["trainingprotocol"]=="regime-eval"
            @test meta["trainingidentity"]==sources.identity
            @test meta["datasetidentity"]==prepared.identity
            @test meta["checkpoint"]==joinpath(sources.folder,"eval-best.bson")
            @test meta["coins"]==join(rc.coins,",")
            @test meta["startdt"]==string(rc.startdt)
            @test length(meta["checkpointsha256"])==64
            @test RegimeTraining.publishclassifier!(prepared,sources,best)===nothing
            cache = joinpath(Classify.predictionsfolder(name),"cached.arrow")
            write(cache,"cache")
            @test RegimeTraining.publishclassifier!(prepared,sources,best)===nothing
            @test isfile(cache)
            meta["checkpointsha256"] = "previous-model"
            Classify.writemeta(name,meta)
            for stem in ("nn-lastepoch.bson","confusion.arrow","xconfusion.arrow")
                write(joinpath(folder,stem),"stale")
            end
            @test RegimeTraining.publishclassifier!(prepared,sources,best)===nothing
            @test !isdir(dirname(cache))
            @test all(!isfile(joinpath(folder,stem)) for stem in
                ("nn-lastepoch.bson","confusion.arrow","xconfusion.arrow"))
            @test Classify.readmeta(name)["checkpointsha256"]!="previous-model"
            @test_throws AssertionError RegimeTraining.publishclassifier!(
                (;prepared...,input=merge(input,(mode="market",))),sources,best)
            @test_throws AssertionError RegimeTraining.publishclassifier!(
                prepared,merge(sources,(cols=["wrong"],)),best)
            @test RegimeTraining.publishclassifier!(
                (;input=(mode="synthetic",)),(;folder=root),best)===nothing
        finally
            rm(folder;recursive=true)
            delete!(TradingStrategy.COINSET_CONFIGS,coinset)
        end
    end
end

@testset "Signal curve predictive values" begin
    EnvConfig.init(EnvConfig.test)
    metrics = DataFrame(signal=fill("longopen",3),grouping=fill("overall",3),
        regime=fill(missing,3),threshold=[0.3,0.4,0.5],coverage=[0.2,0.1,1.0],
        precision=[0.75,1.0,0.5],recall=[0.6,0.2,1.0],tn=[8,9,0],fn=[2,1,0])
    original = copy(metrics)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        @test RegimeTraining.signalplot!(metrics,folder,"eval") === nothing
        svg = read(joinpath(folder,"eval-curves.svg"),String)
        @test occursin("width=\"1480\"",svg)
        @test occursin("<text x=\"60\" y=\"50\">longopen: precision/PPV versus coverage",svg)
        @test occursin("<text x=\"540\" y=\"50\">longopen: recall versus coverage",svg)
        @test occursin("<text x=\"1020\" y=\"50\">longopen: NPV versus coverage",svg)
        @test occursin("overall threshold=0.3 coverage=0.2 precision/PPV=0.75",svg)
        @test occursin("overall threshold=0.3 coverage=0.2 NPV=0.8",svg)
        @test occursin("overall threshold=0.4 coverage=0.1 NPV=0.9",svg)
        @test !occursin("overall threshold=0.5 coverage=1.0 NPV=",svg)
        @test !occursin("NaN",svg)
        @test !occursin("Inf",svg)
    end
    @test isequal(metrics,original)
end

@testset "Coin-scoped preparation and mapped indices preserve pooled results" begin
    EnvConfig.init(EnvConfig.test)
    input = RegimeTraining.inputs(Dict("mode"=>"synthetic","ts"=>"065"))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        pooledfolder = joinpath(root,"pooled")
        streamedfolder = joinpath(root,"streamed")
        pooled = RegimeTraining.prepare!(input,pooledfolder)
        lazyinput = merge(input,(histories=Dict{String,DataFrame}(),))
        references = WeakRef[]
        loadedcoins = String[]
        loader = coin -> begin
            push!(loadedcoins,coin)
            GC.gc()
            @test all(ref -> isnothing(ref.value),references)
            history = RegimeTraining.loadinputhistory(lazyinput,coin)
            push!(references,WeakRef(history))
            history
        end
        streamed = RegimeTraining.preparestreamed!(lazyinput,streamedfolder;historyloader=loader)
        @test loadedcoins==input.coins
        GC.gc()
        @test all(ref -> isnothing(ref.value),references)
        @test streamed.identity==pooled.identity
        @test isempty(streamed.input.histories)
        @test isempty(streamed.stores.featurestores)
        @test isempty(streamed.stores.targetstores)
        @test isempty(streamed.eligible)
        @test isempty(streamed.sets)
        @test isequal(streamed.market,pooled.market)
        @test isequal(streamed.calendar,pooled.calendar)
        for coin in input.coins, stem in ("features","targets","sets")
            actual = RegimeTraining.readpreparedtable(joinpath(streamedfolder,"stores",coin),stem)
            expected = RegimeTraining.readpreparedtable(joinpath(pooledfolder,"stores",coin),stem)
            @test isequal(actual,expected)
        end
        for stem in ("sample-coverage","exclusions","input-identities")
            actual = EnvConfig.readtable(stem;folderpath=streamedfolder,format=:arrow,copycols=true)
            expected = EnvConfig.readtable(stem;folderpath=pooledfolder,format=:arrow,copycols=true)
            if stem=="sample-coverage"
                sort!(actual,[:set,:regime,:foldid])
                sort!(expected,[:set,:regime,:foldid])
            end
            @test isequal(actual,expected)
        end
        sources = RegimeTraining.trainingsources(pooled,pooled.sets,joinpath(pooledfolder,"balanced"))
        compact = RegimeTraining.trainingsources(streamed,streamed.sets,
            joinpath(streamedfolder,"balanced");streaming=true)
        @test compact.identity==sources.identity
        @test isempty(compact.trainindex)
        @test isempty(compact.testindex)
        @test isempty(compact.evalindex)
        @test compact.evalweights == sources.evalweights
        @test isequal(RegimeTraining.loadevalindex(compact),sources.evalindex)
        @test isequal(RegimeTraining.loadtestindex(compact),sources.testindex)
        @test !(RegimeTraining.loadevalindex(compact).row isa Vector)
        for stem in ("train-index","eval-index","test-index")
            actual = RegimeTraining.readpreparedtable(compact.folder,stem)
            expected = RegimeTraining.readpreparedtable(sources.folder,stem)
            @test isequal(actual,expected)
        end
        for (actual,expected) in ((compact.train,sources.train),(compact.eval,sources.eval))
            @test Classify.nsamples(actual)==Classify.nsamples(expected)
            for indices in Iterators.partition(1:Classify.nsamples(actual),4096)
                @test Classify.batch(actual,indices)==Classify.batch(expected,indices)
            end
        end
        @test RegimeTraining.prepare!(lazyinput,streamedfolder;streaming=true).identity==pooled.identity
        requested = merge(lazyinput,(coins=vcat(input.coins,["ZZMISSING"]),))
        skipped = RegimeTraining.preparestreamed!(requested,joinpath(root,"skipped");
            historyloader=coin -> coin=="ZZMISSING" ? nothing : RegimeTraining.loadinputhistory(lazyinput,coin))
        @test skipped.input.coins==input.coins
        @test skipped.identity==pooled.identity
        @test !isdir(joinpath(skipped.folder,"stores","ZZMISSING"))
        featurepath = joinpath(streamedfolder,"stores",first(input.coins),"features.arrow")
        featurebytes = read(featurepath)
        mismatch = merge(lazyinput,(context=lazyinput.context*";changed=true",))
        @test_throws AssertionError RegimeTraining.preparestreamed!(mismatch,streamedfolder)
        @test read(featurepath)==featurebytes
        @test_throws AssertionError RegimeTraining.preparestreamed!(requested,joinpath(root,"empty");
            historyloader=coin -> nothing)
        @test_throws AssertionError RegimeTraining.readpreparedtable(root,"missing")
    end
end

@testset "Synthetic regime training pipeline" begin
    @test RegimeTraining.main(["--help"]) === nothing
    @test_throws AssertionError RegimeTraining.options(["runid=x","runid=y"])
    @test_throws AssertionError RegimeTraining.options(["runid=../invalid"])
    @test_throws AssertionError RegimeTraining.options(["runid=x","unknown=y"])
    EnvConfig.init(EnvConfig.test)
    input = RegimeTraining.inputs(Dict("mode"=>"synthetic","ts"=>"068"))
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
        @test isempty(sources.evalindex)
        @test !isfile(joinpath(sources.folder,"epoch-cells.arrow"))
        evalindex = RegimeTraining.loadevalindex(sources)
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
        predictionprogress = Tuple{Int,Int}[]
        mappedprediction = TradingStrategy.experimentpredictions(best,evalindex,sources.eval;
            onprogress=(completed,total) -> push!(predictionprogress,(completed,total)))
        @test last(predictionprogress)==(nrow(evalindex),nrow(evalindex))
        @test issorted(first.(predictionprogress))
        @test_throws ErrorException TradingStrategy.experimentpredictions(
            best,evalindex,sources.eval;
            onprogress=(completed,total) -> error("intentional prediction progress failure"))
        storedprediction = TradingStrategy.experimentpredictions(best,evalindex,
            prepared.stores,sources.cols)
        @test mappedprediction.label==storedprediction.label
        @test mappedprediction.target==storedprediction.target
        @test mappedprediction.score≈storedprediction.score
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
        @testset "Gain replay uses the configured classifier for active regressions" begin
            strategy = TradingStrategy.strategytemplate(input.rc.tscfg.strategy)
            @test strategy.classifier===nothing
            @test strategy.algorithmconfig.triggerregrwindow==240
            @test strategy.algorithmconfig.trendregrwindow==240
            @test RegimeTraining.gainreport!(prepared,sources,best)===nothing
            gainfolder = joinpath(prepared.folder,"gain",input.rc.folder)
            for coin in input.coins, pass in ("predicted","truth")
                trades = EnvConfig.readtable("$coin-$pass";
                    folderpath=joinpath(gainfolder,"trades-regime"),format=:arrow)
                @test nrow(trades)>0
                @test all(==("test"),trades.set)
                @test all(isfinite,trades.lo_limit)
                @test all(isfinite,trades.so_limit)
            end
        end
        gainfolder = joinpath(prepared.folder,"gain",input.rc.folder)
        @testset "Sealed snapshot replay matches the guarded baseline" begin
            strategy = TradingStrategy.strategytemplate(input.rc.tscfg.strategy)
            snapshotinput = RegimeTraining.snapshotprepared(
                Dict("mode"=>"synthetic","ts"=>"068"),input.rc,folder)
            snapshotsource = RegimeTraining.snapshotsources(snapshotinput;
                trainweights="balanced",folder=sources.folder)
            @test snapshotsource.identity == sources.identity
            @test_throws AssertionError RegimeTraining.loadsnapshot(snapshotinput,snapshotsource)
            golden = Dict((coin,pass)=>EnvConfig.readtable("$coin-$pass";
                folderpath=joinpath(gainfolder,"trades-regime"),format=:arrow)
                for coin in input.coins for pass in ("predicted","truth"))
            goldengains = EnvConfig.readtable("tsmgains-regime";folderpath=gainfolder,format=:arrow)
            predictionsbefore = EnvConfig.readtable("test-predictions";folderpath=sources.folder,format=:arrow)
            trainpath = joinpath(sources.folder,"train",Classify.TRAINFEATURES_FILE)
            trainmodified = mtime(trainpath)
            index = RegimeTraining.loadtestindex(sources)
            group = first(groupby(index,:coin;sort=false))
            coinix = first(group.coinix)
            coin = snapshotinput.stores.coins[coinix]
            history = RegimeTraining.loadinputhistory(snapshotinput.input,coin)
            coverage = TradingStrategy.experimentcoverage(coin,history)
            changedcoverage = merge(coverage,(rows=coverage.rows+1,))
            @test_throws AssertionError RegimeTraining.sealcoin!(snapshotinput,snapshotsource,
                coin,coinix,group,changedcoverage,Int[],folder)
            changedindex = DataFrame(group)
            changedindex.opentime[end] += Minute(1)
            @test_throws AssertionError RegimeTraining.sealcoin!(snapshotinput,snapshotsource,
                coin,coinix,changedindex,coverage,Int[],folder)
            @test RegimeTraining.sealsnapshot!(snapshotinput,snapshotsource) === nothing
            @test_throws AssertionError RegimeTraining.sealsnapshot!(snapshotinput,snapshotsource)
            snapshot = RegimeTraining.loadsnapshot(snapshotinput,snapshotsource)
            @test sum(snapshot.records.rows) == nrow(predictionsbefore)
            @test RegimeTraining.snapshotgainreport!(snapshotinput,snapshotsource) === nothing
            predicted = EnvConfig.readtable("test-predictions";folderpath=sources.folder,format=:arrow)
            @test predicted.label == predictionsbefore.label
            @test predicted.score ≈ predictionsbefore.score
            @test (predicted.score .>= strategy.algorithmconfig.gainlimit.openthreshold) ==
                (predictionsbefore.score .>= strategy.algorithmconfig.gainlimit.openthreshold)
            @test predicted.target == predictionsbefore.target
            for ((coin,pass),expected) in golden
                actual = EnvConfig.readtable("$coin-$pass";
                    folderpath=joinpath(gainfolder,"trades-regime"),format=:arrow)
                @test isequal(actual,expected)
            end
            @test isequal(EnvConfig.readtable("tsmgains-regime";
                folderpath=gainfolder,format=:arrow),goldengains)
            historybefore = nrow(EnvConfig.readtable("test-strategy-history";
                folderpath=sources.folder,format=:arrow))
            rm(joinpath(gainfolder,"gain-report.txt"))
            rm(joinpath(gainfolder,"gain-report.svg"))
            @test RegimeTraining.snapshotgainreport!(snapshotinput,snapshotsource) === nothing
            @test isfile(joinpath(gainfolder,"gain-report.txt"))
            @test occursin("Predicted",read(joinpath(gainfolder,"gain-report.svg"),String))
            @test mtime(trainpath) == trainmodified
            @test nrow(EnvConfig.readtable("test-strategy-history";
                folderpath=sources.folder,format=:arrow)) == historybefore+1
            TradingStrategy.STRATEGY_CONFIGS["SGAINEMPTY"] = ()->TradingStrategy.StrategyConfig(
                algorithm=TradingStrategy.gain_limit_reversal_below_regression!,
                algorithmconfig=TradingStrategy.GainLimitReversalBelowRegressionConfig(triggerdist=100f0))
            try
                emptyrc = merge(snapshotinput.input.rc,
                    (tscfg=merge(snapshotinput.input.rc.tscfg,(strategy="SGAINEMPTY",)),))
                emptyinput = merge(snapshotinput,(input=merge(snapshotinput.input,(rc=emptyrc,)),))
                @test RegimeTraining.snapshotgainreport!(emptyinput,snapshotsource) === nothing
                @test only(EnvConfig.readtable("gain-summary";
                    folderpath=gainfolder,format=:arrow).completed_trades) == 0
                @test isempty(EnvConfig.readtable("tsmgains-regime";folderpath=gainfolder,format=:arrow))
                @test isempty(EnvConfig.readtable("xchgainsreport-regime";folderpath=gainfolder,format=:arrow))
                @test occursin("No completed trades.",read(joinpath(gainfolder,"gain-report.txt"),String))
                @test occursin("No completed trades.",read(joinpath(gainfolder,"gain-report.svg"),String))
            finally
                delete!(TradingStrategy.STRATEGY_CONFIGS,"SGAINEMPTY")
            end
            @test RegimeTraining.snapshotgainreport!(snapshotinput,snapshotsource) === nothing
            coinpath = joinpath(snapshot.folder,first(snapshot.records.coin)*".arrow")
            backup = coinpath*".backup"
            cp(coinpath,backup)
            try
                open(coinpath,"a") do io
                    write(io,"changed")
                end
                @test_throws AssertionError RegimeTraining.snapshotgainreport!(snapshotinput,snapshotsource)
            finally
                mv(backup,coinpath;force=true)
            end
            cache = joinpath(folder,"cache-test")
            @test !RegimeTraining.verifiedcache(cache,"prediction","identity")
            @test RegimeTraining.commitcache(DataFrame(value=[1]),cache,"prediction","identity") === nothing
            @test RegimeTraining.verifiedcache(cache,"prediction","identity")
            @test_throws AssertionError RegimeTraining.verifiedcache(cache,"prediction","other")
            RegimeTraining.atomictable(DataFrame(value=[2]),"prediction",cache)
            @test_throws AssertionError RegimeTraining.verifiedcache(cache,"prediction","identity")
            @test_throws AssertionError RegimeTraining.filedigest(joinpath(cache,"missing.arrow"))
        end
        gainidentity = EnvConfig.readtable("gain-identity";folderpath=gainfolder,format=:arrow)
        @test only(gainidentity.population)=="test"
        gainpredictions = EnvConfig.readtable("test-predictions";folderpath=sources.folder,format=:arrow)
        @test all(==("test"),gainpredictions.set)
        @test Set(zip(gainpredictions.coin,gainpredictions.opentime))==testkeys
        strategy = TradingStrategy.strategytemplate(input.rc.tscfg.strategy)
        index = EnvConfig.readtable("test-index";folderpath=sources.folder,format=:arrow)
        priorreplays = nrow(EnvConfig.readtable("test-strategy-history";
            folderpath=sources.folder,format=:arrow))
        @test RegimeTraining.recordteststrategy!(sources,index,strategy)===nothing
        fields = NamedTuple{fieldnames(TradingStrategy.StrategyConfig)}(
            map(field->getfield(strategy,field),fieldnames(TradingStrategy.StrategyConfig)))
        changedstrategy = TradingStrategy.StrategyConfig(;merge(fields,(maxbudgetquote=201f0,))...)
        @test_logs (:warn,r"test replay strategy or checkpoint changed") RegimeTraining.recordteststrategy!(sources,index,changedstrategy)
        recorded = EnvConfig.readtable("test-strategy";folderpath=sources.folder,format=:arrow)
        @test only(recorded.strategy)==repr(changedstrategy)
        history = EnvConfig.readtable("test-strategy-history";folderpath=sources.folder,format=:arrow)
        @test nrow(history)==priorreplays+2
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
            rc = RegimeTraining.buildrunconfig(RegimeTraining.options(["test","ts=066"]))
            @test rc.coinset=="CS00"
            @test rc.daterange=="D00"
            @test rc.clcfg.features=="F20"
            @test rc.clcfg.targets=="T25"
            @test rc.clcfg.model=="M011"
            @test rc.epochs==10
            @test rc.coins==TradingStrategy.coinsetconfig("CS00").coins
            @test rc.startdt==TradingStrategy.daterangeconfig("D00").startdt
            @test rc.runid==RegimeTraining.buildrunconfig(RegimeTraining.options(["ts=066","test","stage=gain"])).runid
            reinforced = RegimeTraining.buildrunconfig(Dict("ts"=>"066","test"=>"true","reinforce"=>"3"))
            @test reinforced.reinforce==3
            @test reinforced.runid==rc.runid
            @test reinforced.description==rc.description
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","reinforce"=>"-1"))
            for stage in ("snapshot","gain")
                @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","stage"=>stage,"reinforce"=>"3"))
                @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","stage"=>stage,"retrain"=>"true"))
                @test_throws AssertionError RegimeTraining.main(["test","ts=066","stage=$stage","threshold=0.5"])
            end
            @test_throws AssertionError RegimeTraining.options(["test","train"])
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"060"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","startdt"=>"2025-01-01"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","coins"=>"SINE,SINE"))
            @test_throws AssertionError RegimeTraining.buildrunconfig(Dict("ts"=>"066","mincoins"=>"5"))
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
            market = RegimeTraining.buildrunconfig(Dict("ts"=>"066","train"=>"true","coins"=>"AAVE,ADA",
                "startdt"=>"2026-08-01","enddt"=>"2026-10-31","mincoins"=>"5","weightcap"=>"0.2"))
            @test market.coinset=="CS02"
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
        streamed = RegimeTraining.trainingsources(prepared,prepared.sets,
            joinpath(folder,"streamed-model");trainweights="balanced",streaming=true,
            keephistories=true)
        @test isempty(prepared.stores.featurestores)
        @test isempty(prepared.stores.targetstores)
        @test isempty(streamed.trainindex)
        @test isempty(streamed.testindex)
        sampleixs = collect(1:min(16,Classify.nsamples(sources.train)))
        @test first(Classify.batch(streamed.train,sampleixs))==
            first(Classify.batch(sources.train,sampleixs))
        @test isempty(streamed.evalindex)
        @test TradingStrategy.experimentpredictions(best,RegimeTraining.loadevalindex(streamed),streamed.eval).score ≈
            TradingStrategy.experimentpredictions(best,evalindex,sources.eval).score
        @test isequal(RegimeTraining.loadtestindex(streamed),RegimeTraining.loadtestindex(sources))
        @test RegimeTraining.releasetrainingmemory!(prepared,sources)===nothing
        @test isempty(prepared.input.histories)
        @test isempty(prepared.stores.featurestores)
        @test isempty(prepared.stores.targetstores)
        @test isempty(prepared.eligible)
        @test isempty(prepared.sets)
        @test isempty(sources.trainindex)
        @test isempty(sources.evalindex)
        @test isempty(sources.testindex)
        @test nrow(TradingStrategy.experimentpredictions(best,RegimeTraining.loadevalindex(sources),sources.eval))==
            Classify.nsamples(sources.eval)
    end
end
