"""
Terminal runner for regime/fold-aware SimpleChains experiments.
Uses registered trading strategy, coin-set and date-range configurations.
"""
module RegimeTraining

using Dates, DataFrames
import SHA
import EnvConfig, TradingStrategy, Features, Targets, Ohlcv, TestOhlcv, Classify, Xch, Bybit, TSM

"""Return documented CLI usage; test thresholds must be supplied explicitly."""
function helptext()::String
    return """
    julia --project=. scripts/RegimeTraining.jl [test|train] [retrain] [key=value ...]
      test                       default; registered CS00/D00 artificial coins
      train                      registered CS01/D01 local Bybit USDT history
      ts=<id>                    default 063; registered features/targets/model/B flag
      coinset=<CSxx> daterange=<Dxx>  defaults from EnvConfig mode
      coins=<CSV>                override actual training/eval/replay universe
      startdt=<ISO> enddt=<ISO>   together override experiment dates, not just replay
      runid=<immutable-id>        optional; deterministic configuration-derived default
      folder=<name>              gain replay log subfolder
      stage=stores|train|predict|evaluate|gain|test|walkforward  default evaluate
      epochs=<1..200>             default coin-set budget, total including resume
      retrain                    resume last checkpoint; warns after test exposure
      reinforce=<Int>            refresh train misclassification flags every N epochs; 0 off
      mode=synthetic             explicit small SINE/DOUBLESINE smoke fixture
      folds=<comma-separated IDs> optional diagnostic fold subset
      regimes=up,flat,down        optional diagnostic regime subset
      trainweights=class|balanced default class; balanced adds regime/fold weighting
      threshold=<0..1>            required for test/walkforward; fixed across regimes
      mincoins=<Int> weightcap=<0..1>  explicit detector overrides, both required
    Training outputs eval threshold curves without using test for selection.
    Run test only after choosing a fixed threshold from eval reports.
    Market mode refreshes/reuses shared full-history feature and target stores, then selects the experiment window.
    Detector RF01/RG01 settings are recorded in the immutable manifest; no automatic calibration.
    Adaptation ends with eval diagrams and next-step guidance.
    Gain requires an existing checkpoint; it never trains and replays reserved test only.
    Keep the same runid/data options; ts may change if its classifier configuration is identical.
    Ctrl+C stops; completed checkpoints remain. Use retrain to continue.
    """
end

"""Parse CLI keys strictly, rejecting unknown options and duplicate keys."""
function options(args::Vector{String})::Dict{String,String}
    valid = ("mode","runid","stage","epochs","threshold","coins","startdt","enddt",
        "folds","regimes","trainweights","ts","coinset","daterange","folder","mincoins","weightcap","reinforce")
    result = Dict{String,String}()
    for arg in args
        if arg in ("test","train","retrain")
            @assert !haskey(result,arg) "duplicate flag=$arg"
            result[arg] = "true"
            continue
        end
        pair = split(arg,"=";limit=2)
        @assert length(pair)==2 && first(pair) in valid "unknown argument=$arg; use --help"
        @assert !haskey(result,first(pair)) "duplicate argument=$arg"
        result[first(pair)] = last(pair)
    end
    @assert !(haskey(result,"test") && haskey(result,"train")) "test and train flags are exclusive"
    @assert !haskey(result,"runid") || occursin(r"^[A-Za-z0-9_-]+$",result["runid"]) "invalid runid=$(get(result,"runid",""))"
    return result
end

"""Resolve registry configuration and immutable experiment context before reading data."""
function buildrunconfig(opts::Dict{String,String})::NamedTuple
    tscfg = TradingStrategy.tsconfig(get(opts,"ts","063"))
    clcfg = TradingStrategy.classifierconfig(tscfg.classifier)
    @assert clcfg.model in ("M009","M011") "RegimeTraining requires SimpleChains, selected model=$(clcfg.model) ts=$(tscfg.configname)"
    coinset = get(opts,"coinset",TradingStrategy.defaultcoinset())
    daterange = get(opts,"daterange",TradingStrategy.defaultdaterange())
    cs = TradingStrategy.coinsetconfig(coinset)
    dr = TradingStrategy.daterangeconfig(daterange)
    coins = haskey(opts,"coins") ? strip.(split(opts["coins"],",")) : copy(cs.coins)
    @assert !isempty(coins) && allunique(coins) && all(c->!isempty(c)&&uppercase(c)==c,coins) "invalid coins=$coins"
    @assert haskey(opts,"startdt")==haskey(opts,"enddt") "startdt and enddt must be given together, options=$opts"
    startdt = haskey(opts,"startdt") ? DateTime(opts["startdt"]) : dr.startdt
    enddt = haskey(opts,"enddt") ? DateTime(opts["enddt"]) : dr.enddt
    @assert startdt<enddt "invalid dates=$startdt to $enddt"
    mode = get(opts,"mode",EnvConfig.configmode==EnvConfig.test ? "test" : "market")
    @assert mode in ("test","market","synthetic") "invalid mode=$mode"
    @assert haskey(opts,"mincoins")==haskey(opts,"weightcap") "mincoins and weightcap must be given together, options=$opts"
    stage = get(opts,"stage","evaluate")
    @assert stage in ("stores","train","predict","evaluate","gain","test","walkforward") "invalid stage=$stage"
    epochs = parse(Int,get(opts,"epochs",string(cs.maxepochs)))
    @assert 0<epochs<=200 "invalid epoch budget=$epochs"
    reinforce = parse(Int,get(opts,"reinforce","0"))
    @assert reinforce>=0 "reinforce=$reinforce must be nonnegative"
    identityopts = sort([key*"="*value for (key,value) in opts
        if !(key in ("test","train","ts","runid","folder","stage","epochs","retrain","threshold","reinforce"))])
    description = "classifier=$(tscfg.classifier);coinset=$coinset;daterange=$daterange;mode=$mode;coins=$(join(coins,","));start=$startdt;end=$enddt;$(join(identityopts,";"))"
    runid = get(opts,"runid","$(tscfg.classifier)-$coinset-$daterange-"*TradingStrategy.confighash(description)[1:12])
    folder = get(opts,"folder","RegimeTraining-$runid")
    @assert occursin(r"^[A-Za-z0-9_-]+$",folder) "invalid folder=$folder"
    @assert stage!="gain" || !haskey(opts,"retrain") "gain only loads an adapted checkpoint; retrain is not permitted"
    @assert stage!="gain" || reinforce==0 "gain does not train; reinforce=$reinforce belongs to an adaptation run"
    return (;tscfg,clcfg,coinset,daterange,coins,startdt,enddt,mode,stage,epochs,reinforce,runid,folder,description,
        resume=haskey(opts,"retrain"))
end

"""Small real feature implementation used by deterministic synthetic fixtures."""
function syntheticfeatures()::Features.Features006
    f6 = Features.Features006()
    for window in (15,60,240)
        Features.addnormgrad!(f6;window)
    end
    Features.addrelvol!(f6;short=5,long=240)
    return f6
end

"""Synthetic Trend05 target uses a short, explicitly bounded horizon."""
synthetictarget() = Targets.Trend05(leadregr=5,supportregr=15,trendregr=60,
    triggerdist=0.001f0,targetgain=0.005f0,targetminutes=60)

"""Select complete experiment-window rows from the shared full-history feature and target stores."""
function sharedstorewindow(history::DataFrame, featurestore::AbstractDataFrame,
        targetstore::AbstractDataFrame, featurefactory::Function,
        targetfactory::Function, coin::AbstractString)::Tuple{DataFrame,DataFrame}
    required = Features.requiredminutes(featurefactory())
    expectedfeatures = nrow(history) - required + 1
    @assert expectedfeatures > 0 "insufficient history for features of $coin: rows=$(nrow(history)) requiredminutes=$required"
    firstfeature = history.opentime[required]
    featurefirstix = searchsortedfirst(featurestore.opentime,firstfeature)
    featurelastix = searchsortedlast(featurestore.opentime,last(history.opentime))
    @assert featurefirstix <= featurelastix "shared feature store of $coin does not contain expected interval $firstfeature to $(last(history.opentime))"
    features = DataFrame(featurestore[featurefirstix:featurelastix,:];copycols=true)
    @assert (nrow(features) == expectedfeatures) && (features.opentime[begin] == firstfeature) &&
        (features.opentime[end] == last(history.opentime)) "shared feature store of $coin covers $(nrow(features)) rows from $(features.opentime[begin]) to $(features.opentime[end]); expected $expectedfeatures rows from $firstfeature to $(last(history.opentime))"

    targetfirstix = searchsortedfirst(targetstore.opentime,first(history.opentime))
    targetlastix = searchsortedlast(targetstore.opentime,last(history.opentime))
    @assert targetfirstix <= targetlastix "shared target store of $coin does not contain expected interval $(first(history.opentime)) to $(last(history.opentime))"
    targets = DataFrame(targetstore[targetfirstix:targetlastix,:];copycols=true)
    @assert (nrow(targets) == nrow(history)) && (targets.opentime[begin] == history.opentime[begin]) &&
        (targets.opentime[end] == history.opentime[end]) "shared target store of $coin covers $(nrow(targets)) rows from $(targets.opentime[begin]) to $(targets.opentime[end]); expected $(nrow(history)) rows from $(history.opentime[begin]) to $(history.opentime[end])"
    finalcount = max(0,nrow(targets)-Targets.lookaheadminutes(targetfactory()))
    if finalcount < nrow(targets)
        targets[(finalcount+1):end,:label] .= Targets.tradelabelcode(Targets.ignore)
    end
    return features,targets
end

"""Prepare SINE and DOUBLESINE OHLCV, or read an explicitly selected market universe."""
function inputs(opts::Dict{String,String}, rc::NamedTuple=buildrunconfig(opts))::NamedTuple
    mode = rc.mode
    histories = Dict{String,DataFrame}()
    if mode == "synthetic"
        startdt = DateTime(2025,1,1)
        # Lower frequency than the live simulation fixture makes sustained regimes
        # observable while still using TestOhlcv's SINE/DOUBLESINE generator.
        histories["SINE"] = TestOhlcv.sinedata(2400,24000,0,1;firstutc=startdt)
        histories["DOUBLESINE"] = TestOhlcv.sinedata(2400,24000,0,10.5;firstutc=startdt)
        enddt = last(histories["SINE"].opentime)
        liquidity = (minquotevol=5000f0,accumulate=15,checkperiod=60,startthreshold=0.05,
            stopthreshold=0.25,minliquidminutes=120,startdistance=120)
        regime = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
            stepminutes=30,mincoins=2,weightcap=0.5,threshold=0.0025,
            strength=0.0,confirmation=2,liquidity=liquidity)
        folds = TradingStrategy.FoldConfig(anchor=startdt,foldminutes=6000,
            blockminutes=360,gapminutes=60,minminutes=120)
        return (;histories,startdt,enddt,regime,folds,
            featurefactory=syntheticfeatures,targetfactory=synthetictarget,mode,
            modelid=rc.clcfg.model,classbalancing=rc.clcfg.classbalancing,context=rc.description,
            sharedstores=nothing,rc)
    end
    startdt,enddt = rc.startdt,rc.enddt
    EnvConfig.setcoinspath!("Bybit")
    sharedstores = nothing
    if mode == "market"
        TradingStrategy.ensurefeaturetargetstores!(rc.clcfg,rc.coins)
        sharedstores = Dict{String,Tuple{DataFrame,DataFrame}}()
    end
    regime = mode=="test" ? TradingStrategy.RegimeConfig(mincoins=2,weightcap=0.5) : TradingStrategy.RegimeConfig()
    if haskey(opts,"mincoins")
        regime = TradingStrategy.RegimeConfig(mincoins=parse(Int,opts["mincoins"]),
            weightcap=parse(Float64,opts["weightcap"]))
    end
    TradingStrategy.checkregimeconfig(regime)
    featurefactory = TradingStrategy.featurefactory(rc.clcfg.features)
    targetfactory = ()->TradingStrategy.targetconfig(rc.clcfg.targets)
    regression = Targets.regressionfeatures(targetfactory())
    warmup = max(regime.longminutes,regime.liquidity.startdistance+regime.liquidity.minliquidminutes,
        Features.requiredminutes(featurefactory()),isnothing(regression) ? 0 : Features.requiredminutes(regression))
    contextstart = startdt-Minute(warmup+1)
    for coin in rc.coins
        if mode=="test"
            @assert coin in TradingStrategy.coinsetconfig("CS00").coins "test mode requires CS00 artificial symbols, got=$coin"
            minutes = div(Dates.value(enddt-contextstart),60000)
            df = TestOhlcv.sinedata(120,minutes,0,coin=="SINE" ? 1 : 10.5;firstutc=contextstart)
        else
            raw = EnvConfig.readtable("ohlcv";folderpath=EnvConfig.coinfolderpath(coin,"USDT"),format=:arrow)
            @assert !isnothing(raw) "missing OHLCV for $coin"
            df = copy(raw[searchsortedfirst(raw.opentime,contextstart):searchsortedlast(raw.opentime,enddt),:])
            @assert nrow(df)>warmup "insufficient window/context history for $coin: rows=$(nrow(df)) warmup=$warmup"
        end
        Ohlcv.addpivot!(df)
        histories[coin] = df
        if mode == "market"
            featurestore,targetstore,_ = TradingStrategy.coinstores(rc.clcfg,coin;daterange=rc.daterange)
            @assert !isnothing(featurestore) && !isnothing(targetstore) "missing shared feature/target stores for $coin after ensuring full OHLCV coverage"
            sharedstores[coin] = sharedstorewindow(df,featurestore,targetstore,featurefactory,targetfactory,coin)
        end
    end
    return (;histories,startdt,enddt,regime,folds=TradingStrategy.FoldConfig(),
        featurefactory,targetfactory,mode,modelid=rc.clcfg.model,classbalancing=rc.clcfg.classbalancing,
        context=rc.description,sharedstores,rc)
end

"""Persist private stores, shared regimes, folds and per-coin projected memberships."""
function prepare!(input::NamedTuple, folder::AbstractString)::NamedTuple
    description = "regime-experiment-v3;RG01=$(input.regime);RF01=$(input.folds);start=$(input.startdt);end=$(input.enddt);features=$(Features.configdescription(input.featurefactory()));targets=$(Targets.configdescription(input.targetfactory()));model=$(input.modelid);classbalancing=$(input.classbalancing);mode=$(input.mode);$(input.context)"
    identity = TradingStrategy.experimentmanifest!(folder,input.histories,description)
    stores = TradingStrategy.experimentstores!(input.histories,input.featurefactory,input.targetfactory,folder;
        sharedstores=input.sharedstores)
    eligible = TradingStrategy.experimenttimes(input.histories,stores,input.regime)
    market = TradingStrategy.marketcalendar(input.histories,input.regime;
        startdt=input.startdt,enddt=input.enddt)
    EnvConfig.writetable(market,"market";folderpath=folder,format=:arrow)
    horizon = Targets.lookaheadminutes(input.targetfactory())
    calendar = TradingStrategy.foldcalendar(market,eligible,input.folds;horizonminutes=horizon,enddt=input.enddt)
    sets = Dict(coin=>TradingStrategy.calendarsets(eligible[coin],calendar) for coin in stores.coins)
    EnvConfig.writetable(calendar,"calendar";folderpath=folder,format=:arrow)
    for coin in stores.coins
        EnvConfig.writetable(sets[coin],"sets";folderpath=joinpath(folder,"stores",coin),format=:arrow)
    end
    TradingStrategy.writecalendarcoverage!(folder,sets,eligible,market)
    @assert nrow(calendar)>0 "no shared blocks meet regime/liquidity/length constraints; inspect market.arrow and exclusions.arrow in $folder"
    return (;input,identity,stores,eligible,market,calendar,sets,folder)
end

"""Build immutable train/eval indices and separate compact, memory-mapped sources."""
function trainingsources(prepared::NamedTuple, sets::AbstractDict, folder::AbstractString;
        folds::AbstractVector{Int}=Int[],regimes::AbstractVector{<:AbstractString}=String[],
        trainweights::String="class")::NamedTuple
    labels = Targets.uniquelabels(prepared.input.targetfactory())
    trainindex = TradingStrategy.experimentindex(prepared.stores,sets,labels;
        settype="train",folds,regimes)
    evalindex = TradingStrategy.experimentindex(prepared.stores,sets,labels;
        settype="eval",folds,regimes)
    identity = TradingStrategy.experimentmanifest!(folder,
        Dict("train"=>trainindex,"eval"=>evalindex),
        "$(prepared.identity);training=$trainweights;folds=$folds;regimes=$regimes;eval=equal-cells-within-regimes-v1")
    cols = Features.requestedcolumns(prepared.input.featurefactory())
    train = Classify.CompactMmapSource(prepared.stores.featurestores,cols,trainindex;
        folderpath=mkpath(joinpath(folder,"train")),nlabels=length(labels),classbalancing=prepared.input.classbalancing)
    eval = Classify.CompactMmapSource(prepared.stores.featurestores,cols,evalindex;
        folderpath=mkpath(joinpath(folder,"eval")),nlabels=length(labels),classbalancing=false)
    if trainweights == "balanced"
        train = TradingStrategy.WeightedExperimentSource(train,TradingStrategy.balancedcellweights(trainindex))
    end
    for (name,index) in (("train-index",trainindex),("eval-index",evalindex))
        EnvConfig.writetable(index,name;folderpath=folder,format=:arrow)
    end
    return (;train,eval,trainindex,evalindex,identity,labels,cols,folder)
end

"""Write minute/event curves and readable precision/recall/coverage reports."""
function signalreport!(predictions::DataFrame,folder::AbstractString,stem::String;
        thresholds=0.0:0.05:1.0)::DataFrame
    metrics = Classify.signalmetrics(predictions;thresholds,bootstrap=200)
    events = Classify.signaleventmetrics(predictions;thresholds)
    for table in (metrics,events), name in names(table)
        if eltype(table[!,name])===Missing && !(name in ("regime","foldid"))
            table[!,name] = Vector{Union{Missing,Float64}}(missing,nrow(table))
        end
    end
    EnvConfig.writetable(predictions,stem*"-predictions";folderpath=folder,format=:arrow)
    EnvConfig.writetable(metrics,stem*"-signals";folderpath=folder,format=:arrow)
    EnvConfig.writetable(events,stem*"-events";folderpath=folder,format=:arrow)
    thresholdsummary!(metrics,folder,stem)
    signalplot!(metrics,folder,stem)
    for coin in sort!(unique(predictions.coin))
        coinmetrics = Classify.signalmetrics(predictions[predictions.coin .== coin,:];thresholds)
        EnvConfig.writetable(coinmetrics,stem*"-coin-"*coin;
            folderpath=joinpath(folder,"coin-reports"),format=:arrow)
    end
    open(joinpath(folder,stem*"-report.txt"),"w") do io
        println(io,"Signal opportunities, not economics. allclose = no established trend.")
        println(io,"Thresholds are fixed across regimes; use eval, not test, to choose them.")
        println(io,"Missing metrics/insufficient independent blocks are explicit.")
        show(io,MIME"text/plain"(),metrics;allrows=true,allcols=true,truncate=0)
        println(io,"\nTarget-interval opportunity events:")
        show(io,MIME"text/plain"(),events;allrows=true,allcols=true,truncate=0)
        println(io)
    end
    return metrics
end

"""Persist explicit macro means and non-dominated overall operating points."""
function thresholdsummary!(metrics::DataFrame,folder::AbstractString,stem::String)::Nothing
    macrotable = DataFrame(set=String[],signal=String[],threshold=Float64[],
        cells=Int[],precision_cells=Int[],recall_cells=Int[],
        precision=Union{Missing,Float64}[],recall=Union{Missing,Float64}[],
        coverage_cells=Int[],coverage=Union{Missing,Float64}[])
    cells = filter(row->row.grouping=="regime_fold",metrics)
    for group in groupby(cells,[:set,:signal,:threshold])
        precision = collect(skipmissing(group.precision))
        recall = collect(skipmissing(group.recall))
        coverage = collect(skipmissing(group.coverage))
        push!(macrotable,(first(group.set),first(group.signal),first(group.threshold),nrow(group),
            length(precision),length(recall),
            isempty(precision) ? missing : sum(precision)/length(precision),
            isempty(recall) ? missing : sum(recall)/length(recall),
            length(coverage),isempty(coverage) ? missing : sum(coverage)/length(coverage)))
    end
    EnvConfig.writetable(macrotable,stem*"-macro";folderpath=folder,format=:arrow)
    overall = filter(row->row.grouping=="overall" && !row.insufficient_support &&
        !ismissing(row.precision) && !ismissing(row.recall),metrics)
    keep = trues(nrow(overall))
    for ix in 1:nrow(overall), candidate in 1:nrow(overall)
        overall.set[ix]==overall.set[candidate] && overall.signal[ix]==overall.signal[candidate] || continue
        if overall.precision[candidate]>=overall.precision[ix] &&
                overall.recall[candidate]>=overall.recall[ix] &&
                overall.coverage[candidate]>=overall.coverage[ix] &&
                (overall.precision[candidate]>overall.precision[ix] ||
                 overall.recall[candidate]>overall.recall[ix] ||
                 overall.coverage[candidate]>overall.coverage[ix])
            keep[ix] = false
        end
    end
    EnvConfig.writetable(overall[keep,:],stem*"-candidates";folderpath=folder,format=:arrow)
    return nothing
end

"""Write SVG curves with labeled axes, enlarged hover targets and CSS tooltips."""
function signalplot!(metrics::DataFrame,folder::AbstractString,stem::String)::Nothing
    signals = unique(metrics.signal)
    colors = Dict("overall"=>"#111111","up"=>"#267d36","flat"=>"#666666","down"=>"#b32727")
    open(joinpath(folder,stem*"-curves.svg"),"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"$(length(signals)*280+40)\" viewBox=\"0 0 1000 $(length(signals)*280+40)\">")
        println(io,"<style>.point .tooltip{visibility:hidden;pointer-events:none}.point:hover .tooltip,.point:focus .tooltip{visibility:visible}.point{cursor:crosshair}text{font-family:sans-serif;font-size:12px}</style>")
        println(io,"<rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"20\" y=\"25\">Black overall; green up; gray flat; red down. Hover or Tab to points; labels show selected overall thresholds.</text>")
        for (signalix,signal) in enumerate(signals), (axisix,metric) in enumerate((:precision,:recall))
            left = (axisix-1)*480+60
            top = (signalix-1)*280+60
            println(io,"<text x=\"$left\" y=\"$(top-10)\">$signal: $metric versus coverage</text>")
            println(io,"<path d=\"M $left $top v 200 h 400\" fill=\"none\" stroke=\"#aaa\"/><text x=\"$left\" y=\"$(top+220)\">0</text><text x=\"$(left+390)\" y=\"$(top+220)\">1</text>")
            for tick in (0,0.5,1)
                y = top+200*(1-tick)
                println(io,"<text x=\"$(left-10)\" y=\"$(y+4)\" text-anchor=\"end\">$tick</text>")
            end
            println(io,"<text x=\"$(left+180)\" y=\"$(top+240)\">coverage (0-1)</text>")
            for regime in ("overall","up","flat","down")
                rows = filter(row->row.signal==signal && !ismissing(row[metric]) &&
                    (regime=="overall" ? row.grouping=="overall" :
                        row.grouping=="regime" && row.regime==regime),metrics)
                sort!(rows,:coverage)
                points = join(["$(round(left+400row.coverage;digits=2)),$(round(top+200*(1-row[metric]);digits=2))"
                    for row in eachrow(rows)]," ")
                println(io,"<polyline points=\"$points\" fill=\"none\" stroke=\"$(colors[regime])\"/>")
                for row in eachrow(rows)
                    x = round(left+400row.coverage;digits=2)
                    y = round(top+200*(1-row[metric]);digits=2)
                    detail = "$regime threshold=$(row.threshold) coverage=$(round(row.coverage;digits=4)) $metric=$(round(row[metric];digits=4))"
                    println(io,"<g class=\"point\" tabindex=\"0\" aria-label=\"$detail\"><title>$detail</title><circle cx=\"$x\" cy=\"$y\" r=\"8\" fill=\"transparent\" pointer-events=\"all\"/><circle cx=\"$x\" cy=\"$y\" r=\"2\" fill=\"$(colors[regime])\" pointer-events=\"none\"/><g class=\"tooltip\"><rect x=\"$(left+5)\" y=\"$(top+5)\" width=\"390\" height=\"25\" fill=\"white\" stroke=\"$(colors[regime])\"/><text x=\"$(left+10)\" y=\"$(top+22)\">$detail</text></g></g>")
                    if regime=="overall" && row.threshold in (0.3,0.4,0.5)
                        println(io,"<text x=\"$(x+5)\" y=\"$(y-7)\" pointer-events=\"none\">$(row.threshold)</text>")
                    end
                end
            end
        end
        println(io,"</svg>")
    end
    return nothing
end

"""Train and report on eval only; return the best held-out-loss checkpoint."""
function trainreport!(prepared::NamedTuple, sources::NamedTuple;
        epochs::Int,resume::Bool,reinforce::Int=0)
    if isfile(joinpath(sources.folder,"fixed-threshold.arrow"))
        @warn "adapting a classifier previously used for test metric reporting" folder=sources.folder
    end
    if isfile(joinpath(sources.folder,"test-strategy.arrow"))
        @warn "adapting a classifier previously used for test gain replay" folder=sources.folder
    end
    nn = TradingStrategy.modelconfig(prepared.input.modelid)(length(sources.cols),sources.labels,"regime-$(prepared.input.modelid)")
    result = Classify.traineval!(nn,sources.train,sources.eval;
        identity=sources.identity,folderpath=sources.folder,
        evalweights=Classify.fixedevalweights(sources.evalindex.regime,sources.evalindex.foldid),
        regimes=sources.evalindex.regime,folds=sources.evalindex.foldid,
        maxepochs=epochs,resume,reinforce_epochs=reinforce)
    refreshes = DataFrame(epoch=Int[],interval=Int[],hardcount=Int[])
    for row in result.reinforcement.history
        push!(refreshes,(row.epoch,row.interval,row.hardcount))
    end
    EnvConfig.writetable(refreshes,"reinforcement-refreshes";folderpath=sources.folder,format=:arrow)
    records = DataFrame(result.records)
    # Nested cell diagnostics stay in the checkpoint; Arrow history is scalar.
    EnvConfig.writetable(select(records,Not(:cells)),"epoch-losses";
        folderpath=sources.folder,format=:arrow)
    cells = DataFrame(epoch=Int[],regime=String[],foldid=Int[],samples=Int[],
        natural=Float64[],weighted=Float64[])
    for record in result.records, cell in record.cells
        push!(cells,(record.epoch,cell.regime,cell.foldid,cell.count,cell.natural,cell.weighted))
    end
    EnvConfig.writetable(cells,"epoch-cells";folderpath=sources.folder,format=:arrow)
    prediction = TradingStrategy.experimentpredictions(result.best,sources.evalindex,
        prepared.stores,sources.cols)
    signalreport!(prediction,sources.folder,"eval")
    open(joinpath(sources.folder,"training-report.txt"),"w") do io
        println(io,"Dataset identity: ",prepared.identity)
        println(io,"Train/eval membership identity: ",sources.identity)
        println(io,"Stopping: fixed weighted eval loss; patience=50; relative improvement=0.0005; maximum=200 total epochs.")
        println(io,"Stop reason: ",result.stopreason)
        println(io,"Reinforcement refresh interval: ",reinforce,"; train-only 2x emphasis, normalized within class.")
        println(io,"Best eval epoch: ",result.records[argmin(getproperty.(result.records,:evalweighted))].epoch)
        println(io,"Checkpoint: eval-best.bson; resume: eval-last.bson")
        println(io,"Decision: mutually exclusive argmax label; issue only when score >= fixed threshold; otherwise abstain.")
        for key in (:version,:traincount,:evalcount,:labels,:architecture,:description,:batchsize,:seed)
            println(io,key,": ",getproperty(result.manifest,key))
        end
        println(io,"Eval weight mass: ",sum(result.manifest.weights),
            "; maximum: ",maximum(result.manifest.weights))
        println(io,"Full sample weights/regimes/folds: eval-manifest.bson and eval-index.arrow")
    end
    nextsteps!(prepared,sources)
    return result.best
end

"""Persist and print the manual handoff from eval adaptation to held-out strategy replay."""
function nextsteps!(prepared::NamedTuple,sources::NamedTuple)::Nothing
    rc = prepared.input.rc
    message = """
    Adaptation/eval reporting finished. Reserved test samples have not been used by adaptation.
    1. Inspect $(joinpath(sources.folder,"eval-curves.svg")) and eval opportunity reports.
    2. Set the selected entry thresholds in the registered trading strategy.
       Use the same or a different ts with identical features, targets and classifier configuration.
    3. Run RegimeTraining again with stage=gain, runid=$(rc.runid), and the same
       mode/coinset/daterange/coins/date/detector/weight options. Do not use retrain.
       It loads eval-best.bson and replays only reserved test blocks, never adapts.
    Replay settings are recorded; changed strategy/checkpoint settings warn but do not block replay.
    Experiments informed by test results are not independent held-out assessments.
    """
    write(joinpath(sources.folder,"next-steps.txt"),message)
    println(message)
    return nothing
end

"""Evaluate one selected test threshold, warning on changes and retaining invocation history."""
function testreport!(prepared::NamedTuple, sources::NamedTuple, sets::AbstractDict,
        best, threshold::Float64;folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[])::DataFrame
    index = TradingStrategy.experimentindex(prepared.stores,sets,sources.labels;settype="test",folds,regimes)
    testidentity = TradingStrategy.experimentmanifest!(joinpath(sources.folder,"test-identity"),
        Dict("test"=>index),sources.identity)
    checkpoint = bytes2hex(open(SHA.sha256,joinpath(sources.folder,"eval-best.bson")))
    choice = joinpath(sources.folder,"fixed-threshold.arrow")
    current = DataFrame(threshold=[threshold],identity=[testidentity],checkpoint=[checkpoint])
    if isfile(choice)
        previous = EnvConfig.readtable("fixed-threshold";folderpath=sources.folder,format=:arrow)
        if !isequal(previous,current)
            @warn "test metric threshold or checkpoint changed; results are an experiment, not a fresh held-out assessment" folder=sources.folder
        end
    end
    history = EnvConfig.readtable("test-threshold-history";folderpath=sources.folder,format=:arrow)
    if isnothing(history)
        history = DataFrame(evaluation=Int[],recorded_at=DateTime[],threshold=Float64[],
            identity=String[],checkpoint=String[])
        if isfile(choice)
            previous = EnvConfig.readtable("fixed-threshold";folderpath=sources.folder,format=:arrow)
            push!(history,(1,Dates.now(),only(previous.threshold),only(previous.identity),only(previous.checkpoint)))
        end
    end
    history = vcat(history,DataFrame(evaluation=[nrow(history)+1],recorded_at=[Dates.now()],
        threshold=[threshold],identity=[testidentity],checkpoint=[checkpoint]))
    EnvConfig.writetable(history,"test-threshold-history";folderpath=sources.folder,format=:arrow)
    EnvConfig.writetable(current,"fixed-threshold";folderpath=sources.folder,format=:arrow)
    EnvConfig.writetable(index,"test-index";folderpath=sources.folder,format=:arrow)
    prediction = TradingStrategy.experimentpredictions(best,index,prepared.stores,sources.cols)
    signalreport!(prediction,sources.folder,"test";thresholds=[threshold])
    return prediction
end

"""
Record held-out replay membership, checkpoint digest and strategy settings.
Warn on changed settings without restricting strategy experiments; retain replay history.
"""
function recordteststrategy!(sources::NamedTuple,index::DataFrame,strategy::TradingStrategy.StrategyConfig)::Nothing
    identity = TradingStrategy.experimentmanifest!(joinpath(sources.folder,"test-identity"),
        Dict("test"=>index),sources.identity)
    checkpoint = bytes2hex(open(SHA.sha256,joinpath(sources.folder,"eval-best.bson")))
    description = repr(strategy)
    current = DataFrame(identity=[identity],checkpoint=[checkpoint],strategy=[description])
    path = joinpath(sources.folder,"test-strategy.arrow")
    if isfile(path)
        previous = EnvConfig.readtable("test-strategy";folderpath=sources.folder,format=:arrow)
        if !isequal(previous,current)
            @warn "test replay strategy or checkpoint changed; results are an experiment, not a fresh held-out assessment" folder=sources.folder
        end
    end
    history = EnvConfig.readtable("test-strategy-history";folderpath=sources.folder,format=:arrow)
    if isnothing(history)
        history = DataFrame(replay=Int[],recorded_at=DateTime[],identity=String[],
            checkpoint=String[],strategy=String[])
        if isfile(path)
            previous = EnvConfig.readtable("test-strategy";folderpath=sources.folder,format=:arrow)
            push!(history,(1,Dates.now(),only(previous.identity),only(previous.checkpoint),only(previous.strategy)))
        end
    end
    history = vcat(history,DataFrame(replay=[nrow(history)+1],recorded_at=[Dates.now()],
        identity=[identity],checkpoint=[checkpoint],strategy=[description]))
    EnvConfig.writetable(history,"test-strategy-history";folderpath=sources.folder,format=:arrow)
    EnvConfig.writetable(current,"test-strategy";folderpath=sources.folder,format=:arrow)
    return nothing
end

"""Replay reserved test predictions and target references using the selected strategy."""
function gainreport!(prepared::NamedTuple,sources::NamedTuple,best;
        folds::AbstractVector{Int}=Int[],regimes::AbstractVector{<:AbstractString}=String[])::Nothing
    rc = prepared.input.rc
    index = TradingStrategy.experimentindex(prepared.stores,prepared.sets,sources.labels;
        settype="test",folds,regimes)
    strategy = TradingStrategy.strategytemplate(rc.tscfg.strategy)
    recordteststrategy!(sources,index,strategy)
    EnvConfig.writetable(index,"test-index";folderpath=sources.folder,format=:arrow)
    if rc.mode=="market"
        EnvConfig.setlogpath(rc.folder)
        gainfolder = EnvConfig.logfolder()
    else
        gainfolder = mkpath(joinpath(prepared.folder,"gain",rc.folder))
    end
    classifier = Classify.TrendClassifier001(best;featconfig=prepared.input.featurefactory,
        targetconfig=prepared.input.targetfactory)
    ts = TradingStrategy.TsCache(;classifier,strategy,source=TradingStrategy.tsconfigsource(rc.tscfg))
    xc = Xch.XchCache(Bybit.BybitCache();startdt=prepared.input.startdt,enddt=prepared.input.enddt)
    TSM.ensuretradesschema!(xc.tsm,TSM.tradesdf_all_contributors())
    predictions = TradingStrategy.experimentpredictions(best,index,prepared.stores,sources.cols)
    EnvConfig.writetable(predictions,"test-predictions";folderpath=sources.folder,format=:arrow)
    snapshots = [:opentime,:pair,:set,:rangeid,:lastopentrade,:close,:high,:low,:label,:score,
        :lp_amount,:sp_amount,:lol_pavg,:lcl_pavg,:sol_pavg,:scl_pavg,:lo_limit,:lc_limit,:so_limit,:sc_limit]
    gainparts = DataFrame[]
    tradesfolder = mkpath(joinpath(gainfolder,"trades-regime"))
    for coin in sort!(unique(predictions.coin))
        history = prepared.input.histories[coin]
        rows = predictions[predictions.coin .== coin,:]
        results = innerjoin(rows,select(history,:opentime,:high,:low,:close,:pivot);on=:opentime)
        sort!(results,:opentime)
        # Sparse eligibility and block boundaries are independent replay ranges.
        rangeid = cumsum(vcat(1,[results.blockid[ix]!=results.blockid[ix-1] ||
            results.opentime[ix]-results.opentime[ix-1]!=Minute(1) for ix in 2:nrow(results)]))
        results[!,:rangeid] = rangeid
        ohlcv = Ohlcv.defaultohlcv(coin)
        Ohlcv.setdataframe!(ohlcv,copy(history))
        TradingStrategy.addclassifierbase!(ts,ohlcv;datetime=first(results.opentime))
        for (pass,predicted) in (("predicted",true),("truth",false))
            parts = DataFrame[]
            for group in groupby(results,[:set,:rangeid])
                metadata = Dict{Symbol,Any}(:set=>first(group.set),:rangeid=>first(group.rangeid),
                    :freequote=>strategy.maxbudgetquote,:freemargin=>strategy.maxbudgetquote)
                scores = predicted ? group.score : fill(1f0,nrow(group))
                labels = predicted ? group.label : group.target
                tp = TradingStrategy.preparereplaytrades!(ts,xc,coin,group,scores,collect(labels);
                    metadata,datetime=last(group.opentime))
                TradingStrategy.processreplaygains!(tp;strategy)
                push!(parts,select(tp.tradesdf,snapshots))
                TradingStrategy.droppair!(ts,tp.pair)
                TSM.droppair!(xc.tsm,tp.pair)
            end
            trades = vcat(parts...)
            EnvConfig.savedf(trades,"$coin-$pass";folderpath=tradesfolder)
            gains = TSM.compilegains(trades;setpartitions=true)
            if nrow(gains)>0
                gains[!,:predicted] = fill(predicted,nrow(gains))
                push!(gainparts,gains)
            end
        end
        TradingStrategy.dropbase!(ts,coin)
    end
    if isempty(gainparts)
        @warn "gain replay produced no completed trades" tradesfolder
        EnvConfig.writetable(DataFrame(completed_trades=[0]),"gain-summary";
            folderpath=gainfolder,format=:arrow)
    else
        gains = TSM.sortgainsdf!(vcat(gainparts...))
        EnvConfig.savedf(gains,"tsmgains-regime";folderpath=gainfolder)
        TSM.gainsreport(instem="tsmgains-regime",stem="xchgainsreport-regime",folderpath=gainfolder)
        EnvConfig.writetable(DataFrame(completed_trades=[nrow(gains)]),"gain-summary";
            folderpath=gainfolder,format=:arrow)
    end
    EnvConfig.writetable(DataFrame(identity=[sources.identity],ts=[rc.tscfg.configname],
        strategy=[rc.tscfg.strategy],population=["test"],checkpoint=["eval-best.bson"]),
        "gain-identity";folderpath=gainfolder,format=:arrow)
    return nothing
end

"""Execute registered pooled training/replay or independent chronological models."""
function execute!(prepared::NamedTuple;stage::String,epochs::Int,resume::Bool,
        threshold::Union{Nothing,Float64},folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[],trainweights::String="class",reinforce::Int=0)::Nothing
    if stage != "walkforward"
        sources = trainingsources(prepared,prepared.sets,joinpath(prepared.folder,"balanced");
            folds,regimes,trainweights)
        if stage == "test"
            best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
            testreport!(prepared,sources,prepared.sets,best,threshold;folds,regimes)
        elseif stage=="gain"
            @assert !resume "gain cannot resume adaptation"
            best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
            gainreport!(prepared,sources,best;folds,regimes)
        else
            @assert reinforce==0 || resume || !isfile(joinpath(sources.folder,"eval-best.bson")) "reinforce=$reinforce requires retrain to adapt an existing checkpoint"
            adapted = resume || !isfile(joinpath(sources.folder,"eval-best.bson"))
            if adapted
                best = trainreport!(prepared,sources;epochs,resume,reinforce)
            else
                best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
            end
            if !adapted && stage in ("predict","evaluate")
                prediction = TradingStrategy.experimentpredictions(best,sources.evalindex,prepared.stores,sources.cols)
                if stage=="predict"
                    EnvConfig.writetable(prediction,"eval-predictions";folderpath=sources.folder,format=:arrow)
                else
                    signalreport!(prediction,sources.folder,"eval")
                end
            end
            adapted || nextsteps!(prepared,sources)
        end
        return nothing
    end
    @assert isempty(folds) && isempty(regimes) "walk-forward uses chronological windows, not diagnostic subsets"
    synthetic = prepared.input.mode=="synthetic"
    schedule = TradingStrategy.walkforwardschedule(prepared.input.startdt,prepared.input.enddt;
        trainminutes=synthetic ? 6000 : 2*365*1440,
        evalminutes=synthetic ? 3000 : 90*1440,
        testminutes=synthetic ? 3000 : 90*1440,
        gapminutes=prepared.input.folds.gapminutes)
    @assert nrow(schedule)>0 "insufficient history for chronological schedule"
    EnvConfig.writetable(schedule,"walkforward-schedule";folderpath=prepared.folder,format=:arrow)
    tests = DataFrame[]
    for window in eachrow(schedule)
        folder = joinpath(prepared.folder,"walkforward-$(window.walkforwardid)")
        sets = TradingStrategy.chronologicalmemberships(prepared,window)
        sources = trainingsources(prepared,sets,folder;trainweights)
        best = trainreport!(prepared,sources;epochs,resume,reinforce)
        push!(tests,testreport!(prepared,sources,sets,best,threshold))
    end
    combined = vcat(tests...)
    @assert allunique(zip(combined.coin,combined.opentime)) "repeated test samples in chronological aggregate"
    signalreport!(combined,prepared.folder,"walkforward-test";thresholds=[threshold])
    return nothing
end

"""Run the selected experiment stage with strict identities and explicit test thresholds."""
function main(args::Vector{String}=ARGS)::Nothing
    if args in (["help"],["--help"],["-h"])
        println(helptext())
        return nothing
    end
    opts = options(args)
    EnvConfig.init(haskey(opts,"train") ? EnvConfig.training : EnvConfig.test)
    EnvConfig.setpairquote!("USDT")
    EnvConfig.setdfformat!(:arrow)
    rc = buildrunconfig(opts)
    ENV["CTS_RUN_ID"] = rc.runid
    input = inputs(opts,rc)
    folder = joinpath(EnvConfig.defaultdebugfilespath,"regime-training",rc.runid)
    prepared = prepare!(input,folder)
    stage,epochs = rc.stage,rc.epochs
    if stage == "stores"
        println("Calendar and coverage: $folder")
        return nothing
    end
    if stage in ("test","walkforward")
        @assert haskey(opts,"threshold") "choose a fixed eval-based threshold before stage=$stage"
    end
    threshold = haskey(opts,"threshold") ? parse(Float64,opts["threshold"]) : nothing
    @assert isnothing(threshold) || 0 <= threshold <= 1 "invalid fixed threshold=$threshold"
    @assert isnothing(threshold) || stage in ("test","walkforward") "threshold is only supported for test/walkforward, stage=$stage; gain uses registered strategy thresholds"
    folds = haskey(opts,"folds") ? parse.(Int,split(opts["folds"],",")) : Int[]
    regimes = haskey(opts,"regimes") ? collect(split(opts["regimes"],",")) : String[]
    @assert all(in(["up","flat","down"]),regimes) "invalid selected regimes=$regimes"
    trainweights = get(opts,"trainweights","class")
    @assert trainweights in ("class","balanced") "invalid training weights=$trainweights"
    execute!(prepared;stage,epochs,resume=rc.resume,threshold,folds,regimes,trainweights,reinforce=rc.reinforce)
    println("Experiment artifacts: $folder")
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        RegimeTraining.main()
    catch error
        if error isa InterruptException
            println(stderr,"\nInterrupted; completed checkpoints retained. Continue with retrain and the same runid.")
            exit(130)
        end
        rethrow()
    end
end
