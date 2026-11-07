"""
Terminal runner for regime/fold-aware SimpleChains experiments.
Uses registered trading strategy, coin-set and date-range configurations.
"""
module RegimeTraining

using Dates, DataFrames
import SHA
import EnvConfig, TradingStrategy, Features, Targets, Ohlcv, TestOhlcv, Classify, Xch, Bybit, TSM

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1


"""Print and flush a timestamped phase message, including monotonic elapsed time when supplied."""
function progressmessage(message::AbstractString; started::Union{Nothing,UInt64}=nothing,
        io::IO=stdout)::Nothing
    elapsed = isnothing(started) ? "" : " elapsed=$(round((time_ns()-started)/1e9;digits=1))s"
    println(io,"$(EnvConfig.now()) RegimeTraining $message$elapsed")
    flush(io)
    return nothing
end

"""Keep persistent market artifacts under logs and test/synthetic artifacts under debug."""
function experimentfolder(rc::NamedTuple)::String
    root = rc.mode=="market" ? EnvConfig.defaultlogfilespath : EnvConfig.defaultdebugfilespath
    return joinpath(root,"regime-training",rc.runid)
end

"""Return documented CLI usage; test thresholds must be supplied explicitly."""
function helptext()::String
    return """
    julia --project=. scripts/RegimeTraining.jl [test|train] [retrain] [key=value ...]
      test                       default; registered CS00/D00 artificial coins
      train                      registered CS01/D01 local Bybit USDT history through 2026-11-02T22:00:00
      ts=<id>                    default 063; registered features/targets/model/B flag
      coinset=<CSxx> daterange=<Dxx>  defaults from EnvConfig mode
      coins=<CSV>                override actual training/eval/replay universe
      startdt=<ISO> enddt=<ISO>   together override experiment dates, not just replay
      runid=<immutable-id>        optional; deterministic configuration-derived default
      folder=<name>              gain replay log subfolder
      stage=stores|train|predict|evaluate|snapshot|gain|test|walkforward  default evaluate
      epochs=<1..200>             default coin-set budget, total including resume
      retrain                    resume last checkpoint; warns after test exposure
      reinforce=<Int>            refresh train misclassification flags every N epochs; 0 off
      mode=synthetic             explicit small SINE/DOUBLESINE smoke fixture
      folds=<comma-separated IDs> optional diagnostic fold subset
      regimes=up,flat,down        optional diagnostic regime subset
      trainweights=class|balanced default class; balanced adds regime/fold weighting
      threshold=<0..1>            required for test/walkforward; fixed across regimes
      mincoins=<Int> weightcap=<0..1>  explicit detector overrides, both required

    Stage explanations and artifacts:
      Output roots:
        Experiment: ~/crypto/logs/regime-training/<runid> in train/market mode;
                    ~/crypto/debug/regime-training/<runid> in test/synthetic mode.
        Model artifacts: <experiment>/balanced (also with trainweights=class).
        Gain outputs: ~/crypto/logs/<folder> in market mode;
                      <experiment>/gain/<folder> otherwise.
        Default folder: RegimeTraining-<runid>. Use separate folders for strategy comparisons.
        Arrow files are data tables; TXT and SVG files are directly readable.

      stores
        Prepare/verify histories, shared stores, eligible timestamps and assigned blocks.
        Writes <experiment>/manifest.arrow, input-identities.arrow, market.arrow,
        calendar.arrow, sample-coverage.arrow and exclusions.arrow.
        Market mode writes eligible/<coin>.arrow and refreshes/reuses shared full-history
        feature/target stores, without private copies or per-sample membership files.
        Synthetic/test mode writes stores/<coin>/{features,targets,sets}.arrow.
        Does not train or infer.
        Preparation is also performed by train/predict/evaluate/test/walkforward.

      train
        Adapt a missing classifier, or resume with retrain; otherwise reuse its checkpoint.
        Creates/verifies model manifest and train/eval/test-index.arrow, plus compact
        numeric sources in train/ and eval/ (also test/ in market mode).
        Market mode compiles all three directly from shared stores in one coin pass.
        Adaptation writes eval-best.bson,
        eval-last.bson, eval-manifest.bson, epoch-losses.arrow,
        reinforcement-refreshes.arrow and training-report.txt.
        Adaptation also generates the eval report family below; next-steps.txt
        explains the handoff to snapshot/gain. Reusing a checkpoint does not
        regenerate eval reports in this stage. Non-synthetic runs publish the
        best classifier to its configured classifier folder.

      predict
        Infer eval samples from the saved best checkpoint; writes eval-predictions.arrow.
        If no checkpoint exists (or retrain is given), first performs adaptation
        and produces the full training/eval artifacts described above.

      evaluate (default)
        Infer and assess eval signal opportunities over a threshold sweep.
        Uses the best checkpoint, adapting first if missing or retrain is given.
        Eval report family in the model folder:
          eval-predictions.arrow, eval-signals.arrow, eval-events.arrow,
          eval-macro.arrow, eval-candidates.arrow, eval-report.txt,
          eval-curves.svg and coin-reports/eval-coin-<coin>.arrow.
        These assess signals, not economic gains; use eval to choose thresholds.

      snapshot
        One-time seal of the original reserved test inputs; requires saved preparation
        and training-selection artifacts. Verifies original history and test coverage,
        then freezes test features, prices and regression guidance.
        Writes model/gain-snapshot/<coin>.arrow, coins.arrow and manifest.arrow
        (completion seal), plus model/test-identity/ provenance.
        Does not adapt or rebuild train/eval numeric sources. Changed history coverage fails;
        existing seals cannot be overwritten. Later strategies may only request
        regression windows included in the seal.

      gain
        Requires eval-best.bson and a sealed snapshot. Never adapts; infers/reuses
        predictions and replays predicted and target-reference test signals separately.
        Writes model/test-predictions.arrow, test-strategy.arrow and
        test-strategy-history.arrow. Verified prediction and coin/pass replay caches
        live under model/gain-cache/, with digest completion markers.
        Gain output folder contains trades-regime/<coin>-{predicted,truth}.arrow,
        tsmgains-regime.arrow (completed trades), xchgainsreport-regime.arrow
        (combined aggregate), gain-summary.arrow and gain-identity.arrow.
        gain-report.txt and gain-report.svg separate predicted/reference results;
        SVG shows average completed-trade gain (%), not portfolio return.
        Cache reruns regenerate reports; empty results explicitly say no completed trades.

      test
        Requires a saved best checkpoint and threshold=<eval-selected value>.
        Assesses reserved test signals without adaptation. Writes fixed-threshold.arrow,
        test-threshold-history.arrow, test-identity/ provenance, test-index.arrow
        and compact test/ sources in the model folder.
        Generates the same report family as evaluate with the test- prefix and only
        the supplied threshold: test-predictions/signals/events/macro/candidates.arrow,
        test-report.txt, test-curves.svg and coin-reports/test-coin-<coin>.arrow.
        Does not simulate economic gains or create a gain snapshot.

      walkforward
        Train and assess successive chronological train/eval/test windows using
        threshold=<fixed value>; diagnostic folds/regimes subsets are not supported.
        Writes <experiment>/walkforward-schedule.arrow and per-window training,
        checkpoint, eval and test artifacts under walkforward-<id>/.
        Combined test reports use the walkforward-test- prefix at the experiment root,
        including predictions/signals/events/macro/candidates.arrow, report.txt,
        curves.svg and per-coin reports. Does not perform gain replay.

    Training outputs eval threshold curves without using test for selection.
    Curve columns: precision/PPV, recall, NPV (= TN / (TN + FN)), all versus coverage.
    Run test only after choosing a fixed threshold from eval reports.
    Market mode refreshes/reuses shared full-history feature and target stores, then selects the experiment window.
    Input and index identity checks cover configuration, row counts and timestamp bounds,
    not interior data changes. Regenerate artifacts after changing data within those bounds.
    Coins with insufficient window/context history are warned about and skipped.
    No usable histories or shared training blocks still cause an explicit failure.
    Detector RF01/RG01 settings are recorded in the immutable manifest; no automatic calibration.
    Adaptation ends with eval diagrams and next-step guidance.
    Gain requires an existing checkpoint; it never trains and replays reserved test only.
    Run stage=snapshot once to seal existing test inputs against the original experiment.
    Gain reads the sealed snapshot only, not current market histories, and verifies digests.
    Completed predictions and coin/pass replays are reused only with matching identities.
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
    @assert stage in ("stores","train","predict","evaluate","snapshot","gain","test","walkforward") "invalid stage=$stage"
    epochs = parse(Int,get(opts,"epochs",string(cs.maxepochs)))
    @assert 0<epochs<=200 "invalid epoch budget=$epochs"
    reinforce = parse(Int,get(opts,"reinforce","0"))
    @assert reinforce>=0 "reinforce=$reinforce must be nonnegative"
    identityopts = sort([key*"="*value for (key,value) in opts
        if !(key in ("test","train","ts","runid","folder","stage","epochs","retrain","threshold","reinforce"))])
    description = "classifier=$(tscfg.classifier);coinset=$coinset;daterange=$daterange;mode=$mode;coins=$(join(coins,","));start=$startdt;end=$enddt;$(join(identityopts,";"))"
    scheme = mode=="market" ? "shared-blocks-v1;" : "coverage-v1;"
    runid = get(opts,"runid","$(tscfg.classifier)-$coinset-$daterange-"*TradingStrategy.confighash(scheme*description)[1:12])
    folder = get(opts,"folder","RegimeTraining-$runid")
    @assert occursin(r"^[A-Za-z0-9_-]+$",folder) "invalid folder=$folder"
    @assert !(stage in ("gain","snapshot")) || !haskey(opts,"retrain") "$stage does not adapt a checkpoint; retrain is not permitted"
    @assert !(stage in ("gain","snapshot")) || reinforce==0 "$stage does not train; reinforce=$reinforce belongs to an adaptation run"
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

"""Read one market window without retaining the complete raw OHLCV table."""
@noinline function marketinputhistory(coin::AbstractString, startdt::DateTime,
        enddt::DateTime, warmup::Int)::Union{Nothing,DataFrame}
    raw = EnvConfig.readtable("ohlcv";folderpath=EnvConfig.coinfolderpath(coin,"USDT"),format=:arrow)
    @assert !isnothing(raw) "missing OHLCV for $coin"
    return TradingStrategy.regimehistory(raw,coin,startdt,enddt,warmup)
end

"""Load one experiment-window OHLCV history, allowing released histories to be fetched on demand."""
function loadinputhistory(input::NamedTuple,coin::AbstractString)::DataFrame
    haskey(input.histories,coin) && return input.histories[coin]
    if input.mode=="synthetic"
        @assert coin in ("SINE","DOUBLESINE") "unknown synthetic coin=$coin"
        phase = coin=="SINE" ? 1 : 10.5
        return TestOhlcv.sinedata(2400,24000,0,phase;firstutc=input.startdt)
    end
    warmup = TradingStrategy.regimewarmup(input.featurefactory,input.targetfactory,input.regime)
    contextstart = input.startdt-Minute(warmup+1)
    if input.mode=="test"
        @assert coin in TradingStrategy.coinsetconfig("CS00").coins "test mode requires CS00 artificial symbols, got=$coin"
        minutes = div(Dates.value(input.enddt-contextstart),60000)
        df = TestOhlcv.sinedata(120,minutes,0,coin=="SINE" ? 1 : 10.5;
            firstutc=contextstart)
    else
        df = marketinputhistory(coin,input.startdt,input.enddt,warmup)
        @assert !isnothing(df) "previously accepted replay history is now insufficient for $coin: warmup=$warmup"
    end
    Ohlcv.addpivot!(df)
    return df
end

"""Prepare artificial OHLCV or market configuration; market histories are loaded during preparation."""
function inputs(opts::Dict{String,String}, rc::NamedTuple=buildrunconfig(opts); saved::Bool=false)::NamedTuple
    mode = rc.mode
    histories = Dict{String,DataFrame}()
    coverage = DataFrame(coin=String[],rows=Int[],start=DateTime[],stop=DateTime[])
    if mode == "synthetic"
        startdt = DateTime(2025,1,1)
        # Lower frequency than the live simulation fixture makes sustained regimes
        # observable while still using TestOhlcv's SINE/DOUBLESINE generator.
        histories["SINE"] = TestOhlcv.sinedata(2400,24000,0,1;firstutc=startdt)
        histories["DOUBLESINE"] = TestOhlcv.sinedata(2400,24000,0,10.5;firstutc=startdt)
        for coin in sort!(collect(keys(histories)))
            push!(coverage,TradingStrategy.experimentcoverage(coin,histories[coin]))
        end
        enddt = last(histories["SINE"].opentime)
        liquidity = (minquotevol=5000f0,accumulate=15,checkperiod=60,startthreshold=0.05,
            stopthreshold=0.25,minliquidminutes=120,startdistance=120)
        regime = TradingStrategy.RegimeConfig(longminutes=240,shortminutes=60,
            stepminutes=30,mincoins=2,weightcap=0.5,threshold=0.0025,
            strength=0.0,confirmation=2,liquidity=liquidity)
        folds = TradingStrategy.FoldConfig(anchor=startdt,foldminutes=6000,
            blockminutes=360,gapminutes=60,minminutes=120)
        return (;histories,coverage,coins=sort!(collect(keys(histories))),startdt,enddt,regime,folds,
            featurefactory=syntheticfeatures,targetfactory=synthetictarget,mode,
            modelid=rc.clcfg.model,classbalancing=rc.clcfg.classbalancing,context=rc.description,
            sharedstores=nothing,rc)
    end
    startdt,enddt = rc.startdt,rc.enddt
    EnvConfig.setcoinspath!("Bybit")
    sharedstores = nothing
    acceptedcoins = String[]
    regime = mode=="test" ? TradingStrategy.RegimeConfig(mincoins=2,weightcap=0.5) : TradingStrategy.RegimeConfig()
    if haskey(opts,"mincoins")
        regime = TradingStrategy.RegimeConfig(mincoins=parse(Int,opts["mincoins"]),
            weightcap=parse(Float64,opts["weightcap"]))
    end
    TradingStrategy.checkregimeconfig(regime)
    featurefactory = TradingStrategy.featurefactory(rc.clcfg.features)
    targetfactory = ()->TradingStrategy.targetconfig(rc.clcfg.targets)
    warmup = TradingStrategy.regimewarmup(featurefactory,targetfactory,regime)
    contextstart = startdt-Minute(warmup+1)
    if saved || mode=="market"
        return (;histories,coverage,coins=sort(rc.coins),startdt,enddt,regime,folds=TradingStrategy.FoldConfig(),
            featurefactory,targetfactory,mode,modelid=rc.clcfg.model,
            classbalancing=rc.clcfg.classbalancing,context=rc.description,sharedstores,rc)
    end
    started = time_ns()
    progressmessage("checking history: coins=$(length(rc.coins)) range=$startdt to $enddt")
    for (coinix,coin) in enumerate(rc.coins)
        progressmessage("checking history $coin ($coinix/$(length(rc.coins)))";started)
        if mode=="test"
            @assert coin in TradingStrategy.coinsetconfig("CS00").coins "test mode requires CS00 artificial symbols, got=$coin"
            minutes = div(Dates.value(enddt-contextstart),60000)
            df = TestOhlcv.sinedata(120,minutes,0,coin=="SINE" ? 1 : 10.5;firstutc=contextstart)
        else
            df = marketinputhistory(coin,startdt,enddt,warmup)
            isnothing(df) && continue
        end
        push!(acceptedcoins,coin)
        push!(coverage,TradingStrategy.experimentcoverage(coin,df))
        if mode=="test"
            Ohlcv.addpivot!(df)
            histories[coin] = df
        end
        df = nothing
        GC.gc()
    end
    @assert !isempty(acceptedcoins) "no coins satisfy history warmup=$warmup for $startdt to $enddt; requested=$(rc.coins)"
    sort!(acceptedcoins)
    progressmessage("history check complete: accepted=$(length(acceptedcoins))/$(length(rc.coins))";started)
    if mode == "market"
        println("RegimeTraining accepted history: ",repr(acceptedcoins))
    end
    return (;histories,coverage,coins=acceptedcoins,startdt,enddt,regime,folds=TradingStrategy.FoldConfig(),
        featurefactory,targetfactory,mode,modelid=rc.clcfg.model,classbalancing=rc.clcfg.classbalancing,
        context=rc.description,sharedstores,rc)
end

"""Describe the immutable regime experiment independently of its storage lifecycle."""
function preparationdescription(input::NamedTuple)::String
    scheme = input.mode=="market" ? "regime-experiment-v5-shared-blocks" : "regime-experiment-v4-coverage"
    return "$scheme;RG01=$(input.regime);RF01=$(input.folds);start=$(input.startdt);end=$(input.enddt);features=$(Features.configdescription(input.featurefactory()));targets=$(Targets.configdescription(input.targetfactory()));model=$(input.modelid);classbalancing=$(input.classbalancing);mode=$(input.mode);$(input.context)"
end

"""Map a coin's complete shared stores with the configured feature and target identities."""
function inputfullstores(input::NamedTuple, coin::AbstractString)::NamedTuple
    clcfg = input.rc.clcfg
    features = Features.readfeatures(coin,"USDT",clcfg.features;
        confighash=TradingStrategy.featureshash(clcfg.features))
    targets = Targets.readtargets(coin,"USDT",clcfg.targets;
        confighash=TradingStrategy.targetshash(clcfg.targets))
    @assert !isnothing(features) && !isnothing(targets) "missing shared stores for $coin features=$(clcfg.features) targets=$(clcfg.targets)"
    return (;features,targets)
end

"""Read aligned shared windows without letting full-history store mappings escape."""
@noinline function inputsharedstores(input::NamedTuple, coin::AbstractString,
        history::DataFrame)::Tuple{DataFrame,DataFrame}
    (;features,targets) = inputfullstores(input,coin)
    return TradingStrategy.sharedstorewindow(history,features,targets,
        input.featurefactory,input.targetfactory,coin)
end

"""Load, accept and prepare one coin, retaining only coverage and hourly observations."""
@noinline function prepareinputcoin!(input::NamedTuple, coin::AbstractString, historyloader::Function,
        folder::AbstractString, scratch::AbstractString, times::AbstractVector{DateTime};
        coinix::Int, ncoins::Int)::Union{Nothing,NamedTuple}
    history = historyloader(coin)
    isnothing(history) && return nothing
    actual = TradingStrategy.experimentcoverage(coin,history)
    GC.gc()
    sharedstores = nothing
    if input.mode=="market"
        TradingStrategy.ensurefeaturetargetstores!(input.rc.clcfg,[coin];
            progressoffset=coinix-1,progresstotal=ncoins)
        GC.gc()
    end
    progressmessage("checking liquidity $coin ($coinix/$ncoins)")
    liquid = TradingStrategy.regimeliquidity(history,input.regime)
    if input.mode=="market"
        sharedstores = Dict(coin=>inputsharedstores(input,coin,history))
        GC.gc()
    end
    histories = Dict(coin=>history)
    if input.mode=="market"
        features,targets = sharedstores[coin]
        stores = (coins=[coin],featurestores=[features],targetstores=[targets])
    else
        progressmessage("preparing private stores $coin ($coinix/$ncoins): history rows=$(nrow(history))")
        stores = TradingStrategy.experimentstores!(histories,input.featurefactory,
            input.targetfactory,folder;sharedstores)
    end
    progressmessage("calculating eligible minutes $coin ($coinix/$ncoins)")
    eligible = TradingStrategy.experimenttimes(histories,stores,input.regime;
        liquidity=Dict(coin=>liquid))
    EnvConfig.writetable(DataFrame(opentime=eligible[coin]),coin;folderpath=scratch,format=:arrow)
    progressmessage("persisted eligible minutes $coin: rows=$(length(eligible[coin]))")
    stores = nothing
    sharedstores = nothing
    eligible = nothing
    GC.gc()
    progressmessage("calculating hourly observations $coin ($coinix/$ncoins)")
    return (coverage=actual,observation=TradingStrategy.regimeobservations(history,times,input.regime;liquid))
end

"""Read a required persisted experiment table; missing preparation data is an error."""
function readpreparedtable(folder::AbstractString, stem::AbstractString)::DataFrame
    table = EnvConfig.readtable(stem;folderpath=folder,format=:arrow)
    @assert !isnothing(table) "missing prepared table=$stem in $folder"
    return table
end

"""Construct the full-universe market calendar from appended mmap-backed observations."""
@noinline function preparedmarket(scratch::AbstractString, ranges::Vector{UnitRange{Int}},
        input::NamedTuple)::DataFrame
    observations = EnvConfig.readtable("observations";folderpath=scratch,format=:arrow,copycols=false)
    @assert !isnothing(observations) "missing hourly observations in $scratch"
    @assert nrow(observations)==last(last(ranges)) "hourly observation rows=$(nrow(observations)) != appended ranges=$ranges"
    return TradingStrategy.marketcalendar([view(observations,rows,:) for rows in ranges],input.regime;
        startdt=input.startdt,enddt=input.enddt)
end

"""Persist one coin's projected sets without retaining other coins' minute-level assignments."""
@noinline function persistinputsets!(coin::AbstractString, folder::AbstractString,
        timesloader::Function, calendar::DataFrame)::Nothing
    sets = TradingStrategy.calendarsets(timesloader(coin),calendar)
    EnvConfig.writetable(sets,"sets";folderpath=joinpath(folder,"stores",coin),format=:arrow)
    return nothing
end

"""
Accept and prepare histories in one coin loop, staging private stores before coverage verification.
Shared hourly observations
and eligible minutes are disk-backed until the full-universe calendar and coverage are built.
Only coin metadata and the small market/calendar tables survive preparation.
"""
function preparestreamed!(input::NamedTuple, folder::AbstractString;
        historyloader::Function=coin -> input.mode=="market" ?
            marketinputhistory(coin,input.startdt,input.enddt,
                TradingStrategy.regimewarmup(input.featurefactory,input.targetfactory,input.regime)) :
            loadinputhistory(input,coin))::NamedTuple
    coins = input.coins
    started = time_ns()
    return mktempdir(mkpath(folder)) do scratch
        stagedfolder = mkpath(joinpath(scratch,"prepared"))
        coverage = DataFrame(coin=String[],rows=Int[],start=DateTime[],stop=DateTime[])
        acceptedcoins = String[]
        progressmessage("preparing per-coin stores and observations: scratch=$scratch")
        times = collect(ceil(input.startdt,Minute(input.regime.stepminutes)):
            Minute(input.regime.stepminutes):input.enddt)
        ranges = UnitRange{Int}[]
        rows = 0
        for (coinix,coin) in enumerate(coins)
            coinstarted = time_ns()
            progressmessage("loading preparation history $coin ($coinix/$(length(coins)))")
            result = prepareinputcoin!(input,coin,historyloader,stagedfolder,scratch,times;
                coinix,ncoins=length(coins))
            if isnothing(result)
                GC.gc()
                continue
            end
            push!(acceptedcoins,coin)
            push!(coverage,result.coverage)
            EnvConfig.appenddf(result.observation,"observations";folderpath=scratch)
            push!(ranges,(rows+1):(rows+nrow(result.observation)))
            rows += nrow(result.observation)
            result = nothing
            GC.gc()
            progressmessage("prepared $coin ($coinix/$(length(coins))): cumulative hourly rows=$rows";started=coinstarted)
        end
        @assert !isempty(acceptedcoins) "no coins have sufficient history for $(input.startdt) to $(input.enddt); requested=$coins"
        coins = acceptedcoins
        input = merge(input,(coins=coins,coverage=coverage))
        identity = TradingStrategy.experimentmanifest!(folder,coverage,preparationdescription(input))
        for coin in coins
            if input.mode=="market"
                destination = mkpath(joinpath(folder,"eligible"))
                mv(joinpath(scratch,coin*".arrow"),joinpath(destination,coin*".arrow");force=true)
            else
                destination = mkpath(joinpath(folder,"stores",coin))
                for stem in ("features","targets")
                    mv(joinpath(stagedfolder,"stores",coin,stem*".arrow"),
                        joinpath(destination,stem*".arrow");force=true)
                end
            end
        end
        progressmessage("history preparation complete: accepted=$(length(coins)) input identity=$identity";started)
        input.mode=="market" && println("RegimeTraining accepted history: ",repr(coins))
        progressmessage("building full-universe market calendar: coins=$(length(coins)) hourly rows=$rows";started)
        market = preparedmarket(scratch,ranges,input)
        progressmessage("market calendar complete: rows=$(nrow(market))";started)
        GC.gc()
        total = unknown = 0
        timesfolder = input.mode=="market" ? joinpath(folder,"eligible") : scratch
        @assert nrow(market)>1 "insufficient market observations=$(nrow(market))"
        step = market.opentime[2]-market.opentime[1]
        timesloader = coin -> begin
            progressmessage("loading eligible minutes for shared folds: $coin")
            timestamps = readpreparedtable(timesfolder,coin).opentime
            total += length(timestamps)
            unknown += count(timestamps) do timestamp
                ix = searchsortedlast(market.opentime,timestamp)
                ix>0 && timestamp<last(market.opentime)+step && market.regime[ix]=="unknown"
            end
            timestamps
        end
        horizon = Targets.lookaheadminutes(input.targetfactory())
        progressmessage("balancing shared folds";started)
        calendar = TradingStrategy.foldcalendar(market,timesloader,coins,input.folds;
            horizonminutes=horizon,enddt=input.enddt)
        EnvConfig.writetable(market,"market";folderpath=folder,format=:arrow)
        EnvConfig.writetable(calendar,"calendar";folderpath=folder,format=:arrow)
        progressmessage("persisted market and calendar: blocks=$(nrow(calendar)) output=$folder";started)
        if input.mode=="market"
            counts = combine(groupby(calendar,[:set,:regime,:foldid]),:samples=>sum=>:samples)
            counts = counts[counts.samples .> 0,:]
            if nrow(counts)>0
                EnvConfig.writetable(counts,"sample-coverage";folderpath=folder,format=:arrow)
            end
            used = sum(calendar.samples)
            excluded = DataFrame(eligible_samples=[total],assigned_samples=[used],
                excluded_samples=[total-used],unknown_samples=[unknown],
                other_excluded_samples=[total-used-unknown],
                unknown_market_observations=[count(==("unknown"),market.regime)])
            EnvConfig.writetable(excluded,"exclusions";folderpath=folder,format=:arrow)
        else
            for (coinix,coin) in enumerate(coins)
                progressmessage("persisting memberships $coin ($coinix/$(length(coins)))")
                persistinputsets!(coin,folder,coin -> readpreparedtable(scratch,coin).opentime,calendar)
                GC.gc()
            end
            assignmentloader = coin -> (sets=readpreparedtable(joinpath(folder,"stores",coin),"sets"),
                eligible=readpreparedtable(scratch,coin).opentime)
            progressmessage("summarizing calendar coverage";started)
            TradingStrategy.writecalendarcoverage!(folder,coins,assignmentloader,market)
        end
        progressmessage("preparation complete: output=$folder";started)
        @assert nrow(calendar)>0 "no shared blocks meet regime/liquidity/length constraints; inspect market.arrow and exclusions.arrow in $folder"
        stores = (coins=coins,featurestores=DataFrame[],targetstores=DataFrame[])
        return (;input,identity,stores,eligible=Dict{String,Vector{DateTime}}(),
            market,calendar,sets=Dict{String,DataFrame}(),folder)
    end
end

"""Persist private stores and shared memberships; market inputs are always streamed per coin."""
function prepare!(input::NamedTuple, folder::AbstractString;
        streaming::Bool=input.mode=="market")::NamedTuple
    streaming && return preparestreamed!(input,folder)
    description = preparationdescription(input)
    identity = TradingStrategy.experimentmanifest!(folder,input.histories,description)
    stores = TradingStrategy.experimentstores!(input.histories,input.featurefactory,input.targetfactory,folder;
        sharedstores=input.sharedstores)
    horizon = Targets.lookaheadminutes(input.targetfactory())
    assignments = TradingStrategy.regimeassignments(input.histories,stores,input.regime,
        input.folds,input.startdt,input.enddt,horizon)
    (;eligible,market,calendar,sets) = assignments
    EnvConfig.writetable(market,"market";folderpath=folder,format=:arrow)
    EnvConfig.writetable(calendar,"calendar";folderpath=folder,format=:arrow)
    for coin in stores.coins
        EnvConfig.writetable(sets[coin],"sets";folderpath=joinpath(folder,"stores",coin),format=:arrow)
    end
    TradingStrategy.writecalendarcoverage!(folder,sets,eligible,market)
    @assert nrow(calendar)>0 "no shared blocks meet regime/liquidity/length constraints; inspect market.arrow and exclusions.arrow in $folder"
    return (;input,identity,stores,eligible,market,calendar,sets,folder)
end

"""Compile all three compact populations and their metadata in one shared-store coin pass."""
function compileblocks!(prepared::NamedTuple, folder::AbstractString,
        labels::AbstractVector, cols::AbstractVector{<:AbstractString},
        description::AbstractString; folds::AbstractVector{Int},
        regimes::AbstractVector{<:AbstractString},
        storeloader::Function=coin -> inputfullstores(prepared.input,coin))::NamedTuple
    prepared = merge(prepared,(calendar=readpreparedtable(prepared.folder,"calendar"),))
    return mktempdir(mkpath(folder)) do scratch
        for (coinix,coin) in enumerate(prepared.stores.coins)
            progressmessage("compiling train/eval/test from shared stores $coin ($coinix/$(length(prepared.stores.coins)))")
            compileblockcoin!(prepared,scratch,labels,cols,coin,coinix,storeloader;folds,regimes)
            GC.gc()
        end
        indices = (
            trainindex=readpreparedtable(scratch,"train-index"),
            evalindex=readpreparedtable(scratch,"eval-index"),
            testindex=readpreparedtable(scratch,"test-index"))
        for (settype,index) in zip(("train","eval","test"),values(indices))
            @assert nrow(index)>0 "empty compiled source set=$settype folds=$folds regimes=$regimes"
        end
        train = Classify.readcompactsource(joinpath(scratch,"train"),length(cols),nrow(indices.trainindex);
            nlabels=length(labels),classbalancing=prepared.input.classbalancing)
        eval = Classify.readcompactsource(joinpath(scratch,"eval"),length(cols),nrow(indices.evalindex);
            nlabels=length(labels),classbalancing=false)
        test = Classify.readcompactsource(joinpath(scratch,"test"),length(cols),nrow(indices.testindex);
            nlabels=length(labels),classbalancing=false)
        identity = TradingStrategy.experimentmanifest!(folder,
            Dict("train"=>indices.trainindex,"eval"=>indices.evalindex),description)
        for settype in ("train","eval","test")
            mv(joinpath(scratch,settype*"-index.arrow"),joinpath(folder,settype*"-index.arrow");force=true)
            destination = mkpath(joinpath(folder,settype))
            for filename in (Classify.TRAINFEATURES_FILE,Classify.TRAINLABELS_FILE)
                mv(joinpath(scratch,settype,filename),joinpath(destination,filename);force=true)
            end
        end
        return (;indices...,identity,train,eval,test)
    end
end

"""Scope one coin's shared inputs and write all selected sets before releasing them."""
@noinline function compileblockcoin!(prepared::NamedTuple, folder::AbstractString,
        labels::AbstractVector, cols::AbstractVector{<:AbstractString},
        coin::AbstractString, coinix::Int, storeloader::Function;
        folds::AbstractVector{Int}, regimes::AbstractVector{<:AbstractString})::Nothing
    stores = storeloader(coin)
    times = readpreparedtable(joinpath(prepared.folder,"eligible"),coin).opentime
    indices = TradingStrategy.blockexperimentindices(stores.features,stores.targets,
        times,prepared.calendar,labels,coin,coinix;folds,regimes)
    for (settype,index) in zip(("train","eval","test"),values(indices))
        EnvConfig.writetable(index,settype*"-index";folderpath=folder,format=:arrow,
            append=coinix>1,preserveschema=true)
        Classify.writecompactbatch!(joinpath(folder,settype),stores.features,cols,index;
            append=coinix>1,nlabels=length(labels))
    end
    return nothing
end

"""Stage coin-batched indices, verify their immutable identity, then publish mapped tables."""
function persistedtrainingindices!(prepared::NamedTuple, sets::AbstractDict,
        labels::AbstractVector, folder::AbstractString, description::AbstractString;
        folds::AbstractVector{Int}, regimes::AbstractVector{<:AbstractString})::NamedTuple
    storeloader = coin -> begin
        progressmessage("building train/eval/test indices: $coin ($(searchsortedfirst(prepared.stores.coins,coin))/$(length(prepared.stores.coins)))")
        tables = prepared.input.mode=="market" ? inputfullstores(prepared.input,coin) :
            (features=readpreparedtable(joinpath(prepared.folder,"stores",coin),"features"),
                targets=readpreparedtable(joinpath(prepared.folder,"stores",coin),"targets"))
        (features=tables.features,targets=tables.targets,
            sets=isempty(sets) ? readpreparedtable(joinpath(prepared.folder,"stores",coin),"sets") :
                sets[coin])
    end
    return mktempdir(mkpath(folder)) do scratch
        started = time_ns()
        progressmessage("staging training indices: scratch=$scratch output=$folder")
        indices = TradingStrategy.experimentindices(scratch,prepared.stores.coins,
            storeloader,labels;folds,regimes)
        progressmessage("verifying train/eval identity: train=$(nrow(indices.trainindex)) eval=$(nrow(indices.evalindex)) test=$(nrow(indices.testindex))";started)
        identity = TradingStrategy.experimentmanifest!(folder,
            Dict("train"=>indices.trainindex,"eval"=>indices.evalindex),description)
        for stem in ("train-index","eval-index","test-index")
            mv(joinpath(scratch,stem*".arrow"),joinpath(folder,stem*".arrow");force=true)
        end
        progressmessage("published training indices: output=$folder";started)
        return (;indices...,identity)
    end
end

"""Build immutable indices in coin-sized disk batches and separate compact mmap sources."""
function trainingsources(prepared::NamedTuple, sets::AbstractDict, folder::AbstractString;
        folds::AbstractVector{Int}=Int[],regimes::AbstractVector{<:AbstractString}=String[],
        trainweights::String="class",streaming::Bool=false,
        keephistories::Bool=false)::NamedTuple
    labels = Targets.uniquelabels(prepared.input.targetfactory())
    cols = Features.requestedcolumns(prepared.input.featurefactory())
    compiled = prepared.input.mode=="market" && isempty(sets)
    description = "$(prepared.identity);training=$trainweights;folds=$folds;regimes=$regimes;eval=equal-cells-within-regimes-v1"
    if compiled
        (;trainindex,evalindex,testindex,identity,train,eval,test) = compileblocks!(prepared,folder,
            labels,cols,description;folds,regimes)
    elseif streaming
        (;trainindex,evalindex,testindex,identity) = persistedtrainingindices!(
            prepared,sets,labels,folder,description;folds,regimes)
    else
        @assert !isempty(prepared.stores.featurestores) "nonstreaming sources require loaded stores: coins=$(prepared.stores.coins)"
        trainindex = TradingStrategy.experimentindex(prepared.stores,sets,labels;
            settype="train",folds,regimes)
        evalindex = TradingStrategy.experimentindex(prepared.stores,sets,labels;
            settype="eval",folds,regimes)
        testindex = TradingStrategy.experimentindex(prepared.stores,sets,labels;
            settype="test",folds,regimes)
        identity = TradingStrategy.experimentmanifest!(folder,
            Dict("train"=>trainindex,"eval"=>evalindex),description)
        for (name,index) in (("train-index",trainindex),("eval-index",evalindex),("test-index",testindex))
            EnvConfig.writetable(index,name;folderpath=folder,format=:arrow)
        end
    end
    testindexloader = () -> EnvConfig.readtable("test-index";folderpath=folder,format=:arrow)
    evalindexloader = () -> EnvConfig.readtable("eval-index";folderpath=folder,format=:arrow)
    evalweights = Classify.fixedevalweights(evalindex.regime,evalindex.foldid)
    streaming && select!(testindex,Symbol[])
    traincellweights = trainweights == "balanced" ?
        TradingStrategy.balancedcellweights(trainindex) : nothing
    featurestoreloader = coinix -> begin
        coin = prepared.stores.coins[coinix]
        prepared.input.mode=="market" && return inputfullstores(prepared.input,coin).features
        progressmessage("loading compact-source features $coin ($coinix/$(length(prepared.stores.coins)))")
        store = EnvConfig.readtable("features";folderpath=joinpath(prepared.folder,"stores",coin),format=:arrow)
        @assert !isnothing(store) "missing persisted feature store for $coin in $(prepared.folder)"
        store
    end
    streaming && releasepreparedmemory!(prepared;keephistories)
    if !compiled
        progressmessage("compacting train source: samples=$(nrow(trainindex)) features=$(length(cols)) output=$(joinpath(folder,"train"))")
        train = Classify.CompactMmapSource(featurestoreloader,length(prepared.stores.coins),cols,trainindex;
            folderpath=mkpath(joinpath(folder,"train")),nlabels=length(labels),classbalancing=prepared.input.classbalancing)
    end
    streaming && select!(trainindex,Symbol[])
    if !compiled
        progressmessage("compacting eval source: samples=$(nrow(evalindex)) output=$(joinpath(folder,"eval"))")
        eval = Classify.CompactMmapSource(featurestoreloader,length(prepared.stores.coins),cols,evalindex;
            folderpath=mkpath(joinpath(folder,"eval")),nlabels=length(labels),classbalancing=false)
        test = nothing
    end
    streaming && select!(evalindex,Symbol[])
    if trainweights == "balanced"
        train = TradingStrategy.WeightedExperimentSource(train,traincellweights)
    end
    progressmessage("compact sources ready: train=$(Classify.nsamples(train)) eval=$(Classify.nsamples(eval))")
    return (;train,eval,trainindex,evalindex,evalweights,evalindexloader,testindex,testindexloader,identity,labels,cols,folder,
        featurestoreloader,coins=prepared.stores.coins,compiled,test)
end

"""Reload shared market windows or private artificial stores for chronological calculations."""
function reloadtrainingstores!(prepared::NamedTuple)::Nothing
    @assert isempty(prepared.stores.featurestores) && isempty(prepared.stores.targetstores) "experiment stores must be released before reload"
    for coin in prepared.stores.coins
        if prepared.input.mode=="market"
            features,targets = inputsharedstores(prepared.input,coin,prepared.input.histories[coin])
            push!(prepared.stores.featurestores,features)
            push!(prepared.stores.targetstores,targets)
            continue
        end
        folder = joinpath(prepared.folder,"stores",coin)
        features = EnvConfig.readtable("features";folderpath=folder,format=:arrow)
        targets = EnvConfig.readtable("targets";folderpath=folder,format=:arrow)
        @assert !isnothing(features) && !isnothing(targets) "missing persisted experiment stores for $coin in $folder"
        push!(prepared.stores.featurestores,features)
        push!(prepared.stores.targetstores,targets)
    end
    return nothing
end

"""Release preparation-only tables before streaming mmap compaction begins."""
function releasepreparedmemory!(prepared::NamedTuple;keephistories::Bool=false)::Nothing
    keephistories || empty!(prepared.input.histories)
    isnothing(prepared.input.sharedstores) || empty!(prepared.input.sharedstores)
    empty!(prepared.stores.featurestores)
    empty!(prepared.stores.targetstores)
    empty!(prepared.eligible)
    select!(prepared.market,Symbol[])
    select!(prepared.calendar,Symbol[])
    empty!(prepared.sets)
    GC.gc()
    return nothing
end

"""Release all sample indices and preparation tables; retain only weights and compact sources for adaptation."""
function releasetrainingmemory!(prepared::NamedTuple,sources::NamedTuple;
        keephistories::Bool=false)::Nothing
    select!(sources.trainindex,Symbol[])
    select!(sources.evalindex,Symbol[])
    select!(sources.testindex,Symbol[])
    releasepreparedmemory!(prepared;keephistories)
    return nothing
end

"""Normalize undefined numeric report columns and persist a completed metric phase."""
function persistreportmetrics!(table::DataFrame, folder::AbstractString, stem::String)::Nothing
    for name in names(table)
        if eltype(table[!,name])===Missing && !(name in ("regime","foldid"))
            table[!,name] = Vector{Union{Missing,Float64}}(missing,nrow(table))
        end
    end
    EnvConfig.writetable(table,stem;folderpath=folder,format=:arrow)
    return nothing
end

"""Persist predictions and completed metric phases before starting the next expensive report."""
function signalreport!(predictions::DataFrame,folder::AbstractString,stem::String;
        thresholds=0.0:0.05:1.0)::DataFrame
    started = time_ns()
    progressmessage("$stem report: persisting predictions rows=$(nrow(predictions)) output=$folder")
    EnvConfig.writetable(predictions,stem*"-predictions";folderpath=folder,format=:arrow)
    lastprogress = Ref(started)
    reportprogress = status -> begin
        current = time_ns()
        if status.groupix==1 && status.signal=="" || (current-lastprogress[])/1e9>=30
            progressmessage(
                "$stem $(status.phase): threshold=$(status.threshold) group=$(status.grouping) ($(status.groupix)/$(status.groups)) signal=$(status.signal) blocks=$(status.blockix)/$(status.blocks)";started)
            lastprogress[] = current
        end
    end
    progressmessage("$stem minute metrics: thresholds=$(length(thresholds)) bootstrap=200";started)
    metrics = Classify.signalmetrics(predictions;thresholds,bootstrap=200,onprogress=reportprogress)
    persistreportmetrics!(metrics,folder,stem*"-signals")
    progressmessage("$stem event metrics";started)
    events = Classify.signaleventmetrics(predictions;thresholds,onprogress=reportprogress)
    persistreportmetrics!(events,folder,stem*"-events")
    thresholdsummary!(metrics,folder,stem)
    signalplot!(metrics,folder,stem)
    coins = groupby(predictions,:coin;sort=true)
    for (coinix,coinrows) in enumerate(coins)
        coin = first(coinrows.coin)
        progressmessage("$stem coin report $coin ($coinix/$(length(coins)))";started)
        coinmetrics = Classify.signalmetrics(coinrows;thresholds)
        EnvConfig.writetable(coinmetrics,"$stem-coin-$coin";
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
    progressmessage("$stem reports complete: output=$folder";started)
    return metrics
end

"""Return the reserved test index, reloading it after streaming preparation released it."""
function loadtestindex(sources::NamedTuple)::DataFrame
    index = nrow(sources.testindex)>0 ? sources.testindex : sources.testindexloader()
    @assert !isnothing(index) && nrow(index)>0 "missing persisted test index in $(sources.folder)"
    return index
end

"""Load eval metadata for final reports after adaptation released the original index."""
function loadevalindex(sources::NamedTuple)::DataFrame
    index = sources.evalindexloader()
    @assert !isnothing(index) && nrow(index)==Classify.nsamples(sources.eval) "persisted eval index rows=$(isnothing(index) ? nothing : nrow(index)) must match samples=$(Classify.nsamples(sources.eval)) in $(sources.folder)"
    return index
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

"""
Write precision/PPV, recall and NPV SVG curves versus coverage with hover tooltips.
PPV equals precision; NPV is TN / (TN + FN), including abstentions as non-issued
signals in the existing confusion counts. Undefined ratios are omitted.
"""
function signalplot!(metrics::DataFrame,folder::AbstractString,stem::String)::Nothing
    metrics = copy(metrics)
    metrics.npv = [row.tn + row.fn == 0 ? missing : row.tn / (row.tn + row.fn)
        for row in eachrow(metrics)]
    signals = unique(metrics.signal)
    colors = Dict("overall"=>"#111111","up"=>"#267d36","flat"=>"#666666","down"=>"#b32727")
    open(joinpath(folder,stem*"-curves.svg"),"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1480\" height=\"$(length(signals)*280+40)\" viewBox=\"0 0 1480 $(length(signals)*280+40)\">")
        println(io,"<style>.point .tooltip{visibility:hidden;pointer-events:none}.point:hover .tooltip,.point:focus .tooltip{visibility:visible}.point{cursor:crosshair}text{font-family:sans-serif;font-size:12px}</style>")
        println(io,"<rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"20\" y=\"25\">Black overall; green up; gray flat; red down. Hover or Tab to points; labels show selected overall thresholds.</text>")
        for (signalix,signal) in enumerate(signals), (axisix,metric) in enumerate((:precision,:recall,:npv))
            left = (axisix-1)*480+60
            top = (signalix-1)*280+60
            metricname = metric == :precision ? "precision/PPV" : metric == :npv ? "NPV" : string(metric)
            println(io,"<text x=\"$left\" y=\"$(top-10)\">$signal: $metricname versus coverage</text>")
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
                    detail = "$regime threshold=$(row.threshold) coverage=$(round(row.coverage;digits=4)) $metricname=$(round(row[metric];digits=4))"
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

"""
Publish a registered best-eval NN and its provenance for normal classifier loading.
Synthetic fixtures have different features/targets and must not replace registered classifiers.
Unchanged publications preserve prediction caches; replacing a model invalidates derived artifacts.
"""
function publishclassifier!(prepared::NamedTuple, sources::NamedTuple, nn::Classify.NN)::Nothing
    input = prepared.input
    if input.mode=="synthetic"
        progressmessage("synthetic checkpoint retained at $(sources.folder); not publishing a registered classifier")
        return nothing
    end
    @assert (input.mode=="test")== (EnvConfig.configmode==EnvConfig.test) "input mode=$(input.mode) disagrees with classifier storage mode=$(EnvConfig.configmode)"
    rc = input.rc
    name = TradingStrategy.classifiername(rc.clcfg;coinset=rc.coinset,daterange=rc.daterange)
    meta = TradingStrategy.classifiermeta(rc.clcfg;coinset=rc.coinset,daterange=rc.daterange)
    @assert sources.cols==TradingStrategy.classifierfeaturecols(rc.clcfg) "source columns=$(sources.cols) differ from registered classifier=$name"
    @assert string.(nn.labels)==string.(TradingStrategy.classifierlabels(rc.clcfg)) "NN labels=$(nn.labels) differ from registered classifier=$name"
    checkpoint = joinpath(sources.folder,"eval-best.bson")
    merge!(meta,Dict("trainingprotocol"=>"regime-eval", "runid"=>rc.runid,
        "datasetidentity"=>prepared.identity, "trainingidentity"=>sources.identity,
        "checkpoint"=>abspath(checkpoint),
        "checkpointsha256"=>bytes2hex(open(SHA.sha256,checkpoint)),
        "coins"=>join(input.coins,","), "startdt"=>string(input.startdt),
        "enddt"=>string(input.enddt)))
    folder = EnvConfig.classifierfolder(name)
    if Classify.isclassifierstored(name) && isequal(Classify.readmeta(name),meta)
        progressmessage("classifier already published: $(joinpath(folder,"nn.bson"))")
        return nothing
    end
    progressmessage("publishing best-eval classifier: $name output=$folder")
    mktempdir(folder) do staging
        nn.fileprefix = Classify.NN_FILEPREFIX
        Classify.savenn(nn;folderpath=staging,fileprefix=Classify.NN_FILEPREFIX,
            save_lastepoch=false,save_result=true)
        for stem in ("predictions","nn-lastepoch.bson","confusion.arrow","xconfusion.arrow")
            rm(joinpath(folder,stem);recursive=stem=="predictions",force=true)
        end
        mv(joinpath(staging,"nn.bson"),joinpath(folder,"nn.bson");force=true)
        Classify.writemeta(name,meta)
    end
    progressmessage("classifier published: $(joinpath(folder,"nn.bson")); resume checkpoints remain at $(sources.folder)")
    return nothing
end

"""
Train and report on eval only; return the best held-out-loss checkpoint.
Publish registered classifiers before reporting unless `publish=false` for chronological models.
"""
function trainreport!(prepared::NamedTuple, sources::NamedTuple;
        epochs::Int,resume::Bool,reinforce::Int=0,publish::Bool=true)
    if isfile(joinpath(sources.folder,"fixed-threshold.arrow"))
        @warn "adapting a classifier previously used for test metric reporting" folder=sources.folder
    end
    if isfile(joinpath(sources.folder,"test-strategy.arrow"))
        @warn "adapting a classifier previously used for test gain replay" folder=sources.folder
    end
    nn = TradingStrategy.modelconfig(prepared.input.modelid)(length(sources.cols),sources.labels,"regime-$(prepared.input.modelid)")
    started = time_ns()
    select!(sources.trainindex,Symbol[])
    select!(sources.evalindex,Symbol[])
    select!(sources.testindex,Symbol[])
    GC.gc()
    progressmessage("adaptation starting: train=$(Classify.nsamples(sources.train)) eval=$(Classify.nsamples(sources.eval)) epochs=$epochs resume=$resume checkpoints=$(sources.folder)")
    result = Classify.traineval!(nn,sources.train,sources.eval;
        identity=sources.identity,folderpath=sources.folder,
        evalweights=sources.evalweights,
        maxepochs=epochs,resume,reinforce_epochs=reinforce)
    progressmessage("adaptation complete: epochs=$(length(result.records)) reason=$(result.stopreason)";started)
    publish && publishclassifier!(prepared,sources,result.best)
    refreshes = DataFrame(epoch=Int[],interval=Int[],hardcount=Int[])
    for row in result.reinforcement.history
        push!(refreshes,(row.epoch,row.interval,row.hardcount))
    end
    EnvConfig.writetable(refreshes,"reinforcement-refreshes";folderpath=sources.folder,format=:arrow)
    records = DataFrame(result.records)
    # Older checkpoints may contain cell histories; the current Arrow history is scalar.
    EnvConfig.writetable(select(records,Not(:cells)),"epoch-losses";
        folderpath=sources.folder,format=:arrow)
    rm(joinpath(sources.folder,"epoch-cells.arrow");force=true)
    evalindex = loadevalindex(sources)
    progressmessage("predicting eval samples=$(nrow(evalindex))";started)
    lastprediction = Ref(time_ns())
    predictionprogress = (completed,total) -> begin
        current = time_ns()
        if completed==total || (current-lastprediction[])/1e9>=30
            progressmessage("eval prediction samples=$completed/$total";started)
            lastprediction[] = current
        end
    end
    prediction = TradingStrategy.experimentpredictions(result.best,evalindex,sources.eval;
        onprogress=predictionprogress)
    progressmessage("eval predictions ready: rows=$(nrow(prediction)); starting reports";started)
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
        println(io,"Fixed sample weights: eval-manifest.bson; report membership: eval-index.arrow")
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
    3. Seal the original reserved test inputs once with stage=snapshot, runid=$(rc.runid).
       Then run RegimeTraining again with stage=gain, runid=$(rc.runid), and the same
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
    index = loadtestindex(sources)
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
    testsource = sources.compiled ? sources.test :
        Classify.CompactMmapSource(sources.featurestoreloader,length(sources.coins),sources.cols,index;
        folderpath=mkpath(joinpath(sources.folder,"test")),nlabels=length(sources.labels),
        classbalancing=false)
    prediction = TradingStrategy.experimentpredictions(best,index,testsource)
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
    return recordteststrategyidentity!(sources,identity,strategy)
end

"""Record replay exposure using an already verified immutable test-population identity."""
function recordteststrategyidentity!(sources::NamedTuple,identity::AbstractString,
        strategy::TradingStrategy.StrategyConfig)::Nothing
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

"""Replay reserved test predictions and target references using the selected strategy wired to the checkpoint classifier."""
function gainreport!(prepared::NamedTuple,sources::NamedTuple,best;
        folds::AbstractVector{Int}=Int[],regimes::AbstractVector{<:AbstractString}=String[])::Nothing
    rc = prepared.input.rc
    index = loadtestindex(sources)
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
    testsource = sources.compiled ? sources.test :
        Classify.CompactMmapSource(sources.featurestoreloader,length(sources.coins),
        sources.cols,index;folderpath=mkpath(joinpath(sources.folder,"test")),
        nlabels=length(sources.labels),classbalancing=false)
    predictions = TradingStrategy.experimentpredictions(best,index,testsource)
    EnvConfig.writetable(predictions,"test-predictions";folderpath=sources.folder,format=:arrow)
    snapshots = [:opentime,:pair,:set,:rangeid,:lastopentrade,:close,:high,:low,:label,:score,
        :lp_amount,:sp_amount,:lol_pavg,:lcl_pavg,:sol_pavg,:scl_pavg,:lo_limit,:lc_limit,:so_limit,:sc_limit]
    gainparts = DataFrame[]
    tradesfolder = mkpath(joinpath(gainfolder,"trades-regime"))
    for coin in sort!(unique(predictions.coin))
        history = loadinputhistory(prepared.input,coin)
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
                TradingStrategy.processreplaygains!(tp;strategy=ts.cfg)
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
        history = nothing
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
    gainreadablereport!(isempty(gainparts) ? DataFrame() : gains,gainfolder)
    EnvConfig.writetable(DataFrame(identity=[sources.identity],ts=[rc.tscfg.configname],
        strategy=[rc.tscfg.strategy],population=["test"],checkpoint=["eval-best.bson"]),
        "gain-identity";folderpath=gainfolder,format=:arrow)
    return nothing
end

"""Execute registered pooled training/replay or independent chronological models."""
function execute!(prepared::NamedTuple;stage::String,epochs::Int,resume::Bool,
        threshold::Union{Nothing,Float64},folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[],trainweights::String="class",reinforce::Int=0)::Nothing
    if stage == "gain"
        @assert !resume "gain cannot resume adaptation"
        snapshotgainreport!(prepared,snapshotsources(prepared;
            folds=join(folds,","),regimes=join(regimes,","),trainweights))
        return nothing
    end
    if stage != "walkforward"
        sources = trainingsources(prepared,prepared.sets,joinpath(prepared.folder,"balanced");
            folds,regimes,trainweights,streaming=true)
        if stage == "test"
            best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
            testreport!(prepared,sources,prepared.sets,best,threshold;folds,regimes)
        else
            @assert reinforce==0 || resume || !isfile(joinpath(sources.folder,"eval-best.bson")) "reinforce=$reinforce requires retrain to adapt an existing checkpoint"
            adapted = resume || !isfile(joinpath(sources.folder,"eval-best.bson"))
            RegimeTraining.releasetrainingmemory!(prepared,sources)
            if adapted
                best = trainreport!(prepared,sources;epochs,resume,reinforce)
            else
                best = Classify.loadtraincheckpoint(sources.folder;identity=sources.identity)
                publishclassifier!(prepared,sources,best)
            end
            if !adapted && stage in ("predict","evaluate")
                prediction = TradingStrategy.experimentpredictions(best,loadevalindex(sources),sources.eval)
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
    if isempty(prepared.input.histories)
        for coin in prepared.stores.coins
            prepared.input.histories[coin] = loadinputhistory(prepared.input,coin)
        end
    end
    for window in eachrow(schedule)
        isempty(prepared.stores.featurestores) && reloadtrainingstores!(prepared)
        folder = joinpath(prepared.folder,"walkforward-$(window.walkforwardid)")
        sets = TradingStrategy.chronologicalmemberships(prepared,window)
        sources = trainingsources(prepared,sets,folder;trainweights,streaming=true,keephistories=true)
        best = trainreport!(prepared,sources;epochs,resume,reinforce,publish=false)
        push!(tests,testreport!(prepared,sources,sets,best,threshold))
        releasetrainingmemory!(prepared,sources;keephistories=true)
        sources = nothing
        best = nothing
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
    folder = experimentfolder(rc)
    progressmessage("starting run=$(rc.runid) ts=$(rc.tscfg.configname) stage=$(rc.stage) output=$folder")
    @assert !haskey(opts,"threshold") || rc.stage in ("test","walkforward") "threshold is only supported for test/walkforward, stage=$(rc.stage); gain uses registered strategy thresholds"
    if rc.stage in ("snapshot","gain")
        prepared = snapshotprepared(opts,rc,folder)
        sources = snapshotsources(prepared;folds=get(opts,"folds",""),
            regimes=get(opts,"regimes",""),trainweights=get(opts,"trainweights","class"))
        if rc.stage == "snapshot"
            sealsnapshot!(prepared,sources)
        else
            snapshotgainreport!(prepared,sources)
        end
        progressmessage("run complete: artifacts=$folder")
        return nothing
    end
    input = inputs(opts,rc)
    prepared = prepare!(input,folder)
    stage,epochs = rc.stage,rc.epochs
    if stage == "stores"
        progressmessage("stores stage complete: calendar and coverage=$folder")
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
    progressmessage("run complete: artifacts=$folder")
    return nothing
end

include("regimegain.jl")

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
