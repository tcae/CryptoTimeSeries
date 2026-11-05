"""Return the context length required by regime, liquidity, feature and target calculations."""
function regimewarmup(featurefactory::Function, targetfactory::Function,
        config::RegimeConfig)::Int
    regression = Targets.regressionfeatures(targetfactory())
    return max(config.longminutes,config.liquidity.startdistance+config.liquidity.minliquidminutes,
        Features.requiredminutes(featurefactory()),
        isnothing(regression) ? 0 : Features.requiredminutes(regression))
end

"""Select a history window; warn and return nothing when it cannot satisfy regime warmup."""
function regimehistory(raw::AbstractDataFrame, coin::AbstractString,
        startdt::DateTime, enddt::DateTime, warmup::Int)::Union{Nothing,DataFrame}
    firstrow = searchsortedfirst(raw.opentime,startdt-Minute(warmup+1))
    lastrow = searchsortedlast(raw.opentime,enddt)
    rows = max(0,lastrow-firstrow+1)
    if rows <= warmup
        @warn "skipping coin with insufficient window/context history" coin rows warmup startdt enddt
        return nothing
    end
    history = DataFrame(view(raw,firstrow:lastrow,:);copycols=true)
    Ohlcv.addpivot!(history)
    return history
end

"""Select aligned experiment rows from complete shared stores and purge unfinished targets."""
function sharedstorewindow(history::DataFrame, featurestore::AbstractDataFrame,
        targetstore::AbstractDataFrame, featurefactory::Function,
        targetfactory::Function, coin::AbstractString)::Tuple{DataFrame,DataFrame}
    required = Features.requiredminutes(featurefactory())
    expectedfeatures = nrow(history)-required+1
    @assert expectedfeatures > 0 "insufficient history for $coin: rows=$(nrow(history)) requiredminutes=$required"
    firstfeature = history.opentime[required]
    firstrow = searchsortedfirst(featurestore.opentime,firstfeature)
    lastrow = searchsortedlast(featurestore.opentime,last(history.opentime))
    features = DataFrame(view(featurestore,firstrow:lastrow,:);copycols=true)
    @assert nrow(features)==expectedfeatures && first(features.opentime)==firstfeature &&
        last(features.opentime)==last(history.opentime) "shared features misaligned for $coin: rows=$(nrow(features)) expected=$expectedfeatures"
    firstrow = searchsortedfirst(targetstore.opentime,first(history.opentime))
    lastrow = searchsortedlast(targetstore.opentime,last(history.opentime))
    targets = DataFrame(view(targetstore,firstrow:lastrow,:);copycols=true)
    @assert nrow(targets)==nrow(history) && first(targets.opentime)==first(history.opentime) &&
        last(targets.opentime)==last(history.opentime) "shared targets misaligned for $coin: rows=$(nrow(targets)) expected=$(nrow(history))"
    finalcount = max(0,nrow(targets)-Targets.lookaheadminutes(targetfactory()))
    targets[(finalcount+1):end,:label] .= Targets.tradelabelcode(Targets.ignore)
    return features,targets
end

"""Build the full regime/fold preparation shared by training and refresh eligibility reporting."""
function regimeassignments(histories::AbstractDict, stores::NamedTuple,
        config::RegimeConfig, folds::FoldConfig, startdt::DateTime, enddt::DateTime,
        horizon::Int)::NamedTuple
    eligible = experimenttimes(histories,stores,config)
    market = marketcalendar(histories,config;startdt,enddt)
    calendar = foldcalendar(market,eligible,folds;horizonminutes=horizon,enddt)
    sets = Dict(coin=>calendarsets(eligible[coin],calendar) for coin in stores.coins)
    return (;eligible,market,calendar,sets)
end

"""
Persist an input/config identity and fail on reuse of a different dataset.
Hashes contain full OHLCV content, not just coverage endpoints.
"""
function experimentmanifest!(folder::AbstractString, histories::AbstractDict,
        description::AbstractString)::String
    return experimentmanifest!(folder,sort!(collect(keys(histories))),
        coin -> histories[coin],description)
end

"""Load and hash one history inside a non-inlined scope so the caller retains no source roots."""
@noinline function _experimentdigest!(folder::AbstractString, historyloader::Function,
        coin::AbstractString)::String
    history = historyloader(coin)
    staged = EnvConfig.writetable(history,"identity-input";folderpath=folder,format=:arrow)
    try
        return bytes2hex(open(sha256,staged))
    finally
        rm(staged)
    end
end

"""Persist the same immutable input identity while loading and releasing one coin at a time."""
function experimentmanifest!(folder::AbstractString, coins::AbstractVector{<:AbstractString},
        historyloader::Function, description::AbstractString)::String
    mkpath(folder)
    rows = DataFrame(coin=String[], digest=String[])
    for coin in sort(coins)
        push!(rows,(coin,_experimentdigest!(folder,historyloader,coin)))
        GC.gc()
    end
    identity = confighash(description * join(["$(row.coin):$(row.digest)" for row in eachrow(rows)],";"))
    path = joinpath(folder,"manifest.arrow")
    if isfile(path)
        previous = EnvConfig.readtable("manifest";folderpath=folder,format=:arrow)
        @assert only(previous.identity) == identity "experiment dataset/config changed: stored=$(only(previous.identity)), requested=$identity; choose a new run ID"
    else
        EnvConfig.writetable(DataFrame(identity=[identity],description=[description]),
            "manifest";folderpath=folder,format=:arrow)
        EnvConfig.writetable(rows,"input-identities";folderpath=folder,format=:arrow)
    end
    return identity
end

"""
Rebuild or reuse experiment-private features and targets by full input identity.
When full-history shared stores are supplied, select and validate the experiment window from
them; otherwise calculate the window locally. Persisted rows must cover the complete experiment
history, accounting for feature warm-up and incomplete target look-ahead.
"""
function experimentstores!(histories::AbstractDict, featurefactory::Function,
        targetfactory::Function, folder::AbstractString;
        sharedstores::Union{Nothing, AbstractDict}=nothing)::NamedTuple
    coins = sort!(collect(keys(histories)))
    featurestores = DataFrame[]
    targetstores = DataFrame[]
    for coin in coins
        history = histories[coin]
        localfolder = mkpath(joinpath(folder,"stores",coin))
        if isfile(joinpath(localfolder,"features.arrow")) && isfile(joinpath(localfolder,"targets.arrow"))
            fdf = EnvConfig.readtable("features";folderpath=localfolder,format=:arrow)
            tdf = EnvConfig.readtable("targets";folderpath=localfolder,format=:arrow)
            required = Features.requiredminutes(featurefactory())
            expectedfeatures = nrow(history) - required + 1
            if (expectedfeatures > 0) && (nrow(fdf) == expectedfeatures) &&
                    (fdf[begin,:opentime] == history.opentime[required]) &&
                    (fdf[end,:opentime] == history.opentime[end]) &&
                    (nrow(tdf) == nrow(history)) &&
                    (tdf[begin,:opentime] == history.opentime[begin]) &&
                    (tdf[end,:opentime] == history.opentime[end])
                push!(featurestores,fdf)
                push!(targetstores,tdf)
                continue
            end
        end
        if !isnothing(sharedstores) && haskey(sharedstores,coin)
            sharedfeature, sharedtarget = sharedstores[coin]
            fdf = sharedfeature
            tdf = sharedtarget
        else
            ohlcv = Ohlcv.defaultohlcv(coin)
            Ohlcv.setdataframe!(ohlcv,copy(history))
            Ohlcv.addpivot!(ohlcv)
            features = featurefactory()
            Features.setbase!(features,ohlcv;usecache=false)
            fdf = copy(features.fdf[!,vcat(["opentime"],Features.requestedcolumns(features))])
            firsttime = Ohlcv.dataframe(ohlcv).opentime[Features.requiredminutes(features)]
            filter!(row -> row.opentime >= firsttime,fdf)
            target = targetfactory()
            regression = Targets.regressionfeatures(target)
            if isnothing(regression)
                Targets.setbase!(target,ohlcv)
            else
                Features.setbase!(regression,ohlcv;usecache=false)
                Targets.setbase!(target,regression)
            end
            codes = Targets.tradelabelcode.(Targets.labels(target))
            codes[Targets.finallabelcount(target,length(codes))+1:end] .= Targets.tradelabelcode(Targets.ignore)
            tdf = DataFrame(opentime=copy(history.opentime),label=codes)
        end
        required = Features.requiredminutes(featurefactory())
        expectedfeatures = nrow(history) - required + 1
        @assert expectedfeatures > 0 "insufficient history for features of $coin: rows=$(nrow(history)) requiredminutes=$required"
        @assert nrow(fdf) == expectedfeatures "feature store of $coin has $(nrow(fdf)) rows; expected $expectedfeatures rows from $(history.opentime[required]) to $(history.opentime[end])"
        @assert (fdf[begin,:opentime] == history.opentime[required]) && (fdf[end,:opentime] == history.opentime[end]) "feature store of $coin spans $(fdf[begin,:opentime]) to $(fdf[end,:opentime]); expected $(history.opentime[required]) to $(history.opentime[end])"
        @assert nrow(tdf) == nrow(history) "target store of $coin has $(nrow(tdf)) rows; expected $(nrow(history)) rows from $(history.opentime[begin]) to $(history.opentime[end])"
        @assert (tdf[begin,:opentime] == history.opentime[begin]) && (tdf[end,:opentime] == history.opentime[end]) "target store of $coin spans $(tdf[begin,:opentime]) to $(tdf[end,:opentime]); expected $(history.opentime[begin]) to $(history.opentime[end])"
        EnvConfig.writetable(fdf,"features";folderpath=localfolder,format=:arrow)
        EnvConfig.writetable(tdf,"targets";folderpath=localfolder,format=:arrow)
        push!(featurestores,fdf)
        push!(targetstores,tdf)
    end
    return (; coins,featurestores,targetstores)
end

"""Join sorted timestamp vectors without per-row DataFrame access or timestamp dictionaries."""
function _sortedexperimenttimes(historytimes::AbstractVector{DateTime},
        featuretimes::AbstractVector{DateTime}, targettimes::AbstractVector{DateTime},
        targetlabels::AbstractVector, finitefeatures::AbstractVector{Bool},
        liquid::AbstractVector{Bool}, coin::AbstractString)::Vector{DateTime}
    ignorecode = Targets.tradelabelcode(Targets.ignore)
    featureix = targetix = 1
    times = DateTime[]
    for ix in eachindex(liquid)
        liquid[ix] || continue
        time = historytimes[ix]
        while featureix <= length(featuretimes) && featuretimes[featureix] < time
            featureix += 1
        end
        while targetix <= length(targettimes) && targettimes[targetix] < time
            targetix += 1
        end
        featureix <= length(featuretimes) && featuretimes[featureix] == time || continue
        targetix <= length(targettimes) && targettimes[targetix] == time || continue
        targetlabels[targetix] != ignorecode || continue
        @assert finitefeatures[featureix] "invalid feature row for $coin at $time"
        push!(times,time)
    end
    return times
end

"""
Find valid, liquid, nonignored samples through an exact sorted timestamp walk.
All input timestamps must be sorted and unique. Feature validity is calculated column-wise
and checked only for joined, liquid, nonignored samples.
A chronological cutoff recomputes training liquidity from only that prefix.
"""
function experimenttimes(histories::AbstractDict, stores::NamedTuple, config::RegimeConfig;
        cutoff::Union{Nothing, DateTime}=nothing)::Dict{String,Vector{DateTime}}
    result = Dict{String,Vector{DateTime}}()
    for (coinix,coin) in enumerate(stores.coins)
        history = histories[coin]
        tdf = stores.targetstores[coinix]
        fdf = stores.featurestores[coinix]
        historytimes = history[!, :opentime]
        featuretimes = fdf[!, :opentime]
        targettimes = tdf[!, :opentime]
        for (kind,timestamps) in (("history",historytimes),("features",featuretimes),("targets",targettimes))
            @assert _strictregimetimes(timestamps) "$(kind) timestamps for $(coin) must be sorted and unique: rows=$(length(timestamps))"
        end
        lastix = isnothing(cutoff) ? nrow(history) : searchsortedlast(historytimes,cutoff)
        liquid = falses(lastix)
        if lastix > 0
            for range in Ohlcv.liquiditycheck(view(history,1:lastix,:);config.liquidity...)
                liquid[range] .= true
            end
        end
        finitefeatures = trues(nrow(fdf))
        for column in names(fdf,Not(:opentime))
            finitefeatures .&= isfinite.(fdf[!,column])
        end
        result[coin] = _sortedexperimenttimes(historytimes,featuretimes,targettimes,
            tdf[!, :label],finitefeatures,liquid,coin)
    end
    return result
end

"""Create an empty fold-aware source index with stable column types."""
function _experimentindex()::DataFrame
    return DataFrame(coinix=Int[],row=Int[],labelix=Int[],coin=String[],
        opentime=DateTime[],set=String[],regime=String[],foldid=Int[],blockid=Int[])
end

"""Append one coin's source coordinates using strictly sorted, exact timestamp joins."""
function _appendexperimentindex!(index::DataFrame, fdf::AbstractDataFrame,
        tdf::AbstractDataFrame, sets::AbstractDataFrame, labels::AbstractVector,
        coin::AbstractString, coinix::Int; settype::String,
        folds::AbstractVector{Int}, regimes::AbstractVector{<:AbstractString})::Nothing
    featuretimes,targettimes,targetlabels = fdf.opentime,tdf.opentime,tdf.label
    for (kind,timestamps) in (("features",featuretimes),("targets",targettimes),
            ("sets",sets.opentime))
        @assert _strictregimetimes(timestamps) "$kind timestamps for $coin must be sorted and unique: rows=$(length(timestamps))"
    end
    codes = Dict(Targets.tradelabelcode(label)=>ix for (ix,label) in enumerate(labels))
    featureix = targetix = 1
    for row in eachrow(sets)
        row.set == settype && row.regime != "unknown" || continue
        isempty(folds) || row.foldid in folds || continue
        isempty(regimes) || row.regime in regimes || continue
        time = row.opentime
        while featureix <= length(featuretimes) && featuretimes[featureix] < time
            featureix += 1
        end
        while targetix <= length(targettimes) && targettimes[targetix] < time
            targetix += 1
        end
        @assert featureix <= length(featuretimes) && featuretimes[featureix] == time &&
            targetix <= length(targettimes) && targettimes[targetix] == time "sets/features/targets misaligned: $coin $time"
        code = targetlabels[targetix]
        @assert haskey(codes,code) "unsupported target=$code for $coin at $time"
        push!(index,(coinix,featureix,codes[code],coin,time,
            settype,row.regime,row.foldid,row.blockid))
    end
    return nothing
end

"""
Build fold-aware source coordinates and metric metadata using sorted timestamp walks.
Inputs must have strictly increasing, unique timestamps; optional folds/regimes filter
a fixed known-regime population.
"""
function experimentindex(stores::NamedTuple, sets::AbstractDict, labels::AbstractVector;
        settype::String, folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[])::DataFrame
    index = _experimentindex()
    for (coinix,coin) in enumerate(stores.coins)
        _appendexperimentindex!(index,stores.featurestores[coinix],stores.targetstores[coinix],
            sets[coin],labels,coin,coinix;settype,folds,regimes)
    end
    @assert nrow(index)>0 "empty experiment source set=$settype folds=$folds regimes=$regimes"
    return index
end

"""
Write train/eval/test indices in coin-sized Arrow batches into a fresh folder, returning
zero-copy mapped indices. The loader returns features, targets and sets. Input tables and
index batches are released before the next load; coin order and source row coordinates
match `experimentindex`. No accumulated minute-level output vectors are retained.
"""
function experimentindices(folder::AbstractString, coins::AbstractVector{<:AbstractString}, storeloader::Function,
        labels::AbstractVector; folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[])::NamedTuple
    @assert !isempty(coins) "experiment indices require coins, got=$coins"
    for (coinix,coin) in enumerate(coins)
        _writecoinindices!(folder,storeloader,labels,coin,coinix;folds,regimes)
        GC.gc()
    end
    indices = (
        trainindex=EnvConfig.readtable("train-index";folderpath=folder,format=:arrow,copycols=false),
        evalindex=EnvConfig.readtable("eval-index";folderpath=folder,format=:arrow,copycols=false),
        testindex=EnvConfig.readtable("test-index";folderpath=folder,format=:arrow,copycols=false))
    for (settype,index) in zip(("train","eval","test"),values(indices))
        @assert !isnothing(index) && nrow(index)>0 "empty experiment source set=$settype folds=$folds regimes=$regimes"
    end
    return indices
end

"""Scope one coin's source roots and write fixed-schema batches; the first coin replaces output."""
@noinline function _writecoinindices!(folder::AbstractString, storeloader::Function,
        labels::AbstractVector, coin::AbstractString, coinix::Int;
        folds::AbstractVector{Int}, regimes::AbstractVector{<:AbstractString})::Nothing
    stores = storeloader(coin)
    for settype in ("train","eval","test")
        index = _experimentindex()
        _appendexperimentindex!(index,stores.features,stores.targets,stores.sets,
            labels,coin,coinix;settype,folds,regimes)
        EnvConfig.writetable(index,settype*"-index";folderpath=folder,format=:arrow,
            append=coinix>1,preserveschema=true)
    end
    return nothing
end

"""Freeze sample weights: equal cell means within regimes, equal represented regimes."""
function balancedcellweights(index::DataFrame)::Vector{Float32}
    @assert nrow(index)>0 && all(!=("unknown"),index.regime) "invalid weighting population rows=$(nrow(index))"
    counts = combine(groupby(index,[:regime,:foldid]),nrow=>:samples)
    cellsperregime = Dict(regime=>count(==(regime),counts.regime) for regime in unique(counts.regime))
    numberregimes = length(cellsperregime)
    masses = Dict((row.regime,row.foldid)=>1f0/(numberregimes*cellsperregime[row.regime]*row.samples)
        for row in eachrow(counts))
    weights = [masses[(row.regime,row.foldid)] for row in eachrow(index)]
    weights ./= sum(weights)
    return weights
end

"""Write the actual sample distribution and explicitly counted exclusions."""
function writecalendarcoverage!(folder::AbstractString, sets::AbstractDict,
        eligible::AbstractDict, market::DataFrame)::Nothing
    return writecalendarcoverage!(folder,sort!(collect(keys(sets))),
        coin -> (sets=sets[coin],eligible=eligible[coin]),market)
end

"""Scope one coin's source roots and return only cell counts and scalar exclusions."""
@noinline function _calendarcoverage(assignmentloader::Function, coin::AbstractString,
        market::DataFrame, step::Period)::NamedTuple
    (;sets,eligible) = assignmentloader(coin)
    counts = combine(groupby(sets,[:set,:regime,:foldid]),nrow=>:samples;threads=false)
    unknown = count(eligible) do time
        ix = searchsortedlast(market.opentime,time)
        return ix>0 && time<last(market.opentime)+step && market.regime[ix]=="unknown"
    end
    return (;counts,used=nrow(sets),total=length(eligible),unknown)
end

"""Write identical coverage by retaining only per-cell counts between coin-scoped loads."""
function writecalendarcoverage!(folder::AbstractString, coins::AbstractVector{<:AbstractString},
        assignmentloader::Function, market::DataFrame)::Nothing
    @assert nrow(market)>1 "coverage requires at least two market observations, got=$(nrow(market))"
    step = market.opentime[2]-market.opentime[1]
    counts = DataFrame(set=String[],regime=String[],foldid=Int[],samples=Int[])
    used = total = unknown = 0
    for coin in coins
        coverage = _calendarcoverage(assignmentloader,coin,market,step)
        append!(counts,coverage.counts)
        used += coverage.used
        total += coverage.total
        unknown += coverage.unknown
        GC.gc()
    end
    if nrow(counts)>0
        counts = combine(groupby(counts,[:set,:regime,:foldid]),:samples=>sum=>:samples)
        EnvConfig.writetable(counts,"sample-coverage";folderpath=folder,format=:arrow)
    end
    excluded = DataFrame(eligible_samples=[total],assigned_samples=[used],
        excluded_samples=[total-used],unknown_samples=[unknown],
        other_excluded_samples=[total-used-unknown],
        unknown_market_observations=[count(==("unknown"),market.regime)])
    EnvConfig.writetable(excluded,"exclusions";folderpath=folder,format=:arrow)
    return nothing
end

"""
Construct each chronological population using only coverage/regime confirmations
available by its own window end, with final target lookahead purged explicitly.
Do not reuse hindsight-balanced diagnostic assignments for walk-forward training.
"""
function chronologicalmemberships(prepared::NamedTuple, window)::Dict{String,DataFrame}
    input = prepared.input
    horizon = Targets.lookaheadminutes(input.targetfactory())
    @assert input.folds.gapminutes >= horizon "walk-forward gap=$(input.folds.gapminutes) < horizon=$horizon"
    populations = Dict(coin=>DataFrame(opentime=DateTime[],set=String[],regime=String[],
        foldid=Int[],blockid=Int[],rangeid=Int[]) for coin in prepared.stores.coins)
    for (settype,start,stop) in (("train",window.trainstart,window.trainstop),
            ("eval",window.evalstart,window.evalstop),("test",window.teststart,window.teststop))
        times = experimenttimes(input.histories,prepared.stores,input.regime;cutoff=stop)
        market = marketcalendar(input.histories,input.regime;
            startdt=input.startdt,enddt=stop,cutoff=stop)
        calendar = foldcalendar(market,times,input.folds;horizonminutes=horizon,enddt=stop)
        for coin in prepared.stores.coins
            assigned = calendarsets(times[coin],calendar)
            filter!(row->start<=row.opentime<=stop-Minute(horizon),assigned)
            assigned.set .= settype
            append!(populations[coin],assigned)
        end
    end
    return populations
end

"""Optional train-only cell weights layered over a compact source's class weights."""
struct WeightedExperimentSource{S<:Classify.AbstractSampleSource} <: Classify.AbstractSampleSource
    source::S
    weights::Vector{Float32}
end

"""Return the sample count of the underlying compact experiment source."""
Classify.nsamples(source::WeightedExperimentSource) = Classify.nsamples(source.source)

"""Preserve the compact source's sample ordering for weighted training batches."""
Classify.batchindices(source::WeightedExperimentSource, indices::AbstractVector{<:Integer}) =
    Classify.batchindices(source.source,indices)

"""Return labels without changing sample order."""
Classify.classindex(source::WeightedExperimentSource, indices::AbstractVector{<:Integer}) =
    Classify.classindex(source.source,indices)

"""Multiply train-only cell weights into existing optional class weights."""
function Classify.batch(source::WeightedExperimentSource, indices::AbstractVector{<:Integer})
    x,y,w = Classify.batch(source.source,indices)
    extra = source.weights[indices]
    return x,y,isnothing(w) ? extra : w .* extra
end

"""Build metric rows only for the explicitly selected, known-regime population."""
function experimentpredictions(nn, index::DataFrame, stores::NamedTuple,
        featurecols::AbstractVector{<:AbstractString})::DataFrame
    result = select(index,:coin,:opentime,:set,:regime,:foldid,:blockid)
    predicted = [Classify.predictfeatures(nn,store,featurecols) for store in stores.featurestores]
    labels = Dict(Targets.tradelabelcode(label)=>label for label in nn.labels)
    result[!, :label] = [labels[predicted[row.coinix].label[row.row]] for row in eachrow(index)]
    result[!, :score] = [predicted[row.coinix].score[row.row] for row in eachrow(index)]
    result[!, :target] = [nn.labels[row.labelix] for row in eachrow(index)]
    return result
end

"""
Build metric rows from compact sample-source batches without loading per-coin feature stores.
Optional `onprogress(completed,total)` reports completed prediction samples after each batch.
"""
function experimentpredictions(nn, index::DataFrame,
        source::Classify.AbstractSampleSource;
        onprogress::Union{Nothing,Function}=nothing)::DataFrame
    @assert nrow(index) == Classify.nsamples(source) "prediction index rows=$(nrow(index)) differ from compact source samples=$(Classify.nsamples(source))"
    result = select(index,:coin,:opentime,:set,:regime,:foldid,:blockid)
    scores = Vector{Float32}(undef,nrow(index))
    labels = Vector{eltype(nn.labels)}(undef,nrow(index))
    Classify.Flux.testmode!(nn.model)
    for chunk in Iterators.partition(1:nrow(index),8192)
        ordered = Classify.batchindices(source,chunk)
        features,_,_ = Classify.batch(source,ordered)
        chunkscores,chunklabels = Classify.maxpredict(nn,features)
        scores[ordered] .= chunkscores
        labels[ordered] .= chunklabels
        isnothing(onprogress) || onprogress(last(chunk),nrow(index))
    end
    result[!,:label] = labels
    result[!,:score] = scores
    result[!,:target] = [nn.labels[row.labelix] for row in eachrow(index)]
    return result
end
