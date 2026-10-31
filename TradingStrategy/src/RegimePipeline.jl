"""
Persist an input/config identity and fail on reuse of a different dataset.
Hashes contain full OHLCV content, not just coverage endpoints.
"""
function experimentmanifest!(folder::AbstractString, histories::AbstractDict,
        description::AbstractString)::String
    mkpath(folder)
    rows = DataFrame(coin=String[], digest=String[])
    for coin in sort!(collect(keys(histories)))
        # Stream through the repository writer, avoiding materializing a textual
        # representation of millions of candles for content hashing.
        staged = EnvConfig.writetable(histories[coin], "identity-input";
            folderpath=folder,format=:arrow)
        digest = open(sha256,staged)
        push!(rows,(coin,bytes2hex(digest)))
        rm(staged)
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
Never reuse legacy derived caches whose content revision cannot be established.
"""
function experimentstores!(histories::AbstractDict, featurefactory::Function,
        targetfactory::Function, folder::AbstractString)::NamedTuple
    coins = sort!(collect(keys(histories)))
    featurestores = DataFrame[]
    targetstores = DataFrame[]
    for coin in coins
        localfolder = mkpath(joinpath(folder,"stores",coin))
        if isfile(joinpath(localfolder,"features.arrow")) && isfile(joinpath(localfolder,"targets.arrow"))
            push!(featurestores,EnvConfig.readtable("features";folderpath=localfolder,format=:arrow))
            push!(targetstores,EnvConfig.readtable("targets";folderpath=localfolder,format=:arrow))
            continue
        end
        ohlcv = Ohlcv.defaultohlcv(coin)
        Ohlcv.setdataframe!(ohlcv,copy(histories[coin]))
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
        tdf = DataFrame(opentime=copy(histories[coin].opentime),label=codes)
        EnvConfig.writetable(fdf,"features";folderpath=localfolder,format=:arrow)
        EnvConfig.writetable(tdf,"targets";folderpath=localfolder,format=:arrow)
        push!(featurestores,fdf)
        push!(targetstores,tdf)
    end
    return (; coins,featurestores,targetstores)
end

"""
Find valid, liquid, nonignored samples through exact timestamp joins.
A chronological cutoff recomputes training liquidity from only that prefix.
"""
function experimenttimes(histories::AbstractDict, stores::NamedTuple, config::RegimeConfig;
        cutoff::Union{Nothing, DateTime}=nothing)::Dict{String,Vector{DateTime}}
    result = Dict{String,Vector{DateTime}}()
    for (coinix,coin) in enumerate(stores.coins)
        history = histories[coin]
        lastix = isnothing(cutoff) ? nrow(history) : searchsortedlast(history.opentime,cutoff)
        liquid = falses(nrow(history))
        if lastix > 0
            for range in Ohlcv.liquiditycheck(view(history,1:lastix,:);config.liquidity...)
                liquid[range] .= true
            end
        end
        tdf = stores.targetstores[coinix]
        fdf = stores.featurestores[coinix]
        features = Dict(time=>ix for (ix,time) in enumerate(fdf.opentime))
        target = Dict(time=>ix for (ix,time) in enumerate(tdf.opentime))
        times = DateTime[]
        for (ix,time) in enumerate(history.opentime)
            liquid[ix] && haskey(features,time) && haskey(target,time) || continue
            tdf.label[target[time]] != Targets.tradelabelcode(Targets.ignore) || continue
            @assert all(isfinite,fdf[features[time],Not(:opentime)]) "invalid feature row for $coin at $time"
            push!(times,time)
        end
        result[coin] = times
    end
    return result
end

"""
Build fold-aware source coordinates and metric metadata from eligible sets.
Unknown regimes are rejected; optional folds/regimes filter a fixed population.
"""
function experimentindex(stores::NamedTuple, sets::AbstractDict, labels::AbstractVector;
        settype::String, folds::AbstractVector{Int}=Int[],
        regimes::AbstractVector{<:AbstractString}=String[])::DataFrame
    index = DataFrame(coinix=Int[],row=Int[],labelix=Int[],coin=String[],
        opentime=DateTime[],set=String[],regime=String[],foldid=Int[],blockid=Int[])
    codes = Dict(Targets.tradelabelcode(label)=>ix for (ix,label) in enumerate(labels))
    for (coinix,coin) in enumerate(stores.coins)
        fdf = stores.featurestores[coinix]
        tdf = stores.targetstores[coinix]
        frows = Dict(time=>ix for (ix,time) in enumerate(fdf.opentime))
        trows = Dict(time=>ix for (ix,time) in enumerate(tdf.opentime))
        for row in eachrow(sets[coin])
            row.set == settype && row.regime != "unknown" || continue
            isempty(folds) || row.foldid in folds || continue
            isempty(regimes) || row.regime in regimes || continue
            @assert haskey(frows,row.opentime) && haskey(trows,row.opentime) "sets/features/targets misaligned: $coin $(row.opentime)"
            code = tdf.label[trows[row.opentime]]
            @assert haskey(codes,code) "unsupported target=$code for $coin at $(row.opentime)"
            push!(index,(coinix,frows[row.opentime],codes[code],coin,row.opentime,
                settype,row.regime,row.foldid,row.blockid))
        end
    end
    @assert nrow(index)>0 "empty experiment source set=$settype folds=$folds regimes=$regimes"
    return index
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
    populated = [copy(df) for df in values(sets) if nrow(df)>0]
    if !isempty(populated)
        table = vcat(populated...)
        counts = combine(groupby(table,[:set,:regime,:foldid]),nrow=>:samples)
        EnvConfig.writetable(counts,"sample-coverage";folderpath=folder,format=:arrow)
    end

    used = sum(nrow,values(sets))
    total = sum(length,values(eligible))
    @assert nrow(market)>1 "coverage requires at least two market observations, got=$(nrow(market))"
    step = market.opentime[2]-market.opentime[1]
    unknown = sum(count(times) do time
        ix = searchsortedlast(market.opentime,time)
        return ix>0 && time<last(market.opentime)+step && market.regime[ix]=="unknown"
    end for times in values(eligible))
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
