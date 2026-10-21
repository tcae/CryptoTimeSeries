#region ArtifactPipeline

# Build or reuse the layered artifacts of a classifier config:
# per-coin regression/feature/target/sets stores -> training index -> adapted classifier -> per-coin predictions.
# Each step only calculates what is missing, i.e. a new strategy reuses predictions and a new
# target config reuses the feature stores.

"Return the per-coin feature, target and sets stores of a classifier config for `coin` or `nothing` entries if not stored."
function coinstores(clcfg::NamedTuple, coin::AbstractString; daterange::AbstractString, quotecoin::AbstractString=EnvConfig.pairquote)
    return (
        Features.readfeatures(coin, quotecoin, clcfg.features; confighash=featureshash(clcfg.features)),
        Targets.readtargets(coin, quotecoin, clcfg.targets; confighash=targetshash(clcfg.targets)),
        Classify.readsets(coin, quotecoin, setsid(clcfg.partition, daterange); confighash=setshash(clcfg.partition, daterange)),
    )
end

"""
    ensurecoinstores!(clcfg, coins; daterange)

Bring the per-coin feature, target and sets stores of `clcfg` up to date for all `coins` and
return the coins that have OHLCV data and therefore stores.
"""
function ensurecoinstores!(clcfg::NamedTuple, coins::AbstractVector{<:AbstractString}; daterange::AbstractString)::Vector{String}
    dr = daterangeconfig(daterange)
    processed = String[]
    for (coinix, coin) in enumerate(coins)
        ohlcv = Ohlcv.read(coin)
        if size(Ohlcv.dataframe(ohlcv), 1) < 2
            @warn "skipping coin without OHLCV history" coin
            continue
        end
        print("\r$(EnvConfig.now()) ensuring stores $(clcfg.features), $(clcfg.targets), $(setsid(clcfg.partition, daterange)) of $(coin) ($(coinix)/$(length(coins)))                ")
        Features.updatefeatures!(ohlcv, clcfg.features, featureconfig(clcfg.features); confighash=featureshash(clcfg.features))
        Targets.updatetargets!(ohlcv, clcfg.targets, targetconfig(clcfg.targets); confighash=targetshash(clcfg.targets))
        Classify.updatesets!(ohlcv, setsid(clcfg.partition, daterange), partitionconfig(clcfg.partition), dr.startdt, dr.enddt; confighash=setshash(clcfg.partition, daterange))
        push!(processed, String(coin))
    end
    println()
    return processed
end

"""
    trainclassifier!(clcfg; coinset, daterange, retrain=false, reinforce_epochs=0)

Return the adapted classifier of `clcfg` trained on `coinset` within `daterange`. An already
adapted classifier is reused unless `retrain` is set, in which case adaptation continues from
its stored weights, optimizer state and loss history, optionally with reinforcement.
If no adapted classifier exists, create and train a new classifier.
Training samples are compacted into one features × samples Float32 matrix file that is memory mapped for adaptation.
"""
function trainclassifier!(clcfg::NamedTuple; coinset::AbstractString, daterange::AbstractString, retrain::Bool=false, reinforce_epochs::Integer=0)::Classify.TrendClassifier001
    name = classifiername(clcfg; coinset=coinset, daterange=daterange)
    meta = classifiermeta(clcfg; coinset=coinset, daterange=daterange)
    cl = Classify.isclassifierstored(name) ? loadtrendclassifier(clcfg; coinset=coinset, daterange=daterange) : nothing
    if !isnothing(cl) && !retrain
        return cl
    end
    coins = ensurecoinstores!(clcfg, coinsetconfig(coinset).coins; daterange=daterange)
    @assert !isempty(coins) "no coin of coinset $(coinset) has OHLCV data to train classifier $(name)"
    labels = classifierlabels(clcfg)
    loadcoin(coin) = coinstores(clcfg, coin; daterange=daterange)
    indexdf = Classify.buildtrainindex(coins, loadcoin, labels)
    @assert size(indexdf, 1) > 0 "empty training index of classifier $(name)"
    featurestores = [loadcoin(coin)[1] for coin in coins]
    src = Classify.CompactMmapSource(featurestores, classifierfeaturecols(clcfg), indexdf;
        folderpath=EnvConfig.classifierfolder(name), nlabels=length(labels), classbalancing=clcfg.classbalancing)
    # Drop the compaction-only index and feature-store mappings before random-access training.
    indexdf = nothing
    empty!(featurestores)
    GC.gc()
    Classify.writemeta(name, merge(meta, Dict("coins" => join(coins, ","))))
    # predictions and evaluations of a previously adapted NN are invalid for the new one
    rm(Classify.predictionsfolder(name); recursive=true)
    if isnothing(cl)
        cl = Classify.newclassifier(name, modelconfig(clcfg.model), featurefactory(clcfg.features), labels)
    end
    println("$(EnvConfig.now()) adapting classifier $(name) with $(Classify.nsamples(src)) samples of $(length(coins)) coins")
    Classify.adaptnn!(cl.nn, src; maxepochs=coinsetconfig(coinset).maxepochs, reinforce_epochs=reinforce_epochs, folderpath=EnvConfig.classifierfolder(name))
    println("$(EnvConfig.now()) finished adapting classifier $(name) - classifier $(Classify.nnconverged(cl.nn) ? "did" : "did not") converge")
    return loadtrendclassifier(clcfg; coinset=coinset, daterange=daterange)
end

"""
    predictcoins!(clcfg, coins; coinset, daterange)

Bring the per-coin predictions of the adapted classifier of `clcfg` up to date with the feature
stores of `coins` and return the coins with predictions. Predictions that cover a prefix of the
feature store are only extended by the new feature rows.
"""
function predictcoins!(clcfg::NamedTuple, coins::AbstractVector{<:AbstractString}; coinset::AbstractString, daterange::AbstractString)::Vector{String}
    name = classifiername(clcfg; coinset=coinset, daterange=daterange)
    cl = loadtrendclassifier(clcfg; coinset=coinset, daterange=daterange)
    featurecols = classifierfeaturecols(clcfg)
    predicted = String[]
    for (coinix, coin) in enumerate(coins)
        fdf = coinstores(clcfg, coin; daterange=daterange)[1]
        @assert !isnothing(fdf) "missing feature store $(clcfg.features) of $(coin) - run the feature stage first"
        preddf = Classify.readpredictions(name, coin)
        npred = isnothing(preddf) ? 0 : size(preddf, 1)
        if (npred == size(fdf, 1)) && (preddf[end, :opentime] == fdf[end, :opentime])
            push!(predicted, String(coin))
            continue
        end
        extendable = (0 < npred < size(fdf, 1)) && (preddf[begin, :opentime] == fdf[begin, :opentime]) && (preddf[end, :opentime] == fdf[npred, :opentime])
        print("\r$(EnvConfig.now()) predicting $(coin) ($(coinix)/$(length(coins))) with $(name) from row $(extendable ? npred + 1 : 1)                ")
        if extendable
            taildf = Classify.predictfeatures(cl.nn, view(fdf, (npred + 1):size(fdf, 1), :), featurecols)
            Classify.savepredictions(name, coin, vcat(DataFrame(preddf; copycols=true), taildf))
        else
            Classify.savepredictions(name, coin, Classify.predictfeatures(cl.nn, fdf, featurecols))
        end
        push!(predicted, String(coin))
    end
    println()
    return predicted
end

"""
    evaluateclassifier(clcfg, coins; coinset, daterange)

Return the aggregated `(confusion, extendedconfusion)` matrices of the classifier of `clcfg` over
the sets of `coins` and persist them in the classifier folder.
"""
function evaluateclassifier(clcfg::NamedTuple, coins::AbstractVector{<:AbstractString}; coinset::AbstractString, daterange::AbstractString)
    name = classifiername(clcfg; coinset=coinset, daterange=daterange)
    labels = classifierlabels(clcfg)
    setlevels = unique(partitionconfig(clcfg.partition).samplesets)
    cmparts = DataFrame[]
    xcmparts = DataFrame[]
    for coin in coins
        _, tdf, sdf = coinstores(clcfg, coin; daterange=daterange)
        preddf = Classify.readpredictions(name, coin)
        @assert !isnothing(preddf) && !isnothing(tdf) && !isnothing(sdf) "missing predictions, targets or sets of $(coin) for $(name)"
        evaldf = Classify.evaluationdf(preddf, tdf, sdf, setlevels)
        size(evaldf, 1) == 0 && continue
        push!(cmparts, Classify.confusionmatrix(evaldf, labels))
        push!(xcmparts, Classify.extendedconfusionmatrix(evaldf, labels))
    end
    cmdf = Classify.aggregateconfusion(cmparts)
    xcmdf = Classify.aggregateextendedconfusion(xcmparts)
    folder = EnvConfig.classifierfolder(name)
    size(cmdf, 1) > 0 && EnvConfig.savedf(cmdf, "confusion"; folderpath=folder, format=:arrow)
    size(xcmdf, 1) > 0 && EnvConfig.savedf(xcmdf, "xconfusion"; folderpath=folder, format=:arrow)
    return cmdf, xcmdf
end

"""
    replayinput(clcfg, coin; coinset, daterange, startdt=nothing, enddt=nothing)

Return the gain replay input of `coin` with columns `opentime, high, low, close, coin, rangeid, set, target, score, label`.
Without `startdt`/`enddt` the rows of the sets store are used; with both the predicted rows within
`startdt..enddt` form one range of set `all`. Rows with incomplete targets are skipped.
"""
function replayinput(clcfg::NamedTuple, coin::AbstractString; coinset::AbstractString, daterange::AbstractString, startdt::Union{Nothing, DateTime}=nothing, enddt::Union{Nothing, DateTime}=nothing)::DataFrame
    @assert isnothing(startdt) == isnothing(enddt) "startdt=$(startdt) and enddt=$(enddt) must be both given or both nothing"
    name = classifiername(clcfg; coinset=coinset, daterange=daterange)
    _, tdf, sdf = coinstores(clcfg, coin; daterange=daterange)
    preddf = Classify.readpredictions(name, coin)
    @assert !isnothing(preddf) && !isnothing(tdf) "missing predictions or targets of $(coin) for $(name)"
    if isnothing(startdt)
        @assert !isnothing(sdf) "missing sets store of $(coin) for $(name)"
        evaldf = Classify.evaluationdf(preddf, tdf, sdf, unique(partitionconfig(clcfg.partition).samplesets))
    else
        pot = preddf[!, :opentime]
        rows = searchsortedfirst(pot, startdt):searchsortedlast(pot, enddt)
        windowsets = DataFrame(opentime=pot[rows], set=fill("all", length(rows)), rangeid=fill(Int(Classify.RANGEID_SUBRANGE_SPAN), length(rows)))
        evaldf = Classify.evaluationdf(preddf, tdf, windowsets, ["all"])
    end
    odf = Ohlcv.dataframe(Ohlcv.read(coin))
    oot = odf[!, :opentime]
    oix = [searchsortedfirst(oot, dt) for dt in evaldf[!, :opentime]]
    @assert all(ix -> (ix <= lastindex(oot)), oix) && (oot[oix] == evaldf[!, :opentime]) "OHLCV of $(coin) does not cover all replay opentimes"
    return DataFrame(
        opentime=evaldf[!, :opentime], high=odf[oix, :high], low=odf[oix, :low], close=odf[oix, :close],
        coin=fill(String(coin), size(evaldf, 1)), rangeid=evaldf[!, :rangeid], set=evaldf[!, :set],
        target=evaldf[!, :target], score=evaldf[!, :score], label=evaldf[!, :label],
    )
end

#endregion ArtifactPipeline
