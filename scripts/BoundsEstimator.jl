module BoundsEstimator
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics, MLUtils, StatisticalMeasures
using CategoricalArrays, CategoricalDistributions, Distributions
using EnvConfig, Classify, CryptoXch, Ohlcv, Features, Targets, TradingStrategy

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3
const BOUNDS_LABELS = ["center", "width"]
const BOUNDS_RATIO_FORMAT = "relative_v1"
softplus(x::Real) = log1p(exp(-abs(x))) + max(x, 0)

function has_current_target_format(df::AbstractDataFrame)
    reqcols = (:centertarget, :widthtarget)
    all(c -> c in propertynames(df), reqcols) || return false
    if :bounds_target_format in propertynames(df)
        return all(==("$(BOUNDS_RATIO_FORMAT)"), df[!, :bounds_target_format])
    end
    return true
end

function has_current_prediction_format(df::AbstractDataFrame)
    reqcols = (:centerpred, :widthpred)
    all(c -> c in propertynames(df), reqcols) || return false
    return :bounds_prediction_format ∉ propertynames(df)
end

"""
Compute normalized bounds targets from absolute OHLC-derived center/width and pivot.

Returns a tuple `(center_ratio, width_ratio)` with:
- `center_ratio = (center_abs - pivot) / pivot`
- `width_ratio = width_abs / pivot`
"""
function normalize_bounds_targets(center_abs::AbstractVector{<:Real}, width_abs::AbstractVector{<:Real}, pivot::AbstractVector{<:Real})
    @assert length(center_abs) == length(width_abs) == length(pivot) "length mismatch center_abs=$(length(center_abs)) width_abs=$(length(width_abs)) pivot=$(length(pivot))"
    p = Float32.(pivot)
    @assert all(isfinite.(p)) "pivot contains non-finite values"
    @assert all(p .> 0f0) "pivot must be > 0 for ratio targets; min pivot=$(minimum(p))"
    @assert all(center_abs .> 0f0) "center_abs must be > 0 for ratio targets; min center_abs=$(minimum(center_abs))"
    @assert all(width_abs .> 0f0) "width_abs must be > 0 for ratio targets; min width_abs=$(minimum(width_abs))"

    center_ratio = (Float32.(center_abs) .- p) ./ p # can be positive or negative depending on whether center is above or below pivot
    width_ratio = Float32.(width_abs) ./ p # must be non negative as width is an absolute distance
    return center_ratio, width_ratio
end

"""
Reconstruct absolute predicted lower/upper bounds from pivot-relative predictions.

Returns a tuple `(pred_lower_abs, pred_upper_abs)` with:
- `pred_center_abs = (1 + pred_center_ratio) * pivot`
- `pred_width_abs = pred_width_ratio * pivot`
"""
function denormalize_predicted_bounds(pred_center_ratio::AbstractVector{<:Real}, pred_width_ratio::AbstractVector{<:Real}, pivot::AbstractVector{<:Real})
    @assert length(pred_center_ratio) == length(pred_width_ratio) == length(pivot) "length mismatch pred_center=$(length(pred_center_ratio)) pred_width=$(length(pred_width_ratio)) pivot=$(length(pivot))"
    p = Float32.(pivot)
    @assert all(isfinite.(p)) "pivot contains non-finite values"
    @assert all(p .> 0f0) "pivot must be > 0 to reconstruct predicted bands; min pivot=$(minimum(p))"

    c = Float32.(pred_center_ratio)
    w = Float32.(pred_width_ratio)
    pred_center_abs = (1f0 .+ c) .* p
    pred_width_abs = w .* p
    pred_lower_abs = clamp.(pred_center_abs .- pred_width_abs ./ 2f0, 0f0, Inf32)
    pred_upper_abs = clamp.(pred_center_abs .+ pred_width_abs ./ 2f0, 0f0, Inf32)
    return pred_lower_abs, pred_upper_abs
end


"""
inspect = provide a look into files and data structures 
execute = run training and evaluation
special = run special tasks for repair, debugging or refactoring
"""
@enum BoundsEstimatorMode inspect execute special

mutable struct BoundsEstimatorConfig
    configname::String
    folder::String
    featconfig::Features.AbstractFeatures
    targetconfig::Targets.AbstractTargets
    regressormodel
    tradingstrategy::TradingStrategy.GainSegment
    startdt::DateTime
    enddt::DateTime
    opmode::BoundsEstimatorMode
    partitionconfig::NamedTuple
    coins::Vector{String}
    function BoundsEstimatorConfig(;configname, folder="Bounds-$configname-$(EnvConfig.configmode)", featconfig, targetconfig, regressormodel, tradingstrategy, startdt, enddt, opmode=execute, partitionconfig=partitionconfig02(), coins)
        EnvConfig.setlogpath(folder)
        EnvConfig.setdfformat!(:arrow)
        @assert hasproperty(targetconfig, :window) "condition violated: targetconfig=$(typeof(targetconfig)) must provide a positive window field"
        window = Int(getproperty(targetconfig, :window))
        @assert window > 0 "condition violated: targetconfig.window=$(window) must be > 0"
        (verbosity >= 2) && println("log folder: $(EnvConfig.logfolder())")
        (verbosity >= 2) && println("data range: $startdt - $enddt")
        (verbosity >= 2) && println("bounds quality window=$(window)")
        (verbosity >= 2) && println("featuresconfig=$(Features.describe(featconfig))")
        (verbosity >= 2) && println("targetsconfig=$(Targets.describe(targetconfig))")
        return new(configname, folder, featconfig, targetconfig, regressormodel, tradingstrategy, startdt, enddt, opmode, partitionconfig, coins)
    end
end
cfg = nothing # to be set to a BoundsEstimatorConfig instance in main
retrain = false

include("optimizationconfigs.jl")

"""
returns targets as DataFrame with columns :centertarget and :widthtarget aligned to features and ohlcv dataframe rows.
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 4) && println("$(EnvConfig.now()) target calculation from $(fot[begin]) until $(fot[end])")
    Targets.setbase!(trgcfg, ohlcv)
    targets = Targets.centerwidth(trgcfg, fot[begin], fot[end]; relpricediff=true)
    # Targets.labeldistribution(targets)
    @assert size(features, 1) == size(targets, 1) "size(features, 1)=$(size(features, 1)) != size(targets, 1)=$(size(targets, 1))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))
    return targets
end

function _persist_coin_featuretarget_cache(coin::AbstractString, coinresultsdf, coinfeaturesdf; folderpath=EnvConfig.logfolder())::Bool
    @assert isnothing(coinresultsdf) == isnothing(coinfeaturesdf) "unexpected mismatch of coinresultsdf and coinfeaturesdf existence for coin=$(coin) with coinresultsdf existence $(isnothing(coinresultsdf)) and coinfeaturesdf existence $(isnothing(coinfeaturesdf))"
    if isnothing(coinresultsdf) || (size(coinresultsdf, 1) == 0)
        (verbosity >= 3) && println("skipping $coin due to empty results")
        return false
    end
    @assert size(coinresultsdf, 1) == size(coinfeaturesdf, 1) "unexpected mismatch of coinresultsdf and coinfeaturesdf size with coinresultsdf size $(size(coinresultsdf, 1)) and coinfeaturesdf size $(size(coinfeaturesdf, 1))"
    EnvConfig.savedf(coinresultsdf, resultsfilename(coin); folderpath=folderpath)
    EnvConfig.savedf(coinfeaturesdf, featuresfilename(coin); folderpath=folderpath)
    return true
end

"Returns the new rangeid after processing the given coin and its ranges. If it is unchanged then nothing was processed for the coin and it was skipped due to empty ranges or results."
function getfeaturestargets(cfg::BoundsEstimatorConfig, coinix, rangeid, samplesets)
    levels = unique(samplesets)
    coinresultsdf = coinfeaturesdf = nothing
    coin = cfg.coins[coinix]
    (verbosity >= 3) && println("calculating $coin ($coinix/$(length(cfg.coins))) liquid ranges, features and targets")
    ohlcv = Ohlcv.read(coin)
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    cfg.startdt = isnothing(cfg.startdt) ? ot[begin] : cfg.startdt
    startix = Ohlcv.rowix(ot, cfg.startdt)
    cfg.enddt = isnothing(cfg.enddt) ? ot[end] : cfg.enddt
    endix = Ohlcv.rowix(ot, cfg.enddt)
    @assert startix < endix "unexpected startix $startix >= endix $endix for $coin with startdt $(cfg.startdt) and enddt $(cfg.enddt)"
    rv = Ohlcv.liquiditycheck(Ohlcv.ohlcvview(ohlcv, startix:endix))

    for rngix in eachindex(rv) # rng indices are related to the ohlcvview dataframe rows
        rng = rv[rngix]
        rng = rng .+ (startix - 1) # adjust to complete ohlcv dataframe row indices
        if rng[end] - rng[begin] > 0
            (verbosity >= 2) && print("$(EnvConfig.now()) calculating features and targets for $coin ($coinix/$(length(cfg.coins))) range ($rngix/$(length(rv))) $rng from $(ot[rng[begin]]) until $(ot[rng[end]]) with $(rng[end] - rng[begin]) samples                \r")
            (verbosity >= 3) && println()
            rngdf = Ohlcv.dataframe(ohlcv)[rng, :]
            rngohlcv = Ohlcv.ohlcvview(ohlcv, rng)
            Features.setbase!(cfg.featconfig, rngohlcv, usecache=true)
            rngfeatures = Features.features(cfg.featconfig)
            rngresults = calctargets!(cfg.targetconfig, cfg.featconfig)
            @assert size(rngresults, 1) == size(rngfeatures, 1) == size(rngresults, 1) "unexpected mismatch of targets length $(size(rngresults, 1)), features size $(size(rngfeatures, 1)) and ohlcv size $(size(rngresults, 1)) for $(ohlcv.base) range $rng"

            rngresults = hcat(rngresults, rngdf[!, [:opentime, :high, :low, :close, :pivot]]) 
            rngresults[:, :rangeid] .= 0
            rngresults[:, :set] = CategoricalVector(fill(samplesets[1], size(rngfeatures, 1)), levels=levels) # arbitrary value to initialize the column with correct type
            allowmissing!(rngresults, :set)
            rngresults[:, :set] .= missing # initialize with missing to be able to check later if all rows were assigned to a set
            rngresults[:, :coin] = CategoricalVector(fill(coin, size(rngfeatures, 1)), levels=cfg.coins)
            psets = Classify.setpartitions(1:size(rngresults, 1), samplesets, partitionsize=cfg.partitionconfig.partitionsize, gapsize=cfg.partitionconfig.gapsize, minpartitionsize=cfg.partitionconfig.minpartitionsize, maxpartitionsize=cfg.partitionconfig.maxpartitionsize)
            # (verbosity >= 4) && println("$coin length(psets)=$(length(psets)) rng=$rng") #  psets=$psets

            for (pssettype, psrng) in psets # psrng (partition set ranges) is a vector of row ranges with indices that are related to rngresults rows (not to ohlcv dataframe rows)
                rngresults[psrng, :rangeid] .= rangeid
                rngresults[psrng, :set] .= pssettype
                rangeid += 1
            end
            usesamplesmask = .!ismissing.(rngresults[!, :set])
            rngresults = rngresults[usesamplesmask, :]
            rngfeatures = rngfeatures[usesamplesmask, :]
            coinresultsdf = isnothing(coinresultsdf) ? rngresults : vcat(coinresultsdf, rngresults)
            coinfeaturesdf = isnothing(coinfeaturesdf) ? rngfeatures : vcat(coinfeaturesdf, rngfeatures)
        else
            @error "unexpected zero length range for " ohlcv.base rng rv
        end
    end
    ohlcv = ot = rngohlcv = rngresults = rngfeatures = nothing # free memory
    _persist_coin_featuretarget_cache(coin, coinresultsdf, coinfeaturesdf)
    coinfeaturesdf = coinresultsdf = nothing # free memory
    return rangeid
end

function _load_featuretarget_pair(coin::AbstractString)
    resultsdf = EnvConfig.readdf(resultsfilename(coin))
    featuresdf = EnvConfig.readdf(featuresfilename(coin))
    @assert isnothing(resultsdf) == isnothing(featuresdf) "unexpected mismatch of resultsdf and featuresdf existence for coin=$(coin) with resultsdf existence $(isnothing(resultsdf)) and featuresdf existence $(isnothing(featuresdf))"

    if !isnothing(resultsdf)
        resultsdf = DataFrame(resultsdf; copycols=true)
        featuresdf = DataFrame(featuresdf; copycols=true)
        if :sampleix in propertynames(resultsdf)
            select!(resultsdf, Not(:sampleix))
        end
        if !has_current_target_format(resultsdf)
            @warn "ignoring stale bounds feature/target cache with outdated format" coin=coin expected=BOUNDS_RATIO_FORMAT names=names(resultsdf)
            EnvConfig.deletefolder(resultsfilename(coin))
            EnvConfig.deletefolder(featuresfilename(coin))
            return nothing, nothing
        elseif !issorted(resultsdf[!, :rangeid])
            @warn "ignoring stale bounds feature/target cache with non-monotonic rangeid ordering" coin=coin minimum_rangeid=minimum(resultsdf[!, :rangeid]) maximum_rangeid=maximum(resultsdf[!, :rangeid])
            EnvConfig.deletefolder(resultsfilename(coin))
            EnvConfig.deletefolder(featuresfilename(coin))
            return nothing, nothing
        end
        if :bounds_target_format in propertynames(resultsdf)
            select!(resultsdf, Not(:bounds_target_format))
        end
        if :set in propertynames(resultsdf)
            setcol = resultsdf[!, :set]
            if !(setcol isa CategoricalVector) && !(Base.nonmissingtype(eltype(setcol)) <: CategoricalValue)
                resultsdf[!, :set] = CategoricalVector(string.(setcol), levels=settypes())
            end
        end
        @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of resultsdf and featuresdf size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1)) for coin=$(coin)"
    end

    return resultsdf, featuresdf
end

function _featuretarget_cachefiles(cfg::BoundsEstimatorConfig; include_results::Bool=true, include_features::Bool=true, coins::AbstractVector{<:AbstractString}=cfg.coins)
    files = String[]
    for coin in coins
        if include_results && EnvConfig.isfolder(resultsfilename(coin))
            push!(files, resultsfilename(coin))
        end
        if include_features && EnvConfig.isfolder(featuresfilename(coin))
            push!(files, featuresfilename(coin))
        end
    end
    return files
end

function _concat_coin_featuretarget_caches(cfg::BoundsEstimatorConfig, coins::AbstractVector{<:AbstractString}=cfg.coins)
    resultparts = DataFrame[]
    featureparts = DataFrame[]
    cachedcoins = String[]

    for coin in coins
        hasresults = EnvConfig.isfolder(resultsfilename(coin))
        hasfeatures = EnvConfig.isfolder(featuresfilename(coin))
        @assert hasresults == hasfeatures "unexpected mismatch of coin-specific results/features cache existence for coin=$(coin) with hasresults=$(hasresults) and hasfeatures=$(hasfeatures)"
        if hasresults
            coinresultsdf, coinfeaturesdf = _load_featuretarget_pair(coin)
            if !isnothing(coinresultsdf) && (size(coinresultsdf, 1) > 0)
                push!(resultparts, coinresultsdf)
                push!(featureparts, coinfeaturesdf)
                push!(cachedcoins, coin)
            end
        end
    end

    if isempty(resultparts)
        return nothing, nothing, String[]
    end

    resultsdf = length(resultparts) == 1 ? resultparts[1] : vcat(resultparts...; cols=:union)
    featuresdf = length(featureparts) == 1 ? featureparts[1] : vcat(featureparts...; cols=:union)
    @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of concatenated results/features size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
    return resultsdf, featuresdf, cachedcoins
end

function getfeaturestargetsdf(cfg::BoundsEstimatorConfig)
    resultsdf = featuresdf = nothing
    (verbosity >= 2) && println("$(EnvConfig.now()) get features and targets                             ")

    resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg)
    if !isnothing(resultsdf)
        (verbosity >= 2) && println("$(EnvConfig.now()) using $(length(cachedcoins)) coin-specific cached bounds feature/target pairs")
    end

    if isnothing(resultsdf)
        rangeid = Int16(1) # shall be unique across coins
        samplesets = cfg.partitionconfig.samplesets
        samplesets = CategoricalArray(samplesets, levels=unique(samplesets))
        skippedcoins = String[]
        processedcoins = String[]
        for coinix in eachindex(cfg.coins)
            coin = cfg.coins[coinix]
            nextrangeid = getfeaturestargets(cfg, coinix, rangeid, samplesets)
            if nextrangeid > rangeid
                rangeid = nextrangeid
                push!(processedcoins, coin)
            else
                # if rangeid is unchanged then nothing was processed for the coin and it was skipped due to empty ranges or results
                push!(skippedcoins, coin)
            end
        end
        (verbosity >= 2) && println()
        resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg, processedcoins)
        println()
        (verbosity >= 2) && println("$(EnvConfig.now()) processed $(length(processedcoins)), skipped $(length(skippedcoins)) coins")
        (verbosity >= 3) && println("$(EnvConfig.now()) processed $processedcoins")
        (verbosity >= 3) && (length(skippedcoins) > 0) && println("skipped to process $skippedcoins due to no liquid ranges")
    end
    @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(featuresdf, 1) > 0) "unexpected resultsdf and featuresdf size with resultsdf size $(isnothing(resultsdf) ? "nothing" : size(resultsdf, 1)) and featuresdf size $(isnothing(featuresdf) ? "nothing" : size(featuresdf, 1))"
    return resultsdf, featuresdf
end

function df2features(featuresdf::AbstractDataFrame, cfg::BoundsEstimatorConfig)
    if size(featuresdf, 1) > 0
        features = @view featuresdf[!, Features.requestedcolumns(cfg.featconfig)]
        features = Array(features)  # change from df to array
        features = Float32.(permutedims(features, (2, 1)))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("typeof(features)=$(typeof(features)), size(features)=$(size(features)) for settype=$(settype)") 
        return features
    else
        return nothing
    end
end

function df2targets(resultsdf::AbstractDataFrame)
    if size(resultsdf, 1) > 0
        targets = Array(resultsdf[!, [:centertarget, :widthtarget]])  # change from df to array
        targets = Float32.(permutedims(targets, (2, 1)))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("typeof(targets)=$(typeof(targets)), size(targets)=$(size(targets)) for settype=$(settype)") 
        return targets
    else
        return nothing
    end
end

regressormenmonic(coins=nothing, coinix=nothing) = "mix_$(BOUNDS_RATIO_FORMAT)"

function getlatestregressor(cfg::BoundsEstimatorConfig)
    nn = cfg.regressormodel(Features.featurecount(cfg.featconfig), BOUNDS_LABELS, regressormenmonic()) # to get correct filename
    (verbosity >= 3) && println("getlatestregressor regressor file: $(Classify.nnfilename(nn.fileprefix)), isfile=$(isfile(Classify.nnfilename(nn.fileprefix)))")
    if isfile(Classify.nnfilename(nn.fileprefix))
        nn = Classify.loadnn(nn.fileprefix)
        (verbosity >= 3) && println("getlatestregressor loaded: nn=$(nn.fileprefix), labels=$(nn.labels) - regressor $(Classify.nnconverged(nn) ? "did" : "did not") converge")
    else
        (verbosity >= 3) && println("getlatestregressor new: nn=$(nn.fileprefix), labels=$(nn.labels)")
    end
    @assert !isnothing(nn)
    return nn
end

function isfreshcache(cachefile::AbstractString, dependencyfiles::AbstractVector{<:AbstractString})
    EnvConfig.tableexists(cachefile) || return false
    isempty(dependencyfiles) && return false
    cachepath = EnvConfig.tablepath(cachefile; format=:auto)
    cachemtime = stat(cachepath).mtime
    for depfile in dependencyfiles
        if EnvConfig.tableexists(depfile)
            deppath = EnvConfig.tablepath(depfile; format=:auto)
            if stat(deppath).mtime > cachemtime
                return false
            end
        end
    end
    return true
end

function showlosses(nn)
    println("$(EnvConfig.now()) evaluating regressor $(nn.mnemonic)")
    packetsize = length(nn.losses) > 20 ? floor(Int, length(nn.losses) / 20) : 1  # only display 20 lines of loss summary
    startp = lastlosses = nothing
    for i in eachindex(nn.losses)
        if i > firstindex(nn.losses)
            if (i % packetsize == 0) || (i == lastindex(nn.losses))
                plosses = mean(nn.losses[startp:i])
                println("epoch $startp-$i loss: $plosses  lossdiff: $((plosses-lastlosses)/lastlosses*100)%")
                startp = i+1
                lastlosses = plosses
            end
        else
            println("loss: $(nn.losses[i])")
            startp = i+1
            lastlosses = nn.losses[i]
        end
    end
end

function getregressor(cfg::BoundsEstimatorConfig)
    nn = getlatestregressor(cfg)
    if !Classify.isadapted(nn) || (!Classify.nnconverged(nn) && retrain)
        println("$(EnvConfig.now()) adapting one mix regressor for all coins")
        # if regressor file does not exist then create one
        resultsdf, featuresdf = getfeaturestargetsdf(cfg) 
        featuresdf = @view featuresdf[(resultsdf[!, :set] .== "train"), :]
        resultsdf = @view resultsdf[(resultsdf[!, :set] .== "train"), :]
        features = df2features(featuresdf, cfg)
        targets = df2targets(resultsdf)
        (verbosity >= 3) && println("$(EnvConfig.now()) size(featuresdf)=$(isnothing(featuresdf) ? "nothing" : size(featuresdf)), size(resultsdf)=$(isnothing(resultsdf) ? "nothing" : size(resultsdf)), size(features)=$(isnothing(features) ? "nothing" : size(features)), size(targets)=$(isnothing(targets) ? "nothing" : size(targets)) for training bounds regressor"  )
        if isnothing(targets) || isnothing(features) || (size(targets, 2) == 0) || (size(features, 2) == 0)
            return nothing
        end
        resultsdf = featuresdf = nothing # free memory
        Classify.adaptboundsregressor!(nn, features, targets)
        (verbosity >= 3) && showlosses(nn)
        if isnothing(nn)
            # no adaptation took place
            return nothing
        else
            # EnvConfig.savebackup(Classify.nnfilename(nn.fileprefix))
            Classify.savenn(nn)
        end
        println("$(EnvConfig.now()) finished adapting mix regressor - regressor $(Classify.nnconverged(nn) ? "did" : "did not") converge")
    end
    return nn
end

"Returns regression predictions for center/width and derived low/high bounds aligned to results rows."
function getboundspredictionsdf(cfg::BoundsEstimatorConfig)
    function predictbounds(nn::Classify.NN, features::AbstractMatrix)
        yraw = nn.model(Float32.(features))
        centerpred = vec(yraw[1, :]) # relative center can be positive or negative depending on whether predicted center is above or below pivot, but should be finite and not NaN
        widthpred = vec(clamp.(yraw[2, :], 0f0, Inf32)) # width must be non-negative as it represents an absolute distance, but should be finite and not NaN
        return DataFrame(centerpred=centerpred, widthpred=widthpred)
    end

    predictionsdf = EnvConfig.readdf(predictionsfilename())
    depfiles = _featuretarget_cachefiles(cfg)
    if !isnothing(predictionsdf) && !has_current_prediction_format(predictionsdf)
        @warn "ignoring stale predictions cache with legacy marker or missing prediction columns; names=$(names(predictionsdf))"
        EnvConfig.deletefolder(predictionsfilename())
        predictionsdf = nothing
    end
    if !isnothing(predictionsdf) && !isfreshcache(predictionsfilename(), depfiles)
        @warn "ignoring stale bounds prediction cache; rebuilding from newer coin-specific feature/target caches"
        EnvConfig.deletefolder(predictionsfilename())
        predictionsdf = nothing
    end
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        nn = getregressor(cfg)
        if isnothing(nn)
            return nothing
        end
        resultsdf, featuresdf = getfeaturestargetsdf(cfg) 
        (verbosity >= 2) && print("$(EnvConfig.now()) get bounds predictions                             \r")
        (verbosity >= 3) && println()
        if (size(resultsdf, 1) > 0)
            features = df2features(featuresdf, cfg)
            predictionsdf = predictbounds(nn, features)
            @assert size(predictionsdf, 1) == size(featuresdf, 1) == size(resultsdf, 1) "size(predictionsdf, 1)=$(size(predictionsdf, 1)) != size(featuresdf, 1)=$(size(featuresdf, 1)) != size(resultsdf, 1)=$(size(resultsdf, 1)) for mix"
            EnvConfig.savedf(predictionsdf, predictionsfilename())
        end
    end
    if !isnothing(predictionsdf) && size(predictionsdf, 1) > 0
        # now we have the predictions -> add them to resultsdf
        resultsdf, _ = getfeaturestargetsdf(cfg)
        if !isnothing(resultsdf)
            resultsdf = DataFrame(resultsdf; copycols=true)
            if :bounds_target_format in propertynames(resultsdf)
                select!(resultsdf, Not(:bounds_target_format))
            end
            if :sampleix in propertynames(resultsdf)
                select!(resultsdf, Not(:sampleix))
            end
        end
        @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(predictionsdf, 1) > 0) "size mismatch: size(resultsdf, 1)=$(isnothing(resultsdf) ? "nothing" : size(resultsdf, 1)), size(predictionsdf, 1)=$(size(predictionsdf, 1))"
        resultsdf[:, :widthpred] = predictionsdf[!, :widthpred]
        resultsdf[:, :centerpred] = predictionsdf[!, :centerpred]
    else
        resultsdf = nothing
    end
    return resultsdf
end


"""
Create an LSTM contract object from a merged dataframe that already contains
trend probability columns and bounds regressor outputs.

Required by default:
- bounds: `:centerpred`, `:widthpred`
- admin: `:target`, `:set`, `:rangeid`
- trend probabilities: provided via `trendprobcols`

A temporary in-memory row index is generated when no explicit row-id column is present.
"""
function build_lstm_contract(merged_df::AbstractDataFrame; trendprobcols::Vector{Symbol})
    @assert length(trendprobcols) > 0 "trendprobcols must not be empty"

    contractdf = merged_df
    if :sampleix in propertynames(contractdf)
        rixcol = :sampleix
    elseif :rowix in propertynames(contractdf)
        rixcol = :rowix
    else
        contractdf = copy(merged_df)
        contractdf[!, :rowix] = Int32.(1:size(contractdf, 1))
        rixcol = :rowix
    end

    return Classify.lstm_bounds_trend_features(
        contractdf;
        trendprobcols=trendprobcols,
        centercol=:centerpred,
        widthcol=:widthpred,
        targetcol=:target,
        setcol=:set,
        rangeidcol=:rangeid,
        rixcol=rixcol,
    )
end

"""
Convenience wrapper to build an LSTM contract from current bounds predictions.

Note: this requires that `getboundspredictionsdf(cfg)` already includes the trend
probability columns listed in `trendprobcols`. In the standard pipeline, trend
outputs are prepared and merged in `TrendLstm`.
"""
function get_lstm_contract(cfg::BoundsEstimatorConfig; trendprobcols::Vector{Symbol})
    pdf = getboundspredictionsdf(cfg)
    if isnothing(pdf) || (size(pdf, 1) == 0)
        return nothing
    end
    return build_lstm_contract(pdf; trendprobcols=trendprobcols)
end

"""
what is bounds quality? context: the upper bound is a long sell or short buy limit price and the lower bound is a long buy or short sell limit price.
if the price stays within the upper and lower limit across the corresponding window sample span then the bounds are not good because the price does not reach the limits to trigger a buy or sell action.
- get deal done (buy/sell) before trend changes to the worse
  - how many bound estimations matched within the corresponding sample related window (that changes position with every sample?
  - what was the gain vs. close price when estimation was done?
- for those that did not match: how long did it take (sample after the sample under consideration) to exceed bounds and how many percetange lost?
"""
function getboundsqualitydf(cfg::BoundsEstimatorConfig)
    pdf = getboundspredictionsdf(cfg)
    if isnothing(pdf) || size(pdf, 1) == 0
        return (high=DataFrame(), low=DataFrame())
    end
    reqcols = [:centerpred, :widthpred, :centertarget, :widthtarget, :pivot, :high, :low, :close, :set, :coin, :rangeid, :opentime]
    missingcols = [c for c in reqcols if !(c in propertynames(pdf))]
    @assert isempty(missingcols) "missing required columns in predictions/results dataframe: missingcols=$(missingcols), names=$(names(pdf))"

    n = size(pdf, 1)
    @assert hasproperty(cfg.targetconfig, :window) "condition violated: targetconfig=$(typeof(cfg.targetconfig)) must provide a positive window field"
    window = Int(getproperty(cfg.targetconfig, :window))
    @assert window > 0 "invalid targetconfig.window=$(window)"

    centerpred = Float32.(pdf[!, :centerpred])
    widthpred = Float32.(pdf[!, :widthpred])
    centertarget = Float32.(pdf[!, :centertarget])
    widthtarget = Float32.(pdf[!, :widthtarget])
    pivotcol = Float32.(pdf[!, :pivot])
    highcol = Float32.(pdf[!, :high])
    lowcol = Float32.(pdf[!, :low])
    closecol = Float32.(pdf[!, :close])
    setcol = string.(pdf[!, :set])
    setnames = unique(setcol)
    setlookup = Dict(name => ix for (ix, name) in enumerate(setnames))
    setix = Int[setlookup[s] for s in setcol]
    nsets = length(setnames)

    pred_center_abs = (1f0 .+ centerpred) .* pivotcol
    pred_width_abs = widthpred .* pivotcol
    pred_lower = clamp.(pred_center_abs .- pred_width_abs ./ 2f0, 0f0, Inf32)
    pred_upper = clamp.(pred_center_abs .+ pred_width_abs ./ 2f0, 0f0, Inf32)
    target_center_abs = (1f0 .+ centertarget) .* pivotcol
    target_width_abs = widthtarget .* pivotcol

    err_center = abs.(pred_center_abs .- target_center_abs)
    err_width = abs.(pred_width_abs .- target_width_abs)

    rows = zeros(Int, nsets)
    sum_err_center = zeros(Float64, nsets)
    sum_err_width = zeros(Float64, nsets)

    signed_high_distance_vs_close_pct(bound::Float32, close::Float32)::Float32 = begin
        close == 0f0 && return 0f0
        return 100f0 * abs(bound - close) / abs(close)
    end

    signed_low_distance_vs_close_pct(bound::Float32, close::Float32)::Float32 = begin
        close == 0f0 && return 0f0
        return -100f0 * abs(bound - close) / abs(close)
    end

    high_hit_count = zeros(Int, nsets)
    sum_samples_to_first_high_hit = zeros(Float64, nsets)
    count_samples_to_first_high_hit = zeros(Int, nsets)
    sum_high_hit_distance_vs_close_pct = zeros(Float64, nsets)
    count_high_hit_distance_vs_close_pct = zeros(Int, nsets)
    sum_samples_to_first_high_exceed_after_window = zeros(Float64, nsets)
    count_samples_to_first_high_exceed_after_window = zeros(Int, nsets)
    sum_high_hit_distance_vs_close_pct_after_window = zeros(Float64, nsets)
    count_high_hit_distance_vs_close_pct_after_window = zeros(Int, nsets)

    low_hit_count = zeros(Int, nsets)
    sum_samples_to_first_low_hit = zeros(Float64, nsets)
    count_samples_to_first_low_hit = zeros(Int, nsets)
    sum_low_hit_distance_vs_close_pct = zeros(Float64, nsets)
    count_low_hit_distance_vs_close_pct = zeros(Int, nsets)
    sum_samples_to_first_low_exceed_after_window = zeros(Float64, nsets)
    count_samples_to_first_low_exceed_after_window = zeros(Int, nsets)
    sum_low_hit_distance_vs_close_pct_after_window = zeros(Float64, nsets)
    count_low_hit_distance_vs_close_pct_after_window = zeros(Int, nsets)

    @inbounds for i in eachindex(setix)
        s = setix[i]
        rows[s] += 1
        sum_err_center[s] += err_center[i]
        sum_err_width[s] += err_width[i]
    end

    function build_extrema_trees!(max_tree::Vector{Float32}, min_tree::Vector{Float32}, gidx, node::Int, left::Int, right::Int)
        if left == right
            gi = gidx[left]
            max_tree[node] = highcol[gi]
            min_tree[node] = lowcol[gi]
            return nothing
        end
        mid = (left + right) >>> 1
        leftnode = node << 1
        rightnode = leftnode + 1
        build_extrema_trees!(max_tree, min_tree, gidx, leftnode, left, mid)
        build_extrema_trees!(max_tree, min_tree, gidx, rightnode, mid + 1, right)
        max_tree[node] = max(max_tree[leftnode], max_tree[rightnode])
        min_tree[node] = min(min_tree[leftnode], min_tree[rightnode])
        return nothing
    end

    function first_high_hit(max_tree::Vector{Float32}, threshold::Float32, qleft::Int, qright::Int, node::Int, left::Int, right::Int)::Int
        if (qright < left) || (right < qleft) || (max_tree[node] < threshold)
            return 0
        elseif left == right
            return left
        end
        mid = (left + right) >>> 1
        leftnode = node << 1
        hitix = first_high_hit(max_tree, threshold, qleft, qright, leftnode, left, mid)
        return hitix > 0 ? hitix : first_high_hit(max_tree, threshold, qleft, qright, leftnode + 1, mid + 1, right)
    end

    function first_low_hit(min_tree::Vector{Float32}, threshold::Float32, qleft::Int, qright::Int, node::Int, left::Int, right::Int)::Int
        if (qright < left) || (right < qleft) || (min_tree[node] > threshold)
            return 0
        elseif left == right
            return left
        end
        mid = (left + right) >>> 1
        leftnode = node << 1
        hitix = first_low_hit(min_tree, threshold, qleft, qright, leftnode, left, mid)
        return hitix > 0 ? hitix : first_low_hit(min_tree, threshold, qleft, qright, leftnode + 1, mid + 1, right)
    end

    for g in groupby(pdf, [:coin, :rangeid])
        gidx = parentindices(g)[1]
        glen = length(gidx)
        max_tree = fill(-Inf32, max(1, 4 * glen))
        min_tree = fill(Inf32, max(1, 4 * glen))
        build_extrema_trees!(max_tree, min_tree, gidx, 1, 1, glen)

        @inbounds for localix in 1:glen
            firstfuture = localix + 1
            firstfuture > glen && continue

            i = gidx[localix]
            s = setix[i]
            lastfuture = min(localix + window, glen)
            close_i = closecol[i]
            up_i = pred_upper[i]
            low_i = pred_lower[i]

            high_hit_localix = first_high_hit(max_tree, up_i, firstfuture, lastfuture, 1, 1, glen)
            low_hit_localix = first_low_hit(min_tree, low_i, firstfuture, lastfuture, 1, 1, glen)

            if high_hit_localix > 0
                high_hit_count[s] += 1
                sum_samples_to_first_high_hit[s] += high_hit_localix - localix
                count_samples_to_first_high_hit[s] += 1
                sum_high_hit_distance_vs_close_pct[s] += signed_high_distance_vs_close_pct(up_i, close_i)
                count_high_hit_distance_vs_close_pct[s] += 1
            elseif lastfuture < glen
                late_high_hit_localix = first_high_hit(max_tree, up_i, lastfuture + 1, glen, 1, 1, glen)
                if late_high_hit_localix > 0
                    sum_samples_to_first_high_exceed_after_window[s] += late_high_hit_localix - localix
                    count_samples_to_first_high_exceed_after_window[s] += 1
                    sum_high_hit_distance_vs_close_pct_after_window[s] += signed_high_distance_vs_close_pct(up_i, close_i)
                    count_high_hit_distance_vs_close_pct_after_window[s] += 1
                end
            end

            if low_hit_localix > 0
                low_hit_count[s] += 1
                sum_samples_to_first_low_hit[s] += low_hit_localix - localix
                count_samples_to_first_low_hit[s] += 1
                sum_low_hit_distance_vs_close_pct[s] += signed_low_distance_vs_close_pct(low_i, close_i)
                count_low_hit_distance_vs_close_pct[s] += 1
            elseif lastfuture < glen
                late_low_hit_localix = first_low_hit(min_tree, low_i, lastfuture + 1, glen, 1, 1, glen)
                if late_low_hit_localix > 0
                    sum_samples_to_first_low_exceed_after_window[s] += late_low_hit_localix - localix
                    count_samples_to_first_low_exceed_after_window[s] += 1
                    sum_low_hit_distance_vs_close_pct_after_window[s] += signed_low_distance_vs_close_pct(low_i, close_i)
                    count_low_hit_distance_vs_close_pct_after_window[s] += 1
                end
            end
        end
    end

    mean_or_missing(sumv::Float64, countv::Int) = countv == 0 ? missing : (sumv / countv)

    highdf = DataFrame(
        set=setnames,
        mae_center=sum_err_center ./ rows,
        mae_width=sum_err_width ./ rows,
        high_hit_within_window_pct=Float32.(100 .* high_hit_count ./ rows),
        mean_samples_to_first_high_hit=[mean_or_missing(sum_samples_to_first_high_hit[ix], count_samples_to_first_high_hit[ix]) for ix in eachindex(setnames)],
        mean_high_hit_distance_vs_close_pct=[mean_or_missing(sum_high_hit_distance_vs_close_pct[ix], count_high_hit_distance_vs_close_pct[ix]) for ix in eachindex(setnames)],
        mean_samples_to_first_high_exceed_after_window=[mean_or_missing(sum_samples_to_first_high_exceed_after_window[ix], count_samples_to_first_high_exceed_after_window[ix]) for ix in eachindex(setnames)],
        mean_high_hit_distance_vs_close_pct_after_window=[mean_or_missing(sum_high_hit_distance_vs_close_pct_after_window[ix], count_high_hit_distance_vs_close_pct_after_window[ix]) for ix in eachindex(setnames)],
        rows=rows,
    )
    lowdf = DataFrame(
        set=setnames,
        mae_center=sum_err_center ./ rows,
        mae_width=sum_err_width ./ rows,
        low_hit_within_window_pct=Float32.(100 .* low_hit_count ./ rows),
        mean_samples_to_first_low_hit=[mean_or_missing(sum_samples_to_first_low_hit[ix], count_samples_to_first_low_hit[ix]) for ix in eachindex(setnames)],
        mean_low_hit_distance_vs_close_pct=[mean_or_missing(sum_low_hit_distance_vs_close_pct[ix], count_low_hit_distance_vs_close_pct[ix]) for ix in eachindex(setnames)],
        mean_samples_to_first_low_exceed_after_window=[mean_or_missing(sum_samples_to_first_low_exceed_after_window[ix], count_samples_to_first_low_exceed_after_window[ix]) for ix in eachindex(setnames)],
        mean_low_hit_distance_vs_close_pct_after_window=[mean_or_missing(sum_low_hit_distance_vs_close_pct_after_window[ix], count_low_hit_distance_vs_close_pct_after_window[ix]) for ix in eachindex(setnames)],
        rows=rows,
    )
    return (high=highdf, low=lowdf)
end


function safe(f, v; default=missing)
    v = skipmissing(v)
    isempty(v) ? default : f(v)
end

function introspection(cfg::BoundsEstimatorConfig)
    BoundsEstimator.verbosity = 2
    Ohlcv.verbosity = 1
    CryptoXch.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    resultsdf, featdf, cachedcoins = _concat_coin_featuretarget_caches(cfg)
    if !isnothing(featdf) && (size(featdf, 1) > 0)
        println("coin-specific bounds features caches for $(length(cachedcoins)) coins -> size(featdf) = $(size(featdf))")
        println("describe(featdf, :all)=$(describe(featdf, :all))")
    else
        println("No coin-specific bounds features cache found in $(EnvConfig.logfolder())")
    end
    if !isnothing(resultsdf) && (size(resultsdf, 1) > 0)
        println("coin-specific bounds results caches for $(length(cachedcoins)) coins -> size(resultsdf) = $(size(resultsdf))")
        println("describe(resultsdf, :all)=$(describe(resultsdf, :all))")
        println("$(unique(resultsdf[!, :coin])) processable coins")
        println("rangeid sorted = $(issorted(resultsdf[!, :rangeid]))")
    else
        println("No coin-specific bounds results cache found in $(EnvConfig.logfolder())")
    end
    preddf = EnvConfig.readdf(predictionsfilename())
    if !isnothing(preddf) && (size(preddf, 1) > 0)
        println("$(predictionsfilename()): size(preddf) = $(size(preddf))")
        println("describe(preddf, :all)=$(describe(preddf, :all))")
    else
        println("No results file found in $(EnvConfig.logfolder()) - size(preddf)=$(isnothing(preddf) ? "nothing" : size(preddf))")
    end
end

"""
Return whether the CLI arguments request the help output.
"""
function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

function buildcfg(args::Vector{String}, allowedcoins::Vector{String}, startdt::DateTime, enddt::DateTime)
    configref = _argvalue(args, "config", "001")
    basecfg = boundsestimatorconfig(configref)
    configname = _argvalue(args, "configname", string(basecfg.configname))
    folder = _argvalue(args, "folder", "Bounds-$configname-$(EnvConfig.configmode)")
    mergedcfg = merge(basecfg, (configname=configname, folder=folder))
    return BoundsEstimatorConfig(; mergedcfg..., coins=allowedcoins, startdt=startdt, enddt=enddt)
end

function _wants_help(args::Vector{String})::Bool
    for arg in args
        normalized = lowercase(strip(arg))
        if normalized in ("help", "--help", "-h")
            return true
        elseif startswith(normalized, "help=")
            value = split(normalized, "="; limit=2)[2]
            return value in ("1", "true", "yes", "on")
        end
    end
    return false
end

"""
Return CLI help text for `BoundsEstimator.jl`.
"""
function boundsestimatorhelp()::String
    return """
Usage:
  julia --project=. scripts/BoundsEstimator.jl [help] [test|train] [inspect] [special] [retrain] [key=value ...]

Flag parameters:
  help, --help, -h
      Show this message and exit.
      Default: false

  test
      Use `EnvConfig.init(test)` with `testcoins()`.
      Default: true

  train
      Use `EnvConfig.init(training)` with `traincoins()`.
      Default: false

  inspect
      Print cached features, targets, and predictions without running the estimator.
      Default: false

  special
      Enable special/debug mode. Currently no special task is defined and this also enables `inspect`.
      Default: false

  retrain
      Retrain non-converged regressors instead of reusing them.
      Default: false

Key=value parameters:
  config=<configname>
      Bounds preset from `BOUNDS_ESTIMATOR_CONFIGS` in `optimizationconfigs.jl`.
      Default: `001`

  configname=<name>
      Optional output name override.
      Default: same as `config`

  folder=<name>
      Output subfolder.
      Default: `Bounds-<configname>-$(EnvConfig.configmode)`

Fixed date defaults:
  train startdt: `2017-11-17T20:56:00`
  test startdt: `2025-01-17T20:56:00`
  enddt: `2025-08-10T15:00:00`
"""
end

"""
Run the `BoundsEstimator` script with the given CLI arguments.
"""
function main(args::Vector{String}=ARGS)
    if _wants_help(args)
        println(boundsestimatorhelp())
        return nothing
    end

    # startdt = nothing  # means use all what is stored as canned data
    # enddt = nothing  # means use all what is stored as canned data
    startdt = DateTime("2017-11-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$(args)")
    global retrain = "retrain" in args
    retrain && println("retrain mode activated - existing regressors that did not converge will be overwritten")
    testmode = true
    testmode = "test" in args ? true : "train" in args ? false : testmode
    inspectonly = "inspect" in args
    specialonly = "special" in args

    global verbosity = 2
    allowedcoins = String[]

    if testmode
        global verbosity = 2
        Ohlcv.verbosity = 1 # 3
        CryptoXch.verbosity = 1 # 3
        Features.verbosity = 1 # 3
        Targets.verbosity = 1 # 3
        EnvConfig.verbosity = 1
        Classify.verbosity = 1 # 3
        allowedcoins = testcoins()
        EnvConfig.init(test)
        startdt = DateTime("2025-01-17T20:56:00")
        enddt = DateTime("2025-08-10T15:00:00")
    else # training or production
        global verbosity = 2
        Ohlcv.verbosity = 1
        CryptoXch.verbosity = 1
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
        EnvConfig.init(training)
        allowedcoins = traincoins()
    end

    if specialonly
        Ohlcv.verbosity = 3
        CryptoXch.verbosity = 3
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
    end

    global cfg = buildcfg(args, allowedcoins, startdt, enddt)

    if specialonly
        # renamepredictionfiles([mk1config().folder, mk2config().folder, mk3config().folder, mk4config().folder, mk5config().folder])
        println("No special task defined")
    elseif inspectonly
        introspection(cfg)
    else
        pred = getboundspredictionsdf(cfg)
        if !isnothing(pred) && (size(pred, 1) > 0)
            println("$(EnvConfig.now()) bounds predictions rows=$(size(pred, 1))")
            println("$(EnvConfig.now()) bounds predictions sample=$(first(pred, min(5, size(pred, 1))))")
        end
        qdf = getboundsqualitydf(cfg)
        if size(qdf.high, 1) > 0
            println("$(EnvConfig.now()) bounds high quality by set: $(qdf.high)")
        end
        if size(qdf.low, 1) > 0
            println("$(EnvConfig.now()) bounds low quality by set: $(qdf.low)")
        end

        # distdf = getdistances(cfg)
        # if !isnothing(distdf) && (size(distdf, 1) > 0)
        #     println("size(distdf)=$(size(distdf))")
        #     println("describe(distdf)=$(describe(distdf))")
        #     # println(distdf[.!ismissing.(distdf[!, :tpdistnext]),:])
        #     distdfgroup = groupby(distdf, [:set, :trend])
        #     # println(distdfgroup)
        #     # diststatdf = combine(distdfgroup, :tpdistnext => (x -> safe(mean, x)) => :tpdistnext_mean, :tpdistnext => (x -> safe(std, x)) => :tpdistnext_std, :tpdistnext => (x -> (safe(count, x; default=0) / nrow)) => :tpdistnext_pct, :fpdistnext => (x -> safe(mean, x)) => :fpdistnext_mean, :fpdistnext => (x -> safe(std, x)) => :fpdistnext_std, :fpdistnext => (x -> (safe(count, x; default=0) / nrow)) => :fpdistnext_pct, :distfirst => (x -> safe(mean, x)) => :distfirst_mean, :distfirst => (x -> safe(std, x)) => :distfirst_std, :distfirst => (x -> (safe(count, x; default=0) / nrow)) => :distfirst_pct, :distlast => (x -> safe(mean, x)) => :distlast_mean, :distlast => (x -> safe(std, x)) => :distlast_std, :distlast => (x -> (safe(count, x; default=0) / nrow)) => :distlast_pct)
        #     diststatdf = combine(distdfgroup, :tpdistnext => (x -> safe(mean, x)) => :tpdistnext_mean, :tpdistnext => (x -> safe(median, x)) => :tpdistnext_median, :tpdistnext => (x -> safe(std, x)) => :tpdistnext_std, :fpdistnext => (x -> safe(mean, x)) => :fpdistnext_mean, :fpdistnext => (x -> safe(median, x)) => :fpdistnext_median, :fpdistnext => (x -> safe(std, x)) => :fpdistnext_std, :distfirst => (x -> safe(mean, x)) => :distfirst_mean, :distfirst => (x -> safe(median, x)) => :distfirst_median, :distfirst => (x -> safe(std, x)) => :distfirst_std, :distlast => (x -> safe(mean, x)) => :distlast_mean, :distlast => (x -> safe(median, x)) => :distlast_median, :distlast => (x -> safe(std, x)) => :distlast_std)
        #     println("$(EnvConfig.now()) Distances: $(diststatdf)")
        # else
        #     println("$(EnvConfig.now()) no distance data available")
        # end
    end

    println("$(EnvConfig.now()) done @ $(cfg.folder)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

end # of BoundsEstimator
