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
softplus(x::Real) = log1p(exp(-abs(x))) + max(x, 0)

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
    pred_lower_abs = clamp.(pred_center_abs .- pred_width_abs ./ 2f0, 0f0, Inf)
    pred_upper_abs = clamp.(pred_center_abs .+ pred_width_abs ./ 2f0, 0f0, Inf)
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
        (verbosity >= 2) && println("log folder: $(EnvConfig.logfolder())")
        (verbosity >= 2) && println("data range: $startdt - $enddt")
        (verbosity >= 2) && println("featuresconfig=$(Features.describe(featconfig))")
        (verbosity >= 2) && println("targetsconfig=$(Targets.describe(targetconfig))")
        return new(configname, folder, featconfig, targetconfig, regressormodel, tradingstrategy, startdt, enddt, opmode, partitionconfig, coins)
    end
end
cfg = nothing # to be set to a BoundsEstimatorConfig instance in main

include("optimizationconfigs.jl")

"""
returns targets  as DataFrame with columns :lowbound, :highbound and :opentime aligned to features and ohlcv dataframe rows.
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 4) && println("$(EnvConfig.now()) target calculation from $(fot[begin]) until $(fot[end])")
    Targets.setbase!(trgcfg, ohlcv)
    targets = Targets.lowboundhighbound(trgcfg, fot[begin], fot[end])
    targets = Targets.lowhigh2centerwidth(targets[!,:lowbound], targets[!, :highbound])
    # Targets.labeldistribution(targets)
    @assert size(features, 1) == size(targets, 1) "size(features, 1)=$(size(features, 1)) != size(targets, 1)=$(size(targets, 1))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))
    return targets
end

"Returns the new rangeid after processing the given coin and its ranges. If it is unchanged then nothing was processed for the coin and it was skipped due to empty ranges or results."
function getfeaturestargets(cfg::BoundsEstimatorConfig, coin, rangeid, samplesets)
    levels = unique(samplesets)
    coinresultsdf = coinfeaturesdf = nothing
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
            rngresults[:, :coin] .= coin
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
    if size(coinresultsdf, 1) > 0
        @assert size(coinresultsdf, 1) == size(coinfeaturesdf, 1) "unexpected mismatch of coinresultsdf and coinfeaturesdf size with coinresultsdf size $(size(coinresultsdf, 1)) and coinfeaturesdf size $(size(coinfeaturesdf, 1))"
        EnvConfig.savedf(coinresultsdf, resultsfilename(coin))
        EnvConfig.savedf(coinfeaturesdf, featuresfilename(coin))
        coinfeaturesdf = coinresultsdf = nothing # free memory
        return rangeid
    else
        (verbosity >= 3) && println("skipping $coin due to empty results")
        return rangeid
    end
end

function getfeaturestargetsdf(cfg::BoundsEstimatorConfig)
    resultsdf = featuresdf = nothing
    (verbosity >= 2) && println("$(EnvConfig.now()) get features and targets                             ")
    if EnvConfig.isfolder(resultsfilename())
        resultsdf = EnvConfig.readdf(resultsfilename())
        featuresdf = EnvConfig.readdf(featuresfilename())
        @assert isnothing(resultsdf) == isnothing(featuresdf) "unexpected mismatch of resultsdf and featuresdf existence with resultsdf existence $(isnothing(resultsdf)) and featuresdf existence $(isnothing(featuresdf))"
        @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of resultsdf and featuresdf size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
    else
        rangeid = Int16(1) # shall be unique across coins
        samplesets = cfg.partitionconfig.samplesets
        samplesets = CategoricalArray(samplesets, levels=unique(samplesets))
        skippedcoins = String[]
        processedcoins = String[]
        for coinix in eachindex(cfg.coins)
            coin = cfg.coins[coinix]
            if getfeaturestargets(cfg, coin, rangeid, samplesets) > rangeid 
                push!(processedcoins, coin)
            else
                # if rangeid is unchanged then nothing was processed for the coin and it was skipped due to empty ranges or results
                push!(skippedcoins, coin)
            end
        end
        (verbosity >= 2) && println()
        if length(processedcoins) > 0
            @assert isnothing(featuresdf)
            for coinix in eachindex(processedcoins)
                coin = processedcoins[coinix]
                (verbosity >= 2) && print("$(EnvConfig.now()) concatenating $coin ($coinix/$(length(cfg.coins))) features                \r")
                (verbosity >= 3) && println()
                coinfeaturesdf = EnvConfig.readdf(featuresfilename(coin))
                @assert !isnothing(coinfeaturesdf)
                @assert size(coinfeaturesdf, 1) > 0 "unexpected empty features for $coin size(coinfeaturesdf, 1)=$(size(coinfeaturesdf, 1))"
                featuresdf = isnothing(featuresdf) ? coinfeaturesdf : vcat(featuresdf, coinfeaturesdf)
            end
            @assert (!isnothing(featuresdf) && (size(featuresdf, 1) > 0)) "unexpected inconsistency: length(processedcoins)=$(length(processedcoins)), (isnothing(featuresdf)=$(isnothing(featuresdf)) || (size(featuresdf, 1)=$((isnothing(featuresdf) ? "nothing" : (size(featuresdf, 1)))) == 0))"
            if size(featuresdf, 1) > 0
                EnvConfig.savedf(featuresdf, featuresfilename())
            end
            coinfeaturesdf = featuresdf = nothing # free memory
            (verbosity >= 2) && println()

            @assert isnothing(resultsdf)
            for coinix in eachindex(processedcoins)
                coin = processedcoins[coinix]
                (verbosity >= 2) && print("$(EnvConfig.now()) concatenating $coin ($coinix/$(length(cfg.coins))) targets/results                                            \r")
                (verbosity >= 3) && println()
                coinresultsdf = EnvConfig.readdf(resultsfilename(coin))
                @assert !isnothing(coinresultsdf)
                @assert size(coinresultsdf, 1) > 0 "unexpected empty results for $coin size(coinresultsdf, 1)=$(size(coinresultsdf, 1))"
                resultsdf = isnothing(resultsdf) ? coinresultsdf : vcat(resultsdf, coinresultsdf)
            end
            @assert (!isnothing(resultsdf) && (size(resultsdf, 1) > 0)) "unexpected inconsistency: length(processedcoins)=$(length(processedcoins)), (isnothing(resultsdf)=$(isnothing(resultsdf)) || (size(resultsdf, 1)=$((isnothing(resultsdf) ? "nothing" : (size(resultsdf, 1)))) == 0))"
            if size(resultsdf, 1) > 0
                resultsdf[:, :sampleix] = collect(1:size(resultsdf, 1)) # sampleix is a unique index for each sample in the complete resultsdf 
                EnvConfig.savedf(resultsdf, resultsfilename())
            end
            coinresultsdf = resultsdf = nothing # free memory
            (verbosity >= 2) && println()

            for coinix in eachindex(processedcoins)
                coin = processedcoins[coinix]
                (verbosity >= 2) && print("$(EnvConfig.now()) deleting $coin ($coinix/$(length(cfg.coins))) specific features and results                                   \r")
                (verbosity >= 3) && println()
                EnvConfig.deletefolder(resultsfilename(coin))
                EnvConfig.deletefolder(featuresfilename(coin))
            end
            resultsdf = EnvConfig.readdf(resultsfilename())
            featuresdf = EnvConfig.readdf(featuresfilename())
        end
        println()
        (verbosity >= 2) && println("$(EnvConfig.now()) processed $(length(processedcoins)), skipped $(length(skippedcoins)) coins")
        (verbosity >= 3) && println("$(EnvConfig.now()) processed $processedcoins")
        (verbosity >= 3) && (length(skippedcoins) > 0) && println("skipped to process $skippedcoins due to no liquid ranges")
    end
    @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(featuresdf, 1) > 0) "unexpected resultsdf and featuresdf size with resultsdf size $(isnothing(resultsdf) ? "nothing" : size(resultsdf, 1)) and featuresdf size $(isnothing(featuresdf) ? "nothing" : size(featuresdf, 1))"
    return resultsdf, featuresdf
end

function df2features(featuresdf::AbstractDataFrame, cfg::BoundsEstimatorConfig, settype=nothing)
    featuresdf = isnothing(settype) ? featuresdf : @view featuresdf[(featuresdf[!, :set] .== settype), :]
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

function df2targets(resultsdf::AbstractDataFrame, settype=nothing)
    resultsdf = isnothing(settype) ? resultsdf : @view resultsdf[(resultsdf[!, :set] .== settype), :]
    if size(resultsdf, 1) > 0
        targets = Array(resultsdf[!, [:centertarget, :widthtarget]])  # change from df to array
        targets = Float32.(permutedims(targets, (2, 1)))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("typeof(targets)=$(typeof(targets)), size(targets)=$(size(targets)) for settype=$(settype)") 
        return targets
    else
        return nothing
    end
end

regressormenmonic(coins=nothing, coinix=nothing) = "mix"

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
        features = df2features(featuresdf, cfg, "train")
        targets = df2targets(resultsdf, "train")
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
    if !isnothing(predictionsdf)
        @assert all([:centerpred, :widthpred] .∈ propertynames(predictionsdf)) "unexpected columns in predictionsdf with names=$(names(predictionsdf))"
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
            EnvConfig.savedf(resultsdf, predictionsfilename())
        end
    end
    if !isnothing(predictionsdf) && size(predictionsdf, 1) > 0
        # now we have the predictions -> add them to resultsdf
        @assert EnvConfig.isfolder(resultsfilename()) "unexpected missing resultsfile"
        resultsdf = EnvConfig.readdf(resultsfilename())
        @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(predictionsdf, 1) > 0) "size mismatch: size(resultsdf, 1)=$(snothing(resultsdf) ? "nothing" : size(resultsdf, 1)), size(predictionsdf, 1)=$(size(predictionsdf, 1))"
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
- admin: `:target`, `:set`, `:rangeid`, `:sampleix`
- trend probabilities: provided via `trendprobcols`
"""
function build_lstm_contract(merged_df::AbstractDataFrame; trendprobcols::Vector{Symbol})
    @assert length(trendprobcols) > 0 "trendprobcols must not be empty"
    return Classify.lstm_bounds_trend_features(
        merged_df;
        trendprobcols=trendprobcols,
        centercol=:centerpred,
        widthcol=:widthpred,
        targetcol=:target,
        setcol=:set,
        rangeidcol=:rangeid,
        rixcol=:sampleix,
    )
end

"""
Convenience wrapper to build an LSTM contract from current bounds predictions.

Note: this requires that `getboundspredictionsdf(cfg)` already includes the trend
probability columns listed in `trendprobcols`. In the standard pipeline, trend
outputs are prepared and merged in `TradeAdviceLstm`.
"""
function get_lstm_contract(cfg::BoundsEstimatorConfig; trendprobcols::Vector{Symbol})
    pdf = getboundspredictionsdf(cfg)
    if isnothing(pdf) || (size(pdf, 1) == 0)
        return nothing
    end
    return build_lstm_contract(pdf; trendprobcols=trendprobcols)
end

"""
what is bounds quality?
- get deal done (buy/sell) before trend changes to the worse
  - how many bound estimations matched within window?
  - how many bound estimations matched with the every sample changing window?
  - what was the gain vs. close price when estimation was done?
- for those that did not match: how long did it take and how many percetange lost?
"""
function getboundsqualitydf(cfg::BoundsEstimatorConfig)
    pdf = getboundspredictionsdf(cfg)
    if isnothing(pdf) || size(pdf, 1) == 0
        return DataFrame()
    end
    evaldf = copy(pdf)
    evaldf[!, :err_center] = abs.(evaldf[!, :centerpred] .- evaldf[!, :centertarget])
    evaldf[!, :err_width] = abs.(evaldf[!, :widthpred] .- evaldf[!, :widthtarget])
    evaldf[!, :contains_close] = (evaldf[!, :pred_lower] .<= evaldf[!, :close]) .&& (evaldf[!, :close] .<= evaldf[!, :pred_upper])
    grp = groupby(evaldf, :set)
    return combine(grp,
        :err_center => mean => :mae_center,
        :err_width => mean => :mae_width,
        :contains_close => (x -> mean(Float32.(x)) * 100f0) => :close_in_predicted_band_pct,
        nrow => :rows)
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
    resultsdf = EnvConfig.readdf(resultsfilename())
    if !isnothing(resultsdf) && (size(resultsdf, 1) > 0)
        println("size(resultsdf) = $(size(resultsdf))")
        println("describe(resultsdf, :all)=$(describe(resultsdf, :all))")
        println("$(unique(resultsdf[!, :coin])) processable coins")
        println("used targets: $(unique(resultsdf[!, :target]))")
        println("rangeid sorted = $(issorted(resultsdf[!, :rangeid]))")
        for coin in cfg.coins
            coin_results = @view resultsdf[resultsdf[!, :coin] .== coin, :]
            print("\rcoin=$coin, sopentime sorted = $(issorted(coin_results[!, :opentime])), rangeid sorted = $(issorted(coin_results[!, :rangeid]))")
        end
        # println("target distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(resultsdf[!, :targets]))))")
        # println("set distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(resultsdf[!, :set]))))")
        gainsdf = getgainsdf(cfg)
        if !isnothing(gainsdf) && (size(gainsdf, 1) > 0)
            println("describe(gainsdf, :all)=$(describe(gainsdf, :all))")
            for coin in cfg.coins
                coinview = @view gainsdf[gainsdf[!, :coin] .== coin, :]
                # println("coin=$coin, describe(coinview, :all)=$(describe(coinview, :all))")
                println("coinview[1:10, :]=$(coinview[1:10, :])")
            end
        end
    else
        println("No results file found in $(EnvConfig.logfolder()) - size(resultsdf)=$(isnothing(resultsdf) ? "nothing" : size(resultsdf))")
    end
end

# startdt = nothing  # means use all what is stored as canned data
# enddt = nothing  # means use all what is stored as canned data
startdt = DateTime("2017-11-17T20:56:00")
enddt = DateTime("2025-08-10T15:00:00")

if abspath(PROGRAM_FILE) == @__FILE__
    println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")
    retrain = false
    retrain = "retrain" in ARGS
    retrain && println("retrain mode activated - existing regressors that did not converge will be overwritten")
    # retrain = true
    testmode = true
    testmode = "test" in ARGS ? true : "train" in ARGS ? false : testmode
    inspectonly = "inspect" in ARGS
    # inspectonly = true
    specialonly = "special" in ARGS
    # specialonly = true


    verbosity = 2
    allowedcoins = []
    if specialonly
        Ohlcv.verbosity = 3
        CryptoXch.verbosity = 3
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
        allowedcoins = ["SINE", "DOUBLESINE"]
        EnvConfig.init(test)
    else 
        if testmode 
            verbosity = 2
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
            verbosity = 2
            Ohlcv.verbosity = 1
            CryptoXch.verbosity = 1
            Features.verbosity = 1
            Targets.verbosity = 1
            EnvConfig.verbosity = 1
            Classify.verbosity = 1
            EnvConfig.init(training)
            allowedcoins = traincoins()
        end
    end
    cfg = BoundsEstimatorConfig(;boundsmk025config()..., coins=allowedcoins, startdt=startdt, enddt=enddt)

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
        if size(qdf, 1) > 0
            println("$(EnvConfig.now()) bounds quality by set: $qdf")
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
end
end # of BoundsEstimator
