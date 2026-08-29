module TrendDetector
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics, MLUtils, StatisticalMeasures
using CategoricalArrays, CategoricalDistributions, Distributions
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy, Trade, Xch, Bybit, TSM

#TODO regression from last trend pivot as feature 
"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3

# Compatibility wrappers used by tests and scripts that still access these
# helper symbols via the TrendDetector module.
resultsfilename(coin=nothing) = TradingStrategy.resultsfilename(coin)
featuresfilename(coin=nothing) = TradingStrategy.featuresfilename(coin)
trendf6config01() = TradingStrategy.trendf6config01()
targetconfig01() = TradingStrategy.targetconfig01()
tradingstrategy02() = TradingStrategy.tradingstrategy02()


"""
inspect = provide a look into files and data structures 
execute = run training and evaluation
special = run special tasks for repair, debugging or refactoring
gain = run inference-only gains/trades pipeline
"""
@enum TrendDetectorMode inspect execute special gain

mutable struct TrendDetectorConfig
    configname::String
    folder::String
    featconfig::Features.AbstractFeatures
    targetconfig::Targets.AbstractTargets
    classifiermodel
    classifiertype::Type{<:Classify.AbstractClassifier}
    tradingstrategy::TradingStrategy.StrategyConfig
    startdt::DateTime
    enddt::DateTime
    opmode::TrendDetectorMode
    partitionconfig::NamedTuple
    coins::Vector{String}
    classbalancing::Bool
    function TrendDetectorConfig(;configname, folder="Trend-$configname-$(EnvConfig.configmode)", featconfig, targetconfig, classifiermodel, classifiertype::Type{<:Classify.AbstractClassifier}=Classify.TrendClassifier001, tradingstrategy, startdt, enddt, opmode=execute, partitionconfig=TradingStrategy.partitionconfig02(), coins, classbalancing=true)
        EnvConfig.setlogpath(folder)
        EnvConfig.setdfformat!(:arrow)
        (verbosity >= 2) && println("verbosity: $verbosity")
        (verbosity >= 2) && println("log folder: $(EnvConfig.logfolder())")
        (verbosity >= 2) && println("data range: $startdt - $enddt")
        (verbosity >= 2) && println("featuresconfig=$(Features.describe(featconfig))")
        (verbosity >= 2) && println("targetsconfig=$(Targets.describe(targetconfig))")
        (verbosity >= 2) && println("classbalancing=$(classbalancing)")
        return new(configname, folder, featconfig, targetconfig, classifiermodel, classifiertype, tradingstrategy, startdt, enddt, opmode, partitionconfig, coins, classbalancing)
    end
end
cfg = nothing # to be set to a TrendDetectorConfig instance in main
retrain = false

"""
returns targets
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 4) && println("$(EnvConfig.now()) target calculation from $(fot[begin]) until $(fot[end])")
    if trgcfg isa Targets.TrendRegression
        if Features.issupplementedcurrent(featcfg)
            Targets.setbase!(trgcfg, featcfg)
        else
            @error "features not supplemented current for target calculation, cannot calculate targets for $(Targets.describe(trgcfg)) with feature base from $(fot[begin]) until $(fot[end])"
            throw(AssertionError("features not supplemented current for target calculation"))
        end
    else
        Targets.setbase!(trgcfg, ohlcv)
    end
    targets = Targets.labels(trgcfg, fot[begin], fot[end])
    # Targets.labeldistribution(targets)
    @assert size(features, 1) == length(targets) "size(features, 1)=$(size(features, 1)) != length(targets)=$(length(targets))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))
    return targets
end

@inline function _normalize_tradelabel(value)
    if value isa Targets.TradeLabel
        return value
    elseif value isa Integer
        return Targets.TradeLabel(Int(value))
    else
        return Targets.tradelabel(string(value))
    end
end

"""
Return `(ix, reason)` for the first invalid score where valid scores are finite
numbers within `(0.0, 1.0]`. Returns `(nothing, "")` when all scores are valid.

Zero is excluded: it is the live-path "not yet classified" sentinel that
`gain_limit_reversal!` acts on, so it must never appear in prediction output.
"""
function _first_invalid_score(scores)
    for ix in eachindex(scores)
        value = scores[ix]
        if ismissing(value)
            return ix, "missing"
        elseif isnan(value)
            return ix, "NaN"
        elseif !isfinite(value)
            return ix, "non-finite"
        elseif (value <= 0.0) || (value > 1.0)
            return ix, "out-of-range"
        end
    end
    return nothing, ""
end

"""Ensure Trades column `set` exists. Owner: TrendDetector. Eltype: `String`."""
function tradesdf_set(df::DataFrame)::DataFrame
    if :set ∉ propertynames(df)
        df[!, :set] = fill("", nrow(df))
    end
    return df
end

"""Ensure Trades column `rangeid` exists. Owner: TrendDetector. Eltype: `Int`."""
function tradesdf_rangeid(df::DataFrame)::DataFrame
    if :rangeid ∉ propertynames(df)
        df[!, :rangeid] = zeros(Int, nrow(df))
    end
    return df
end

"""Return TrendDetector-contributed Trades schema initializer functions."""
function tradesdf_contributors()::Vector{Function}
    return Function[
        tradesdf_set,
        tradesdf_rangeid,
    ]
end

"""Resolve the canonical timestamp column name from common OHLCV variants."""
function _resolve_opentime_col(df::AbstractDataFrame)::Union{Symbol, Nothing}
    for col in (:opentime, :open_time, :timestamp, :time, :datetime)
        if col in propertynames(df)
            return col
        end
    end
    return nothing
end

function _load_featuretarget_pair(coin::Union{AbstractString, Nothing})
    resultsdf = EnvConfig.readdf(TradingStrategy.resultsfilename(coin))
    featuresdf = EnvConfig.readdf(TradingStrategy.featuresfilename(coin))
    @assert isnothing(resultsdf) == isnothing(featuresdf) "unexpected mismatch of resultsdf and featuresdf existence for coin=$(string(coin)) with resultsdf existence $(isnothing(resultsdf)) and featuresdf existence $(isnothing(featuresdf))"

    if !isnothing(resultsdf)
        resultsdf = DataFrame(resultsdf)
        featuresdf = DataFrame(featuresdf)
        if :sampleix in propertynames(resultsdf)
            select!(resultsdf, Not(:sampleix))
        end
        if :target in propertynames(resultsdf)
            resultsdf[!, :target] = [_normalize_tradelabel(value) for value in resultsdf[!, :target]]
        end
        @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of resultsdf and featuresdf size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1)) for coin=$(coin)"
    end

    return resultsdf, featuresdf
end

function _featuretarget_cachefiles(cfg::TrendDetectorConfig; include_results::Bool=true, include_features::Bool=true, coins::AbstractVector{<:AbstractString}=cfg.coins)
    files = String[]
    for coin in coins
        if include_results && EnvConfig.isfolder(TradingStrategy.resultsfilename(coin))
            push!(files, TradingStrategy.resultsfilename(coin))
        end
        if include_features && EnvConfig.isfolder(TradingStrategy.featuresfilename(coin))
            push!(files, TradingStrategy.featuresfilename(coin))
        end
    end
    return files
end

function _persist_coin_featuretarget_cache(coin::AbstractString, coinresultsdf, coinfeaturesdf, targetissuesdf::AbstractDataFrame=DataFrame(); folderpath=EnvConfig.logfolder())::Bool
    @assert isnothing(coinresultsdf) == isnothing(coinfeaturesdf) "unexpected mismatch of coinresultsdf and coinfeaturesdf existence for coin=$(coin) with coinresultsdf existence $(isnothing(coinresultsdf)) and coinfeaturesdf existence $(isnothing(coinfeaturesdf))"
    if isnothing(coinresultsdf) || (size(coinresultsdf, 1) == 0)
        (verbosity >= 3) && println("skipping $coin due to empty results")
        return false
    end
    @assert size(coinresultsdf, 1) == size(coinfeaturesdf, 1) "unexpected mismatch of coinresultsdf and coinfeaturesdf size with coinresultsdf size $(size(coinresultsdf, 1)) and coinfeaturesdf size $(size(coinfeaturesdf, 1))"
    EnvConfig.savedf(coinresultsdf, TradingStrategy.resultsfilename(coin); folderpath=folderpath)
    EnvConfig.savedf(coinfeaturesdf, TradingStrategy.featuresfilename(coin); folderpath=folderpath)
    if size(targetissuesdf, 1) > 0
        EnvConfig.savedf(targetissuesdf, TradingStrategy.targetissuesfilename(); folderpath=folderpath)
    end
    return true
end

function _concat_coin_featuretarget_caches(cfg::TrendDetectorConfig, coins::AbstractVector{<:AbstractString}=cfg.coins)
    resultparts = DataFrame[]
    featureparts = DataFrame[]
    cachedcoins = String[]
    resultsdf = featuresdf = nothing

    hasresults = EnvConfig.isfolder(TradingStrategy.resultsfilename(nothing))
    hasfeatures = EnvConfig.isfolder(TradingStrategy.featuresfilename(nothing))
    @assert hasresults == hasfeatures "unexpected mismatch of coin-specific results/features cache existence for coin=all with hasresults=$(hasresults) and hasfeatures=$(hasfeatures)"
    if hasresults
        resultsdf, featuresdf = _load_featuretarget_pair(nothing)
        cachedcoins = string.(unique(resultsdf[!, :coin]))
        @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of concatenated results/features size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
        if Set(cachedcoins) == Set(coins)
            return resultsdf, featuresdf, cachedcoins
        end
        # A previous run can be interrupted after only some coins were processed, leaving a
        # stale results/all + features/all cache that covers a strict subset of cfg.coins.
        # Trusting it here would silently pin every future "from scratch" run to that subset.
        @warn "ignoring stale results/all + features/all cache: covers $(length(cachedcoins)) coins but $(length(coins)) are requested; recomputing all coins" cachedcoins=cachedcoins missingcoins=setdiff(coins, cachedcoins)
        resultsdf = featuresdf = nothing
        cachedcoins = String[]
    end
    for coin in coins
        hasresults = EnvConfig.isfolder(TradingStrategy.resultsfilename(coin))
        hasfeatures = EnvConfig.isfolder(TradingStrategy.featuresfilename(coin))
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

    if isempty(resultparts) || (Set(cachedcoins) != Set(coins))
        # Same reasoning as above: only reuse per-coin caches when every requested coin has one,
        # otherwise fall through so getfeaturestargetsdf! recomputes the full requested coin set.
        return nothing, nothing, String[]
    end

    resultsdf = length(resultparts) == 1 ? resultparts[1] : vcat(resultparts...; cols=:union)
    featuresdf = length(featureparts) == 1 ? featureparts[1] : vcat(featureparts...; cols=:union)
    @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of concatenated results/features size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
    EnvConfig.savedf(resultsdf, TradingStrategy.resultsfilename(nothing))
    EnvConfig.savedf(featuresdf, TradingStrategy.featuresfilename(nothing))
    return resultsdf, featuresdf, cachedcoins
end

function getfeaturestargetsdf!(cfg::TrendDetectorConfig)
    resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg)
    if !isnothing(resultsdf) && !isnothing(featuresdf) 
        (verbosity >= 2) && println("$(EnvConfig.now()) using $(length(cachedcoins)) coin-specific cached trend feature/target pairs")
    else
        (verbosity >= 2) && println("$(EnvConfig.now()) calculating features and targets                             ")

        cl = _trendclassifierseed(cfg)
        rangeid = Classify.RANGEID_SUBRANGE_SPAN # liquidity range base id; unique across coins
        skippedcoins = String[]
        processedcoins = String[]
        targetissuesdf = DataFrame()

        for coinix in eachindex(cfg.coins)
            coin = cfg.coins[coinix]
            coinresultsdf = coinfeaturesdf = nothing
            resultsdf = featuresdf = nothing
            (verbosity >= 2) && print("calculating $coin ($coinix/$(length(cfg.coins))) features and targets                                                          \r")
            (verbosity >= 3) && println()
            ohlcv = Ohlcv.read(coin)
            odf = Ohlcv.dataframe(ohlcv)
            otcol = _resolve_opentime_col(odf)
            if size(odf, 1) == 0
                (verbosity >= 1) && @warn "skipping coin due to empty OHLCV data" coin
                push!(skippedcoins, coin)
                continue
            end
            @assert !isnothing(otcol) "non-empty OHLCV data must contain :opentime-compatible timestamp column for coin=$(coin); available columns=$(propertynames(odf))"
            ot = odf[!, otcol]
            cfg.startdt = isnothing(cfg.startdt) ? ot[begin] : cfg.startdt
            startix = Ohlcv.rowix(ot, cfg.startdt)
            cfg.enddt = isnothing(cfg.enddt) ? ot[end] : cfg.enddt
            endix = Ohlcv.rowix(ot, cfg.enddt)
            @assert startix < endix "unexpected startix $startix >= endix $endix for $coin with startdt $(cfg.startdt) and enddt $(cfg.enddt)              "
            reqmins = max(0, Int(Classify.requiredminutes(cl)))

            if cfg.opmode == gain
                history_begin_ix = max(firstindex(ot), startix - reqmins)
                history_startdt = ot[history_begin_ix]
                real_startdt = ot[startix]
                real_enddt = ot[endix]
                (verbosity >= 2) && print("$(EnvConfig.now()) calculating features and targets for $coin ($coinix/$(length(cfg.coins))) gain window $(real_startdt) → $(real_enddt), preload $(history_startdt) (feature_req=$(reqmins))                \r")
                (verbosity >= 3) && println()
                rngohlcv = Ohlcv.ohlcvview(ohlcv, history_begin_ix:endix)
                trgcfg = cfg.targetconfig
                rngresults, rngfeatures = Classify.featurestargetsdf(
                    cl,
                    rngohlcv,
                    trgcfg;
                    startdt=real_startdt,
                    enddt=real_enddt,
                    partitionconfig=nothing,
                    coin=coin,
                    rangeid_start=rangeid,
                )
                issues = Targets.crosscheck(trgcfg, rngresults[!, :target], rngresults[!, :pivot])
                issues_count = isnothing(issues) ? 0 : (issues isa AbstractDataFrame ? size(issues, 1) : length(issues))
                if issues_count > 0
                    issue_values = issues isa AbstractDataFrame ? issues[!, :issue] : issues
                    if size(targetissuesdf, 1) > 0
                        targetissuesdf = vcat(targetissuesdf, DataFrame(issue=issue_values, coin=CategoricalVector(fill(coin, issues_count), levels=cfg.coins), rangeid=fill(rangeid, issues_count)))
                    else
                        targetissuesdf = DataFrame(issue=issue_values, coin=CategoricalVector(fill(coin, issues_count), levels=cfg.coins), rangeid=fill(rangeid, issues_count))
                    end
                end
                if size(rngresults, 1) > 0
                    rangeid += Classify.RANGEID_SUBRANGE_SPAN
                end
                coinresultsdf = isnothing(coinresultsdf) ? rngresults : vcat(coinresultsdf, rngresults)
                coinfeaturesdf = isnothing(coinfeaturesdf) ? rngfeatures : vcat(coinfeaturesdf, rngfeatures)
            else
                window_minutes = endix - startix + 1
                liq_checkperiod = min(Ohlcv.ld.checkperiod, max(15, fld(window_minutes, 4)))
                liq_accumulate = min(Ohlcv.ld.accumulate, liq_checkperiod)
                liq_startdistance = min(Ohlcv.ld.startdistance, max(0, window_minutes - liq_checkperiod))
                liq_minliquidminutes = min(Ohlcv.ld.minliquidminutes, max(1, window_minutes - liq_checkperiod + 1))
                rv = Ohlcv.liquiditycheck(
                    Ohlcv.ohlcvview(ohlcv, startix:endix);
                    minquotevol=Ohlcv.ld.minquotevol,
                    accumulate=liq_accumulate,
                    checkperiod=liq_checkperiod,
                    startthreshold=Ohlcv.ld.startthreshold,
                    stopthreshold=Ohlcv.ld.stopthreshold,
                    minliquidminutes=liq_minliquidminutes,
                    startdistance=liq_startdistance,
                )
                liq_historymins = max(liq_checkperiod, liq_accumulate, liq_startdistance)
                preload_historymins = max(reqmins, liq_historymins)

                for rngix in eachindex(rv) # rng indices are related to the ohlcvview dataframe rows
                    rng = rv[rngix]
                    rng = rng .+ (startix - 1) # adjust to complete ohlcv dataframe row indices
                    if rng[end] - rng[begin] > 0
                        real_startdt = ot[rng[begin]]
                        real_enddt = ot[rng[end]]
                        history_begin_ix = max(firstindex(ot), rng[begin] - preload_historymins)
                        history_startdt = ot[history_begin_ix]
                        (verbosity >= 2) && print("$(EnvConfig.now()) calculating features and targets for $coin ($coinix/$(length(cfg.coins))) range ($rngix/$(length(rv))) $rng real $(real_startdt) → $(real_enddt), preload $(history_startdt) (feature_req=$(reqmins), liquidity_req=$(liq_historymins), used=$(preload_historymins))                \r")
                        (verbosity >= 3) && println()
                        rngohlcv = Ohlcv.ohlcvview(ohlcv, history_begin_ix:rng[end])
                        trgcfg = cfg.targetconfig
                        #TODO verbosity == 3 shows that the features.arrow file of the base under consideration is read for each range while it should not be read at all during feature generation
                        rngresults, rngfeatures = Classify.featurestargetsdf(
                            cl,
                            rngohlcv,
                            trgcfg;
                            startdt=real_startdt,
                            enddt=real_enddt,
                            partitionconfig=cfg.partitionconfig,
                            coin=coin,
                            rangeid_start=rangeid,
                        )
                        issues = DataFrame() # disabled Targets.crosscheck(trgcfg, rngresults[!, :target], rngresults[!, :pivot])
                        issues_count = isnothing(issues) ? 0 : (issues isa AbstractDataFrame ? size(issues, 1) : length(issues))
                        if issues_count > 0
                            issue_values = issues isa AbstractDataFrame ? issues[!, :issue] : issues
                            if size(targetissuesdf, 1) > 0
                                targetissuesdf = vcat(targetissuesdf, DataFrame(issue=issue_values, coin=CategoricalVector(fill(coin, issues_count), levels=cfg.coins), rangeid=fill(rangeid, issues_count)))
                            else
                                targetissuesdf = DataFrame(issue=issue_values, coin=CategoricalVector(fill(coin, issues_count), levels=cfg.coins), rangeid=fill(rangeid, issues_count))
                            end
                        end
                        if size(rngresults, 1) > 0
                            rangeid += Classify.RANGEID_SUBRANGE_SPAN
                        end
                        coinresultsdf = isnothing(coinresultsdf) ? rngresults : vcat(coinresultsdf, rngresults)
                        coinfeaturesdf = isnothing(coinfeaturesdf) ? rngfeatures : vcat(coinfeaturesdf, rngfeatures)
                    else
                        @error "unexpected zero length range for " ohlcv.base rng rv
                    end
                end
            end
            ohlcv = ot = rngohlcv = rngresults = rngfeatures = nothing # free memory
            if _persist_coin_featuretarget_cache(coin, coinresultsdf, coinfeaturesdf, targetissuesdf)
                coinfeaturesdf = coinresultsdf = nothing # free memory
                push!(processedcoins, coin)
            else
                push!(skippedcoins, coin)
            end
        end

        (verbosity >= 2) && println()
        resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg, processedcoins)
        (verbosity >= 2) && println("$(EnvConfig.now()) processed $(length(processedcoins)), skipped $(length(skippedcoins)) coins")
        (verbosity >= 3) && println("$(EnvConfig.now()) processed $processedcoins")
        if (verbosity >= 3) && (length(skippedcoins) > 0)
            if cfg.opmode == gain
                println("skipped to process $skippedcoins due to empty data/results")
            else
                println("skipped to process $skippedcoins due to no liquid ranges")
            end
        end
    end

    @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(featuresdf, 1) > 0) "unexpected resultsdf and featuresdf size with resultsdf size $(isnothing(resultsdf) ? "nothing" : size(resultsdf, 1)) and featuresdf size $(isnothing(featuresdf) ? "nothing" : size(featuresdf, 1))"
    return resultsdf, featuresdf
end

function df2features(featuresdf, cfg::TrendDetectorConfig, settype=nothing)
    if size(featuresdf, 1) > 0
        featuresdf = isnothing(settype) ? featuresdf : @view featuresdf[(featuresdf[!, :set] .== settype), :]
        features = @view featuresdf[!, Features.requestedcolumns(cfg.featconfig)]
        features = Array(features)  # change from df to array
        features = permutedims(features, (2, 1))  # Flux expects observations as columns with features of an oberservation as one column
        (verbosity >= 3) && println("typeof(features)=$(typeof(features)), size(features)=$(size(features)) for settype=$(settype)") 
        return features
    else
        return nothing
    end
end

classifiermenmonic(coins=nothing, coinix=nothing) = "mix"

function _classifierfolder(cfg::TrendDetectorConfig)::String
    override = strip(get(ENV, "TRENDDETECTOR_CLASSIFIER_FOLDER", ""))
    if !isempty(override)
        return isabspath(override) ? override : normpath(joinpath(EnvConfig.logfolder(), override))
    end
    if cfg.opmode == gain
        # Gain mode reuses the configured phase artifact folder, but it must still be
        # resolved as an absolute log-path so loading works from any working directory.
        phasefolder = "Trend-$(cfg.configname)-$(String(Symbol(EnvConfig.configmode)))"
        return normpath(joinpath(dirname(EnvConfig.logfolder()), phasefolder))
    end
    # Training and execution artifacts live in the current run log folder; this is
    # where the classifier is created and therefore where it must be found again.
    return EnvConfig.logfolder()
end

function _trendclassifierspec(cfg::TrendDetectorConfig)
    return (
        config_ref=cfg.configname,
        featconfig=() -> cfg.featconfig,
        targetconfig=() -> cfg.targetconfig,
        folder=_classifierfolder(cfg),
    )
end

function _trendclassifierseed(cfg::TrendDetectorConfig)::Classify.AbstractClassifier
    featurecount = Features.featurecount(cfg.featconfig)
    labels = Targets.uniquelabels(cfg.targetconfig)
    mnemonic = classifiermenmonic()
    classifierfolder = _classifierfolder(cfg)
    spec = _trendclassifierspec(cfg)

    if cfg.opmode == gain
        nntmp = cfg.classifiermodel(featurecount, labels, mnemonic)
        loadspec = merge(spec, (nn_fileprefix=nntmp.fileprefix,))
        return Classify.load(
            cfg.classifiertype,
            loadspec;
            mode=EnvConfig.configmode,
            folder=classifierfolder,
        )
    end

    return Classify.loadorbuild(
        cfg.classifiertype,
        spec,
        featurecount,
        labels,
        mnemonic,
        cfg.classifiermodel;
        mode=EnvConfig.configmode,
        folder=classifierfolder,
    )
end

function getruntimeclassifier(cfg::TrendDetectorConfig)::Classify.AbstractClassifier
    cl = _trendclassifierseed(cfg)

    if cfg.opmode == gain
        @assert !retrain "retrain mode is not allowed in gain mode"
        @assert Classify.isadapted(cl) "gain mode requires an existing adapted classifier in $(_classifierfolder(cfg)); train it first"
        return cl
    end

    model = Classify.model(cl)

    if !Classify.isadapted(cl) || retrain
        println("$(EnvConfig.now()) adapting one mix classifier for all coins")
        resultsdf, featuresdf = getfeaturestargetsdf!(cfg)
        if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
            return cl
        end
        x, y, sampleweights = Classify.prepareadaptation(
            cl,
            resultsdf,
            featuresdf;
            settype="train",
            classbalancing=cfg.classbalancing
        )
        resultsdf = featuresdf = nothing # free memory
        if retrain
            Classify.adaptnn!(cl.nn, x, y; sampleweights=sampleweights, reinforce_epochs=10)
        else
            Classify.adaptnn!(cl.nn, x, y; sampleweights=sampleweights, reinforce_epochs=0)
        end


        (verbosity >= 3) && showlosses(model)
        println("$(EnvConfig.now()) finished adapting mix classifier - classifier $(Classify.nnconverged(cl) ? "did" : "did not") converge")
        modelprefix = "$(cfg.configname)-$(String(Symbol(EnvConfig.configmode)))"
        Classify.savenn(cl.nn; folderpath=EnvConfig.logfolder(), fileprefix=modelprefix, save_lastepoch=false, save_result=true)
        # Keep the reviewed/classification-export artifacts in neuralnets, but do not
        # force the runtime training loop to read/write there before evaluation.
    end

    return cl
end

function getlatestclassifier(cfg::TrendDetectorConfig)
    cl = getruntimeclassifier(cfg)
    return Classify.model(cl)
end

# getlatestclassifier(coins, coinix, featureconfig, targetconfig) = getlatestclassifier((isnothing(coinix) ? coins : coins[coinix]), featureconfig, targetconfig)

function showlosses(nn)
    println("$(EnvConfig.now()) evaluating classifier $(nn.mnemonic)")
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


function getclassifier(cfg::TrendDetectorConfig)
    return Classify.model(getruntimeclassifier(cfg))
end

function checkpredictionsdf(predictionsdf::Union{AbstractDataFrame, Nothing}, refdf::Union{AbstractDataFrame, Nothing}=nothing)
    @assert !isnothing(predictionsdf) "missing predictionsdf"
    @assert size(predictionsdf, 1) > 0 "unexpected empty predictionsdf"
    @assert isnothing(refdf) || (size(predictionsdf, 1) == size(refdf, 1)) "size mismatch: size(predictionsdf, 1)=$(size(predictionsdf, 1)) != size(refdf, 1)=$(size(refdf, 1))"
    @assert :score in propertynames(predictionsdf) "missing :score column in predictionsdf with columns=$(propertynames(predictionsdf))"
    badix, badreason = _first_invalid_score(predictionsdf[!, :score])
    @assert isnothing(badix) "invalid score in predictionsdf at row $(badix) due to $(badreason): $(predictionsdf[badix, :]) of $(describe(predictionsdf, :all))"
end

"""
Split and persist `predictionsdf` per coin (mirrors the `results/features` per-coin caches),
so downstream single/few-coin readers (e.g. `tradesim.jl`) can load a coin's predictions
without loading the full multi-coin `predictions/maxpredictions` file. `resultsdf` and
`predictionsdf` must be row-aligned (same order, same length) - `predictionsdf` itself carries
no coin/opentime identity of its own. Skips coins that already have a cache unless `force`.
"""
function _persist_coin_predictions_cache!(resultsdf::AbstractDataFrame, predictionsdf::AbstractDataFrame; force::Bool=false)
    @assert :coin in propertynames(resultsdf) "resultsdf missing :coin column required for per-coin predictions split"
    n = size(resultsdf, 1)
    @assert n == size(predictionsdf, 1) "resultsdf/predictionsdf row count mismatch: $(n) != $(size(predictionsdf, 1))"
    coins = string.(resultsdf[!, :coin])
    coinixs = Dict{String, Vector{Int}}()
    for ix in 1:n
        push!(get!(() -> Int[], coinixs, coins[ix]), ix)
    end
    for (coin, ixs) in coinixs
        if !force && EnvConfig.isfolder(TradingStrategy.predictionsfilename(coin))
            continue
        end
        EnvConfig.savedf(predictionsdf[ixs, :], TradingStrategy.predictionsfilename(coin))
    end
    return nothing
end

"""
Rebuild per-coin prediction caches from an already-generated combined `results/all` +
`predictions/maxpredictions` pair, e.g. for a replay-source folder created before the
per-coin split existed. Overwrites any existing per-coin cache.
"""
function backfillcoinpredictions!()
    resultsdf = DataFrame(EnvConfig.readdf(TradingStrategy.resultsfilename()))
    predictionsdf = DataFrame(EnvConfig.readdf(TradingStrategy.predictionsfilename()))
    @assert nrow(resultsdf) > 0 "missing or empty $(TradingStrategy.resultsfilename())"
    @assert nrow(predictionsdf) > 0 "missing or empty $(TradingStrategy.predictionsfilename())"
    _persist_coin_predictions_cache!(resultsdf, predictionsdf; force=true)
    return nothing
end

"""
Returns the max prediction with its corresponding trade label for the samples of all coins. 
The returned DataFrame provides one score::Float32 column and one label::TradeLabel column representing the best sample prediction + the original targets::TradeLabel and set::CategoricalVector.
"""
function getmaxpredictionsdf(cfg::TrendDetectorConfig)
    predictionsdf = nothing
    if !retrain
        predictionsdf = EnvConfig.readdf(TradingStrategy.predictionsfilename())
        depfiles = _featuretarget_cachefiles(cfg)
        if !isnothing(predictionsdf) 
            if isfreshcache(TradingStrategy.predictionsfilename(), depfiles)
                checkpredictionsdf(predictionsdf)
            else
                @warn "ignoring stale max predictions cache; rebuilding from newer coin-specific trend feature/target caches"
                EnvConfig.deletefolder(TradingStrategy.predictionsfilename())
                predictionsdf = nothing
            end
        end
    end
    resultsdf = featuresdf = nothing
    freshlycomputed = false
    # predictions are stored in a predictionsdf to avoid loading every time also features bu eventually you want the whole resultdf with predictions
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        cl = getruntimeclassifier(cfg)
        if isnothing(resultsdf) || isnothing(featuresdf)
            resultsdf, featuresdf = getfeaturestargetsdf!(cfg) 
        end
        (verbosity >= 2) && print("$(EnvConfig.now()) classify maximum predictions                             \r")
        (verbosity >= 3) && println()
        predictionsdf = Classify.maxpredictdf(cl, featuresdf)
        checkpredictionsdf(predictionsdf, featuresdf)
        if (size(resultsdf, 1) > 0)
            EnvConfig.savedf(predictionsdf, TradingStrategy.predictionsfilename())
        end
        freshlycomputed = true
    end
    if !isnothing(predictionsdf) && (size(predictionsdf, 1) > 0)
        if isnothing(resultsdf)
            resultsdf, _ = getfeaturestargetsdf!(cfg)
        end
        checkpredictionsdf(predictionsdf, resultsdf)
        _persist_coin_predictions_cache!(resultsdf, predictionsdf; force=freshlycomputed)
        resultsdf[:, :score] = predictionsdf[!, :score]
        resultsdf[:, :label] = predictionsdf[!, :label]
        badix, badreason = _first_invalid_score(resultsdf[!, :score])
        @assert isnothing(badix) "invalid score after assigning predictionsdf to resultsdf at row $(badix) due to $(badreason): $(resultsdf[badix, :])"
    else
        resultsdf = nothing
    end
    return resultsdf
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

const TRUE_GAIN_THRESHOLD = (0.9f0, 0.9f0)

"""Columns kept in the per-range Trades snapshot accumulated by `getgainsdf`.

These are exactly what `TSM.compilegainsdf` and `TradeAdviceCompare` consume. The
remaining ~45 Trades columns are order id/status/msg and account placeholders that stay
constant during gain replay; carrying them would roughly triple the snapshot memory that
accumulates over all range groups."""
const TRADES_SNAPSHOT_COLUMNS = Symbol[
    :opentime, :pair, :set, :rangeid, :lastopentrade,
    :close, :high, :low, :label, :score,
    :lp_amount, :sp_amount,
    :lol_pavg, :lcl_pavg, :sol_pavg, :scl_pavg,
    :lo_limit, :lc_limit, :so_limit, :sc_limit,
]

"""Result columns replay needs for gain compilation.

The per-coin results cache also carries `pivot` and the full feature-side payload, which
replay drops again; projecting here keeps the loaded frame small."""
const GAIN_RESULT_COLUMNS = Symbol[:opentime, :high, :low, :close, :coin, :rangeid, :set, :target]

"""Return the coins that have a per-coin results and predictions cache.

Predictions are deliberately reused regardless of age so `algorithm` variants can be
compared without recomputing them; `_load_coin_gaininput` asserts row alignment, which is
the property gain compilation actually depends on."""
function _gaininputcoins(cfg::TrendDetectorConfig)::Vector{String}
    coins = String[]
    for coin in cfg.coins
        coinstr = String(coin)
        if EnvConfig.isfolder(TradingStrategy.resultsfilename(coinstr)) && EnvConfig.isfolder(TradingStrategy.predictionsfilename(coinstr))
            push!(coins, coinstr)
        end
    end
    return coins
end

"""Return the replay window spanning every gain-input coin.

Only the `opentime` column is touched, so the Arrow payload stays unread."""
function _gainreplaywindow(coins::AbstractVector{<:AbstractString})
    startdt = nothing
    enddt = nothing
    for coin in coins
        table = EnvConfig.readtable(TradingStrategy.resultsfilename(coin); materialize=false)
        @assert !isnothing(table) "missing results cache for coin=$(coin)"
        opentimes = DataFrame(table; copycols=false)[!, :opentime]
        isempty(opentimes) && continue
        lo, hi = extrema(opentimes)
        startdt = isnothing(startdt) ? lo : min(startdt, lo)
        enddt = isnothing(enddt) ? hi : max(enddt, hi)
    end
    return startdt, enddt
end

"""Load one coin's replay input by joining its per-coin results and predictions caches.

Both caches are row aligned by construction (`_persist_coin_predictions_cache!`), so the
score/label columns are taken over positionally."""
function _load_coin_gaininput(coin::AbstractString)::DataFrame
    resultstable = EnvConfig.readtable(TradingStrategy.resultsfilename(coin); materialize=false)
    predictionstable = EnvConfig.readtable(TradingStrategy.predictionsfilename(coin); materialize=false)
    @assert !isnothing(resultstable) "missing results cache for coin=$(coin)"
    @assert !isnothing(predictionstable) "missing predictions cache for coin=$(coin)"

    # Arrow columns are lazy until materialized, so wrapping first and projecting second
    # only reads the columns replay actually needs.
    lazyresults = DataFrame(resultstable; copycols=false)
    available = propertynames(lazyresults)
    projected = Symbol[col for col in GAIN_RESULT_COLUMNS if col in available]
    @assert :opentime in projected "results cache for coin=$(coin) misses :opentime; available=$(available)"
    resultsdf = DataFrame(lazyresults[!, projected])

    predictionsdf = DataFrame(predictionstable; copycols=false)
    @assert nrow(resultsdf) == nrow(predictionsdf) "results/predictions row mismatch for coin=$(coin): $(nrow(resultsdf)) != $(nrow(predictionsdf))"
    resultsdf[!, :score] = collect(predictionsdf[!, :score])
    resultsdf[!, :label] = collect(predictionsdf[!, :label])

    if :target in propertynames(resultsdf)
        resultsdf[!, :target] = [_normalize_tradelabel(value) for value in resultsdf[!, :target]]
    end
    badix, badreason = _first_invalid_score(resultsdf[!, :score])
    @assert isnothing(badix) "invalid score in gain input for coin=$(coin) at row $(badix) due to $(badreason)"
    return resultsdf
end

"""Stem of the compiled gain segments produced by gain replay."""
const COMPILED_GAINS_STEM = "tsmgains-td"

"""Concatenate per-pair compiled gains, persist them and derive the gain report."""
function _collectxchgains(gainparts::Vector{DataFrame}; gainsstem::AbstractString=COMPILED_GAINS_STEM, reportstem::AbstractString="xchgainsreport-td")
    folderpath = EnvConfig.logfolder()
    xchgainsdf = isempty(gainparts) ? DataFrame() : TSM.sortgainsdf!(reduce(vcat, gainparts))
    empty!(gainparts)
    EnvConfig.savedf(xchgainsdf, String(gainsstem); folderpath=folderpath)
    xchreportdf = TSM.gainsreport(instem=gainsstem, stem=reportstem, folderpath=folderpath)
    return xchgainsdf, xchreportdf
end

"""Log the compiled gains report of the current run."""
function _report_compiled_gains(xchreportdf::Union{AbstractDataFrame, Nothing})
    if !isnothing(xchreportdf) && (size(xchreportdf, 1) > 0)
        println("$(EnvConfig.now()) compiled gains report: $xchreportdf")
    end
    return nothing
end

"""Subfolder of the run log folder holding one Trades artifact per trading pair."""
const TRADES_TD_SUBFOLDER = "trades-td"

"""Return the per-pair Trades artifact folder of the current run."""
tradestdfolder(folderpath::AbstractString=EnvConfig.logfolder())::String = joinpath(String(folderpath), TRADES_TD_SUBFOLDER)

"""Persist one pair's accumulated Trades snapshots of one replay pass and compile its gains.

Consumes `tradeparts`. Each pass is stored and compiled separately because both replay the
same minutes with different scores/labels; mixing them in one partition would match an open
of one pass against a close of the other."""
function _flushpairtrades!(gainparts::Vector{DataFrame}, tradeparts::Vector{DataFrame}, folderpath::AbstractString, pass::AbstractString, predicted::Bool, openthreshold::Float32, closethreshold::Float32)
    isempty(tradeparts) && return nothing
    pairdf = reduce(vcat, tradeparts)
    empty!(tradeparts)
    pair = String(pairdf[1, :pair])
    opentimes = pairdf[!, :opentime]
    @assert issorted(opentimes) "expected opentime ordered Trades rows for pair=$(pair), pass=$(pass); rows=$(nrow(pairdf)), first=$(first(opentimes)), last=$(last(opentimes))"
    EnvConfig.savedf(pairdf, "$(pair)-$(pass)"; folderpath=folderpath)
    # Each range is replayed independently, so a position still open at a range end has no
    # close of its own; scoping to (set, rangeid) drops it instead of matching it against
    # the next range's first row.
    gdf = TSM.compilegains(pairdf; setpartitions=true)
    if nrow(gdf) > 0
        gdf[!, :predicted] = fill(predicted, nrow(gdf))
        gdf[!, :openthreshold] = fill(openthreshold, nrow(gdf))
        gdf[!, :closethreshold] = fill(closethreshold, nrow(gdf))
        push!(gainparts, gdf)
    end
    return nothing
end

function getgainsdf(cfg::TrendDetectorConfig)
    EnvConfig.setlogpath(cfg.folder)
    gaindeps = vcat(_featuretarget_cachefiles(cfg; include_features=false), [TradingStrategy.predictionsfilename()])
    if isfreshcache(COMPILED_GAINS_STEM, gaindeps)
        gaindf = EnvConfig.readdf(COMPILED_GAINS_STEM)
        if !isnothing(gaindf) && (size(gaindf, 1) > 0)
            return DataFrame(gaindf)
        end
    end

    # Per-coin results/predictions caches are the gain-compilation input. Build them once if
    # they are missing; loading them per coin avoids materializing every coin (and the far
    # larger features cache) before the first range is processed.
    gaincoins = _gaininputcoins(cfg)
    if length(gaincoins) != length(cfg.coins)
        resultsdf = getmaxpredictionsdf(cfg)
        if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
            return nothing
        end
        resultsdf = nothing
        # getmaxpredictionsdf resolves the classifier, which repoints the log path.
        EnvConfig.setlogpath(cfg.folder)
        gaincoins = _gaininputcoins(cfg)
    end
    if isempty(gaincoins)
        return nothing
    end
    tradesfolderpath = tradestdfolder()

    ts = TradingStrategy.TsCache(strategy=TradingStrategy.strategyconfig(cfg.configname), source="trenddetector:$(cfg.configname)")
    replay_startdt, replay_enddt = _gainreplaywindow(gaincoins)
    xc = Xch.XchCache(Bybit.BybitCache(); startdt=replay_startdt, enddt=replay_enddt)
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

    xchgainparts = DataFrame[]
    predparts = DataFrame[]
    truthparts = DataFrame[]
    totalranges = 0

    for (coinix, coin) in enumerate(gaincoins)
        coinresultsdf = _load_coin_gaininput(coin)
        if nrow(coinresultsdf) == 0
            continue
        end

        # Assembly order matters: replay ranges are processed in chronological order so the
        # accumulated pair snapshot is built from the beginning of time without reordering.
        sort!(coinresultsdf, [:opentime, :set, :rangeid])
        # Range ids can collide across independently cached coins/runs. Replay must stay
        # scoped to set+rangeid within one coin to avoid mixing samples across ranges.
        # Grouping yields set-major order, but sets partition the timeline into interleaved
        # blocks, so the groups are resequenced by their first opentime.
        rangegroups = groupby(coinresultsdf, [:set, :rangeid])
        grouporder = sortperm([first(group[!, :opentime]) for group in rangegroups])
        totalranges += length(grouporder)

        for (rngix, groupix) in enumerate(grouporder)
            resultsview = rangegroups[groupix]
            rng = resultsview[begin, :rangeid]
            (verbosity >= 2) && print("$(EnvConfig.now()) calculating gains for $coin ($coinix/$(length(gaincoins))) range ($rngix/$(length(rangegroups))) $rng                             \r")
            (verbosity >= 3) && println()
            @assert size(resultsview, 1) > 0 "unexpected empty resultsview for rangeid $rng"

            sampleset = resultsview[begin, :set]
            scores = resultsview[!, :score]
            labels = [_normalize_tradelabel(value) for value in resultsview[!, :label]]
            targets = [_normalize_tradelabel(value) for value in resultsview[!, :target]]
            truescores = fill(1f0, size(resultsview, 1))
            evaldt = resultsview[end, :opentime]
            # Replay has no account, so equity is a constant: one lane budget is always
            # fundable and repeated opens are prevented by the lane position amount.
            replaybudget = Dict{Symbol, Any}(
                :set => String(sampleset),
                :rangeid => Int(rng),
                :freequote => cfg.tradingstrategy.maxbudgetquote,
                :freemargin => cfg.tradingstrategy.maxbudgetquote,
            )

            # Process predicted gains using strategy config thresholds
            open_threshold = cfg.tradingstrategy.openthreshold
            close_threshold = cfg.tradingstrategy.closethreshold
            tp = TradingStrategy.preparereplaytrades!(
                ts,
                xc,
                coin,
                resultsview,
                scores,
                labels,
                metadata=replaybudget,
                datetime=evaldt,
            )
            gdf = TradingStrategy.processreplaygains!(
                tp;
                strategy=cfg.tradingstrategy,
            )
            # Snapshot before the truth pass reuses and overwrites this same tradesdf.
            push!(predparts, select(tp.tradesdf, TRADES_SNAPSHOT_COLUMNS))

            # Process labeled truth gains using TRUE_GAIN_THRESHOLD
            true_open, true_close = TRUE_GAIN_THRESHOLD
            tp = TradingStrategy.preparereplaytrades!(
                ts,
                xc,
                coin,
                resultsview,
                truescores,
                targets,
                metadata=replaybudget,
                datetime=evaldt,
            )
            gdf = TradingStrategy.processreplaygains!(
                tp;
                strategy=cfg.tradingstrategy,
            )
            # Keep one Trades snapshot per range/set; replay state in `xc` is
            # overwritten on each loop iteration, so we must collect snapshots.
            push!(truthparts, select(tp.tradesdf, TRADES_SNAPSHOT_COLUMNS))

            # Release the full-width replay Trades DataFrame of this range; otherwise both
            # caches keep one per coin alive until the end of the run.
            TradingStrategy.droppair!(ts, tp.pair)
            TSM.droppair!(xc.tsm, tp.pair)
        end
        rangegroups = nothing
        coinresultsdf = nothing
        # One coin is one pair, so its snapshots are complete once its ranges are processed.
        _flushpairtrades!(xchgainparts, predparts, tradesfolderpath, "predicted", true, cfg.tradingstrategy.openthreshold, cfg.tradingstrategy.closethreshold)
        _flushpairtrades!(xchgainparts, truthparts, tradesfolderpath, "truth", false, TRUE_GAIN_THRESHOLD[1], TRUE_GAIN_THRESHOLD[2])
    end

    gaindf, xchreportdf = _collectxchgains(xchgainparts)
    _report_compiled_gains(xchreportdf)

    if size(gaindf, 1) > 0
        expected = Set(Xch.tradingpairkey(String(coin), EnvConfig.pairquote) for coin in cfg.coins)
        present = Set(String.(unique(gaindf[!, :pair])))
        missing_pairs = sort(collect(setdiff(expected, present)))
        if !isempty(missing_pairs)
            @warn "missing pairs in gains output" missing_pairs present_pairs=sort(collect(present))
        end
    end

    (verbosity >= 2) && println("$(EnvConfig.now()) calculated gains for $(totalranges) ranges")
    return gaindf
end

function getconfusionmatrices(cfg::TrendDetectorConfig)
    xcmdf = DataFrame()
    cmdf = DataFrame()
    if EnvConfig.isfolder(EnvConfig.logpath(TradingStrategy.confusionfilename()))
        cmdf = EnvConfig.readdf(TradingStrategy.confusionfilename())
    end
    if EnvConfig.isfolder(EnvConfig.logpath(TradingStrategy.xconfusionfilename()))
        xcmdf = EnvConfig.readdf(TradingStrategy.xconfusionfilename())
    end
    if !isnothing(cmdf) && !isnothing(xcmdf) && (size(cmdf, 1) > 0) && (size(xcmdf, 1) > 0)
        return cmdf, xcmdf
    end
    resultsdf = getmaxpredictionsdf(cfg)
    if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
        return nothing, nothing
    end
    badix, badreason = _first_invalid_score(resultsdf[!, :score])
    @assert isnothing(badix) "invalid score before confusion matrix evaluation at row $(badix) due to $(badreason): $(resultsdf[badix, :])"
    resultsdf = @view resultsdf[.!ismissing.(resultsdf[!, :set]), :] # exclude gaps between set partitions
    (verbosity >= 2) && print("$(EnvConfig.now()) calculating confusion matrices                             \r")
    (verbosity >= 3) && println()
    if (size(resultsdf, 1) > 0)
        # predictedlabel = categorical(string.(dfp[!, :label]), levels=string.(Targets.uniquelabels(cfg.targetconfig)))
        # println("predictedllabels=$(unique(predictedlabel)), levels=$(levels(predictedlabel))")
        # targetlabel = categorical(string.(dfp[!, :target]), levels=string.(Targets.uniquelabels(cfg.targetconfig)))
        # println("targetlabels=$(unique(targetlabel)), levels=$(levels(targetlabel))")
        # cm = StatisticalMeasures.ConfusionMatrices.confmat(predictedlabel, targetlabel)
        # println("describe(predictions): $(describe(dfp))")
        # display(cm)
        cmdf = Classify.confusionmatrix(resultsdf, Targets.uniquelabels(cfg.targetconfig))
        if size(cmdf, 1) > 0
            EnvConfig.savedf(cmdf, TradingStrategy.confusionfilename())
        end
        xcmdf = Classify.extendedconfusionmatrix(resultsdf, Targets.uniquelabels(cfg.targetconfig))
        if size(xcmdf, 1) > 0
            EnvConfig.savedf(xcmdf, TradingStrategy.xconfusionfilename())
        end
    else
        (verbosity >= 1) && println("skipping evaluation of $(cfg.coins) due to missing predictions (size(dfp)= $(size(resultsdf)))")
    end
    (verbosity >= 2) && print("$(EnvConfig.now()) calculated confusion matrices                             \r")
    (verbosity >= 3) && println()
    return cmdf, xcmdf
end

function averageconfusionmatrix(cfg::TrendDetectorConfig)
    # calc positive prediction value (ppv) 
    cmdf, xcmdf = getconfusionmatrices(cfg)
    println("describe(confusion matrix: $(describe(cmdf)))")
    println("describe(extended confusion matrix: $(describe(xcmdf)))")
    if size(cmdf, 1) > 0
        # cmdfgrp = groupby(cmdf, [:coin, :set, :prediction])
        cmdf = @view cmdf[.!ismissing.(cmdf[!, :set]), :] # exclude gaps between set partitions
        cmdfgrp = groupby(cmdf, [:set, :prediction])
        ccmdf = combine(cmdfgrp, 
                        [:truth_longopen, :truth_longhold, :truth_allclose, :truth_shorthold, :truth_shortopen] => ((lb, lh, ac, sh, sb) -> sum(lb) / (sum(lb) + sum(lh) + sum(sum(ac)) + sum(sh) + sum(sb)) * 100) => "longopen_ppv%",
                        [:truth_longhold, :truth_longopen, :truth_allclose, :truth_shorthold, :truth_shortopen] => ((lh, lb, ac, sh, sb) -> sum(lh) / (sum(lh) + sum(lb) + sum(sum(ac)) + sum(sh) + sum(sb)) * 100) => "longhold_ppv%",
                        [:truth_allclose, :truth_longopen, :truth_longhold, :truth_shorthold, :truth_shortopen] => ((ac, lb, lh, sh, sb) -> sum(ac) / (sum(ac) + sum(lb) + sum(lh) + sum(sh) + sum(sb)) * 100) => "allclose_ppv%",
                        [:truth_shorthold, :truth_longopen, :truth_longhold, :truth_allclose, :truth_shortopen] => ((sh, lb, lh, ac, sb) -> sum(sh) / (sum(sh) + sum(lb) + sum(lh) + sum(ac) + sum(sb)) * 100) => "shorthold_ppv%",
                        [:truth_shortopen, :truth_longopen, :truth_longhold, :truth_allclose, :truth_shorthold] => ((sb, lb, lh, ac, sh) -> sum(sb) / (sum(sb) + sum(lb) + sum(lh) + sum(ac) + sum(sh)) * 100) => "shortopen_ppv%")
    else
        (verbosity >= 2) && println("cannot get confusion matrices")
        ccmdf = DataFrame()
    end
    if size(xcmdf, 1) > 0
        # cmdfgrp = groupby(xcmdf, [:coin, :set, :prediction])
        xcmdf = @view xcmdf[.!ismissing.(xcmdf[!, :set]), :] # exclude gaps between set partitions
        # println("DEBUG xcmdf=$(xcmdf[1:100, :])")
        xcmdfgrp = groupby(xcmdf, [:set, :pred_label, :bin])
        cxcmdf = combine(xcmdfgrp, [:tp, :fp] => ((tp, fp) -> sum(tp) / (sum(tp) + sum(fp)) * 100) => "ppv%")
    else
        (verbosity >= 2) && println("cannot get confusion matrices")
        cxcmdf = DataFrame()
    end
    return ccmdf, cxcmdf
end

function gainspipeline(cfg)
    # getclassifier(cfg) # ensure preparation of baseline mix classifier
    cmdf, xcmdf = getconfusionmatrices(cfg)
    @assert isnothing(cmdf) == isnothing(xcmdf) "unexpected cmdf and xcmdf existence mismatch with isnothing(cmdf)=$(isnothing(cmdf)) and isnothing(xcmdf)=$(isnothing(xcmdf))"
    if !isnothing(cmdf) && (size(cmdf, 1) > 0)
        println("$(EnvConfig.now()) Confusion matrix: $cmdf")
        println("$(EnvConfig.now()) Extended confusion matrix: $xcmdf")
        # ccmdf,cxcmdf = averageconfusionmatrix(cfg)
        # println("Average extended confusion matrix: $cxcmdf")
        # println("Average confusion matrix: $ccmdf")
    end
    gaindf = getgainsdf(cfg)
    if !isnothing(gaindf) && (size(gaindf, 1) > 0)
        gaindfgroup = groupby(gaindf, [:set, :side, :predicted, :openthreshold, :closethreshold])
        cgaindf = combine(gaindfgroup, :gain => mean, nrow, :gain => sum, :gainquote => sum)
        sort!(cgaindf, [:set, :side, :openthreshold, :closethreshold])
        println("$(EnvConfig.now()) cgaindf=$cgaindf")
    end
end

function safe(f, v; default=missing)
    v = skipmissing(v)
    isempty(v) ? default : f(v)
end

function introspection(cfg::TrendDetectorConfig)
    TrendDetector.verbosity = 2
    Ohlcv.verbosity = 1
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    if EnvConfig.tableexists(TradingStrategy.targetissuesfilename())
        targetissuespath = EnvConfig.tablepath(TradingStrategy.targetissuesfilename(); format=:auto)
        targetissuesdf = EnvConfig.readdf(TradingStrategy.targetissuesfilename())
        println("target issues file: $(targetissuespath)")
        if isnothing(targetissuesdf) || size(targetissuesdf, 1) == 0
            println("targetissues.arrow is present but empty")
        else
            println("size(targetissuesdf) = $(size(targetissuesdf))")
            println("describe(targetissuesdf, :all)=$(describe(targetissuesdf, :all))")
            show(targetissuesdf, truncate=100)
            println()
        end
    else
        println("No target issues file found in $(EnvConfig.logfolder())")
    end
    resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg)
    if isnothing(featuresdf) || size(featuresdf, 1) == 0
        println("No coin-specific trend features cache found in $(EnvConfig.logfolder())")
    else
        println("coin-specific trend features caches for $(length(cachedcoins)) coins -> concatenated size(featuresdf) = $(size(featuresdf))")
        println("describe(featuresdf, :all)=$(describe(featuresdf, :all))")
    end
    if isnothing(resultsdf) || size(resultsdf, 1) == 0
        println("No coin-specific trend results cache found in $(EnvConfig.logfolder())")
    else
        println("coin-specific trend results caches for $(length(cachedcoins)) coins -> concatenated size(resultsdf) = $(size(resultsdf))")
        println("describe(resultsdf, :all)=$(describe(resultsdf, :all))")
        println("$(unique(resultsdf[!, :coin])) processable coins")
        println("used targets: $(unique(resultsdf[!, :target]))")
        println("rangeid sorted = $(issorted(resultsdf[!, :rangeid]))")
        for coin in cachedcoins
            coin_results = @view resultsdf[resultsdf[!, :coin] .== coin, :]
            print("\rcoin=$coin, opentime sorted = $(issorted(coin_results[!, :opentime])), rangeid sorted = $(issorted(coin_results[!, :rangeid]))")
        end
    end
    preddf = EnvConfig.readdf(TradingStrategy.predictionsfilename())
    if !isnothing(preddf) && (size(preddf, 1) > 0)
        println("$(TradingStrategy.predictionsfilename()): size(preddf) = $(size(preddf))")
        println("describe(preddf, :all)=$(describe(preddf, :all))")
        if :score in propertynames(preddf)
            badix, badreason = _first_invalid_score(preddf[!, :score])
            first_invalid_ix = isnothing(badix) ? "none" : string(badix)
            reason = isnothing(badix) ? "none" : badreason
            println("predictions score integrity: valid=$(isnothing(badix)) first_invalid_ix=$(first_invalid_ix) reason=$(reason)")
            if !isnothing(badix)
                println("first invalid prediction row: $(preddf[badix, :])")
            end
        else
            println("predictions score integrity: missing :score column in preddf with columns=$(propertynames(preddf))")
        end
    else
        println("No results file found in $(EnvConfig.logfolder()) - size(preddf)=$(isnothing(preddf) ? "nothing" : size(preddf))")
    end
end

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing,AbstractString}=nothing)
    prefix = key * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "="; limit=2)[2]
        end
    end
    return default
end

function _normalize_runid_token(value)::String
    token = replace(lowercase(strip(String(value))), r"[^a-z0-9._-]+" => "_")
    return isempty(token) ? "na" : token
end

function _set_deterministic_run_id!(args::Vector{String}, context::Vector{Pair{String, String}}=Pair{String, String}[])
    explicit = _argvalue(args, "runid", nothing)
    if !isnothing(explicit)
        runid = _normalize_runid_token(explicit)
        ENV["CTS_RUN_ID"] = runid
        println("$(EnvConfig.now()) CTS_RUN_ID=$(runid) (explicit)")
        return runid
    end

    argtokens = String[]
    for arg in args
        startswith(arg, "runid=") && continue
        if occursin("=", arg)
            parts = split(arg, "="; limit=2)
            push!(argtokens, "$( _normalize_runid_token(parts[1]) )=$( _normalize_runid_token(parts[2]) )")
        else
            push!(argtokens, _normalize_runid_token(arg))
        end
    end
    sort!(argtokens)
    ctxtokens = ["$( _normalize_runid_token(kv.first) )=$( _normalize_runid_token(kv.second) )" for kv in context]
    sort!(ctxtokens)
    runid = join(vcat(["trenddetector"], ctxtokens, argtokens), "__")
    ENV["CTS_RUN_ID"] = runid
    println("$(EnvConfig.now()) CTS_RUN_ID=$(runid)")
    return runid
end

function _parse_bool(raw::AbstractString)::Bool
    value = lowercase(strip(raw))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    error("classbalancing=$(raw) must be one of true/false, yes/no, on/off, 1/0")
end

function _parse_csv_tokens(raw::AbstractString)::Vector{String}
    tokens = [uppercase(strip(token)) for token in split(String(raw), ",") if !isempty(strip(token))]
    @assert !isempty(tokens) "coins=$(raw) must provide at least one non-empty token"
    return unique(tokens)
end

function _clear_test_trade_cache!()
    EnvConfig.deletefolder("trades")
    return nothing
end

function buildcfg(args::Vector{String}, allowedcoins::Vector{String}, startdt::DateTime, enddt::DateTime, defaultfoldersuffix::AbstractString, opmode::TrendDetectorMode)
    configref = _argvalue(args, "config", "046")
    basecfg = TradingStrategy.trenddetectorconfig(configref)
    configname = _argvalue(args, "configname", string(basecfg.configname))
    folder = _argvalue(args, "folder", "Trend-$configname-$defaultfoldersuffix")
    classbalancing_default = (:classbalancing in keys(basecfg)) ? string(getfield(basecfg, :classbalancing)) : "true"
    classbalancing = _parse_bool(_argvalue(args, "classbalancing", classbalancing_default))
    mergedcfg = merge(basecfg, (configname=configname, folder=folder, classbalancing=classbalancing))
    return TrendDetectorConfig(; mergedcfg..., coins=allowedcoins, startdt=startdt, enddt=enddt, opmode=opmode)
end

"""
Return whether the CLI arguments request the help output.
"""
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
Return CLI help text for `TrendDetector.jl`.
"""
function trenddetectorhelp()::String
    return """
Usage:
    julia --project=. scripts/TrendDetector.jl [help] [test|train|gain] [inspect] [special] [retrain] [key=value ...]

Flag parameters:
  help, --help, -h
      Show this message and exit.
      Default: false

  test
    Use `EnvConfig.init(test)` with `TradingStrategy.testcoins()`.
      Default: true

    train
        Use `EnvConfig.init(training)` with `TradingStrategy.traincoins()`.
            Default: false

    gain
        Inference-only mode. Uses the selected data phase (`test` or `train`) and
        loads an existing classifier from `Trend-<config>-<phase>` unless
        `TRENDDETECTOR_CLASSIFIER_FOLDER` overrides it.
        Uses the explicit `startdt..enddt` window directly (no liquidity-range
        filtering and no partition split).
        Writes outputs to `Trend-<config>-gain-<phase>` by default.
            Default: false

  inspect
      Print cached features, targets, predictions, and `results/targetissues.arrow` when present, without training/evaluation.
      Default: false

  special
      Enable special mode, which currently a defined limited time range with 2 trading pairs to have a limited comparison for tradesim.
      Default: false

  retrain
      Retrain non-converged classifiers instead of reusing them.
      Default: false

Key=value parameters:
  config=<configname>
    Trend preset from `TREND_DETECTOR_CONFIGS` in `TradingStrategy/src/tradingstrategyconfig.jl`.
      Default: `029`

  configname=<name>
      Optional output name override.
      Default: same as `config`

  folder=<name>
      Output subfolder.
      Default: `Trend-<configname>-<mode>` where mode is `test`, `training`, `gain-test`, or `gain-training`

  classbalancing=<Bool>
      Apply inverse-frequency class weights during training.
      Default: preset value (for `029`: `false`)

  startdt=<DateTime>
      Override start datetime (ISO-8601 format).
      Example: `2025-07-01T01:00:00`

  enddt=<DateTime>
      Override end datetime (ISO-8601 format).
      Example: `2025-07-30T01:00:00`

  coins=<CSV>
      Override trading pair bases as a comma-separated list.
      Example: `coins=SINE,BTC,ETH`

Fixed date defaults:
  train startdt: `2017-11-17T20:56:00`
  test startdt: `2025-01-17T20:56:00`
  enddt: `2025-08-10T15:00:00`
"""
end

"""
Run the `TrendDetector` script with the given CLI arguments.
"""
function main(args::Vector{String}=ARGS)
    if _wants_help(args)
        println(trenddetectorhelp())
        return nothing
    end

    # startdt = nothing  # means use all what is stored as canned data
    # enddt = nothing  # means use all what is stored as canned data
    startdt = DateTime("2017-11-17T20:56:00")
    enddt = DateTime("2025-08-10T15:00:00")

    println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$(args)")
    global retrain = "retrain" in args
    retrain && println("retrain mode activated - existing classifiers that did not converge will be overwritten")
    has_test = "test" in args
    has_train = "train" in args
    has_gain = "gain" in args
    train_or_test_count = (has_test ? 1 : 0) + (has_train ? 1 : 0)
    @assert train_or_test_count <= 1 "mode flags are exclusive for phase selection; use only one of test or train"

    testmode = true
    trainmode = false
    if has_train
        testmode = false
        trainmode = true
    elseif has_test
        testmode = true
        trainmode = false
    end
    inspectonly = "inspect" in args
    specialonly = "special" in args
    opmode = has_gain ? gain : (specialonly ? special : (inspectonly ? inspect : execute))
    # inspectonly = specialonly ? true : inspectonly # if specialonly then also do inspection

    global verbosity = 2
    allowedcoins = String[]
    if testmode
        global verbosity = 2
        Ohlcv.verbosity = 1 # 3
        Features.verbosity = 1 # 3
        Targets.verbosity = 1 # 3
        EnvConfig.verbosity = 1
        Classify.verbosity = 3
        allowedcoins = TradingStrategy.testcoins()
        EnvConfig.init(test)
        startdt = DateTime("2025-01-17T20:56:00")
        enddt = DateTime("2025-08-10T15:00:00")
    else # training or production
        global verbosity = 2
        Ohlcv.verbosity = 1
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
        EnvConfig.init(training)
        allowedcoins = TradingStrategy.traincoins()
    end

    if opmode == gain
        # Gain mode is inference-only and uses the selected phase context.
        EnvConfig.verbosity = 1
    end

    if specialonly
        Ohlcv.verbosity = 1
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
        allowedcoins = ["SINE"] # , "DOUBLESINE"
        startdt = DateTime("2025-06-01T04:01:00")
        enddt = DateTime("2025-07-20T04:01:00") # DateTime("2025-07-30T01:00:00")
    end

    startdt_arg = _argvalue(args, "startdt", nothing)
    if !isnothing(startdt_arg)
        startdt = DateTime(String(startdt_arg))
    end
    enddt_arg = _argvalue(args, "enddt", nothing)
    if !isnothing(enddt_arg)
        enddt = DateTime(String(enddt_arg))
    end
    coins_arg = _argvalue(args, "coins", nothing)
    if !isnothing(coins_arg)
        allowedcoins = _parse_csv_tokens(String(coins_arg))
    end
    @assert startdt <= enddt "startdt=$(startdt) must be <= enddt=$(enddt)"

    EnvConfig.setcoinspath!("Bybit")
    (verbosity >= 2) && println("coinspath: $(EnvConfig.coinspath())")

    phase = string(Symbol(EnvConfig.configmode))
    folder_suffix = opmode == gain ? "gain-$phase" : phase
    global cfg = buildcfg(args, allowedcoins, startdt, enddt, folder_suffix, opmode)
    testmode # && _clear_test_trade_cache!()
    _set_deterministic_run_id!(args, [
        "mode" => (opmode == gain ? "gain-$phase" : phase),
        "configname" => cfg.configname,
        "folder" => cfg.folder,
        "testmode" => string(testmode),
        "gainmode" => string(opmode == gain),
        "retrain" => string(retrain),
    ])

    if specialonly
        # renamepredictionfiles([TradingStrategy.mk001config().folder, TradingStrategy.mk002config().folder, TradingStrategy.mk003config().folder, TradingStrategy.mk004config().folder, TradingStrategy.mk005config().folder])
        println("create comparison basis for tradesim using $allowedcoins in special mode with startdt=$startdt and enddt=$enddt")
        gainspipeline(cfg)
    elseif inspectonly
        introspection(cfg)
    else
        gainspipeline(cfg)
    end

    println("$(EnvConfig.now()) done @ $(cfg.folder)")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end

end # of TrendDetector
