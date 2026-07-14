module TrendDetector
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics, MLUtils, StatisticalMeasures
using CategoricalArrays, CategoricalDistributions, Distributions
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy, Trade, Xch, Bybit

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
"""
@enum TrendDetectorMode inspect execute special

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

function _normalize_tradelabel_column!(df::AbstractDataFrame, col::Symbol)
    if col in propertynames(df)
        df[!, col] = [_normalize_tradelabel(value) for value in df[!, col]]
    end
    return df
end

function _normalize_set_column!(df::AbstractDataFrame)
    if :set in propertynames(df)
        setcol = df[!, :set]
        if !(setcol isa CategoricalVector) && !(Base.nonmissingtype(eltype(setcol)) <: CategoricalValue)
            df[!, :set] = CategoricalVector(string.(setcol), levels=TradingStrategy.settypes())
        end
    end
    return df
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

function _load_featuretarget_pair(coin::AbstractString)
    resultsdf = EnvConfig.readdf(TradingStrategy.resultsfilename(coin))
    featuresdf = EnvConfig.readdf(TradingStrategy.featuresfilename(coin))
    @assert isnothing(resultsdf) == isnothing(featuresdf) "unexpected mismatch of resultsdf and featuresdf existence for coin=$(coin) with resultsdf existence $(isnothing(resultsdf)) and featuresdf existence $(isnothing(featuresdf))"

    if !isnothing(resultsdf)
        resultsdf = DataFrame(resultsdf; copycols=true)
        featuresdf = DataFrame(featuresdf; copycols=true)
        if :sampleix in propertynames(resultsdf)
            select!(resultsdf, Not(:sampleix))
        end
        _normalize_tradelabel_column!(resultsdf, :target)
        _normalize_tradelabel_column!(resultsdf, :label)
        _normalize_set_column!(resultsdf)
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

    if isempty(resultparts)
        return nothing, nothing, String[]
    end

    resultsdf = length(resultparts) == 1 ? resultparts[1] : vcat(resultparts...; cols=:union)
    featuresdf = length(featureparts) == 1 ? featureparts[1] : vcat(featureparts...; cols=:union)
    @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of concatenated results/features size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
    return resultsdf, featuresdf, cachedcoins
end

function getfeaturestargetsdf(cfg::TrendDetectorConfig)
    resultsdf = featuresdf = nothing
    (verbosity >= 2) && println("$(EnvConfig.now()) get features and targets                             ")

    resultsdf, featuresdf, cachedcoins = _concat_coin_featuretarget_caches(cfg)
    if !isnothing(resultsdf)
        (verbosity >= 2) && println("$(EnvConfig.now()) using $(length(cachedcoins)) coin-specific cached trend feature/target pairs")
    end

    if isnothing(resultsdf)
        cl = _trendclassifierseed(cfg)
        rangeid = UInt16(1) # shall be unique across coins
        skippedcoins = String[]
        processedcoins = String[]
        targetissuesdf = DataFrame()

        for coinix in eachindex(cfg.coins)
            coin = cfg.coins[coinix]
            coinresultsdf = coinfeaturesdf = nothing
            resultsdf = featuresdf = nothing
            (verbosity >= 2) && print("calculating $coin ($coinix/$(length(cfg.coins))) liquid ranges, features and targets                                                          \r")
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
            rv = Ohlcv.liquiditycheck(Ohlcv.ohlcvview(ohlcv, startix:endix))

            for rngix in eachindex(rv) # rng indices are related to the ohlcvview dataframe rows
                rng = rv[rngix]
                rng = rng .+ (startix - 1) # adjust to complete ohlcv dataframe row indices
                if rng[end] - rng[begin] > 0
                    (verbosity >= 2) && print("$(EnvConfig.now()) calculating features and targets for $coin ($coinix/$(length(cfg.coins))) range ($rngix/$(length(rv))) $rng from $(ot[rng[begin]]) until $(ot[rng[end]]) with $(rng[end] - rng[begin]) samples                \r")
                    (verbosity >= 3) && println()
                    rngohlcv = Ohlcv.ohlcvview(ohlcv, rng)
                    trgcfg = cfg.targetconfig
                    rngresults, rngfeatures = Classify.featurestargetsdf(
                        cl,
                        rngohlcv,
                        trgcfg;
                        partitionconfig=cfg.partitionconfig,
                        coin=coin,
                        rangeid_start=rangeid,
                    )
                    issues = Targets.crosscheck(trgcfg, rngresults[!, :target], rngresults[!, :pivot])
                    if !isnothing(issues) && (length(issues) > 0)
                        if size(targetissuesdf, 1) > 0
                            targetissuesdf = vcat(targetissuesdf, DataFrame(issue=issues, coin=CategoricalVector(fill(coin, length(issues)), levels=cfg.coins), rangeid=fill(rangeid, length(issues))))
                        else
                            targetissuesdf = DataFrame(issue=issues, coin=CategoricalVector(fill(coin, length(issues)), levels=cfg.coins), rangeid=fill(rangeid, length(issues)))
                        end
                    end
                    if size(rngresults, 1) > 0
                        rangeid = UInt16(maximum(rngresults[!, :rangeid]) + 1)
                    end
                    coinresultsdf = isnothing(coinresultsdf) ? rngresults : vcat(coinresultsdf, rngresults)
                    coinfeaturesdf = isnothing(coinfeaturesdf) ? rngfeatures : vcat(coinfeaturesdf, rngfeatures)
                else
                    @error "unexpected zero length range for " ohlcv.base rng rv
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
        (verbosity >= 3) && (length(skippedcoins) > 0) && println("skipped to process $skippedcoins due to no liquid ranges")
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

function _trendclassifierspec(cfg::TrendDetectorConfig)
    return (
        config_ref=cfg.configname,
        featconfig=() -> cfg.featconfig,
        targetconfig=() -> cfg.targetconfig,
        folder=cfg.folder,
    )
end

function _trendclassifierseed(cfg::TrendDetectorConfig)::Classify.AbstractClassifier
    featurecount = Features.featurecount(cfg.featconfig)
    labels = Targets.uniquelabels(cfg.targetconfig)
    mnemonic = classifiermenmonic()
    spec = _trendclassifierspec(cfg)
    return Classify.loadorbuild(
        cfg.classifiertype,
        spec,
        featurecount,
        labels,
        mnemonic,
        cfg.classifiermodel;
        mode=EnvConfig.configmode,
        folder=cfg.folder,
    )
end

function getruntimeclassifier(cfg::TrendDetectorConfig)::Classify.AbstractClassifier
    cl = _trendclassifierseed(cfg)
    model = Classify.model(cl)

    if !Classify.isadapted(cl) || (!Classify.nnconverged(cl) && retrain)
        println("$(EnvConfig.now()) adapting one mix classifier for all coins")
        resultsdf, featuresdf = getfeaturestargetsdf(cfg)
        if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
            return cl
        end
        Classify.adapt!(
            cl,
            resultsdf,
            featuresdf;
            settype="train",
            classbalancing=cfg.classbalancing,
            retrain=retrain,
            save_after=true,
            mode=EnvConfig.configmode,
            folder=cfg.folder,
        )
        (verbosity >= 3) && showlosses(model)
        println("$(EnvConfig.now()) finished adapting mix classifier - classifier $(Classify.nnconverged(cl) ? "did" : "did not") converge")
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

"""
Returns the max prediction with its corresponding trade label for the samples of all coins. 
The returned DataFrame provides one score::Float32 column and one label::TradeLabel column representing the best sample prediction + the original targets::TradeLabel and set::CategoricalVector.
"""
function getmaxpredictionsdf(cfg::TrendDetectorConfig)
    predictionsdf = EnvConfig.readdf(TradingStrategy.predictionsfilename())
    if !isnothing(predictionsdf)
        predictionsdf = DataFrame(predictionsdf; copycols=true)
        _normalize_tradelabel_column!(predictionsdf, :label)
    end
    depfiles = _featuretarget_cachefiles(cfg)
    if !isnothing(predictionsdf) && !isfreshcache(TradingStrategy.predictionsfilename(), depfiles)
        @warn "ignoring stale max predictions cache; rebuilding from newer coin-specific trend feature/target caches"
        EnvConfig.deletefolder(TradingStrategy.predictionsfilename())
        predictionsdf = nothing
    end
    # predictions are stored in a predictionsdf to avoid loading every time also features bu eventually you want the whole resultdf with predictions
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        cl = getruntimeclassifier(cfg)
        resultsdf, featuresdf = getfeaturestargetsdf(cfg) 
        (verbosity >= 2) && print("$(EnvConfig.now()) get maximum predictions                             \r")
        (verbosity >= 3) && println()
        predictionsdf = Classify.maxpredictdf(cl, featuresdf)
        _normalize_tradelabel_column!(predictionsdf, :label)
        @assert size(predictionsdf, 1) == size(featuresdf, 1) == size(resultsdf, 1) "size(predictionsdf, 1)=$(size(predictionsdf, 1)) != size(featuresdf, 1)=$(size(featuresdf, 1)) != size(resultsdf, 1)=$(size(resultsdf, 1)) for mix"
        if (size(resultsdf, 1) > 0)
            EnvConfig.savedf(predictionsdf, TradingStrategy.predictionsfilename())
        end
    end
    if !isnothing(predictionsdf) && (size(predictionsdf, 1) > 0)
        resultsdf, _ = getfeaturestargetsdf(cfg)
        if !isnothing(resultsdf)
            resultsdf = DataFrame(resultsdf; copycols=true)
            if :sampleix in propertynames(resultsdf)
                select!(resultsdf, Not(:sampleix))
            end
        end
        @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(predictionsdf, 1) > 0) "size mismatch: size(resultsdf, 1)=$(isnothing(resultsdf) ? "nothing" : size(resultsdf, 1)), size(predictionsdf, 1)=$(size(predictionsdf, 1))"
        _normalize_tradelabel_column!(resultsdf, :target)
        resultsdf[:, :score] = predictionsdf[!, :score]
        resultsdf[:, :label] = predictionsdf[!, :label]
    else
        resultsdf = nothing
    end
    return resultsdf
end

function addgainadmin!(gdf, coin, sampleset, predicted, rangeid, openthreshold, closethreshold; pair=nothing)
    coinstr = String(coin)
    pairstr = isnothing(pair) ? Xch.tradingpairkey(coinstr, EnvConfig.pairquote) : String(pair)
    gdf[!, :coin] = CategoricalVector(fill(coinstr, size(gdf, 1)))
    gdf[!, :pair] = fill(pairstr, size(gdf, 1))
    gdf[!, :set] = fill(sampleset, size(gdf, 1))
    gdf[!, :predicted] = fill(predicted, size(gdf, 1))
    gdf[!, :rangeid] = fill(rangeid, size(gdf, 1))
    gdf[!, :openthreshold] = fill(openthreshold, size(gdf, 1))
    gdf[!, :closethreshold] = fill(closethreshold, size(gdf, 1))
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

const GAIN_THRESHOLDS = Tuple((openthreshold, closethreshold) for openthreshold in TradingStrategy.default_openthresholds() for closethreshold in TradingStrategy.default_closethresholds() if closethreshold <= openthreshold)
const TRUE_GAIN_THRESHOLD = (0.9f0, 0.9f0)

function getgainsdf(cfg::TrendDetectorConfig)
    gaindeps = vcat(_featuretarget_cachefiles(cfg; include_features=false), [TradingStrategy.predictionsfilename()])
    if isfreshcache(TradingStrategy.gainsfilename(), gaindeps)
        gaindf = TradingStrategy.loadtrades(; stem="gains")
        if size(gaindf, 1) > 0
            return gaindf
        end
    end

    resultsdf = getmaxpredictionsdf(cfg) # DataFrame with columns: target, opentime, high, low, close, pivot, coin, rangeid, set, score, label
    if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
        return nothing
    end

    ts = TradingStrategy.TsCache(strategy=TradingStrategy.strategyconfig(cfg.configname), source="trenddetector:$(cfg.configname)")
    xc = Xch.XchCache(Bybit.BybitCache(); startdt=cfg.startdt, enddt=cfg.enddt)
    Xch.ensuretradesschema(xc, Xch.tradesdf_all_contributors())

    # Range ids can collide across independently cached coins/runs. Replay must
    # stay scoped to coin+set+rangeid to avoid mixing samples across ranges.
    rangegroups = groupby(resultsdf, [:coin, :set, :rangeid])
    gainparts = DataFrame[]
    sizehint!(gainparts, (length(GAIN_THRESHOLDS) + 1) * length(rangegroups))

    for (rngix, resultsview) in enumerate(rangegroups)
        rng = resultsview[begin, :rangeid]
        (verbosity >= 2) && print("$(EnvConfig.now()) calculating gains for range ($rngix/$(length(rangegroups))) $rng                             \r")
        (verbosity >= 3) && println()
        @assert size(resultsview, 1) > 0 "unexpected empty resultsview for rangeid $rng"

        coin = resultsview[begin, :coin]
        sampleset = resultsview[begin, :set]
        scores = resultsview[!, :score]
        labels = resultsview[!, :label]
        targets = resultsview[!, :target]
        truescores = fill(1f0, size(resultsview, 1))
        evaldt = resultsview[end, :opentime]

        # Process predicted gains using strategy config thresholds
        open_threshold = cfg.tradingstrategy.openthreshold
        close_threshold = cfg.tradingstrategy.closethreshold
        tp = TradingStrategy.preparereplaytrades!(
            ts,
            xc,
            String(coin),
            resultsview,
            scores,
            labels,
            metadata=Dict{Symbol, Any}(:set => String(sampleset), :rangeid => Int(rng)),
            datetime=evaldt,
        )
        gdf = TradingStrategy.processreplaygains!(
            tp;
            strategy=cfg.tradingstrategy,
        )
        if size(gdf, 1) > 0
            addgainadmin!(gdf, coin, sampleset, true, rng, open_threshold, close_threshold; pair=tp.pair)
            push!(gainparts, gdf)
        end

        # Process labeled truth gains using TRUE_GAIN_THRESHOLD
        true_open, true_close = TRUE_GAIN_THRESHOLD
        tp = TradingStrategy.preparereplaytrades!(
            ts,
            xc,
            String(coin),
            resultsview,
            truescores,
            targets,
            metadata=Dict{Symbol, Any}(:set => String(sampleset), :rangeid => Int(rng)),
            datetime=evaldt,
        )
        gdf = TradingStrategy.processreplaygains!(
            tp;
            strategy=cfg.tradingstrategy,
        )
        if size(gdf, 1) > 0
            addgainadmin!(gdf, coin, sampleset, false, rng, true_open, true_close; pair=tp.pair)
            push!(gainparts, gdf)
        end
    end

    gaindf = isempty(gainparts) ? nothing : reduce(vcat, gainparts)
    if !isnothing(gaindf) && (size(gaindf, 1) > 0)
        gaindf = gaindf[.!ismissing.(gaindf[!, :set]), :] # exclude gaps between set partitions
        if size(gaindf, 1) > 0
            keycols = Symbol[:coin, :set, :predicted, :rangeid, :openthreshold, :closethreshold, :trend, :startdt, :enddt]
            if all(col -> col in names(gaindf), keycols)
                keycounts = combine(groupby(gaindf, keycols), nrow => :rows)
                dupmask = keycounts[!, :rows] .> 1
                @assert !any(dupmask) "duplicate gain segments detected per key $(keycols); duplicates=$(sum(dupmask))"
            end
            sort!(gaindf, [:coin, :predicted, :openthreshold, :closethreshold, :startdt, :trend])
            TradingStrategy.savetrades(gaindf; stem="gains")

            present = Set(String.(unique(gaindf[!, :coin])))
            missing_coins = [coin for coin in cfg.coins if !(String(coin) in present)]
            if !isempty(missing_coins)
                @warn "missing coins in gains output" missing_coins present_coins=collect(present)
            end
        end
    end

    pairkeys = Xch.tradingpairs(xc)
    if !isempty(pairkeys)
        tradeparts = DataFrame[]
        for pair in pairkeys
            tdf = Xch.trades(xc, pair)
            size(tdf, 1) > 0 || continue
            push!(tradeparts, DataFrame(tdf; copycols=true))
        end
        if !isempty(tradeparts)
            tradesv1 = reduce(vcat, tradeparts; cols=:union)
            sortcols = Symbol[]
            for c in [:coin, :set, :rangeid, :opentime]
                (c in names(tradesv1)) && push!(sortcols, c)
            end
            !isempty(sortcols) && sort!(tradesv1, sortcols)
            TradingStrategy.savetrades(tradesv1; stem="tradesV1")
        end
    end

    (verbosity >= 2) && println("$(EnvConfig.now()) calculated gains for $(length(rangegroups)) ranges")
    return gaindf
end


"""
Provides distance information between neighboring predicted gain segments in form of one data frame row for each precited positive segment with the following columns:  
- :coin and :set as taken over from gains
- :trend is the predicted trend
- :tpdistnext = in case the predicted segment does overlap with a labeled segment (= true positive) and the next segment is also a true positive segment of the same labeled segment, distance to the next true positive predicted segment
- :fpdistnext = in case the predicted segment does not overlap with a labeled segment (= false positive), distance to the next predicted segment
- :distfirst = in case of the first true positive predicted segment of a labeled segment, distance to the beginning of the labeled  sgement to see how long the prediction needs to detect a true positive situation
- :distlast - in case of the last true positive predcited segment, distance to the end of the labelled segment
- :startdt, :enddt provide the timestamps of the predicted segment
- :truestartdt, :trueenddt provide the timestamps of the labelled segment
"""
function getdistances(cfg::TrendDetectorConfig)
    if isfreshcache(TradingStrategy.distancesfilename(), [TradingStrategy.gainsfilename()])
        distdf = EnvConfig.readdf(TradingStrategy.distancesfilename())
        if !isnothing(distdf) && (size(distdf, 1) > 0)
            return distdf
        end
    end

    gaindf = getgainsdf(cfg)
    if isnothing(gaindf) || (size(gaindf, 1) == 0)
        (verbosity >= 1) && println("skipping distances collection due to missing gains")
        return DataFrame()
    end

    predmask = gaindf[!, :predicted] .== true
    if !any(predmask)
        (verbosity >= 1) && println("skipping distances collection due to missing predicted gains")
        return DataFrame()
    end

    # Limit predicted gains to the best threshold pair, but keep all labeled truth gains.
    openmin = minimum(gaindf[predmask, :openthreshold])
    closemin = minimum(gaindf[predmask, :closethreshold])
    usemask = (gaindf[!, :predicted] .== false) .|| ((gaindf[!, :predicted] .== true) .&& (gaindf[!, :openthreshold] .== openmin) .&& (gaindf[!, :closethreshold] .== closemin))
    gaindf1 = @view gaindf[usemask, :]
    trendlevels = unique(gaindf1[!, :trend])

    coinvals = String[]
    setvals = String[]
    trendvals = eltype(gaindf1[!, :trend])[]
    tpdistnextvals = Union{Missing, Int64}[]
    fpdistnextvals = Union{Missing, Int64}[]
    distfirstvals = Union{Missing, Int64}[]
    distlastvals = Union{Missing, Int64}[]
    startdtvals = DateTime[]
    enddtvals = DateTime[]
    truestartdtvals = Union{Missing, DateTime}[]
    trueenddtvals = Union{Missing, DateTime}[]

    for coinix in eachindex(cfg.coins)
        coin = cfg.coins[coinix]
        (verbosity >= 2) && print("$(EnvConfig.now()) calculating distances for $coin ($coinix/$(length(cfg.coins)))                             \r")
        (verbosity >= 3) && println()

        coingaindf = @view gaindf1[gaindf1[!, :coin] .== coin, :]
        if size(coingaindf, 1) == 0
            continue
        end
        
        gaindfgrp = groupby(coingaindf, [:predicted, :trend])
        haspredictions = false

        for trend in trendlevels
            cpgaindf = get(gaindfgrp, (true, trend), DataFrame())   # predicted gains of one trend
            if size(cpgaindf, 1) == 0
                continue
            end
            haspredictions = true
            ctgaindf = get(gaindfgrp, (false, trend), DataFrame())  # labeled gains of the same trend

            if (size(cpgaindf, 1) > 1) && !issorted(cpgaindf[!, :startdt])
                cpgaindf = sort(cpgaindf, :startdt)
            end
            if (size(ctgaindf, 1) > 1) && !issorted(ctgaindf[!, :startdt])
                ctgaindf = sort(ctgaindf, :startdt)
            end

            cpstart = cpgaindf[!, :startdt]
            cpend = cpgaindf[!, :enddt]
            cpset = cpgaindf[!, :set]

            if size(ctgaindf, 1) > 0
                ctstart = ctgaindf[!, :startdt]
                ctend = ctgaindf[!, :enddt]
                ctix = firstindex(ctstart)
            else
                ctix = 0
            end

            for cpix in eachindex(cpstart)
                cpnix = cpix < lastindex(cpstart) ? cpix + 1 : 0
                distnext = cpnix == 0 ? missing : Minute(cpstart[cpnix] - cpend[cpix]).value
                @assert (cpnix == 0) || (cpend[cpix] < cpstart[cpnix]) "cpgaindf[cpix=$cpix, :]=$(cpgaindf[cpix, :]), cpgaindf[cpnix=$cpnix, :]=$(cpgaindf[cpnix, :])"

                while (ctix != 0) && (cpstart[cpix] > ctend[ctix])
                    ctix = ctix < lastindex(ctend) ? ctix + 1 : 0
                end

                tpdistnext = fpdistnext = distfirst = distlast = missing
                truestartdt = trueenddt = missing

                if ctix == 0
                    fpdistnext = distnext
                else
                    truestartdt = ctstart[ctix]
                    trueenddt = ctend[ctix]
                    if cpend[cpix] < truestartdt
                        fpdistnext = distnext
                    else
                        prev_same_true = (cpix > firstindex(cpstart)) && (cpend[cpix - 1] >= truestartdt)
                        distfirst = prev_same_true ? missing : Minute(cpstart[cpix] - truestartdt).value
                        next_same_true = (cpnix != 0) && !(trueenddt < cpstart[cpnix])
                        distlast = next_same_true ? missing : Minute(cpend[cpix] - trueenddt).value
                        if next_same_true
                            tpdistnext = distnext
                        else
                            fpdistnext = distnext
                        end
                    end
                end

                push!(coinvals, coin)
                push!(setvals, string(cpset[cpix]))
                push!(trendvals, trend)
                push!(tpdistnextvals, tpdistnext)
                push!(fpdistnextvals, fpdistnext)
                push!(distfirstvals, distfirst)
                push!(distlastvals, distlast)
                push!(startdtvals, cpstart[cpix])
                push!(enddtvals, cpend[cpix])
                push!(truestartdtvals, truestartdt)
                push!(trueenddtvals, trueenddt)
            end
        end

        if !haspredictions
            (verbosity >= 1) && println("skipping distances collection of $(coin) due to missing gain predictions")
        end
    end

    distdf = DataFrame(
        coin=coinvals,
        set=setvals,
        trend=trendvals,
        tpdistnext=tpdistnextvals,
        fpdistnext=fpdistnextvals,
        distfirst=distfirstvals,
        distlast=distlastvals,
        startdt=startdtvals,
        enddt=enddtvals,
        truestartdt=truestartdtvals,
        trueenddt=trueenddtvals,
    )
    if size(distdf, 1) > 0
        EnvConfig.savedf(distdf, TradingStrategy.distancesfilename())
    end
    (verbosity >= 2) && print("$(EnvConfig.now()) calculated distances for $(length(cfg.coins)) coins                             \r")
    (verbosity >= 3) && println()
    return distdf
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
    dfp = getmaxpredictionsdf(cfg)
    if isnothing(dfp) || (size(dfp, 1) == 0)
        return nothing, nothing
    end
    dfp = @view dfp[.!ismissing.(dfp[!, :set]), :] # exclude gaps between set partitions
    (verbosity >= 2) && print("$(EnvConfig.now()) calculating confusion matrices                             \r")
    (verbosity >= 3) && println()
    if (size(dfp, 1) > 0)
        # predictedlabel = categorical(string.(dfp[!, :label]), levels=string.(Targets.uniquelabels(cfg.targetconfig)))
        # println("predictedllabels=$(unique(predictedlabel)), levels=$(levels(predictedlabel))")
        # targetlabel = categorical(string.(dfp[!, :target]), levels=string.(Targets.uniquelabels(cfg.targetconfig)))
        # println("targetlabels=$(unique(targetlabel)), levels=$(levels(targetlabel))")
        # cm = StatisticalMeasures.ConfusionMatrices.confmat(predictedlabel, targetlabel)
        # println("describe(predictions): $(describe(dfp))")
        # display(cm)
        cmdf = Classify.confusionmatrix(dfp, Targets.uniquelabels(cfg.targetconfig))
        if size(cmdf, 1) > 0
            EnvConfig.savedf(cmdf, TradingStrategy.confusionfilename())
        end
        xcmdf = Classify.extendedconfusionmatrix(dfp, Targets.uniquelabels(cfg.targetconfig))
        if size(xcmdf, 1) > 0
            EnvConfig.savedf(xcmdf, TradingStrategy.xconfusionfilename())
        end
    else
        (verbosity >= 1) && println("skipping evaluation of $(cfg.coins) due to missing predictions (size(dfp)= $(size(dfp)))")
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
        # println(describe(gaindf))
        # println(gaindf[1:2,:])
        gaindfgroup = groupby(gaindf, [:set, :trend, :predicted, :openthreshold, :closethreshold])
        # cgaindf = combine(gaindfgroup, [:truth_longbuy, :truth_allclose] => ((lb, ac) -> sum(lb) / (sum(lb) + sum(ac)) * 100) => "longbuy_ppv%")
        cgaindf = combine(gaindfgroup, :gain => mean, :samplecount => mean, nrow, :gain => sum, :gainfee => sum)
        sort!(cgaindf, [:set, :trend, :openthreshold, :closethreshold])
        println("$(EnvConfig.now()) cgaindf=$cgaindf")
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
            println(targetissuesdf)
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

function _clear_test_trade_cache!()
    EnvConfig.deletefolder("trades")
    return nothing
end

function buildcfg(args::Vector{String}, allowedcoins::Vector{String}, startdt::DateTime, enddt::DateTime)
    configref = _argvalue(args, "config", "046")
    basecfg = TradingStrategy.trenddetectorconfig(configref)
    configname = _argvalue(args, "configname", string(basecfg.configname))
    folder = _argvalue(args, "folder", "Trend-$configname-$(EnvConfig.configmode)")
    classbalancing_default = (:classbalancing in keys(basecfg)) ? string(getfield(basecfg, :classbalancing)) : "true"
    classbalancing = _parse_bool(_argvalue(args, "classbalancing", classbalancing_default))
    mergedcfg = merge(basecfg, (configname=configname, folder=folder, classbalancing=classbalancing))
    return TrendDetectorConfig(; mergedcfg..., coins=allowedcoins, startdt=startdt, enddt=enddt)
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
  julia --project=. scripts/TrendDetector.jl [help] [test|train] [inspect] [special] [retrain] [key=value ...]

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
      Default: `Trend-<configname>-$(EnvConfig.configmode)`

  classbalancing=<Bool>
      Apply inverse-frequency class weights during training.
      Default: preset value (for `029`: `false`)

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
    testmode = specialonly = true
    testmode = "test" in args ? true : "train" in args ? false : testmode
    inspectonly = "inspect" in args
    specialonly = "special" in args
    # inspectonly = specialonly ? true : inspectonly # if specialonly then also do inspection

    global verbosity = 2
    allowedcoins = String[]
    if testmode
        global verbosity = 2
        Ohlcv.verbosity = 1 # 3
        Features.verbosity = 1 # 3
        Targets.verbosity = 1 # 3
        EnvConfig.verbosity = 1
        Classify.verbosity = 1 # 3
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

    if specialonly
        Ohlcv.verbosity = 1
        Features.verbosity = 1
        Targets.verbosity = 1
        EnvConfig.verbosity = 1
        Classify.verbosity = 1
        allowedcoins = ["SINE"] # , "DOUBLESINE"
        startdt = DateTime("2025-07-01T01:00:00")
        enddt = DateTime("2025-07-30T01:00:00")
    end

    EnvConfig.setcoinspath!("Bybit")
    (verbosity >= 2) && println("coinspath: $(EnvConfig.coinspath())")

    global cfg = buildcfg(args, allowedcoins, startdt, enddt)
    testmode && _clear_test_trade_cache!()
    _set_deterministic_run_id!(args, [
        "mode" => string(Symbol(EnvConfig.configmode)),
        "configname" => cfg.configname,
        "folder" => cfg.folder,
        "testmode" => string(testmode),
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

main(ARGS)

end # of TrendDetector
