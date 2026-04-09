module TrendDetector
using Test, Dates, Logging, CSV, JDF, DataFrames, Statistics, MLUtils, StatisticalMeasures
using CategoricalArrays, CategoricalDistributions, Distributions
using EnvConfig, Classify, Ohlcv, Features, Targets, TradingStrategy

#TODO regression from last trend pivot as feature 
"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3


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
    tradingstrategy::TradingStrategy.GainSegment
    startdt::DateTime
    enddt::DateTime
    opmode::TrendDetectorMode
    partitionconfig::NamedTuple
    coins::Vector{String}
    oversampling::Bool
    function TrendDetectorConfig(;configname, folder="Trend-$configname-$(EnvConfig.configmode)", featconfig, targetconfig, classifiermodel, tradingstrategy, startdt, enddt, opmode=execute, partitionconfig=partitionconfig02(), coins, oversampling=true)
        EnvConfig.setlogpath(folder)
        (verbosity >= 2) && println("verbosity: $verbosity")
        (verbosity >= 2) && println("log folder: $(EnvConfig.logfolder())")
        (verbosity >= 2) && println("data range: $startdt - $enddt")
        (verbosity >= 2) && println("featuresconfig=$(Features.describe(featconfig))")
        (verbosity >= 2) && println("targetsconfig=$(Targets.describe(targetconfig))")
        (verbosity >= 2) && println("oversampling=$oversampling")
        return new(configname, folder, featconfig, targetconfig, classifiermodel, tradingstrategy, startdt, enddt, opmode, partitionconfig, coins, oversampling)
    end
end
cfg = nothing # to be set to a TrendDetectorConfig instance in main

include("optimizationconfigs.jl")

"""
returns targets
feature base has to be set before calling because that determines the ohlcv and relevant time range
"""
function calctargets!(trgcfg::Targets.AbstractTargets, featcfg::Features.AbstractFeatures)
    ohlcv = Features.ohlcv(featcfg)
    features = Features.features(featcfg)
    fot = Features.opentime(featcfg)
    (verbosity >= 4) && println("$(EnvConfig.now()) target calculation from $(fot[begin]) until $(fot[end])")
    Targets.setbase!(trgcfg, ohlcv)
    targets = Targets.labels(trgcfg, fot[begin], fot[end])
    # Targets.labeldistribution(targets)
    @assert size(features, 1) == length(targets) "size(features, 1)=$(size(features, 1)) != length(targets)=$(length(targets))"
    # (verbosity >= 3) && println(describe(trgcfg.df, :all))
    return targets
end


function getfeaturestargetsdf(cfg::TrendDetectorConfig)
    resultsdf = featuresdf = nothing
    (verbosity >= 2) && println("$(EnvConfig.now()) get features and targets                             ")
    if EnvConfig.isfolder(resultsfilename())
        resultsdf = EnvConfig.readdf(resultsfilename())
        featuresdf = EnvConfig.readdf(featuresfilename())
        @assert isnothing(resultsdf) == isnothing(featuresdf) "unexpected mismatch of resultsdf and featuresdf existence with resultsdf existence $(isnothing(resultsdf)) and featuresdf existence $(isnothing(featuresdf))"
        if !isnothing(resultsdf) && (:sampleix in propertynames(resultsdf))
            resultsdf = DataFrame(resultsdf)
            select!(resultsdf, Not(:sampleix))
            EnvConfig.savedf(resultsdf, resultsfilename())
        end
        @assert size(resultsdf, 1) == size(featuresdf, 1) "unexpected mismatch of resultsdf and featuresdf size with resultsdf size $(size(resultsdf, 1)) and featuresdf size $(size(featuresdf, 1))"
    else
        rangeid = Int16(1) # shall be unique across coins
        samplesets = cfg.partitionconfig.samplesets
        levels = unique(samplesets)
        # push!(levels, "noop")
        samplesets = CategoricalArray(samplesets, levels=levels)
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
            ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
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
                    Features.setbase!(cfg.featconfig, rngohlcv, usecache=true)
                    rngfeatures = Features.features(cfg.featconfig)
                    rngresults = DataFrame(target=calctargets!(cfg.targetconfig, cfg.featconfig))
                    @assert size(rngresults, 1) == size(rngfeatures, 1) == size(Ohlcv.dataframe(rngohlcv), 1) "unexpected mismatch of targets length $(size(rngresults, 1)), features size $(size(rngfeatures, 1)) and ohlcv size $(size(Ohlcv.dataframe(rngohlcv), 1)) for $(ohlcv.base) range $rng"
                    rngresults = hcat(rngresults, Ohlcv.dataframe(rngohlcv)[!, [:opentime, :high, :low, :close, :pivot]]) 
                    issues = Targets.crosscheck(cfg.targetconfig, rngresults[!, :target], rngresults[!, :pivot])
                    if !isnothing(issues) && (length(issues) > 0)
                        if size(targetissuesdf, 1) > 0
                            targetissuesdf = vcat(targetissuesdf, DataFrame(issue=issues, coin=fill(coin, length(issues)), rangeid=fill(rangeid, length(issues))))
                        else
                            targetissuesdf = DataFrame(issue=issues, coin=fill(coin, length(issues)), rangeid=fill(rangeid, length(issues)))
                        end
                    end
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
                if size(targetissuesdf, 1) > 0
                    EnvConfig.savedf(targetissuesdf, targetissuesfilename())
                end
                coinfeaturesdf = coinresultsdf = nothing # free memory
                push!(processedcoins, coin)
            else
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
                @assert size(coinfeaturesdf, 1) > 0 "unexpected empty results for $coin size(coinfeaturesdf, 1)=$(size(coinfeaturesdf, 1))"
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
                # resultsdf[:, :label] = fill(allclose, size(resultsdf, 1))
                # resultsdf[:, :score] = zeros(Float32, size(resultsdf, 1))
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
        (verbosity >= 2) && println()
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

function getlatestclassifier(cfg::TrendDetectorConfig)
    nn = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), classifiermenmonic()) # to get correct filename
    (verbosity >= 3) && println("getlatestclassifier classifier file: $(Classify.nnfilename(nn.fileprefix)), isfile=$(isfile(Classify.nnfilename(nn.fileprefix)))")
    if isfile(Classify.nnfilename(nn.fileprefix))
        nn = Classify.loadnn(nn.fileprefix)
        (verbosity >= 3) && println("getlatestclassifier loaded: nn=$(nn.fileprefix), labels=$(nn.labels) - classifier $(Classify.nnconverged(nn) ? "did" : "did not") converge")
    else
        (verbosity >= 3) && println("getlatestclassifier new: nn=$(nn.fileprefix), labels=$(nn.labels)")
    end
    @assert !isnothing(nn)
    return nn
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
    nn = getlatestclassifier(cfg)
    if !Classify.isadapted(nn) || (!Classify.nnconverged(nn) && retrain)
        println("$(EnvConfig.now()) adapting one mix classifier for all coins")
        # if classifier file does not exist then create one
        resultsdf, featuresdf = getfeaturestargetsdf(cfg) 
        featuresdf = @view featuresdf[(resultsdf[!, :set] .== "train"), :]
        resultsdf = @view resultsdf[(resultsdf[!, :set] .== "train"), :]
        if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
            return nothing
        end
        features = df2features(featuresdf, cfg)
        targets = resultsdf[!, :target]
        (verbosity >= 3) && println("$(EnvConfig.now()) size(featuresdf)=$(size(featuresdf)), size(features)=$(size(features)), size(targets)=$(size(targets)) for training mix classifier"  )
        resultsdf = featuresdf = nothing # free memory
        if cfg.oversampling
            (verbosity >= 2) && println("$(EnvConfig.now()) before correction: $(Distributions.fit(UnivariateFinite, categorical(string.(targets)))))")
            (features), targets = oversample((features), targets)  # all classes are equally trained
            # (features), targets = undersample((features), targets)  # all classes are equally trained
            (verbosity >= 2) && println("after oversampling: $(Distributions.fit(UnivariateFinite, categorical(string.(targets)))))")
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no oversampling applied - class distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(targets)))))")
        end
        Classify.adaptnn!(nn, features, targets)
        (verbosity >= 3) && showlosses(nn)
        if isnothing(nn)
            # no adaptation took place
            return nothing
        else
            # EnvConfig.savebackup(Classify.nnfilename(nn.fileprefix))
            Classify.savenn(nn)
        end
        println("$(EnvConfig.now()) finished adapting mix classifier - classifier $(Classify.nnconverged(nn) ? "did" : "did not") converge")
    end
    return nn
end

"""
Returns the max prediction with its corresponding trade label for the samples of all coins. 
The returned DataFrame provides one score::Float32 column and one label::TradeLabel column representing the best sample prediction + the original targets::TradeLabel and set::CategoricalVector.
"""
function getmaxpredictionsdf(cfg::TrendDetectorConfig)
    predictionsdf = EnvConfig.readdf(predictionsfilename()) 
    # predictions are stored in a predictionsdf to avoid loading every time also features bu eventually you want the whole resultdf with predictions
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        nn = getclassifier(cfg)
        resultsdf, featuresdf = getfeaturestargetsdf(cfg) 
        (verbosity >= 2) && print("$(EnvConfig.now()) get maximum predictions                             \r")
        (verbosity >= 3) && println()
        features = df2features(featuresdf, cfg)
        predictionsdf = Classify.maxpredictdf(nn, features)
        @assert size(predictionsdf, 1) == size(featuresdf, 1) == size(resultsdf, 1) "size(predictionsdf, 1)=$(size(predictionsdf, 1)) != size(featuresdf, 1)=$(size(featuresdf, 1)) != size(resultsdf, 1)=$(size(resultsdf, 1)) for mix"
        if (size(resultsdf, 1) > 0)
            EnvConfig.savedf(predictionsdf, predictionsfilename())
        end
    end
    if !isnothing(predictionsdf) && (size(predictionsdf, 1) > 0)
        @assert EnvConfig.isfolder(resultsfilename()) "unexpected missing resultsfile"
        resultsdf = EnvConfig.readdf(resultsfilename())
        if !isnothing(resultsdf)
            resultsdf = DataFrame(resultsdf)
            if :sampleix in propertynames(resultsdf)
                select!(resultsdf, Not(:sampleix))
                EnvConfig.savedf(resultsdf, resultsfilename())
            end
        end
        @assert !isnothing(resultsdf) && (size(resultsdf, 1) == size(predictionsdf, 1) > 0) "size mismatch: size(resultsdf, 1)=$(snothing(resultsdf) ? "nothing" : size(resultsdf, 1)), size(predictionsdf, 1)=$(size(predictionsdf, 1))"
        resultsdf[:, :score] = predictionsdf[!, :score]
        resultsdf[:, :label] = predictionsdf[!, :label]
    else
        resultsdf = nothing
    end
    return resultsdf
end

function addgainadmin!(gdf, coin, sampleset, predicted, rangeid, openthreshold, closethreshold)
    gdf[!, :coin] = fill(coin, size(gdf, 1))
    gdf[!, :set] = fill(sampleset, size(gdf, 1))
    gdf[!, :predicted] = fill(predicted, size(gdf, 1))
    gdf[!, :rangeid] = fill(rangeid, size(gdf, 1))
    gdf[!, :openthreshold] = fill(openthreshold, size(gdf, 1))
    gdf[!, :closethreshold] = fill(closethreshold, size(gdf, 1))
end

function isfreshcache(cachefile::AbstractString, dependencyfiles::AbstractVector{<:AbstractString})
    EnvConfig.tableexists(cachefile) || return false
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

const GAIN_THRESHOLDS = ((0.8f0, 0.5f0), (0.7f0, 0.5f0), (0.6f0, 0.5f0), (0.8f0, 0.6f0), (0.7f0, 0.6f0), (0.6f0, 0.55f0))
const TRUE_GAIN_THRESHOLD = (0.9f0, 0.9f0)

function getgainsdf(cfg::TrendDetectorConfig)
    if isfreshcache(gainsfilename(), [resultsfilename(), predictionsfilename()])
        gaindf = TradingStrategy.loadtrades(; stem="gains")
        if size(gaindf, 1) > 0
            return gaindf
        end
    end

    resultsdf = getmaxpredictionsdf(cfg)
    if isnothing(resultsdf) || (size(resultsdf, 1) == 0)
        return nothing
    end

    rangegroups = groupby(resultsdf, :rangeid)
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

        for (openthreshold, closethreshold) in GAIN_THRESHOLDS
            TradingStrategy.reset!(cfg.tradingstrategy)
            gdf = TradingStrategy.getgains(cfg.tradingstrategy, resultsview, scores, labels, true, openthreshold=openthreshold, closethreshold=closethreshold)
            if size(gdf, 1) > 0
                addgainadmin!(gdf, coin, sampleset, true, rng, openthreshold, closethreshold)
                push!(gainparts, gdf)
            end
        end

        TradingStrategy.reset!(cfg.tradingstrategy)
        true_open, true_close = TRUE_GAIN_THRESHOLD
        gdf = TradingStrategy.getgains(cfg.tradingstrategy, resultsview, truescores, targets, true, openthreshold=true_open, closethreshold=true_close)
        if size(gdf, 1) > 0
            addgainadmin!(gdf, coin, sampleset, false, rng, true_open, true_close)
            push!(gainparts, gdf)
        end
    end

    gaindf = isempty(gainparts) ? nothing : reduce(vcat, gainparts)
    if !isnothing(gaindf) && (size(gaindf, 1) > 0)
        gaindf = gaindf[.!ismissing.(gaindf[!, :set]), :] # exclude gaps between set partitions
        if size(gaindf, 1) > 0
            sort!(gaindf, [:coin, :predicted, :openthreshold, :closethreshold, :startdt, :trend])
            TradingStrategy.savetrades(gaindf; stem="gains")
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
    if isfreshcache(distancesfilename(), [gainsfilename()])
        distdf = EnvConfig.readdf(distancesfilename())
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
    gaindfgrp = groupby(gaindf1, [:coin, :predicted, :trend])
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

        haspredictions = false
        for trend in trendlevels
            cpgaindf = get(gaindfgrp, (coin, true, trend), DataFrame())   # predicted gains of one trend
            if size(cpgaindf, 1) == 0
                continue
            end
            haspredictions = true
            ctgaindf = get(gaindfgrp, (coin, false, trend), DataFrame())  # labeled gains of the same trend

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
        EnvConfig.savedf(distdf, distancesfilename())
    end
    (verbosity >= 2) && print("$(EnvConfig.now()) calculated distances for $(length(cfg.coins)) coins                             \r")
    (verbosity >= 3) && println()
    return distdf
end

function getconfusionmatrices(cfg::TrendDetectorConfig)
    xcmdf = DataFrame()
    cmdf = DataFrame()
    if EnvConfig.isfolder(EnvConfig.logpath(confusionfilename()))
        cmdf = EnvConfig.readdf(confusionfilename())
    end
    if EnvConfig.isfolder(EnvConfig.logpath(xconfusionfilename()))
        xcmdf = EnvConfig.readdf(xconfusionfilename())
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
            EnvConfig.savedf(cmdf, confusionfilename())
        end
        xcmdf = Classify.extendedconfusionmatrix(dfp, Targets.uniquelabels(cfg.targetconfig))
        if size(xcmdf, 1) > 0
            EnvConfig.savedf(xcmdf, xconfusionfilename())
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
                        [:truth_longbuy, :truth_longhold, :truth_allclose, :truth_shorthold, :truth_shortbuy] => ((lb, lh, ac, sh, sb) -> sum(lb) / (sum(lb) + sum(lh) + sum(sum(ac)) + sum(sh) + sum(sb)) * 100) => "longbuy_ppv%",
                        [:truth_longhold, :truth_longbuy, :truth_allclose, :truth_shorthold, :truth_shortbuy] => ((lh, lb, ac, sh, sb) -> sum(lh) / (sum(lh) + sum(lb) + sum(sum(ac)) + sum(sh) + sum(sb)) * 100) => "longhold_ppv%",
                        [:truth_allclose, :truth_longbuy, :truth_longhold, :truth_shorthold, :truth_shortbuy] => ((ac, lb, lh, sh, sb) -> sum(ac) / (sum(ac) + sum(lb) + sum(lh) + sum(sh) + sum(sb)) * 100) => "allclose_ppv%",
                        [:truth_shorthold, :truth_longbuy, :truth_longhold, :truth_allclose, :truth_shortbuy] => ((sh, lb, lh, ac, sb) -> sum(sh) / (sum(sh) + sum(lb) + sum(lh) + sum(ac) + sum(sb)) * 100) => "shorthold_ppv%",
                        [:truth_shortbuy, :truth_longbuy, :truth_longhold, :truth_allclose, :truth_shorthold] => ((sb, lb, lh, ac, sh) -> sum(sb) / (sum(sb) + sum(lb) + sum(lh) + sum(ac) + sum(sh)) * 100) => "shortbuy_ppv%")
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
    targetissuesdf = EnvConfig.readdf(targetissuesfilename())
    if isnothing(targetissuesdf) || size(targetissuesdf, 1) == 0
        println("No target issues file found")
    else
        println("size(targetissuesdf) = $(size(targetissuesdf))")
        println("describe(targetissuesdf, :all)=$(describe(targetissuesdf, :all))")
        println(targetissuesdf)
    end
    featuresdf = EnvConfig.readdf(featuresfilename())
    if isnothing(featuresdf) || size(featuresdf, 1) == 0
        println("No features file found in $(EnvConfig.logfolder()) - size(featuresdf)=$(isnothing(featuresdf) ? "nothing" : size(featuresdf))")
    else
        println("size(featuresdf) = $(size(featuresdf))")
        println("describe(featuresdf, :all)=$(describe(featuresdf, :all))")
    end
    resultsdf = EnvConfig.readdf(resultsfilename())
    if isnothing(resultsdf) || size(resultsdf, 1) == 0
        println("No results file found in $(EnvConfig.logfolder()) - size(resultsdf)=$(isnothing(resultsdf) ? "nothing" : size(resultsdf))")
    else
        println("size(resultsdf) = $(size(resultsdf))")
        println("describe(resultsdf, :all)=$(describe(resultsdf, :all))")
        println("$(unique(resultsdf[!, :coin])) processable coins")
        println("used targets: $(unique(resultsdf[!, :target]))")
        println("rangeid sorted = $(issorted(resultsdf[!, :rangeid]))")
        for coin in cfg.coins
            coin_results = @view resultsdf[resultsdf[!, :coin] .== coin, :]
            print("\rcoin=$coin, sopentime sorted = $(issorted(coin_results[!, :opentime])), rangeid sorted = $(issorted(coin_results[!, :rangeid]))")
        end
        # println()
        # # println("target distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(resultsdf[!, :targets]))))")
        # # println("set distribution: $(Distributions.fit(UnivariateFinite, categorical(string.(resultsdf[!, :set]))))")
        # gainsdf = getgainsdf(cfg)
        # if !isnothing(gainsdf) && (size(gainsdf, 1) > 0)
        #     println("describe(gainsdf, :all)=$(describe(gainsdf, :all))")
        #     for coin in cfg.coins
        #         coinview = @view gainsdf[gainsdf[!, :coin] .== coin, :]
        #         # println("coin=$coin, describe(coinview, :all)=$(describe(coinview, :all))")
        #         println("coinview[1:10, :]=$(coinview[1:10, :])")
        #     end
        # end
    end
    preddf = EnvConfig.readdf(predictionsfilename())
    if !isnothing(preddf) && (size(preddf, 1) > 0)
        println("$(predictionsfilename()): size(preddf) = $(size(preddf))")
        println("describe(preddf, :all)=$(describe(preddf, :all))")
    else
        println("No results file found in $(EnvConfig.logfolder()) - size(preddf)=$(isnothing(preddf) ? "nothing" : size(preddf))")
    end
end

# startdt = nothing  # means use all what is stored as canned data
# enddt = nothing  # means use all what is stored as canned data
startdt = DateTime("2017-11-17T20:56:00")
enddt = DateTime("2025-08-10T15:00:00")

println("$(EnvConfig.now()) $PROGRAM_FILE ARGS=$ARGS")
retrain = false
retrain = "retrain" in ARGS
retrain && println("retrain mode activated - existing classifiers that did not converge will be overwritten")
# retrain = true
testmode = true
testmode = "test" in ARGS ? true : "train" in ARGS ? false : testmode
inspectonly = "inspect" in ARGS
# inspectonly = true
specialonly = "special" in ARGS
# specialonly = true
inspectonly = specialonly ? true : inspectonly # if specialonly then also do inspection

verbosity = 2
allowedcoins = []
if testmode 
    verbosity = 2
    Ohlcv.verbosity = 1 # 3
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
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
    EnvConfig.init(training)
    allowedcoins = traincoins()
end

if specialonly
    Ohlcv.verbosity = 3
    Features.verbosity = 1
    Targets.verbosity = 1
    EnvConfig.verbosity = 1
    Classify.verbosity = 1
end

cfg = TrendDetectorConfig(;mk029config()..., coins=allowedcoins, startdt=startdt, enddt=enddt)

if specialonly
    # renamepredictionfiles([mk1config().folder, mk2config().folder, mk3config().folder, mk4config().folder, mk5config().folder])
    println("No special task defined")
elseif inspectonly
    introspection(cfg)
else
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


println("$(EnvConfig.now()) done @ $(cfg.folder)")
end # of TrendDetector
