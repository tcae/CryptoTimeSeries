"""
    TrendClassifier001

Lightweight runtime classifier that wraps a trained neural network and a
feature-configuration factory. It is intended for simulation/inference scripts
that need `AbstractClassifier` compatibility without introducing a dedicated
training classifier type.
"""
mutable struct TrendClassifier001 <: AbstractClassifier
    bc::Dict{AbstractString, NamedTuple}
    nn::NN
    featconfig::Function
    targetconfig::Union{Nothing, Function}
    required_minutes::Int
    cfgid::Int
end

"""
    TrendClassifier001(nn::NN; featconfig::Function, targetconfig::Union{Nothing,Function}=nothing, required_minutes::Union{Nothing,Integer}=nothing, cfgid::Integer=1)

Construct a runtime classifier. When `required_minutes` is not provided, it is
derived from the feature configuration.
"""
function TrendClassifier001(
    nn::NN;
    featconfig::Function,
    targetconfig::Union{Nothing, Function}=nothing,
    required_minutes::Union{Nothing, Integer}=nothing,
    cfgid::Integer=1,
)
    reqmins = isnothing(required_minutes) ? max(Features.requiredminutes(featconfig()), 2) : Int(required_minutes)
    return TrendClassifier001(Dict{AbstractString, NamedTuple}(), nn, featconfig, targetconfig, reqmins, Int(cfgid))
end

"""
    TrendClassifier001(featurecount, labels, mnemonic; classifiermodel, featconfig, required_minutes, cfgid=1)

Construct a runtime classifier from a model factory signature
`classifiermodel(featurecount, labels, mnemonic)`.
"""
function TrendClassifier001(
    featurecount::Integer,
    labels,
    mnemonic::AbstractString;
    classifiermodel::Function,
    featconfig::Function,
    targetconfig::Union{Nothing, Function}=nothing,
    required_minutes::Integer,
    cfgid::Integer=1,
)
    nn = classifiermodel(featurecount, labels, String(mnemonic))
    return TrendClassifier001(Dict{AbstractString, NamedTuple}(), nn, featconfig, targetconfig, Int(required_minutes), Int(cfgid))
end

"""
    addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData; regressionwindows=Int[])

Attach a base OHLCV stream and initialize the feature configuration for cache-backed inference.
Register strategy-only regression dependencies before initialization. Reuse existing
stores and log missing history before calculating it in memory; never write runtime
regressions to stores, including for limited replay ranges.
"""
function addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData; regressionwindows::AbstractVector{<:Integer}=Int[])
    featcfg = cl.featconfig()
    for window in unique(regressionwindows)
        Features.require_regression!(featcfg; window=window)
    end
    if !isempty(regressionwindows)
        visible = Ohlcv.dataframe(ohlcv)
        source = parent(visible)
        @assert nrow(visible) > 0 && nrow(source) > 1 "strategy regressions for base=$(ohlcv.base) require a visible sample and at least two source samples; visible=$(nrow(visible)), source=$(nrow(source)), windows=$(regressionwindows)"
        firstrow = max(1, searchsortedfirst(source[!, :opentime], visible[1, :opentime]) - Features.requiredminutes(featcfg) + 1)
        for window in unique(regressionwindows)
            stored = Features.readregression(ohlcv.base, ohlcv.quotecoin, window)
            if isnothing(stored) || nrow(stored) == 0 || stored[1, :opentime] > source[firstrow, :opentime] || stored[end, :opentime] < visible[end, :opentime]
                @info "Calculating missing strategy regression in memory; initialization may take additional time (store remains unchanged)" base=ohlcv.base quotecoin=ohlcv.quotecoin window startdt=source[firstrow, :opentime] enddt=visible[end, :opentime]
            end
        end
    end
    Features.setbase!(featcfg, ohlcv, usecache=true)
    cl.bc[ohlcv.base] = (ohlcv=ohlcv, featcfg=featcfg)
    return cl
end

"""
    supplement!(cl::TrendClassifier001)

Advance all per-base feature configurations to match their OHLCV cursor.
Bases that fail (e.g. insufficient history for configured windows) are warned
and dropped so the trading loop continues with the remaining bases.
"""
function supplement!(cl::TrendClassifier001)
    failed_bases = String[]
    for (base, basecfg) in cl.bc
        try
            Features.supplement!(basecfg.featcfg)
        catch err
            push!(failed_bases, String(base))
            @warn "dropping base from runtime classifier due to supplement failure" base exception=(err, catch_backtrace())
        end
    end
    for base in failed_bases
        delete!(cl.bc, base)
    end
    return nothing
end

"""
    requiredminutes(cl::TrendClassifier001) -> Integer

Return the warm-up horizon required before advice can be produced.
"""
requiredminutes(cl::TrendClassifier001)::Integer = cl.required_minutes

"""
    featconfig(cl::TrendClassifier001, base::AbstractString)

Return the feature configuration attached to `base`, or `nothing` when the base was not
added. Gives strategies read access to the cached features (regressions, distances,
relative volume) that were already computed for inference.
"""
function featconfig(cl::TrendClassifier001, base::AbstractString)
    entry = get(cl.bc, uppercase(String(base)), nothing)
    return isnothing(entry) ? nothing : entry.featcfg
end

function _featurematrix(cl::TrendClassifier001, featuresdf::AbstractDataFrame)
    cols = Features.requestedcolumns(cl.featconfig())
    feats = Matrix(featuresdf[!, cols])
    return permutedims(feats, (2, 1))
end

"""
    maxpredictdf(cl::TrendClassifier001, featuresdf::AbstractDataFrame)

Run max prediction on a feature table matching `cl.featconfig` requested columns.
"""
function maxpredictdf(cl::TrendClassifier001, featuresdf::AbstractDataFrame)
    x = _featurematrix(cl, featuresdf)
    return maxpredictdf(cl.nn, x)
end

"""
    maxpredictdf(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData; startdt=nothing, enddt=nothing)

Derive features from an arbitrary OHLCV sequence and return score/label max predictions.
"""
function maxpredictdf(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData; startdt::Union{Nothing, DateTime}=nothing, enddt::Union{Nothing, DateTime}=nothing)
    featcfg = cl.featconfig()
    Features.setbase!(featcfg, ohlcv, usecache=true)
    fdf = if isnothing(startdt) || isnothing(enddt)
        Features.features(featcfg)
    else
        Features.features(featcfg, startdt, enddt)
    end
    return maxpredictdf(cl, fdf)
end

"""
    advice(cl::TrendClassifier001, base::AbstractString, dt::DateTime; investment=nothing)

Infer one-step trade advice at datetime `dt` for `base`.
"""
function advice(cl::TrendClassifier001, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    haskey(cl.bc, base) || return nothing
    basecfg = cl.bc[base]
    fdf = Features.features(basecfg.featcfg, dt, dt)
    (isnothing(fdf) || size(fdf, 1) == 0) && return nothing

    x = permutedims(Matrix(fdf), (2, 1))
    scores, labels = maxpredict(cl.nn, x)
    isempty(labels) && return nothing

    label = labels[1] isa Targets.TradeLabel ? labels[1] : Targets.tradelabel(string(labels[1]))
    oix = Ohlcv.rowix(basecfg.ohlcv, dt)
    price = Ohlcv.dataframe(basecfg.ohlcv)[oix, :pivot]
    return TradeAdvice(cl, cl.cfgid, label, 1f0, base, price, dt, 0f0, (scores[1]), investment)
end

Classify.isadapted(cl::TrendClassifier001)::Bool = Classify.isadapted(cl.nn)
Classify.nnconverged(cl::TrendClassifier001)::Bool = Classify.nnconverged(cl.nn)
Classify.model(cl::TrendClassifier001) = cl.nn
