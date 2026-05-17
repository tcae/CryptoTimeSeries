"""
    RuntimeNNClassifier

Lightweight runtime classifier that wraps a trained neural network and a
feature-configuration factory. It is intended for simulation/inference scripts
that need `AbstractClassifier` compatibility without introducing a dedicated
training classifier type.
"""
mutable struct RuntimeNNClassifier <: AbstractClassifier
    bc::Dict{AbstractString, NamedTuple}
    nn::NN
    featconfig_factory::Function
    required_minutes::Int
    cfgid::Int
end

"""
    RuntimeNNClassifier(nn::NN; featconfig_factory::Function, required_minutes::Integer, cfgid::Integer=1)

Construct a runtime classifier that creates one feature config per base via
`featconfig_factory` and uses `required_minutes` as warm-up horizon.
"""
function RuntimeNNClassifier(nn::NN; featconfig_factory::Function, required_minutes::Integer, cfgid::Integer=1)
    return RuntimeNNClassifier(Dict{AbstractString, NamedTuple}(), nn, featconfig_factory, Int(required_minutes), Int(cfgid))
end

"""
    addbase!(cl::RuntimeNNClassifier, ohlcv::Ohlcv.OhlcvData)

Attach a base OHLCV stream and initialize the feature configuration for cache-backed inference.
"""
function addbase!(cl::RuntimeNNClassifier, ohlcv::Ohlcv.OhlcvData)
    featcfg = cl.featconfig_factory()
    Features.setbase!(featcfg, ohlcv, usecache=true)
    cl.bc[ohlcv.base] = (ohlcv=ohlcv, featcfg=featcfg)
    return cl
end

"""
    supplement!(cl::RuntimeNNClassifier)

Advance all per-base feature configurations to match their OHLCV cursor.
"""
function supplement!(cl::RuntimeNNClassifier)
    for basecfg in values(cl.bc)
        Features.supplement!(basecfg.featcfg)
    end
end

"""
    requiredminutes(cl::RuntimeNNClassifier) -> Integer

Return the warm-up horizon required before advice can be produced.
"""
requiredminutes(cl::RuntimeNNClassifier)::Integer = cl.required_minutes

"""
    advice(cl::RuntimeNNClassifier, base::AbstractString, dt::DateTime; investment=nothing)

Infer one-step trade advice at datetime `dt` for `base`.
"""
function advice(cl::RuntimeNNClassifier, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
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
    return TradeAdvice(cl, cl.cfgid, label, 1f0, base, price, dt, 0f0, Float32(scores[1]), investment)
end