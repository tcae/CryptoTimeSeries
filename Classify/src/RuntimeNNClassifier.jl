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
Bases that fail (e.g. insufficient history for configured windows) are warned
and dropped so the trading loop continues with the remaining bases.
"""
function supplement!(cl::RuntimeNNClassifier)
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

"""
    loadclassifier(nn_fileprefix, featconfig_factory, required_minutes; search_folders, cfgid=1)

Search `search_folders` for a saved neural-network file identified by `nn_fileprefix`,
load it, and return a `RuntimeNNClassifier` configured with `featconfig_factory` and `required_minutes`.

`search_folders` is iterated in order; the first folder that contains the NN file wins.
`EnvConfig.setlogpath` is called temporarily for each folder to resolve the file path.
Throws an error if the file is not found or cannot be loaded.
"""
function loadclassifier(
    nn_fileprefix::AbstractString,
    featconfig_factory::Function,
    required_minutes::Integer;
    search_folders::AbstractVector{<:AbstractString},
    cfgid::Integer=1,
)::RuntimeNNClassifier
    for folder in unique(String.(search_folders))
        EnvConfig.setlogpath(folder)
        nnpath = nnfilename(nn_fileprefix)
        if isfile(nnpath)
            try
                nn = loadnn(nn_fileprefix)
                return RuntimeNNClassifier(nn; featconfig_factory=featconfig_factory, required_minutes=required_minutes, cfgid=cfgid)
            catch err
                shorterr = sprint(showerror, err)
                error("classifier file found but could not be loaded: nnpath=$nnpath. Cause=$shorterr. Likely artifact compatibility mismatch (Flux/Optimisers/BSON versions).")
            end
        end
    end
    error("classifier file not found for fileprefix=$nn_fileprefix, checked folders=$(collect(search_folders))")
end