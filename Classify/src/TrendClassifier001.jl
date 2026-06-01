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
    featconfig_factory::Function
    required_minutes::Int
    cfgid::Int
end

"""
    TrendClassifier001(nn::NN; featconfig_factory::Function, required_minutes::Integer, cfgid::Integer=1)

Construct a runtime classifier that creates one feature config per base via
`featconfig_factory` and uses `required_minutes` as warm-up horizon.
"""
function TrendClassifier001(nn::NN; featconfig_factory::Function, required_minutes::Integer, cfgid::Integer=1)
    return TrendClassifier001(Dict{AbstractString, NamedTuple}(), nn, featconfig_factory, Int(required_minutes), Int(cfgid))
end

"""
    addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData)

Attach a base OHLCV stream and initialize the feature configuration for cache-backed inference.
"""
function addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData)
    featcfg = cl.featconfig_factory()
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
    return TradeAdvice(cl, cl.cfgid, label, 1f0, base, price, dt, 0f0, Float32(scores[1]), investment)
end

"""
    loadclassifier(nn_fileprefix, featconfig_factory, required_minutes; search_folders, cfgid=1)

Search `search_folders` for a saved neural-network file identified by `nn_fileprefix`,
load it, and return a `TrendClassifier001` configured with `featconfig_factory` and `required_minutes`.

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
)::TrendClassifier001
    build = (nn, fcfg, reqmin, id) -> TrendClassifier001(nn; featconfig_factory=fcfg, required_minutes=reqmin, cfgid=id)
    return trend_runtime_loadclassifier(
        build,
        nn_fileprefix,
        featconfig_factory,
        required_minutes;
        search_folders=search_folders,
        cfgid=cfgid,
    )::TrendClassifier001
end

function load(::Type{TrendClassifier001}, spec::NamedTuple; mode=EnvConfig.configmode, folder::Union{Nothing, AbstractString}=nothing, cfgid::Integer=1)::TrendClassifier001
    hasproperty(spec, :nn_fileprefix) || error("missing nn_fileprefix in TrendClassifier001 load spec")
    hasproperty(spec, :featconfig_factory) || error("missing featconfig_factory in TrendClassifier001 load spec")
    hasproperty(spec, :required_minutes) || error("missing required_minutes in TrendClassifier001 load spec")

    target_folder = isnothing(folder) ? trend_runtime_folder_from_spec(spec, mode) : String(folder)
    return loadclassifier(
        String(getproperty(spec, :nn_fileprefix)),
        getproperty(spec, :featconfig_factory),
        Int(getproperty(spec, :required_minutes));
        search_folders=[target_folder],
        cfgid=cfgid,
    )
end

function save(cl::TrendClassifier001; mode=EnvConfig.configmode, folder::Union{Nothing, AbstractString}=nothing)
    isnothing(folder) && error("missing folder for TrendClassifier001 save; pass folder explicitly")
    target_folder = String(folder)
    EnvConfig.setlogpath(target_folder)
    savenn(cl.nn)
    return cl
end
