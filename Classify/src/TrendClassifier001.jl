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
    addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData)

Attach a base OHLCV stream and initialize the feature configuration for cache-backed inference.
"""
function addbase!(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData)
    featcfg = cl.featconfig()
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

function _required_minutes_from_spec(spec::NamedTuple)::Int
    if hasproperty(spec, :required_minutes)
        return Int(getproperty(spec, :required_minutes))
    end
    hasproperty(spec, :featconfig) || error("missing required_minutes in TrendClassifier001 spec and featconfig unavailable to derive it")
    return max(Features.requiredminutes(getproperty(spec, :featconfig)()), 2)
end

function _targetconfig_from_spec(spec::NamedTuple)
    return hasproperty(spec, :targetconfig) ? getproperty(spec, :targetconfig) : nothing
end

function _featurematrix(cl::TrendClassifier001, featuresdf::AbstractDataFrame)
    cols = Features.requestedcolumns(cl.featconfig())
    feats = Matrix(featuresdf[!, cols])
    return permutedims(feats, (2, 1))
end

function _resolve_opentime_col(df::AbstractDataFrame)::Union{Symbol, Nothing}
    for col in (:opentime, :open_time, :timestamp, :time, :datetime)
        if col in propertynames(df)
            return col
        end
    end
    return nothing
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
    featurestargetsdf(cl::TrendClassifier001, ohlcv::Ohlcv.OhlcvData, targetconfig::Targets.AbstractTargets; startdt=nothing, enddt=nothing, partitionconfig=nothing)

Create aligned feature and target tables for one OHLCV sequence.
When `partitionconfig` is provided (NamedTuple from TrendDetector), rows are split
into train/eval/test sets using `setpartitions`.
"""
function featurestargetsdf(
    cl::TrendClassifier001,
    ohlcv::Ohlcv.OhlcvData,
    targetconfig::Targets.AbstractTargets;
    startdt::Union{Nothing, DateTime}=nothing,
    enddt::Union{Nothing, DateTime}=nothing,
    partitionconfig=nothing,
    coin::AbstractString=ohlcv.base,
    rangeid_start::UInt16=UInt16(1),
)
    featcfg = cl.featconfig()
    Features.setbase!(featcfg, ohlcv, usecache=true)
    fdf = if isnothing(startdt) || isnothing(enddt)
        Features.features(featcfg)
    else
        Features.features(featcfg, startdt, enddt)
    end

    if targetconfig isa Targets.TrendRegression
        if Features.issupplementedcurrent(featcfg)
            Targets.setbase!(targetconfig, featcfg)
        else
            error("features not supplemented current for target calculation")
        end
    else
        Targets.setbase!(targetconfig, ohlcv)
    end

    fotcol = _resolve_opentime_col(fdf)
    if isnothing(fotcol)
        fot = Features.opentime(featcfg)
        @assert size(fdf, 1) == length(fot) "feature rows $(size(fdf, 1)) must match feature opentime length $(length(fot)) for $(ohlcv.base)"
        firstdt = fot[begin]
        lastdt = fot[end]
    else
        firstdt = fdf[begin, fotcol]
        lastdt = fdf[end, fotcol]
    end
    targets = Targets.labels(targetconfig, firstdt, lastdt)
    @assert size(fdf, 1) == length(targets) "size(features)=$(size(fdf, 1)) must equal targets=$(length(targets))"

    rdf = DataFrame(target=targets)
    odf = Ohlcv.dataframe(ohlcv)
    ootcol = _resolve_opentime_col(odf)
    @assert !isnothing(ootcol) "OHLCV dataframe must contain :opentime-compatible timestamp column for $(ohlcv.base); available columns=$(propertynames(odf))"
    pricedf = select(odf, [ootcol, :high, :low, :close, :pivot])
    if ootcol != :opentime
        rename!(pricedf, ootcol => :opentime)
    end
    pfirst = findfirst(==(firstdt), pricedf[!, :opentime])
    plast = findfirst(==(lastdt), pricedf[!, :opentime])
    @assert !isnothing(pfirst) && !isnothing(plast) "failed to align feature date range with OHLCV for $(ohlcv.base)"
    pricedf = @view pricedf[pfirst:plast, :]
    @assert size(pricedf, 1) == size(rdf, 1) "price rows $(size(pricedf, 1)) must match targets $(size(rdf, 1))"
    rdf = hcat(rdf, pricedf)

    rdf[:, :coin] = CategoricalVector(fill(String(coin), size(rdf, 1)), levels=[String(coin)])
    rdf[:, :rangeid] = fill(Int(rangeid_start), size(rdf, 1))

    if !isnothing(partitionconfig)
        samplesets = partitionconfig.samplesets
        levels = unique(samplesets)
        rdf[:, :set] = CategoricalVector(fill(string(samplesets[1]), size(rdf, 1)), levels=string.(levels))
        allowmissing!(rdf, :set)
        rdf[:, :set] .= missing

        psets = setpartitions(
            1:size(rdf, 1),
            samplesets,
            partitionsize=partitionconfig.partitionsize,
            gapsize=partitionconfig.gapsize,
            minpartitionsize=partitionconfig.minpartitionsize,
            maxpartitionsize=partitionconfig.maxpartitionsize,
        )
        rid = Int(rangeid_start)
        for (pssettype, psrng) in psets
            rdf[psrng, :rangeid] .= rid
            rdf[psrng, :set] .= pssettype
            rid += 1
        end
        mask = .!ismissing.(rdf[!, :set])
        rdf = rdf[mask, :]
        fdf = fdf[mask, :]
    else
        rdf[:, :set] = CategoricalVector(fill("all", size(rdf, 1)), levels=["all"])
    end

    return rdf, DataFrame(fdf; copycols=true)
end

"""
    adapt!(cl::TrendClassifier001, resultsdf::AbstractDataFrame, featuresdf::AbstractDataFrame; settype="train", classbalancing=true, retrain=false, save_after=false, mode=EnvConfig.configmode, folder=nothing)

Adapt or retrain the underlying NN using provided feature/target tables.
"""
function adapt!(
    cl::TrendClassifier001,
    resultsdf::AbstractDataFrame,
    featuresdf::AbstractDataFrame;
    settype::AbstractString="train",
    classbalancing::Bool=true,
    retrain::Bool=false,
    save_after::Bool=false,
    mode=EnvConfig.configmode,
    folder::Union{Nothing, AbstractString}=nothing,
)
    if !retrain && isadapted(cl.nn) && nnconverged(cl.nn)
        return cl
    end

    mask = string.(resultsdf[!, :set]) .== String(settype)
    tresults = @view resultsdf[mask, :]
    tfeatures = @view featuresdf[mask, :]
    size(tresults, 1) == 0 && return cl

    x = _featurematrix(cl, tfeatures)
    y = tresults[!, :target]

    sampleweights = nothing
    if classbalancing
        weightinfo = classweighting(y, cl.nn.labels)
        sampleweights = weightinfo.sampleweights
    end
    adaptnn!(cl.nn, x, y; sampleweights=sampleweights)

    if save_after
        save(cl; mode=mode, folder=folder)
    end
    return cl
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

"""
    loadclassifier(nn_fileprefix, featconfig, required_minutes; search_folders, cfgid=1)

Search `search_folders` for a saved neural-network file identified by `nn_fileprefix`,
load it, and return a `TrendClassifier001` configured with `featconfig` and `required_minutes`.

`search_folders` is iterated in order; the first folder that contains the NN file wins.
`EnvConfig.setlogpath` is called temporarily for each folder to resolve the file path.
Throws an error if the file is not found or cannot be loaded.
"""
function loadclassifier(
    nn_fileprefix::AbstractString,
    featconfig::Function,
    required_minutes::Integer;
    search_folders::AbstractVector{<:AbstractString},
    cfgid::Integer=1,
)::TrendClassifier001
    build = (nn, fcfg, reqmin, id) -> TrendClassifier001(nn; featconfig=fcfg, required_minutes=reqmin, cfgid=id)
    return runtime_loadclassifier(
        build,
        nn_fileprefix,
        featconfig,
        required_minutes,
        cfgid;
        search_folders=search_folders,
    )::TrendClassifier001
end

function load(::Type{TrendClassifier001}, spec::NamedTuple; mode=EnvConfig.configmode, folder::Union{Nothing, AbstractString}=nothing, cfgid::Integer=1)::TrendClassifier001
    hasproperty(spec, :nn_fileprefix) || error("missing nn_fileprefix in TrendClassifier001 load spec")
    hasproperty(spec, :featconfig) || error("missing featconfig in TrendClassifier001 load spec")

    target_folder = isnothing(folder) ? trend_runtime_folder_from_spec(spec, mode) : String(folder)
    required_minutes = _required_minutes_from_spec(spec)
    return loadclassifier(
        String(getproperty(spec, :nn_fileprefix)),
        getproperty(spec, :featconfig),
        required_minutes;
        search_folders=[target_folder],
        cfgid=cfgid,
    )
end

"""
    loadorbuild(::Type{TrendClassifier001}, spec, featurecount, labels, mnemonic, classifiermodel; mode, folder, cfgid)

Load a persisted runtime classifier when present, otherwise build a fresh instance
using `classifiermodel(featurecount, labels, mnemonic)`.
"""
function loadorbuild(
    ::Type{TrendClassifier001},
    spec::NamedTuple,
    featurecount::Integer,
    labels,
    mnemonic::AbstractString,
    classifiermodel::Function;
    mode=EnvConfig.configmode,
    folder::Union{Nothing, AbstractString}=nothing,
    cfgid::Integer=1,
)::TrendClassifier001
    hasproperty(spec, :featconfig) || error("missing featconfig in TrendClassifier001 spec")

    target_folder = isnothing(folder) ? trend_runtime_folder_from_spec(spec, mode) : String(folder)
    required_minutes = _required_minutes_from_spec(spec)
    targetconfig = _targetconfig_from_spec(spec)

    nntmp = classifiermodel(featurecount, labels, String(mnemonic))
    EnvConfig.setlogpath(target_folder)
    if isfile(nnfilename(nntmp.fileprefix))
        loadspec = merge(spec, (nn_fileprefix=nntmp.fileprefix,))
        try
            return load(TrendClassifier001, loadspec; mode=mode, folder=target_folder, cfgid=cfgid)
        catch err
            # Legacy BSON artifacts may fail to deserialize after package/model schema changes.
            # Fall back to building a fresh runtime classifier so callers can retrain and resave.
            @warn "failed to load persisted TrendClassifier001 artifact; rebuilding classifier" folder=target_folder fileprefix=nntmp.fileprefix exception=(err, catch_backtrace())
        end
    end

    return TrendClassifier001(
        featurecount,
        labels,
        String(mnemonic);
        classifiermodel=classifiermodel,
        featconfig=getproperty(spec, :featconfig),
        targetconfig=targetconfig,
        required_minutes=required_minutes,
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

Classify.isadapted(cl::TrendClassifier001)::Bool = Classify.isadapted(cl.nn)
Classify.nnconverged(cl::TrendClassifier001)::Bool = Classify.nnconverged(cl.nn)
Classify.model(cl::TrendClassifier001) = cl.nn
