#region ClassifierStore

# Artifacts of a named classifier in `EnvConfig.classifierfolder(classifiername)`:
# - `nn.bson`, `nn-lastepoch.bson`: adapted NN (best loss) and last epoch checkpoint
# - `meta.arrow`: key/value table with component ids, hashes, labels, feature columns and coins
# - `trainindex.arrow`: training samples as references into the per-coin feature stores
# - `trainfeatures.f32`: compact features × samples Float32 matrix of the training index, memory mapped during adaptation
# - `predictions/<coin>.arrow`: opentime, score, label (Int8 TradeLabel code) over the feature store history

const NN_FILEPREFIX = "nn"
const TRAININDEX_STEM = "trainindex"
const META_STEM = "meta"

"Return (and create) the per-coin predictions folder of the classifier `classifiername`."
function predictionsfolder(classifiername::AbstractString)::String
    path = joinpath(EnvConfig.classifierfolder(classifiername), "predictions")
    isdir(path) || mkpath(path)
    return path
end

"Return whether the adapted NN of the classifier `classifiername` is stored."
isclassifierstored(classifiername::AbstractString)::Bool = isfile(nnfilename(NN_FILEPREFIX; folderpath=EnvConfig.classifierfolder(classifiername)))

"Persist the key/value meta data of the classifier `classifiername`."
function writemeta(classifiername::AbstractString, meta::AbstractDict{String, String})
    keyvec = sort!(collect(keys(meta)))
    EnvConfig.savedf(DataFrame(key=keyvec, value=[meta[key] for key in keyvec]), META_STEM; folderpath=EnvConfig.classifierfolder(classifiername), format=:arrow)
    return meta
end

"Return the key/value meta data of the classifier `classifiername` or `nothing` if not stored."
function readmeta(classifiername::AbstractString)::Union{Nothing, Dict{String, String}}
    df = EnvConfig.readdf(META_STEM; folderpath=EnvConfig.classifierfolder(classifiername), format=:arrow)
    return isnothing(df) ? nothing : Dict{String, String}(String(k) => String(v) for (k, v) in zip(df[!, :key], df[!, :value]))
end

"""
    newclassifier(classifiername, classifiermodel, featconfig, labels)

Return an untrained `TrendClassifier001` built with `classifiermodel(featurecount, labels, mnemonic)`
whose NN checkpoints under the classifier artifact name.
"""
function newclassifier(classifiername::AbstractString, classifiermodel::Function, featconfig::Function, labels::AbstractVector{Targets.TradeLabel})::TrendClassifier001
    f6 = featconfig()
    cl = TrendClassifier001(Features.featurecount(f6), labels, String(classifiername);
        classifiermodel=classifiermodel, featconfig=featconfig, required_minutes=max(Features.requiredminutes(f6), 2))
    cl.nn.fileprefix = NN_FILEPREFIX
    return cl
end

"""
    loadclassifier(classifiername; featconfig)

Load the adapted `TrendClassifier001` named `classifiername` from its classifier folder.
"""
function loadclassifier(classifiername::AbstractString; featconfig::Function)::TrendClassifier001
    folder = EnvConfig.classifierfolder(classifiername)
    nnpath = nnfilename(NN_FILEPREFIX; folderpath=folder)
    @assert isfile(nnpath) "classifier $(classifiername) is not adapted: missing $(nnpath)"
    nn = try
        loadnn(NN_FILEPREFIX; folderpath=folder)
    catch err
        error("classifier file found but could not be loaded: nnpath=$(nnpath). Cause=$(sprint(showerror, err)). Likely artifact compatibility mismatch (Flux/Optimisers/BSON versions).")
    end
    return TrendClassifier001(nn; featconfig=featconfig)
end

"Append the rows of one coin that belong to `settype`, have complete features and a valid target to the index vectors."
function _appendindexrows!(coinixs, rows, labelixs, coinix::Integer, fot::AbstractVector, tot::AbstractVector, tcodes::AbstractVector{<:Integer}, sot::AbstractVector, sset::AbstractVector, settype::AbstractString, codemap::Dict{Int8, UInt8})
    ignorecode = Targets.tradelabelcode(Targets.ignore)
    for six in eachindex(sot)
        sset[six] == settype || continue
        dt = sot[six]
        fix = searchsortedfirst(fot, dt)
        # rows before the feature warm-up end are not part of the feature store
        ((fix <= lastindex(fot)) && (fot[fix] == dt)) || continue
        tix = searchsortedfirst(tot, dt)
        @assert (tix <= lastindex(tot)) && (tot[tix] == dt) "target store misses opentime $(dt) of the sets store for coin index $(coinix)"
        code = tcodes[tix]
        code == ignorecode && continue
        @assert haskey(codemap, code) "target label code $(code) at $(dt) of coin index $(coinix) is not part of the classifier labels $(codemap)"
        push!(coinixs, coinix)
        push!(rows, fix)
        push!(labelixs, codemap[code])
    end
    return nothing
end

"""
    buildtrainindex(coins, loadcoin, labels; settype="train", classbalancing)

Return the training index DataFrame with columns `coinix, row, labelix, weight` of all samples
of `settype` over `coins`. `loadcoin(coin)` returns the `(features, targets, sets)` stores of `coin`.
`labelix` is the position of the target label in `labels`; with `classbalancing` the weights are
inverse class frequencies, otherwise 1.
"""
function buildtrainindex(coins::AbstractVector{<:AbstractString}, loadcoin::Function, labels::AbstractVector{Targets.TradeLabel}; settype::AbstractString="train", classbalancing::Bool)::DataFrame
    @assert length(coins) <= typemax(UInt16) "$(length(coins)) coins exceed the UInt16 coin index"
    codemap = Dict{Int8, UInt8}(Targets.tradelabelcode(label) => UInt8(ix) for (ix, label) in enumerate(labels))
    coinixs = UInt16[]
    rows = UInt32[]
    labelixs = UInt8[]
    for (coinix, coin) in enumerate(coins)
        fdf, tdf, sdf = loadcoin(coin)
        @assert !isnothing(fdf) && !isnothing(tdf) && !isnothing(sdf) "missing store of $(coin): features=$(!isnothing(fdf)) targets=$(!isnothing(tdf)) sets=$(!isnothing(sdf))"
        _appendindexrows!(coinixs, rows, labelixs, UInt16(coinix), fdf[!, :opentime], tdf[!, :opentime], tdf[!, :label], sdf[!, :opentime], sdf[!, :set], settype, codemap)
        (verbosity >= 2) && print("\r$(EnvConfig.now()) indexed $(coin) ($(coinix)/$(length(coins))) total samples=$(length(rows))                ")
    end
    (verbosity >= 2) && println()
    counts = [count(==(UInt8(ix)), labelixs) for ix in eachindex(labels)]
    (verbosity >= 2) && println("$(EnvConfig.now()) training index class counts $(Dict(zip(string.(labels), counts)))")
    cw = classbalancing ? classweights(counts) : ones(Float32, length(labels))
    return DataFrame(coinix=coinixs, row=rows, labelix=labelixs, weight=cw[labelixs])
end

"Persist the training index of the classifier `classifiername`."
function savetrainindex(classifiername::AbstractString, indexdf::AbstractDataFrame)
    EnvConfig.savedf(indexdf, TRAININDEX_STEM; folderpath=EnvConfig.classifierfolder(classifiername), format=:arrow)
end

"Return the (memory mapped) training index of the classifier `classifiername` or `nothing` if not stored."
readtrainindex(classifiername::AbstractString) = EnvConfig.readdf(TRAININDEX_STEM; folderpath=EnvConfig.classifierfolder(classifiername), format=:arrow)

"""
    predictfeatures(nn, fdf, featurecols; chunksize=8192)

Return a DataFrame with `opentime, score, label` (Int8 TradeLabel code) of the max prediction of
`nn` for every row of the feature store `fdf`, evaluated chunk wise to bound memory.
"""
function predictfeatures(nn::NN, fdf::AbstractDataFrame, featurecols::AbstractVector{<:AbstractString}; chunksize::Integer=8192)::DataFrame
    n = size(fdf, 1)
    columns = [fdf[!, col] for col in featurecols]
    scores = Vector{Float32}(undef, n)
    codes = Vector{Int8}(undef, n)
    Flux.testmode!(nn.model)
    for chunk in Iterators.partition(1:n, chunksize)
        x = Matrix{Float32}(undef, length(columns), length(chunk))
        for (sampleix, row) in enumerate(chunk), featureix in eachindex(columns)
            x[featureix, sampleix] = columns[featureix][row]
        end
        chunkscores, chunklabels = maxpredict(nn, x)
        scores[chunk] .= chunkscores
        codes[chunk] .= Targets.tradelabelcode.(chunklabels)
    end
    return DataFrame(opentime=copy(fdf[!, :opentime]), score=scores, label=codes)
end

"Persist the predictions of `coin` of the classifier `classifiername`."
savepredictions(classifiername::AbstractString, coin::AbstractString, preddf::AbstractDataFrame) = EnvConfig.savedf(preddf, String(coin); folderpath=predictionsfolder(classifiername), format=:arrow)

"Return the (memory mapped) predictions of `coin` of the classifier `classifiername` or `nothing` if not stored."
readpredictions(classifiername::AbstractString, coin::AbstractString) = EnvConfig.readdf(String(coin); folderpath=predictionsfolder(classifiername), format=:arrow)

"""
    evaluationdf(preddf, tdf, sdf, setlevels)

Return the rows of the sets store `sdf` joined with predictions and targets as DataFrame with
`opentime, set, rangeid, score, label, target` (TradeLabel) skipping rows without prediction or
with incomplete (`ignore`) targets.
"""
function evaluationdf(preddf::AbstractDataFrame, tdf::AbstractDataFrame, sdf::AbstractDataFrame, setlevels::AbstractVector{<:AbstractString})::DataFrame
    pot = preddf[!, :opentime]
    tot = tdf[!, :opentime]
    ignorecode = Targets.tradelabelcode(Targets.ignore)
    pix = Int[]
    tix = Int[]
    six = Int[]
    for (ix, dt) in enumerate(sdf[!, :opentime])
        pi = searchsortedfirst(pot, dt)
        ((pi <= lastindex(pot)) && (pot[pi] == dt)) || continue
        ti = searchsortedfirst(tot, dt)
        @assert (ti <= lastindex(tot)) && (tot[ti] == dt) "target store misses opentime $(dt) of the sets store"
        tdf[ti, :label] == ignorecode && continue
        push!(pix, pi)
        push!(tix, ti)
        push!(six, ix)
    end
    return DataFrame(
        opentime=sdf[six, :opentime],
        set=categorical(String.(sdf[six, :set]); levels=collect(setlevels)),
        rangeid=sdf[six, :rangeid],
        score=preddf[pix, :score],
        label=Targets.TradeLabel.(preddf[pix, :label]),
        target=Targets.TradeLabel.(tdf[tix, :label]),
    )
end

"Aggregate per-coin `confusionmatrix` results by summing counts and recalculating the ppv."
function aggregateconfusion(parts::AbstractVector{<:AbstractDataFrame})::DataFrame
    isempty(parts) && return DataFrame()
    df = reduce(vcat, parts; cols=:union)
    countcols = [col for col in names(df) if startswith(col, "truth_") || (col in ("allpredicted", "truepositive"))]
    for col in countcols
        df[!, col] = coalesce.(df[!, col], 0)
    end
    cdf = combine(groupby(df, [:set, :prediction]), countcols .=> sum .=> countcols)
    cdf[!, "ppv%"] = Float32[ap > 0 ? round(tp / ap * 100; digits=0) : 0f0 for (tp, ap) in zip(cdf[!, :truepositive], cdf[!, :allpredicted])]
    return sort!(cdf, [order(:set, rev=true), :prediction])
end

"Aggregate per-coin `extendedconfusionmatrix` results by summing counts and recalculating the derived rates."
function aggregateextendedconfusion(parts::AbstractVector{<:AbstractDataFrame})::DataFrame
    isempty(parts) && return DataFrame()
    df = reduce(vcat, parts)
    df[!, :set] = string.(df[!, :set])
    df[!, :bin] = string.(df[!, :bin])
    cdf = combine(groupby(df, [:set, :pred_label, :bin]), [:tp, :tn, :fp, :fn] .=> sum .=> [:tp, :tn, :fp, :fn])
    allvec = cdf.tp + cdf.tn + cdf.fp + cdf.fn
    cdf[!, "tp%"] = round.(cdf.tp ./ allvec .* 100.0; digits=0)
    cdf[!, "tn%"] = round.(cdf.tn ./ allvec .* 100.0; digits=0)
    cdf[!, "fp%"] = round.(cdf.fp ./ allvec .* 100.0; digits=0)
    cdf[!, "fn%"] = round.(cdf.fn ./ allvec .* 100.0; digits=0)
    cdf[!, "tpr"] = round.(cdf.tp ./ (cdf.tp + cdf.fn); digits=2)
    cdf[!, "fpr"] = round.(cdf.fp ./ (cdf.fp + cdf.tn); digits=2)
    cdf[!, "ppv%"] = round.(cdf.tp ./ (cdf.tp + cdf.fp) .* 100.0; digits=0)
    cdf[!, "npv%"] = round.(cdf.tn ./ (cdf.tn + cdf.fn) .* 100.0; digits=0)
    return cdf
end

#endregion ClassifierStore
