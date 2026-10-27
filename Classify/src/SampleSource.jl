#region SampleSource

"""
A sample source provides adaptation batches to `adaptnn!` independent of where the samples are stored.

Implementations provide
- `nsamples(src)`: number of samples
- `batch(src, ixs)`: `(x, y, w)` with `x` features × samples Float32 matrix, `y` targets matrix and `w` sample weights or `nothing`
- `classindex(src, ixs)`: true class index per sample (classification sources only)
"""
abstract type AbstractSampleSource end

"In-memory sample source with features × samples matrix `X`, target matrix `Y` (one-hot or regression targets) and optional sample weights `W`."
struct MatrixSampleSource{XM<:AbstractMatrix{Float32}, YM<:AbstractMatrix, WV<:Union{Nothing, AbstractVector{Float32}}} <: AbstractSampleSource
    X::XM
    Y::YM
    W::WV
    function MatrixSampleSource(X::AbstractMatrix{Float32}, Y::AbstractMatrix, W::Union{Nothing, AbstractVector{Float32}}=nothing)
        @assert size(X, 2) == size(Y, 2) > 0 "size(X, 2)=$(size(X, 2)) must equal size(Y, 2)=$(size(Y, 2)) > 0"
        @assert isnothing(W) || (length(W) == size(X, 2)) "length(W)=$(isnothing(W) ? 0 : length(W)) must equal size(X, 2)=$(size(X, 2))"
        return new{typeof(X), typeof(Y), typeof(W)}(X, Y, W)
    end
end

"Return a classification `MatrixSampleSource` with one-hot targets of `targets` in the label order of `nn.labels`."
function MatrixSampleSource(nn::NN, X::AbstractMatrix{Float32}, targets::AbstractVector, W::Union{Nothing, AbstractVector{Float32}}=nothing)
    return MatrixSampleSource(X, Flux.onehotbatch(targets, nn.labels), W)
end

nsamples(src::MatrixSampleSource) = size(src.X, 2)
batch(src::MatrixSampleSource, ixs::AbstractVector{<:Integer}) = (src.X[:, ixs], src.Y[:, ixs], isnothing(src.W) ? nothing : src.W[ixs])
classindex(src::MatrixSampleSource, ixs::AbstractVector{<:Integer}) = [argmax(view(src.Y, :, ix)) for ix in ixs]

"""
Memory mapped sample source of a classifier training index.

Each sample references row `row[i]` of the per-coin feature store `features[coinix[i]]` and carries its
class index `labelix[i]` (position in the classifier label order). Class weights are calculated once
from the per-class sample counts.
"""
struct ArrowIndexSource{V<:AbstractVector{Float32}, C<:AbstractVector{<:Integer}, R<:AbstractVector{<:Integer}, L<:AbstractVector{<:Integer}} <: AbstractSampleSource
    features::Vector{Vector{V}}  # [coinix][featureix] memory mapped feature store column
    coinix::C
    row::R
    labelix::L
    classweight::Vector{Float32}
    nlabels::Int
    useweights::Bool
end

"""
    ArrowIndexSource(featurestores, featurecols, index; nlabels, useweights)

`featurestores` is a vector of per-coin feature store tables in coin index order, `featurecols`
the feature column names in classifier input order and `index` the training index table with
columns `coinix, row, labelix`.
"""
function ArrowIndexSource(featurestores::AbstractVector, featurecols::AbstractVector{<:AbstractString}, index; nlabels::Integer, useweights::Bool)
    @assert !isempty(featurestores) "ArrowIndexSource requires at least one feature store"
    @assert 0 < nlabels <= typemax(UInt8) "nlabels=$(nlabels) must be within 1:$(typemax(UInt8))"
    columns = [[store[!, col] for col in featurecols] for store in featurestores]
    V = typeof(columns[1][1])
    @assert all(col -> col isa V, Iterators.flatten(columns)) "all feature store columns must share type $(V); found $(unique(typeof.(Iterators.flatten(columns))))"
    @assert eltype(V) == Float32 "feature store columns must be Float32 but are $(eltype(V))"
    features = Vector{Vector{V}}([Vector{V}(cols) for cols in columns])
    labelix = index[!, :labelix]
    counts = [count(==(UInt8(ix)), labelix) for ix in 1:nlabels]
    src = ArrowIndexSource(features, index[!, :coinix], index[!, :row], labelix, classweights(counts), Int(nlabels), useweights)
    @assert nsamples(src) > 0 "ArrowIndexSource requires a non-empty training index"
    return src
end

nsamples(src::ArrowIndexSource) = length(src.coinix)

function batch(src::ArrowIndexSource, ixs::AbstractVector{<:Integer})
    sortedixs = sort(ixs; by=ix -> (src.coinix[ix], src.row[ix]))
    nfeatures = length(src.features[1])
    x = Matrix{Float32}(undef, nfeatures, length(sortedixs))
    for (sampleix, ix) in enumerate(sortedixs)
        columns = src.features[src.coinix[ix]]
        row = src.row[ix]
        for featureix in 1:nfeatures
            x[featureix, sampleix] = columns[featureix][row]
        end
    end
    y = Flux.onehotbatch(src.labelix[sortedixs], 1:src.nlabels)
    w = src.useweights ? src.classweight[src.labelix[sortedixs]] : nothing
    return x, y, w
end

classindex(src::ArrowIndexSource, ixs::AbstractVector{<:Integer}) = src.labelix[ixs]
batchindices(src::ArrowIndexSource, ixs::AbstractVector{<:Integer}) = sort(ixs; by=ix -> (src.coinix[ix], src.row[ix]))

"File name of the compact training feature matrix within a classifier folder."
const TRAINFEATURES_FILE = "trainfeatures.f32"
const TRAINLABELS_FILE = "trainlabels.u8"

"""
Memory-mapped sample source backed by one compact features × samples Float32 matrix file.

The matrix holds only the samples referenced by the training index, in index order, so all
features of a sample are contiguous and a randomly picked sample touches one memory page.
The file is headerless: its dimensions follow from the feature count and the index length.
"""
struct CompactMmapSource{L<:AbstractVector{UInt8}} <: AbstractSampleSource
    X::Matrix{Float32}  # features × samples, memory mapped read-only
    labelix::L
    classweight::Vector{Float32}
    nlabels::Int
    useweights::Bool
end

"""
    _writecompactmatrix(path, featurestores, featurecols, coinixs, rows)

Write the features × samples Float32 matrix of the samples `(coinixs[i], rows[i])` to `path`.
The matrix is filled directly in a writable memory map of a temporary file that is renamed
to `path` afterwards, because truncating a file in place breaks still memory mapped readers.
"""
function _writecompactmatrix(path::AbstractString, featurestores::AbstractVector, featurecols::AbstractVector{<:AbstractString},
    coinixs::AbstractVector{<:Integer}, rows::AbstractVector{<:Integer})
    columns = [[store[!, col] for col in featurecols] for store in featurestores]
    @assert all(column -> eltype(column) == Float32, Iterators.flatten(columns)) "feature store columns must be Float32; found $(unique(eltype.(Iterators.flatten(columns))))"
    nfeatures = length(featurecols)
    tmppath = path * ".tmp"
    open(tmppath, "w+") do io
        X = Mmap.mmap(io, Matrix{Float32}, (nfeatures, length(coinixs)))
        for sampleix in eachindex(coinixs)
            coincolumns = columns[coinixs[sampleix]]
            row = rows[sampleix]
            for featureix in 1:nfeatures
                X[featureix, sampleix] = coincolumns[featureix][row]
            end
        end
        Mmap.sync!(X)
    end
    mv(tmppath, path; force=true)
    return path
end

"""
    CompactMmapSource(featurestores, featurecols, index; folderpath, nlabels, classbalancing)

Write the compact features × samples matrix and aligned UInt8 labels for samples in `index` to
`TRAINFEATURES_FILE` and `TRAINLABELS_FILE` in `folderpath`, then memory-map both read-only.
`featurestores` contains per-coin feature tables in coin-index order, `featurecols` the feature
column names in classifier input order and `index` has columns `coinix, row, labelix`. Original
coin and row coordinates are used only while compacting; sample indices then address both files.
"""
function CompactMmapSource(featurestores::AbstractVector, featurecols::AbstractVector{<:AbstractString}, index;
    folderpath::AbstractString, nlabels::Integer, classbalancing::Bool)
    @assert !isempty(featurestores) "CompactMmapSource requires at least one feature store"
    @assert !isempty(featurecols) "CompactMmapSource requires at least one feature column"
    @assert 0 < nlabels <= typemax(UInt8) "nlabels=$(nlabels) must be within 1:$(typemax(UInt8))"
    coinixs = index[!, :coinix]
    rows = index[!, :row]
    labelix = index[!, :labelix]
    nsamp = length(coinixs)
    @assert nsamp > 0 "CompactMmapSource requires a non-empty training index"
    @assert length(rows) == length(labelix) == nsamp "index column lengths differ: coinix=$(nsamp) row=$(length(rows)) labelix=$(length(labelix))"
    @assert all(coinix -> 1 <= coinix <= length(featurestores), coinixs) "coinix values must be within 1:$(length(featurestores))"
    @assert all(label -> 1 <= label <= nlabels, labelix) "labelix values must be within 1:$(nlabels)"
    mkpath(folderpath)
    path = joinpath(folderpath, TRAINFEATURES_FILE)
    (verbosity >= 2) && println("$(EnvConfig.now()) writing compact training features $(length(featurecols))×$(nsamp) to $(path)")
    _writecompactmatrix(path, featurestores, featurecols, coinixs, rows)
    labelpath = joinpath(folderpath, TRAINLABELS_FILE)
    labeltmppath = labelpath * ".tmp"
    open(labeltmppath, "w+") do io
        labels = Mmap.mmap(io, Vector{UInt8}, nsamp)
        labels .= labelix
        Mmap.sync!(labels)
    end
    mv(labeltmppath, labelpath; force=true)
    expectedbytes = sizeof(Float32) * length(featurecols) * nsamp
    @assert filesize(path) == expectedbytes "$(path) has $(filesize(path)) bytes; expected $(expectedbytes)"
    @assert filesize(labelpath) == nsamp "$(labelpath) has $(filesize(labelpath)) bytes; expected $(nsamp)"
    X = open(io -> Mmap.mmap(io, Matrix{Float32}, (length(featurecols), nsamp)), path, "r")
    labels = open(io -> Mmap.mmap(io, Vector{UInt8}, nsamp), labelpath, "r")
    counts = [count(==(UInt8(ix)), labels) for ix in 1:nlabels]
    classweight = classbalancing ? classweights(counts) : ones(Float32, nlabels)
    return CompactMmapSource(X, labels, classweight, Int(nlabels), classbalancing)
end

"Return the number of samples in the compact memory-mapped source."
nsamples(src::CompactMmapSource) = size(src.X, 2)

"""
Return the feature, one-hot label and optional weight batch for sample indices `ixs`.

Indices are sorted so samples sharing a memory page are read together.
"""
function batch(src::CompactMmapSource, ixs::AbstractVector{<:Integer})
    sortedixs = sort(ixs)
    x = src.X[:, sortedixs]
    y = Flux.onehotbatch(src.labelix[sortedixs], 1:src.nlabels)
    w = src.useweights ? src.classweight[src.labelix[sortedixs]] : nothing
    return x, y, w
end

"Return true class indices for sample indices `ixs`."
classindex(src::CompactMmapSource, ixs::AbstractVector{<:Integer}) = src.labelix[ixs]

"Return sample indices in the order used by `batch`."
batchindices(src::AbstractSampleSource, ixs::AbstractVector{<:Integer}) = ixs
batchindices(src::CompactMmapSource, ixs::AbstractVector{<:Integer}) = sort(ixs)

"Return checked UInt32 sample indices for a sample count, to keep adaptation index vectors compact."
function _sampleindices(nsamples::Integer)::UnitRange{UInt32}
    @assert 0 <= nsamples <= typemax(UInt32) "nsamples=$(nsamples) must be within 0:$(typemax(UInt32))"
    return UInt32(1):UInt32(nsamples)
end

"Return inverse-frequency class weights `1 / (n_active_classes * p(class))` for class sample `counts`; absent classes get weight 0."
function classweights(counts::AbstractVector{<:Integer})::Vector{Float32}
    total = sum(counts)
    @assert total > 0 "class counts must not all be zero: $(counts)"
    active = count(>(0), counts)
    return Float32[c > 0 ? total / (active * c) : 0f0 for c in counts]
end

"Return `(sum of weighted sample losses, sum of weights)` of `nn` over the samples `ixs` evaluated in chunks of `chunksize`."
function _lossaccumulation(nn::NN, src::AbstractSampleSource, ixs::AbstractVector{<:Integer}, chunksize::Integer)
    losssum = 0f0
    weightsum = 0f0
    for chunk in Iterators.partition(ixs, chunksize)
        x, y, w = batch(src, chunk)
        losses = samplelosses(nn, nn.valuecorrection(nn.model(x)), y)
        losssum += isnothing(w) ? sum(losses) : sum(w .* losses)
        weightsum += isnothing(w) ? length(losses) : sum(w)
    end
    return losssum, weightsum
end

"Return the (weighted) mean loss of `nn` over all samples of `src` with deterministic inference behavior."
function fullsetloss(nn::NN, src::AbstractSampleSource; chunksize::Integer=8192)
    Flux.testmode!(nn.model)
    losssum, weightsum = _lossaccumulation(nn, src, 1:nsamples(src), chunksize)
    Flux.trainmode!(nn.model)
    @assert weightsum > 0f0 "sum of sample weights must be > 0 but is $(weightsum)"
    return losssum / weightsum
end

"Return a packed bit vector marking samples of `src` whose predicted class differs from the target class."
function misclassificationflags(nn::NN, src::AbstractSampleSource; chunksize::Integer=8192)
    Flux.testmode!(nn.model)
    flags = falses(nsamples(src))
    for chunk in Iterators.partition(_sampleindices(nsamples(src)), chunksize)
        orderedixs = batchindices(src, chunk)
        x, _, _ = batch(src, orderedixs)
        predix = vec(map(argmax, eachcol(nn.valuecorrection(nn.model(x)))))
        trueix = classindex(src, orderedixs)
        flags[orderedixs] .= predix .!= trueix
    end
    Flux.trainmode!(nn.model)
    return flags
end

"Return sample weights with hard-example emphasis normalized to preserve each class's total weight."
function _reinforcementweights(baseweights::Union{Nothing, AbstractVector{Float32}}, labels::AbstractVector{<:Integer},
    flags::AbstractVector{Bool}, hardnessmeans::AbstractVector{Float32})
    @assert length(labels) == length(flags) "length(labels)=$(length(labels)) must equal length(flags)=$(length(flags))"
    @assert all(label -> 1 <= label <= length(hardnessmeans), labels) "labels must be within 1:$(length(hardnessmeans))"
    @assert isnothing(baseweights) || length(baseweights) == length(labels) "length(baseweights)=$(isnothing(baseweights) ? 0 : length(baseweights)) must equal length(labels)=$(length(labels))"
    weights = ones(Float32, length(labels))
    isnothing(baseweights) || (weights .= baseweights)
    for sampleix in eachindex(labels)
        label = labels[sampleix]
        weights[sampleix] *= (flags[sampleix] ? 2f0 : 1f0) / hardnessmeans[label]
    end
    return weights
end

"Return per-class mean hardness for binary 1×/2× emphasis flags."
function _classhardnessmeans(src::AbstractSampleSource, flags::AbstractVector{Bool}, nlabels::Integer; chunksize::Integer=8192)
    @assert length(flags) == nsamples(src) "length(flags)=$(length(flags)) must equal nsamples(src)=$(nsamples(src))"
    @assert nlabels > 0 "nlabels=$(nlabels) must be > 0"
    classcounts = zeros(Int, nlabels)
    hardcounts = zeros(Int, nlabels)
    for chunk in Iterators.partition(_sampleindices(nsamples(src)), chunksize)
        labels = classindex(src, chunk)
        for (offset, sampleix) in enumerate(chunk)
            label = labels[offset]
            classcounts[label] += 1
            flags[sampleix] && (hardcounts[label] += 1)
        end
    end
    means = ones(Float32, nlabels)
    for label in eachindex(means)
        classcounts[label] > 0 && (means[label] += hardcounts[label] / classcounts[label])
    end
    return means
end

"""
    adaptnn!(nn, src; maxepochs, reinforce_epochs=0, batchsize=64, folderpath=EnvConfig.logfolder())

Adapt `nn` with shuffled batches of `src` for at most `maxepochs` epochs or until the loss over
the full adaptation set (registered per epoch in `nn.losses`) converges. `nn` is checkpointed
per epoch under `nn.fileprefix` in `folderpath`.

If `reinforce_epochs > 0`, training still uses every sample. Every `reinforce_epochs`, the
misclassification flags are refreshed over the full adaptation set; misclassified samples receive
2× emphasis, normalized within their class so each class retains its original total weight.
The packed flags are in-memory training state and are rebuilt after a restart.
"""
function adaptnn!(nn::NN, src::AbstractSampleSource; maxepochs::Integer, reinforce_epochs::Integer=0, batchsize::Integer=64, folderpath::AbstractString=EnvConfig.logfolder())
    @assert maxepochs > 0 "maxepochs=$(maxepochs) must be > 0"
    @assert reinforce_epochs >= 0 "reinforce_epochs=$(reinforce_epochs) must be >= 0"
    @assert batchsize > 0 "batchsize=$(batchsize) must be > 0"
    active = collect(_sampleindices(nsamples(src)))
    hardflags = falses(nsamples(src))
    hardnessmeans = ones(Float32, length(nn.labels))
    (verbosity >= 3) && println("adaptnn! with $(nsamples(src)) samples of $(typeof(src).name.name), maxepochs=$(maxepochs), batchsize=$(batchsize)")
    Flux.trainmode!(nn.model)
    minloss = maxloss = missing
    breakmsg = "epoch loop finished without convergence"
    @showprogress for epoch in 1:maxepochs
        shuffle!(active)
        for chunk in Iterators.partition(active, batchsize)
            orderedixs = batchindices(src, chunk)
            x, y, w = batch(src, orderedixs)
            if (reinforce_epochs > 0) && (nn.lossfunc === Flux.logitcrossentropy)
                labels = classindex(src, orderedixs)
                w = _reinforcementweights(w, labels, hardflags[orderedixs], hardnessmeans)
            end
            loss = trainbatch!(nn, x, y, w)
            minloss = ismissing(minloss) ? loss : min(minloss, loss)
            maxloss = ismissing(maxloss) ? loss : max(maxloss, loss)
        end
        push!(nn.losses, fullsetloss(nn, src))
        savenn(nn; folderpath=folderpath)
        if nnconverged(nn)
            breakmsg = "stopping adaptation after epoch $epoch due to no loss reduction ($(nn.losses[end-4:end])) in last 4 epochs"
            break
        end
        if (reinforce_epochs > 0) && (epoch % reinforce_epochs == 0)
            if nn.lossfunc === Flux.logitcrossentropy
                hardflags = misclassificationflags(nn, src)
                hardnessmeans = _classhardnessmeans(src, hardflags, length(nn.labels))
                breakmsg = "refreshed soft reinforcement after epoch $epoch with $(count(hardflags)) hard samples"
            else
                breakmsg = "reinforcement skipped at epoch $epoch because nn.lossfunc=$(nn.lossfunc) is not Flux.logitcrossentropy"
            end
        end
    end
    Flux.testmode!(nn.model)
    println(breakmsg)  # print after showprogress loop to avoid cluttered text output
    println("minloss=$minloss  maxloss=$maxloss")
    return nn
end

#endregion SampleSource
