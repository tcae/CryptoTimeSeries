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
class index `labelix[i]` (position in the classifier label order) and sample weight `weight[i]`.
Batches gather the features of their samples sorted by storage location and build lazy one-hot targets.
"""
struct ArrowIndexSource{V<:AbstractVector{Float32}, C<:AbstractVector{<:Integer}, R<:AbstractVector{<:Integer}, L<:AbstractVector{<:Integer}, WV<:AbstractVector{Float32}} <: AbstractSampleSource
    features::Vector{Vector{V}}  # [coinix][featureix] memory mapped feature store column
    coinix::C
    row::R
    labelix::L
    weight::WV
    nlabels::Int
    useweights::Bool
end

"""
    ArrowIndexSource(featurestores, featurecols, index; nlabels, useweights)

`featurestores` is a vector of per-coin feature store tables in coin index order, `featurecols`
the feature column names in classifier input order and `index` the training index table with
columns `coinix, row, labelix, weight`.
"""
function ArrowIndexSource(featurestores::AbstractVector, featurecols::AbstractVector{<:AbstractString}, index; nlabels::Integer, useweights::Bool)
    @assert !isempty(featurestores) "ArrowIndexSource requires at least one feature store"
    columns = [[store[!, col] for col in featurecols] for store in featurestores]
    V = typeof(columns[1][1])
    @assert all(col -> col isa V, Iterators.flatten(columns)) "all feature store columns must share type $(V); found $(unique(typeof.(Iterators.flatten(columns))))"
    @assert eltype(V) == Float32 "feature store columns must be Float32 but are $(eltype(V))"
    features = Vector{Vector{V}}([Vector{V}(cols) for cols in columns])
    src = ArrowIndexSource(features, index[!, :coinix], index[!, :row], index[!, :labelix], index[!, :weight], Int(nlabels), useweights)
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
    w = src.useweights ? src.weight[sortedixs] : nothing
    return x, y, w
end

classindex(src::ArrowIndexSource, ixs::AbstractVector{<:Integer}) = src.labelix[ixs]

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

"Return the sample indices of `src` whose predicted class differs from the target class."
function misclassifiedindices(nn::NN, src::AbstractSampleSource; chunksize::Integer=8192)
    Flux.testmode!(nn.model)
    wrongix = Int[]
    for chunk in Iterators.partition(1:nsamples(src), chunksize)
        x, _, _ = batch(src, chunk)
        predix = vec(map(argmax, eachcol(nn.valuecorrection(nn.model(x)))))
        trueix = classindex(src, chunk)
        append!(wrongix, chunk[predix .!= trueix])
    end
    Flux.trainmode!(nn.model)
    return wrongix
end

"""
    adaptnn!(nn, src; maxepochs, reinforce_epochs=0, batchsize=64, folderpath=EnvConfig.logfolder())

Adapt `nn` with shuffled batches of `src` for at most `maxepochs` epochs or until the loss over
the full adaptation set (registered per epoch in `nn.losses`) converges. `nn` is checkpointed
per epoch under `nn.fileprefix` in `folderpath`.

If `reinforce_epochs > 0`, hard-sample filtering is applied every `reinforce_epochs` epochs:
training continues on the currently misclassified samples, but the misclassification check
always reevaluates the full adaptation set so previously correct samples can be reintroduced.
"""
function adaptnn!(nn::NN, src::AbstractSampleSource; maxepochs::Integer, reinforce_epochs::Integer=0, batchsize::Integer=64, folderpath::AbstractString=EnvConfig.logfolder())
    @assert maxepochs > 0 "maxepochs=$(maxepochs) must be > 0"
    active = collect(1:nsamples(src))
    (verbosity >= 3) && println("adaptnn! with $(nsamples(src)) samples of $(typeof(src).name.name), maxepochs=$(maxepochs), batchsize=$(batchsize)")
    Flux.trainmode!(nn.model)
    minloss = maxloss = missing
    breakmsg = "epoch loop finished without convergence"
    @showprogress for epoch in 1:maxepochs
        for chunk in Iterators.partition(shuffle(active), batchsize)
            x, y, w = batch(src, chunk)
            loss, grads = Flux.withgradient(nn.model) do m
                y_hat = nn.valuecorrection(m(x))
                isnothing(w) ? nn.lossfunc(y_hat, y) : weightedloss(nn, y_hat, y, w)
            end
            minloss = ismissing(minloss) ? loss : min(minloss, loss)
            maxloss = ismissing(maxloss) ? loss : max(maxloss, loss)
            Flux.update!(nn.optim, nn.model, grads[1])
        end
        push!(nn.losses, fullsetloss(nn, src))
        savenn(nn; folderpath=folderpath)
        if nnconverged(nn)
            breakmsg = "stopping adaptation after epoch $epoch due to no loss reduction ($(nn.losses[end-4:end])) in last 4 epochs"
            break
        end
        if (reinforce_epochs > 0) && (epoch % reinforce_epochs == 0)
            if nn.lossfunc === Flux.logitcrossentropy
                wrongix = misclassifiedindices(nn, src)
                if isempty(wrongix)
                    breakmsg = "stopping adaptation after epoch $epoch because reinforce filtering found no misclassified samples"
                    break
                end
                active = wrongix
                breakmsg = "applied reinforce filtering after epoch $epoch with $(length(wrongix)) hard samples; continuing training"
            else
                breakmsg = "reinforce filtering skipped at epoch $epoch because nn.lossfunc=$(nn.lossfunc) is not Flux.logitcrossentropy"
            end
        end
    end
    Flux.testmode!(nn.model)
    println(breakmsg)  # print after showprogress loop to avoid cluttered text output
    println("minloss=$minloss  maxloss=$maxloss")
    return nn
end

#endregion SampleSource
