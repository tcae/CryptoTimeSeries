"""
    fixedevalweights(regimes, folds)::Vector{Float32}

Give equal mass to each represented regime, then to each represented fold within
that regime, then to each sample in the cell. Unknown regimes are forbidden.
Membership and these per-sample weights must be frozen before training.
"""
function fixedevalweights(regimes::AbstractVector, folds::AbstractVector)::Vector{Float32}
    @assert length(regimes) == length(folds) > 0 "regimes=$(length(regimes)), folds=$(length(folds)) must be equal and nonempty"
    @assert all(regime -> !ismissing(regime) && string(regime) != "unknown", regimes) "regimes=$(unique(regimes)) must exclude unknown and missing"
    represented = unique(regimes)
    weights = zeros(Float32, length(regimes))
    for regime in represented
        regimefolds = unique(folds[regimes .== regime])
        for fold in regimefolds
            cell = (regimes .== regime) .& (folds .== fold)
            weights[cell] .= 1 / (length(represented) * length(regimefolds) * count(cell))
        end
    end
    return weights
end

"""Write a complete BSON payload to a sibling staging file and atomically replace its destination."""
function _atomicevalsave(path::AbstractString, payload)
    staging = path * ".staging"
    try
        BSON.bson(staging, Dict(:payload => payload))
        mv(staging, path; force=true)
    finally
        isfile(staging) && rm(staging)
    end
    return path
end

"""Read an eval checkpoint payload in Classify's module context."""
_loadevalpayload(path::AbstractString) = BSON.load(path, @__MODULE__)[:payload]

"""
    loadtraincheckpoint(folderpath; identity::String, best::Bool=true)::NN

Load an inference-mode NN without training. Verify the immutable run manifest and
checkpoint manifest against `identity`; missing or incompatible artifacts fail.
Default deployment/test selection is best eval, never last epoch. The authoritative
last checkpoint contains the best NN, so loading best also repairs an interrupted
best-file replacement without running optimization.
"""
function loadtraincheckpoint(folderpath::AbstractString; identity::String, best::Bool=true)::NN
    manifest = _loadevalpayload(joinpath(folderpath, "eval-manifest.bson"))
    @assert manifest.identity == identity "manifest identity=$(repr(manifest.identity)) differs from requested identity=$(repr(identity))"
    payload = _loadevalpayload(joinpath(folderpath, "eval-last.bson"))
    @assert isequal(payload.manifest, manifest) "last checkpoint manifest differs from immutable manifest for identity=$(identity)"
    @assert !isempty(payload.records) && length(payload.nn.losses) == length(payload.records) "checkpoint train epochs=$(length(payload.nn.losses)), eval epochs=$(length(payload.records)) must agree and be nonempty"
    if best
        @assert payload.best isa NN "best checkpoint model=$(typeof(payload.best)) must be NN for identity=$(identity)"
        bestpath = joinpath(folderpath, "eval-best.bson")
        if isfile(bestpath)
            stored = _loadevalpayload(bestpath)
            @assert isequal(stored.manifest, manifest) "best checkpoint manifest differs from immutable manifest for identity=$(identity)"
        end
        _atomicevalsave(bestpath, (; nn=payload.best, records=payload.records, manifest,
            bestepoch=argmin([record.evalweighted for record in payload.records])))
    end
    nn = best ? payload.best : payload.nn
    Flux.testmode!(nn.model)
    return nn
end

"""Reject ignore-target membership in opt-in training/evaluation sources."""
function _checkevalmembership(nn::NN, src::AbstractSampleSource)
    ignored = findall(label -> string(label) == "ignore", nn.labels)
    isempty(ignored) && return nothing
    for chunk in Iterators.partition(1:nsamples(src), 8192)
        indices = classindex(src, chunk)
        @assert all(index -> !(index in ignored), indices) "source contains ignore target indices=$(indices)"
    end
    return nothing
end

"""
    evallosses(nn, src; weights, regimes, folds, chunksize=8192)

Return natural pooled loss, fixed-weight loss and cell counts/losses. Source
class weights are deliberately ignored. Native BatchNorm modes and running state
are preserved, including when evaluation fails.
"""
function evallosses(nn::NN, src::AbstractSampleSource; weights::AbstractVector{Float32},
    regimes::AbstractVector, folds::AbstractVector, chunksize::Integer=8192)
    @assert nn.model isa SimpleChainsModel "model=$(typeof(nn.model)) must be a native SimpleChains model"
    n = nsamples(src)
    @assert n == length(weights) == length(regimes) == length(folds) > 0 "samples=$(n), weights=$(length(weights)), regimes=$(length(regimes)), folds=$(length(folds)) must agree"
    @assert chunksize > 0 "chunksize=$(chunksize) must be positive"
    @assert all(isfinite, weights) && all(>(0), weights) "weights=$(weights) must be finite and strictly positive"
    @assert all(regime -> !ismissing(regime) && string(regime) != "unknown", regimes) "regimes=$(unique(regimes)) must exclude unknown and missing"
    _checkevalmembership(nn, src)
    stateful = filter(layer -> layer isa SimpleChainsBatchNorm, nn.model.chain.layers)
    modes = map(layer -> layer.training, stateful)
    losses = Vector{Float64}(undef, n)
    Flux.testmode!(nn.model)
    try
        for chunk in Iterators.partition(1:n, chunksize)
            ordered = batchindices(src, chunk)
            x, y, _ = batch(src, ordered)
            losses[ordered] .= samplelosses(nn, nn.valuecorrection(nn.model(x)), y)
        end
    finally
        foreach((layer, mode) -> (layer.training = mode), stateful, modes)
    end
    @assert all(loss -> isfinite(loss) && loss >= 0, losses) "eval losses=$(losses) must be finite and nonnegative"
    cells = [(; regime, foldid=fold, count=count(cell),
        natural=mean(losses[cell]), weighted=sum(losses[cell] .* weights[cell]) / sum(weights[cell]))
        for (regime, fold) in unique(collect(zip(regimes, folds)))
        for cell in [(regimes .== regime) .& (folds .== fold)]]
    return (; natural=mean(losses), weighted=sum(losses .* weights) / sum(weights),
        cells, count=n, weightmass=sum(weights), maxweight=maximum(weights))
end

"""Return shuffled training chunks, merging a singleton tail for native BatchNorm."""
function _evaltrainchunks(active::AbstractVector, batchsize::Integer)
    chunks = collect.(Iterators.partition(active, batchsize))
    if length(chunks) > 1 && length(last(chunks)) == 1
        append!(chunks[end-1], pop!(chunks))
    end
    return chunks
end

"""
Refresh train-only hard-example flags and normalize emphasis by each class's
existing sample-weight mass, including optional regime/fold weights.
"""
function _evalreinforcement(nn::NN, train::AbstractSampleSource, interval::Integer,
        epoch::Integer, history::AbstractVector)
    layers = filter(layer -> layer isa SimpleChainsBatchNorm, nn.model.chain.layers)
    modes = map(layer -> layer.training,layers)
    flags = try
        misclassificationflags(nn,train)
    finally
        foreach((layer,mode)->(layer.training=mode),layers,modes)
    end
    mass = zeros(Float32,length(nn.labels))
    hardmass = zeros(Float32,length(nn.labels))
    for chunk in Iterators.partition(1:nsamples(train),8192)
        ordered = batchindices(train,chunk)
        _,_,weights = batch(train,ordered)
        labels = classindex(train,ordered)
        for (offset,index) in enumerate(ordered)
            weight = isnothing(weights) ? 1f0 : weights[offset]
            mass[labels[offset]] += weight
            flags[index] && (hardmass[labels[offset]] += weight)
        end
    end
    means = ones(Float32,length(nn.labels))
    for label in eachindex(means)
        mass[label]>0 && (means[label] += hardmass[label]/mass[label])
    end
    refreshed = vcat(history,[(;epoch,interval,hardcount=count(flags))])
    @info "refreshed training reinforcement" epoch interval hardcount=count(flags)
    return (;interval,flags,means,history=refreshed)
end

"""
    traineval!(nn, train, eval; identity::String, folderpath, evalweights=nothing,
               regimes=fill("overall", nsamples(eval)), folds=fill(0, nsamples(eval)),
               maxepochs=200, batchsize=64, resume=true, seed=1, reinforce_epochs=0)

Opt-in native SimpleChains/M011 training. Stop on separately persisted fixed-weight
eval loss (50-epoch patience, 0.0005 meaningful improvement, 200 total epochs).
`nn.losses` remains train history. Default eval weights balance regime/fold cells;
explicit positive Float32 weights override this. Eval source class weights are
never used. Callers must exclude ignore/unknown samples and encode exact train/eval
membership, data/config revisions and training-weight identity in `identity`.

Return `(best, last, records, manifest, stopreason)` as a named tuple. `best` and
`last` are inference-mode NNs. Resume automatically from `eval-last.bson`, including
optimizer and private RNG. An immutable `eval-manifest.bson` rejects changes to
identity, membership counts, eval weights/cells, labels, architecture, batchsize or
seed. `maxepochs` may increase on resume, never beyond 200.

Each checkpoint is an atomic replacement. The authoritative last payload also
contains the selected best NN so interruption between last/best file writes is
repaired on resume. Deploy `eval-best.bson`'s `payload.nn`, not the last checkpoint.
Use `loadtraincheckpoint(folderpath; identity)` for deployment/test without training.
Use a separate folder from legacy `adaptnn!` artifacts.

`reinforce_epochs` is a nonnegative refresh interval, not an extra epoch budget.
After each interval, misclassified train samples receive 2× emphasis normalized
within class to preserve existing train weight mass. All samples remain in batches.
Flags/normalizers and refresh history are checkpointed for exact resume. Enabling
or changing reinforcement on resume refreshes flags before the next epoch and
logs the change. Eval weights/losses, patience history and total cap never reset.
"""
function traineval!(nn::NN, train::AbstractSampleSource, eval::AbstractSampleSource;
    identity::String, folderpath::AbstractString, evalweights::Union{Nothing,AbstractVector{Float32}}=nothing,
    regimes::AbstractVector=fill("overall", nsamples(eval)), folds::AbstractVector=fill(0, nsamples(eval)),
    maxepochs::Integer=200, batchsize::Integer=64, resume::Bool=true, seed::Integer=1,
    reinforce_epochs::Integer=0)
    @assert !isempty(identity) "identity=$(repr(identity)) must not be empty"
    @assert nn.model isa SimpleChainsModel "model=$(typeof(nn.model)) must be native SimpleChains"
    @assert reinforce_epochs>=0 "reinforce_epochs=$reinforce_epochs must be nonnegative"
    @assert reinforce_epochs==0 || nn.lossfunc===Flux.logitcrossentropy "reinforcement requires logitcrossentropy, got=$(nn.lossfunc)"
    @assert maxepochs > 0 && batchsize >= 2 && nsamples(train) >= 2 "maxepochs=$(maxepochs), batchsize=$(batchsize), train samples=$(nsamples(train)) must permit training"
    _checkevalmembership(nn, train)
    weights = isnothing(evalweights) ? fixedevalweights(regimes, folds) : copy(evalweights)
    # Validate the complete frozen eval contract before persisting anything.
    evallosses(nn, eval; weights, regimes, folds)
    manifest = (; version=1, identity, traincount=nsamples(train), evalcount=nsamples(eval),
        weights, regimes=copy(regimes), folds=copy(folds), labels=copy(nn.labels),
        architecture=(string(typeof(nn.model.chain)), map(nn.model.chain.layers) do layer
            layer isa SimpleChainsBatchNorm ?
                (length(layer.mean), layer.epsilon, layer.momentum) : repr(layer)
        end), description=nn.description, batchsize, seed)
    mkpath(folderpath)
    manifestpath = joinpath(folderpath, "eval-manifest.bson")
    lastpath = joinpath(folderpath, "eval-last.bson")
    bestpath = joinpath(folderpath, "eval-best.bson")
    if isfile(manifestpath)
        @assert resume "resume=$(resume) cannot overwrite an existing eval manifest"
        @assert isequal(manifest, _loadevalpayload(manifestpath)) "eval manifest mismatch for identity=$(identity)"
    else
        @assert !isfile(lastpath) && !isfile(bestpath) "orphan checkpoints in folderpath=$(folderpath)"
        @assert isempty(nn.losses) "fresh eval run requires empty train history; epochs=$(length(nn.losses))"
        _atomicevalsave(manifestpath, manifest)
    end
    records = NamedTuple[]
    rng = MersenneTwister(seed)
    best = nothing
    reinforcement = (;interval=0,flags=falses(nsamples(train)),
        means=ones(Float32,length(nn.labels)),history=NamedTuple[])
    if isfile(lastpath)
        payload = _loadevalpayload(lastpath)
        @assert isequal(payload.manifest, manifest) "last checkpoint manifest mismatch for identity=$(identity)"
        nn, best, records, rng = payload.nn, payload.best, payload.records, payload.rng
        if hasproperty(payload,:reinforcement)
            reinforcement = payload.reinforcement
        elseif reinforce_epochs>0
            @info "initializing reinforcement for an existing unreinforced checkpoint" identity
        end
        @assert length(reinforcement.flags)==nsamples(train) && length(reinforcement.means)==length(nn.labels) "reinforcement state mismatch: flags=$(length(reinforcement.flags)), samples=$(nsamples(train)), means=$(length(reinforcement.means)), labels=$(length(nn.labels))"
        @assert length(nn.losses) == length(records) "train epochs=$(length(nn.losses)) differ from eval epochs=$(length(records))"
    end
    limit = min(maxepochs, 200)
    if reinforcement.interval!=reinforce_epochs
        @info "training reinforcement interval changed" previous=reinforcement.interval requested=reinforce_epochs epoch=length(records)
        reinforcement = if reinforce_epochs>0 && !isempty(records)
            _evalreinforcement(nn,train,reinforce_epochs,length(records),reinforcement.history)
        else
            (;interval=reinforce_epochs,flags=falses(nsamples(train)),
                means=ones(Float32,length(nn.labels)),history=reinforcement.history)
        end
    end
    active = collect(1:nsamples(train))
    while length(records) < limit && !nnconverged([record.evalweighted for record in records])
        started = time()
        Flux.trainmode!(nn.model)
        # Start from canonical indices each epoch so saved RNG alone reproduces shuffle.
        active .= 1:nsamples(train)
        shuffle!(rng, active)
        for chunk in _evaltrainchunks(active, batchsize)
            ordered = batchindices(train, chunk)
            x, y, w = batch(train, ordered)
            if reinforce_epochs>0
                w = _reinforcementweights(w,classindex(train,ordered),
                    reinforcement.flags[ordered],reinforcement.means)
            end
            trainbatch!(nn, x, y, w)
        end
        trainloss = fullsetloss(nn, train)
        @assert isfinite(trainloss) && trainloss >= 0 "train loss=$(trainloss) must be finite and nonnegative"
        evaluation = evallosses(nn, eval; weights, regimes, folds)
        previousbest = isempty(records) ? Inf : minimum(record.evalweighted for record in records)
        push!(nn.losses, trainloss)
        push!(records, (; epoch=length(records)+1, trainloss, evalnatural=evaluation.natural,
            evalweighted=evaluation.weighted, cells=evaluation.cells, evalcount=evaluation.count,
            traincount=nsamples(train), weightmass=evaluation.weightmass,
            maxweight=evaluation.maxweight, seconds=time()-started))
        Flux.testmode!(nn.model)
        evaluation.weighted < previousbest && (best = deepcopy(nn))
        if reinforce_epochs>0 && length(records)%reinforce_epochs==0
            reinforcement = _evalreinforcement(nn,train,reinforce_epochs,
                length(records),reinforcement.history)
        end
        _atomicevalsave(lastpath, (; nn, best, records, manifest, rng, reinforcement))
        _atomicevalsave(bestpath, (; nn=best, records, manifest,
            bestepoch=argmin([record.evalweighted for record in records])))
    end
    @assert !isnothing(best) "no completed eval checkpoint for identity=$(identity)"
    Flux.testmode!(nn.model)
    Flux.testmode!(best.model)
    stopreason = nnconverged([record.evalweighted for record in records]) ? :patience : :epochcap
    _atomicevalsave(bestpath, (; nn=best, records, manifest,
        bestepoch=argmin([record.evalweighted for record in records])))
    @info "eval training stopped" identity epochs=length(records) stopreason
    return (; best, last=nn, records, manifest, stopreason, reinforcement)
end
