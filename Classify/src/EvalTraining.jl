"""
    fixedevalweights(regimes, folds)::Vector{Float32}

Give equal mass to each represented regime, then to each represented fold within
that regime, then to each sample in the cell. Unknown regimes are forbidden.
Membership and these per-sample weights must be frozen before training.
"""
function fixedevalweights(regimes::AbstractVector, folds::AbstractVector)::Vector{Float32}
    @assert length(regimes) == length(folds) > 0 "regimes=$(length(regimes)), folds=$(length(folds)) must be equal and nonempty"
    @assert all(regime -> !ismissing(regime) && string(regime) != "unknown", regimes) "regimes=$(unique(regimes)) must exclude unknown and missing"
    counts = Dict{Tuple{eltype(regimes),eltype(folds)},Int}()
    regimefolds = Dict{eltype(regimes),Int}()
    for (regime, fold) in zip(regimes, folds)
        key = (regime, fold)
        if !haskey(counts, key)
            regimefolds[regime] = get(regimefolds, regime, 0) + 1
        end
        counts[key] = get(counts, key, 0) + 1
    end
    weights = zeros(Float32, length(regimes))
    for index in eachindex(weights)
        regime, fold = regimes[index], folds[index]
        weights[index] = 1 / (length(regimefolds) * regimefolds[regime] * counts[(regime, fold)])
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

"""Identify the separately persisted immutable manifest by its SHA-256 digest."""
_evalmanifestreference(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

"""Project an existing v1 training contract to v2 without retaining obsolete eval membership arrays."""
function _evalcontract(manifest)
    if hasproperty(manifest, :version) && manifest.version == 1
        compact = (; (key => value for (key, value) in pairs(manifest)
            if key != :regimes && key != :folds)...)
        return merge(compact, (; version=2))
    end
    return manifest
end

"""
Read an eval payload, resolving and verifying compact manifest references.
Existing checkpoints with embedded manifests remain resumable. An already verified
manifest/reference pair may be supplied to avoid reading the large manifest again.
"""
function _loadevalpayload(path::AbstractString; manifest=nothing, reference=nothing)
    payload = BSON.load(path, @__MODULE__)[:payload]
    if hasproperty(payload, :manifest)
        payload = merge(payload, (; manifest=_evalcontract(payload.manifest)))
    elseif hasproperty(payload, :weights)
        payload = _evalcontract(payload)
    end
    if hasproperty(payload, :manifestreference)
        manifestpath = joinpath(dirname(path), "eval-manifest.bson")
        expected = isnothing(reference) ? _evalmanifestreference(manifestpath) : reference
        @assert payload.manifestreference == expected "checkpoint manifest reference=$(payload.manifestreference) differs from immutable manifest digest=$expected"
        resolved = isnothing(manifest) ? _loadevalpayload(manifestpath) : manifest
        if hasproperty(payload, :manifest)
            @assert isequal(payload.manifest, resolved) "embedded checkpoint manifest differs from referenced immutable manifest for path=$path"
        end
        return merge(payload, (; manifest=resolved))
    end
    return payload
end

"""Save an eval checkpoint with a digest reference instead of embedded population metadata."""
function _saveevalcheckpoint(path::AbstractString, payload, reference::AbstractString)
    compact = (; (key => value for (key, value) in pairs(payload)
        if key != :manifest && key != :manifestreference)...)
    return _atomicevalsave(path, merge(compact, (; manifestreference=reference)))
end

"""Repair a missing or stale best checkpoint from authoritative last state; leave a current file untouched."""
function _repairevalbest(path::AbstractString, best::NN, records::AbstractVector, manifest,
        reference::AbstractString)::Nothing
    @assert !isempty(records) "eval records=$(length(records)) must be nonempty to repair the best checkpoint"
    bestepoch = argmin([record.evalweighted for record in records])
    if isfile(path)
        stored = _loadevalpayload(path; manifest, reference)
        @assert isequal(stored.manifest, manifest) "best checkpoint manifest differs from immutable manifest for identity=$(manifest.identity)"
        stored.bestepoch == bestepoch && return nothing
    end
    _saveevalcheckpoint(path, (; nn=best, records=records[1:bestepoch], bestepoch), reference)
    return nothing
end

"""
    loadtraincheckpoint(folderpath; identity::String, best::Bool=true)::NN

Load an inference-mode NN without training. Verify the immutable run manifest and
checkpoint manifest against `identity`; missing or incompatible artifacts fail.
Default deployment/test selection is best eval, never last epoch. The authoritative
last checkpoint contains the best NN, so loading best also repairs a missing or
stale best file without running optimization. A current best file is not rewritten.
"""
function loadtraincheckpoint(folderpath::AbstractString; identity::String, best::Bool=true)::NN
    manifest = _loadevalpayload(joinpath(folderpath, "eval-manifest.bson"))
    @assert manifest.identity == identity "manifest identity=$(repr(manifest.identity)) differs from requested identity=$(repr(identity))"
    reference = _evalmanifestreference(joinpath(folderpath, "eval-manifest.bson"))
    payload = _loadevalpayload(joinpath(folderpath, "eval-last.bson"); manifest, reference)
    @assert isequal(payload.manifest, manifest) "last checkpoint manifest differs from immutable manifest for identity=$(identity)"
    @assert !isempty(payload.records) && length(payload.nn.losses) == length(payload.records) "checkpoint train epochs=$(length(payload.nn.losses)), eval epochs=$(length(payload.records)) must agree and be nonempty"
    if best
        @assert payload.best isa NN "best checkpoint model=$(typeof(payload.best)) must be NN for identity=$(identity)"
        _repairevalbest(joinpath(folderpath, "eval-best.bson"), payload.best, payload.records, manifest, reference)
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
    evallosses(nn, src; weights, chunksize=8192)

Return natural pooled loss, fixed-weight loss, and separate
`inference_seconds` and `diagnostic_seconds` timings. Source
class weights are deliberately ignored. Native BatchNorm modes and running state
are preserved, including when evaluation fails. Losses are reduced per chunk;
only fixed sample weights are needed, not regime/fold membership.
"""
function evallosses(nn::NN, src::AbstractSampleSource; weights::AbstractVector{Float32},
    chunksize::Integer=8192)
    @assert nn.model isa SimpleChainsModel "model=$(typeof(nn.model)) must be a native SimpleChains model"
    n = nsamples(src)
    @assert n == length(weights) > 0 "samples=$n and weights=$(length(weights)) must agree and be nonempty"
    @assert chunksize > 0 "chunksize=$(chunksize) must be positive"
    @assert all(isfinite, weights) && all(>(0), weights) "weights=$(weights) must be finite and strictly positive"
    _checkevalmembership(nn, src)
    stateful = filter(layer -> layer isa SimpleChainsBatchNorm, nn.model.chain.layers)
    modes = map(layer -> layer.training, stateful)
    inference_seconds = 0.0
    diagnostic_seconds = 0.0
    losssum = 0.0
    weightedsum = 0.0
    weightmass = 0.0
    Flux.testmode!(nn.model)
    try
        for chunk in Iterators.partition(1:n, chunksize)
            inference_started = time_ns()
            ordered = batchindices(src, chunk)
            x, y, _ = batch(src, ordered)
            losses = samplelosses(nn, nn.valuecorrection(nn.model(x)), y)
            inference_seconds += (time_ns() - inference_started) / 1e9
            diagnostic_started = time_ns()
            for (offset, index) in enumerate(ordered)
                loss = losses[offset] + 0.0
                @assert isfinite(loss) && loss >= 0 "eval loss=$loss at index=$index must be finite and nonnegative"
                weight = weights[index]
                losssum += loss
                weightedsum += loss * weight
                weightmass += weight
            end
            diagnostic_seconds += (time_ns() - diagnostic_started) / 1e9
        end
    finally
        foreach((layer, mode) -> (layer.training = mode), stateful, modes)
    end
    diagnostic_started = time_ns()
    natural = losssum / n
    weighted = weightedsum / weightmass
    maxweight = maximum(weights)
    diagnostic_seconds += (time_ns() - diagnostic_started) / 1e9
    return (; natural, weighted, count=n, weightmass, maxweight,
        inference_seconds, diagnostic_seconds)
end

"""Lazily return views of shuffled indices, merging a singleton tail for native BatchNorm."""
function _evaltrainchunks(active::AbstractVector, batchsize::Integer)
    @assert batchsize >= 2 "batchsize=$batchsize must be at least two for native BatchNorm"
    n = length(active)
    chunkcount = cld(n, batchsize)
    n > batchsize && n % batchsize == 1 && (chunkcount -= 1)
    return (view(active, (index-1)*batchsize+1:(index == chunkcount ? n : index*batchsize))
        for index in 1:chunkcount)
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
               regimes=nothing, folds=nothing,
               maxepochs=200, batchsize=64, resume=true, seed=1, reinforce_epochs=0)

Opt-in native SimpleChains/M011 training. Stop on separately persisted fixed-weight
eval loss (50-epoch patience, 0.0005 meaningful improvement, 200 total epochs).
`nn.losses` and each record's `trainloss` store training-time batch loss, aggregated
by effective sample-weight mass (including reinforcement), not a final-model
inference loss. No extra training-set inference pass is performed. Earlier records
in resumed runs retain their original loss semantics. Logs identify the metric
and report adaptation, eval inference, eval diagnostics and checkpoint timings.
Explicit positive Float32 eval weights are required unless regime/fold vectors
are supplied to construct balanced weights once; without either, samples are
weighted uniformly. Membership is not retained in the training contract or used
for per-epoch diagnostics. Eval source class weights are
never used. Callers must exclude ignore/unknown samples and encode exact train/eval
membership, data/config revisions and training-weight identity in `identity`.

Return `(best, last, records, manifest, stopreason)` as a named tuple. `best` and
`last` are inference-mode NNs. Resume automatically from `eval-last.bson`, including
optimizer and private RNG. An immutable `eval-manifest.bson` rejects changes to
identity, membership counts, eval weights, labels, architecture, batchsize or
seed. `maxepochs` may increase on resume, never beyond 200.
Shuffled sample indices use UInt32 and batches are lazy views of that one vector.

Each checkpoint is an atomic replacement. Last is saved every epoch with complete
history; best is saved only for a strictly lower weighted eval loss, with history
through that best epoch. The authoritative last payload also contains the selected
best NN so a missing/stale best file is repaired on resume, even without further
epochs. Deploy `eval-best.bson`'s `payload.nn`, not the last checkpoint.
Use `loadtraincheckpoint(folderpath; identity)` for deployment/test without training.
Checkpoint payloads reference the separate immutable manifest by SHA-256 rather
than embedding its population metadata. Load with `_loadevalpayload` to resolve
that reference. Existing embedded-manifest checkpoints can still be resumed.
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
    regimes::Union{Nothing,AbstractVector}=nothing, folds::Union{Nothing,AbstractVector}=nothing,
    maxepochs::Integer=200, batchsize::Integer=64, resume::Bool=true, seed::Integer=1,
    reinforce_epochs::Integer=0)
    @assert !isempty(identity) "identity=$(repr(identity)) must not be empty"
    @assert nn.model isa SimpleChainsModel "model=$(typeof(nn.model)) must be native SimpleChains"
    @assert reinforce_epochs>=0 "reinforce_epochs=$reinforce_epochs must be nonnegative"
    @assert reinforce_epochs==0 || nn.lossfunc===Flux.logitcrossentropy "reinforcement requires logitcrossentropy, got=$(nn.lossfunc)"
    @assert maxepochs > 0 && batchsize >= 2 && nsamples(train) >= 2 "maxepochs=$(maxepochs), batchsize=$(batchsize), train samples=$(nsamples(train)) must permit training"
    println("$(EnvConfig.now()) validating train/eval populations: train=$(nsamples(train)) eval=$(nsamples(eval)) checkpoints=$folderpath")
    flush(stdout)
    _checkevalmembership(nn, train)
    @assert isnothing(regimes) == isnothing(folds) "regimes=$(typeof(regimes)) and folds=$(typeof(folds)) must both be provided or omitted"
    weights = if !isnothing(evalweights)
        evalweights
    elseif !isnothing(regimes)
        fixedevalweights(regimes, folds)
    else
        fill(1f0 / nsamples(eval), nsamples(eval))
    end
    regimes = folds = nothing
    # Validate the complete frozen eval contract before persisting anything.
    evallosses(nn, eval; weights)
    manifest = (; version=2, identity, traincount=nsamples(train), evalcount=nsamples(eval),
        weights, labels=copy(nn.labels),
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
    reference = _evalmanifestreference(manifestpath)
    records = NamedTuple[]
    rng = MersenneTwister(seed)
    best = nothing
    reinforcement = (;interval=0,flags=falses(nsamples(train)),
        means=ones(Float32,length(nn.labels)),history=NamedTuple[])
    if isfile(lastpath)
        payload = _loadevalpayload(lastpath; manifest, reference)
        @assert isequal(payload.manifest, manifest) "last checkpoint manifest mismatch for identity=$(identity)"
        nn, best, records, rng = payload.nn, payload.best, payload.records, payload.rng
        if hasproperty(payload,:reinforcement)
            reinforcement = payload.reinforcement
        elseif reinforce_epochs>0
            @info "initializing reinforcement for an existing unreinforced checkpoint" identity
        end
        @assert length(reinforcement.flags)==nsamples(train) && length(reinforcement.means)==length(nn.labels) "reinforcement state mismatch: flags=$(length(reinforcement.flags)), samples=$(nsamples(train)), means=$(length(reinforcement.means)), labels=$(length(nn.labels))"
        @assert length(nn.losses) == length(records) "train epochs=$(length(nn.losses)) differ from eval epochs=$(length(records))"
        @assert best isa NN "best checkpoint model=$(typeof(best)) must be NN for identity=$(identity)"
        _repairevalbest(bestpath, best, records, manifest, reference)
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
    active = collect(_sampleindices(nsamples(train)))
    println("$(EnvConfig.now()) adaptation ready: completed=$(length(records)) limit=$limit batchsize=$batchsize resume=$resume")
    flush(stdout)
    while length(records) < limit && !nnconverged([record.evalweighted for record in records])
        started = time()
        epoch = length(records)+1
        Flux.trainmode!(nn.model)
        # Start from canonical indices each epoch so saved RNG alone reproduces shuffle.
        active .= 1:nsamples(train)
        shuffle!(rng, active)
        chunks = _evaltrainchunks(active, batchsize)
        println("$(EnvConfig.now()) adapting epoch=$epoch/$limit batches=$(length(chunks))")
        flush(stdout)
        meter = ProgressMeter.Progress(length(chunks);dt=5,
            desc="$(EnvConfig.now()) epoch $epoch/$limit batches ",enabled=verbosity>=1)
        trainlosssum = 0.0
        trainweightmass = 0.0
        for chunk in chunks
            ordered = batchindices(train, chunk)
            x, y, w = batch(train, ordered)
            if reinforce_epochs>0
                w = _reinforcementweights(w,classindex(train,ordered),
                    reinforcement.flags[ordered],reinforcement.means)
            end
            batchloss = trainbatch!(nn, x, y, w)
            @assert isfinite(batchloss) && batchloss >= 0 "batch loss=$batchloss must be finite and nonnegative"
            weightmass = isnothing(w) ? length(ordered) : sum(w)
            trainlosssum += batchloss * weightmass
            trainweightmass += weightmass
            ProgressMeter.next!(meter)
        end
        adaptation_seconds = time() - started
        @assert trainweightmass > 0 "training weight mass=$trainweightmass must be positive"
        trainloss = trainlosssum / trainweightmass
        println("$(EnvConfig.now()) epoch=$epoch calculating held-out eval losses; training-time loss=$trainloss adaptation_seconds=$(round(adaptation_seconds;digits=3))")
        flush(stdout)
        @assert isfinite(trainloss) && trainloss >= 0 "train loss=$(trainloss) must be finite and nonnegative"
        evaluation = evallosses(nn, eval; weights)
        println("$(EnvConfig.now()) epoch=$epoch eval complete: eval_inference_seconds=$(round(evaluation.inference_seconds;digits=3)) eval_diagnostic_seconds=$(round(evaluation.diagnostic_seconds;digits=3))")
        flush(stdout)
        previousbest = isempty(records) ? Inf : minimum(record.evalweighted for record in records)
        push!(nn.losses, trainloss)
        trainloss = nn.losses[end]
        push!(records, (; epoch=length(records)+1, trainloss, evalnatural=evaluation.natural,
            evalweighted=evaluation.weighted, cells=NamedTuple[], evalcount=evaluation.count,
            traincount=nsamples(train), weightmass=evaluation.weightmass,
            maxweight=evaluation.maxweight, seconds=time()-started))
        Flux.testmode!(nn.model)
        improved = evaluation.weighted < previousbest
        improved && (best = deepcopy(nn))
        if reinforce_epochs>0 && length(records)%reinforce_epochs==0
            reinforcement = _evalreinforcement(nn,train,reinforce_epochs,
                length(records),reinforcement.history)
        end
        println("$(EnvConfig.now()) saving checkpoints: epoch=$epoch output=$folderpath")
        flush(stdout)
        checkpoint_started = time_ns()
        _saveevalcheckpoint(lastpath, (; nn, best, records, rng, reinforcement), reference)
        if improved
            _saveevalcheckpoint(bestpath, (; nn=best, records, bestepoch=epoch), reference)
        end
        checkpoint_seconds = (time_ns() - checkpoint_started) / 1e9
        println("$(EnvConfig.now()) checkpoint saved: epoch=$epoch/$limit trainloss=$trainloss trainloss_kind=training-time evalnatural=$(evaluation.natural) evalweighted=$(evaluation.weighted) bestepoch=$(argmin([record.evalweighted for record in records])) bestupdated=$improved checkpoint_seconds=$(round(checkpoint_seconds;digits=3)) elapsed=$(round(time()-started;digits=1))s")
        flush(stdout)
    end
    @assert !isnothing(best) "no completed eval checkpoint for identity=$(identity)"
    Flux.testmode!(nn.model)
    Flux.testmode!(best.model)
    stopreason = nnconverged([record.evalweighted for record in records]) ? :patience : :epochcap
    @info "$(EnvConfig.now()) eval training stopped" identity epochs=length(records) stopreason
    return (; best, last=nn, records, manifest, stopreason, reinforcement)
end
