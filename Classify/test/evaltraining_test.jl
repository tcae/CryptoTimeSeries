using Test, Random, Flux, Classify, DataFrames
import EnvConfig
import BSON

"""Build a deterministic small M011 and disjoint train/eval sources."""
function evaltrainingfixture(; frozen::Bool=false)
    Random.seed!(871)
    nn = Classify.model011(2, ["longopen", "allclose"], "eval-test")
    frozen && (nn.optim = Flux.setup(Flux.Adam(0.0), nn.model.parameters))
    train = Classify.MatrixSampleSource(nn, zeros(Float32, 2, 5),
        ["longopen", "longopen", "allclose", "allclose", "allclose"])
    eval = Classify.MatrixSampleSource(nn, zeros(Float32, 2, 4),
        ["longopen", "longopen", "longopen", "allclose"], Float32[100, 1, 1, 1])
    return nn, train, eval
end

@testset "Compact lazy epoch indices" begin
    for n in (0, 2, 4, 5, 6, 7, 8), batchsize in (2, 3, 4)
        active = collect(Classify._sampleindices(n))
        chunks = Classify._evaltrainchunks(active, batchsize)
        materialized = collect(chunks)
        @test length(chunks) == length(materialized)
        @test all(chunk -> chunk isa SubArray && parent(chunk) === active, materialized)
        @test reduce(vcat, materialized; init=UInt32[]) == active
        @test all(chunk -> length(chunk) >= 2, materialized)
    end
    active = collect(Classify._sampleindices(1_000_000))
    @test sizeof(active) == 4_000_000
    Classify._evaltrainchunks(active, 64)
    allocated = @allocated Classify._evaltrainchunks(active, 64)
    @test allocated < 4096
    chunks = Classify._evaltrainchunks(active, 64)
    @test length(chunks) == 15_625
    @test Base.summarysize(chunks) < sizeof(active) + 4096
    @test_throws AssertionError Classify._evaltrainchunks(active, 1)
    oldshuffle = shuffle!(MersenneTwister(1), collect(1:1000))
    compactshuffle = shuffle!(MersenneTwister(1), collect(Classify._sampleindices(1000)))
    @test compactshuffle == oldshuffle
end

@testset "Compact checkpoint references and existing checkpoint resume" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        nn, train, eval = evaltrainingfixture(; frozen=true)
        initial = Classify.traineval!(nn, train, eval; identity="compact",
            folderpath=root, maxepochs=1, batchsize=2)
        @test initial.manifest.version == 2
        @test !hasproperty(initial.manifest, :regimes)
        @test !hasproperty(initial.manifest, :folds)
        @test isempty(initial.records[1].cells)
        for name in ("eval-last.bson", "eval-best.bson")
            raw = BSON.load(joinpath(root, name), Classify)[:payload]
            @test !hasproperty(raw, :manifest)
            @test length(raw.manifestreference) == 64
            @test Classify._loadevalpayload(joinpath(root, name)).manifest == initial.manifest
        end
        mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
            nn, train, eval = evaltrainingfixture(; frozen=true)
            initial = Classify.traineval!(nn, train, eval; identity="v1-resume",
                folderpath=root, maxepochs=1, batchsize=2)
            oldmanifest = merge(initial.manifest,
                (; version=1, regimes=fill("overall", 4), folds=zeros(Int, 4)))
            for name in ("eval-last.bson", "eval-best.bson")
                path = joinpath(root, name)
                payload = Classify._loadevalpayload(path)
                embedded = (; (key => value for (key, value) in pairs(payload)
                    if key != :manifestreference)...)
                Classify._atomicevalsave(path, merge(embedded, (; manifest=oldmanifest)))
            end
            Classify._atomicevalsave(joinpath(root, "eval-manifest.bson"), oldmanifest)
            fresh, _, _ = evaltrainingfixture(; frozen=true)
            resumed = Classify.traineval!(fresh, train, eval; identity="v1-resume",
                folderpath=root, maxepochs=2, batchsize=2)
            @test length(resumed.records) == 2
            @test resumed.manifest == initial.manifest
            @test !hasproperty(resumed.manifest, :regimes)
            @test Classify.loadtraincheckpoint(root; identity="v1-resume").model.parameters ==
                initial.best.model.parameters
        end
        path = joinpath(root, "eval-last.bson")
        payload = Classify._loadevalpayload(path)
        # Existing embedded-manifest checkpoints must resume into compact writes.
        embedded = (; (key => value for (key, value) in pairs(payload)
            if key != :manifestreference)...)
        Classify._atomicevalsave(path, embedded)
        fresh, _, _ = evaltrainingfixture(; frozen=true)
        resumed = Classify.traineval!(fresh, train, eval; identity="compact",
            folderpath=root, maxepochs=2, batchsize=2)
        @test length(resumed.records) == 2
        @test !hasproperty(BSON.load(path, Classify)[:payload], :manifest)
        @test Classify.loadtraincheckpoint(root; identity="compact").model.parameters ==
            initial.best.model.parameters
        raw = BSON.load(path, Classify)[:payload]
        Classify._atomicevalsave(path, merge(raw, (; manifestreference="wrong")))
        @test_throws AssertionError Classify.loadtraincheckpoint(root; identity="compact")
        Classify._atomicevalsave(path, raw)
        manifestpath = joinpath(root, "eval-manifest.bson")
        Classify._atomicevalsave(manifestpath,
            merge(initial.manifest, (; weights=reverse(initial.manifest.weights) .* 2)))
        @test_throws AssertionError Classify.loadtraincheckpoint(root; identity="compact")
    end
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        manifestpath = joinpath(root, "eval-manifest.bson")
        manifest = (; identity="large", weights=ones(Float32, 500_000),
            regimes=fill("overall", 500_000), folds=zeros(Int, 500_000))
        Classify._atomicevalsave(manifestpath, manifest)
        reference = Classify._evalmanifestreference(manifestpath)
        path = joinpath(root, "compact.bson")
        Classify._saveevalcheckpoint(path, (; manifest, epoch=1), reference)
        @test filesize(path) < 1024
        @test filesize(manifestpath) > 1_000_000
        @test Classify._loadevalpayload(path).manifest == manifest
    end
end

"""Record training batch reads to detect unintended full-training-set inference."""
struct EvalTrainingReadSource <: Classify.AbstractSampleSource
    source::Classify.MatrixSampleSource
    reads::Vector{Vector{Int}}
end

"""Return the wrapped training sample count."""
Classify.nsamples(src::EvalTrainingReadSource) = Classify.nsamples(src.source)

"""Read a training batch and retain its indices for pass-count assertions."""
function Classify.batch(src::EvalTrainingReadSource, indices::AbstractVector{<:Integer})
    push!(src.reads, collect(indices))
    return Classify.batch(src.source, indices)
end

"""Return wrapped target indices without reading training features."""
Classify.classindex(src::EvalTrainingReadSource, indices::AbstractVector{<:Integer}) =
    Classify.classindex(src.source, indices)

@testset "Training-time loss reuses batches with effective mass weighting" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        for weighted in (false, true)
            nn, train, eval = evaltrainingfixture()
            weights = weighted ? Float32[1, 3, 2, 4, 8] : nothing
            train = Classify.MatrixSampleSource(train.X, train.Y, weights)
            tracked = EvalTrainingReadSource(train, Vector{Int}[])
            reference = deepcopy(nn)
            active = shuffle!(MersenneTwister(1), collect(1:Classify.nsamples(train)))
            chunks = Classify._evaltrainchunks(active, 2)
            Flux.trainmode!(reference.model)
            losses = Float64[]
            masses = Float64[]
            for chunk in chunks
                x, y, w = Classify.batch(train, chunk)
                push!(losses, Classify.trainbatch!(reference, x, y, w))
                push!(masses, isnothing(w) ? length(chunk) : sum(w))
            end
            result = Classify.traineval!(nn, tracked, eval; identity="batch-loss",
                folderpath=joinpath(root, string(weighted)), maxepochs=1, batchsize=2)
            expected = sum(losses .* masses) / sum(masses)
            @test isapprox(result.records[1].trainloss, expected; rtol=1e-6)
            @test result.last.losses == [result.records[1].trainloss]
            @test tracked.reads == collect(chunks)
            @test length.(tracked.reads) == [2, 3]
            @test result.last.model.parameters == reference.model.parameters
            @test result.records[1].trainloss != Classify.fullsetloss(reference, train)
            evaluation = Classify.evallosses(reference, eval;
                weights=fill(0.25f0, 4))
            @test result.records[1].evalweighted == evaluation.weighted
            @test isfinite(evaluation.inference_seconds) && evaluation.inference_seconds >= 0
            @test isfinite(evaluation.diagnostic_seconds) && evaluation.diagnostic_seconds >= 0
        end

        nn, train, eval = evaltrainingfixture()
        train = Classify.MatrixSampleSource(train.X, train.Y, Float32[1, 3, 2, 4, 8])
        reference = deepcopy(nn)
        rng = MersenneTwister(1)
        reinforcement = nothing
        expectedlosses = Float64[]
        for epoch in 1:2
            active = shuffle!(rng, collect(1:Classify.nsamples(train)))
            Flux.trainmode!(reference.model)
            losses, masses = Float64[], Float64[]
            for chunk in Classify._evaltrainchunks(active, 2)
                x, y, w = Classify.batch(train, chunk)
                if epoch > 1
                    w = Classify._reinforcementweights(w, Classify.classindex(train, chunk),
                        reinforcement.flags[chunk], reinforcement.means)
                end
                push!(losses, Classify.trainbatch!(reference, x, y, w))
                push!(masses, sum(w))
            end
            push!(expectedlosses, sum(losses .* masses) / sum(masses))
            Flux.testmode!(reference.model)
            reinforcement = Classify._evalreinforcement(reference, train, 1, epoch, NamedTuple[])
        end
        result = Classify.traineval!(nn, train, eval; identity="reinforced-batch-loss",
            folderpath=joinpath(root, "reinforced"), maxepochs=2, batchsize=2, reinforce_epochs=1)
        @test isapprox(result.last.losses, expectedlosses; rtol=1e-6)
        @test result.last.model.parameters == reference.model.parameters
    end
end

@testset "Timestamped adaptation and checkpoint progress" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        nn,train,eval = evaltrainingfixture()
        path = joinpath(folder,"progress.log")
        result = open(path,"w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    Classify.traineval!(nn,train,eval;identity="progress",
                        folderpath=folder,maxepochs=1,batchsize=2)
                end
            end
        end
        output = read(path,String)
        @test occursin(r"\d{6} \d{2}:\d{2} validating train/eval populations",output)
        @test occursin("adaptation ready: completed=0 limit=1",output)
        @test occursin("adapting epoch=1/1",output)
        @test occursin("epoch=1/1 batches=2",output)
        @test occursin("calculating held-out eval losses",output)
        @test !occursin("calculating full train",output)
        @test occursin("trainloss_kind=training-time",output)
        for field in ("adaptation_seconds", "eval_inference_seconds",
                "eval_diagnostic_seconds", "checkpoint_seconds")
            timing = match(Regex(field * "=([0-9.]+)"), output)
            @test !isnothing(timing)
            @test parse(Float64, timing.captures[1]) >= 0
        end
        @test occursin("saving checkpoints: epoch=1",output)
        @test occursin("checkpoint saved: epoch=1/1",output)
        @test occursin("evalweighted=",output)
        @test occursin("bestupdated=true",output)
        @test length(result.records)==1
        @test isfile(joinpath(folder,"eval-best.bson"))
    end
end

@testset "Best checkpoint writes only on improvement or repair" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        folder = joinpath(root, "unchanged")
        frozen, train, eval = evaltrainingfixture(; frozen=true)
        initial = Classify.traineval!(frozen, train, eval; identity="unchanged",
            folderpath=folder, maxepochs=1, batchsize=2)
        bestpath = joinpath(folder, "eval-best.bson")
        original = read(bestpath)
        originalstat = stat(bestpath)
        frozen, _, _ = evaltrainingfixture(; frozen=true)
        resumed = Classify.traineval!(frozen, train, eval; identity="unchanged",
            folderpath=folder, maxepochs=3, batchsize=2)
        @test all(record.evalweighted == initial.records[1].evalweighted for record in resumed.records)
        @test length(resumed.records) == 3
        @test length(Classify._loadevalpayload(joinpath(folder, "eval-last.bson")).records) == 3
        @test length(Classify._loadevalpayload(bestpath).records) == 1
        @test read(bestpath) == original
        @test stat(bestpath).inode == originalstat.inode
        @test mtime(bestpath) == originalstat.mtime

        # Neither inference nor a no-op resume may replace a current best file.
        loaded = Classify.loadtraincheckpoint(folder; identity="unchanged")
        @test loaded.model.parameters == initial.best.model.parameters
        frozen, _, _ = evaltrainingfixture(; frozen=true)
        Classify.traineval!(frozen, train, eval; identity="unchanged",
            folderpath=folder, maxepochs=3, batchsize=2)
        @test read(bestpath) == original
        @test stat(bestpath).inode == originalstat.inode
        @test mtime(bestpath) == originalstat.mtime

        for repair in (:load, :resume), damage in (:missing, :stale)
            if damage == :missing
                rm(bestpath)
            else
                payload = Classify._loadevalpayload(bestpath)
                Classify._atomicevalsave(bestpath, merge(payload, (; bestepoch=0)))
            end
            if repair == :load
                repaired = Classify.loadtraincheckpoint(folder; identity="unchanged")
            else
                frozen, _, _ = evaltrainingfixture(; frozen=true)
                result = Classify.traineval!(frozen, train, eval; identity="unchanged",
                    folderpath=folder, maxepochs=3, batchsize=2)
                @test length(result.records) == 3
                repaired = result.best
            end
            @test repaired.model.parameters == initial.best.model.parameters
            payload = Classify._loadevalpayload(bestpath)
            @test payload.bestepoch == 1
            @test length(payload.records) == 1
            @test payload.nn.model.parameters == initial.best.model.parameters
            @test !isfile(bestpath * ".staging")
        end

        folder = joinpath(root, "improving")
        nn, _, _ = evaltrainingfixture()
        train = Classify.MatrixSampleSource(nn, zeros(Float32, 2, 5), fill("allclose", 5))
        eval = Classify.MatrixSampleSource(nn, zeros(Float32, 2, 4), fill("allclose", 4))
        firstepoch = Classify.traineval!(nn, train, eval; identity="improving",
            folderpath=folder, maxepochs=1, batchsize=5)
        bestpath = joinpath(folder, "eval-best.bson")
        stale = Classify._loadevalpayload(bestpath)
        fresh, _, _ = evaltrainingfixture()
        improved = Classify.traineval!(fresh, train, eval; identity="improving",
            folderpath=folder, maxepochs=2, batchsize=5)
        @test improved.records[2].evalweighted < firstepoch.records[1].evalweighted
        payload = Classify._loadevalpayload(bestpath)
        @test payload.bestepoch == 2
        @test length(payload.records) == 2
        @test payload.nn.model.parameters == improved.best.model.parameters
        @test payload.nn.model.parameters != stale.nn.model.parameters

        # Simulate interruption after committing last but before replacing best.
        Classify._atomicevalsave(bestpath, stale)
        repaired = Classify.loadtraincheckpoint(folder; identity="improving")
        @test repaired.model.parameters == improved.best.model.parameters
        @test Classify._loadevalpayload(bestpath).bestepoch == 2
        Classify._atomicevalsave(bestpath, stale)
        fresh, _, _ = evaltrainingfixture()
        Classify.traineval!(fresh, train, eval; identity="improving",
            folderpath=folder, maxepochs=2, batchsize=5)
        @test Classify._loadevalpayload(bestpath).bestepoch == 2
    end
end

@testset "Train-only reinforcement and exact checkpoint resume" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do root
        nn,train,eval = evaltrainingfixture()
        train = Classify.MatrixSampleSource(nn,zeros(Float32,2,5),
            ["longopen","longopen","allclose","allclose","allclose"],Float32[1,3,2,4,8])
        initial = Classify.traineval!(nn,train,eval;identity="reinforced",
            folderpath=joinpath(root,"resume"),maxepochs=2,batchsize=2,reinforce_epochs=2)
        @test initial.reinforcement.interval==2
        @test [row.epoch for row in initial.reinforcement.history]==[2]
        @test initial.reinforcement.flags==Classify.misclassificationflags(initial.last,train)
        labels = Classify.classindex(train,collect(1:Classify.nsamples(train)))
        _,_,baseweights = Classify.batch(train,collect(1:Classify.nsamples(train)))
        actual = Classify._reinforcementweights(baseweights,labels,
            initial.reinforcement.flags,initial.reinforcement.means)
        for label in unique(labels)
            selected = labels .== label
            @test sum(actual[selected]) ≈ (isnothing(baseweights) ? count(selected) : sum(baseweights[selected]))
        end
        fresh,_,_ = evaltrainingfixture()
        resumed = Classify.traineval!(fresh,train,eval;identity="reinforced",
            folderpath=joinpath(root,"resume"),maxepochs=4,batchsize=2,reinforce_epochs=2)
        fresh,_,_ = evaltrainingfixture()
        continuous = Classify.traineval!(fresh,train,eval;identity="reinforced",
            folderpath=joinpath(root,"continuous"),maxepochs=4,batchsize=2,reinforce_epochs=2)
        @test resumed.last.model.parameters==continuous.last.model.parameters
        @test resumed.last.losses==continuous.last.losses
        @test [r.evalweighted for r in resumed.records]==[r.evalweighted for r in continuous.records]
        @test resumed.reinforcement==continuous.reinforcement
        @test resumed.manifest.weights==initial.manifest.weights
        @test resumed.manifest.evalcount==4
        frozen,train,eval = evaltrainingfixture(;frozen=true)
        plain = Classify.traineval!(frozen,train,eval;identity="enable",
            folderpath=joinpath(root,"enable"),maxepochs=2,batchsize=2)
        frozen,_,_ = evaltrainingfixture(;frozen=true)
        enabled = Classify.traineval!(frozen,train,eval;identity="enable",
            folderpath=joinpath(root,"enable"),maxepochs=1000,batchsize=2,reinforce_epochs=3)
        @test enabled.records[1:2]==plain.records
        @test enabled.manifest==plain.manifest
        @test length(enabled.records)==51
        @test enabled.stopreason==:patience
        @test first(enabled.reinforcement.history).epoch==2
        @test all(row->row.epoch==2 || row.epoch%3==0,enabled.reinforcement.history)
        frozen,_,_ = evaltrainingfixture(;frozen=true)
        disabled = Classify.traineval!(frozen,train,eval;identity="enable",
            folderpath=joinpath(root,"enable"),maxepochs=200,batchsize=2)
        @test length(disabled.records)==51
        @test disabled.stopreason==:patience
        @test disabled.reinforcement.interval==0
        @test_throws AssertionError Classify.traineval!(frozen,train,eval;identity="invalid",
            folderpath=joinpath(root,"invalid"),reinforce_epochs=-1)
        payload = Classify._loadevalpayload(joinpath(root,"enable","eval-last.bson"))
        records = [merge(first(payload.records),(;epoch,evalweighted=2.0-epoch*0.001)) for epoch in 1:199]
        payload.nn.losses = fill(first(payload.nn.losses),199)
        Classify._atomicevalsave(joinpath(root,"enable","eval-last.bson"),merge(payload,(;records,)))
        frozen,_,_ = evaltrainingfixture(;frozen=true)
        capped = Classify.traineval!(frozen,train,eval;identity="enable",
            folderpath=joinpath(root,"enable"),maxepochs=1000,batchsize=2,reinforce_epochs=2)
        @test length(capped.records)==200
        @test capped.stopreason==:epochcap
    end
end

@testset "Fixed eval cell weights and inference isolation" begin
    regimes = ["up", "up", "up", "down"]
    folds = [1, 1, 2, 1]
    weights = Classify.fixedevalweights(regimes, folds)
    @test weights == Float32[0.125, 0.125, 0.25, 0.5]
    @test_throws AssertionError Classify.fixedevalweights(["unknown"], [1])
    @test_throws AssertionError Classify.fixedevalweights(String[], Int[])
    @test collect(Classify._evaltrainchunks(collect(1:5), 2)) == [[1, 2], [3, 4, 5]]
    @test collect(Classify._evaltrainchunks(collect(1:4), 2)) == [[1, 2], [3, 4]]
    nn, _, eval = evaltrainingfixture()
    Flux.trainmode!(nn.model)
    layers = filter(layer -> layer isa Classify.SimpleChainsBatchNorm, nn.model.chain.layers)
    before = deepcopy(layers)
    parameters = copy(nn.model.parameters)
    result = Classify.evallosses(nn, eval; weights, chunksize=1)
    @test result.count == 4
    @test result.weightmass == 1
    @test result.maxweight == 0.5
    @test result.natural ≈ log(2)
    @test result.weighted ≈ log(2)
    @test !hasproperty(result, :cells)
    @test nn.model.parameters == parameters
    @test all(layer.training for layer in layers)
    @test all(pair -> pair[1].mean == pair[2].mean && pair[1].variance == pair[2].variance, zip(layers, before))
    # Unequal label losses distinguish natural/fixed weighting from source class weights.
    nn.model.parameters[end] = 2f0
    weighted = Classify.evallosses(nn, eval; weights, chunksize=1)
    @test weighted.natural != weighted.weighted
    @test weighted.natural ≈ (3log1p(exp(2.0)) + log1p(exp(-2.0))) / 4 rtol=1e-6
    @test weighted.weighted ≈ (log1p(exp(2.0)) + log1p(exp(-2.0))) / 2 rtol=1e-6
    ignorednn = Classify.model011(2, ["ignore", "longopen"], "ignored")
    ignoredsrc = Classify.MatrixSampleSource(ignorednn, zeros(Float32, 2, 2), ["ignore", "longopen"])
    @test_throws AssertionError Classify.evallosses(ignorednn, ignoredsrc;
        weights=ones(Float32, 2))
    nn.valuecorrection = x -> error("intentional evaluation failure")
    @test_throws ErrorException Classify.evallosses(nn, eval; weights)
    @test all(layer.training for layer in layers)
end

@testset "Streaming eval diagnostics match population reductions" begin
    nn, _, _ = evaltrainingfixture()
    rng = MersenneTwister(951)
    n = 101
    regimes = rand(rng, ["up", "flat", "down"], n)
    folds = rand(rng, 1:5, n)
    weights = Classify.fixedevalweights(regimes, folds)
    represented = unique(regimes)
    expectedweights = [1 / (length(represented) *
        length(unique(folds[regimes .== regimes[index]])) *
        count((regimes .== regimes[index]) .& (folds .== folds[index]))) for index in 1:n]
    @test isapprox(weights, expectedweights; rtol=1e-6)
    src = Classify.MatrixSampleSource(nn, randn(rng, Float32, 2, n),
        rand(rng, ["longopen", "allclose"], n))
    Flux.testmode!(nn.model)
    x, y, _ = Classify.batch(src, 1:n)
    losses = Classify.samplelosses(nn, nn.model(x), y) .+ 0.0
    for chunksize in (1, 7, 64, 200)
        actual = Classify.evallosses(nn, src; weights, chunksize)
        @test isapprox(actual.natural, sum(losses)/n; rtol=1e-6)
        @test isapprox(actual.weighted, sum(losses .* weights)/sum(weights); rtol=1e-6)
        @test !hasproperty(actual, :cells)
    end
end

@testset "Eval checkpoints, exact resume and stopping" begin
    # All fixture files remain package-local and are removed after the test.
    root = joinpath(@__DIR__, "eval-training-fixture-" * string(getpid()))
    @assert !ispath(root) "fixture root=$(root) must not already exist"
    mkpath(root)
    try
        nn, train, eval = evaltrainingfixture()
        initial = Classify.traineval!(nn, train, eval; identity="memberships-v1",
            folderpath=joinpath(root, "resumed"), maxepochs=2, batchsize=2)
        @test length(initial.records) == 2
        @test length(initial.last.losses) == 2
        @test initial.last.losses == [record.trainloss for record in initial.records]
        @test initial.records[1].evalnatural != initial.records[1].trainloss
        @test initial.best.losses[end] == initial.records[argmin([r.evalweighted for r in initial.records])].trainloss
        @test all(isfile(joinpath(root, "resumed", name)) for name in
            ("eval-manifest.bson", "eval-last.bson", "eval-best.bson"))
        # A stale/missing best file is repaired from authoritative last state.
        rm(joinpath(root, "resumed", "eval-best.bson"))
        fresh, _, _ = evaltrainingfixture()
        resumed = Classify.traineval!(fresh, train, eval; identity="memberships-v1",
            folderpath=joinpath(root, "resumed"), maxepochs=4, batchsize=2)
        fresh, _, _ = evaltrainingfixture()
        continuous = Classify.traineval!(fresh, train, eval; identity="memberships-v1",
            folderpath=joinpath(root, "continuous"), maxepochs=4, batchsize=2)
        @test resumed.last.model.parameters == continuous.last.model.parameters
        @test resumed.last.losses == continuous.last.losses
        @test [r.evalweighted for r in resumed.records] == [r.evalweighted for r in continuous.records]
        @test resumed.best.model.parameters == continuous.best.model.parameters
        @test resumed.stopreason == continuous.stopreason == :epochcap
        folder = joinpath(root, "resumed")
        loadedbest = Classify.loadtraincheckpoint(folder; identity="memberships-v1")
        loadedlast = Classify.loadtraincheckpoint(folder; identity="memberships-v1", best=false)
        @test loadedbest.model.parameters == resumed.best.model.parameters
        @test loadedlast.model.parameters == resumed.last.model.parameters
        @test all(layer -> !(layer isa Classify.SimpleChainsBatchNorm) || !layer.training,
            loadedbest.model.chain.layers)
        @test_throws AssertionError Classify.loadtraincheckpoint(folder; identity="wrong")
        bestpath = joinpath(folder, "eval-best.bson")
        bestpayload = Classify._loadevalpayload(bestpath)
        Classify._atomicevalsave(bestpath, merge(bestpayload,
            (; manifest=merge(bestpayload.manifest, (; identity="wrong")))))
        @test_throws AssertionError Classify.loadtraincheckpoint(folder; identity="memberships-v1")
        rm(bestpath)
        @test Classify.loadtraincheckpoint(folder; identity="memberships-v1").model.parameters ==
            resumed.best.model.parameters
        @test isfile(bestpath)
        @test !any(endswith(".staging"), readdir(joinpath(root, "resumed")))
        fresh, _, _ = evaltrainingfixture()
        @test_throws AssertionError Classify.traineval!(fresh, train, eval; identity="changed",
            folderpath=joinpath(root, "resumed"), maxepochs=4, batchsize=2)
        @test_throws AssertionError Classify.traineval!(fresh, train, eval; identity="memberships-v1",
            folderpath=joinpath(root, "resumed"), maxepochs=4, batchsize=2, evalweights=ones(Float32, 4))
        @test_throws AssertionError Classify.traineval!(fresh, train, eval; identity="memberships-v1",
            folderpath=joinpath(root, "resumed"), maxepochs=4, batchsize=2, resume=false)
        opposing, _, _ = evaltrainingfixture()
        opposetrain = Classify.MatrixSampleSource(opposing, zeros(Float32, 2, 5), fill("allclose", 5))
        opposeval = Classify.MatrixSampleSource(opposing, zeros(Float32, 2, 4), fill("longopen", 4))
        divergent = Classify.traineval!(opposing, opposetrain, opposeval; identity="opposed",
            folderpath=joinpath(root, "opposed"), maxepochs=3, batchsize=5)
        @test argmin(divergent.last.losses) == 3
        @test argmin([record.evalweighted for record in divergent.records]) == 1
        @test length(divergent.best.losses) == 1
        frozen, train, eval = evaltrainingfixture(; frozen=true)
        patience = Classify.traineval!(frozen, train, eval; identity="patience",
            folderpath=joinpath(root, "patience"), maxepochs=1000, batchsize=2)
        @test length(patience.records) == 51
        @test patience.stopreason == :patience
        @test length(patience.best.losses) == 1
        @test !Classify.nnconverged(fill(1.0, 50))
        @test Classify.nnconverged(fill(1.0, 51))
        @test !Classify.nnconverged([1.0, 1.0-0.0005]; patience=1)
        @test Classify.nnconverged([1.0, 1.0-0.00049]; patience=1)
        # Resume at epoch 199, with a nonconverged eval series, exercises the total cap.
        path = joinpath(root, "patience", "eval-last.bson")
        payload = Classify._loadevalpayload(path)
        records = [merge(first(payload.records), (; epoch, evalweighted=2.0-epoch*0.001)) for epoch in 1:199]
        payload.nn.losses = fill(first(payload.nn.losses), 199)
        Classify._atomicevalsave(path, merge(payload, (; records)))
        frozen, _, _ = evaltrainingfixture(; frozen=true)
        capped = Classify.traineval!(frozen, train, eval; identity="patience",
            folderpath=joinpath(root, "patience"), maxepochs=1000, batchsize=2)
        @test length(capped.records) == 200
        @test capped.stopreason == :epochcap
        frozen, _, _ = evaltrainingfixture(; frozen=true)
        stopped = Classify.traineval!(frozen, train, eval; identity="patience",
            folderpath=joinpath(root, "patience"), maxepochs=200, batchsize=2)
        @test length(stopped.records) == 200
    finally
        rm(root; recursive=true)
    end
end
