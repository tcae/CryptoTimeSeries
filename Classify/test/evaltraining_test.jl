using Test, Random, Flux, Classify, DataFrames
import EnvConfig

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
    @test Classify._evaltrainchunks(collect(1:5), 2) == [[1, 2], [3, 4, 5]]
    @test Classify._evaltrainchunks(collect(1:4), 2) == [[1, 2], [3, 4]]
    nn, _, eval = evaltrainingfixture()
    Flux.trainmode!(nn.model)
    layers = filter(layer -> layer isa Classify.SimpleChainsBatchNorm, nn.model.chain.layers)
    before = deepcopy(layers)
    parameters = copy(nn.model.parameters)
    result = Classify.evallosses(nn, eval; weights, regimes, folds, chunksize=1)
    @test result.count == 4
    @test result.weightmass == 1
    @test result.maxweight == 0.5
    @test result.natural ≈ log(2)
    @test result.weighted ≈ log(2)
    @test sum(cell.count for cell in result.cells) == 4
    @test nn.model.parameters == parameters
    @test all(layer.training for layer in layers)
    @test all(pair -> pair[1].mean == pair[2].mean && pair[1].variance == pair[2].variance, zip(layers, before))
    # Unequal label losses distinguish natural/fixed weighting from source class weights.
    nn.model.parameters[end] = 2f0
    weighted = Classify.evallosses(nn, eval; weights, regimes, folds, chunksize=1)
    @test weighted.natural != weighted.weighted
    @test weighted.natural ≈ (3log1p(exp(2.0)) + log1p(exp(-2.0))) / 4 rtol=1e-6
    @test weighted.weighted ≈ (log1p(exp(2.0)) + log1p(exp(-2.0))) / 2 rtol=1e-6
    ignorednn = Classify.model011(2, ["ignore", "longopen"], "ignored")
    ignoredsrc = Classify.MatrixSampleSource(ignorednn, zeros(Float32, 2, 2), ["ignore", "longopen"])
    @test_throws AssertionError Classify.evallosses(ignorednn, ignoredsrc;
        weights=ones(Float32, 2), regimes=["up", "up"], folds=[1, 1])
    nn.valuecorrection = x -> error("intentional evaluation failure")
    @test_throws ErrorException Classify.evallosses(nn, eval; weights, regimes, folds)
    @test all(layer.training for layer in layers)
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
