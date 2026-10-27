using Dates, Random, Statistics, LinearAlgebra
using CSV, DataFrames, JSON3, Flux
using EnvConfig, TestOhlcv, Features, Targets, Classify, TradingStrategy

"""Build an in-memory, class-balanced source using the pipeline's exact index and label order."""
function benchmarksource(stores::AbstractDict, coins::Vector{String}, labels::AbstractVector,
    featurecols::Vector{String}, settype::String; classbalancing::Bool)
    index = Classify.buildtrainindex(coins, coin -> stores[coin], labels; settype)
    @assert nrow(index) > 0 "empty benchmark set=$(settype), coins=$(coins)"
    features = Matrix{Float32}(undef, length(featurecols), nrow(index))
    for sample in 1:nrow(index), (feature, column) in enumerate(featurecols)
        features[feature, sample] = stores[coins[index[sample, :coinix]]][1][index[sample, :row], column]
    end
    labelindices = index[!, :labelix]
    targets = Flux.onehotbatch(labelindices, eachindex(labels))
    counts = [count(==(label), labelindices) for label in eachindex(labels)]
    weights = classbalancing ? Classify.classweights(counts)[labelindices] : nothing
    return Classify.MatrixSampleSource(features, targets, weights)
end

"""Train one shuffled epoch through the shared batch API, without evaluation or disk I/O."""
function benchmarkepoch!(nn::Classify.NN, source::Classify.MatrixSampleSource,
    order::Vector{Int}, rng::AbstractRNG, batchsize::Int)
    shuffle!(rng, order)
    Flux.trainmode!(nn.model)
    for indices in Iterators.partition(order, batchsize)
        features, targets, weights = Classify.batch(source, indices)
        loss = Classify.trainbatch!(nn, features, targets, weights)
        @assert isfinite(loss) "nonfinite training loss=$(loss), model=$(nn.mnemonic)"
    end
    return nn
end

"""Measure warmed probability inference including the owned output and softmax."""
function benchmarkinference(nn::Classify.NN, features::Matrix{Float32}; repeats::Int)
    Flux.testmode!(nn.model)
    Classify.predict(nn, features)
    timings = Float64[]
    for _ in 1:7
        measurement = @timed for _ in 1:repeats
            Classify.predict(nn, features)
        end
        push!(timings, measurement.time / repeats)
    end
    return median(timings)
end

"""Return accuracy, cross entropy and entry precision/recall for a held-out pattern."""
function benchmarkquality(nn::Classify.NN, source::Classify.MatrixSampleSource, labels::AbstractVector)
    Flux.testmode!(nn.model)
    logits = nn.model(source.X)
    loss = nn.lossfunc(logits, source.Y)
    truth = map(argmax, eachcol(source.Y))
    predicted = map(argmax, eachcol(logits))
    entries = Dict{String,Any}()
    for label in (Targets.longopen, Targets.shortopen)
        labelindex = findfirst(==(label), labels)
        @assert !isnothing(labelindex) "entry label=$(label) absent from labels=$(labels)"
        positives = count(==(labelindex), predicted)
        support = count(==(labelindex), truth)
        correct = count(ix -> predicted[ix] == truth[ix] == labelindex, eachindex(truth))
        entries[string(label)] = (precision=positives == 0 ? missing : correct / positives,
            recall=support == 0 ? missing : correct / support, predictions=positives, support=support)
    end
    return (accuracy=mean(predicted .== truth), crossentropy=loss, entries=entries)
end

"""
    benchmarkmodel008(; epochs=10, trials=3, batchsize=64, seed=713, batchnorm=false)

Benchmark unchanged 059/M008, SimpleChains M009 and dense-only Flux M010 on
CS00 (SINE/DOUBLESINE), D00, F19/T24/P02/B1. Data generation, compilation,
full-set evaluation and checkpoint I/O are outside training timings. Each trial
starts afresh and uses identical seeded shuffles; dense controls also share
initial parameters. Writes measurements and metadata only under ~/crypto/debug.
With `batchnorm=true`, compare only M008 and architecture-matched native
SimpleChains BatchNorm M011 instead. M008 and M011 share initial parameters.
"""
function benchmarkmodel008(; epochs::Int=10, trials::Int=3, batchsize::Int=64, seed::Int=713,
    batchnorm::Bool=false)
    @assert epochs > 0 && trials > 0 && batchsize > 0 "epochs=$(epochs), trials=$(trials), batchsize=$(batchsize) must be positive"
    @assert !batchnorm || batchsize > 1 "native BatchNorm requires batchsize>1, got $(batchsize)"
    EnvConfig.init(EnvConfig.test)
    BLAS.set_num_threads(1)
    runfolder = mktempdir(mkpath(joinpath(EnvConfig.defaultdebugfilespath, "model008-simplechains"));
        prefix="run-", cleanup=false)
    oldcoinspath = EnvConfig.activecoinspath
    EnvConfig.setcoinspath!(joinpath(runfolder, "coins"))
    try
        config = TradingStrategy.classifierconfig(TradingStrategy.tsconfig("059").classifier)
        dates = TradingStrategy.daterangeconfig("D00")
        coins = TradingStrategy.coinsetconfig("CS00").coins
        labels = TradingStrategy.classifierlabels(config)
        featurecols = TradingStrategy.classifierfeaturecols(config)
        stores = Dict{String,Tuple}()
        for coin in coins
            ohlcv = TestOhlcv.testohlcv(coin, dates.startdt, dates.enddt)
            stores[coin] = (
                Features.updatefeatures!(ohlcv, config.features, TradingStrategy.featureconfig(config.features);
                    confighash=TradingStrategy.featureshash(config.features)),
                Targets.updatetargets!(ohlcv, config.targets, TradingStrategy.targetconfig(config.targets);
                    confighash=TradingStrategy.targetshash(config.targets)),
                Classify.updatesets!(ohlcv, "P02_D00", TradingStrategy.partitionconfig(config.partition),
                    dates.startdt, dates.enddt; confighash=TradingStrategy.setshash(config.partition, "D00")))
        end
        source = benchmarksource(stores, coins, labels, featurecols, "train"; classbalancing=true)
        heldout = Dict((coin, settype) => benchmarksource(stores, [coin], labels, featurecols,
            settype; classbalancing=false) for coin in coins for settype in ("test", "eval"))
        metadata = (julia=string(VERSION), cpu=Sys.cpu_info()[1].model,
            julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
            config="059", classifier=TradingStrategy.classifierbasename(config),
            coinset="CS00", daterange="D00", startdt=string(dates.startdt), enddt=string(dates.enddt),
            coins, featurecols, labels=string.(labels), epochs, trials, batchsize, seed, batchnorm,
            flux_version=string(pkgversion(Flux)), simplechains_version=string(pkgversion(Classify.SimpleChains)),
            training_samples=Classify.nsamples(source),
            training_class_counts=vec(sum(source.Y; dims=2)),
            heldout_samples=Dict("$(coin)-$(settype)" => Classify.nsamples(data) for ((coin, settype), data) in heldout))
        open(joinpath(runfolder, "metadata.json"), "w") do io
            JSON3.write(io, metadata)
        end
        training = DataFrame()
        inference = DataFrame()
        quality = DataFrame()
        backends = batchnorm ?
            (("Flux-M008-BatchNorm", Classify.model008), ("SimpleChains-M011-BatchNorm", Classify.model011)) :
            (("Flux-M008-BatchNorm", Classify.model008), ("SimpleChains-M009-dense", Classify.model009),
                ("Flux-M010-dense", Classify.model010))
        for (backend, factory) in backends
            # Compile the exact training and evaluation paths on disposable state.
            Random.seed!(seed)
            warm = factory(length(featurecols), labels, "benchmark-warmup")
            Flux.trainmode!(warm.model)
            warmfeatures, warmtargets, warmweights = Classify.batch(source, collect(1:batchsize))
            Classify.trainbatch!(warm, warmfeatures, warmtargets, warmweights)
            tail = rem(Classify.nsamples(source), batchsize)
            if tail > 0
                Classify.trainbatch!(warm, Classify.batch(source, collect(1:tail))...)
            end
            warmorder = collect(1:Classify.nsamples(source))
            benchmarkepoch!(warm, Classify.MatrixSampleSource(warmfeatures, warmtargets, warmweights),
                collect(1:batchsize), MersenneTwister(seed), batchsize)
            for trial in 1:trials
                Random.seed!(seed)
                nn = factory(length(featurecols), labels, "benchmark")
                order = copy(warmorder)
                rng = MersenneTwister(seed)
                GC.gc()
                for epoch in 1:epochs
                    measurement = @timed benchmarkepoch!(nn, source, order, rng, batchsize)
                    trainingloss = Classify.fullsetloss(nn, source)
                    push!(training, (backend, trial, epoch, seconds=measurement.time,
                        bytes=measurement.bytes, gcseconds=measurement.gctime, loss=trainingloss); cols=:union)
                end
                Flux.testmode!(nn.model)
                for coin in coins, settype in ("test", "eval")
                    data = heldout[(coin, settype)]
                    metrics = benchmarkquality(nn, data, labels)
                    for label in (Targets.longopen, Targets.shortopen)
                        entry = metrics.entries[string(label)]
                        push!(quality, (backend, trial, coin, settype, accuracy=metrics.accuracy,
                            crossentropy=metrics.crossentropy, label=string(label),
                            precision=entry.precision, recall=entry.recall,
                            predictions=entry.predictions, support=entry.support); cols=:union)
                    end
                end
                for coin in coins, samples in (1, 64, 8192)
                    features = heldout[(coin, "test")].X[:, 1:samples]
                    seconds = benchmarkinference(nn, features; repeats=samples == 1 ? 1000 : samples == 64 ? 200 : 10)
                    push!(inference, (; backend, trial, coin, samples, seconds); cols=:union)
                end
                println("$(backend) trial $(trial): training seconds=", sum(training[(training.backend .== backend) .& (training.trial .== trial), :seconds]))
                CSV.write(joinpath(runfolder, "training.csv"), training)
                CSV.write(joinpath(runfolder, "inference.csv"), inference)
                CSV.write(joinpath(runfolder, "quality.csv"), quality)
            end
        end
        summary = combine(groupby(combine(groupby(training, [:backend, :trial]), :seconds => sum => :seconds), :backend),
            :seconds => median => :training_seconds)
        println(summary)
        println(combine(groupby(inference, [:backend, :samples]), :seconds => median => :seconds))
        println("Benchmark results: ", runfolder)
        return (; runfolder, training, inference, quality)
    finally
        EnvConfig.setcoinspath!(oldcoinspath)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert isempty(ARGS) || ARGS == ["--batchnorm"] "expected no arguments or --batchnorm, got ARGS=$(ARGS)"
    benchmarkmodel008(; batchnorm=ARGS == ["--batchnorm"])
end
