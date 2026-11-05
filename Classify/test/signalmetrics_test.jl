using Test, Classify, DataFrames, Dates, Targets, CategoricalArrays
using Random, Statistics
include("signalmetrics_reference.jl")

"""Build shuffled multi-coin reports with gaps, interleaved cells, reused blocks and excluded first rows."""
function randomsignalfixture(seed::Int;minutes::Int=40)
    rng = MersenneTwister(seed)
    labels = [Classify.SIGNAL_LABELS...,"ignore","longstrongopen","shortstronghold"]
    df = DataFrame(coin=String[],opentime=DateTime[],set=String[],regime=String[],
        foldid=Int[],blockid=Int[],target=String[],label=String[],score=Float32[])
    for coin in ("AAA","BBB"), set in ("train","eval"), minute in 1:minutes
        rand(rng)<0.1 && continue
        push!(df,(coin,DateTime(2026)+Minute(minute),set,
            rand(rng,["up","down","unknown"]),rand(rng,1:2),rand(rng,1:3),
            rand(rng,labels),rand(rng,labels),rand(rng,Float32[0,0.5,0.8,1])))
    end
    return df[randperm(rng,nrow(df)),:]
end

@testset "Optimized reporting equals independent row-scanning oracles" begin
    for seed in 1:8
        df = randomsignalfixture(seed)
        thresholds = [0.0,0.5,0.8,1.0]
        for bootstrap in (0,11)
            expected = referencesignalmetrics(df;thresholds,bootstrap,seed)
            actual = Classify.signalmetrics(df;thresholds,bootstrap,seed)
            @test isequal(actual,expected)
        end
        @test isequal(Classify.signaleventmetrics(df;thresholds),
            referencesignalevents(df;thresholds))
        sort!(df,[:coin,:set,:opentime])
        @test isequal(Classify.signaleventmetrics(df;thresholds),
            referencesignalevents(df;thresholds))
    end
    df = randomsignalfixture(91)
    for column in (:coin,:set,:regime,:target,:label)
        df[!,column] = categorical(df[!,column])
    end
    @test isequal(Classify.signalmetrics(df;thresholds=[0.5],bootstrap=17),
        referencesignalmetrics(df;thresholds=[0.5],bootstrap=17))
    @test isequal(Classify.signaleventmetrics(df;thresholds=[0.5]),
        referencesignalevents(df;thresholds=[0.5]))
    subset = view(df,2:2:nrow(df),:)
    @test isequal(Classify.signalmetrics(subset;thresholds=[0.5],bootstrap=17),
        referencesignalmetrics(subset;thresholds=[0.5],bootstrap=17))
    @test isequal(Classify.signaleventmetrics(subset;thresholds=[0.5]),
        referencesignalevents(subset;thresholds=[0.5]))
    cutoffs = Dict(name=>0.5 for name in Classify.SIGNAL_LABELS)
    @test isequal(Classify.signalmetrics(df;thresholds=[cutoffs]),
        referencesignalmetrics(df;thresholds=[cutoffs]))
    emptydf = df[1:0,:]
    @test isempty(Classify.signalmetrics(emptydf))
    @test isempty(Classify.signaleventmetrics(emptydf))
end

@testset "Sorted event matching handles spanning and repeated episodes" begin
    for targetbits in 0:31, episodebits in 0:31
        df = DataFrame(coin=fill("AAA",5),set=fill("eval",5),regime=fill("up",5),
            foldid=ones(Int,5),blockid=ones(Int,5),opentime=DateTime(2026).+Minute.(1:5),
            target=[isodd(targetbits>>ix) ? "longhold" : "allclose" for ix in 0:4],
            label=[isodd(episodebits>>ix) ? "longhold" : "allclose" for ix in 0:4],
            score=ones(Float32,5))
        @test isequal(Classify.signaleventmetrics(df;thresholds=[0.5]),
            referencesignalevents(df;thresholds=[0.5]))
    end
end

"""Build a small wrong-direction, abstention, exclusion and missing-minute fixture."""
function signalmetricfixture()
    return DataFrame(coin=fill("SINE", 8), opentime=DateTime(2026, 1, 1) .+ Minute.([0, 1, 2, 3, 4, 6, 7, 8]),
        set=fill("eval", 8), regime=["up", "up", "up", "up", "down", "down", "unknown", "up"],
        foldid=[1, 1, 1, 1, 2, 2, 2, 1], blockid=[1, 1, 1, 1, 2, 2, 3, 1],
        target=["longopen", "longhold", "allclose", "shortopen", "shorthold", "allclose", "longopen", "ignore"],
        label=["longopen", "shortopen", "allclose", "shortopen", "shorthold", "longhold", "longopen", "longopen"],
        score=Float32[0.5, 0.9, 0.4, 0.9, 0.8, 0.7, 1, 1])
end

@testset "Reporting allocation regression on shared calendar blocks" begin
    n = 2000
    df = DataFrame(coin=fill("AAA",n),set=fill("eval",n),regime=fill("up",n),
        foldid=[isodd(ix) ? 1 : 2 for ix in 1:n],
        blockid=[div(ix-1,500)+1 for ix in 1:n],
        opentime=DateTime(2026).+Minute.(1:n),
        target=[isodd(ix) ? "longhold" : "allclose" for ix in 1:n],
        label=[ix%3==0 ? "longhold" : "allclose" for ix in 1:n],
        score=fill(0.7f0,n))
    for (optimized,reference,kwargs) in (
            (Classify.signalmetrics,referencesignalmetrics,(bootstrap=7,)),
            (Classify.signaleventmetrics,referencesignalevents,NamedTuple()))
        expected = reference(df;thresholds=[0.3,0.8],kwargs...)
        @test isequal(optimized(df;thresholds=[0.3,0.8],kwargs...),expected)
        oldbytes = @allocated reference(df;thresholds=[0.3,0.8],kwargs...)
        newbytes = @allocated optimized(df;thresholds=[0.3,0.8],kwargs...)
        @test newbytes<oldbytes/2
    end
end

@testset "Report progress callbacks preserve metrics and surface failures" begin
    predictions = signalmetricfixture()
    for (report,kwargs,phase) in (
            (Classify.signalmetrics,(bootstrap=5,seed=713),"minute metrics"),
            (Classify.signaleventmetrics,NamedTuple(),"event metrics"))
        statuses = NamedTuple[]
        expected = report(predictions;thresholds=[0.5,0.8],kwargs...)
        actual = report(predictions;thresholds=[0.5,0.8],kwargs...,
            onprogress=status -> push!(statuses,status))
        @test isequal(actual,expected)
        @test !isempty(statuses)
        @test first(statuses).phase==phase
        @test first(statuses).threshold==0.5
        @test first(statuses).groupix==1
        @test all(status -> 1<=status.groupix<=status.groups,statuses)
        @test Set(status.threshold for status in statuses)==Set([0.5,0.8])
        @test any(status -> status.signal=="longopen",statuses)
        @test_throws ErrorException report(predictions;thresholds=[0.5],kwargs...,
            onprogress=status -> error("intentional progress failure"))
        if report===Classify.signalmetrics
            @test any(status -> status.phase=="minute block counts" &&
                1<=status.blockix<=status.blocks,statuses)
        end
    end
end

@testset "Argmax decisions and minute signal metrics" begin
    @test Classify.signaldecision("longopen", 0.5, 0.5) == "longopen"
    @test ismissing(Classify.signaldecision("allclose", 0.49, 0.5))
    @test ismissing(Classify.signaldecision("ignore", 1.0, 0.5))
    @test Classify.signaldecision("longstrongopen", 0.9, 0.5) == "longopen"
    @test Classify.signaldecision([0.5, 0.5], ["longopen", "shortopen"], 0.5) == "longopen"
    @test Classify.signaldecision("longopen", 0.5, Dict("longopen"=>0.5)) == "longopen"
    @test_throws AssertionError Classify.signaldecision("longopen", NaN, 0.5)
    @test_throws AssertionError Classify.signaldecision("longopen", 0.5, 2.0)
    df = signalmetricfixture()
    metrics = Classify.signalmetrics(df; thresholds=[0.5])
    overall = metrics[metrics.grouping .== "overall", :]
    @test nrow(overall) == 7
    long = only(eachrow(overall[overall.signal .== "long", :]))
    @test (long.tp, long.fp, long.fn, long.tn) == (1, 1, 1, 3)
    @test long.precision == long.recall == 0.5
    @test long.coverage == 2/6
    @test long.eligible == 6 && long.excluded == 2 && long.abstained == 1
    close = only(eachrow(overall[overall.signal .== "allclose", :]))
    @test close.tp == 0 && close.fn == 2 && ismissing(close.precision)
    @test close.recall == 0
    for grouping in ["regime", "fold", "regime_fold"], signal in overall.signal
        grouped = metrics[(metrics.grouping .== grouping) .& (metrics.signal .== signal), :]
        pooled = only(eachrow(overall[overall.signal .== signal, :]))
        for column in [:tp, :fp, :fn, :tn, :eligible, :excluded]
            @test sum(grouped[!, column]) == pooled[column]
        end
    end
    unknown = metrics[(metrics.grouping .== "regime") .& coalesce.(metrics.regime .== "unknown", false), :]
    @test all(==(0), unknown.eligible)
    @test all(ismissing, unknown.coverage)
    # TradeLabel inputs use the same decision normalization as string inputs.
    df.label = [Targets.tradelabel(label, Targets.uniquelabels()) for label in df.label]
    df.target = [Targets.tradelabel(label, Targets.uniquelabels()) for label in df.target]
    @test isequal(Classify.signalmetrics(df; thresholds=[0.5]), metrics)
end

@testset "Shared block bootstrap confidence intervals" begin
    # Ten synchronized coins per block still provide only two independent blocks.
    df = DataFrame(coin=repeat(["coin$(ix)" for ix in 1:10]; inner=2),
        opentime=repeat([DateTime(2026, 1, 1), DateTime(2026, 1, 2)], 10),
        set=fill("eval", 20), regime=fill("up", 20), foldid=fill(1, 20),
        blockid=repeat([1, 2], 10), target=fill("longopen", 20),
        label=repeat(["longopen", "shortopen"], 10), score=ones(Float32, 20))
    report = Classify.signalmetrics(df; thresholds=[0.5], bootstrap=200, seed=713)
    @test isequal(report, Classify.signalmetrics(df; thresholds=[0.5], bootstrap=200, seed=713))
    row = only(eachrow(report[(report.grouping .== "overall") .& (report.signal .== "longopen"), :]))
    @test row.blocks == 2
    @test row.eligible == 20
    @test row.recall == row.coverage == 0.5
    @test (row.recall_lower, row.recall_upper) == (0.0, 1.0)
    @test (row.coverage_lower, row.coverage_upper) == (0.0, 1.0)
    @test (row.precision_lower, row.precision_upper) == (1.0, 1.0)
    @test 0 < row.precision_bootstrap_valid < 200
    @test row.recall_bootstrap_valid == row.coverage_bootstrap_valid == 200
    # Replicating an entire coin universe cannot narrow block-based intervals.
    replicated = vcat(df, transform(df, :coin => ByRow(coin -> coin * "-copy") => :coin))
    doubled = Classify.signalmetrics(replicated; thresholds=[0.5], bootstrap=200, seed=713)
    other = only(eachrow(doubled[(doubled.grouping .== "overall") .& (doubled.signal .== "longopen"), :]))
    @test other.blocks == 2
    @test (other.recall_lower, other.recall_upper) == (row.recall_lower, row.recall_upper)
    unsupported = Classify.signalmetrics(df; thresholds=[0.5], bootstrap=200, minblocks=3)
    @test all(ismissing, unsupported.recall_lower)
    @test all(unsupported.insufficient_support)
    disabled = Classify.signalmetrics(df; thresholds=[0.5])
    @test all(ismissing, disabled.precision_lower)
    @test all(==(0), disabled.precision_bootstrap_valid)
    @test_throws AssertionError Classify.signalmetrics(df; bootstrap=-1)
    never = report[(report.grouping .== "overall") .& (report.signal .== "allclose"), :]
    @test all(ismissing, never.precision_lower) && all(ismissing, never.recall_lower)
    @test all(==(0.0), never.coverage_upper)
end

@testset "Exact target opportunities, duplicate episodes and boundaries" begin
    df = DataFrame(coin=fill("SINE", 7), opentime=DateTime(2026, 1, 1) .+ Minute.([0, 1, 2, 3, 4, 6, 7]),
        set=fill("eval", 7), regime=fill("up", 7), foldid=fill(1, 7),
        blockid=[1, 1, 1, 1, 2, 2, 2], target=fill("longhold", 7),
        label=fill("longhold", 7), score=Float32[1, 0.1, 1, 1, 1, 1, 1])
    result = Classify.signaleventmetrics(df; thresholds=[0.5])
    row = only(eachrow(result[(result.grouping .== "overall") .& (result.signal .== "longhold"), :]))
    @test row.opportunities == 3 # block split and missing minute split
    @test row.episodes == 4
    @test (row.tp, row.fp, row.fn) == (2, 2, 1)
    @test row.detected == 3 && row.duplicate_episodes == 1
    @test row.repeated_signal_minutes == 2
    @test row.mean_delay_minutes == 0
    @test row.detection_rate == 1
    @test row.precision == 0.5 && row.recall == 2/3
    @test row.blocks == 2 && !row.insufficient_support
    @test_throws AssertionError Classify.signaleventmetrics(vcat(df, df[1:1, :]))
    df.regime[3] = "unknown"
    excluded = Classify.signaleventmetrics(df; thresholds=[0.5])
    row = only(eachrow(excluded[(excluded.grouping .== "overall") .& (excluded.signal .== "longhold"), :]))
    @test row.eligible == 6 && row.excluded == 1
    @test row.opportunities == 4 # excluded minute cannot bridge intervals
    bridge = DataFrame(coin=fill("SINE", 3), opentime=DateTime(2026, 1, 1) .+ Minute.(0:2),
        set=fill("eval", 3), regime=fill("up", 3), foldid=fill(1, 3), blockid=fill(1, 3),
        target=["longhold", "allclose", "longhold"], label=fill("longhold", 3), score=ones(Float32, 3))
    bridgemetrics = Classify.signaleventmetrics(bridge; thresholds=[0.5])
    row = only(eachrow(bridgemetrics[(bridgemetrics.grouping .== "overall") .& (bridgemetrics.signal .== "longhold"), :]))
    @test row.opportunities == 2 && row.episodes == 1 && row.detected == 1
    @test row.tp == 0 && row.detection_rate == 0.5
end

@testset "Extended confusion true one-versus-rest parity" begin
    labels = Targets.uniquelabels()
    df = signalmetricfixture()
    df = df[1:6, :]
    df.set = categorical(df.set)
    result = Classify.extendedconfusionmatrix(df, labels, 10)
    for row in eachrow(result)
        bin = parse(Int, first(split(string(row.bin), "/")))
        positive = df.target .== string(row.pred_label)
        issued = (df.label .== string(row.pred_label)) .& (Classify.score2bin.(df.score, 10) .>= bin)
        @test row.tp == count(positive .& issued)
        @test row.fp == count(.!positive .& issued)
        @test row.fn == count(positive .& .!issued)
        @test row.tn == count(.!positive .& .!issued)
    end
end
