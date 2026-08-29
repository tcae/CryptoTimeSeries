using Test
using Dates
using DataFrames
using Targets
using EnvConfig
using Xch
using TestOhlcv
using TradingStrategy
using TSM

# TrendDetector calls preparereplaytrades! with a `groupby` SubDataFrame of the shared
# multi-coin results table. Building the Trades frame with copycols=false made it alias
# that table, so the replay wrote :label (and :score on the truth pass) back into the
# predictions. No test exercised a SubDataFrame input, which is how that reached a run.

"""Build a multi-coin results table of the shape TrendDetector groups over."""
function replay_results_probe(n::Int)
    dt = DateTime(2026, 2, 1)
    return DataFrame(
        opentime=vcat([dt + Minute(i) for i in 1:n], [dt + Minute(i) for i in 1:n]),
        high=Float32[101f0 for _ in 1:2n],
        low=Float32[99f0 for _ in 1:2n],
        close=Float32[100f0 for _ in 1:2n],
        score=Float32[0.9f0 for _ in 1:2n],
        label=vcat(fill(Targets.longopen, n), fill(Targets.shortopen, n)),
        target=vcat(fill(Targets.longopen, n), fill(Targets.shortopen, n)),
        coin=vcat(fill("SINE", n), fill("DOUBLESINE", n)),
    )
end

@testset "replay does not alias the results table it was grouped from" begin
    EnvConfig.init(EnvConfig.test)
    n = 20
    startdt = DateTime(2026, 2, 1)
    xc = Xch.XchCache(startdt=startdt)
    ts = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(), source="test")
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

    resultsdf = replay_results_probe(n)
    before = deepcopy(resultsdf)
    resultsview = groupby(resultsdf, :coin)[1]
    @test resultsview isa SubDataFrame

    scores = resultsview[!, :score]
    labels = collect(resultsview[!, :label])

    tp = TradingStrategy.preparereplaytrades!(
        ts, xc, "SINE", resultsview, scores, labels;
        quotecoin="USDC", datetime=resultsview[end, :opentime],
    )

    # the Trades frame must own its columns, not view the results table
    @test tp.tradesdf[!, :score] isa Vector{Float32}
    @test TSM.TradesColumns(tp.tradesdf) isa TSM.TradesColumns

    TradingStrategy.processreplaygains!(tp; strategy=TradingStrategy.StrategyConfig(), lastix=n)

    # the truth pass reuses the prepared frame and writes a fresh score array
    truescores = fill(1f0, n)
    targets = collect(resultsview[!, :target])
    tp2 = TradingStrategy.preparereplaytrades!(
        ts, xc, "SINE", resultsview, truescores, targets;
        quotecoin="USDC", datetime=resultsview[end, :opentime],
    )
    TradingStrategy.processreplaygains!(tp2; strategy=TradingStrategy.StrategyConfig(), lastix=n)

    for col in propertynames(resultsdf)
        @test isequal(collect(resultsdf[!, col]), collect(before[!, col]))
    end
end
