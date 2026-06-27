using Test
using Dates
using DataFrames
using Targets
using EnvConfig
using Xch
using Classify
using Ohlcv
using TestOhlcv
using TradingStrategy

Base.@kwdef mutable struct MockClassifier <: Classify.AbstractClassifier
    bc::Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}} = Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}}()
    advice_calls::Int = 0
end

function init_runtime_columns!(tdf::DataFrame)
    for contributor in TradingStrategy.tradesdf_contributors()
        contributor(tdf)
    end
    return tdf
end

function Classify.addbase!(cl::MockClassifier, ohlcv::Ohlcv.OhlcvData)
    cl.bc[String(ohlcv.base)] = (ohlcv=ohlcv,)
    return cl
end

Classify.supplement!(cl::MockClassifier) = cl
Classify.requiredminutes(::MockClassifier) = 0

function Classify.advice(cl::MockClassifier, base::AbstractString, datetime::DateTime; investment=nothing)
    _ = investment
    cl.advice_calls += 1
    return (
        tradelabel=Targets.longopen,
        probability=0.75f0,
        configid=42,
        datetime=datetime,
        base=String(base),
    )
end

@testset "TsCache classifier-call gating is variant scoped" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 11)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    # Legacy algorithm keeps current behavior and classifies each snapshot call.
    cl_plain = MockClassifier()
    rt_plain = TradingStrategy.TsCache(classifier=cl_plain, strategy=TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt_plain, xc, ["SINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    init_runtime_columns!(Xch.trades(xc, "SINE", EnvConfig.pairquote))
    evaldt = enddt
    _ = TradingStrategy.gettradesrow!(rt_plain, xc, "SINE", evaldt)
    _ = TradingStrategy.gettradesrow!(rt_plain, xc, "SINE", evaldt)
    @test cl_plain.advice_calls == 2

    # New variant gates classifier calls by price delta and minimum interval.
    cl_gated = MockClassifier()
    gs_gated = TradingStrategy.StrategyConfig(
        algorithm=TradingStrategy.gain_limit_reversal!,
        minpricedelta=0.001f0,
        max_classify_staleness_minutes=1,
    )
    rt_gated = TradingStrategy.TsCache(classifier=cl_gated, strategy=gs_gated, source="test")
    TradingStrategy.preparebases!(rt_gated, xc, ["SINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    init_runtime_columns!(Xch.trades(xc, "SINE", EnvConfig.pairquote))
    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=1))
    _ = TradingStrategy.gettradesrow!(rt_gated, xc, "SINE", evaldt; reconciliation=recon)
    _ = TradingStrategy.gettradesrow!(rt_gated, xc, "SINE", evaldt; reconciliation=recon)
    @test cl_gated.advice_calls == 1
end

@testset "TsCache classifier-call gating reclassifies on interval OR price delta" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 11)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    cl = MockClassifier()
    gs = TradingStrategy.StrategyConfig(
        algorithm=TradingStrategy.gain_limit_reversal!,
        minpricedelta=0.5f0,
        max_classify_staleness_minutes=1,
    )
    rt = TradingStrategy.TsCache(classifier=cl, strategy=gs, source="test")
    TradingStrategy.preparebases!(rt, xc, ["SINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    init_runtime_columns!(Xch.trades(xc, "SINE", EnvConfig.pairquote))

    evaldt = enddt
    _ = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt)
    _ = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt + Minute(1))
    @test cl.advice_calls == 2
end

@testset "StrategyConfig max staleness naming" begin
    gs_new = TradingStrategy.StrategyConfig(max_classify_staleness_minutes=3)

    @test gs_new.max_classify_staleness_minutes == 3
    @test TradingStrategy.max_classify_staleness_minutes(gs_new) == 3
end

@testset "Runtime API compatibility adapter" begin
    @test_throws ArgumentError TradingStrategy.TsCache(source="test")
    rt = TradingStrategy.TsCache(classifier=MockClassifier())

    @test TradingStrategy.requiredhistoryminutes(rt) >= 0
    @test isempty(TradingStrategy.acceptedbases(rt))

    gs = TradingStrategy.StrategyConfig(maxwindow=12)
    push!(rt.accepted, "BTC")
    TradingStrategy.apply_strategy!(rt, gs; source="test")
    @test isempty(rt.pairs)
    @test isempty(rt.classifier_gate_state)
    @test isempty(TradingStrategy.acceptedbases(rt))

    TradingStrategy.dropbase!(rt, "BTC")
    push!(rt.accepted, "ETH")
    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))

    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=7))
    @test recon.has_long_open
    @test recon.long_open_ix == 7

    startdt = DateTime(2026, 1, 1)
    evaldt = startdt + Minute(1)
    xc = Xch.XchCache(startdt=startdt)
    Xch.addbase!(xc, "BTC", startdt, startdt + Minute(120))
    xc.currentdt = evaldt
    TradingStrategy.preparebases!(rt, xc, ["BTC"]; history_startdt=startdt, datetime=evaldt, updatecache=false)
    init_runtime_columns!(Xch.trades(xc, "BTC", EnvConfig.pairquote))
    rowmeta = TradingStrategy.gettradesrow!(rt, xc, "BTC", evaldt; reconciliation=recon)
    @test !isnothing(rowmeta)
    @test rowmeta.base == "BTC"
    @test rowmeta.rowix >= 1
    @test rowmeta.tradesdf[rowmeta.rowix, :pair] == Xch.tradingpairkey("BTC", EnvConfig.pairquote)
end

@testset "TsCache pair-state scaffolding syncs Xch trades" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 1)
    evaldt = startdt + Minute(10)
    xc = Xch.XchCache(startdt=startdt)

    seed = DataFrame(opentime=[startdt], lastopentrade=Union{Missing, DateTime}[missing])
    Xch.settrades!(xc, "btc", "usdt", seed)

    @test_throws ArgumentError TradingStrategy.TsCache(source="test")
    cfg = Dict{Symbol, Any}(:classifier_factory => () -> MockClassifier())
    ts = TradingStrategy.TsCache(configuration=cfg, source="test")
    @test TradingStrategy.pairkeys(ts) == String[]
    @test TradingStrategy.tspairkey("btc", "usdt") == "BTCUSDT"

    tp = TradingStrategy.syncpairtrades!(ts, xc, "btc", "usdt"; datetime=evaldt)
    @test tp.pair == "BTCUSDT"
    @test tp.last_update_dt == evaldt
    @test TradingStrategy.haspairstate(ts, "btcusdt")
    @test tp.tradesdf === Xch.trades(xc, "BTCUSDT")

    push!(tp.tradesdf, (opentime=startdt + Minute(1), lastopentrade=missing); cols=:subset)
    @test nrow(Xch.trades(xc, "BTCUSDT")) == 2

    tp2 = TradingStrategy.getpairstate!(ts, "eth", "usdt")
    @test tp2.pair == "ETHUSDT"
    @test TradingStrategy.pairkeys(ts) == ["BTCUSDT", "ETHUSDT"]

    TradingStrategy.droppair!(ts, "ethusdt")
    @test !TradingStrategy.haspairstate(ts, "ETHUSDT")
    @test TradingStrategy.pairkeys(ts) == ["BTCUSDT"]
end

@testset "TsCache getgains writes pair trades" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 1)
    xc = Xch.XchCache(startdt=startdt)

    cfg = Dict{Symbol, Any}(
        :classifier_factory => () -> MockClassifier(),
        :strategy_template => TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!),
    )
    ts = TradingStrategy.TsCache(configuration=cfg, source="test")

    df = DataFrame(
        opentime=[startdt + Minute(i) for i in 0:4],
        high=Float32[101f0, 102f0, 103f0, 104f0, 105f0],
        low=Float32[99f0, 100f0, 101f0, 102f0, 103f0],
        close=Float32[100f0, 101f0, 102f0, 103f0, 104f0],
    )
    init_runtime_columns!(df)
    scores = Float32[0.8f0, 0.7f0, 0.9f0, 0.4f0, 0.3f0]
    labels = Targets.TradeLabel[Targets.longopen, Targets.longopen, Targets.longopen, Targets.longclose, Targets.longclose]

    gdf = TradingStrategy.getgains(
        ts,
        xc,
        "sine",
        df,
        scores,
        labels,
        true;
        strategy=ts.strategy_template,
        openthreshold=0.6f0,
        closethreshold=0.1f0,
        metadata=Dict{Symbol, Any}(:predicted => true, :rangeid => 1),
    )

    @test Xch.hastrades(xc, "SINEUSDT")
    tdf = Xch.trades(xc, "SINEUSDT")
    @test nrow(tdf) == nrow(df)
    @test "predicted" in names(tdf)
    @test "rangeid" in names(tdf)
    @test "longopenlimit" in names(tdf)
    @test "longcloselimit" in names(tdf)
    @test "shortopenlimit" in names(tdf)
    @test "shortcloselimit" in names(tdf)
    @test "tradelabel" in names(tdf)
    @test "labelscore" in names(tdf)
    @test "lastopentrade" in names(tdf)
    @test tdf[1, :pair] == "SINEUSDT"
    @test size(gdf, 1) >= 0
end

@testset "TsCache multi-base lifecycle is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 1)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)
    xc.bases["DOUBLESINE"] = TestOhlcv.testohlcv("DOUBLESINE", startdt, enddt)

    rt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(), source="test")

    TradingStrategy.preparebases!(rt, xc, ["SINE", "DOUBLESINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["SINE", "DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["SINE", "DOUBLESINE"])

    TradingStrategy.dropbase!(rt, "SINE")
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.preparebases!(rt, xc, ["DOUBLESINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.apply_strategy!(rt, TradingStrategy.StrategyConfig(maxwindow=60); source="reconfigured")
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.classifier))))

    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.classifier))))
end

@testset "TsCache row production is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 10)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    Xch.addbase!(xc, "SINE", startdt, enddt)

    rt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt, xc, ["SINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    init_runtime_columns!(Xch.trades(xc, "SINE", EnvConfig.pairquote))

    evaldt = enddt
    xc.currentdt = evaldt
    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=5))

    snap1 = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt; reconciliation=recon)
    snap2 = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt; reconciliation=recon)

    @test !isnothing(snap1)
    @test !isnothing(snap2)
    @test snap1.base == snap2.base
    @test snap1.datetime == snap2.datetime
    @test snap1.tradesdf[snap1.rowix, :tradelabel] == snap2.tradesdf[snap2.rowix, :tradelabel]
    @test isequal(snap1.tradesdf[snap1.rowix, :longopenlimit], snap2.tradesdf[snap2.rowix, :longopenlimit])
    @test isequal(snap1.tradesdf[snap1.rowix, :longcloselimit], snap2.tradesdf[snap2.rowix, :longcloselimit])
    @test isequal(snap1.tradesdf[snap1.rowix, :shortopenlimit], snap2.tradesdf[snap2.rowix, :shortopenlimit])
    @test isequal(snap1.tradesdf[snap1.rowix, :shortcloselimit], snap2.tradesdf[snap2.rowix, :shortcloselimit])
    @test snap1.configid == snap2.configid
end
