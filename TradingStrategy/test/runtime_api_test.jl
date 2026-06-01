using Test
using Dates
using Targets
using EnvConfig
using CryptoXch
using Classify
using Ohlcv
using TestOhlcv
using TradingStrategy

Base.@kwdef mutable struct IncompleteRuntime <: TradingStrategy.AbstractStrategyRuntime
end

Base.@kwdef mutable struct MockClassifier <: Classify.AbstractClassifier
    bc::Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}} = Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}}()
end

function Classify.addbase!(cl::MockClassifier, ohlcv::Ohlcv.OhlcvData)
    cl.bc[String(ohlcv.base)] = (ohlcv=ohlcv,)
    return cl
end

Classify.supplement!(cl::MockClassifier) = cl

function Classify.advice(cl::MockClassifier, base::AbstractString, datetime::DateTime; investment=nothing)
    _ = cl
    _ = investment
    return (
        tradelabel=Targets.longbuy,
        probability=0.75f0,
        configid=42,
        datetime=datetime,
        base=String(base),
    )
end

@testset "Runtime API compatibility adapter" begin
    rt = TradingStrategy.GainSegmentRuntime()

    @test TradingStrategy.requiredhistoryminutes(rt) >= 0
    @test isempty(TradingStrategy.acceptedbases(rt))

    snap = TradingStrategy.StrategySnapshot(
        base="BTC",
        datetime=DateTime(2026, 1, 1),
        label=Targets.longbuy,
        long_openprice=100f0,
        long_closeprice=101f0,
        long_openix=1,
    )
    @test snap.label == Targets.longbuy
    @test snap.long_openix == 1

    gs = TradingStrategy.GainSegment(maxwindow=12)
    push!(rt.accepted, "BTC")
    TradingStrategy.apply_strategy!(rt, gs; source="test")
    @test isempty(rt.strategy_state)
    @test isempty(rt.strategy_history)
    @test isempty(TradingStrategy.acceptedbases(rt))

    TradingStrategy.dropbase!(rt, "BTC")
    push!(rt.accepted, "ETH")
    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))

    recon = TradingStrategy.StrategyReconciliationInput(has_long_open=true, long_avg_entry=100f0, long_open_ix=7)
    @test recon.has_long_open
    @test recon.long_open_ix == 7
end

@testset "Runtime API mandatory abstract methods fail fast" begin
    EnvConfig.init(EnvConfig.test)
    rt = IncompleteRuntime()
    xc = CryptoXch.XchCache()
    dt = DateTime(2026, 1, 1)

    @test_throws ArgumentError TradingStrategy.preparebases!(rt, xc, ["BTC"]; history_startdt=dt - Minute(120), datetime=dt, updatecache=false)
    @test_throws ArgumentError TradingStrategy.getsnapshot!(rt, xc, "BTC", dt)
    @test_throws ArgumentError TradingStrategy.getsnapshots!(rt, xc, ["BTC"], dt)
end

@testset "GainSegmentRuntime multi-base lifecycle is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 1)
    enddt = startdt + Minute(240)
    xc = CryptoXch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)
    xc.bases["DOUBLESINE"] = TestOhlcv.testohlcv("DOUBLESINE", startdt, enddt)

    rt = TradingStrategy.GainSegmentRuntime(classifier=MockClassifier(), strategy=TradingStrategy.GainSegment(), source="test")

    TradingStrategy.preparebases!(rt, xc, ["SINE", "DOUBLESINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["SINE", "DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["SINE", "DOUBLESINE"])

    TradingStrategy.dropbase!(rt, "SINE")
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.preparebases!(rt, xc, ["DOUBLESINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.apply_strategy!(rt, TradingStrategy.GainSegment(maxwindow=60); source="reconfigured")
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.classifier))))

    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.classifier))))
end

@testset "GainSegmentRuntime snapshot production is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 10)
    enddt = startdt + Minute(240)
    xc = CryptoXch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    rt = TradingStrategy.GainSegmentRuntime(classifier=MockClassifier(), strategy=TradingStrategy.GainSegment(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt, xc, ["SINE"]; history_startdt=startdt, datetime=enddt, updatecache=false)

    evaldt = enddt
    recon = TradingStrategy.StrategyReconciliationInput(has_long_open=true, long_avg_entry=100f0, long_open_ix=5)

    snap1 = TradingStrategy.getsnapshot!(rt, xc, "SINE", evaldt; reconciliation=recon)
    snap2 = TradingStrategy.getsnapshot!(rt, xc, "SINE", evaldt; reconciliation=recon)

    @test !isnothing(snap1)
    @test !isnothing(snap2)
    @test snap1.base == snap2.base
    @test snap1.datetime == snap2.datetime
    @test snap1.label == snap2.label
    @test snap1.long_openprice == snap2.long_openprice
    @test snap1.long_closeprice == snap2.long_closeprice
    @test snap1.short_openprice == snap2.short_openprice
    @test snap1.short_closeprice == snap2.short_closeprice
    @test snap1.configid == snap2.configid
end
