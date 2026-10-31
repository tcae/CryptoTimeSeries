using Test
using Dates
using DataFrames
using Targets
using EnvConfig
using Xch, Bybit
using Classify
using Ohlcv
using TestOhlcv
using TradingStrategy
using TSM

Base.@kwdef mutable struct MockClassifier <: Classify.AbstractClassifier
    bc::Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}} = Dict{String, NamedTuple{(:ohlcv,), Tuple{Ohlcv.OhlcvData}}}()
    advice_calls::Int = 0
end

function init_runtime_columns!(tdf::DataFrame)
    TSM.xch_tradesdf_lastopentrade(tdf)
    for contributor in TSM.tradesdf_all_contributors()
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

"""Minimal classifier input configuration for testing strategy-only regressions."""
function strategyregressiontestfeatures()
    f6 = Features.Features006()
    Features.addgrad!(f6; window=5)
    return f6
end

@testset "strategy regressions share live and replay initialization" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        oldpath = EnvConfig.coinspath()
        EnvConfig.setcoinspath!(folder)
        try
            startdt = DateTime("2025-02-17T13:30:00")
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(80))
            labels = Targets.TradeLabel[Targets.longopen, Targets.shortopen]
            nn = Classify.model008(Features.featurecount(strategyregressiontestfeatures()), labels, "strategy-regressions")
            cl = Classify.TrendClassifier001(nn; featconfig=strategyregressiontestfeatures)
            cfg = TradingStrategy.GainLimitReversalBelowRegressionConfig(triggerregrwindow=180, trendregrwindow=180)
            strategy = TradingStrategy.StrategyConfig(classifier=cl, algorithm=TradingStrategy.gain_limit_reversal_below_regression!, algorithmconfig=cfg)
            rt = TradingStrategy.TsCache(strategy=strategy)
            @test TradingStrategy._strategyregressionwindows(cfg) == [180]
            @test isempty(TradingStrategy._strategyregressionwindows(TradingStrategy.GainLimitReversalConfig()))
            @test isempty(TradingStrategy._strategyregressionwindows(TradingStrategy.strategytemplate("S08").algorithmconfig))
            @test TradingStrategy.requiredhistoryminutes(rt) == 180
            dt = Ohlcv.dataframe(ohlcv)[end, :opentime]
            @test_throws r"requires 180 minutes" TradingStrategy.addclassifierbase!(rt, ohlcv; datetime=dt)
            xc = Xch.XchCache(Bybit.BybitCache(); startdt=startdt)
            EnvConfig.setcoinspath!(folder)
            xc.bases["SINE"] = ohlcv
            @test !isnothing(TradingStrategy.acceptbase!(rt, xc, "SINE"; datetime=dt))
            f6 = Classify.featconfig(cl, "SINE")
            @test Features.requestedcolumns(f6) == Features.requestedcolumns(strategyregressiontestfeatures())
            @test Features.regressionat(f6, 180, dt) isa NamedTuple
            path = joinpath(Features.regressionfolder("SINE", ohlcv.quotecoin), "regr180.arrow")
            @test !isfile(path)
            extended = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(81))
            Ohlcv.setdataframe!(ohlcv, DataFrame(Ohlcv.dataframe(extended)))
            @test !isnothing(TradingStrategy.acceptbase!(rt, xc, "SINE"; datetime=dt + Minute(1)))
            @test !isfile(path)
            @test Features.regressionat(f6, 180, dt + Minute(1)) isa NamedTuple

            probe = DataFrame(opentime=[dt + Minute(1)], pair=["SINEUSDT"], close=Float32[2], high=Float32[2], low=Float32[2], score=Float32[0.9], label=Targets.TradeLabel[Targets.longopen])
            init_runtime_columns!(probe)
            cols = TSM.TradesColumns(probe)
            regr = Features.regressionat(f6, 180, dt + Minute(1))
            @test TradingStrategy._relative_to_regression(cfg.gainlimit, cl, cols, 1, 180) ≈ (probe.close[1] - regr.regry) / regr.regry
            @test TradingStrategy._regression_slope_gain_per_hour(cfg.gainlimit, cl, cols, 1, 180) ≈ Features.relativegain(regr.regry, regr.grad, 60)
            @test_throws AssertionError TradingStrategy._relative_to_regression(cfg.gainlimit, nothing, cols, 1, 180)

            full = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(240))
            TradingStrategy.reset!(rt)
            TradingStrategy.addclassifierbase!(rt, Ohlcv.ohlcvview(full, 200:240); datetime=startdt + Minute(200))
            @test Classify.featconfig(cl, "SINE") !== f6
            @test Features.regressionat(Classify.featconfig(cl, "SINE"), 180, startdt + Minute(200)) isa NamedTuple
            TradingStrategy.reset!(rt)
            replayxc = Xch.XchCache(Bybit.BybitCache(); startdt=startdt, enddt=startdt + Minute(240))
            EnvConfig.setcoinspath!(folder)
            replayxc.bases["SINE"] = ohlcv
            @test_throws AssertionError TradingStrategy.acceptbase!(rt, replayxc, "SINE"; datetime=dt)
            replayxc.bases["SINE"] = full
            @test !isnothing(TradingStrategy.acceptbase!(rt, replayxc, "SINE"; datetime=startdt + Minute(240)))
            TradingStrategy.apply_strategy!(rt, TradingStrategy.StrategyConfig(classifier=cl))
            TradingStrategy.addclassifierbase!(rt, full)
            @test !("ry+180" in names(Classify.featconfig(cl, "SINE").fdfno))
            mockrt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(algorithmconfig=cfg))
            @test_throws r"require TrendClassifier001" TradingStrategy.addclassifierbase!(mockrt, full)
            preloaded = Classify.TrendClassifier001(nn; featconfig=strategyregressiontestfeatures)
            Classify.addbase!(preloaded, full)
            preloadedrt = TradingStrategy.TsCache(classifier=preloaded, strategy=TradingStrategy.StrategyConfig(algorithmconfig=cfg))
            @test_throws r"not initialized" TradingStrategy.addclassifierbase!(preloadedrt, full)
        finally
            EnvConfig.setcoinspath!(oldpath)
        end
    end
end

function Classify.advice(cl::MockClassifier, base::AbstractString, datetime::DateTime; investment=nothing)
    _ = investment
    cl.advice_calls += 1
    return (
        tradelabel=longopen,
        probability=0.75f0,
        configid=42,
        datetime=datetime,
        base=String(base),
    )
end

@testset "TsCache classifier advice refreshes each invocation" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 11)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    cl_plain = MockClassifier()
    rt_plain = TradingStrategy.TsCache(classifier=cl_plain, strategy=TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt_plain, xc, ["SINE"]; datetime=enddt, updatecache=false)
    init_runtime_columns!(TSM.trades(xc.tsm, "SINE", EnvConfig.pairquote))
    evaldt = enddt
    _ = TradingStrategy.gettradesrow!(rt_plain, xc, "SINE", evaldt)
    _ = TradingStrategy.gettradesrow!(rt_plain, xc, "SINE", evaldt)
    @test cl_plain.advice_calls == 2

end

@testset "Runtime API compatibility adapter" begin
    startdt = DateTime(2026, 1, 1)
    evaldt = startdt + Minute(1)
    xc = Xch.XchCache(Bybit.BybitCache(), startdt=startdt)
    Xch.addbase!(xc, "BTC", startdt, startdt + Minute(120))

    @test_throws ArgumentError TradingStrategy.TsCache(source="test")
    rt = TradingStrategy.TsCache(classifier=MockClassifier())

    @test TradingStrategy.requiredhistoryminutes(rt) >= 0
    @test isempty(TradingStrategy.acceptedbases(rt))

    gs = TradingStrategy.StrategyConfig(algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=12))
    push!(rt.accepted, "BTC")
    TradingStrategy.apply_strategy!(rt, gs; source="test")
    @test isempty(rt.pairs)
    @test isempty(TradingStrategy.acceptedbases(rt))

    TradingStrategy.dropbase!(rt, "BTC")
    push!(rt.accepted, "ETH")
    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))

    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=7))
    @test recon.has_long_open
    @test recon.long_open_ix == 7

    xc.currentdt = evaldt
    TradingStrategy.preparebases!(rt, xc, ["BTC"]; datetime=evaldt, updatecache=false)
    init_runtime_columns!(TSM.trades(xc.tsm, "BTC", EnvConfig.pairquote))
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
    TSM.settrades!(xc.tsm, "btc", "usdt", seed)

    @test_throws ArgumentError TradingStrategy.TsCache(source="test")
    ts = TradingStrategy.TsCache(strategy=TradingStrategy.StrategyConfig(classifier=MockClassifier()), source="test")
    @test TradingStrategy.pairkeys(ts) == String[]
    @test TradingStrategy.tspairkey("btc", "usdt") == "BTCUSDT"

    tp = TradingStrategy.syncpairtrades!(ts, xc, "btc", "usdt"; datetime=evaldt)
    @test tp.pair == "BTCUSDT"
    @test tp.last_update_dt == evaldt
    @test TradingStrategy.haspairstate(ts, "btcusdt")
    @test tp.tradesdf === TSM.trades(xc.tsm, "BTCUSDT")

    push!(tp.tradesdf, (opentime=startdt + Minute(1), lastopentrade=missing); cols=:subset)
    @test nrow(TSM.trades(xc.tsm, "BTCUSDT")) == 2

    tp2 = TradingStrategy.getpairstate!(ts, "eth", "usdt")
    @test tp2.pair == "ETHUSDT"
    @test TradingStrategy.pairkeys(ts) == ["BTCUSDT", "ETHUSDT"]

    TradingStrategy.droppair!(ts, "ethusdt")
    @test !TradingStrategy.haspairstate(ts, "ETHUSDT")
    @test TradingStrategy.pairkeys(ts) == ["BTCUSDT"]
end

@testset "Explicit replay processing fails fast on malformed state" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 3)

    badcols_tp = TradingStrategy.TsTp(
        pair="SINEUSDT",
        tradesdf=DataFrame(opentime=[startdt], high=Float32[101f0], low=Float32[99f0]),
    )
    @test_throws AssertionError TradingStrategy.processreplaygains!(
        badcols_tp;
        strategy=TradingStrategy.StrategyConfig(),
        lastix=1,
    )

    tdf = DataFrame(
        opentime=[startdt],
        high=Float32[101f0],
        low=Float32[99f0],
        pair=["SINEUSDT"],
    )
    init_runtime_columns!(tdf)
    tdf[!, :score] = Float32[0.8f0]
    tdf[!, :label] = TradeLabel[longopen]
    noclose_tp = TradingStrategy.TsTp(pair="SINEUSDT", tradesdf=tdf)
    @test_throws AssertionError TradingStrategy.processreplaygains!(
        noclose_tp;
        strategy=TradingStrategy.StrategyConfig(),
        lastix=1,
    )
end

@testset "TsCache multi-base lifecycle is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 1)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)
    xc.bases["DOUBLESINE"] = TestOhlcv.testohlcv("DOUBLESINE", startdt, enddt)

    rt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(), source="test")

    TradingStrategy.preparebases!(rt, xc, ["SINE", "DOUBLESINE"]; datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["SINE", "DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.cfg.classifier))) == Set(["SINE", "DOUBLESINE"])

    TradingStrategy.dropbase!(rt, "SINE")
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.cfg.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.preparebases!(rt, xc, ["DOUBLESINE"]; datetime=enddt, updatecache=false)
    @test TradingStrategy.acceptedbases(rt) == Set(["DOUBLESINE"])
    @test Set(String.(Classify.bases(rt.cfg.classifier))) == Set(["DOUBLESINE"])

    TradingStrategy.apply_strategy!(rt, TradingStrategy.StrategyConfig(algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=60)); source="reconfigured")
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.cfg.classifier))))

    TradingStrategy.reset!(rt)
    @test isempty(TradingStrategy.acceptedbases(rt))
    @test isempty(Set(String.(Classify.bases(rt.cfg.classifier))))
end

@testset "TsCache row production is deterministic" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 10)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    rt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt, xc, ["SINE"]; datetime=enddt, updatecache=false)
    init_runtime_columns!(TSM.trades(xc.tsm, "SINE", EnvConfig.pairquote))

    evaldt = enddt
    xc.currentdt = evaldt
    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=5))

    snap1 = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt; reconciliation=recon)
    snap2 = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt; reconciliation=recon)

    @test !isnothing(snap1)
    @test !isnothing(snap2)
    @test snap1.base == snap2.base
    @test snap1.datetime == snap2.datetime
    @test snap1.tradesdf[snap1.rowix, :label] == snap2.tradesdf[snap2.rowix, :label]
    @test isequal(snap1.tradesdf[snap1.rowix, :lo_limit], snap2.tradesdf[snap2.rowix, :lo_limit])
    @test isequal(snap1.tradesdf[snap1.rowix, :lc_limit], snap2.tradesdf[snap2.rowix, :lc_limit])
    @test isequal(snap1.tradesdf[snap1.rowix, :so_limit], snap2.tradesdf[snap2.rowix, :so_limit])
    @test isequal(snap1.tradesdf[snap1.rowix, :sc_limit], snap2.tradesdf[snap2.rowix, :sc_limit])
    @test snap1.configid == snap2.configid
end

@testset "TsCache advice phase composes with row application phase" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime(2026, 1, 12)
    enddt = startdt + Minute(240)
    xc = Xch.XchCache(startdt=startdt)
    xc.bases["SINE"] = TestOhlcv.testohlcv("SINE", startdt, enddt)

    rt = TradingStrategy.TsCache(classifier=MockClassifier(), strategy=TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!), source="test")
    TradingStrategy.preparebases!(rt, xc, ["SINE"]; datetime=enddt, updatecache=false)
    init_runtime_columns!(TSM.trades(xc.tsm, "SINE", EnvConfig.pairquote))

    evaldt = enddt
    recon = merge(TradingStrategy.defaultreconciliationinput(), (has_long_open=true, long_avg_entry=100f0, long_open_ix=9))

    rowmeta = TradingStrategy.gettradesrow!(rt, xc, "SINE", evaldt; reconciliation=recon)
    @test !isnothing(rowmeta)
    @test rowmeta.base == "SINE"
    @test rowmeta.datetime == evaldt
    @test rowmeta.rowix >= 1
    @test (rowmeta.tradesdf[rowmeta.rowix, :close]) > 0f0
    @test (rowmeta.tradesdf[rowmeta.rowix, :lo_limit]) > 0f0
end
