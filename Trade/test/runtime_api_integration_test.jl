using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, Xch, Targets

"Test runtime used to validate Trade's mandatory strategy-runtime integration path."
Base.@kwdef mutable struct FakeRuntime <: TradingStrategy.AbstractStrategyRuntime
    rows::Vector{NamedTuple} = NamedTuple[]
    mins::Int = 0
    reconciliation_by_base::Dict{String, NamedTuple} = Dict{String, NamedTuple}()
    dropped::Vector{String} = String[]
end

"Expose deterministic history requirement for Trade runtime-history delegation tests."
TradingStrategy.requiredhistoryminutes(rt::FakeRuntime)::Int = rt.mins

"Record bases dropped by Trade via the mandatory runtime API path."
function TradingStrategy.dropbase!(rt::FakeRuntime, base::AbstractString)::Nothing
    push!(rt.dropped, uppercase(String(base)))
    return nothing
end

"Return injected tradesdf row metadata and capture reconciliation inputs passed by Trade."
function TradingStrategy.gettradesrows!(
    rt::FakeRuntime,
    xc::Xch.XchCache,
    bases::AbstractVector{<:AbstractString},
    datetime::DateTime;
    reconciliation_by_base::AbstractDict=Dict{String, Any}(),
)::Vector{NamedTuple}
    _ = bases
    rt.reconciliation_by_base = Dict{String, NamedTuple}(reconciliation_by_base)
    out = NamedTuple[]
    for row in rt.rows
        base = uppercase(String(row.base))
        tdf = Xch.trades(xc, base, EnvConfig.pairquote)
        rowix = findlast(==(datetime), tdf[!, :opentime])
        if isnothing(rowix)
            push!(tdf, (opentime=datetime, lastopentrade=missing, pair=Xch.tradingpairkey(base, EnvConfig.pairquote), coin=base); cols=:subset)
            rowix = nrow(tdf)
        end
        tdf[rowix, :tradelabel] = row.tradelabel
        tdf[rowix, :longopenlimit] = row.longopenlimit
        tdf[rowix, :longcloselimit] = row.longcloselimit
        tdf[rowix, :shortopenlimit] = row.shortopenlimit
        tdf[rowix, :shortcloselimit] = row.shortcloselimit
        push!(out, (
            base=base,
            datetime=datetime,
            tradesdf=tdf,
            rowix=rowix,
            probability=Float32(get(row, :probability, 0f0)),
            configid=Int(get(row, :configid, 0)),
            source=:test,
        ))
    end
    return out
end

@testset "Restricted base removal delegates to runtime" begin
    EnvConfig.init(EnvConfig.test)

    tc = Trade.TradeCache(xc=Xch.XchCache(), cl=Classify.Classifier011(), trademode=Trade.notrade)
    tc.cfg = DataFrame(basecoin=["BTC", "ETH"])

    fake = FakeRuntime()
    tc.mc[:strategy_runtime] = fake

    Trade._disablerestrictedbase!(tc, "BTC", "test")
    @test "BTC" in fake.dropped
end

@testset "Runtime API advice path" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    xc.mc[:simmode] = Xch.nosimulation
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.notrade)

    @test !isnothing(Trade._strategyruntime(tc))

    tc.cfg = DataFrame(basecoin=["BTC"])
    tc.xc.currentdt = DateTime("2026-05-30T12:00:00")

    fake = FakeRuntime(
        rows=[
            (
                base="BTC",
                tradelabel=Targets.longopen,
                longopenlimit=100f0,
                longcloselimit=110f0,
                shortopenlimit=0f0,
                shortcloselimit=0f0,
                probability=0.75f0,
                configid=42,
            ),
        ],
        mins=2000,
    )

    tc.mc[:strategy_runtime] = fake

    assets = DataFrame(
        coin=["BTC", EnvConfig.pairquote],
        free=Float32[0.5f0, 1000f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0.2f0, 0f0],
        usdtprice=Float32[100f0, 1f0],
        usdtvalue=Float32[50f0, 1000f0],
    )

    advices = Trade._collect_strategy_advices(tc, assets)
    labels = Set(ta.tradelabel for ta in advices)
    advice_by_label = Dict(ta.tradelabel => ta for ta in advices)

    @test length(advices) == 2
    @test Targets.longopen in labels
    @test Targets.longclose in labels
    @test advice_by_label[Targets.longopen].source == :tradingstrategy
    @test advice_by_label[Targets.longopen].relativeamount == 1f0
    @test advice_by_label[Targets.longopen].price == 100f0
    @test advice_by_label[Targets.longclose].relativeamount == 1f0
    @test advice_by_label[Targets.longclose].price == 110f0

    recon = fake.reconciliation_by_base["BTC"]
    @test recon.has_long_open
    @test recon.has_short_open
    @test recon.long_avg_entry == 100f0
    @test recon.short_avg_entry == 100f0
    @test recon.long_open_ix == 0
    @test recon.short_open_ix == 0

    histmins = Trade._tradeselection_history_minutes(tc)
    @test histmins >= 2001
end

@testset "Apply trading strategy stores only canonical template" begin
    EnvConfig.init(EnvConfig.test)

    mc = Dict{Symbol, Any}()
    gs = TradingStrategy.makestrategy(
        ;
        algorithm=TradingStrategy.gain_limit_reversal!,
        openthreshold=0.25f0,
        closethreshold=0.35f0,
        buygain=0.45f0,
        sellgain=0.55f0,
        limitreduction=0.15f0,
        maxwindow=12,
    )

    Trade.apply_tradingstrategy!(mc, gs; source="test")

    @test mc[:strategy_template] isa TradingStrategy.StrategyConfig
    @test mc[:strategy_algorithm] == TradingStrategy.gain_limit_reversal!
    @test mc[:strategy_source] == "test"
    @test !haskey(mc, :strategy_openthreshold)
    @test !haskey(mc, :strategy_closethreshold)
    @test !haskey(mc, :strategy_buygain)
    @test !haskey(mc, :strategy_sellgain)
    @test !haskey(mc, :strategy_limitreduction)
    @test !haskey(mc, :strategy_maxwindow)
end

@testset "usenewtrade delegates order request/status to Xch" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Day(2)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)

    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)
    tc.mc[:usenewtrade] = true
    tc.xc.currentdt = startdt
    tc.cfg = DataFrame(basecoin=["BTC"], openenabled=[true], closeenabled=[true])
    basecfg = tc.cfg[1, :]

    # Ensure simulated account has quote buying power so request can be accepted.
    bc = Xch._routedbc(tc.xc, Xch.trade_exchange_spot)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote],
        free=Float32[5_000f0],
        locked=Float32[0f0],
        borrowed=Float32[0f0],
        accruedinterest=Float32[0f0],
    )

    assets = DataFrame(
        coin=String[EnvConfig.pairquote],
        free=Float32[5_000f0],
        locked=Float32[0f0],
        borrowed=Float32[0f0],
        usdtprice=Float32[1f0],
        usdtvalue=Float32[5_000f0],
    )

    account = (
        equity_quote=5_000.0,
        free_quote=5_000.0,
        free_margin_quote=5_000.0,
        capacity=(initial_margin_quote=0.0, available_short_quote=5_000.0),
    )

    ta = Trade.StrategyAdvice(
        configid=1,
        tradelabel=Targets.longopen,
        relativeamount=1f0,
        base="BTC",
        price=nothing,
        datetime=tc.xc.currentdt,
        hourlygain=0.1f0,
        probability=0.8f0,
        investmentid=nothing,
        source=:tradingstrategy,
        allowreversal=true,
    )

    req = Trade._trade_via_xchdf!(tc, basecfg, ta, assets; account=account)
    @test req.action == :long_open

    tdf = Xch.trades(tc.xc, "BTC", EnvConfig.pairquote)
    @test nrow(tdf) >= 1
    rowix = nrow(tdf)
    @test tdf[rowix, :pair] == "BTC$(EnvConfig.pairquote)"
    @test tdf[rowix, :tradelabel] == Targets.longopen
    if req.accepted
        @test !ismissing(tdf[rowix, :longid])
        @test tdf[rowix, :longstatus] != "none"
    else
        @test req.reason == "insufficient_free_quote"
        @test tdf[rowix, :longstatus] == "Rejected"
        @test !ismissing(tdf[rowix, :longmsgid])
    end
end

@testset "open_amount applies account constraints" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Day(2)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)

    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)
    tc.xc.currentdt = startdt

    assets = DataFrame(
        coin=String["BTC", EnvConfig.pairquote],
        free=Float32[0f0, 5_000f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        usdtprice=Float32[100f0, 1f0],
        usdtvalue=Float32[0f0, 5_000f0],
    )

    longacct = (
        equity_quote=5_000.0,
        free_quote=5_000.0,
        free_margin_quote=5_000.0,
        capacity=(initial_margin_quote=0.0, available_short_quote=5_000.0),
    )

    longamount = Trade.open_amount(tc, longacct, assets, "BTC", Targets.longopen; leverage=1, unfilled_open_amount=0f0)
    @test longamount > 0f0

    shortacct_nomargin = (
        equity_quote=5_000.0,
        free_quote=5_000.0,
        free_margin_quote=0.0,
        capacity=(initial_margin_quote=100.0, available_short_quote=0.0),
    )

    assets_short = DataFrame(
        coin=String["BTC", EnvConfig.pairquote],
        free=Float32[0f0, 5_000f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0.2f0, 0f0],
        usdtprice=Float32[100f0, 1f0],
        usdtvalue=Float32[-20f0, 5_000f0],
    )

    shortamount = Trade.open_amount(tc, shortacct_nomargin, assets_short, "BTC", Targets.shortopen; leverage=2, unfilled_open_amount=0f0)
    @test shortamount < 0f0
end

@testset "usenewtrade blocks open while opposite exposure exists" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Day(2)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)

    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.buysell)
    tc.mc[:usenewtrade] = true
    tc.xc.currentdt = startdt
    tc.cfg = DataFrame(basecoin=["BTC"], openenabled=[true], closeenabled=[true])
    basecfg = tc.cfg[1, :]

    assets = DataFrame(
        coin=String["BTC", EnvConfig.pairquote],
        free=Float32[0f0, 5_000f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0.2f0, 0f0],
        usdtprice=Float32[100f0, 1f0],
        usdtvalue=Float32[-20f0, 5_000f0],
    )

    account = (
        equity_quote=4_980.0,
        free_quote=5_000.0,
        free_margin_quote=5_000.0,
        capacity=(initial_margin_quote=100.0, available_short_quote=5_000.0),
        balances=DataFrame(),
        assets=assets,
    )

    ta = Trade.StrategyAdvice(
        configid=1,
        tradelabel=Targets.longopen,
        relativeamount=1f0,
        base="BTC",
        price=nothing,
        datetime=tc.xc.currentdt,
        hourlygain=0.1f0,
        probability=0.8f0,
        investmentid=nothing,
        source=:tradingstrategy,
        allowreversal=true,
    )

    req = Trade._trade_via_xchdf!(tc, basecfg, ta, assets; account=account)
    @test !req.accepted
    @test req.reason == "sequencing_opposite_exposure"
end
