using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, Xch, Targets

"Test runtime used to validate Trade's mandatory strategy-runtime integration path."
Base.@kwdef mutable struct FakeRuntime <: TradingStrategy.AbstractStrategyRuntime
    snapshots::Vector{TradingStrategy.StrategySnapshot} = TradingStrategy.StrategySnapshot[]
    mins::Int = 0
    reconciliation_by_base::Dict{String, TradingStrategy.StrategyReconciliationInput} = Dict{String, TradingStrategy.StrategyReconciliationInput}()
    dropped::Vector{String} = String[]
end

"Expose deterministic history requirement for Trade runtime-history delegation tests."
TradingStrategy.requiredhistoryminutes(rt::FakeRuntime)::Int = rt.mins

"Record bases dropped by Trade via the mandatory runtime API path."
function TradingStrategy.dropbase!(rt::FakeRuntime, base::AbstractString)::Nothing
    push!(rt.dropped, uppercase(String(base)))
    return nothing
end

"Return injected snapshots and capture reconciliation inputs passed by Trade."
function TradingStrategy.getsnapshots!(
    rt::FakeRuntime,
    xc::Xch.XchCache,
    bases::AbstractVector{<:AbstractString},
    datetime::DateTime;
    reconciliation_by_base::AbstractDict{String, TradingStrategy.StrategyReconciliationInput}=Dict{String, TradingStrategy.StrategyReconciliationInput}(),
)::Vector{TradingStrategy.StrategySnapshot}
    _ = xc
    _ = bases
    _ = datetime
    rt.reconciliation_by_base = Dict{String, TradingStrategy.StrategyReconciliationInput}(reconciliation_by_base)
    return copy(rt.snapshots)
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
        snapshots=[
            TradingStrategy.StrategySnapshot(
                base="BTC",
                datetime=tc.xc.currentdt,
                label=Targets.longopen,
                long_openprice=100f0,
                long_closeprice=110f0,
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
    gs = TradingStrategy.GainSegment(
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

    @test mc[:strategy_template] isa TradingStrategy.GainSegment
    @test mc[:strategy_algorithm] == TradingStrategy.gain_limit_reversal!
    @test mc[:strategy_source] == "test"
    @test !haskey(mc, :strategy_openthreshold)
    @test !haskey(mc, :strategy_closethreshold)
    @test !haskey(mc, :strategy_buygain)
    @test !haskey(mc, :strategy_sellgain)
    @test !haskey(mc, :strategy_limitreduction)
    @test !haskey(mc, :strategy_maxwindow)
end
