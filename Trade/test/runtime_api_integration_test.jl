using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, CryptoXch, Targets

"Test runtime used to validate Trade's optional strategy-runtime integration path."
Base.@kwdef mutable struct FakeRuntime <: TradingStrategy.AbstractStrategyRuntime
    snapshots::Vector{TradingStrategy.StrategySnapshot} = TradingStrategy.StrategySnapshot[]
    mins::Int = 0
    reconciliation_by_base::Dict{String, TradingStrategy.StrategyReconciliationInput} = Dict{String, TradingStrategy.StrategyReconciliationInput}()
    dropped::Vector{String} = String[]
end

"Expose deterministic history requirement for Trade runtime-history delegation tests."
TradingStrategy.requiredhistoryminutes(rt::FakeRuntime)::Int = rt.mins

"Record bases dropped by Trade when runtime API path is active."
function TradingStrategy.dropbase!(rt::FakeRuntime, base::AbstractString)::Nothing
    push!(rt.dropped, uppercase(String(base)))
    return nothing
end

"Return injected snapshots and capture reconciliation inputs passed by Trade."
function TradingStrategy.getsnapshots!(
    rt::FakeRuntime,
    xc::CryptoXch.XchCache,
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

    tc = Trade.TradeCache(xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=Trade.notrade)
    tc.cfg = DataFrame(basecoin=["BTC", "ETH"])

    fake = FakeRuntime()
    tc.mc[:strategy_runtime] = fake
    tc.mc[:use_strategy_runtime_api] = true

    Trade._disablerestrictedbase!(tc, "BTC", "test")
    @test "BTC" in fake.dropped
end

@testset "Runtime API default and advice path" begin
    EnvConfig.init(EnvConfig.test)

    xc = CryptoXch.XchCache()
    xc.mc[:simmode] = CryptoXch.nosimulation
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.notrade)

    @test tc.mc[:use_strategy_runtime_api]

    tc.cfg = DataFrame(basecoin=["BTC"])
    tc.xc.currentdt = DateTime("2026-05-30T12:00:00")

    fake = FakeRuntime(
        snapshots=[
            TradingStrategy.StrategySnapshot(
                base="BTC",
                datetime=tc.xc.currentdt,
                label=Targets.longbuy,
                long_openprice=100f0,
                long_closeprice=110f0,
                probability=0.75f0,
                configid=42,
            ),
        ],
        mins=2000,
    )

    tc.mc[:strategy_runtime] = fake
    tc.mc[:use_strategy_runtime_api] = true

    assets = DataFrame(
        coin=["BTC", EnvConfig.cryptoquote],
        free=Float32[0.5f0, 1000f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0.2f0, 0f0],
        usdtprice=Float32[100f0, 1f0],
        usdtvalue=Float32[50f0, 1000f0],
    )

    advices = Trade._collect_strategy_advices(tc, assets)
    labels = Set(ta.tradelabel for ta in advices)

    @test length(advices) == 2
    @test Targets.longbuy in labels
    @test Targets.longclose in labels

    recon = fake.reconciliation_by_base["BTC"]
    @test recon.has_long_open
    @test recon.has_short_open

    histmins = Trade._tradeselection_history_minutes(tc)
    @test histmins >= 2001
end

@testset "Runtime API env override" begin
    prev = get(ENV, "CTS_USE_STRATEGY_RUNTIME_API", nothing)
    try
        EnvConfig.init(EnvConfig.test)

        ENV["CTS_USE_STRATEGY_RUNTIME_API"] = "false"
        tc_disabled = Trade.TradeCache(xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=Trade.notrade)
        @test !tc_disabled.mc[:use_strategy_runtime_api]

        ENV["CTS_USE_STRATEGY_RUNTIME_API"] = "true"
        tc_enabled = Trade.TradeCache(xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=Trade.notrade)
        @test tc_enabled.mc[:use_strategy_runtime_api]
    finally
        if isnothing(prev)
            pop!(ENV, "CTS_USE_STRATEGY_RUNTIME_API", nothing)
        else
            ENV["CTS_USE_STRATEGY_RUNTIME_API"] = String(prev)
        end
    end
end
