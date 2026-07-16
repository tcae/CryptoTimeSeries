using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, Xch, Targets

const TEST_RECONCILIATION_BY_BASE = Dict{String, NamedTuple}()
const TEST_ROWS = NamedTuple[]
const TEST_PREPARE_CALLS = NamedTuple[]

"Return injected per-base tradesdf row state used by Trade._collect_strategy_row."
function TradingStrategy.gettradesrow!(
    rt::TradingStrategy.TsCache,
    xc::Xch.XchCache,
    base::AbstractString,
    datetime::DateTime;
    reconciliation=nothing,
)::Union{Nothing, NamedTuple}
    _ = rt
    _ = reconciliation
    empty!(TEST_RECONCILIATION_BY_BASE)
    empty!(TEST_PREPARE_CALLS)
    basekey = uppercase(String(base))
    Xch.ensuretradesschema(xc, Xch.tradesdf_all_contributors())
    row = findfirst(r -> uppercase(String(r.base)) == basekey, TEST_ROWS)
    isnothing(row) && return nothing

    tdf = Xch.trades(xc, basekey, EnvConfig.pairquote)
    rowix = findlast(==(datetime), tdf[!, :opentime])
    if isnothing(rowix)
        push!(tdf, (opentime=datetime, lastopentrade=missing, pair=Xch.tradingpairkey(basekey, EnvConfig.pairquote), coin=basekey); cols=:subset)
        rowix = nrow(tdf)
    end
    spec = TEST_ROWS[row]
    tdf[rowix, :label] = spec.tradelabel
    tdf[rowix, :lo_limit] = hasproperty(spec, :lo_limit) ? spec.lo_limit : spec.longopenlimit
    tdf[rowix, :lc_limit] = hasproperty(spec, :lc_limit) ? spec.lc_limit : spec.longcloselimit
    tdf[rowix, :so_limit] = hasproperty(spec, :so_limit) ? spec.so_limit : spec.shortopenlimit
    tdf[rowix, :sc_limit] = hasproperty(spec, :sc_limit) ? spec.sc_limit : spec.shortcloselimit
    tdf[rowix, :score] = (get(spec, :probability, 0f0))

    return (
        base=basekey,
        datetime=datetime,
        tradesdf=tdf,
        rowix=rowix,
        probability=(get(spec, :probability, 0f0)),
        configid=Int(get(spec, :configid, 0)),
        source=:test,
    )
end

function TradingStrategy.preparebases!(
    rt::TradingStrategy.TsCache,
    xc::Xch.XchCache,
    bases::AbstractVector{<:AbstractString};
    datetime::DateTime,
    updatecache::Bool=false,
)::Nothing
    _ = xc
    push!(TEST_PREPARE_CALLS, (
        bases=String[uppercase(String(base)) for base in bases],
        datetime=datetime,
        updatecache=Bool(updatecache),
    ))
    return nothing
end

function _prepare_strategy_runtime_for_cfg!(cache::TradeCache, datetime::DateTime; updatecache::Bool=false)::Nothing
    hasproperty(cache.cfg, :basecoin) || return nothing
    size(cache.cfg, 1) == 0 && return nothing

    rt = Trade._strategyruntime(cache)
    bases = String.(cache.cfg[!, :basecoin])
    isempty(bases) && return nothing

    TradingStrategy.preparebases!(rt, cache.xc, bases; datetime=datetime, updatecache=updatecache)
    return nothing
end

function _collect_strategy_row(cache::TradeCache, base::AbstractString)
    dt = cache.xc.currentdt
    isnothing(dt) && return nothing
    rt = Trade._strategyruntime(cache)
    rowmeta = TradingStrategy.gettradesrow!(rt, cache.xc, uppercase(String(base)), dt)
    isnothing(rowmeta) && return nothing
    return (tradesdf=rowmeta.tradesdf, rowix=Int(rowmeta.rowix))
end

"Blacklist one base in the current runtime config to avoid repeated order attempts."
function _blacklistbase!(cache::TradeCache, base::AbstractString, reason::AbstractString)::Nothing
    base_upper = uppercase(String(base))
    blacklist = get!(cache.mc, :blacklistbases, String[])
    !(base_upper in blacklist) && push!(blacklist, base_upper)

    if !hasproperty(cache.cfg, :basecoin)
        (verbosity >= 1) && @warn "blacklisted base cannot be removed from runtime config because :basecoin column is missing" base=base_upper reason=String(reason)
        return nothing
    end

    rowix = findfirst(==(base_upper), cache.cfg[!, :basecoin])
    if isnothing(rowix)
        (verbosity >= 1) && @warn "blacklisted base not found in runtime config" base=base_upper reason=String(reason)
        return nothing
    end
    cache.cfg = cache.cfg[cache.cfg[!, :basecoin] .!= base_upper, :]
    try
        Xch.removebase!(cache.xc, base_upper)
    catch err
        (verbosity >= 1) && @warn "failed removing blacklisted base from exchange cache" base=base_upper error=sprint(showerror, err)
    end
    (verbosity >= 1) && @warn "removed blacklisted base from trading universe" base=base_upper reason=String(reason)
    return nothing
end


@testset "Blacklisted base removal stays outside runtime until prepare" begin
    EnvConfig.init(EnvConfig.test)

    tc = Trade.TradeCache(xc=Xch.XchCache(), strategy=TradingStrategy.strategyconfig("046"), trademode=Trade.notrade)
    tc.cfg = DataFrame(basecoin=["BTC", "ETH"])

    rt = TradingStrategy.TsCache(classifier=Classify.Classifier011(), strategy=TradingStrategy.StrategyConfig(), source="test")
    rt.accepted = Set(["BTC", "ETH"])
    tc.ts = rt

    @test !isnothing(Trade._strategyruntime(tc))
    @test tc.ts.cfg == rt.cfg

    _blacklistbase!(tc, "BTC", "test")
    @test tc.mc[:blacklistbases] == ["BTC"]
    @test tc.cfg[!, :basecoin] == ["ETH"]
    @test "BTC" in TradingStrategy.acceptedbases(rt)
end

@testset "Runtime API row path" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    tc = Trade.TradeCache(xc=xc, strategy=TradingStrategy.strategyconfig("046"), trademode=Trade.notrade)

    @test !isnothing(Trade._strategyruntime(tc))

    tc.cfg = DataFrame(basecoin=["BTC"])
    tc.xc.currentdt = DateTime("2026-05-30T12:00:00")

    rt = TradingStrategy.TsCache(classifier=Classify.Classifier011(), strategy=TradingStrategy.StrategyConfig(), source="test")
    empty!(TEST_ROWS)
    push!(TEST_ROWS,
        (
                base="BTC",
                tradelabel=Targets.longopen,
                lo_limit=100f0,
                lc_limit=110f0,
                so_limit=0f0,
                sc_limit=0f0,
                probability=0.75f0,
                configid=42,
            ),
    )
    tc.ts = rt

    rowstate = _collect_strategy_row(tc, "BTC")

    @test !isnothing(rowstate)
    @test rowstate.tradesdf[rowstate.rowix, :label] == Targets.longopen
    @test rowstate.tradesdf[rowstate.rowix, :score] == 0.75f0
    @test rowstate.tradesdf[rowstate.rowix, :lo_limit] == 100f0
    @test rowstate.tradesdf[rowstate.rowix, :lc_limit] == 110f0

    @test isempty(TEST_RECONCILIATION_BY_BASE)
    @test isempty(TEST_PREPARE_CALLS)

    histmins = Trade._tradeselection_history_minutes(tc)
    @test histmins >= 2001
end

@testset "Runtime preparation follows selected cfg lifecycle" begin
    EnvConfig.init(EnvConfig.test)

    xc = Xch.XchCache()
    tc = Trade.TradeCache(xc=xc, strategy=TradingStrategy.strategyconfig("046"), trademode=Trade.notrade)
    tc.cfg = DataFrame(basecoin=["BTC", "ETH"])

    rt = TradingStrategy.TsCache(classifier=Classify.Classifier011(), strategy=TradingStrategy.StrategyConfig(), source="test")
    tc.ts = rt

    dt = DateTime("2026-05-30T12:34:00")
    _prepare_strategy_runtime_for_cfg!(tc, dt; updatecache=true)

    @test !isempty(TEST_PREPARE_CALLS)
    @test length(TEST_PREPARE_CALLS) == 1

    preparecall = only(TEST_PREPARE_CALLS)
    @test preparecall.bases == ["BTC", "ETH"]
    @test preparecall.datetime == dt
    @test preparecall.updatecache == true
end

@testset "Apply trading strategy stores runtime only" begin
    EnvConfig.init(EnvConfig.test)

    mc = Dict{Symbol, Any}()
    rt = TradingStrategy.TsCache("046"; source="test")
    gs = TradingStrategy.StrategyConfig(
        ;
        algorithm=TradingStrategy.gain_limit_reversal!,
        openthreshold=0.25f0,
        closethreshold=0.35f0,
        buygain=0.45f0,
        sellgain=0.55f0,
        limitreduction=0.15f0,
        maxwindow=12,
    )

    TradingStrategy.apply_strategy!(rt, gs; source="test")

    @test rt isa TradingStrategy.TsCache
    @test rt.cfg.algorithm == gs.algorithm
    @test rt.cfg.openthreshold == gs.openthreshold
    @test rt.cfg.closethreshold == gs.closethreshold
    @test rt.cfg.buygain == gs.buygain
    @test rt.cfg.sellgain == gs.sellgain
    @test rt.cfg.limitreduction == gs.limitreduction
    @test rt.cfg.maxwindow == gs.maxwindow
    @test rt.source == "test"
    @test !haskey(mc, :strategy_template)
    @test !haskey(mc, :strategy_source)
    @test !haskey(mc, :strategy_openthreshold)
    @test !haskey(mc, :strategy_closethreshold)
    @test !haskey(mc, :strategy_buygain)
    @test !haskey(mc, :strategy_sellgain)
    @test !haskey(mc, :strategy_limitreduction)
    @test !haskey(mc, :strategy_maxwindow)
end
