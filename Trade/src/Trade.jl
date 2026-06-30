# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable trade advice and is responsible to allocate assets.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, Xch, Classify, Features, Targets, TradeLog, TradingStrategy

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

"""
- buysell is the normal trade mode
- closeonly disables opening trades and only closes existing long/short positions
- quickexit sells all assets as soon as possible
- notrade for testing
"""
@enum TradeMode buysell closeonly quickexit notrade

# Backward compatibility alias (deprecated): `sellonly` == `closeonly`.
const sellonly = closeonly

"""
Loop lifecycle states stored in `TradeCache.mc[:loop_state]`.
- `loop_idle`: loop has not been started yet
- `loop_running`: loop is executing ticks
- `loop_paused`: loop is suspended between ticks
- `loop_stopping`: stop has been requested; loop will exit after current tick
- `loop_stopped`: loop has finished (either normally or after stop request)
"""
@enum LoopState loop_idle loop_running loop_paused loop_stopping loop_stopped


"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: essential status messages, e.g. load and save messages, are reported
- 3: print debug info
"""
verbosity = 2

"""Ensure Trades column `longamount` exists. Owner: Trade. Eltype: `Union{Missing,Float32}`."""
function tradesdf_longamount(tradesdf::DataFrame)::DataFrame
    if :longamount ∉ propertynames(tradesdf)
        tradesdf[!, :longamount] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `shortamount` exists. Owner: Trade. Eltype: `Union{Missing,Float32}`."""
function tradesdf_shortamount(tradesdf::DataFrame)::DataFrame
    if :shortamount ∉ propertynames(tradesdf)
        tradesdf[!, :shortamount] = Vector{Union{Missing, Float32}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `longleverage` exists. Owner: Trade. Eltype: `Union{Missing,UInt8}`."""
function tradesdf_longleverage(tradesdf::DataFrame)::DataFrame
    if :longleverage ∉ propertynames(tradesdf)
        tradesdf[!, :longleverage] = Vector{Union{Missing, UInt8}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `shortleverage` exists. Owner: Trade. Eltype: `Union{Missing,UInt8}`."""
function tradesdf_shortleverage(tradesdf::DataFrame)::DataFrame
    if :shortleverage ∉ propertynames(tradesdf)
        tradesdf[!, :shortleverage] = Vector{Union{Missing, UInt8}}(missing, nrow(tradesdf))
    end
    return tradesdf
end

"""Return Trade-contributed Trades schema initializer functions."""
function tradesdf_contributors()::Vector{Function}
    return Function[
        tradesdf_longamount,
        tradesdf_shortamount,
        tradesdf_longleverage,
        tradesdf_shortleverage,
    ]
end

# Extra minute buffer for liquidity lookback window to absorb minute-boundary rounding
# and small OHLCV gaps without underfetching the required continuity check horizon.
const LIQUIDITY_LOOKBACK_MARGIN_MINUTES = 5

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : Float64(sum(assets[!, :usdtvalue]))
end

"Return the effective trading budget in quote currency, capped by `mc[:maxbudgetquote]` when configured."
function _effectivebudgetquote(cache, assets::AbstractDataFrame)::Float64
    totalusdt = _portfoliototal(assets)
    maxbudget = get(cache.mc, :maxbudgetquote, nothing)
    if isnothing(maxbudget)
        return totalusdt
    end
    cap = Float64(maxbudget)
    if !isfinite(cap) || (cap <= 0.0)
        return totalusdt
    end
    return min(totalusdt, cap)
end

"Return the explicit limit price used for order creation in simulation mode."
function _orderlimitprice(cache, price::Real)
    return cache.xc.mc[:simmode] == Xch.bybitsim ? price : nothing
end

function _portfolioquotevalue(assets::AbstractDataFrame)::Union{Missing, Float64}
    if size(assets, 1) == 0 || !any(name -> name == "coin", names(assets))
        return missing
    end
    quoteix = findfirst(==(EnvConfig.pairquote), assets[!, :coin])
    if isnothing(quoteix)
        return missing
    end
    return Float64((assets[quoteix, :free] + assets[quoteix, :locked]) - assets[quoteix, :borrowed])
end

function _writeportfoliosnapshot!(cache, assets::AbstractDataFrame; source_module::AbstractString="Trade")
    rowcount = size(assets, 1)
    simmode = String(Symbol(cache.xc.mc[:simmode]))
    event_time = Dates.now(Dates.UTC)
    portfolio_total = _portfoliototal(assets)
    cash_after = _portfolioquotevalue(assets)
    exchange_name = Xch._routeexchange(cache.xc.routing, Xch.trade_exchange_spot, Xch.exchange(cache.xc))
    account_alias = exchange_name
    try
        if rowcount == 0
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=Xch.tradelogrunmode(cache.xc),
                run_id=Xch.tradelogrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeLog.routing_trade_exchange_spot,
                market_type=TradeLog.market_unknown,
                asset_class=TradeLog.crypto,
                instrument_type=TradeLog.instrument_unknown,
                symbol="PORTFOLIO",
                cash_after=cash_after,
                portfolio_value_after=portfolio_total,
                notes="rows=0; simmode=$(simmode)"
            )
            TradeLog.writeeventwithhash(event)
            return nothing
        end

        hascoin = "coin" in names(assets)
        hasfree = "free" in names(assets)
        haslocked = "locked" in names(assets)
        hasborrowed = "borrowed" in names(assets)
        hasusdtvalue = "usdtvalue" in names(assets)
        for row in eachrow(assets)
            coin = hascoin ? String(row[:coin]) : "UNKNOWN"
            freeqty = hasfree ? Float64(row[:free]) : 0.0
            lockedqty = haslocked ? Float64(row[:locked]) : 0.0
            borrowedqty = hasborrowed ? Float64(row[:borrowed]) : 0.0
            positionqty = freeqty + lockedqty - borrowedqty
            positionvalue = hasusdtvalue ? Float64(row[:usdtvalue]) : missing
            event = TradeLog.AuditEventRow(
                event_type=TradeLog.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=Xch.tradelogrunmode(cache.xc),
                run_id=Xch.tradelogrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeLog.routing_trade_exchange_spot,
                market_type=TradeLog.market_unknown,
                asset_class=TradeLog.crypto,
                instrument_type=TradeLog.spot_pair,
                symbol=coin,
                baseasset=coin,
                quoteasset=EnvConfig.pairquote,
                settlement_asset=EnvConfig.pairquote,
                position_qty_after=positionqty,
                cash_after=(coin == EnvConfig.pairquote ? positionqty : cash_after),
                portfolio_value_after=portfolio_total,
                fill_notional=positionvalue,
                notes="asset=$(coin); rows=$(rowcount); simmode=$(simmode)"
            )
            TradeLog.writeeventwithhash(event)
        end
    catch tradelog_error
        (verbosity >= 1) && @warn "failed to persist portfolio snapshot" exception=(tradelog_error, catch_backtrace())
    end
    return nothing
end

"Write portfolio tradelog snapshots according to `cache.mc[:tradelog_portfolio_snapshot_mode]`."
function _maybe_writeportfoliosnapshot!(cache, assets::AbstractDataFrame)
    mode = get(cache.mc, :tradelog_portfolio_snapshot_mode, :all)
    if mode == :none
        return nothing
    elseif mode == :session_start
        if !get(cache.mc, :tradelog_portfolio_snapshot_written, false)
            _writeportfoliosnapshot!(cache, assets)
            cache.mc[:tradelog_portfolio_snapshot_written] = true
        end
        return nothing
    elseif mode == :all
        _writeportfoliosnapshot!(cache, assets)
        return nothing
    end
    @warn "unknown tradelog portfolio snapshot mode=$(mode); expected :all, :session_start or :none"
    return nothing
end

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::Xch.XchCache  # required to connect to exchange
    cfg::AbstractDataFrame    # maintains the bases to trade and their classifiers
    cl::Classify.AbstractClassifier
    mc::Dict # MC = module constants
    dbgdf
    looplock::ReentrantLock
    loopcond::Threads.Condition
    function TradeCache(; xc=Xch.XchCache(), cl=Classify.Classifier011(), trademode=notrade)
        looplock = ReentrantLock()
        cache = new(xc, DataFrame(), cl, Dict(), DataFrame(), looplock, Threads.Condition(looplock))
        cache.mc[:exitcoins] = [] # exit specific coins
        cache.mc[:longopencoins] = []  # force open long
        cache.mc[:shortopencoins] = [] # force open short
        cache.mc[:restrictedcoins] = String[] # coins excluded from the robot universe (e.g. account-region restrictions)
        cache.mc[:whitelistcoins] = ["ADA", "AI16Z", "APEX", "AAVE", "BNB", "BTC", "CAKE", "DOGE", "ELX", "ENA", "ETH", "HBAR", "HFT", "JUP", "LINK", "LTC", "MNT", "ONDO", "PEPE", "POPCAT", "S", "SOL", "SUI", "TON", "TRX", "VIRTUAL", "W", "WAL", "WIF", "WLD", "X", "XLM", "XRP"] 
        # not whitelisted: "ANIME", "B3", "BERA", "CMETH", "LDO", "PLUME", "SOSO", "TRUMP"
        cache.mc[:hourlygainlimit] = 0.1f0 # limit hourly gain to a realistic 10% max
        cache.mc[:maxassetfraction] = 0.1f0 # defines the maximum ratio of (a specific asset) / ( total assets) - only close trades, if this is exceeded
        cache.mc[:maxbudgetquote] = nothing # optional overall quote-currency budget cap; if set, trading uses min(totalusdt, maxbudgetquote)
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:last_traderefresh_dt] = nothing
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:usenewtrade] = false # implementation switch between old and new trade! method
        cache.mc[:managed_close_orders] = Dict{String, Dict{Symbol, Any}}()  # per-base reconstructed/managed close orders
        cache.mc[:openorders_snapshot] = DataFrame()
        cache.mc[:strategy_runtime] = TradingStrategy.TsCache(classifier=cl, strategy=TradingStrategy.StrategyConfig(), source="default")
        cache.mc[:tradelog_portfolio_snapshot_mode] = :all  # :all, :session_start, :none
        cache.mc[:tradelog_portfolio_snapshot_written] = false
        cache.mc[:loop_state] = loop_idle
        (verbosity >= 4) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes]), exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins]), longopencoins = $(cache.mc[:longopencoins]), shortopencoins = $(cache.mc[:shortopencoins])")
        return cache
    end
end

function _openorderssnapshot(cache::TradeCache)::DataFrame
    oo = get(cache.mc, :openorders_snapshot, DataFrame())
    return oo isa DataFrame ? oo : DataFrame()
end

function _activeopenbuysymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_buy_symbols)
        cache.mc[:active_open_buy_symbols] = Set{String}()
    end
    return cache.mc[:active_open_buy_symbols]
end

function _refreshactiveopenbuysymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopenbuysymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        Xch.openstatus(String(orow.status)) || continue
        lowercase(String(orow.side)) == "buy" || continue
        if hasproperty(orow, :isLeverage) && Bool(getproperty(orow, :isLeverage))
            continue
        end
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

function _activeopensellsymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_sell_symbols)
        cache.mc[:active_open_sell_symbols] = Set{String}()
    end
    return cache.mc[:active_open_sell_symbols]
end

function _refreshactiveopensellsymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopensellsymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        Xch.openstatus(String(orow.status)) || continue
        lowercase(String(orow.side)) == "sell" || continue
        if hasproperty(orow, :isLeverage) && !Bool(getproperty(orow, :isLeverage))
            continue
        end
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

function _rememberactiveopenbuy!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopenbuysymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _rememberactiveopensell!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopensellsymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _hasactiveopenbuy(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopenbuysymbols!(cache)
end

function _hasactiveopensell(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopensellsymbols!(cache)
end

function _tradeselection_history_minutes(tc::TradeCache)::Int
    classifier_minutes = try
        Int(Classify.requiredminutes(tc.cl))
    catch
        0
    end
    liquidity_minutes = Int(Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate + LIQUIDITY_LOOKBACK_MARGIN_MINUTES)
    return max(classifier_minutes + 1, liquidity_minutes, 24 * 60)
end

function _wait_for_live_usdtmarket!(tc::TradeCache, datetime::DateTime; requestedbases::Union{Nothing, AbstractVector{<:AbstractString}}=nothing)
    down_start = Dates.now(Dates.UTC)
    attempts = 0
    quotecoin = uppercase(String(EnvConfig.pairquote))
    requested = isnothing(requestedbases) ? String[] : unique([uppercase(String(b)) for b in requestedbases if !isempty(String(b)) && (uppercase(String(b)) != quotecoin)])
    while true
        marketdf = Xch.screeningUSDTmarket(tc.xc; dt=datetime)
        if size(marketdf, 1) > 0
            if attempts > 0
                downtime = Dates.now(Dates.UTC) - down_start
                @warn "$(quotecoin) market snapshot restored after downtime" datetime attempts downtime
            end
            return marketdf
        end

        # Fallback: query only requested bases to reduce load and isolate pair-specific failures.
        if !isempty(requested)
            scoped = Xch.valuationUSDTmarket(tc.xc, requested; dt=datetime)
            if size(scoped, 1) > 0
                if attempts > 0
                    downtime = Dates.now(Dates.UTC) - down_start
                    @warn "$(quotecoin) market snapshot restored from scoped fallback" datetime attempts downtime symbols=length(requested)
                end
                return scoped
            end
        end

        attempts += 1
        if attempts == 1
            @warn "$(quotecoin) market snapshot unavailable; polling every second until restored" datetime
        elseif attempts % 60 == 0
            @warn "$(quotecoin) market snapshot still unavailable" datetime attempts
        end
        sleep(1)
    end
end

"Use OHLCV-derived marketview in replay/simulation modes instead of persisted trade config snapshots."
function _uses_simulated_marketview(tc::TradeCache)::Bool
    return !isnothing(tc.xc.enddt) || (tc.xc.mc[:simmode] != Xch.nosimulation)
end

@inline function _rowix_at_or_before(opentimes, datetime::DateTime)::Int
    return searchsortedlast(opentimes, datetime)
end

function _rolling_quotevolume24h(df::AbstractDataFrame, endix::Int, enddt::DateTime)::Float64
    startdt = enddt - Day(1)
    ot = df[!, :opentime]
    stopix = min(endix, searchsortedlast(ot, enddt))
    startix = searchsortedlast(ot, startdt) + 1  # strictly greater than startdt
    if (startix > stopix) || (stopix < 1)
        return 0.0
    end
    if :quotevolume in propertynames(df)
        qv = @view df[startix:stopix, :quotevolume]
        return Float64(sum(qv))
    end
    @assert (:basevolume in propertynames(df)) && (:close in propertynames(df)) "OHLCV dataframe must include quotevolume or basevolume+close; names=$(names(df))"
    basevol = @view df[startix:stopix, :basevolume]
    closes = @view df[startix:stopix, :close]
    s = 0.0
    @inbounds for ix in eachindex(basevol)
        s += Float64(basevol[ix]) * Float64(closes[ix])
    end
    return s
end

function _rolling_pricechangepercent24h(df::AbstractDataFrame, endix::Int, enddt::DateTime)::Float32
    startdt = enddt - Day(1)
    ot = df[!, :opentime]
    startix = searchsortedfirst(ot, startdt)
    if !(1 <= startix <= endix)
        return 0f0
    end
    firstclose = Float64(df[startix, :close])
    lastclose = Float64(df[endix, :close])
    if firstclose <= 0.0
        return 0f0
    end
    return Float32(((lastclose / firstclose) - 1.0) * 100.0)
end

"""
Fast liquidity gate for trade selection at `datetime`.

The objective is to admit coins that are liquid overall (24h quote volume gate)
and currently liquid continuously over the recent `checkperiod` window.
"""
function _continuous_liquidity_now(df::AbstractDataFrame, datetime::DateTime;
    minquotevol::Float32=Ohlcv.ld.minquotevol,
    accumulate::Int=Int(Ohlcv.ld.accumulate),
    checkperiod::Int=Int(Ohlcv.ld.checkperiod),
    threshold::Float64=Float64(Ohlcv.ld.startthreshold))::Bool
    rows = size(df, 1)
    rows == 0 && return false
    endix = min(_rowix_at_or_before(df[!, :opentime], datetime), rows)
    endix <= 0 && return false

    required = checkperiod + accumulate - 1
    endix < required && return false

    startix = endix - required + 1
    qv = if :quotevolume in propertynames(df)
        Float32.(df[startix:endix, :quotevolume])
    else
        @assert (:pivot in propertynames(df)) && (:basevolume in propertynames(df)) "OHLCV dataframe must include quotevolume or pivot+basevolume; names=$(names(df))"
        Float32.(df[startix:endix, :pivot] .* df[startix:endix, :basevolume])
    end
    accqv = 0.0f0
    insufficient = 0
    for ix in eachindex(qv)
        accqv += qv[ix]
        if ix > accumulate
            accqv -= qv[ix - accumulate]
        end
        if ix >= accumulate
            insufficient += accqv < minquotevol ? 1 : 0
        end
    end

    startnok = round(Int, checkperiod * threshold)
    return insufficient < startnok
end

function _ensure_marketview_ohlcv!(tc::TradeCache, base::AbstractString, startdt::DateTime, enddt::DateTime, loaded::Set)
    basekey = String(base)
    if basekey in loaded
        ohlcv = Xch.ohlcv(tc.xc, base)
        Xch.cryptoupdate!(tc.xc, ohlcv, startdt, enddt)
        return ohlcv
    end
    ohlcv = Xch.cryptodownload(tc.xc, base, "1m", startdt, enddt)
    push!(loaded, basekey)
    return ohlcv
end

function _ensure_marketview_ohlcv!(tc::TradeCache, base::AbstractString, startdt::DateTime, enddt::DateTime)
    loaded = Set{String}(String.(Xch.bases(tc.xc)))
    return _ensure_marketview_ohlcv!(tc, base, startdt, enddt, loaded)
end

"Build a synthetic USDT market snapshot from OHLCV at `datetime` for simulation/backtest selection."
function _simulated_usdtmarketview(tc::TradeCache, datetime::DateTime, bases::Set{String}, history_startdt::DateTime)::DataFrame
    bases_sorted = sort!(collect(bases))
    loaded = Set{String}(String.(Xch.bases(tc.xc)))
    basecoins = String[]
    quotevolumes = Float64[]
    pricechanges = Float32[]
    lastprices = Float32[]
    sizehint!(basecoins, length(bases_sorted))
    sizehint!(quotevolumes, length(bases_sorted))
    sizehint!(pricechanges, length(bases_sorted))
    sizehint!(lastprices, length(bases_sorted))

    for base in bases_sorted
        isempty(base) && continue
        Xch.validbase(tc.xc, base) || continue
        ohlcv = _ensure_marketview_ohlcv!(tc, base, history_startdt, datetime, loaded)
        df = Ohlcv.dataframe(ohlcv)
        if size(df, 1) == 0
            continue
        end
        rowix = _rowix_at_or_before(df[!, :opentime], datetime)
        if rowix < 1
            continue
        end
        lastprice = Float32(df[rowix, :close])
        quotevolume24h = _rolling_quotevolume24h(df, rowix, datetime)
        pricechangepercent = _rolling_pricechangepercent24h(df, rowix, datetime)
        push!(basecoins, String(base))
        push!(quotevolumes, Float64(quotevolume24h))
        push!(pricechanges, Float32(pricechangepercent))
        push!(lastprices, Float32(lastprice))
    end

    if isempty(basecoins)
        return DataFrame(basecoin=String[], quotevolume24h=Float64[], pricechangepercent=Float32[], lastprice=Float32[])
    end
    return DataFrame(basecoin=basecoins, quotevolume24h=quotevolumes, pricechangepercent=pricechanges, lastprice=lastprices)
end

"""Synchronize `buyenabled` and `sellenabled` flags from the currently computed criteria columns."""
function _sync_tradeflags!(tc::TradeCache; assetonly::Bool=false)
    if assetonly
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :inportfolio]
    else
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& tc.cfg[!, :minquotevol] .&& tc.cfg[!, :continuousminvol] .&& tc.cfg[!, :whitelisted]
        tc.cfg[:, :sellenabled] .= tc.cfg[:, :buyenabled] .|| tc.cfg[!, :inportfolio]
    end
    return tc
end

"Return one optional boolean trade-selection field from a `DataFrameRow`."
function _cfgbool(row::DataFrameRow, field::Symbol, default::Bool=false)::Bool
    if !hasproperty(row, field)
        return default
    end
    value = getproperty(row, field)
    return (ismissing(value) || isnothing(value)) ? default : Bool(value)
end

"Log enriched diagnostics for Kraken margin order failures with expected vs available margin." 
function _log_margin_order_diagnostics(cache::TradeCache, basecfg::DataFrameRow, ta, base::AbstractString, side::AbstractString, requested_leverage::Signed, requested_limitprice::Union{Nothing, Real}, basequantity::Real, freebase::Real, borrowedbase::Real, freeusdt::Real, totalborrowedusdt::Real, effectivebudgetquote::Real, err)
    symbol = Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)
    additional_base = max(0.0, Float64(basequantity) - Float64(freebase))
    requested_limitprice_value = isnothing(requested_limitprice) ? missing : Float64(requested_limitprice)
    expected_margin_quote = isnothing(requested_limitprice) ? missing : (additional_base * Float64(requested_limitprice))
    limits = Xch.marginlimits(cache.xc, symbol; role=Xch.trade_exchange_spot)
    @error "margin order submission failed" exchange=Xch.exchange(cache.xc) base=String(base) symbol=String(symbol) side=String(side) tradelabel=String(Symbol(ta.tradelabel)) requested_leverage=requested_leverage requested_baseqty=Float64(basequantity) requested_limitprice=requested_limitprice_value expected_margin_quote=expected_margin_quote available_free_quote=Float64(freeusdt) freebase=Float64(freebase) borrowedbase=Float64(borrowedbase) totalborrowedquote=Float64(totalborrowedusdt) effectivebudgetquote=Float64(effectivebudgetquote) buyenabled=_cfgbool(basecfg, :buyenabled, false) sellenabled=_cfgbool(basecfg, :sellenabled, false) inportfolio=_cfgbool(basecfg, :inportfolio, false) maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell error_message=sprint(showerror, err)
end

"Return true when an order error indicates exchange/account permission restrictions for the symbol."
function _ispermissionrestrictederror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("invalid permissions", msg) || occursin("trading restricted", msg) || occursin("permission denied", msg)
end

"Return true when an order error indicates temporary/per-position funding insufficiency."
function _isinsufficientfundserror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("insufficient funds", msg)
end

"Return true when Kraken private-read cooldown/rate-limit transiently blocks order flow." 
function _isprivatecooldownerror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("private read cooldown", msg) || occursin("rate limit", msg)
end

"Return true when an order error indicates the target order no longer exists (race with fill/cancel)."
function _isunknownordererror(err)::Bool
    msg = lowercase(sprint(showerror, err))
    return occursin("unknown order", msg) || occursin("order not found", msg)
end

"Disable trading flags for one base in the current runtime config to avoid repeated restricted-order attempts."
function _disablerestrictedbase!(cache::TradeCache, base::AbstractString, reason::AbstractString)::Nothing
    base_upper = uppercase(String(base))
    restricted = get!(cache.mc, :restrictedcoins, String[])
    !(base_upper in restricted) && push!(restricted, base_upper)

    if !hasproperty(cache.cfg, :basecoin)
        (verbosity >= 1) && @warn "permission-restricted base cannot be removed from runtime config because :basecoin column is missing" base=base_upper reason=String(reason)
        return nothing
    end

    rowix = findfirst(==(base_upper), cache.cfg[!, :basecoin])
    if isnothing(rowix)
        (verbosity >= 1) && @warn "permission-restricted base not found in runtime config" base=base_upper reason=String(reason)
        return nothing
    end
    cache.cfg = cache.cfg[cache.cfg[!, :basecoin] .!= base_upper, :]
    try
        Xch.removebase!(cache.xc, base_upper)
    catch err
        (verbosity >= 1) && @warn "failed removing restricted base from exchange cache" base=base_upper error=sprint(showerror, err)
    end
    (verbosity >= 1) && @warn "removed restricted base from trading universe" base=base_upper reason=String(reason)
    return nothing
end

"Return normalized set of base coins excluded from trading by runtime restrictions."
function _restrictedbaseset(tc::TradeCache, quotecoin::AbstractString)::Set{String}
    tokens = get(tc.mc, :restrictedcoins, String[])
    normalized = [_normalize_basecoin_token(x, quotecoin) for x in tokens]
    return Set(String.(filter(!isnothing, normalized)))
end

"Normalize a base/pair token to a base coin symbol for the configured quote coin."
function _normalize_basecoin_token(token, quotecoin::AbstractString)::Union{Nothing, String}
    q = uppercase(String(quotecoin))
    t = uppercase(strip(String(token)))
    isempty(t) && return nothing
    t == q && return nothing
    if occursin('/', t)
        parts = split(t, '/'; limit=2)
        length(parts) == 2 || return nothing
        base, quotetoken = parts
        quotetoken == q || return nothing
        base == q && return nothing
        return base
    end
    if endswith(t, q)
        base = t[1:(end - length(q))]
        isempty(base) && return nothing
        base == q && return nothing
        return base
    end
    return t
end

"""
Loads all USDT coins, checks liquidity volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored, consisting of the union of buyenabled and sellenabled pairs.
`assetonly` is an input parameter to limit coins for backtesting.
"""
function tradeselection!(tc::TradeCache, assetbases::Vector; datetime=tc.xc.startdt, assetonly=false, updatecache=false)
    datetime = floor(datetime, Minute(1))
    quotecoin = uppercase(EnvConfig.pairquote)
    assetbase_tokens = [_normalize_basecoin_token(x, quotecoin) for x in assetbases]
    whitelist_tokens = [_normalize_basecoin_token(x, quotecoin) for x in tc.mc[:whitelistcoins]]
    assetbaseset = Set{String}(String.(filter(!isnothing, assetbase_tokens)))
    portfolioassetbaseset = copy(assetbaseset)
    whitelistset = Set{String}(String.(filter(!isnothing, whitelist_tokens)))
    restrictedset = _restrictedbaseset(tc, quotecoin)
    assetbaseset = setdiff(assetbaseset, restrictedset)
    whitelistset = setdiff(whitelistset, restrictedset)
    if !assetonly
        balancesdf = Xch.balances(tc.xc; ignoresmallvolume=false)
        if size(balancesdf, 1) > 0
            hasfree = :free in names(balancesdf)
            haslocked = :locked in names(balancesdf)
            hasborrowed = :borrowed in names(balancesdf)
            for row in eachrow(balancesdf)
                base = _normalize_basecoin_token(row.coin, quotecoin)
                isnothing(base) && continue
                freeqty = hasfree ? Float64(row.free) : 0.0
                lockedqty = haslocked ? Float64(row.locked) : 0.0
                borrowedqty = hasborrowed ? Float64(row.borrowed) : 0.0
                if (abs(freeqty) + abs(lockedqty) + abs(borrowedqty)) > 0.0
                    push!(portfolioassetbaseset, String(base))
                end
            end
        end
        # Keep restricted held bases in inportfolio to allow close/monitor flows.
        # Restricted filtering is still applied to non-portfolio candidate expansion.
    end
    history_minutes = _tradeselection_history_minutes(tc)
    history_startdt = datetime - Minute(history_minutes)

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # Xch.removeallbases(tc.xc)  #* reuse what is in cache
    # Classify.removebase!(tc.cl, nothing)  #* reuse what is in cache

    marketbases = assetonly ? Set(String.(collect(portfolioassetbaseset))) : Set(String.(collect(union(portfolioassetbaseset, whitelistset, Set(String.(Xch.bases(tc.xc)))))))
    marketbases = union(portfolioassetbaseset, setdiff(marketbases, restrictedset))
    if _uses_simulated_marketview(tc)
        usdtdf = _simulated_usdtmarketview(tc, datetime, marketbases, history_startdt)
        if size(usdtdf, 1) == 0
            requestedbases = filter(!=(quotecoin), collect(marketbases))
            error("empty simulated marketview at datetime=$(datetime), requestedbases=$(requestedbases). Check OHLCV availability for configured bases.")
        end
    else
        usdtdf = Xch.screeningUSDTmarket(tc.xc; dt=datetime)  # superset of coins with 24h volume price change and last price
        if size(usdtdf, 1) == 0
            usdtdf = _wait_for_live_usdtmarket!(tc, datetime; requestedbases=collect(marketbases))
        end
        if assetonly
            usdtdf = filter(row -> row.basecoin in assetbaseset, usdtdf)
        end
    end
    if !isempty(restrictedset) && (size(usdtdf, 1) > 0)
        usdtdf = filter(row -> !((String(row.basecoin) in restrictedset) && !(String(row.basecoin) in portfolioassetbaseset)), usdtdf)
    end
    if !assetonly
        knownbases = Set(String.(usdtdf[!, :basecoin]))
        missingportfoliobases = setdiff(portfolioassetbaseset, knownbases)
        if !isempty(missingportfoliobases)
            valuationdf = Xch.valuationUSDTmarket(tc.xc, collect(missingportfoliobases); dt=datetime)
            for row in eachrow(valuationdf)
                base = String(row.basecoin)
                if ((base in restrictedset) && !(base in portfolioassetbaseset)) || (base in knownbases)
                    continue
                end
                push!(usdtdf, (
                    basecoin=base,
                    quotevolume24h=Float32(row.quotevolume24h),
                    pricechangepercent=Float32(row.pricechangepercent),
                    lastprice=Float32(row.lastprice),
                    askprice=Float32(row.askprice),
                    bidprice=Float32(row.bidprice),
                ))
                push!(knownbases, base)
            end
        end
    end
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent, :lastprice)
    if size(tc.cfg, 1) == 0
        tc.cfg[:, :datetime] = DateTime[]
        tc.cfg[:, :minquotevol] = Bool[]
        tc.cfg[:, :continuousminvol] = Bool[]
        tc.cfg[:, :inportfolio] = Bool[]
        tc.cfg[:, :classifieraccepted] = Bool[]
        tc.cfg[:, :buyenabled] = Bool[]
        tc.cfg[:, :sellenabled] = Bool[]
        tc.cfg[:, :whitelisted] = Bool[]
        (verbosity >= 1) && @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :datetime] .= datetime
    # tc.cfg[:, :validbase] = [Xch.validbase(tc.xc, base) for base in tc.cfg[!, :basecoin]] # is already filtered by getUSDTmarket
    minimumdayquotevolumemillion = round(Ohlcv.liquiddailyminimumquotevolume() / 1000000, digits=0) # ignore allcoins with less than liquiddailyminimumquotevolume
    tc.cfg[:, :minquotevol] = tc.cfg[:, :quotevolume24h_M] .>= minimumdayquotevolumemillion
    tc.cfg[:, :continuousminvol] .= false
    tc.cfg[:, :inportfolio] = [base in portfolioassetbaseset for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :classifieraccepted] .= false
    tc.cfg[:, :buyenabled] .= false
    tc.cfg[:, :sellenabled] .= false
    tc.cfg[:, :whitelisted] = [base in whitelistset for base in tc.cfg[!, :basecoin]]

    # download latest OHLCV and classifier features
    tc.cfg = tc.cfg[tc.cfg[:, :minquotevol] .|| tc.cfg[:, :inportfolio], :]
    (verbosity >= 3) && println("#minquotevol=$(sum(tc.cfg[:, :minquotevol])) #inportfolio=$(sum(tc.cfg[:, :inportfolio]))")
    count = size(tc.cfg, 1)
    xcbases = Xch.bases(tc.xc)
    removebases = setdiff(xcbases, tc.cfg[!, :basecoin])
    for rb in removebases  # remove coins that were loaded but are no longer part of the new configuration
        Xch.removebase!(tc.xc, rb)
        Classify.removebase!(tc.cl, rb)
    end
    xcbaseset = Set(Xch.bases(tc.xc))
    candidatebaseset = Set{String}()
    (verbosity >= 3) && println("trade selection history window=$(history_minutes) minutes from $(history_startdt) to $(datetime)")
    for (ix, row) in enumerate(eachrow(tc.cfg))
        (verbosity >= 2) && updatecache &&  print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) including cache update                           ")
        (verbosity >= 2) && !updatecache && print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) without cache update                             ")
        if row.basecoin in xcbaseset
            ohlcv = Xch.ohlcv(tc.xc, row.basecoin)
            Xch.cryptoupdate!(tc.xc, ohlcv, history_startdt, datetime)
        else
            ohlcv = Xch.cryptodownload(tc.xc, row.basecoin, "1m", history_startdt, datetime)
        end
        if updatecache
            Ohlcv.write(ohlcv) # write ohlcv even if data length is too short to calculate features
        end
        row.continuousminvol = true #TODO check disabled until debugged _continuous_liquidity_now(Ohlcv.dataframe(ohlcv), datetime, minquotevol=5000f0, accumulate=60, checkperiod=24*60, threshold=0.8)
        if row.inportfolio || (row.whitelisted && row.minquotevol)
            push!(candidatebaseset, String(row.basecoin))
        end
    end

    # Keep classifier/feature workload limited to liquidity candidates and portfolio holdings.
    for rb in setdiff(Set(Xch.bases(tc.xc)), candidatebaseset)
        Xch.removebase!(tc.xc, rb)
        Classify.removebase!(tc.cl, rb)
    end

    classifierloadedset = Set(String.(Classify.bases(tc.cl)))
    for row in eachrow(tc.cfg)
        base = String(row.basecoin)
        if (base in candidatebaseset) && !(base in classifierloadedset)
            Classify.addbase!(tc.cl, Xch.ohlcv(tc.xc, base))
            push!(classifierloadedset, base)
        end
    end

    if !isempty(classifierloadedset)
        Classify.supplement!(tc.cl)
        if updatecache
            Classify.writetargetsfeatures(tc.cl)
        end
    end
    xcbases = Xch.bases(tc.xc)
    classifierbases = Classify.bases(tc.cl)
    remove_xc_bases = setdiff(xcbases, classifierbases)
    for rb in remove_xc_bases  # remove coins not accepted by classifier (e.g. insufficient requiredminutes)
        Xch.removebase!(tc.xc, rb)
    end
    remove_classifier_bases = setdiff(classifierbases, xcbases)
    for rb in remove_classifier_bases  # drop stale classifier-only bases that are no longer in the exchange cache
        Classify.removebase!(tc.cl, rb)
    end
    xcbases = Xch.bases(tc.xc)
    classifierbases = Classify.bases(tc.cl)
    classifierbaseset = Set(classifierbases)
    @assert Set(xcbases) == classifierbaseset "Set(xcbases)=$(xcbases) != Set(classifierbases)=$(classifierbases)"

    tc.cfg[:, :classifieraccepted] = [base in classifierbaseset for base in tc.cfg[!, :basecoin]]
    _sync_tradeflags!(tc; assetonly=assetonly)
    (verbosity >= 4) && println("$(Xch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    # tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    (verbosity >= 2) && println("$(EnvConfig.now()) #tc.cfg=$(size(tc.cfg, 1)) sum(classifieraccepted)=$(sum(tc.cfg[!, :classifieraccepted])) classifierbases($(length(classifierbases)))=$(classifierbases) ")

    if !assetonly
        (verbosity >= 2) && println("\r$(Xch.ttstr(tc.xc)) trained trade config on the fly including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    end
    return tc
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2


currenttime(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :opentime]
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :close]
closelongset = [shortstrongopen, shortopen, shorthold, allclose, longstrongclose, longclose]
closeshortset = [shortclose, shortstrongclose, allclose, longhold, longopen, longstrongopen]

mutable struct Investment  #TODO bookkeeping for consistency checks
    investmentid
    tradeadvice::Vector{Classify.TradeAdvice} # vector of all used trade advices
    orderid::Vector # vector of all used orders
    classifiername
    configid
end

_isclosetrade(tl) = tl in [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
_isopentrade(tl) = tl in [shortstrongopen, shortopen, longopen, longstrongopen]
_isopenshorttrade(tl) = tl in [shortstrongopen, shortopen]

_traderank(tl) = _isclosetrade(tl) ? 1 : _isopentrade(tl) ? 2 : 3

function _tradetolabeltext(label)
    return String(Symbol(label))
end

function _withtradelogcontext(f::Function, cache::TradeCache, ta)
    signal_score = try
        Float64(ta.probability)
    catch
        missing
    end
    strategy_ref = _strategyruntime(cache).source
    Xch.settradelogcontext!(
        cache.xc;
        strategy_engine="tradingstrategy",
        strategy_config_ref=strategy_ref,
        signal_label=_tradetolabeltext(ta.tradelabel),
        signal_score=signal_score,
    )
    try
        return f()
    finally
        Xch.cleartradelogcontext!(cache.xc)
    end
end

function _requested_limitprice(cache::TradeCache, ta, fallback_price::Real)
    return isnothing(ta.price) ? _orderlimitprice(cache, fallback_price) : Float32(ta.price)
end

function _material_order_change(old_price, new_price, old_qty::Real, new_qty::Real; price_reltol::Real=1f-3, qty_reltol::Real=1f-3)::Bool
    oldp = (ismissing(old_price) || isnothing(old_price)) ? nothing : Float32(old_price)
    newp = (ismissing(new_price) || isnothing(new_price)) ? nothing : Float32(new_price)
    if isnothing(oldp) != isnothing(newp)
        return true
    end
    if !isnothing(oldp) && !isnothing(newp)
        denom = max(abs(oldp), 1f-6)
        if abs(newp - oldp) / denom > Float32(price_reltol)
            return true
        end
    end

    oldq = Float32(old_qty)
    newq = Float32(new_qty)
    qdenom = max(abs(oldq), 1f-6)
    return abs(newq - oldq) / qdenom > Float32(qty_reltol)
end

function trade!(cache::TradeCache, basecfg::DataFrameRow, ta, assets::AbstractDataFrame)
    sellbuyqtyratio = 2 # longclose qty / longbuy qty per order, if > 1 longclose quicker than buying it
    qtyacceleration = 4 # if > 1 then increase longbuy and longclose order qty by this factor
    short_margin_leverage = 2
    result = nothing
    base = ta.base
    totalusdt = sum(assets.usdtvalue)
    if totalusdt <= 0
        @warn "totalusdt=$totalusdt is insufficient, assets=$assets"
        return nothing
    end
    effectivebudgetquote = _effectivebudgetquote(cache, assets)
    if effectivebudgetquote <= 0
        (verbosity > 2) && println("$(tradetime(cache)) skip $base: effectivebudgetquote=$effectivebudgetquote is insufficient")
        return nothing
    end
    basequantity = missing
    freeusdtfractionmargin = 0.05
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.pairquote, :free]) - totalborrowedusdt
    freebase = sum(assets[assets[!, :coin] .== base, :free]) *(1-eps(Float32))
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    quotequantity = cache.mc[:maxassetfraction] * effectivebudgetquote / 10  # distribute over 10 trades within effective budget
    ohlcv = Xch.ohlcv(cache.xc, base)
    price = currentprice(ohlcv)
    symbol = Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)
    @assert base == ohlcv.base == ta.base
    minimumbasequantity = Xch.minimumbasequantity(cache.xc, base, price)
    if isnothing(minimumbasequantity)
        (verbosity > 2) && println("$(tradetime(cache)) skip $base due to missing minimum base quantity at price=$price")
        return nothing
    end
    # (verbosity > 2) && println("$(tradetime(cache)) entry $base , $(ta.tradelabel)")
    # Xch.portfolio subtracts the borrowed amount from usdtvalue of each base
    tradelabel = ta.tradelabel
    if (cache.mc[:trademode] == quickexit) || (base in cache.mc[:exitcoins])
        tradelabel = allclose
    end
    if (tradelabel in [allclose, longhold]) && (borrowedbase > 0)
        tradelabel = shortclose
    end
    if (tradelabel in [allclose, shorthold]) && (freebase > 0)
        tradelabel = longclose
    end
    if (tradelabel in [allclose, shorthold, longhold])
        return nothing
    end
    if (tradelabel in [longstrongclose, longclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
        existing = _managedcloseget(cache, base, longclose)
        closeablelong = freebase
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeablelong <= 2 * minimumbasequantity ? (closeablelong >= minimumbasequantity ? closeablelong : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), closeablelong)
        sufficientsellbalance = (basequantity <= closeablelong) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if sufficientsellbalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                existing_limit = get(existing, :limitprice, nothing)
                existing_qty = Float32(get(existing, :baseqty, 0f0))
                if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity)
                    oid = String(existing[:orderid])
                else
                    amended = try
                        (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradelogcontext(cache, ta) do
                            Xch.changeorder(cache.xc, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                        end
                    catch err
                        if _isunknownordererror(err)
                            (verbosity >= 1) && @warn "managed longclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            _managedcloseclear!(cache, base, longclose)
                            nothing
                        else
                            rethrow(err)
                        end
                    end
                    if !isnothing(amended)
                        oid = amended
                    else
                        if !isnothing(_managedcloseget(cache, base, longclose))
                            try
                                Xch.cancelorder(cache.xc, base, String(existing[:orderid]))
                            catch err
                                (verbosity >= 1) && @warn "failed to cancel managed longclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            end
                            _managedcloseclear!(cache, base, longclose)
                        end
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : _withtradelogcontext(cache, ta) do
                    Xch.createsellorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0)
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, longclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=longclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no longclose $base due to sufficientsellbalance=$sufficientsellbalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (tradelabel in [longopen, longstrongopen]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(0f0, min(max(qtyacceleration * quotequantity/price, minimumbasequantity) * price, freeusdt - freeusdtfractionmargin * effectivebudgetquote) / price) #* keep 5% * effective budget as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && ((basequantity + borrowedbase) > 0.0)
        # basequantity += borrowedbase # buy all short as well when switching to long
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / effectivebudgetquote
        existing_longbuy_oid = let
            oo = _openorderssnapshot(cache)
            found = nothing
            for orow in eachrow(oo)
                Xch.openstatus(String(orow.status)) || continue
                (String(orow.symbol) == symbol) || continue
                (String(orow.side) == "Buy") || continue
                if hasproperty(orow, :isLeverage) && Bool(getproperty(orow, :isLeverage))
                    continue
                end
                found = String(orow.orderid)
                break
            end
            found
        end
        # basefraction = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt)

        # if base == "ADA"
        #     println("coin=$(ta.base) tradelabel=$(ta.tradelabel) price=$price basequantity=$basequantity sufficientbuybalance=$sufficientbuybalance minimumbasequantity=$minimumbasequantity quotequantity=$quotequantity freeusdt=$freeusdt totalusdt=$totalusdt")
        # end
    
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 3) && println("$(tradetime(cache)) skip $base longbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif !isnothing(existing_longbuy_oid) || _hasactiveopenbuy(cache, symbol)
            (verbosity >= 2) && println("$(tradetime(cache)) skip $base longbuy: existing open longbuy order $(existing_longbuy_oid) is still active")
        elseif sufficientbuybalance && exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : _withtradelogcontext(cache, ta) do
                Xch.createbuyorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=0)
            end
            if !isnothing(oid)
                result = (trade=longopen, oid=oid)
                _rememberactiveopenbuy!(cache, symbol)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.pairquote, :free])), EnvConfig.pairquote=$(EnvConfig.pairquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base longbuy due to sufficientbuybalance=$sufficientbuybalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
        elseif (tradelabel in [shortstrongopen, shortopen]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(qtyacceleration * quotequantity / price, minimumbasequantity)
        sufficientbuybalance = ((basequantity - freebase) * price < freeusdt) && (basequantity > 0.0)
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / (effectivebudgetquote + totalborrowedusdt)
        marginok = Xch.marginpermitted(cache.xc, symbol, "Sell", short_margin_leverage; role=Xch.trade_exchange_spot)
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif _hasactiveopensell(cache, symbol)
            (verbosity >= 2) && println("$(tradetime(cache)) skip $base shortbuy: existing open short-entry sell order is still active")
        elseif !marginok
            limits = Xch.marginlimits(cache.xc, symbol; role=Xch.trade_exchange_spot)
            (verbosity >= 1) && @warn "skip $base shortbuy due to Kraken margin metadata limits" symbol=symbol requested_leverage=short_margin_leverage maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell
        elseif sufficientbuybalance
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : _withtradelogcontext(cache, ta) do
                try
                    Xch.createsellorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage)
                catch err
                    _log_margin_order_diagnostics(cache, basecfg, ta, base, "Sell", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeusdt, totalborrowedusdt, effectivebudgetquote, err)
                    rethrow(err)
                end
            end
            if !isnothing(oid)
                result = (trade=shortopen, oid=oid)
                _rememberactiveopensell!(cache, symbol)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortbuy order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortbuy order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * effectivebudgetquote <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.pairquote, :free])), EnvConfig.pairquote=$(EnvConfig.pairquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base shortbuy due to sufficientbuybalance=$sufficientbuybalance")
        end
    elseif (tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, closeonly, quickexit]) && basecfg.sellenabled
        existing = _managedcloseget(cache, base, shortclose)
        closeableshort = borrowedbase
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = closeableshort <= 2 * minimumbasequantity ? (closeableshort >= minimumbasequantity ? closeableshort : minimumbasequantity) : minimumbasequantity # increase minimumbasequantity if otherwise a too small base amount remains that cannot be sold
        basequantity = max(0f0, min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), closeableshort))
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if exceedsminimumbasequantity
            requested_limitprice = _requested_limitprice(cache, ta, price)
            oid = nothing
            if !isnothing(existing)
                existing_limit = get(existing, :limitprice, nothing)
                existing_qty = Float32(get(existing, :baseqty, 0f0))
                if !_material_order_change(existing_limit, requested_limitprice, existing_qty, basequantity)
                    oid = String(existing[:orderid])
                else
                    amended = try
                        (cache.mc[:trademode] == notrade) ? String(existing[:orderid]) : _withtradelogcontext(cache, ta) do
                            Xch.changeorder(cache.xc, String(existing[:orderid]); limitprice=requested_limitprice, basequantity=basequantity)
                        end
                    catch err
                        if _isunknownordererror(err)
                            (verbosity >= 1) && @warn "managed shortclose amend skipped because order is no longer present" base=base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            _managedcloseclear!(cache, base, shortclose)
                            nothing
                        else
                            rethrow(err)
                        end
                    end
                    if !isnothing(amended)
                        oid = amended
                    else
                        if !isnothing(_managedcloseget(cache, base, shortclose))
                            try
                                Xch.cancelorder(cache.xc, base, String(existing[:orderid]))
                            catch err
                                (verbosity >= 1) && @warn "failed to cancel managed shortclose before recreate" base orderid=String(existing[:orderid]) error=sprint(showerror, err)
                            end
                            _managedcloseclear!(cache, base, shortclose)
                        end
                    end
                end
            end
            if isnothing(oid)
                oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : _withtradelogcontext(cache, ta) do
                    try
                        Xch.createbuyorder(cache.xc, base; limitprice=requested_limitprice, basequantity=basequantity, maker=true, marginleverage=short_margin_leverage, reduceonly=true)
                    catch err
                        _log_margin_order_diagnostics(cache, basecfg, ta, base, "Buy", short_margin_leverage, requested_limitprice, basequantity, freebase, borrowedbase, freeusdt, totalborrowedusdt, effectivebudgetquote, err)
                        rethrow(err)
                    end
                end
            end
            if !isnothing(oid)
                _managedcloseset!(cache, base, oid, shortclose; limitprice=requested_limitprice, baseqty=basequantity)
                result = (trade=shortclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortclose order with oid $oid, limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortclose order with limitprice=$requested_limitprice and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no shortclose $base due to exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    end
    push!(cache.dbgdf, (
        taconfigid = isnothing(ta) ? missing : ta.configid,
        tatradelabel = isnothing(ta) ? missing : tradelabel,
        tabase = isnothing(ta) ? missing : ta.base,
        tahourlygain = isnothing(ta) ? missing : ta.hourlygain,
        oid = isnothing(result) ? missing : result.oid,
        baseqty = basequantity,
        minimumbasequantity = minimumbasequantity,
        freebase = freebase,
        totalborrowedusdt = totalborrowedusdt,
        freeusdt = freeusdt,
        quoteqty = quotequantity,
        price = price,
        opentime=currenttime(ohlcv)
    ), promote=true)
    if !isnothing(result)

    end 
    return result
end

tradetime(cache::TradeCache) = Xch.ttstr(cache.xc)
# USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")
function USDTmsg(assets)
    totalusdt = sum(assets.usdtvalue)
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.pairquote, :free]) - totalborrowedusdt
    freepct = totalusdt > 0f0 ? min(100, round(Int, freeusdt / totalusdt * 100)) : 0
    return string("$(EnvConfig.pairquote): total=$(round(Int, totalusdt)), quotefree=$(freepct)%")
end
function tradeadvicelessthan(ta1, ta2)
    closeset = [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
    buyset = [shortstrongopen, shortopen, longopen, longstrongopen]
    holdset = [shorthold, longhold]
    if (ta1.tradelabel in closeset) && !(ta2.tradelabel in closeset)
        return true
    elseif (ta1.tradelabel in buyset) && (ta2.tradelabel in buyset)
        if ta1.hourlygain < ta2.hourlygain
            return true
        end
    end
    return false
end

# ── Loop control ────────────────────────────────────────────────────────────

"Returns the current loop lifecycle state."
_loopstate_nolock(cache::TradeCache) = LoopState(Int(cache.mc[:loop_state]))
_setloopstate_nolock!(cache::TradeCache, s::LoopState) = (cache.mc[:loop_state] = s; nothing)

function _setloopstate!(cache::TradeCache, s::LoopState)
    lock(cache.looplock)
    try
        _setloopstate_nolock!(cache, s)
        notify(cache.loopcond; all=true)
    finally
        unlock(cache.looplock)
    end
    return nothing
end

function _waitforactive_loopstate!(cache::TradeCache)
    lock(cache.looplock)
    try
        while _loopstate_nolock(cache) == loop_paused
            wait(cache.loopcond)
        end
        return _loopstate_nolock(cache)
    finally
        unlock(cache.looplock)
    end
end

# ── Strategy config ─────────────────────────────────────────────────────────

function _validatestrategyconfig!(spec::TradingStrategy.StrategyConfig)
    openthreshold = Float32(spec.openthreshold)
    closethreshold = Float32(spec.closethreshold)
    buygain = Float32(spec.buygain)
    sellgain = Float32(spec.sellgain)
    limitreduction = Float32(spec.limitreduction)
    maxwindow = Int(spec.maxwindow)

    @assert 0f0 <= openthreshold <= 1f0 "strategy_openthreshold must be in [0, 1], got $(openthreshold)"
    @assert 0f0 <= closethreshold <= 1f0 "strategy_closethreshold must be in [0, 1], got $(closethreshold)"
    @assert 0f0 <= buygain <= 1f0 "strategy_buygain must be in [0, 1], got $(buygain)"
    @assert 0f0 <= sellgain <= 1f0 "strategy_sellgain must be in [0, 1], got $(sellgain)"
    @assert 0f0 <= limitreduction <= 1f0 "strategy_limitreduction must be in [0, 1], got $(limitreduction)"
    @assert maxwindow > 0 "strategy_maxwindow must be > 0, got $(maxwindow)"
    return spec
end

"Validate strategy runtime parameters stored in `TradeCache.mc[:strategy_runtime]`."
function _validatestrategyconfig!(cache::TradeCache)
    _validatestrategyconfig!(_strategyruntime(cache).strategy_config)
    return cache
end

"Apply strategy runtime settings from a `TradingStrategy.StrategyConfig` and reset derived per-base state."
function apply_tradingstrategy!(mc::AbstractDict, spec::TradingStrategy.StrategyConfig; source::AbstractString="manual")
    _validatestrategyconfig!(spec)

    managed_close_orders = get!(mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
    empty!(managed_close_orders)

    if haskey(mc, :strategy_runtime) && (mc[:strategy_runtime] isa TradingStrategy.TsCache)
        rt = mc[:strategy_runtime]
        TradingStrategy.apply_strategy!(rt, spec; source=source)
    else
        mc[:strategy_runtime] = TradingStrategy.TsCache(strategy=spec, source=source)
    end
    return mc
end

function apply_tradingstrategy!(cache::TradeCache, spec::TradingStrategy.StrategyConfig; source::AbstractString="manual")
    apply_tradingstrategy!(cache.mc, spec; source=source)
    return cache
end

function _strategyruntime(cache::TradeCache)::TradingStrategy.TsCache
    if !haskey(cache.mc, :strategy_runtime) || !(cache.mc[:strategy_runtime] isa TradingStrategy.TsCache)
        cache.mc[:strategy_runtime] = TradingStrategy.TsCache(
            classifier=cache.cl,
            strategy=TradingStrategy.StrategyConfig(),
            source="default",
        )
    end
    return cache.mc[:strategy_runtime]
end

function _prepare_strategy_runtime_for_cfg!(cache::TradeCache, datetime::DateTime; updatecache::Bool=false)::Nothing
    hasproperty(cache.cfg, :basecoin) || return nothing
    size(cache.cfg, 1) == 0 && return nothing

    rt = _strategyruntime(cache)
    bases = String.(cache.cfg[!, :basecoin])
    isempty(bases) && return nothing

    TradingStrategy.preparebases!(rt, cache.xc, bases; datetime=datetime, updatecache=updatecache)
    return nothing
end

function _managedclosestate(cache::TradeCache)::Dict{String, Dict{Symbol, Any}}
    return get!(cache.mc, :managed_close_orders, Dict{String, Dict{Symbol, Any}}())
end

function _managedcloseside(tradelabel::Targets.TradeLabel)::String
    if tradelabel in [longclose, longstrongclose]
        return "Sell"
    elseif tradelabel in [shortclose, shortstrongclose]
        return "Buy"
    end
    throw(ArgumentError("managed close label=$(tradelabel) must be a close label"))
end

function _managedclosekey(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)::String
    symbol = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)))
    return string(symbol, "|", _managedcloseside(tradelabel))
end

function _managedclosekey(base::AbstractString, tradelabel::Targets.TradeLabel)::String
    # Backward-compatible helper for tests/callers that only provide base.
    symbol = uppercase(String(base)) * uppercase(String(EnvConfig.pairquote))
    return string(symbol, "|", _managedcloseside(tradelabel))
end

function _managedcloseget(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)
    return get(_managedclosestate(cache), _managedclosekey(cache, base, tradelabel), nothing)
end

function _managedcloseset!(cache::TradeCache, base::AbstractString, orderid, tradelabel::Targets.TradeLabel; limitprice=nothing, baseqty::Real=0f0)
    symbol = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)))
    _managedclosestate(cache)[_managedclosekey(cache, base, tradelabel)] = Dict{Symbol, Any}(
        :base => uppercase(String(base)),
        :symbol => symbol,
        :orderid => String(orderid),
        :tradelabel => tradelabel,
        :limitprice => isnothing(limitprice) ? nothing : Float32(limitprice),
        :baseqty => Float32(baseqty),
        :updated => Dates.now(Dates.UTC),
    )
    return nothing
end

function _managedcloseclear!(cache::TradeCache, base::AbstractString, tradelabel::Targets.TradeLabel)
    delete!(_managedclosestate(cache), _managedclosekey(cache, base, tradelabel))
    return nothing
end

function _positioncloselabels(assets::AbstractDataFrame, base::AbstractString; sellenabled::Bool=true)::Vector{Targets.TradeLabel}
    basekey = uppercase(String(base))
    freebase = Float32(sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :free]))
    borrowedbase = Float32(sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :borrowed]))
    labels = Targets.TradeLabel[]
    if sellenabled && (freebase > 0f0)
        push!(labels, longclose)
    end
    if sellenabled && (borrowedbase > 0f0)
        push!(labels, shortclose)
    end
    return labels
end

function _cfgrow_for_base(cache::TradeCache, base::AbstractString)
    hasproperty(cache.cfg, :basecoin) || return nothing
    rowix = findfirst(==(uppercase(String(base))), uppercase.(String.(cache.cfg[!, :basecoin])))
    if isnothing(rowix)
        return nothing
    end
    return cache.cfg[rowix, :]
end

function _basecfg_for_close(cache::TradeCache, base::AbstractString, sellenabled::Bool)::DataFrameRow
    cfgrow = _cfgrow_for_base(cache, base)
    if !isnothing(cfgrow)
        return cfgrow
    end

    # Fallback row allows close-only maintenance for held bases that are not in runtime cfg.
    fallback = DataFrame(
        basecoin=[uppercase(String(base))],
        buyenabled=[false],
        sellenabled=[Bool(sellenabled)],
        classifieraccepted=[false],
        inportfolio=[true],
        minquotevol=[false],
        continuousminvol=[false],
        whitelisted=[false],
        datetime=[isnothing(cache.xc.currentdt) ? Dates.now() : cache.xc.currentdt],
    )
    return fallback[1, :]
end

function _close_management_bases(cache::TradeCache, assets::AbstractDataFrame)::Vector{String}
    quote_coin = uppercase(String(EnvConfig.pairquote))
    bases = String[]
    if hasproperty(cache.cfg, :basecoin)
        for base in String.(cache.cfg[!, :basecoin])
            push!(bases, uppercase(base))
        end
    end
    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        (base == quote_coin) && continue
        freebase = Float32(getproperty(row, :free))
        borrowedbase = Float32(getproperty(row, :borrowed))
        if (freebase > 0f0) || (borrowedbase > 0f0)
            push!(bases, base)
        end
    end
    return unique(bases)
end

function _orderbase_from_symbol(cache::TradeCache, assets::AbstractDataFrame, symbol::AbstractString)::Union{Nothing, String}
    sym = uppercase(String(symbol))
    for base in _close_management_bases(cache, assets)
        expected = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)))
        if sym == expected
            return base
        end
    end
    return nothing
end

function _managed_close_label_from_order(orow)::Union{Nothing, Targets.TradeLabel}
    side = uppercase(String(getproperty(orow, :side)))
    is_leverage = hasproperty(orow, :isLeverage) ? Bool(getproperty(orow, :isLeverage)) : false
    if (side == "SELL") && !is_leverage
        return longclose
    elseif (side == "BUY") && is_leverage
        return shortclose
    end
    return nothing
end

function _managed_order_baseqty(orow)::Float32
    if hasproperty(orow, :baseqty)
        return Float32(getproperty(orow, :baseqty))
    elseif hasproperty(orow, :qty)
        return Float32(getproperty(orow, :qty))
    end
    return 0f0
end

function _managed_order_limitprice(orow)
    if hasproperty(orow, :price)
        v = Float32(getproperty(orow, :price))
        return v > 0f0 ? v : nothing
    elseif hasproperty(orow, :limitprice)
        v = Float32(getproperty(orow, :limitprice))
        return v > 0f0 ? v : nothing
    end
    return nothing
end

function _reconstruct_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    state = _managedclosestate(cache)
    empty!(state)
    for orow in eachrow(oo)
        Xch.openstatus(String(getproperty(orow, :status))) || continue
        closelabel = _managed_close_label_from_order(orow)
        isnothing(closelabel) && continue
        base = _orderbase_from_symbol(cache, assets, String(getproperty(orow, :symbol)))
        isnothing(base) && continue
        _managedcloseset!(
            cache,
            base,
            String(getproperty(orow, :orderid)),
            closelabel;
            limitprice=_managed_order_limitprice(orow),
            baseqty=_managed_order_baseqty(orow),
        )
    end
    return nothing
end

function _cancel_unmanaged_open_orders!(cache::TradeCache, oo::AbstractDataFrame)
    managed_ids = Set{String}(String(v[:orderid]) for v in values(_managedclosestate(cache)))
    for orow in eachrow(oo)
        Xch.openstatus(String(getproperty(orow, :status))) || continue
        oid = String(getproperty(orow, :orderid))
        oid in managed_ids && continue
        symbol = String(getproperty(orow, :symbol))
        base = _normalize_basecoin_token(symbol, EnvConfig.pairquote)
        isnothing(base) && continue
        try
            Xch.cancelorder(cache.xc, String(base), oid)
            (verbosity >= 1) && @warn "cancelled unmanaged open order" base=String(base) symbol=symbol orderid=oid
        catch err
            if _isunknownordererror(err)
                continue
            elseif _isprivatecooldownerror(err)
                (verbosity >= 1) && @warn "skip unmanaged-order cancellation due to transient private-read cooldown" base=String(base) symbol=symbol orderid=oid error=sprint(showerror, err)
            else
                rethrow(err)
            end
        end
    end
    return nothing
end

function _advicebybase(tradeadvices)::Dict{String, Any}
    bybase = Dict{String, Any}()
    for ta in tradeadvices
        base = uppercase(String(ta.base))
        haskey(bybase, base) && continue
        bybase[base] = ta
    end
    return bybase
end

function _ensure_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, tradeadvices)
    advbybase = _advicebybase(tradeadvices)
    for base in _close_management_bases(cache, assets)
        cfgrow = _cfgrow_for_base(cache, base)
        sellenabled = isnothing(cfgrow) ? true : _cfgbool(cfgrow, :sellenabled, true)
        basecfg = _basecfg_for_close(cache, base, sellenabled)
        for closelabel in _positioncloselabels(assets, base; sellenabled=sellenabled)

            baseadvice = haskey(advbybase, base) ? advbybase[base] : (
                classifier=cache.cl,
                configid=0,
                tradelabel=ignore,
                relativeamount=1f0,
                base=base,
                price=nothing,
                datetime=isnothing(cache.xc.currentdt) ? Dates.now() : cache.xc.currentdt,
                hourlygain=0f0,
                probability=0f0,
                investmentid=nothing,
                source=:managedclose,
                allowreversal=false,
            )
            ta = (
                classifier=baseadvice.classifier,
                configid=baseadvice.configid,
                tradelabel=closelabel,
                relativeamount=baseadvice.relativeamount,
                base=base,
                price=nothing,
                datetime=baseadvice.datetime,
                hourlygain=baseadvice.hourlygain,
                probability=baseadvice.probability,
                investmentid=baseadvice.investmentid,
                source=:managedclose,
                allowreversal=false,
            )

            try
                trade!(cache, basecfg, ta, assets)
            catch err
                if _ispermissionrestrictederror(err)
                    _disablerestrictedbase!(cache, base, sprint(showerror, err))
                elseif _isinsufficientfundserror(err)
                    (verbosity >= 1) && @warn "skip managed close order due to insufficient funds" base=base error=sprint(showerror, err)
                elseif _isprivatecooldownerror(err)
                    (verbosity >= 1) && @warn "skip managed close order due to transient private-read cooldown" base=base error=sprint(showerror, err)
                else
                    rethrow(err)
                end
            end
        end
    end
    return nothing
end

function _advice_price_from_row(tdf::AbstractDataFrame, rowix::Integer, label::Targets.TradeLabel)
    if label in (longopen, longstrongopen)
        return tdf[rowix, :longopenlimit]
    elseif label in (shortopen, shortstrongopen)
        return tdf[rowix, :shortopenlimit]
    elseif label in (longclose, longstrongclose)
        return tdf[rowix, :longcloselimit]
    elseif label in (shortclose, shortstrongclose)
        return tdf[rowix, :shortcloselimit]
    end
    return nothing
end

"Collects the TradingStrategy advice for all bases in the current TradeCache configuration and returns a vector of trade advice tuples."
function _collect_strategy_advices(cache::TradeCache)
    dt = cache.xc.currentdt
    isnothing(dt) && return Any[]

    bases = hasproperty(cache.cfg, :basecoin) ? String.(cache.cfg[!, :basecoin]) : String[]
    isempty(bases) && return Any[]

    rt = _strategyruntime(cache)

    rows = TradingStrategy.gettradesrows!(rt, cache.xc, bases, dt) # get advoices for all bases in the current cfg of teh current sample time
    tradeadvices = Any[]
    for rowmeta in rows
        tdf = rowmeta.tradesdf
        rowix = Int(rowmeta.rowix)
        label = tdf[rowix, :tradelabel]
        label == ignore && continue
        rawprice = _advice_price_from_row(tdf, rowix, label)
        price = (ismissing(rawprice) || isnothing(rawprice)) ? nothing : Float32(rawprice)
        push!(tradeadvices, (
            classifier=cache.cl,
            configid=Int(rowmeta.configid),
            tradelabel=label,
            relativeamount=1f0,
            base=String(rowmeta.base),
            price=price,
            datetime=dt,
            hourlygain=0f0,
            probability=Float32(rowmeta.probability),
            investmentid=nothing,
            source=:tradingstrategy,
            allowreversal=false,
        ))
    end
    return tradeadvices
end

function _should_refresh_tradeselection(cache::TradeCache)::Bool
    currentdt = cache.xc.currentdt
    if isnothing(currentdt)
        return false
    end
    refresh_times = get(cache.mc, :reloadtimes, Time[])
    currentminute = floor(currentdt, Minute(1))
    if !(Time(currentminute) in refresh_times)
        return false
    end
    lastrefresh = get(cache.mc, :last_traderefresh_dt, nothing)
    return isnothing(lastrefresh) || (lastrefresh != currentminute)
end

function _mark_tradeselection_refreshed!(cache::TradeCache)
    currentdt = cache.xc.currentdt
    cache.mc[:last_traderefresh_dt] = isnothing(currentdt) ? nothing : floor(currentdt, Minute(1))
    return cache
end

function _maybe_refresh_tradeselection!(cache::TradeCache; assets::Union{Nothing, AbstractDataFrame}=nothing)
    if !_should_refresh_tradeselection(cache)
        return false
    end
    assets_df = isnothing(assets) ? Xch.portfolio!(cache.xc) : assets
    (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
    tradeselection!(cache, assets_df[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
    cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
    _prepare_strategy_runtime_for_cfg!(cache, cache.xc.currentdt; updatecache=true)
    _mark_tradeselection_refreshed!(cache)
    (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
    return true
end

"Return position-side gaps where no matching open close order currently exists."
function _positions_without_close_orders(cache::TradeCache, assets::AbstractDataFrame, oo::AbstractDataFrame)
    quote_coin = uppercase(String(EnvConfig.pairquote))
    missing = NamedTuple{(:base, :side, :qty, :required, :covered, :minimum), Tuple{String, String, Float32, Float32, Float32, Float32}}[]

    function _managed_covered_qty(symbol::AbstractString, side::AbstractString)::Float32
        total = 0f0
        sideu = uppercase(String(side))
        symbolu = uppercase(String(symbol))
        for (mkey, managed) in pairs(_managedclosestate(cache))
            msymbol = ""
            if haskey(managed, :symbol)
                msymbol = uppercase(String(get(managed, :symbol, "")))
            elseif haskey(managed, :base)
                mbase = String(get(managed, :base, ""))
                if !isempty(mbase)
                    msymbol = uppercase(String(Xch.symboltoken(cache.xc, mbase, EnvConfig.pairquote; role=Xch.trade_exchange_spot)))
                end
            end
            if isempty(msymbol)
                msymbol = uppercase(String(split(String(mkey), "|"; limit=2)[1]))
            end
            (msymbol == symbolu) || continue
            mlabel = get(managed, :tradelabel, ignore)
            if (sideu == "SELL") && (mlabel in [longclose, longstrongclose])
                total += Float32(get(managed, :baseqty, 0f0))
            elseif (sideu == "BUY") && (mlabel in [shortclose, shortstrongclose])
                total += Float32(get(managed, :baseqty, 0f0))
            end
        end
        return total
    end

    function _remaining_open_qty(orow)::Float32
        total = Float32(getproperty(orow, :baseqty))
        executed = hasproperty(orow, :executedqty) ? Float32(getproperty(orow, :executedqty)) : 0f0
        return max(0f0, total - executed)
    end

    function _covered_qty(symbol::AbstractString, side::AbstractString; require_leverage::Union{Nothing, Bool}=nothing)::Float32
        wanted_side = uppercase(String(side))
        total = 0f0
        for orow in eachrow(oo)
            Xch.openstatus(String(orow.status)) || continue
            (String(orow.symbol) == String(symbol)) || continue
            (uppercase(String(orow.side)) == wanted_side) || continue
            if !isnothing(require_leverage)
                if hasproperty(orow, :isLeverage)
                    (Bool(getproperty(orow, :isLeverage)) == require_leverage) || continue
                end
            end
            total += _remaining_open_qty(orow)
        end
        return total + _managed_covered_qty(symbol, side)
    end

    function _min_base_qty(base::AbstractString, symbol::AbstractString, row)::Float32
        price = try
            hasproperty(row, :usdtprice) ? Float32(getproperty(row, :usdtprice)) : 0f0
        catch
            0f0
        end
        if !(price > 0f0)
            price = try
                Float32(currentprice(Xch.ohlcv(cache.xc, base)))
            catch
                0f0
            end
        end
        syminfo = try
            Xch.minimumqty(cache.xc, symbol)
        catch
            nothing
        end
        if isnothing(syminfo)
            return 0f0
        end
        if price > 0f0
            return Float32(1.01f0 * max(Float32(syminfo.minbaseqty), Float32(syminfo.minquoteqty) / price))
        end
        return Float32(1.01f0 * Float32(syminfo.minbaseqty))
    end

    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == quote_coin && continue
        freebase = Float32(row.free)
        borrowedbase = Float32(row.borrowed)
        if (freebase <= 0f0) && (borrowedbase <= 0f0)
            continue
        end

        symbol = Xch.symboltoken(cache.xc, base, EnvConfig.pairquote; role=Xch.trade_exchange_spot)
        minimumbasequantity = _min_base_qty(base, symbol, row)
        if freebase > 0f0
            # Long-close orders are spot sells (non-leverage). Exclude short-entry margin sells.
            sell_covered = _covered_qty(symbol, "Sell"; require_leverage=false)
            sell_gap = max(0f0, freebase - sell_covered)
            (sell_gap >= minimumbasequantity) && push!(missing, (base=base, side="Sell", qty=sell_gap, required=freebase, covered=sell_covered, minimum=minimumbasequantity))
        end
        if borrowedbase > 0f0
            # Short-close orders are margin buys. Exclude long-entry spot buys.
            buy_covered = _covered_qty(symbol, "Buy"; require_leverage=true)
            buy_gap = max(0f0, borrowedbase - buy_covered)
            (buy_gap >= minimumbasequantity) && push!(missing, (base=base, side="Buy", qty=buy_gap, required=borrowedbase, covered=buy_covered, minimum=minimumbasequantity))
        end
    end
    return missing
end

"""
Execute one trading tick: reconstruct managed close-order state from live open orders,
cancel unmanaged orders, keep one close order active per open robot-owned position,
execute open/reversal entries, and handle daily trade-selection reload.
Called by the loop runners once per iterate step.
"""
function _tradestep!(cache::TradeCache)
    (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
    oo = try
        Xch.getopenorders(cache.xc)
    catch err
        if _isprivatecooldownerror(err)
            prev = _openorderssnapshot(cache)
            if size(prev, 1) > 0
                (verbosity >= 1) && @warn "using previous openorders snapshot due to transient private-read cooldown" rows=size(prev, 1) error=sprint(showerror, err)
                prev
            else
                (verbosity >= 1) && @warn "skip tradestep due to transient private-read cooldown and missing openorders snapshot" error=sprint(showerror, err)
                return nothing
            end
        else
            rethrow(err)
        end
    end
    cache.mc[:openorders_snapshot] = oo
    _refreshactiveopenbuysymbols!(cache, oo)
    _refreshactiveopensellsymbols!(cache, oo)
    assets = Xch.portfolio!(cache.xc)
    _reconstruct_managed_close_orders!(cache, assets, oo)
    _cancel_unmanaged_open_orders!(cache, oo)
    _maybe_writeportfoliosnapshot!(cache, assets)
    tradeadvices = _collect_strategy_advices(cache) # advice from TradingStrategy for all bases in runtime cfg
    _ensure_managed_close_orders!(cache, assets, tradeadvices)
    if cache.mc[:usenewtrade]
    else # legacy trade!()
        sellbases = []
        buybases = []
        sort!(tradeadvices, lt=tradeadvicelessthan)  # close first, then buy high-gain first
        for ta in tradeadvices
            rowix = hasproperty(cache.cfg, :basecoin) ? findfirst(==(ta.base), cache.cfg[!, :basecoin]) : nothing
            if isnothing(rowix)
                (verbosity >= 1) && @warn "skip trade advice because base is missing in runtime config" base=ta.base
                continue
            end
            basecfg = cache.cfg[rowix, :]
            res = try
                trade!(cache, basecfg, ta, assets)
            catch err
                if _ispermissionrestrictederror(err)
                    _disablerestrictedbase!(cache, ta.base, sprint(showerror, err))
                    nothing
                elseif _isinsufficientfundserror(err)
                    (verbosity >= 1) && @warn "skip trade advice due to insufficient funds" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                elseif _isprivatecooldownerror(err)
                    (verbosity >= 1) && @warn "skip trade advice due to transient private-read cooldown" base=ta.base tradelabel=String(Symbol(ta.tradelabel)) error=sprint(showerror, err)
                    nothing
                else
                    rethrow(err)
                end
            end
            if !isnothing(res) && (res.trade in [longopen, longstrongopen, shortclose, shortstrongclose])
                push!(buybases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [longstrongclose, longclose, shortstrongopen, shortopen])
                push!(sellbases, basecfg.basecoin)
            elseif !isnothing(res)
                @warn "case not handled: $res"
            end
        end
        (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                          ")
    end

    # Avoid extra live API calls: only refresh post-trade portfolio in simulation mode.
    assets_after = (cache.xc.mc[:simmode] == Xch.nosimulation) ? assets : Xch.portfolio!(cache.xc)

    # Live safety summary: highlight open positions that currently have no opposite-side close order.
    # In simulation mode orders can fill immediately, so there is no persistent open-order coverage to inspect.
    if (cache.xc.mc[:simmode] == Xch.nosimulation) && (cache.mc[:trademode] in [buysell, closeonly, quickexit])
        oo_after = oo
        cache.mc[:openorders_snapshot] = oo_after
        missing = _positions_without_close_orders(cache, assets_after, oo_after)
        if !isempty(missing)
            details = ["$(x.base):$(x.side):gap=$(round(Float64(x.qty); digits=6)) req=$(round(Float64(x.required); digits=6)) cov=$(round(Float64(x.covered); digits=6)) min=$(round(Float64(x.minimum); digits=6))" for x in missing]
            (verbosity >= 1) && @warn "open positions without active close order" count=length(missing) details=details
        end
    end

    _maybe_refresh_tradeselection!(cache; assets=assets_after)
    #TODO low prio: for closed orders check fees
    #TODO low prio: aggregate orders and transactions in bookkeeping
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = Xch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start calculating trading strategy on the fly")
        tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        _prepare_strategy_runtime_for_cfg!(cache, cache.xc.startdt; updatecache=false)
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
end

"Log end-of-loop summary statistics."
function _tradefinish!(cache::TradeCache)
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    (verbosity >= 3) && @info (size(cache.xc.closedorders, 1) > 0) ? "$(EnvConfig.now()): closed orders log $(cache.xc.closedorders)" : "$(EnvConfig.now()): no closed orders"
    (verbosity >= 3) && @info (size(cache.xc.orders, 1) > 0) ? "$(EnvConfig.now()): open orders log $(cache.xc.orders)" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): closed orders $(size(cache.xc.closedorders, 1)), open orders $(size(cache.xc.orders, 1))"
    assets = Xch.portfolio!(cache.xc)
    (verbosity >= 3) && @info "assets = $assets"
    (verbosity >= 2) && @info "total $(EnvConfig.pairquote) = $(sum(assets.usdtvalue))"
end

"""
Shared iteration engine used by both backtest and live runners.
Advances through `cache.xc` one tick at a time, calling `_tradestep!` each step.
Respects `pause!`/`resume!`/`stop!` loop control requests.
"""
function _run_tradeloop!(cache::TradeCache)
    _setloopstate!(cache, loop_running)
    try
        for c in cache.xc
            st = _waitforactive_loopstate!(cache)
            (st == loop_stopping) && break
            _tradestep!(cache)
        end
    catch ex
        if isa(ex, InterruptException)
            (verbosity >= 0) && println("\nCtrl+C pressed within tradeloop")
        else
            (verbosity >= 0) && @error "exception=$ex"
            bt = catch_backtrace()
            for ptr in bt
                frame = StackTraces.lookup(ptr)
                for fr in frame
                    if occursin("CryptoTimeSeries", string(fr.file))
                        (verbosity >= 1) && println("fr.func=$(fr.func) fr.file=$(fr.file) fr.line=$(fr.line)")
                    end
                end
            end
        end
    finally
        _setloopstate!(cache, loop_stopped)
    end
    _tradefinish!(cache)
    return cache
end

"""
Run a full backtest replay over the cached OHLCV window defined by `cache.xc.startdt`…`cache.xc.enddt`.
When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.
"""
function run_backtest!(cache::TradeCache; skip_init::Bool=false)
    skip_init || _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

"""
Run the live trading loop, advancing one minute per tick and sleeping until the next wall-clock minute.
When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.
"""
function run_live!(cache::TradeCache; skip_init::Bool=false)
    skip_init || _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

end  # module

