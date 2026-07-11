# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable trade advice and is responsible to allocate assets.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, Xch, Features, Targets, TradingStrategy

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

"""Ensure Trades column `lo_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-open consumed by Xch order processing."""
function tradesdf_lo_amount(tradesdf::DataFrame)::DataFrame
    if :lo_amount ∉ propertynames(tradesdf)
        tradesdf[!, :lo_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `lc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for long-close consumed by Xch order processing."""
function tradesdf_lc_amount(tradesdf::DataFrame)::DataFrame
    if :lc_amount ∉ propertynames(tradesdf)
        tradesdf[!, :lc_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `so_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-open consumed by Xch order processing."""
function tradesdf_so_amount(tradesdf::DataFrame)::DataFrame
    if :so_amount ∉ propertynames(tradesdf)
        tradesdf[!, :so_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Ensure Trades column `sc_amount` exists. Owner: Trade. Eltype: `Float32` with `0f0` as the default. Note: Request order size for short-close consumed by Xch order processing."""
function tradesdf_sc_amount(tradesdf::DataFrame)::DataFrame
    if :sc_amount ∉ propertynames(tradesdf)
        tradesdf[!, :sc_amount] = fill(0f0, nrow(tradesdf))
    end
    return tradesdf
end

"""Return Trade-contributed Trades schema initializer functions."""
function tradesdf_contributors()::Vector{Function}
    return Function[
        tradesdf_lo_amount,
        tradesdf_lc_amount,
        tradesdf_so_amount,
        tradesdf_sc_amount,
    ]
end

# Extra minute buffer for liquidity lookback window to absorb minute-boundary rounding
# and small OHLCV gaps without underfetching the required continuity check horizon.
const LIQUIDITY_LOOKBACK_MARGIN_MINUTES = 5

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : (sum(assets[!, :usdtvalue]))
end

"Return the effective trading budget in quote currency, capped by `mc[:maxbudgetquote]` when configured."
function _effectivebudgetquote(cache, assets::AbstractDataFrame)::Float64
    totalusdt = _portfoliototal(assets)
    maxbudget = get(cache.mc, :maxbudgetquote, nothing)
    if isnothing(maxbudget)
        return totalusdt
    end
    cap = (maxbudget)
    if !isfinite(cap) || (cap <= 0.0)
        return totalusdt
    end
    return min(totalusdt, cap)
end

"Return the explicit limit price used for order creation in simulation mode."
function _orderlimitprice(cache, price::Real)
    return Xch.exchange(cache.xc) == Xch.EXCHANGE_BYBITSIM ? price : nothing
end

function _portfolioquotevalue(assets::AbstractDataFrame)::Union{Missing, Float64}
    if size(assets, 1) == 0 || !any(name -> name == "coin", names(assets))
        return missing
    end
    quoteix = findfirst(==(EnvConfig.pairquote), assets[!, :coin])
    if isnothing(quoteix)
        return missing
    end
    return ((assets[quoteix, :free] + assets[quoteix, :locked]) - assets[quoteix, :borrowed])
end

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::Xch.XchCache  # required to connect to exchange
    cfg::AbstractDataFrame    # maintains the bases to trade and their strategy acceptance state
    ts::TradingStrategy.TsCache
    mc::Dict # MC = module constants
    looplock::ReentrantLock
    loopcond::Threads.Condition
    function TradeCache(; xc=Xch.XchCache(), strategy=TradingStrategy.TsCache("046"; source="default"), trademode=notrade)
        looplock = ReentrantLock()
        ts = strategy isa TradingStrategy.TsCache ? strategy : strategy isa AbstractString ? TradingStrategy.TsCache(strategy; source="default") : strategy isa TradingStrategy.StrategyConfig ? TradingStrategy.TsCache(strategy=strategy, source="default") : throw(ArgumentError("strategy must be TradingStrategy.TsCache, TradingStrategy.StrategyConfig, or a config ref string"))
        cache = new(xc, DataFrame(), ts, Dict(), looplock, Threads.Condition(looplock))
        cache.mc[:blacklistbases] = String[] # bases excluded from new trading; held positions may still be closed
        cache.mc[:maxassetfraction] = 0.1f0 # defines the maximum ratio of (a specific asset) / ( total assets) - only close trades, if this is exceeded
        cache.mc[:maxbudgetquote] = nothing # optional overall quote-currency budget cap; if set, trading uses min(totalusdt, maxbudgetquote)
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:last_traderefresh_dt] = nothing
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:managed_close_orders] = Dict{String, Dict{Symbol, Any}}()  # per-base reconstructed/managed close orders
        cache.mc[:loop_state] = loop_idle
        (verbosity >= 4) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), maxbudgetquote = $(cache.mc[:maxbudgetquote]), reloadtimes = $(cache.mc[:reloadtimes]), blacklistbases = $(cache.mc[:blacklistbases])")
        return cache
    end
end

function _orderreduceonly(orow)::Bool
    if hasproperty(orow, :reduceonly)
        return Bool(getproperty(orow, :reduceonly))
    elseif hasproperty(orow, :reduceOnly)
        return Bool(getproperty(orow, :reduceOnly))
    end
    return false
end

function _orderislongclose(orow)::Bool
    side = uppercase(String(getproperty(orow, :side)))
    side == "SELL" || return false
    _orderreduceonly(orow) && return true
    if hasproperty(orow, :isLeverage)
        return !Bool(getproperty(orow, :isLeverage))
    end
    return false
end

function _orderisshortclose(orow)::Bool
    side = uppercase(String(getproperty(orow, :side)))
    return (side == "BUY") && _orderreduceonly(orow)
end

function _orderislongentry(orow)::Bool
    side = uppercase(String(getproperty(orow, :side)))
    return (side == "BUY") && !_orderisshortclose(orow)
end

function _orderisshortentry(orow)::Bool
    side = uppercase(String(getproperty(orow, :side)))
    return (side == "SELL") && !_orderislongclose(orow)
end

function _activeopenlongsymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_long_symbols)
        cache.mc[:active_open_long_symbols] = Set{String}()
    end
    return cache.mc[:active_open_long_symbols]
end

function _refreshactiveopenlongsymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopenlongsymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        Xch.openstatus(String(orow.status)) || continue
        _orderislongentry(orow) || continue
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

function _activeopenshortsymbols!(cache::TradeCache)::Set{String}
    if !haskey(cache.mc, :active_open_short_symbols)
        cache.mc[:active_open_short_symbols] = Set{String}()
    end
    return cache.mc[:active_open_short_symbols]
end

function _refreshactiveopenshortsymbols!(cache::TradeCache, oo::AbstractDataFrame)
    active = _activeopenshortsymbols!(cache)
    empty!(active)
    for orow in eachrow(oo)
        Xch.openstatus(String(orow.status)) || continue
        _orderisshortentry(orow) || continue
        push!(active, uppercase(String(orow.symbol)))
    end
    return active
end

function _rememberactiveopenlong!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopenlongsymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _rememberactiveopenshort!(cache::TradeCache, symbol::AbstractString)
    push!(_activeopenshortsymbols!(cache), uppercase(String(symbol)))
    return cache
end

function _hasactiveopenlong(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopenlongsymbols!(cache)
end

function _hasactiveopenshort(cache::TradeCache, symbol::AbstractString)::Bool
    return uppercase(String(symbol)) in _activeopenshortsymbols!(cache)
end

function _tradeselection_history_minutes(tc::TradeCache)::Int
    classifier_minutes = try
        Int(TradingStrategy.requiredhistoryminutes(_strategyruntime(tc)))
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
    return !isnothing(tc.xc.enddt)
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
        return (sum(qv))
    end
    @assert (:basevolume in propertynames(df)) && (:close in propertynames(df)) "OHLCV dataframe must include quotevolume or basevolume+close; names=$(names(df))"
    basevol = @view df[startix:stopix, :basevolume]
    closes = @view df[startix:stopix, :close]
    s = 0.0
    @inbounds for ix in eachindex(basevol)
        s += (basevol[ix]) * (closes[ix])
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
    firstclose = (df[startix, :close])
    lastclose = (df[endix, :close])
    if firstclose <= 0.0
        return 0f0
    end
    return (((lastclose / firstclose) - 1.0) * 100.0)
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
    threshold::Float64=(Ohlcv.ld.startthreshold))::Bool
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
        lastprice = (df[rowix, :close])
        quotevolume24h = _rolling_quotevolume24h(df, rowix, datetime)
        pricechangepercent = _rolling_pricechangepercent24h(df, rowix, datetime)
        push!(basecoins, String(base))
        push!(quotevolumes, (quotevolume24h))
        push!(pricechanges, (pricechangepercent))
        push!(lastprices, (lastprice))
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
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& tc.cfg[!, :minquotevol] .&& tc.cfg[!, :continuousminvol] .&& .!tc.cfg[!, :blacklisted]
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
    symbol = Xch.symboltoken(cache.xc, base, EnvConfig.pairquote)
    additional_base = max(0.0, (basequantity) - (freebase))
    requested_limitprice_value = isnothing(requested_limitprice) ? missing : (requested_limitprice)
    expected_margin_quote = isnothing(requested_limitprice) ? missing : (additional_base * (requested_limitprice))
    limits = Xch.marginlimits(cache.xc, symbol)
    @error "margin order submission failed" exchange=Xch.exchange(cache.xc) base=String(base) symbol=String(symbol) side=String(side) tradelabel=String(Symbol(ta.tradelabel)) requested_leverage=requested_leverage requested_baseqty=(basequantity) requested_limitprice=requested_limitprice_value expected_margin_quote=expected_margin_quote available_free_quote=(freeusdt) freebase=(freebase) borrowedbase=(borrowedbase) totalborrowedquote=(totalborrowedusdt) effectivebudgetquote=(effectivebudgetquote) buyenabled=_cfgbool(basecfg, :buyenabled, false) sellenabled=_cfgbool(basecfg, :sellenabled, false) inportfolio=_cfgbool(basecfg, :inportfolio, false) maxleveragebuy=limits.maxleveragebuy maxleveragesell=limits.maxleveragesell error_message=sprint(showerror, err)
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

"Return normalized set of base coins excluded from new trading by runtime configuration."
function _blacklistbaseset(tc::TradeCache, quotecoin::AbstractString)::Set{String}
    tokens = get(tc.mc, :blacklistbases, String[])
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
    assetbaseset = Set{String}(String.(filter(!isnothing, assetbase_tokens)))
    portfolioassetbaseset = copy(assetbaseset)
    blacklistset = _blacklistbaseset(tc, quotecoin)
    assetbaseset = setdiff(assetbaseset, blacklistset)
    if !assetonly
        balancesdf = Xch.balances(tc.xc; ignoresmallvolume=false)
        if size(balancesdf, 1) > 0
            hasfree = :free in names(balancesdf)
            haslocked = :locked in names(balancesdf)
            hasborrowed = :borrowed in names(balancesdf)
            for row in eachrow(balancesdf)
                base = _normalize_basecoin_token(row.coin, quotecoin)
                isnothing(base) && continue
                freeqty = hasfree ? (row.free) : 0.0
                lockedqty = haslocked ? (row.locked) : 0.0
                borrowedqty = hasborrowed ? (row.borrowed) : 0.0
                if (abs(freeqty) + abs(lockedqty) + abs(borrowedqty)) > 0.0
                    push!(portfolioassetbaseset, String(base))
                end
            end
        end
        # Keep blacklisted held bases in inportfolio to allow close/monitor flows.
        # Blacklist filtering is still applied to non-portfolio candidate expansion.
    end
    history_minutes = _tradeselection_history_minutes(tc)
    history_startdt = datetime - Minute(history_minutes)

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # Xch.removeallbases(tc.xc)  #* reuse what is in cache

    marketbases = assetonly ? Set(String.(collect(portfolioassetbaseset))) : Set(String.(collect(union(portfolioassetbaseset, Set(String.(Xch.bases(tc.xc)))))))
    marketbases = union(portfolioassetbaseset, setdiff(marketbases, blacklistset))
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
    if !isempty(blacklistset) && (size(usdtdf, 1) > 0)
        usdtdf = filter(row -> !((String(row.basecoin) in blacklistset) && !(String(row.basecoin) in portfolioassetbaseset)), usdtdf)
    end
    if !assetonly
        knownbases = Set(String.(usdtdf[!, :basecoin]))
        missingportfoliobases = setdiff(portfolioassetbaseset, knownbases)
        if !isempty(missingportfoliobases)
            valuationdf = Xch.valuationUSDTmarket(tc.xc, collect(missingportfoliobases); dt=datetime)
            for row in eachrow(valuationdf)
                base = String(row.basecoin)
                if ((base in blacklistset) && !(base in portfolioassetbaseset)) || (base in knownbases)
                    continue
                end
                push!(usdtdf, (
                    basecoin=base,
                    quotevolume24h=(row.quotevolume24h),
                    pricechangepercent=(row.pricechangepercent),
                    lastprice=(row.lastprice),
                    askprice=(row.askprice),
                    bidprice=(row.bidprice),
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
        tc.cfg[:, :blacklisted] = Bool[]
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
    tc.cfg[:, :blacklisted] = [base in blacklistset for base in tc.cfg[!, :basecoin]]

    # download latest OHLCV and classifier features
    tc.cfg = tc.cfg[tc.cfg[:, :minquotevol] .|| tc.cfg[:, :inportfolio], :]
    (verbosity >= 3) && println("#minquotevol=$(sum(tc.cfg[:, :minquotevol])) #inportfolio=$(sum(tc.cfg[:, :inportfolio]))")
    count = size(tc.cfg, 1)
    xcbases = Xch.bases(tc.xc)
    removebases = setdiff(xcbases, tc.cfg[!, :basecoin])
    for rb in removebases  # remove coins that were loaded but are no longer part of the new configuration
        Xch.removebase!(tc.xc, rb)
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
        if row.inportfolio || (row.minquotevol && !row.blacklisted)
            push!(candidatebaseset, String(row.basecoin))
        end
    end

    # Keep classifier/feature workload limited to liquidity candidates and portfolio holdings.
    for rb in setdiff(Set(Xch.bases(tc.xc)), candidatebaseset)
        Xch.removebase!(tc.xc, rb)
    end
    rt = _strategyruntime(tc)
    TradingStrategy.preparebases!(rt, tc.xc, sort!(collect(candidatebaseset)); datetime=datetime, updatecache=updatecache)
    xcbases = Xch.bases(tc.xc)
    classifierbases = TradingStrategy.acceptedbases(rt)
    remove_xc_bases = setdiff(xcbases, classifierbases)
    for rb in remove_xc_bases  # remove coins not accepted by classifier (e.g. insufficient requiredminutes)
        Xch.removebase!(tc.xc, rb)
    end
    remove_classifier_bases = setdiff(classifierbases, xcbases)
    for rb in remove_classifier_bases  # drop stale classifier-only bases that are no longer in the exchange cache
        TradingStrategy.dropbase!(rt, rb)
    end
    xcbases = Xch.bases(tc.xc)
    classifierbases = TradingStrategy.acceptedbases(rt)
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

_isclosetrade(tl) = tl in [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
_isopentrade(tl) = tl in [shortstrongopen, shortopen, longopen, longstrongopen]
_isopenshorttrade(tl) = tl in [shortstrongopen, shortopen]

_traderank(tl) = _isclosetrade(tl) ? 1 : _isopentrade(tl) ? 2 : 3

function _tradetolabeltext(label)
    return String(Symbol(label))
end

function _rowbase(tradesdf::DataFrame, rowix::Integer)::String
    if hasproperty(tradesdf, :coin)
        return uppercase(String(tradesdf[rowix, :coin]))
    end
    if hasproperty(tradesdf, :pair)
        pair = uppercase(String(tradesdf[rowix, :pair]))
        quotecoin = uppercase(String(EnvConfig.pairquote))
        if endswith(pair, quotecoin)
            base = first(pair, max(0, length(pair) - length(quotecoin)))
            isempty(base) || return base
        end
    end
    error("trades row must provide :coin or :pair to derive base")
end

function _labelaction(label::Targets.TradeLabel)::Symbol
    if label in [longopen, longstrongopen]
        return :long_open
    elseif label in [longclose, longstrongclose]
        return :long_close
    elseif label in [shortopen, shortstrongopen]
        return :short_open
    elseif label in [shortclose, shortstrongclose]
        return :short_close
    end
    return :none
end

function _action_columns(action::Symbol)
    if action == :long_open
        return (limitcol=:lo_limit, amountcol=:lo_amount, idcol=:lo_id, statuscol=:lo_status)
    elseif action == :long_close
        return (limitcol=:lc_limit, amountcol=:lc_amount, idcol=:lc_id, statuscol=:lc_status)
    elseif action == :short_open
        return (limitcol=:so_limit, amountcol=:so_amount, idcol=:so_id, statuscol=:so_status)
    elseif action == :short_close
        return (limitcol=:sc_limit, amountcol=:sc_amount, idcol=:sc_id, statuscol=:sc_status)
    end
    error("unsupported action=$(action)")
end

function _row_signal(tradesdf::DataFrame, rowix::Integer, base::AbstractString)
    return (
        tradelabel=tradesdf[rowix, :label],
        base=uppercase(String(base)),
        probability=hasproperty(tradesdf, :score) ? (tradesdf[rowix, :score]) : 0f0,
    )
end

function _current_order_price(orow)
    if hasproperty(orow, :limitprice)
        return getproperty(orow, :limitprice)
    elseif hasproperty(orow, :price)
        return getproperty(orow, :price)
    end
    return missing
end

function _current_order_qty(orow)::Float32
    if hasproperty(orow, :baseqty)
        return (getproperty(orow, :baseqty))
    elseif hasproperty(orow, :qty)
        return (getproperty(orow, :qty))
    end
    return 0f0
end

function _material_order_change(old_price, new_price, old_qty::Real, new_qty::Real; price_reltol::Real=1f-3, qty_reltol::Real=1f-3)::Bool
    oldp = (ismissing(old_price) || isnothing(old_price)) ? nothing : (old_price)
    newp = (ismissing(new_price) || isnothing(new_price)) ? nothing : (new_price)
    if isnothing(oldp) != isnothing(newp)
        return true
    end
    if !isnothing(oldp) && !isnothing(newp)
        denom = max(abs(oldp), 1f-6)
        if abs(newp - oldp) / denom > (price_reltol)
            return true
        end
    end

    oldq = (old_qty)
    newq = (new_qty)
    qdenom = max(abs(oldq), 1f-6)
    return abs(newq - oldq) / qdenom > (qty_reltol)
end

function trade!(cache::TradeCache, tradesdfdict::Dict)
    opencount = 0f0
    closequote = 0f0
    freequote = nothing
    equity = nothing
    for basecfg in eachrow(cache.cfg)
        base = uppercase(String(basecfg.basecoin))
        tradesix = tradesdfdict[base].rowix
        tradesdf = tradesdfdict[base].tradesdf
        tradesrow = tradesdf[tradesix, :]
        if (cache.mc[:trademode] == quickexit) || (base in cache.mc[:blacklistbases])
            tradesrow.label = allclose
            tradesrow.lo_limit = tradesrow.lc_limit = tradesrow.so_limit = tradesrow.sc_limit = 0f0
        elseif cache.mc[:trademode] == notrade
            tradesrow.label = ignore
            tradesrow.lo_limit = tradesrow.lc_limit = tradesrow.so_limit = tradesrow.sc_limit = 0f0
        else
            ts.cfg.algorithm(ts.cfg, tradesdf, tradesix)
            opencount += tradesrow.label in [shortstrongopen, shortopen, longopen, longstrongopen] ? 1 : 0
            if tradesrow.label in [shortstrongclose, shortclose, allclose, longstrongclose, longclose]
                closequote += (tradesrow.lp_amount + tradesrow.sp_amount) * tradesrow.close
            end
        end
        equity = isnothing(equity) ? tradesrow.equity : equity
        freequote = isnothing(freequote) ? tradesrow.freequote : freequote
    end

    tradeamount = min(freequote + closequote, equity, cache.mc[:maxbudgetquote]) 
    tradeamount = tradeamount / opencount
    tradeamount = min(tradeamount, equity * cache.mc[:maxassetfraction])  # limit per base to maxassetfraction of total equity

    for basecfg in eachrow(cache.cfg)
        base = uppercase(String(basecfg.basecoin))
        tradesix = tradesdfdict[base].rowix
        tradesdf = tradesdfdict[base].tradesdf
        tradesrow = tradesdf[tradesix, :]
        if tradesrow.label in [longopen, longstrongopen]
            tradesrow.lp_amount = tradeamount
        elseif tradesrow.label in [shortstrongopen, shortopen]
            tradesrow.sp_amount = tradeamount
        end
        Xch.process_order_request(cache.xc, tradesdf, tradesix) #TODO check implementation

    end

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
    openthreshold = (spec.openthreshold)
    closethreshold = (spec.closethreshold)
    buygain = (spec.buygain)
    sellgain = (spec.sellgain)
    limitreduction = (spec.limitreduction)
    maxwindow = Int(spec.maxwindow)

    @assert 0f0 <= openthreshold <= 1f0 "strategy_openthreshold must be in [0, 1], got $(openthreshold)"
    @assert 0f0 <= closethreshold <= 1f0 "strategy_closethreshold must be in [0, 1], got $(closethreshold)"
    @assert 0f0 <= buygain <= 1f0 "strategy_buygain must be in [0, 1], got $(buygain)"
    @assert 0f0 <= sellgain <= 1f0 "strategy_sellgain must be in [0, 1], got $(sellgain)"
    @assert 0f0 <= limitreduction <= 1f0 "strategy_limitreduction must be in [0, 1], got $(limitreduction)"
    @assert maxwindow > 0 "strategy_maxwindow must be > 0, got $(maxwindow)"
    return spec
end

"Validate strategy runtime parameters stored in `TradeCache.ts`."
function _validatestrategyconfig!(cache::TradeCache)
    _validatestrategyconfig!(cache.ts.cfg)
    return cache
end

function _strategyruntime(cache::TradeCache)::TradingStrategy.TsCache
    return cache.ts
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
    symbol = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote)))
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
    symbol = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote)))
    _managedclosestate(cache)[_managedclosekey(cache, base, tradelabel)] = Dict{Symbol, Any}(
        :base => uppercase(String(base)),
        :symbol => symbol,
        :orderid => String(orderid),
        :label => tradelabel,
        :limitprice => isnothing(limitprice) ? nothing : (limitprice),
        :baseqty => (baseqty),
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
    freebase = (sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :free]))
    borrowedbase = (sum(assets[uppercase.(String.(assets[!, :coin])) .== basekey, :borrowed]))
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
        blacklisted=[false],
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
        freebase = (getproperty(row, :free))
        borrowedbase = (getproperty(row, :borrowed))
        if (freebase > 0f0) || (borrowedbase > 0f0)
            push!(bases, base)
        end
    end
    return unique(bases)
end

function _reconstruct_managed_close_orders!(cache::TradeCache, rowsbybase::AbstractDict{String, Any})
    state = _managedclosestate(cache)
    empty!(state)
    for (base, rowstate) in pairs(rowsbybase)
        tradesdf = rowstate.tradesdf
        rowix = Int(rowstate.rowix)
        for (idcol, stcol, amountcol, limitcol, closelabel) in [
            (:lc_id, :lc_status, :lc_amount, :lc_limit, longclose),
            (:sc_id, :sc_status, :sc_amount, :sc_limit, shortclose),
        ]
            oid = String(tradesdf[rowix, idcol])
            (isempty(strip(oid)) || (lowercase(strip(oid)) == Xch.NO_ORDER_ID)) && continue
            status = String(tradesdf[rowix, stcol])
            Xch.openstatus(status) || continue
            limitprice = (hasproperty(tradesdf, limitcol) && !ismissing(tradesdf[rowix, limitcol])) ? (tradesdf[rowix, limitcol]) : nothing
            baseqty = (tradesdf[rowix, amountcol])
            _managedcloseset!(cache, base, oid, closelabel; limitprice=limitprice, baseqty=baseqty)
        end
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

function _strategyrowlabel(rowstate)::Targets.TradeLabel
    return rowstate.tradesdf[Int(rowstate.rowix), :label]
end

function _strategyrowprice(rowstate)
    tdf = rowstate.tradesdf
    rowix = Int(rowstate.rowix)
    label = _strategyrowlabel(rowstate)
    if label in (longopen, longstrongopen)
        return tdf[rowix, :lo_limit]
    elseif label in (shortopen, shortstrongopen)
        return tdf[rowix, :so_limit]
    elseif label in (longclose, longstrongclose)
        return tdf[rowix, :lc_limit]
    elseif label in (shortclose, shortstrongclose)
        return tdf[rowix, :sc_limit]
    end
    return nothing
end

function _strategyrowprobability(rowstate)::Float32
    tdf = rowstate.tradesdf
    rowix = Int(rowstate.rowix)
    return hasproperty(tdf, :score) ? (tdf[rowix, :score]) : 0f0
end

function _syncroworder(tdf::DataFrame, rowix::Integer, ordertype::Symbol, oo::AbstractDataFrame)
    orderidcol = Symbol(string(ordertyp, "_id"))
    orderstatuscol = Symbol(string(ordertyp, "_status"))
    if hasproperty(tdf, orderidcol) && hasproperty(tdf, orderstatuscol)
        oid = String(tdf[rowix, orderidcol])
        if !isempty(strip(oid)) && (lowercase(strip(oid)) != Xch.NO_ORDER_ID)
            orowix = findfirst(==(oid), String.(oo[!, :orderid]))
            if !isnothing(orowix)
                tdf[rowix, orderstatuscol] = String(oo[orowix, :status])
            end
        end
    end
    return nothing
end

function _collect_strategy_row(cache::TradeCache, base::AbstractString)
    dt = cache.xc.currentdt
    isnothing(dt) && return nothing
    rt = _strategyruntime(cache)
    rowmeta = TradingStrategy.gettradesrow!(rt, cache.xc, uppercase(String(base)), dt)
    isnothing(rowmeta) && return nothing
    return (tradesdf=rowmeta.tradesdf, rowix=Int(rowmeta.rowix))
end

function _ensure_managed_close_orders!(cache::TradeCache, assets::AbstractDataFrame, rowsbybase::AbstractDict{String, Any})
    for base in _close_management_bases(cache, assets)
        cfgrow = _cfgrow_for_base(cache, base)
        sellenabled = isnothing(cfgrow) ? true : _cfgbool(cfgrow, :sellenabled, true)
        basecfg = _basecfg_for_close(cache, base, sellenabled)
        for closelabel in _positioncloselabels(assets, base; sellenabled=sellenabled)
            rowstate = get(rowsbybase, uppercase(String(base)), nothing)
            if isnothing(rowstate)
                dt = isnothing(cache.xc.currentdt) ? Dates.now(Dates.UTC) : cache.xc.currentdt
                rowstate = Xch.ensuretradesrow!(cache.xc, base, EnvConfig.pairquote, dt)
            end
            tradesdf = rowstate.tradesdf
            rowix = Int(rowstate.rowix)
            tradesdf[rowix, :label] = closelabel

            try
                trade!(cache, tradesdf, rowix, assets; basecfg=basecfg)
            catch err
                if _ispermissionrestrictederror(err)
                    _blacklistbase!(cache, base, sprint(showerror, err))
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

"Refreshes the trading strategy selection if the current time matches a configured refresh time and it hasn't been refreshed yet for this minute."
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
    _mark_tradeselection_refreshed!(cache)
    (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
    return true
end

"Return position-side gaps where no matching open close order currently exists."
function _positions_without_close_orders(cache::TradeCache, assets::AbstractDataFrame, rowsbybase::AbstractDict{String, Any})
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
                    msymbol = uppercase(String(Xch.symboltoken(cache.xc, mbase, EnvConfig.pairquote)))
                end
            end
            if isempty(msymbol)
                msymbol = uppercase(String(split(String(mkey), "|"; limit=2)[1]))
            end
            (msymbol == symbolu) || continue
            mlabel = get(managed, :label, ignore)
            if (sideu == "SELL") && (mlabel in [longclose, longstrongclose])
                total += (get(managed, :baseqty, 0f0))
            elseif (sideu == "BUY") && (mlabel in [shortclose, shortstrongclose])
                total += (get(managed, :baseqty, 0f0))
            end
        end
        return total
    end

    function _covered_qty(symbol::AbstractString, side::AbstractString; closekind::Union{Nothing, Symbol}=nothing)::Float32
        symbolu = uppercase(String(symbol))
        sideu = uppercase(String(side))
        total = 0f0
        for (base, rowstate) in pairs(rowsbybase)
            rowsymbol = uppercase(String(Xch.symboltoken(cache.xc, base, EnvConfig.pairquote)))
            rowsymbol == symbolu || continue
            tradesdf = rowstate.tradesdf
            rowix = Int(rowstate.rowix)
            for (stcol, amountcol, filledcol, orderside, orderclosekind) in [
                (:lo_status, :lo_amount, :lo_filled, "BUY", :none),
                (:lc_status, :lc_amount, :lc_filled, "SELL", :long),
                (:so_status, :so_amount, :so_filled, "SELL", :none),
                (:sc_status, :sc_amount, :sc_filled, "BUY", :short),
            ]
                status = String(tradesdf[rowix, stcol])
                Xch.openstatus(status) || continue
                uppercase(orderside) == sideu || continue
                if closekind == :long
                    orderclosekind == :long || continue
                elseif closekind == :short
                    orderclosekind == :short || continue
                end
                baseqty = (tradesdf[rowix, amountcol])
                executed = hasproperty(tradesdf, filledcol) && !ismissing(tradesdf[rowix, filledcol]) ? (tradesdf[rowix, filledcol]) : 0f0
                remaining = max(0f0, baseqty - executed)
                total += max(0f0, remaining)
            end
        end
        return total + _managed_covered_qty(symbolu, sideu)
    end

    function _min_base_qty(base::AbstractString, symbol::AbstractString, row)::Float32
        price = try
            hasproperty(row, :usdtprice) ? (getproperty(row, :usdtprice)) : 0f0
        catch
            0f0
        end
        if !(price > 0f0)
            price = try
                (currentprice(Xch.ohlcv(cache.xc, base)))
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
            return (1.01f0 * max((syminfo.minbaseqty), (syminfo.minquoteqty) / price))
        end
        return (1.01f0 * (syminfo.minbaseqty))
    end

    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == quote_coin && continue
        freebase = (row.free)
        borrowedbase = (row.borrowed)
        if (freebase <= 0f0) && (borrowedbase <= 0f0)
            continue
        end

        symbol = Xch.symboltoken(cache.xc, base, EnvConfig.pairquote)
        minimumbasequantity = _min_base_qty(base, symbol, row)
        if freebase > 0f0
            # Long-close orders are explicit reduce-only sells where supported, with a
            # conservative non-leverage fallback for legacy spot rows that lack the flag.
            sell_covered = _covered_qty(symbol, "Sell"; closekind=:long)
            sell_gap = max(0f0, freebase - sell_covered)
            (sell_gap >= minimumbasequantity) && push!(missing, (base=base, side="Sell", qty=sell_gap, required=freebase, covered=sell_covered, minimum=minimumbasequantity))
        end
        if borrowedbase > 0f0
            # Short-close orders must be explicit reduce-only buys.
            buy_covered = _covered_qty(symbol, "Buy"; closekind=:short)
            buy_gap = max(0f0, borrowedbase - buy_covered)
            (buy_gap >= minimumbasequantity) && push!(missing, (base=base, side="Buy", qty=buy_gap, required=borrowedbase, covered=buy_covered, minimum=minimumbasequantity))
        end
    end
    return missing
end

"""
Execute one trading tick: reconstruct managed close-order state from live open orders,
cancel unmanaged orders, keep one close order active per open position,
execute open/reversal entries, and handle daily trade-selection reload.
Called by the loop runners once per iterate step.
"""
function _tradestep!(cache::TradeCache)
    (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")

    syncpairs = [Xch.tradingpairkey(String(base), EnvConfig.pairquote) for base in cache.cfg[!, :basecoin]]
    rowsbybase = Xch.sync_latest_trades_rows!(cache.xc, syncpairs)
    # rowsbybase is a Dict[base] => (tradesdf, rowix, ohlcv) where rowix is the index of the current trade row.

    trade!(cache, rowsbybase)
    _maybe_refresh_tradeselection!(cache; assets=assets_after)
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = Xch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start calculating trading strategy on the fly")
        tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
end

"Log end-of-loop summary statistics."
function _tradefinish!(cache::TradeCache)
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    oo = Xch.getopenorders(cache.xc)
    (verbosity >= 3) && @info (size(oo, 1) > 0) ? "$(EnvConfig.now()): open orders snapshot $(oo)" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): open orders $(size(oo, 1))"
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

