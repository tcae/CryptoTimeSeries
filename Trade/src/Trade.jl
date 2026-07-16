# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable trade advice and is responsible to allocate assets.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, Xch, Features, Targets, TradingStrategy

function _summarize_symbols(symbols; limit::Int=8)::String
    values = sort!(String.(collect(symbols)))
    total = length(values)
    shown = values[1:min(limit, total)]
    suffix = total > limit ? ", ..." : ""
    return "count=$(total) [$(join(shown, ", "))$(suffix)]"
end

"""
- buysell is the normal trade mode
- closeonly disables opening trades and only closes existing long/short positions
- quickexit sells all assets as soon as possible
- notrade for testing
"""
@enum TradeMode buysell closeonly quickexit notrade

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

# Extra minute buffer for liquidity lookback window to absorb minute-boundary rounding
# and small OHLCV gaps without underfetching the required continuity check horizon.
const LIQUIDITY_LOOKBACK_MARGIN_MINUTES = 5

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : (sum(assets[!, :usdtvalue]))
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
        basekey = String(base)
        (Xch.validbase(tc.xc, base) || (basekey in loaded)) || continue
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
        tc.cfg[:, :pair] = String[]
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
    tc.cfg[:, :pair] = [Xch.tradingpairkey(String(base), quotecoin) for base in tc.cfg[!, :basecoin]]
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
    missing_in_classifier = setdiff(Set(xcbases), classifierbaseset)
    missing_in_xc = setdiff(classifierbaseset, Set(xcbases))
    @assert isempty(missing_in_classifier) && isempty(missing_in_xc) "exchange/classifier base mismatch: xc=$(_summarize_symbols(xcbases)), classifier=$(_summarize_symbols(classifierbases)), xc_only=$(_summarize_symbols(missing_in_classifier)), classifier_only=$(_summarize_symbols(missing_in_xc))"

    tc.cfg[:, :classifieraccepted] = [base in classifierbaseset for base in tc.cfg[!, :basecoin]]
    _sync_tradeflags!(tc; assetonly=assetonly)
    (verbosity >= 4) && println("$(Xch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    # tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    (verbosity >= 2) && println("$(EnvConfig.now()) #tc.cfg=$(size(tc.cfg, 1)) sum(classifieraccepted)=$(sum(tc.cfg[!, :classifieraccepted])) classifierbases=$(_summarize_symbols(classifierbases))")

    if !assetonly
        (verbosity >= 2) && println("\r$(Xch.ttstr(tc.xc)) trained trade config on the fly including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    end
    return tc
end

function trade!(cache::TradeCache, tradesdfdict::Dict; assets::AbstractDataFrame)
    opencount = 0f0
    closequote = 0f0
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
            cache.ts.cfg.algorithm(cache.ts.cfg, tradesdf, tradesix)
            opencount += tradesrow.label in [shortstrongopen, shortopen, longopen, longstrongopen] ? 1 : 0
            if tradesrow.label in [shortstrongclose, shortclose, allclose, longstrongclose, longclose]
                closequote += (tradesrow.lp_amount + tradesrow.sp_amount) * tradesrow.close
            end
        end
    end

    quotefree = _portfolioquotevalue(assets)
    freequote = ismissing(quotefree) ? 0f0 : quotefree
    equity = _portfoliototal(assets)

    maxbudgetquote = get(cache.mc, :maxbudgetquote, nothing)
    availablequote = (freequote + closequote)
    cappedquote = if isnothing(maxbudgetquote)
        min(availablequote, equity)
    else
        cap = (maxbudgetquote)
        if !isfinite(cap) || (cap <= 0.0)
            min(availablequote, equity)
        else
            min(availablequote, equity, cap)
        end
    end

    tradeamount = opencount > 0f0 ? (cappedquote / opencount) : 0f0
    tradeamount = min(tradeamount, equity * cache.mc[:maxassetfraction])  # limit per base to maxassetfraction of total equity
    # (verbosity >= 3) && println("$(tradetime(cache)) trade sizing: opencount=$(opencount), freequote=$(round(freequote, digits=4)), closequote=$(round(closequote, digits=4)), equity=$(round(equity, digits=4)), cappedquote=$(round(cappedquote, digits=4)), tradeamount=$(round(tradeamount, digits=4))")

    for basecfg in eachrow(cache.cfg)
        base = uppercase(String(basecfg.basecoin))
        tradesix = tradesdfdict[base].rowix
        tradesdf = tradesdfdict[base].tradesdf
        tradesrow = tradesdf[tradesix, :]
        if tradesrow.label in [longopen, longstrongopen]
            tradesrow.lo_amount = tradeamount
        elseif tradesrow.label in [shortstrongopen, shortopen]
            tradesrow.so_amount = tradeamount
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

function _strategyruntime(cache::TradeCache)::TradingStrategy.TsCache
    return cache.ts
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

function _summarize_cfg(cfg::AbstractDataFrame)::String
    rows = size(cfg, 1)
    bases = (:basecoin in names(cfg)) ? String.(cfg[!, :basecoin]) : String[]
    buys = (:buyenabled in names(cfg)) ? sum(cfg[!, :buyenabled]) : 0
    sells = (:sellenabled in names(cfg)) ? sum(cfg[!, :sellenabled]) : 0
    return "rows=$(rows), bases=$(_summarize_symbols(bases)), buyenabled=$(buys), sellenabled=$(sells)"
end

function _summarize_openorders(oo::AbstractDataFrame)::String
    rows = size(oo, 1)
    symbols = (:symbol in names(oo)) ? String.(oo[!, :symbol]) : String[]
    sides = (:side in names(oo)) ? String.(oo[!, :side]) : String[]
    return "rows=$(rows), symbols=$(_summarize_symbols(symbols)), sides=$(_summarize_symbols(sides))"
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
    (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(_summarize_cfg(cache.cfg))"
    return true
end


"""
Execute one trading tick: reconstruct managed close-order state from live open orders,
cancel unmanaged orders, keep one close order active per open position,
execute open/reversal entries, and handle daily trade-selection reload.
Called by the loop runners once per iterate step.
"""
function _tradestep!(cache::TradeCache)
    (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")

    acct = Xch.account_status(cache.xc; force_refresh=true, ttl_seconds=0)
        syncpairs = String.(cache.cfg[!, :pair])
    rowsbybase = Xch.sync_latest_trades_rows!(cache.xc, syncpairs; acct=acct)
    # rowsbybase is a Dict[base] => (tradesdf, rowix, ohlcv) where rowix is the index of the current trade row.

    trade!(cache, rowsbybase; assets=acct.assets)
    _maybe_refresh_tradeselection!(cache; assets=acct.assets)
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = Xch.portfolio!(cache.xc)
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
    (verbosity >= 3) && @info (size(oo, 1) > 0) ? "$(EnvConfig.now()): open orders summary $(_summarize_openorders(oo))" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): open orders $(size(oo, 1))"
    assets = Xch.portfolio!(cache.xc)
    if verbosity >= 3
        assetcoins = (:coin in names(assets)) ? String.(assets[!, :coin]) : String[]
        @info "assets summary: rows=$(size(assets, 1)), coins=$(_summarize_symbols(assetcoins))"
    end
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

