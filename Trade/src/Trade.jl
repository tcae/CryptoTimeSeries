# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, longbuy when an uptrend starts and longclose when it ends.
It generates the OHLCV data, executes the trades in a loop and selects the basecoins to trade.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, Statistics
using EnvConfig, Ohlcv, CryptoXch, Classify, Features, Targets, TradeAudit, TradingStrategy

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

"""
- buysell is the normal trade mode
- sellonly disables buying but sells according to normal longclose behavior
- quickexit sells all assets as soon as possible
- notrade for testing
"""
@enum TradeMode buysell sellonly quickexit notrade

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

function _portfoliototal(assets::AbstractDataFrame)::Float64
    return size(assets, 1) == 0 ? 0.0 : Float64(sum(assets[!, :usdtvalue]))
end

function _portfolioquotevalue(assets::AbstractDataFrame)::Union{Missing, Float64}
    if size(assets, 1) == 0 || !any(name -> name == "coin", names(assets))
        return missing
    end
    quoteix = findfirst(==(EnvConfig.cryptoquote), assets[!, :coin])
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
    exchange_name = CryptoXch._routeexchange(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.exchange(cache.xc))
    account_alias = something(CryptoXch._routeauthname(cache.xc.routing, CryptoXch.trade_exchange_spot, CryptoXch.authname(cache.xc)), "")
    try
        if rowcount == 0
            event = TradeAudit.AuditEventRow(
                event_type=TradeAudit.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=CryptoXch.auditrunmode(cache.xc),
                run_id=CryptoXch.auditrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeAudit.routing_trade_exchange_spot,
                market_type=TradeAudit.market_unknown,
                asset_class=TradeAudit.crypto,
                instrument_type=TradeAudit.instrument_unknown,
                symbol="PORTFOLIO",
                cash_after=cash_after,
                portfolio_value_after=portfolio_total,
                notes="rows=0; simmode=$(simmode)"
            )
            TradeAudit.writeeventwithhash(event)
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
            event = TradeAudit.AuditEventRow(
                event_type=TradeAudit.PORTFOLIO_SNAPSHOT,
                event_time_utc=event_time,
                created_at_utc=event_time,
                source_module=String(source_module),
                environment=string(Symbol(EnvConfig.configmode)),
                run_mode=CryptoXch.auditrunmode(cache.xc),
                run_id=CryptoXch.auditrunid(cache.xc),
                exchange=exchange_name,
                account_alias=account_alias,
                routing_role=TradeAudit.routing_trade_exchange_spot,
                market_type=TradeAudit.market_unknown,
                asset_class=TradeAudit.crypto,
                instrument_type=TradeAudit.spot_pair,
                symbol=coin,
                baseasset=coin,
                quoteasset=EnvConfig.cryptoquote,
                settlement_asset=EnvConfig.cryptoquote,
                position_qty_after=positionqty,
                cash_after=(coin == EnvConfig.cryptoquote ? positionqty : cash_after),
                portfolio_value_after=portfolio_total,
                fill_notional=positionvalue,
                notes="asset=$(coin); rows=$(rowcount); simmode=$(simmode)"
            )
            TradeAudit.writeeventwithhash(event)
        end
    catch audit_error
        (verbosity >= 1) && @warn "failed to persist portfolio snapshot" exception=(audit_error, catch_backtrace())
    end
    return nothing
end

"""
*TradeCache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *TradeCache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct TradeCache
    xc::CryptoXch.XchCache  # required to connect to exchange
    cfg::AbstractDataFrame    # maintains the bases to trade and their classifiers
    cl::Classify.AbstractClassifier
    mc::Dict # MC = module constants
    dbgdf
    looplock::ReentrantLock
    loopcond::Threads.Condition
    function TradeCache(; xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=notrade)
        looplock = ReentrantLock()
        cache = new(xc, DataFrame(), cl, Dict(), DataFrame(), looplock, Threads.Condition(looplock))
        cache.mc[:exitcoins] = [] # exit specific coins
        cache.mc[:longopencoins] = []  # force open long
        cache.mc[:shortopencoins] = [] # force open short
        cache.mc[:whitelistcoins] = ["ADA", "AI16Z", "APEX", "AAVE", "BNB", "BTC", "CAKE", "DOGE", "ELX", "ENA", "ETH", "HBAR", "HFT", "JUP", "LINK", "LTC", "MNT", "ONDO", "PEPE", "POPCAT", "S", "SOL", "SUI", "TON", "TRX", "VIRTUAL", "W", "WAL", "WIF", "WLD", "X", "XLM", "XRP"] 
        # not whitelisted: "ANIME", "B3", "BERA", "CMETH", "LDO", "PLUME", "SOSO", "TRUMP"
        cache.mc[:hourlygainlimit] = 0.1f0 # limit hourly gain to a realistic 10% max
        cache.mc[:maxassetfraction] = 0.1f0 # defines the maximum ratio of (a specific asset) / ( total assets) - only close trades, if this is exceeded
        cache.mc[:reloadtimes] = [Time("04:00:00")]
        cache.mc[:trademode] = trademode  # see TradeMode definition above
        cache.mc[:usenewtrade] = false # implementation switch between old and new trade! method
        cache.mc[:strategy_engine] = :classifier  # :classifier (legacy) or :getgainsalgo
        cache.mc[:strategy_algorithm] = TradingStrategy.gain_reversal!  # configured gain algorithm
        cache.mc[:strategy_state] = Dict{String, Any}()  # per-base TradingStrategy.GainSegment
        cache.mc[:strategy_history] = Dict{String, Any}()  # per-base rolling price+signal history
        cache.mc[:strategy_openthreshold] = 0.6f0
        cache.mc[:strategy_closethreshold] = 0.5f0
        cache.mc[:strategy_buygain] = 0.001f0
        cache.mc[:strategy_sellgain] = 0.01f0
        cache.mc[:strategy_limitreduction] = 0f0
        cache.mc[:strategy_maxwindow] = 4 * 60
        cache.mc[:strategy_source] = "default"
        cache.mc[:loop_state] = loop_idle
        (verbosity >= 2) && println("TradeCache trademode = $(cache.mc[:trademode]), maxassetfraction = $(cache.mc[:maxassetfraction]), reloadtimes = $(cache.mc[:reloadtimes]), exitcoins = $(cache.mc[:exitcoins]), whitelistcoins = $(cache.mc[:whitelistcoins]), longopencoins = $(cache.mc[:longopencoins]), shortopencoins = $(cache.mc[:shortopencoins])")
        return cache
    end
end

function Base.show(io::IO, cache::TradeCache)
    print(io::IO, "TradeCache: startdt=$(cache.xc.startdt) currentdt=$(cache.xc.currentdt) enddt=$(cache.xc.enddt)")
end

ohlcvdf(cache, base) = Ohlcv.dataframe(cache.bd[base].ohlcv)
ohlcv(cache, base) = cache.bd[base].ohlcv
classifier(cache, base) = cache.bd[base].classifier
backtest(cache) = cache.backtestperiod >= Dates.Minute(1)
dummytime() = DateTime("2000-01-01T00:00:00")

function _tradeselection_history_minutes(tc::TradeCache)::Int
    classifier_minutes = try
        Int(Classify.requiredminutes(tc.cl))
    catch
        0
    end
    liquidity_minutes = max(
        Int(Ohlcv.ld.startdistance + Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate),
        Int(Ohlcv.ld.minliquidminutes + Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate),
    )
    return max(classifier_minutes + 1, liquidity_minutes, 24 * 60)
end

"""
Build a minimal 24h USDT market snapshot for requested bases by downloading their
OHLCV slices directly. This is a fallback path used when `getUSDTmarket` returns
no rows (for example when canned market snapshots are unavailable).
"""
function _fallback_usdtmarket(tc::TradeCache, datetime::DateTime, bases::AbstractVector)::DataFrame
    usdtdf = DataFrame(
        basecoin=String[],
        quotevolume24h=Float32[],
        pricechangepercent=Float32[],
        lastprice=Float32[],
        askprice=Float32[],
        bidprice=Float32[],
    )
    startdt = datetime - Day(1) + Minute(1)
    for rawbase in unique(bases)
        base = rawbase === nothing ? "" : String(rawbase)
        isempty(base) && continue
        base == EnvConfig.cryptoquote && continue
        ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", startdt, datetime)
        odf = Ohlcv.dataframe(ohlcv)
        if size(odf, 1) == 0
            (verbosity >= 2) && @warn "fallback usdtmarket: missing OHLCV" base datetime startdt
            continue
        end
        firstopen = Float32(odf[begin, :open])
        lastclose = Float32(odf[end, :close])
        lastpivot = hasproperty(odf, :pivot) ? Float32(odf[end, :pivot]) : lastclose
        vol24h = Float32(sum(odf[!, :basevolume] .* odf[!, :pivot]))
        change = firstopen == 0f0 ? 0f0 : Float32((lastclose - firstopen) / firstopen)
        push!(usdtdf, (
            basecoin=base,
            quotevolume24h=vol24h,
            pricechangepercent=change,
            lastprice=lastclose,
            askprice=lastpivot * 1.00001f0,
            bidprice=lastpivot * 0.99999f0,
        ))
    end
    return usdtdf
end


TRADECONFIGFILE = "TradeConfig"

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
Loads all USDT coins, checks continuous minimum volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to enable backtesting.
"""
function tradeselection!(tc::TradeCache, assetbases::Vector; datetime=tc.xc.startdt, assetonly=false, updatecache=false)
    datetime = floor(datetime, Minute(1))
    quotecoin = uppercase(EnvConfig.cryptoquote)
    assetbase_tokens = [_normalize_basecoin_token(x, quotecoin) for x in assetbases]
    whitelist_tokens = [_normalize_basecoin_token(x, quotecoin) for x in tc.mc[:whitelistcoins]]
    assetbaseset = Set(filter(!isnothing, assetbase_tokens))
    whitelistset = Set(filter(!isnothing, whitelist_tokens))
    history_minutes = _tradeselection_history_minutes(tc)
    history_startdt = datetime - Minute(history_minutes)

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # CryptoXch.removeallbases(tc.xc)  #* reuse what is in cache
    # Classify.removebase!(tc.cl, nothing)  #* reuse what is in cache

    usdtdf = CryptoXch.getUSDTmarket(tc.xc; dt=datetime)  # superset of coins with 24h volume price change and last price
    if size(usdtdf, 1) == 0
        fallbackbases = filter(!=(quotecoin), collect(union(assetbaseset, whitelistset)))
        (verbosity >= 1) && @warn "empty USDT market snapshot; using OHLCV fallback reconstruction" datetime fallbackbases
        usdtdf = _fallback_usdtmarket(tc, datetime, fallbackbases)
    end
    if assetonly
        usdtdf = filter(row -> row.basecoin in assetbaseset, usdtdf)
    end
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent, :lastprice)
    if size(tc.cfg, 1) == 0
        tc.cfg[:, :datetime] = DateTime[]
        tc.cfg[:, :minquotevol] = Bool[]
        tc.cfg[:, :continuousminvol] = Bool[]
        tc.cfg[:, :inportfolio] = Bool[]
        tc.cfg[:, :classifieraccepted] = Bool[]
        tc.cfg[:, :noinvest] = Bool[]
        tc.cfg[:, :buyenabled] = Bool[]
        tc.cfg[:, :sellenabled] = Bool[]
        tc.cfg[:, :whitelisted] = Bool[]
        (verbosity >= 1) && @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :datetime] .= datetime
    # tc.cfg[:, :validbase] = [CryptoXch.validbase(tc.xc, base) for base in tc.cfg[!, :basecoin]] # is already filtered by getUSDTmarket
    minimumdayquotevolumemillion = round(Ohlcv.liquiddailyminimumquotevolume() / 1000000, digits=0) # ignore allcoins with less than liquiddailyminimumquotevolume
    tc.cfg[:, :minquotevol] = tc.cfg[:, :quotevolume24h_M] .>= minimumdayquotevolumemillion
    tc.cfg[:, :continuousminvol] .= false
    tc.cfg[:, :inportfolio] = [base in assetbaseset for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :classifieraccepted] .= false
    tc.cfg[:, :noinvest] .= false
    tc.cfg[:, :buyenabled] .= false
    tc.cfg[:, :sellenabled] .= false
    tc.cfg[:, :whitelisted] = [base in whitelistset for base in tc.cfg[!, :basecoin]]

    # download latest OHLCV and classifier features
    tc.cfg = tc.cfg[tc.cfg[:, :minquotevol] .|| tc.cfg[:, :inportfolio], :]
    (verbosity >= 3) && println("#minquotevol=$(sum(tc.cfg[:, :minquotevol])) #inportfolio=$(sum(tc.cfg[:, :inportfolio]))")
    count = size(tc.cfg, 1)
    xcbases = CryptoXch.bases(tc.xc)
    removebases = setdiff(xcbases, tc.cfg[!, :basecoin])
    for rb in removebases  # remove coins that were loaded but are no longer part of the new configuration
        CryptoXch.removebase!(tc.xc, rb)
        Classify.removebase!(tc.cl, rb)
    end
    xcbaseset = Set(CryptoXch.bases(tc.xc))
    (verbosity >= 3) && println("trade selection history window=$(history_minutes) minutes from $(history_startdt) to $(datetime)")
    for (ix, row) in enumerate(eachrow(tc.cfg))
        (verbosity >= 2) && updatecache &&  print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) including cache update                           ")
        (verbosity >= 2) && !updatecache && print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) without cache update                             ")
        if row.basecoin in xcbaseset
            ohlcv = CryptoXch.ohlcv(tc.xc, row.basecoin)
            CryptoXch.cryptoupdate!(tc.xc, ohlcv, history_startdt, datetime)
        else
            ohlcv = CryptoXch.cryptodownload(tc.xc, row.basecoin, "1m", history_startdt, datetime)
            Classify.addbase!(tc.cl, ohlcv)
        end
        if updatecache
            Ohlcv.write(ohlcv) # write ohlcv even if data length is too short to calculate features
        end
        rv = Ohlcv.liquiditycheck(ohlcv)
        row.continuousminvol = (length(rv) > 0) && (rv[end][end] == lastindex(Ohlcv.dataframe(ohlcv), 1))
    end
    Classify.supplement!(tc.cl)
    if updatecache
        Classify.writetargetsfeatures(tc.cl)
    end
    xcbases = CryptoXch.bases(tc.xc)
    classifierbases = Classify.bases(tc.cl)
    classifierbaseset = Set(classifierbases)
    removebases = setdiff(xcbases, classifierbases)
    for rb in removebases  # remove coins that were not accepted by the classifier, e.g. if requiredminutes is insufficient
        CryptoXch.removebase!(tc.xc, rb)
    end
    xcbases = CryptoXch.bases(tc.xc)
    @assert Set(xcbases) == classifierbaseset "Set(xcbases)=$(xcbases) != Set(classifierbases)=$(classifierbases)"

    tc.cfg[:, :classifieraccepted] = [base in classifierbaseset for base in tc.cfg[!, :basecoin]]

    if assetonly
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
    else
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& (tc.cfg[:, :minquotevol] .&& tc.cfg[!, :continuousminvol]) .&& tc.cfg[!, :whitelisted]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :buyenabled] .|| (tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted])
    end
    (verbosity >= 3) && println("$(CryptoXch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    # tc.cfg = tc.cfg[(tc.cfg[!, :buyenabled] .|| tc.cfg[:, :sellenabled]), :]
    (verbosity >= 3) && println("$(EnvConfig.now()) #tc.cfg=$(size(tc.cfg, 1)) sum(classifieraccepted)=$(sum(tc.cfg[!, :classifieraccepted])) classifierbases($(length(classifierbases)))=$(classifierbases) ")

    if !assetonly
        write(tc, datetime)
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) trained and saved trade config data including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    end
    return tc
end

_cfgstem(timestamp::Union{Nothing, DateTime}) = isnothing(timestamp) ? TRADECONFIGFILE : join([TRADECONFIGFILE, Dates.format(timestamp, "yy-mm-dd")], "_")
_cfgfolder(folderpath=nothing) = isnothing(folderpath) ? EnvConfig.datafolderpath(TRADECONFIGFILE) : normpath(folderpath)

function _cfgfilename(timestamp::Union{Nothing, DateTime}; folderpath=nothing, format::Symbol=:auto)
    return EnvConfig.tablepath(_cfgstem(timestamp); folderpath=_cfgfolder(folderpath), format=format)
end

"Saves the trade configuration. If timestamp!=nothing then save 2x with and without timestamp in filename otherwise only without timestamp"
function write(tc::TradeCache, timestamp::Union{Nothing, DateTime}=nothing; folderpath=nothing, format::Symbol=EnvConfig.dfformat())
    if (size(tc.cfg, 1) == 0)
        @warn "trade config is empty - not stored"
        return
    end
    cfgfolder = _cfgfolder(folderpath)
    cfgstem = _cfgstem(nothing)
    EnvConfig.deletefolder(cfgstem; folderpath=cfgfolder)
    cfgfilename = EnvConfig.savedf(tc.cfg, cfgstem; folderpath=cfgfolder, format=format)
    (verbosity >= 3) && println("saving trade config in cfgfilename=$cfgfilename")
    if !isnothing(timestamp)
        cfgstem = _cfgstem(timestamp)
        EnvConfig.deletefolder(cfgstem; folderpath=cfgfolder)
        cfgfilename = EnvConfig.savedf(tc.cfg, cfgstem; folderpath=cfgfolder, format=format)
        (verbosity >= 3) && println("saving trade config in cfgfilename=$cfgfilename")
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists but does not load the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function readconfig!(tc::TradeCache, datetime; folderpath=nothing, format::Symbol=EnvConfig.dfformat())
    tc.cfg = df = DataFrame() # old cfg is in either case discarded
    cfgfolder = _cfgfolder(folderpath)
    cfgstem = _cfgstem(datetime)
    cfgfilename = _cfgfilename(datetime; folderpath=cfgfolder, format=:auto)
    df = something(EnvConfig.readdf(cfgstem; folderpath=cfgfolder, format=format, copycols=true), DataFrame())
    if size(df, 1) > 0
        (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config from $cfgfilename")
        tc.cfg = df
        tc.cfg[:, :whitelisted] = [coin in tc.mc[:whitelistcoins] for coin in tc.cfg[!, :basecoin]]
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& (tc.cfg[:, :minquotevol] .&& tc.cfg[!, :continuousminvol]) .&& tc.cfg[!, :whitelisted]
        return tc
    else
        (verbosity >= 2) && println("Loading $cfgfilename failed")
        return nothing
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists. Also loads the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function read!(tc::TradeCache, datetime=nothing; folderpath=nothing, format::Symbol=EnvConfig.dfformat())
    datetime = isnothing(datetime) ? nothing : floor(datetime, Minute(1))
    tc = readconfig!(tc, datetime; folderpath=folderpath, format=format)
    df = nothing
    checkinclude(basecoin) = !(basecoin in CryptoXch.baseignore)
    if !isnothing(tc) && !isnothing(tc.cfg) && (size(tc.cfg, 1) > 0)
        datetime = tc.cfg[begin, :datetime]
        clvec = []
        tc.cfg = tc.cfg[checkinclude.(tc.cfg[!, :basecoin]), :] 
        df = tc.cfg
        rows = size(df, 1)
        for ix in eachindex(df[!, :basecoin])
            (verbosity >= 2) && print("\r$(EnvConfig.now()) loading $(df[ix, :basecoin]) from trade config ($ix of $rows)                                                  ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", datetime-Minute(Classify.requiredminutes(tc.cl)-1), datetime)
            Classify.addbase!(tc.cl, ohlcv)
        end
        classifierbases = Classify.bases(tc.cl)
        if (length(classifierbases) != count(tc.cfg[:, :classifieraccepted]))
            #TODO that can happen, e.g. loading production coins in cockpit and then switching to test - requires xc and classifier cleanup with loading of config
            @error "length(classifierbases)=$(length(classifierbases)) != count(tc.cfg[:, :classifieraccepted])=$(count(tc.cfg[:, :classifieraccepted]))"
        end
        tc.cfg[:, :classifieraccepted] = [base in classifierbases for base in tc.cfg[!, :basecoin]] #! redundant with previous if check?
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) loaded trade config data including $rows base classifier (ohlcv, features) data      ")
    end
    return !isnothing(df) && (size(df, 1) > 0) ? tc : nothing
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
function addassetsconfig!(tc::TradeCache, assets=CryptoXch.portfolio!(tc.xc))
    sort!(assets, [:coin])  # for readability only

    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:borrowed, :accruedinterest, :locked, :free])]
    tc.cfg[:, :inportfolio] = .!ismissing.(tc.cfg[:, :usdtvalue])
    tc.cfg[:, :sellenabled] .= tc.cfg[!, :buyenabled] .|| (tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted])
    sort!(tc.cfg, [:basecoin])  # for readability only
    sort!(tc.cfg, rev=true, [:buyenabled])  # for readability only
    return tc.cfg, assets
end

"Returns the current TradeConfig dataframe with usdtprice and usdtvalue added as well as the portfolio dataframe as a tuple"
function assetsconfig!(tc::TradeCache, datetime=nothing; folderpath=nothing, format::Symbol=EnvConfig.dfformat())
    tc = readconfig!(tc, datetime; folderpath=folderpath, format=format)
    return addassetsconfig!(tc)
end

significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2


currenttime(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :opentime]
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv)[ohlcv.ix, :close]
closelongset = [shortstrongbuy, shortbuy, shorthold, allclose, longstrongclose, longclose]
closeshortset = [shortclose, shortstrongclose, allclose, longhold, longbuy, longstrongbuy]

mutable struct Investment  #TODO bookkeeping for consistency checks
    investmentid
    tradeadvice::Vector{Classify.TradeAdvice} # vector of all used trade advices
    orderid::Vector # vector of all used orders
    classifiername
    configid
end

function dbgrow(cache::TradeCache, ta)
    return (
        taconfigid = ta.configid,
        tatradelabel = ta.tradelabel,
        tabase = ta.base,
        tahourlygain = ta.hourlygain,
        baseqty = missing,
        minimumbasequantity = missing,
        freebase = missing,
        totalborrowedusdt = missing,
        freeusdt = missing,
        quoteqty = missing,
    )
end

"Adds a dataframe trade advice row "
function _traderow!(df, cache; basecoin="XXX", tradelabel=allclose, hourlygain=0f0, probability=1f0, relativeamount=1f0, investmentid="XXX", price=0f0, datetime=cache.xc.currentdt, classifier=cache.cl, configid=0, oid="", enforced="n/a")
    hourlygain = min(hourlygain, cache.mc[:hourlygainlimit])
    push!(df, (basecoin=basecoin, tradelabel=tradelabel, hourlygain=hourlygain, probability=probability, relativeamount=relativeamount, investmentid=investmentid, price=price, datetime=datetime, classifier=classifier, configid=configid, oid=oid, enforced=enforced))
    return last(df)
end

"Adds a dataframe trade advice row based ona trade advice input"
_tradeadvice2df!(df, cache::TradeCache, ta) = _traderow!(df, cache, basecoin=ta.base, tradelabel=ta.tradelabel, hourlygain=ta.hourlygain, probability=ta.probability, relativeamount=ta.relativeamount, investmentid=ta.investmentid, price=ta.price, datetime=ta.datetime, classifier=ta.classifier, configid=ta.configid)

"Adds enforced trade advics according to black lists and enforced trades constraints"
function _forcetradelabel!(df::DataFrame, cache::TradeCache, coins, tradelabel, hourlygain, enforced)
    for base in coins
        rowix = findfirst(x -> x == base, df[!, :basecoin])
        if isnothing(rowix)
            if base in CryptoXch.bases(cache.xc)
                _traderow!(df, cache, basecoin=base, tradelabel=tradelabel, probability=1f0, relativeamount=1f0, hourlygain=hourlygain, enforced=enforced)
            end
        else
            df[rowix, :tradelabel] = tradelabel
            df[rowix, :investmentid] = trade_enforced
        end
    end
end

_isclosetrade(tl) = tl in [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
_isopentrade(tl) = tl in [shortstrongbuy, shortbuy, longbuy, longstrongbuye]
_isopenshorttrade(tl) = tl in [shortstrongbuy, shortbuy]

"""
Creates dataframe from trade advice vector plus corresponding asset info and adds/changes rows to enforce trades,  
i.e. add trades for enforced long open and short open and long/short exits, removes black listed coins
"""
function policyenforcement(cache::TradeCache, tavec::Vector{Classify.TradeAdvice}, assets::AbstractDataFrame)
    df = DataFrame()
    _traderow!(df, cache)  # create columns
    pop!(df)  # remove dummy row
    if cache.mc[:trademode] == quickexit
        for base in assets[!, :coin]
            uppercase(String(base)) == uppercase(EnvConfig.cryptoquote) && continue
            _traderow!(df, cache, basecoin=base, tradelabel=allclose, enforced="quickexit")
        end
    else # no quick exit
        # don't check against other trade modes to enable debugging of tradeamount()
        for ta in tavec
            if !(ta.base in CryptoXch.baseignore)
                #TODO baseignore -> invalid symbol
                #TODO noinvest => buyenabled = false
                rowix = findfirst(row -> row.basecoin == ta.base, cache.cfg)
                if !isnothing(rowix) 
                    if cache.cfg[rowix, :sellenabled] && _isclosetrade(ta.tradelabel)
                        _tradeadvice2df!(df, cache, ta)
                    elseif cache.cfg[rowix, :buyenabled] && _isopentrade(ta.tradelabel)
                        _tradeadvice2df!(df, cache, ta)
                    # else allhold tradeadvices are skipped
                    end
                end
            end
        end
        _forcetradelabel!(df, cache, cache.mc[:longopencoins], longstrongbuy, 1f0, "longopen")
        _forcetradelabel!(df, cache, cache.mc[:shortopencoins], shortstrongbuy, -1x0, "shortopen")
        _forcetradelabel!(df, cache, cache.mc[:exitcoins], allclose, 0f0, "exit")
    end
    if size(df, 1) > 0
        leftjoin!(df, assets, on = :basecoin => :coin)
    end
    return df
end

"Distributes the available quote assets across all open trades and returns result in quoteamount"
function _tradeamounts!(tadf)
    tadf.quoteamount[_isclosetrade(tadf.tradelabel)] .= abs.(tadf.free .* tadf.usdtprice) # add close amounts (which are not constrained by free quote)

    freequote = sum(tadf[tadf[!, :basecoin] .== EnvConfig.cryptoquote, :free])
    maxassetquote = sum(tadf[!, :usdtvalue]) * cache.mc[:maxassetfraction]
    opentradeweights = abs(tadf[!, :hourlygain] .* tadf[!, :probability] .* tadf[!, :relativeamount])
    opentradeweights = opentradeweights ./ sum(opentradeweights)  # normalize weights that they add up to 1
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= opentradeweights .* freequote   # open trades have positive (long) or negative (short) amount, close and hold trades have zero amount
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= min.(tadf[!, :quoteamount], maxassetquote)  # limit max amount of trade
    tadf.quoteamount[_isopentrade(tadf.tradelabel)] .= tadf[!, :quoteamount] .- abs.(tadf[!, :free]) .* tadf[!, :usdtprice]  # deduct already opened amount
    reduceamount = _isopentrade(tadf.tradelabel) .&& (tadf[!, :quoteamount] .< (-0.1 .* abs.(tadf[!, :free]) .* tadf[!, :usdtprice]))  # 0f0) # reduce if current open position is >10% larger than target
    if any(reduceamount)  # instead of open -> change to close with the amount that is above target volume
        tadf.tradelabel[reduceamount] .= allclose
        tadf.quoteamount[reduceamount] .= abs.(trade.quoteamount)
    end
    tadf.quoteamount[_isopentrade(tadf.tradelabel) .&& (tadf[!, :quoteamount] .< 0f0)] .== 0f0  # those who have a reduce amount less than 10% will be ignored

    subset!(tadf, [:quoteamount, :minquoteqty] => (amt, minq) -> abs.(amt) .> minq) # remove alltrades below level threshold
end

_traderank(tl) = _isclosetrade(tl) ? 1 : _isopentrade(tl) ? 2 : 3

function _tradetolabeltext(label)
    return String(Symbol(label))
end

function _withtradeauditcontext(f::Function, cache::TradeCache, ta::Classify.TradeAdvice)
    signal_score = try
        Float64(ta.probability)
    catch
        missing
    end
    strategy_engine = String(Symbol(_strategyengine(cache)))
    strategy_ref = string(get(cache.mc, :strategy_source, "default"))
    CryptoXch.setauditcontext!(
        cache.xc;
        strategy_engine=strategy_engine,
        strategy_config_ref=strategy_ref,
        signal_label=_tradetolabeltext(ta.tradelabel),
        signal_score=signal_score,
    )
    try
        return f()
    finally
        CryptoXch.clearauditcontext!(cache.xc)
    end
end

"""
Provides the amount that should be used for the tradeadvice including all considerations in a dataframe.

  - take out all not tradable coins from the trade advice set
    - quotecoins
    - black listed coins
  - calculate the overall amount that can be spend now based on free available portfolio budget
  - sort remaining trade advices according to expected hourly gain
  - determine overall investment for that basecoin 
    - it should not dominate due to risk and hold back a head room part (e.g. 5%) of free usdt
    - consider the hourly gain when calculating the basecoin specific amount
    - reduce investments according to overall investment amount if delta is more than 10% (hysteresis) if advice is buy or hold
  - close investments if 
    - hourly gain of current investments are x% (e.g. 50%) less gain than best investments and those invested coins above that limit are not at dominating limit
    - coin is above dominating investment limit
    - quickexit for specific or all coins
  - split investment in chunks of reasoable size
  - if one chunk is smaller that minimum limits of exchange then merge them if possible
  - too small amounts cannot be traded if they are below exchange limits
  - insuffient free coins will also prevent trading
  - margin trades should only be done if borrowed amount is covered by free quotecoin
  - same amounts are applicable for margin and spot trading, i.e. buy amount also applies to a margin sell without basecoin assets (short buy)

  Returned dataframe should include (all amounts in usdt to better compare magnitudes)

  - basecoin, currentcloseprice, totalwallet, totalbasecoin, minquoteqty, minbaseqty, maxtotalbasecoin, maxbuyamount, maxsellamount, targetchunksize, buyamount, sellamount

  ## Short trades:

  - marginfree and marginlocked can be negative
  - the total absolute value is sum(abs.(marginfree, marginlocked), free, locked, - borrowed per coin)
  - there should be no case of free / locked positive and marginfree / marginlocked negative because there are balanced with each other -> save to add all absolute amounts by abs.(df)
"""
function tradeamount(cache::TradeCache, tavec::Vector{Classify.TradeAdvice}, assets::AbstractDataFrame) #TODO consider negative short amounts
    tadf = policyenforcement(cache, tavec, assets) # returns a dataframe with tradeadvice per line plus corresponding asset info
    if sze(tadf, 1) > 0
        tadf.minquoteqty = [minimumquotequantity(xc, base) for base in tadf[!, :basecoin]]
        _tradeamounts!(tadf)
        sort!(tadf, [order(:tradelabel, by=_traderank), order(:hourlygain, rev=true)])  # order such that close before open and high before low hourlygain
        ta.basetradeqty .=0f0
        ta.vol1hmedian .= 0f0
        ta.baseamount .= 0f0
        for tarow in tadf
            ohlcv = CryptoXch.ohlcv(cache.xc, base)
            price = currentprice(ohlcv)
            tarow.baseamount = tarow.quoteamount / price
        
            tarow.basevol1hmedian = Ohlcv.dataframe(ohlcv)[Ohlcv.rowix(ohlcv, cache.xc.currentdt-Hour(1)):Ohlcv.rowix(ohlcv, cache.xc.currentdt), :basevolume]
            tarow.basetradeqty = max(tarow.baseamount, median(tarow.basevol1hmedian)/2) # don't trade more than 50% of the houry minute median
        end
    end
    return tadf
end


"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, tadf::DataFrameRow)
    for ta in eachrow(tadf)
        if (ta.tradelabel in [longbuy, longstrongbuy]) && (cache.mc[:trademode] == buysell)
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : CryptoXch.createbuyorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=0)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [longstrongclose, longclose]) && (cache.mc[:trademode] in [buysell, sellonly, quickexit])
            oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : CryptoXch.createsellorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=basequta.basetradeqtyantity, maker=true, marginleverage=0)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, sellonly, quickexit]) && basecfg.sellenabled
            oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : CryptoXch.createbuyorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=2)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        elseif (ta.tradelabel in [shortstrongbuy, shortbuy]) && (cache.mc[:trademode] == buysell)
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : CryptoXch.createsellorder(cache.xc, ta.basecoin; limitprice=nothing, basequantity=ta.basetradeqty, maker=true, marginleverage=2)
            if !isnothing(oid)
                ta.oid = oid
            else
                ta.oid = "failed"
            end
        end
        push!(cache.dbgdf, ta, promote=true)
    end
end

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, basecfg::DataFrameRow, ta::Classify.TradeAdvice, assets::AbstractDataFrame)
    sellbuyqtyratio = 2 # longclose qty / longbuy qty per order, if > 1 longclose quicker than buying it
    qtyacceleration = 4 # if > 1 then increase longbuy and longclose order qty by this factor
    result = nothing
    base = ta.base
    totalusdt = sum(assets.usdtvalue)
    if totalusdt <= 0
        @warn "totalusdt=$totalusdt is insufficient, assets=$assets"
        return nothing
    end
    basequantity = missing
    freeusdtfractionmargin = 0.05
    totalborrowedusdt = sum(assets[!, :borrowed] .* assets[!, :usdtprice])
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free]) - totalborrowedusdt
    freebase = sum(assets[assets[!, :coin] .== base, :free]) *(1-eps(Float32))
    borrowedbase = sum(assets[assets[!, :coin] .== base, :borrowed])
    quotequantity = cache.mc[:maxassetfraction] * totalusdt / 10  # distribute over 10 trades
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    price = currentprice(ohlcv)
    @assert base == ohlcv.base == ta.base
    minimumbasequantity = CryptoXch.minimumbasequantity(cache.xc, base, price)
    if isnothing(minimumbasequantity)
        (verbosity > 2) && println("$(tradetime(cache)) skip $base due to missing minimum base quantity at price=$price")
        return nothing
    end
    # (verbosity > 2) && println("$(tradetime(cache)) entry $base , $(ta.tradelabel)")
    # CryptoXch.portfolio subtracts the borrowed amount from usdtvalue of each base
    if (cache.mc[:trademode] == quickexit) || (base in cache.mc[:exitcoins])
        ta.tradelabel = allclose
    end
    if (ta.tradelabel in [allclose, longhold]) && (borrowedbase > 0)
        ta.tradelabel = shortclose
    end
    if (ta.tradelabel in [allclose, shorthold]) && (freebase > 0)
        ta.tradelabel = longclose
    end
    if (ta.tradelabel in [allclose, shorthold, longhold])
        return nothing
    end
    if (ta.tradelabel in [longstrongclose, longclose]) && (cache.mc[:trademode] in [buysell, sellonly, quickexit]) && basecfg.sellenabled
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = freebase <= 2 * minimumbasequantity ? (freebase >= minimumbasequantity ? freebase : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), freebase)
        sufficientsellbalance = (basequantity <= freebase) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if sufficientsellbalance && exceedsminimumbasequantity
            oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : _withtradeauditcontext(cache, ta) do
                CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=0)
            end
            if !isnothing(oid)
                result = (trade=longclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longclose order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longclose order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no longclose $base due to sufficientsellbalance=$sufficientsellbalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (ta.tradelabel in [longbuy, longstrongbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(0f0, min(max(qtyacceleration * quotequantity/price, minimumbasequantity) * price, freeusdt - freeusdtfractionmargin * totalusdt) / price) #* keep 5% * totalusdt as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && ((basequantity + borrowedbase) > 0.0)
        # basequantity += borrowedbase # buy all short as well when switching to long
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / totalusdt 
        # basefraction = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt)

        # if base == "ADA"
        #     println("coin=$(ta.base) tradelabel=$(ta.tradelabel) price=$price basequantity=$basequantity sufficientbuybalance=$sufficientbuybalance minimumbasequantity=$minimumbasequantity quotequantity=$quotequantity freeusdt=$freeusdt totalusdt=$totalusdt")
        # end
    
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 3) && println("$(tradetime(cache)) skip $base longbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif sufficientbuybalance && exceedsminimumbasequantity
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : _withtradeauditcontext(cache, ta) do
                CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=0)
            end
            if !isnothing(oid)
                result = (trade=longbuy, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base longbuy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * totalusdt <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker longbuy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base longbuy due to sufficientbuybalance=$sufficientbuybalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (ta.tradelabel in [shortstrongbuy, shortbuy]) && (cache.mc[:trademode] == buysell) && basecfg.buyenabled
        basequantity = max(qtyacceleration * quotequantity/price, minimumbasequantity) * price
        sufficientbuybalance = ((basequantity - freebase) * price < freeusdt) && (basequantity > 0.0)
        basefraction = (sum(sum(eachcol(assets[assets.coin .== base, [:free, :locked, :borrowed]]))) + basequantity) * price / (totalusdt + totalborrowedusdt)
        if basefraction > cache.mc[:maxassetfraction] # base dominates assets
            (verbosity > 2) && println("$(tradetime(cache)) skip $base shortbuy: base dominates assets due to basefraction=$(basefraction) > maxassetfraction=$(cache.mc[:maxassetfraction])")
        elseif sufficientbuybalance
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : _withtradeauditcontext(cache, ta) do
                CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=2)
            end
            if !isnothing(oid)
                result = (trade=shortbuy, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortbuy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * totalusdt <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortbuy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), ($freeusdtfractionmargin * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no $base shortbuy due to sufficientbuybalance=$sufficientbuybalance")
        end
    elseif (ta.tradelabel in [shortclose, shortstrongclose]) && (cache.mc[:trademode] in [buysell, sellonly, quickexit]) && basecfg.sellenabled
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = borrowedbase <= 2 * minimumbasequantity ? (borrowedbase >= minimumbasequantity ? borrowedbase : minimumbasequantity) : minimumbasequantity # increase minimumbasequantity if otherwise a too small base aount remains that cannot be sold
        basequantity = max(0f0, min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), borrowedbase))
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if exceedsminimumbasequantity
            oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : _withtradeauditcontext(cache, ta) do
                CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=2)
            end
            if !isnothing(oid)
                result = (trade=shortclose, oid=oid)
                (verbosity > 2) && println("$(tradetime(cache)) created $base shortclose order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker shortclose order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets))")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no shortclose $base due to exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    end
    push!(cache.dbgdf, (
        taconfigid = isnothing(ta) ? missing : ta.configid,
        tatradelabel = isnothing(ta) ? missing : ta.tradelabel,
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

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
# USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")
function USDTmsg(assets)
    return string("USDT: total=$(round(Int, sum(assets.usdtvalue))), free=$(sum(assets.usdtvalue) > 0f0 ? round(Int, sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])/sum(assets.usdtvalue)*100) : 0f0)%")
end
function tradeadvicelessthan(ta1, ta2)
    closeset = [shortclose, shortstrongclose, allclose, longstrongclose, longclose]
    buyset = [shortstrongbuy, shortbuy, longbuy, longstrongbuy]
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

_strategyengine(cache::TradeCache) = Symbol(get(cache.mc, :strategy_engine, :classifier))

# ── Loop control ────────────────────────────────────────────────────────────

"Returns the current loop lifecycle state."
_loopstate_nolock(cache::TradeCache) = LoopState(Int(cache.mc[:loop_state]))
_setloopstate_nolock!(cache::TradeCache, s::LoopState) = (cache.mc[:loop_state] = s; nothing)

function loopstate(cache::TradeCache)
    lock(cache.looplock)
    try
        return _loopstate_nolock(cache)
    finally
        unlock(cache.looplock)
    end
end

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

"""
Request the loop to pause after the current tick.
Only effective when the loop state is `loop_running`.
"""
function pause!(cache::TradeCache)
    lock(cache.looplock)
    try
        (_loopstate_nolock(cache) == loop_running) && _setloopstate_nolock!(cache, loop_paused)
    finally
        unlock(cache.looplock)
    end
    return cache
end

"""
Resume a paused loop.
Only effective when the loop state is `loop_paused`.
"""
function resume!(cache::TradeCache)
    lock(cache.looplock)
    try
        if _loopstate_nolock(cache) == loop_paused
            _setloopstate_nolock!(cache, loop_running)
            notify(cache.loopcond; all=true)
        end
    finally
        unlock(cache.looplock)
    end
    return cache
end

"""
Request the loop to stop gracefully after the current tick completes.
Effective when loop state is `loop_running` or `loop_paused`.
"""
function stop!(cache::TradeCache)
    lock(cache.looplock)
    try
        st = _loopstate_nolock(cache)
        if st in (loop_running, loop_paused)
            _setloopstate_nolock!(cache, loop_stopping)
            notify(cache.loopcond; all=true)
        end
    finally
        unlock(cache.looplock)
    end
    return cache
end

# ── Strategy config ─────────────────────────────────────────────────────────

function _validatestrategyconfig!(mc::AbstractDict)
    openthreshold = Float32(mc[:strategy_openthreshold])
    closethreshold = Float32(mc[:strategy_closethreshold])
    buygain = Float32(mc[:strategy_buygain])
    sellgain = Float32(mc[:strategy_sellgain])
    limitreduction = Float32(mc[:strategy_limitreduction])
    maxwindow = Int(mc[:strategy_maxwindow])

    @assert 0f0 <= openthreshold <= 1f0 "strategy_openthreshold must be in [0, 1], got $(openthreshold)"
    @assert 0f0 <= closethreshold <= 1f0 "strategy_closethreshold must be in [0, 1], got $(closethreshold)"
    @assert 0f0 <= buygain <= 1f0 "strategy_buygain must be in [0, 1], got $(buygain)"
    @assert 0f0 <= sellgain <= 1f0 "strategy_sellgain must be in [0, 1], got $(sellgain)"
    @assert 0f0 <= limitreduction <= 1f0 "strategy_limitreduction must be in [0, 1], got $(limitreduction)"
    @assert maxwindow > 0 "strategy_maxwindow must be > 0, got $(maxwindow)"
    return mc
end

"Validate strategy runtime parameters stored in `TradeCache.mc`."
function _validatestrategyconfig!(cache::TradeCache)
    _validatestrategyconfig!(cache.mc)
    return cache
end

"Apply strategy runtime settings from a `TradingStrategy.GainSegment` and reset derived per-base state."
function apply_tradingstrategy!(mc::AbstractDict, gs::TradingStrategy.GainSegment; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    mc[:strategy_engine] = strategy_engine
    mc[:strategy_algorithm] = gs.algorithm
    mc[:strategy_openthreshold] = Float32(gs.openthreshold)
    mc[:strategy_closethreshold] = Float32(gs.closethreshold)
    mc[:strategy_buygain] = Float32(gs.buygain)
    mc[:strategy_sellgain] = Float32(gs.sellgain)
    mc[:strategy_limitreduction] = Float32(gs.limitreduction)
    mc[:strategy_maxwindow] = Int(gs.maxwindow)
    mc[:strategy_source] = String(source)

    strategy_state = get!(mc, :strategy_state, Dict{String, Any}())
    strategy_history = get!(mc, :strategy_history, Dict{String, Any}())
    empty!(strategy_state)
    empty!(strategy_history)
    return _validatestrategyconfig!(mc)
end

function apply_tradingstrategy!(cache::TradeCache, gs::TradingStrategy.GainSegment; strategy_engine::Symbol=:getgainsalgo, source::AbstractString="manual")
    apply_tradingstrategy!(cache.mc, gs; strategy_engine=strategy_engine, source=source)
    return cache
end

"Apply strategy runtime settings from a TrendDetector-style configuration reference."
function apply_trenddetector_strategy!(mc::AbstractDict, tdref)
    @assert hasproperty(tdref, :tradingstrategy) "tdref must expose field :tradingstrategy, got type=$(typeof(tdref))"
    gs = getproperty(tdref, :tradingstrategy)
    @assert gs isa TradingStrategy.GainSegment "tdref.tradingstrategy must be TradingStrategy.GainSegment, got type=$(typeof(gs))"
    source = hasproperty(tdref, :configname) ? "trenddetector:$(getproperty(tdref, :configname))" : "trenddetector"
    return apply_tradingstrategy!(mc, gs; strategy_engine=:getgainsalgo, source=source)
end

function apply_trenddetector_strategy!(cache::TradeCache, tdref)
    apply_trenddetector_strategy!(cache.mc, tdref)
    return cache
end

function _strategyhistory!(cache::TradeCache, base::AbstractString)
    return get!(cache.mc[:strategy_history], String(base)) do
        (
            predictionsdf=DataFrame(opentime=DateTime[], high=Float32[], low=Float32[], close=Float32[]),
            scores=Float32[],
            labels=Targets.TradeLabel[],
        )
    end
end

function _strategystate!(cache::TradeCache, base::AbstractString)
    return get!(cache.mc[:strategy_state], String(base)) do
        _validatestrategyconfig!(cache)
        gs = TradingStrategy.GainSegment(
            ;
            maxwindow=Int(cache.mc[:strategy_maxwindow]),
            openthreshold=Float32(cache.mc[:strategy_openthreshold]),
            closethreshold=Float32(cache.mc[:strategy_closethreshold]),
            algorithm=get(cache.mc, :strategy_algorithm, TradingStrategy.gain_reversal!),
            limitreduction=Float32(cache.mc[:strategy_limitreduction]),
        )
        gs.buygain = Float32(cache.mc[:strategy_buygain])
        gs.sellgain = Float32(cache.mc[:strategy_sellgain])
        gs
    end
end

"update if the record already exists, otherwise insert it."
function _upsert_getgainsalgo_sample!(history, ohlcv::Ohlcv.OhlcvData, label::Targets.TradeLabel, score)
    rowix = ohlcv.ix
    odf = Ohlcv.dataframe(ohlcv)
    @assert (1 <= rowix <= size(odf, 1)) "rowix=$(rowix) out of bounds for ohlcv rows=$(size(odf, 1))"
    opentime = odf[rowix, :opentime]
    high = Float32(odf[rowix, :high])
    low = Float32(odf[rowix, :low])
    close = Float32(odf[rowix, :close])
    sc = Float32(score)

    if size(history.predictionsdf, 1) > 0 && (history.predictionsdf[end, :opentime] == opentime)
        history.predictionsdf[end, :high] = high
        history.predictionsdf[end, :low] = low
        history.predictionsdf[end, :close] = close
        history.scores[end] = sc
        history.labels[end] = label
    else
        push!(history.predictionsdf, (opentime=opentime, high=high, low=low, close=close))
        push!(history.scores, sc)
        push!(history.labels, label)
    end
    return history
end

function _getgainsalgo_action2label(gs::TradingStrategy.GainSegment, fallback::Targets.TradeLabel=allclose)::Targets.TradeLabel
    if gs.buyta.orderlabel in [longbuy, longstrongbuy]
        return longbuy
    elseif gs.buyta.orderlabel in [shortbuy, shortstrongbuy]
        return shortbuy
    elseif gs.sellta.orderlabel in [longclose, longstrongclose]
        return longclose
    elseif gs.sellta.orderlabel in [shortclose, shortstrongclose]
        return shortclose
    end
    return fallback
end

function _getgainsalgo_advice!(cache::TradeCache, base::AbstractString, ta::Classify.TradeAdvice)
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    history = _strategyhistory!(cache, base)
    _upsert_getgainsalgo_sample!(history, ohlcv, ta.tradelabel, ta.probability)
    gs = _strategystate!(cache, base)
    lastix = length(history.scores)
    if lastix > 0
        TradingStrategy.getgains(gs, history.predictionsdf, history.scores, history.labels, false; lastix=lastix, openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
        ta.tradelabel = _getgainsalgo_action2label(gs, ta.tradelabel)
    end
    return ta
end

"""
Execute one trading tick: cancel pending orders, collect classifier/strategy advice for all
configured bases, execute trades, and handle daily trade-selection reload.
Called by the loop runners once per iterate step.
"""
function _tradestep!(cache::TradeCache)
    (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
    oo = CryptoXch.getopenorders(cache.xc)
    for ooe in eachrow(oo)  # cancel all open orders; amending maker orders causes rejections
        if CryptoXch.openstatus(ooe.status)
            CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(ooe.symbol).basecoin, ooe.orderid)
        end
    end
    assets = CryptoXch.portfolio!(cache.xc)
    _writeportfoliosnapshot!(cache, assets)
    tradeadvices = []
    for basecfg in eachrow(cache.cfg)
        Classify.supplement!(cache.cl)
        #TODO handle multiple classifiers per base
        tradeadvice = Classify.advice(cache.cl, basecfg.basecoin, cache.xc.currentdt, investment=nothing)
        if !isnothing(tradeadvice)
            if _strategyengine(cache) == :getgainsalgo
                tradeadvice = _getgainsalgo_advice!(cache, basecfg.basecoin, tradeadvice)
            end
            push!(tradeadvices, tradeadvice)
        else
            (verbosity > 3) && println("no trade advice for $(basecfg.basecoin)")
        end
    end
    if cache.mc[:usenewtrade]
    else # legacy trade!()
        sellbases = []
        buybases = []
        sort!(tradeadvices, lt=tradeadvicelessthan)  # close first, then buy high-gain first
        for ta in tradeadvices
            basecfg = first(filter(row -> row.basecoin == ta.base, cache.cfg))
            res = trade!(cache, basecfg, ta, assets)
            if !isnothing(res) && (res.trade in [longbuy, longstrongbuy, shortclose, shortstrongclose])
                push!(buybases, basecfg.basecoin)
            elseif !isnothing(res) && (res.trade in [longstrongclose, longclose, shortstrongbuy, shortbuy])
                push!(sellbases, basecfg.basecoin)
            elseif !isnothing(res)
                @warn "case not handled: $res"
            end
        end
        (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                          ")
    end
    if Time(cache.xc.currentdt) in cache.mc[:reloadtimes]
        assets = CryptoXch.portfolio!(cache.xc)
        (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
        tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
    end
    #TODO low prio: for closed orders check fees
    #TODO low prio: aggregate orders and transactions in bookkeeping
    return nothing
end

"Load or derive the initial trade configuration if `cache.cfg` is empty."
function _ensure_tradeloop_initialized!(cache::TradeCache)
    if size(cache.cfg, 1) == 0
        assets = CryptoXch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start loading trading strategy")
        if isnothing(read!(cache, cache.xc.startdt))
            (verbosity >= 2) && print("\r$(tradetime(cache)): start reassessing trading strategy")
            tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        end
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
end

"Log end-of-loop summary statistics."
function _tradefinish!(cache::TradeCache)
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    (verbosity >= 3) && @info (size(cache.xc.closedorders, 1) > 0) ? "$(EnvConfig.now()): closed orders log $(cache.xc.closedorders)" : "$(EnvConfig.now()): no closed orders"
    (verbosity >= 3) && @info (size(cache.xc.orders, 1) > 0) ? "$(EnvConfig.now()): open orders log $(cache.xc.orders)" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): closed orders $(size(cache.xc.closedorders, 1)), open orders $(size(cache.xc.orders, 1))"
    assets = CryptoXch.portfolio!(cache.xc)
    (verbosity >= 3) && @info "assets = $assets"
    (verbosity >= 2) && @info "total USDT = $(sum(assets.usdtvalue))"
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

"""
Start the trading loop (blocking). Selects backtest or live mode from `cache.xc.enddt` presence.
Use `stop!` to request early termination from another task.
"""
function start!(cache::TradeCache)
    if loopstate(cache) != loop_idle
        @warn "start! called but loop is not idle (state=$(loopstate(cache)))"
    end
    _ensure_tradeloop_initialized!(cache)
    _run_tradeloop!(cache)
    return cache
end

"""
Asynchronously start the trading loop in a background task.
Returns immediately with a `Task` handle. The loop executes in the background, and the caller
can control it from another task/thread using `pause!()`, `resume!()`, and `stop!()`.

When `skip_init=false` (default) the trade configuration is loaded or rebuilt if `cache.cfg` is empty.
Pass `skip_init=true` when the caller has already populated `cache.cfg`.

Returns:
    `Task`: a background task running `_run_tradeloop!(cache)`. 
    Caller can `wait(task)` for completion or check task status.

Example:
```julia
cache = Trade.setup_backtest(...)
task = Trade.async_start!(cache)
# ... from another task:
Trade.pause!(cache)   # pause the loop
Trade.resume!(cache)  # resume it
Trade.stop!(cache)    # request exit
result = wait(task)   # block until loop finishes
```
"""
function async_start!(cache::TradeCache; skip_init::Bool=false)
    if loopstate(cache) != loop_idle
        @warn "async_start! called but loop is not idle (state=$(loopstate(cache)))"
    end
    skip_init || _ensure_tradeloop_initialized!(cache)
    return @async _run_tradeloop!(cache)
end

"""
Execute exactly one trading tick without the iteration engine.
Useful for step-by-step debugging or custom replay harnesses.
The caller is responsible for advancing `cache.xc.currentdt` before calling `step!`.
"""
function step!(cache::TradeCache)
    _tradestep!(cache)
    return cache
end

"""
**`tradeloop`** — compatibility wrapper calling `start!`.
Prefer using `run_backtest!`, `run_live!`, or `start!` directly for new code.

+ get initial TradeStrategy config (if not present at entry) and refresh daily according to `reloadtimes`
+ get new exchange data (preferably non-blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non-blocking)
+ follow up on open orders (preferably non-blocking)
"""
function tradeloop(cache::TradeCache)
    start!(cache)
end

function tradelooptest(cache::TradeCache)
    for c in cache.xc
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
    end
end


end  # module

