# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, longbuy when an uptrend starts and longclose when it ends.
It generates the OHLCV data, executes the trades in a loop and selects the basecoins to trade.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, JDF, Statistics
using EnvConfig, Ohlcv, CryptoXch, Classify, Features, Targets

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
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: essential status messages, e.g. load and save messages, are reported
- 3: print debug info
"""
verbosity = 2

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
    function TradeCache(; xc=CryptoXch.XchCache(), cl=Classify.Classifier011(), trademode=notrade)
        cache = new(xc, DataFrame(), cl, Dict(), DataFrame())
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


TRADECONFIGFILE = "TradeConfig"

"""
Loads all USDT coins, checks continuous minimum volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to enable backtesting.
"""
function tradeselection!(tc::TradeCache, assetbases::Vector; datetime=tc.xc.startdt, assetonly=false, updatecache=false)
    datetime = floor(datetime, Minute(1))

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    # CryptoXch.removeallbases(tc.xc)  #* reuse what is in cache
    # Classify.removebase!(tc.cl, nothing)  #* reuse what is in cache

    usdtdf = CryptoXch.getUSDTmarket(tc.xc)  # superset of coins with 24h volume price change and last price
    if assetonly
        usdtdf = filter(row -> row.basecoin in assetbases, usdtdf)
    end
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent, :lastprice)
    if size(tc.cfg, 1) == 0
        (verbosity >= 1) && @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :datetime] .= datetime
    # tc.cfg[:, :validbase] = [CryptoXch.validbase(tc.xc, base) for base in tc.cfg[!, :basecoin]] # is already filtered by getUSDTmarket
    minimumdayquotevolumemillion = round(Ohlcv.liquiddailyminimumquotevolume() / 1000000, digits=0) # ignore allcoins with less than liquiddailyminimumquotevolume
    tc.cfg[:, :minquotevol] = tc.cfg[:, :quotevolume24h_M] .>= minimumdayquotevolumemillion
    tc.cfg[:, :continuousminvol] .= false
    tc.cfg[:, :inportfolio] = [base in assetbases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :classifieraccepted] .= false
    tc.cfg[:, :noinvest] .= false
    tc.cfg[:, :buyenabled] .= false
    tc.cfg[:, :sellenabled] .= false
    tc.cfg[:, :whitelisted] = [base in tc.mc[:whitelistcoins] for base in tc.cfg[!, :basecoin]]

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
    xcbases = CryptoXch.bases(tc.xc)
    for (ix, row) in enumerate(eachrow(tc.cfg))
        (verbosity >= 2) && updatecache &&  print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) including cache update                           ")
        (verbosity >= 2) && !updatecache && print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) without cache update                             ")
        if row.basecoin in xcbases
            ohlcv = CryptoXch.ohlcv(tc.xc, row.basecoin)
            CryptoXch.cryptoupdate!(tc.xc, ohlcv, datetime - Year(20), datetime)
        else
            ohlcv = CryptoXch.cryptodownload(tc.xc, row.basecoin, "1m", datetime - Year(20), datetime)
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
    removebases = setdiff(xcbases, classifierbases)
    for rb in removebases  # remove coins that were not accepted by the classifier, e.g. if requiredminutes is insufficient
        CryptoXch.removebase!(tc.xc, rb)
    end
    xcbases = CryptoXch.bases(tc.xc)
    @assert Set(xcbases) == Set(classifierbases) "Set(xcbases)=$(xcbases) != Set(classifierbases)=$(classifierbases)"

    tc.cfg[:, :classifieraccepted] = [base in classifierbases for base in tc.cfg[!, :basecoin]]

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

function _cfgfilename(timestamp::Union{Nothing, DateTime}, ext="jdf")
    if isnothing(timestamp)
        cfgfilename = TRADECONFIGFILE
    else
        cfgfilename = join([TRADECONFIGFILE, Dates.format(timestamp, "yy-mm-dd")], "_")
    end
    return EnvConfig.datafile(cfgfilename, TRADECONFIGFILE, ".jdf")
end

"Saves the trade configuration. If timestamp!=nothing then save 2x with and without timestamp in filename otherwise only without timestamp"
function write(tc::TradeCache, timestamp::Union{Nothing, DateTime}=nothing)
    if (size(tc.cfg, 1) == 0)
        @warn "trade config is empty - not stored"
        return
    end
    cfgfilename = _cfgfilename(nothing)
    # EnvConfig.savebackup(cfgfilename)
    if isdir(cfgfilename)
        rm(cfgfilename; force=true, recursive=true)
    end
    (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
    JDF.savejdf(cfgfilename, tc.cfg)  # parent(tc.cfg))
    if !isnothing(timestamp)
        cfgfilename = _cfgfilename(timestamp)
        (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
        JDF.savejdf(cfgfilename, tc.cfg)  # parent(tc.cfg))
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists but does not load the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function readconfig!(tc::TradeCache, datetime)
    tc.cfg = df = DataFrame() # old cfg is in either case discarded
    cfgfilename = _cfgfilename(datetime, "jdf")
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
    end
    if !isnothing(df) && (size(df, 1) > 0 )
        (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config from $cfgfilename")
        tc.cfg = df
        tc.cfg[:, :whitelisted] = [coin in tc.mc[:whitelistcoins] for coin in tc.cfg[!, :basecoin]]
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& (tc.cfg[:, :minquotevol] .&& tc.cfg[!, :continuousminvol]) .&& tc.cfg[!, :whitelisted]
        return tc
    else
        (verbosity >=2) && !isnothing(df) && println("Loading $cfgfilename failed")
        return nothing
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists. Also loads the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function read!(tc::TradeCache, datetime=nothing)
    datetime = isnothing(datetime) ? nothing : floor(datetime, Minute(1))
    tc = readconfig!(tc, datetime)
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
function assetsconfig!(tc::TradeCache, datetime=nothing)
    tc = readconfig!(tc, datetime)
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
            oid = (cache.mc[:trademode] == notrade) ? "SellSpotSim" : CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=0)
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
            oid = (cache.mc[:trademode] == notrade) ? "BuySpotSim" : CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=0)
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
            oid = (cache.mc[:trademode] == notrade) ? "SellMarginSim" : CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=2)
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
            oid = (cache.mc[:trademode] == notrade) ? "BuyMarginSim" : CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true, marginleverage=2)
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

"""
**`tradeloop`** has to
+ get initial TradinStrategy config (if not present at entry) and refresh daily according to `reloadtimes`
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)
"""
function tradeloop(cache::TradeCache)
    # TODO add hooks to enable coupling to the cockpit visualization
    # @info "$(EnvConfig.now()): trading bases=$(CryptoXch.bases(cache.xc)) period=$(isnothing(cache.xc.startdt) ? "start canned" : cache.xc.startdt) enddt=$(isnothing(cache.xc.enddt) ? "start canned" : cache.xc.enddt)"
    # @info "$(EnvConfig.now()): trading with open orders $(CryptoXch.getopenorders(cache.xc))"
    # @info "$(EnvConfig.now()): trading with assets $(CryptoXch.balances(cache.xc))"
    # for base in keys(cache.xc.bases)
    #     syminfo = CryptoXch.minimumqty(cache.xc, CryptoXch.symboltoken(base))
    #     @info "$syminfo"
    # end
    if size(cache.cfg, 1) == 0
        assets = CryptoXch.balances(cache.xc)
        (verbosity >= 2) && print("\r$(tradetime(cache)): start loading trading strategy")
        if isnothing(read!(cache, cache.xc.startdt))
            (verbosity >= 2) && print("\r$(tradetime(cache)): start reassessing trading strategy")
            tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.startdt)
        end
        cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :] # limit coins to those that are tradable
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
    try
        for c in cache.xc
            (verbosity > 3) && println("startdt=$(cache.xc.startdt), currentdt=$(cache.xc.currentdt), enddt=$(cache.xc.enddt)")
            oo = CryptoXch.getopenorders(cache.xc)
            for ooe in eachrow(oo)  # all orders to be cancelled because amending maker orders will lead to rejections and "Order does not exist" returns
                if CryptoXch.openstatus(ooe.status)
                    CryptoXch.cancelorder(cache.xc, CryptoXch.basequote(ooe.symbol).basecoin, ooe.orderid)
                end
            end
            assets = CryptoXch.portfolio!(cache.xc)
            tradeadvices = []
            for basecfg in eachrow(cache.cfg)
                Classify.supplement!(cache.cl)
                #TODO handle multiple classifier with different longbuy/longclose signals for one base
                #TODO  - provide multiple advices from differen classifiers
                #TODO  - limits per classifier + config and base => classifier provides relative amount for that base
                #TODO  - classifier provides: class name, cfgid, relative invest amount, price, trade label
                #TODO  - Trade adds: investment ID, average actual longbuy price, startdt buying, enddt buying, actual amount bought
                #TODO  - ask for advice per investment with specific classifier + config bought at specific price
                tradeadvice = Classify.advice(cache.cl, basecfg.basecoin, cache.xc.currentdt, investment=nothing)
                if !isnothing(tradeadvice)
                    push!(tradeadvices, tradeadvice)  #TODO old trade advice to be added
                else 
                    (verbosity > 3) && println("no trade advice for $(basecfg.basecoin)")
                end
                # print("\r$(tradetime(cache)): $(USDTmsg(assets))")
            end
            if cache.mc[:usenewtrade]
            else # old trade!()
                sellbases = []
                buybases = []
                    # sort to fill execute sell advices to create free quote to buy then sort according to hourlygain to buy high hourly gain first
                sort!(tradeadvices, lt=tradeadvicelessthan)
                for ta in tradeadvices
                    # (verbosity > 3) && println("trade advice for $(ta.base): $(ta.tradelabel)")
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
                (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                                                    ")
            end
            if Time(cache.xc.currentdt) in cache.mc[:reloadtimes]  # e.g. [Time("04:00:00"))]
                assets = CryptoXch.portfolio!(cache.xc)
                (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
                tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
                cache.cfg = cache.cfg[(cache.cfg[!, :buyenabled] .|| cache.cfg[:, :sellenabled]), :]  # limit coins to those that are tradable
                (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
            end
            #TODO low prio: for closed orders check fees
            #TODO low prio: aggregate orders and transactions in bookkeeping
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
                        (verbosity >= 1) && println("fr.func=$(fr.func) fr.linfo=$(fr.linfo) fr.file=$(fr.file) fr.line=$(fr.line) fr.from_c=$(fr.from_c) fr.inlined=$(fr.inlined) fr.pointer=$(fr.pointer)")
                    end
                end
            end
        end
    end
    (verbosity >= 2) && println("$(tradetime(cache)): finished trading core loop")
    (verbosity >= 3) && @info (size(cache.xc.closedorders, 1) > 0) ? "$(EnvConfig.now()): verbosity=$verbosity closed orders log $(cache.xc.closedorders)" : "$(EnvConfig.now()): no closed orders"
    (verbosity >= 3) && @info (size(cache.xc.orders, 1) > 0) ? "$(EnvConfig.now()): open orders log $(cache.xc.orders)" : "$(EnvConfig.now()): no open orders"
    (verbosity >= 2) && @info "$(EnvConfig.now()): closed orders $(size(cache.xc.closedorders, 1)), open orders $(size(cache.xc.orders, 1))"
    assets = CryptoXch.portfolio!(cache.xc)
    (verbosity >= 3) && @info "assets = $assets"
    totalusdt = sum(assets.usdtvalue)
    (verbosity >= 2) && @info "total USDT = $totalusdt"
    #TODO save investlog
end

function tradelooptest(cache::TradeCache)
    for c in cache.xc
        println("$(Dates.now(UTC)) $c  $(CryptoXch.ohlcv(c.xc, "BTC"))")
    end
end


end  # module
