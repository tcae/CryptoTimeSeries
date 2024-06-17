# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.
It generates the OHLCV data, executes the trades in a loop and selects the basecoins to trade.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV, JDF
using EnvConfig, Ohlcv, CryptoXch, Classify, Features

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

"""
- buysell is the normal trade mode
- sellonly disables buying but sells according to normal sell behavior
- quickexit sells all assets as soon as possible
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
    tradegapminutes::Integer  # required to buy/sell in Minute(tradegapminutes) time gaps - will be extended on low budget to satisfy minimum spending constraints per trade
    topx::Integer  # defines how many best candidates are considered for trading
    tradeassetfraction # defines the default trade fraction of an assets versus total assets value as a function(maxassetfraction, count buy sell coins)
    maxassetfraction # defines the maximum ration of (a specific asset) / ( total assets) - sell only if this is exceeded
    lastbuy::Dict  # Dict(base, DateTime) required to buy in = Minute(tradegapminutes) time gaps
    lastsell::Dict  # Dict(base, DateTime) required to sell in = Minute(tradegapminutes) time gaps
    reloadtimes::Vector{Time}  # provides time info when the portfolio of coin candidates shall be reassessed
    trademode::TradeMode
    function TradeCache(; tradegapminutes=1, topx=50, maxassetfraction=0.10, xc=CryptoXch.XchCache(true), cl=Classify.Classifier001(), reloadtimes=[], trademode=buysell)
        new(xc, DataFrame(), cl, tradegapminutes, topx, 1f0, maxassetfraction, Dict(), Dict(), reloadtimes, trademode)
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


MINIMUMDAYUSDTVOLUME = 2*1000000
TRADECONFIGFILE = "TradeConfig"

function continuousminimumvolume(ohlcv::Ohlcv.OhlcvData, datetime::Union{DateTime, Nothing}, checkperiod=Day(1), accumulateminutes=5, minimumaccumulatequotevolume=1000f0)::Bool
    if size(ohlcv.df, 1) == 0
        (verbosity >= 4) && println("$(ohlcv.base) has an empty dataframe")
        return false
    end
    datetime = isnothing(datetime) ? ohlcv.df[end, :opentime] : datetime
    endix = Ohlcv.rowix(ohlcv, datetime)
    startdt = datetime - checkperiod
    startix = Ohlcv.rowix(ohlcv, startdt)
    vol = minimumaccumulatequotevolume
    count = countok = 0
    for ix in startix:endix
        if ((ix - startix) % accumulateminutes) == 0
            if ix > startix
                count += 1
                if vol >= minimumaccumulatequotevolume
                    countok += 1
                end
            end
            vol = ohlcv.df[ix, :basevolume] * ohlcv.df[ix, :pivot]
        else
            vol += ohlcv.df[ix, :basevolume] * ohlcv.df[ix, :pivot]
        end
    end
    if count == countok
        return true
    else
        (verbosity >= 3) && println("\r$(ohlcv.base) has ($(count - countok) of $count) in $(round(((count - countok) / count) * 100.0))% insuficient continuous $(accumulateminutes) minimum volume of $minimumaccumulatequotevolume $(EnvConfig.cryptoquote) over a period of $checkperiod ending $datetime")
        return false
    end
end

"""
Loads all USDT coins, checks last24h volume and other continuous minimum volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to enable backtesting.
"""
function tradeselection!(tc::TradeCache, assetbases::Vector; datetime=tc.xc.startdt, minimumdayquotevolume=MINIMUMDAYUSDTVOLUME, assetonly=false, updatecache=false)
    datetime = floor(datetime, Minute(1))

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    CryptoXch.removeallbases(tc.xc)
    Classify.removebase!(tc.cl, nothing)

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
    minimumdayquotevolume = minimumdayquotevolume / 1000000
    tc.cfg[:, :minquotevol] = tc.cfg[:, :quotevolume24h_M] .>= minimumdayquotevolume
    tc.cfg[:, :continuousminvol] .= false
    tc.cfg[:, :inportfolio] = [base in assetbases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :classifieraccepted] .= false
    tc.cfg[:, :buyenabled] .= false
    tc.cfg[:, :sellenabled] .= false
    (verbosity >= 3) && println("verbosity=$verbosity describe(tc.cfg, :all)")

    # download latest OHLCV and classifier features
    df = @view tc.cfg[tc.cfg[:, :minquotevol], :]
    count = size(df, 1)
    if updatecache
        for (ix, row) in enumerate(eachrow(df))
            (verbosity >= 2) && print("\r$(EnvConfig.now()) updating $(row.basecoin) ($ix of $count) including cache update                           ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, row.basecoin, "1m", datetime - Year(10), datetime)
            Classify.addbase!(tc.cl, ohlcv)
            Classify.writetargetsfeatures(tc.cl)
            CryptoXch.removeallbases(tc.xc)  # avoid slow down due to memory overload
            Classify.removebase!(tc.cl, nothing)  # avoid slow down due to memory overload
        end
    end
    for (ix, row) in enumerate(eachrow(df))
        (verbosity >= 2) && print("\r$(EnvConfig.now()) loading $(row.basecoin) ($ix of $count)                                                  ")
        CryptoXch.setstartdt(tc.xc, datetime - Minute(Classify.requiredminutes(tc.cl)-1))
        ohlcv = CryptoXch.addbase!(tc.xc, row.basecoin, datetime - Minute(Classify.requiredminutes(tc.cl)-1), tc.xc.enddt)
        Classify.addbase!(tc.cl, ohlcv)
        row.continuousminvol = continuousminimumvolume(ohlcv, datetime)
    end
    classifierbases = Classify.bases(tc.cl)
    tc.cfg[:, :classifieraccepted] = [base in classifierbases for base in tc.cfg[!, :basecoin]]
    tc.cfg = @view tc.cfg[tc.cfg[!, :minquotevol] .&& tc.cfg[!, :classifieraccepted], :]

    if assetonly
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted]
    else
        tc.cfg[:, :buyenabled] .= tc.cfg[!, :classifieraccepted] .&& tc.cfg[!, :continuousminvol]
        tc.cfg[:, :sellenabled] .= tc.cfg[!, :buyenabled] .|| (tc.cfg[!, :inportfolio] .&& tc.cfg[!, :classifieraccepted])
    end

    (verbosity >= 3) && println("$(CryptoXch.ttstr(tc.xc)) result of tradeselection! $(tc.cfg)")
    if !assetonly
        write(tc, datetime)
    end
    (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) trained and saved trade config data including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    return tc
end

"""
tradeselection! loads all data up to enddt. This function will reduce the memory starting from startdt plus OHLCV data required for feature calculation.
"""
function timerangecut!(tc::TradeCache, startdt, enddt)
    for ohlcv in CryptoXch.ohlcv(tc.xc)
        tcix = findfirst(x -> x == ohlcv.base, tc.cfg[!, :basecoin])
        if isnothing(tcix)
            CryptoXch.removebase!(tc.xc, ohlcv.base)
        else
            Classify.timerangecut!(tc.cl, startdt, enddt)
        end
    end
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
    # EnvConfig.checkbackup(cfgfilename)
    if isdir(cfgfilename)
        rm(cfgfilename; force=true, recursive=true)
    end
    (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
    JDF.savejdf(cfgfilename, parent(tc.cfg))
    if !isnothing(timestamp)
        cfgfilename = _cfgfilename(timestamp)
        (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
        JDF.savejdf(cfgfilename, parent(tc.cfg))
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists but does not load the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function readconfig!(tc::TradeCache, datetime)
    df = DataFrame()
    cfgfilename = _cfgfilename(datetime, "jdf")
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
    end
    if !isnothing(df) && (size(df, 1) > 0 )
        (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config from $cfgfilename")
        tc.cfg = df
        tc.cfg = @view tc.cfg[tc.cfg[!, :minquotevol] .&& tc.cfg[!, :classifieraccepted], :]
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
    if !isnothing(tc) && !isnothing(tc.cfg) && (size(tc.cfg, 1) > 0)
        clvec = []
        df = tc.cfg
        rows = size(df, 1)
        for ix in eachindex(df[!, :basecoin])
            (verbosity >= 2) && print("\r$(EnvConfig.now()) loading $(df[ix, :basecoin]) from trade config ($ix of $rows)                                                  ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", datetime-Minute(Classify.requiredminutes(tc.cl)-1), datetime)
            Classify.addbase!(tc.cl, ohlcv)
        end
        classifierbases = Classify.bases(tc.cl)
        if (length(classifierbases) != count(tc.cfg[:, :classifieraccepted]))
            @error "length(classifierbases)=$(length(classifierbases)) != count(tc.cfg[:, :classifieraccepted])=$(count(tc.cfg[:, :classifieraccepted]))"
        end
        tc.cfg[:, :classifieraccepted] = [base in classifierbases for base in tc.cfg[!, :basecoin]]
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) loaded trade config data including $rows base classifier (ohlcv, features) data      ")
    end
    return !isnothing(df) && (size(df, 1) > 0) ? tc : nothing
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
function addassetsconfig!(tc::TradeCache, assets=CryptoXch.portfolio!(tc.xc))
    sort!(assets, [:coin])  # for readability only

    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:locked, :free])]
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


MAKER_CORRECTION = 0.0005
makerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? MAKER_CORRECTION : -MAKER_CORRECTION))
TAKER_CORRECTION = 0.0001
takerfeeprice(ohlcv::Ohlcv.OhlcvData, tp::Classify.InvestProposal) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix] * (1 + (tp == sell ? -TAKER_CORRECTION : TAKER_CORRECTION))
currentprice(ohlcv::Ohlcv.OhlcvData) = Ohlcv.dataframe(ohlcv).close[ohlcv.ix]
maxconcurrentbuycount() = 10  # regrwindow / 2.0  #* heuristic

"Iterate through all orders and adjust or create new order. All open orders should be cancelled before."
function trade!(cache::TradeCache, basecfg::DataFrameRow, tp::Classify.InvestProposal, assets::AbstractDataFrame)
    stopbuying = false
    sellbuyqtyratio = 2 # sell qty / buy qty per order, if > 1 sell quicker than buying it
    qtyacceleration = 4 # if > 1 then increase buy and sell order qty by this factor
    executed = noop
    base = basecfg.basecoin
    totalusdt = sum(assets.usdtvalue)
    freeusdt = sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])
    freebase = sum(assets[assets[!, :coin] .== base, :free])
    # 1800 = likely maxccbuycount
    quotequantity = cache.tradeassetfraction * totalusdt / maxconcurrentbuycount()
    ohlcv = CryptoXch.ohlcv(cache.xc, base)
    dtnow = Ohlcv.dataframe(ohlcv).opentime[Ohlcv.ix(ohlcv)]
    price = currentprice(ohlcv)
    minimumbasequantity = CryptoXch.minimumbasequantity(cache.xc, base, price)
    minqteratio = round(Int, (minimumbasequantity * price) / quotequantity)  # if quotequantity target exceeds minimum quote constraints then extend gaps because spending budget is low
    tradegapminutes = minqteratio > 1 ? cache.tradegapminutes * minqteratio : cache.tradegapminutes
    basedominatesassets = (sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt) > cache.maxassetfraction
    if (tp == sell) && (cache.trademode in [buysell, sellonly, quickexit]) && basecfg.sellenabled
        # now adapt minimum, if otherwise a too small remainder would be left
        minimumbasequantity = freebase <= 2 * minimumbasequantity ? (freebase >= minimumbasequantity ? freebase : minimumbasequantity) : minimumbasequantity
        basequantity = min(max(sellbuyqtyratio * qtyacceleration * quotequantity/price, minimumbasequantity), freebase)
        sufficientsellbalance = (basequantity <= freebase) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        tradegapminutes = basedominatesassets ? 1 : tradegapminutes  # accelerate selloff if basedominatesassets
        if sufficientsellbalance && exceedsminimumbasequantity && (!(base in keys(cache.lastsell)) || (cache.lastsell[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createsellorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastsell[base] = dtnow
                executed = sell
                (verbosity > 2) && println("$(tradetime(cache)) created $base sell order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker sell order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes")
            end
        else
            (verbosity > 3) && println("$(tradetime(cache)) no sell $base due to sufficientsellbalance=$sufficientsellbalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    elseif (tp == buy) && (cache.trademode == buysell) && basecfg.buyenabled
        basequantity = min(max(qtyacceleration * quotequantity/price, minimumbasequantity) * price, freeusdt - 0.01 * totalusdt) / price #* keep 1% * totalusdt as head room
        sufficientbuybalance = (basequantity * price < freeusdt) && (basequantity > 0.0)
        exceedsminimumbasequantity = basequantity >= minimumbasequantity
        if basedominatesassets
            (verbosity > 2) && println("$(tradetime(cache)) skip $base buy due to basefraction=$(sum(assets[assets.coin .== base, :usdtvalue]) / totalusdt) > maxassetfraction=$(cache.maxassetfraction)")
            return
        end
        if sufficientbuybalance && exceedsminimumbasequantity && (!(base in keys(cache.lastbuy)) || (cache.lastbuy[base] + Minute(tradegapminutes) <= dtnow))
            oid = CryptoXch.createbuyorder(cache.xc, base; limitprice=nothing, basequantity=basequantity, maker=true)
            if !isnothing(oid)
                cache.lastbuy[base] = dtnow
                executed = buy
                (verbosity > 2) && println("$(tradetime(cache)) created $base buy order with oid $oid, limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
            else
                (verbosity > 2) && println("$(tradetime(cache)) failed to create $base maker buy order with limitprice=$price and basequantity=$basequantity (min=$minimumbasequantity) = quotequantity=$(basequantity*price), $(USDTmsg(assets)), tgm=$tradegapminutes, (0.01 * totalusdt <= $freeusdt)")
                # println("sufficientbuybalance=$sufficientbuybalance=($basequantity * $price * $(1 + cache.xc.feerate + 0.001) < $freeusdt), exceedsminimumbasequantity=$exceedsminimumbasequantity=($basequantity > $minimumbasequantity=(1.01 * max($(syminfo.minbaseqty), $(syminfo.minquoteqty)/$price)))")
                # println("freeusdt=sum($(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])), EnvConfig.cryptoquote=$(EnvConfig.cryptoquote)")
            end
        else
            (verbosity > 2) && println("$(tradetime(cache)) no buy due to sufficientbuybalance=$sufficientbuybalance, exceedsminimumbasequantity=$exceedsminimumbasequantity")
        end
    end
    return executed
end

tradetime(cache::TradeCache) = CryptoXch.ttstr(cache.xc)
# USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), locked=$(round(Int, sum(assets.locked .* assets.usdtprice))), free=$(round(Int, sum(assets.free .* assets.usdtprice)))")
USDTmsg(assets) = string("USDT: total=$(round(Int, sum(assets.usdtvalue))), free=$(round(Int, sum(assets[assets[!, :coin] .== EnvConfig.cryptoquote, :free])/sum(assets.usdtvalue)*100))%")

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
        (verbosity > 2) && @info "$(tradetime(cache)) initial trading strategy: $(cache.cfg)"
    end
    cache.tradeassetfraction = min(cache.maxassetfraction, 1/(count(cache.cfg[!, :buyenabled]) > 0 ? count(cache.cfg[!, :buyenabled]) : 1))
    if (verbosity > 2)
        println("\n$(tradetime(cache)): cache.xc.startdt=$(cache.xc.startdt), cache.xc.currentdt=$(cache.xc.currentdt), cache.xc.enddt=$(cache.xc.enddt)")
        for ohlcv in cache.cl.bd
            println(ohlcv)
        end
    end
    # timerangecut!(cache, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
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
            sellbases = []
            buybases = []
            for basecfg in eachrow(cache.cfg)
                Classify.supplement!(cache.cl)
                tp = Classify.advice(cache.cl, basecfg.basecoin, cache.xc.currentdt)
                # print("\r$(tradetime(cache)): $(USDTmsg(assets))")
                executed = trade!(cache, basecfg, tp, assets)
                if executed == buy
                    push!(buybases, basecfg.basecoin)
                elseif executed == sell
                    push!(sellbases, basecfg.basecoin)
                end
            end
            (verbosity >= 2) && print("\r$(tradetime(cache)): $(USDTmsg(assets)), bought: $(buybases), sold: $(sellbases)                                                                    ")
            if Time(cache.xc.currentdt) in cache.reloadtimes  # e.g. [Time("04:00:00"))]
                assets = CryptoXch.portfolio!(cache.xc)
                (verbosity >= 2) && println("\n$(tradetime(cache)): start reassessing trading strategy")
                tradeselection!(cache, assets[!, :coin]; datetime=cache.xc.currentdt, updatecache=true)
                (verbosity >= 2) && @info "$(tradetime(cache)) reassessed trading strategy: $(cache.cfg)"
                # timerangecut!(cache, cache.xc.startdt, isnothing(cache.xc.enddt) ? cache.xc.currentdt : cache.xc.enddt)
                cache.tradeassetfraction = min(cache.maxassetfraction, 1/(count(cache.cfg[!, :buyenabled]) > 0 ? count(cache.cfg[!, :buyenabled]) : 1))
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
