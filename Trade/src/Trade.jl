# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
*Trade* is the top level module that shall  follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.
It generates the OHLCV data, executes the trades in a loop but delegates the trade strategy to *TradingStrategy*.
"""
module Trade

using Dates, DataFrames, Profile, Logging, CSV
using EnvConfig, Ohlcv, TradingStrategy, CryptoXch, Features, Classify

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

minimumdayusdtvolume = 10000000  # per day = 6944 per minute

mutable struct BaseInfo
    ohlcv::Ohlcv.OhlcvData
    classifier::Classify.AbstractClassifier
end

"""
*Cache* contains the recipe and state parameters for the **tradeloop** as parameter. Recipe parameters to create a *Cache* are
+ *backtestperiod* is the *Dates* period of the backtest (in case *backtestchunk* > 0)
+ *backtestenddt* specifies the last *DateTime* of the backtest
+ *baseconstraint* is an array of base crypto strings that constrains the crypto bases for trading else if *nothing* there is no constraint

"""
mutable struct Cache
    backtestperiod  # nothing or < Dates.Minute(1) in case of no backtest, otherwise execute backtest
    backtestenddt  # only considered if backtestperiod >= Dates.Minute(1)
    bd  # ::Dict{String, BaseInfo}
    openorders  # DataFrame as provided by CryptoXch
    portfolio  # DatFrame as provided by CryptoXch
    messagelog  # fileid
    function Cache(backtestperiod::Dates.Period=Dates.Minute(0), backtestenddt::Dates.DateTime=dummytime(), symbolclassifiers::Dict{String, Classify.AbstractClassifier})
        bd = Dict()
        for sc in symbolclassifiers
            bq = CryptoXch.basequote(sc[1])
            if isnothing(bq)
                @warning "Don't recognize $(sc[1]) as valid symbol with quotecoin $(EnvConfig.cryptoquote) - not considered as valid symbol/classifier pair"
            else
                cls = bd[bq[1]] = sc[2]
                base = sc[1]
                enddt = floor(Dates.now(Dates.UTC), Dates.Minute(1))
                initialperiod = Dates.Minute(Classify.requiredminutes(cls))
                startbacktest = nothing
                if backtestperiod >= Dates.Minute(1)  # this is a backtest session
                    enddt = floor(backtestenddt, Dates.Minute(1))
                    initialperiod += backtestperiod
                    startbacktest = floor(enddt - backtestperiod, Dates.Minute(1))
                end
                startdt = floor(enddt - initialperiod, Dates.Minute)
                ohlcv = CryptoXch.cryptodownload(base, "1m", startdt, enddt)
                CryptoXch.timerangecut!(ohlcv, startdt, enddt)
                Ohlcv.setix!(ohlcv, isnothing(startbacktest) ? nothing : Ohlcv.rowix(ohlcv, startbacktest))
                bd[base] = BaseInfo(ohlcv, cls)
            end
        end
        messagelog = open(EnvConfig.logpath("messagelog_$(EnvConfig.runid()).txt"), "w")
        new(backtestperiod, backtestenddt, bd, nothing, nothing, messagelog)
    end
end

ohlcvdf(cache, base) = Ohlcv.dataframe(cache.bd[base].ohlcv)
ohlcv(cache, base) = cache.bd[base].ohlcv
classifier(cache, base) = cache.bd[base].classifier
backtest(cache) = cache.backtestperiod >= Dates.Minute(1)
dummytime() = DateTime("2000-01-01T00:00:00")

function freelocked(portfoliodf, base)
    pdf = portfoliodf[portfoliodf.base .== base, :]
    if size(pdf, 1) > 0
        return sum(pdf.free), sum(pdf.locked)
    else
        return 0.0, 0.0
    end
end


"""
Determines which bases (always with USDT as quotecoin) to trade, load their free and locked amounts, load their recent ohlcv.
- bases that are in the wallet - traded only so far to close the open orders
- bases that are in a prescriped list
- bases that fullfill certain criteria to be selected, i.e. 24h USDT trade volume > 10million USDT and confirmation from Classifier
"""
function preparetradecache!(cache::Cache)
    usdtdf = CryptoXch.getUSDTmarket()
    cache.openorders = CryptoXch.getopenorders()
    cache.portfolio = CryptoXch.balances()
    cache.portfolio = CryptoXch.portfolio!(cache.portfolio, usdtdf)
    usdtdf = isnothing(cache.baseconstraint) ? usdtdf : filter(row -> (row.basecoin in cache.baseconstraint), usdtdf)
    # cache.usdtfree, cache.usdtlocked = freelocked(cache.portfolio, "usdt")

    println("trading $(keys(cache.bd))")
    @info "trading $(keys(cache.bd))"
    # no need to cache assets because they are implicitly stored via keys(bd)
    reportliquidity(cache, nothing)
    return cache
end

function sleepuntilnextminute(lastdt)
    enddt = floor(Dates.now(Dates.UTC), Dates.Minute)
    if lastdt == enddt
        nowdt = Dates.now(Dates.UTC)
        nextdt = lastdt + Dates.Minute(1)
        period = Dates.Millisecond(nextdt - floor(nowdt, Dates.Millisecond))
        sleepseconds = floor(period, Dates.Second)
        sleepseconds = Dates.value(sleepseconds) + 1
        @info "trade loop sleep seconds: $sleepseconds"
        sleep(sleepseconds)
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)
    end
    return enddt
end

function writetradelogs(cache::Cache)
    CSV.write(EnvConfig.logpath("openorders.csv"), cache.openorders)
    flush(cache.messagelog)
end

function readopenorders(cache::Cache)
    CSV.read(EnvConfig.logpath("openorders.csv"))
    cache.openorders = CSV.File(EnvConfig.logpath("openorders.csv")) |> DataFrame
end

"""
append most recent ohlcv data as well as corresponding features
returns `true` if successful appended else `false`
"""
function appendmostrecent!(cache::Cache, base)
    global count = 0
    continuetrading = false
    df = ohlcvdf(cache, base)
    ohlcv = ohlcv(cache, base)
    if backtest(cache) # consume next backtest data
        ix = Ohlcv.setix!(ohlcv, Ohlcv.ix(ohlcv) + 1)
        if (ix > lastindex(df, 1)) || (df.opentime[ix] > cache.backtestenddt)
            continuetrading = false
            @info "stop trading loop due to backtest ohlcv for $base exhausted - count = $count"
            return continuetrading
        end
        continuetrading = true
        if (Ohlcv.ix(ohlcv) % 1000) == 0
            println("continue at ix=$(Ohlcv.ix(ohlcv)) < size=$(size(df, 1)) continue=$continuetrading")
            writetradelogs(cache)
        end
    else  # no backtest
        lastdt = df.opentime[end]
        enddt = sleepuntilnextminute(lastdt)
        startdt = df.opentime[begin]  # stay with start until tradeloop cleanup
        ohlcvix = lastindex(df, 1)
        CryptoXch.cryptoupdate!(ohlcv, startdt, enddt)
        df = ohlcvdf(cache, base)
        println("extended from $lastdt to $enddt -> check df: $(df.opentime[begin]) - $(df.opentime[end]) size=$(size(df,1))")
        # ! TODO implement error handling
        @assert lastdt == df.opentime[ohlcvix]  # make sure begin wasn't cut
        ohlcvix = Ohlcv.setix!(ohlcv, lastindex(df, 1))
        if lastdt < df.opentime[end]
            continuetrading = true
        else
            continuetrading = false
            @warn "stop trading loop due to no reloading progress for $base"
        end
    end
    return continuetrading
end

"Returns the cumulated portfolio liquidity in USDT as (total, free, locked)"
function totalusdtliquidity(cache)
    cache.portfolio = isnothing(cache.portfolio) ? CryptoXch.portfolio!() : cache.portfolio
    usdtfree = sum(cache.portfolio.free .* cache.portfolio.usdtprice)
    usdtlocked = sum(cache.portfolio.locked .* cache.portfolio.usdtprice)
    usdttotal = usdtfree + usdtlocked
    return usdttotal, usdtfree, usdtlocked
end

"Returns the asset liquidity in USDT as (total, free, locked)"
function usdtliquidity(cache, base)
    cache.portfolio = isnothing(cache.portfolio) ? CryptoXch.portfolio!() : cache.portfolio
    df = findfirst(x -> x == base, cache.portfolio)
    if isnothing(df)
        @warn "no $base found in portfolio"
        return 0.0, 0.0, 0.0
    end
    free = df.free[begin] * df.usdtprice[begin]
    locked = df.locked[begin] * df.usdtprice[begin]
    return  free+locked, free, locked
end

"prints liquidity of `base` and USDT or of accumulated portfolio if `base === nothing`"
function reportliquidity(cache, base)
    if isnothing(base)
        total, free, locked = totalusdtliquidity(cache)
        @info "liquidity portfolio total free: $(round(free;digits=3)) USDT, locked: $(round(locked;digits=3)) USDT, total: $(round(total;digits=3)) USDT"
    elseif base == "usdt"
        @info "liquidity free: $(round(cache.usdtfree;digits=3)) USDT, locked: $(round(cache.usdtlocked;digits=3)) USDT, total: $(round((cache.usdtfree + cache.usdtlocked);digits=3)) USDT"
    else
        total, free, locked = usdtliquidity(cache, base)
        @info "liquidity free: $(round(cache.bd[base].assetfree;digits=3)) $(base) = $(round(free;digits=3)) USDT, locked: $(round(cache.bd[base].assetlocked;digits=3)) $(base) = $(round(locked;digits=3)) USDT -- USDT free: $(round(cache.usdtfree;digits=3)), locked: $(round(cache.usdtlocked;digits=3)) USDT"
    end
end

ordertime(cache, base)  = Ohlcv.dataframe(cache.bd[base].features.ohlcv)[cache.bd[base].ohlcvix, :opentime]

function closeorder!(cache, order, side, status)
    df = Ohlcv.dataframe(cache.bd[order.base].features.ohlcv)
    opentime = df[!, :opentime]
    timeix = cache.bd[order.base].ohlcvix
    # @info "close $side order #$(order.orderId) of $(order.origQty) $(order.base) (executed $(order.executedQty) $(order.base)) because $status at $(round(order.price;digits=3))USDT on ix:$(timeix) / $(EnvConfig.timestr(opentime[timeix]))"
    order = (;order..., closetime=ordertime(cache, order.base))
    if backtest(cache)
        if status == "FILLED"
            order = (;order..., status = status, executedQty=order.origQty)
            # order.executedQty = order.origQty
        end
    else  # no backtest
        xchorder = CryptoXch.getorder(order.base, order.orderId)
        if (status == "CANCELED") && (xchorder.status != "FILLED")
            xchorder = CryptoXch.cancelorder(order.base, order.orderId)
        end
        order = (;order...,
            price = xchorder["price"],
            origQty = xchorder["origQty"],
            executedQty = xchorder["executedQty"],
            status = xchorder["status"],
            timeInForce = xchorder["timeInForce"],
            type = xchorder["type"],
            side = xchorder["side"]
        )
        # TODO read order and fill executedQty and confirm FILLED (if not CANCELED) and read order fills (=transactions) to get the right fee
    end
    executedusdt = CryptoXch.ceilbase("usdt", order.executedQty * order.price)
    if (side == "BUY")
        orderusdt = CryptoXch.ceilbase("usdt", order.origQty * order.price)
        if order.executedQty > 0
            TradingStrategy.registerbuy!(cache.tradechances, timeix, order.price, order.orderId, cache.bd[order.base].features)
        end
        if !isapprox(cache.usdtlocked, orderusdt)
            if (cache.usdtlocked < orderusdt)
                msg = "closeorder! locked $(cache.usdtlocked) USDT insufficient to fill buy order #$(order.orderId) of $(order.origQty) $(order.base) (== $orderusdt USDT)"
                msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
                order = (;order..., message=msg);
                @warn msg
            end
        end
        cache.usdtlocked = cache.usdtlocked < orderusdt ? 0.0 : cache.usdtlocked - orderusdt
        cache.usdtfree += (orderusdt - executedusdt)
        cache.bd[order.base].assetfree += CryptoXch.floorbase(order.base, order.executedQty * (1 - tradingfee)) #! TODO minus fee or fee is deducted from BNB
    elseif (side == "SELL")
        if !isapprox(cache.bd[order.base].assetlocked, order.origQty)
            if cache.bd[order.base].assetlocked < order.origQty
                msg = "closeorder! locked $(cache.bd[order.base].assetlocked) $(order.base) insufficient to fill sell order #$(order.orderId) of $(order.origQty) $(order.base)"
                msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
                order = (;order..., message=msg);
                @warn msg
            end
        end
        cache.bd[order.base].assetlocked = cache.bd[order.base].assetlocked < order.origQty ? 0.0 : cache.bd[order.base].assetlocked - order.origQty
        cache.bd[order.base].assetfree += order.origQty - order.executedQty
        cache.usdtfree += CryptoXch.floorbase("usdt", executedusdt * (1 - tradingfee)) #! TODO minus fee or fee is deducted from BNB
        TradingStrategy.deletetradechanceoforder!(cache.tradechances, order.orderId)
    end
    cache.openorders = filter(row -> !(row.orderId == order.orderId), cache.openorders)
    totalusdt, _, _ = totalusdtliquidity(cache)

    msg = "closeorder! close $side order #$(order.orderId) of $(order.origQty) $(order.base) (executed $(order.executedQty) $(order.base)) because $status at $(round(order.price;digits=3))USDT on ix:$(timeix) / $(EnvConfig.timestr(opentime[timeix]))  new total USDT = $(round(totalusdt;digits=3))"
    msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
    order = (;order..., message=msg);
    @info msg
end

function buyqty(cache, base)
    usdttotal, usdtfree, _ = totalusdtliquidity(cache)
    usdtmin = 2 * CryptoXch.minimumquotevolume
    orderportfoliopercentage = 2.0 / 100
    usdtqty = min((usdttotal * orderportfoliopercentage), usdtfree)
    usdtqty = max(usdtqty, usdtmin)
    df = ohlcvdf(cache, base)
    baseqty = usdtqty / df.close[cache.bd[base].ohlcvix]
    baseqty = CryptoXch.floorbase(base, baseqty)
    return baseqty
end

function neworder!(cache, base, price, qty, side, tc)
    if cache.bd[base].symbolfilter.pricestep > 0.0
        price = price - (price % cache.bd[base].symbolfilter.pricestep)
    end
    qty = floor(qty, ; digits=cache.bd[base].symbolfilter.baseprecision)
    if cache.bd[base].symbolfilter.basestep > 0.0
        qty = qty - (qty % cache.bd[base].symbolfilter.basestep)
    end
    if backtest(cache)
        orderid = Dates.value(convert(Millisecond, Dates.now())) # unique id in backtest
        while !isnothing(TradingStrategy.tradechanceoforder(cache.tradechances, orderid))
            orderid = Dates.value(convert(Millisecond, Dates.now())) # unique id in backtest
        end
    else
        orderid = 0
    end
    order = (
        base = base,
        orderId = orderid, # unique id in backtest
        price = price,
        origQty = qty,
        executedQty = 0.0,
        status = "NEW",
        timeInForce = "GTC",
        type = "LIMIT",
        side = side,
        opentime = ordertime(cache, base),
        closetime = dummytime(),
        message = ""
    )
    # TODO check balances and whether it matches with fills of transactions including fees
    # reportliquidity(cache, base)
    # @info "open $(order.side) order #$(order.orderId) of $(order.origQty) $(order.base) because $(order.status) at $priceusdt USDT tc: $tc"
    if qty < cache.bd[base].symbolfilter.basemin
        msg = "neworder! $side order $qty $(order.base)=$qty $base is less than required minimum amount of $(cache.bd[base].symbolfilter.basemin) - order #$(order.orderId) is not executed"
        msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
        order = (;order..., status = "REJECTED", orderId=0, message=msg);
        @warn msg
        return
    end
    orderusdt = qty * price
    orderusdt = round(orderusdt, ; digits=cache.bd[base].symbolfilter.quoteprecision)
    if orderusdt < cache.bd[base].symbolfilter.quotemin
        msg = "neworder! $side order $qty $(order.base)=$orderusdt USDT is less than required minimum volume $(cache.bd[base].symbolfilter.quotemin) USDT - order #$(order.orderId) is not executed"
        msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
        order = (;order..., status = "REJECTED", orderId=0, message=msg);
        @warn msg
        return
    end
    if side == "BUY"
        if orderusdt > cache.usdtfree
            msg = "neworder! BUY order $qty $(order.base)=$orderusdt USDT has insufficient free coverage $(cache.usdtfree) USDT - order #$(order.orderId) is not executed"
            msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
            order = (;order..., status = "REJECTED", orderId=0, message=msg);
            @warn msg
        else
            if !backtest(cache)
                xchorder = CryptoXch.createbuyorder(base, price, qty)  #! check
                order = (
                    base = xchorder["base"],
                    orderId = xchorder["orderId"],
                    price = xchorder["price"],
                    origQty = xchorder["origQty"],
                    executedQty = xchorder["executedQty"],
                    status = xchorder["status"],
                    timeInForce = xchorder["timeInForce"],
                    type = xchorder["type"],
                    side = xchorder["side"],
                    opentime = xchorder["time"],
                    closetime = order.closetime,
                    message = order.message
                )
            end
            cache.usdtfree -= orderusdt
            cache.usdtlocked += orderusdt
            push!(cache.openorders, order)
            TradingStrategy.registerbuyorder!(cache.tradechances, orderid, tc)
        end
    elseif side == "SELL"
        if order.origQty > cache.bd[order.base].assetfree
            msg = "neworder! SELL order $(order.origQty) $(order.base) has insufficient free coverage $(cache.bd[order.base].assetfree) $(order.base) - order #$(order.orderId) is reduced"
            msg = length(order.message) > 0 ? msg = "$(order.message); $msg" : msg;
            @warn msg

            qty = CryptoXch.floorbase(base, cache.bd[order.base].assetfree)
            if !backtest(cache)
                xchorder = CryptoXch.createsellorder(base, price, qty)  #! check
                order = (
                    base = xchorder["base"],
                    orderId = xchorder["orderId"],
                    price = xchorder["price"],
                    origQty = xchorder["origQty"],
                    executedQty = xchorder["executedQty"],
                    status = xchorder["status"],
                    timeInForce = xchorder["timeInForce"],
                    type = xchorder["type"],
                    side = xchorder["side"],
                    opentime = xchorder["time"],
                    closetime = order.closetime,
                    message = order.message
                )
            else
                order = (;order..., origQty=qty)
            end
        end
        cache.bd[order.base].assetfree  -= order.origQty
        cache.bd[order.base].assetlocked  += order.origQty
        push!(cache.openorders, order)
        TradingStrategy.registersellorder!(cache.tradechances, orderid, tc)
    end
end

significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.2

"""
- selects the trades to be executed and places orders
- corrects orders that are not yet executed
- cancels orders that are not yet executed and where sufficient gain seems unlikely

to be considered:
- portfolio that has to be tracked to sell
- new buy chances but consider the already traded bases and multiple buy chances of the same base
- log how many chances are offered on average to determine an appropriate trade granularity
    - start with 2% granuality of portfolio value per trade
    - start with max 1 active trade per base
    - start accepting all buy chances as long as USDT is available

- check open sell orders
    - if (partially) closed, update assets to free locked assets
    - is sell order completely sold? if so, delete trade chance
    - if sell order still (partially) open, then update sell price
        - if order sell price not within 0.1% of chance sell price and > minimal volume then cancel sell order and issue new order
- check open buy orders if (partially) closed
    - if so, then register buy in trade chance and issue sell order
- is there a buy(base) chance? (max 1 buy order per base)
    - is there an existing open buy(base) order?
        - is the buy price of the existing buy order within 0.1% of the buy chance?
            - if so, do nothing
            - if not and > minimal volume , cancel existing order and issue new order
        - if not issue a new buy order
    - adapt buy chance with orderid
"""
function trade!(cache)
    #TODO this is the function that needs to be adapted to the new strategy
    """
    - need to introduce a sequence concept that varies the gap and amount according to free USDT, open orders, median order closure time, number of bases to be invested
      - a function as asset allocator that
        - returns USDT amount and minute gap and
        - receives as input per tracked base the tracked regression length, median investment length, free USDT, open orders
      - here the openorders have to be checked for orders for that base and the time gap to be considered

    """
    for order in copy.(eachrow(cache.openorders))
        tc = TradingStrategy.tradechanceoforder(cache.tradechances, order.orderId)
        if isnothing(tc)
            @warn "no tradechance found for $(order.side) order #$(order.orderId)" tc maxlog=10
            continue
        end
        if order.side == "SELL"
            # @info "trade! close order $tc"

            df = ohlcvdf(cache, order.base)
            if df.high[cache.bd[order.base].ohlcvix] > order.price
                closeorder!(cache, order, "SELL", "FILLED")
            elseif significantsellpricechange(tc, order.price)
                # set new limit price
                closeorder!(cache, order, "SELL", "CANCELED")
                neworder!(cache, order.base, tc.sellprice, order.origQty, "SELL", tc)
                #? order.origQty remains unchanged?
            end
        end
        if order.side == "BUY"
            df = ohlcvdf(cache, order.base)
            if df.low[cache.bd[order.base].ohlcvix] < order.price
                closeorder!(cache, order, "BUY", "FILLED")
                # buy order is closed, now issue a sell order
                qty = cache.bd[order.base].assetfree
                neworder!(cache, order.base, tc.sellprice, qty, "SELL", tc)
                # getorder and check that it is filled
            else
                newtc = TradingStrategy.tradechanceofbase(cache.tradechances, order.base)
                if !isnothing(newtc)
                    # buy price not reached but other new buy chance available - hence, cancel old buy order and use new chance
                    closeorder!(cache, order, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order
                    neworder!(cache, order.base, newtc.buyprice, buyqty(cache, order.base), "BUY", newtc)
                    TradingStrategy.deletenewbuychanceofbase!(cache.tradechances, order.base)
                elseif significantbuypricechange(tc, order.price)
                    closeorder!(cache, order, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order with adapted price
                    neworder!(cache, order.base, tc.buyprice, buyqty(cache, order.base), "BUY", tc)
                end
            end
        end
    end
    # check remaining new base buy chances
    for base in TradingStrategy.baseswithnewtradechances(cache.tradechances)
        usdttotal, _, _ = usdtliquidity(cache, base)
        if usdttotal <= CryptoXch.minimumquotevolume
            # no new order if asset is already in possession
            tc = TradingStrategy.tradechanceofbase(cache.tradechances, order.base)
            neworder!(cache, base, tc.buyprice, buyqty(cache, base), "BUY", tc)
            TradingStrategy.deletenewbuychanceofbase!(cache.tradechances, base)
        end
    end
end

# function performancecheck()

# end

function coretradeloop(cache::Cache)
    global count = 1
    continuetrading = false
    for base in keys(cache.bd)
        continuetrading = appendmostrecent!(cache, base) # add either next backtest data or wait until new realtime data
        if continuetrading
            # assess trade chances
            cache.tradechances = TradingStrategy.assesstrades!(cache.tradechances, cache.bd[base].features, cache.bd[base].ohlcvix)
        else
            assetrefresh = true
            break
        end
    end
    if continuetrading
        trade!(cache) # execute trades on basis of assessed trade chances
    end
    return continuetrading
end

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop(cache)
    # TODO add hooks to enable coupling to the cockpit visualization
    logger = SimpleLogger(cache.messagelog)
    defaultlogger = global_logger(logger)
    @info "backtest period=$(cache.backtestperiod) enddt=$(cache.backtestenddt)"
    profileinit = false
    # Profile.clear()
    continuetrading = true
    while continuetrading
        preparetradecache!(cache) # determine the bases to be considered for trading and prepare their initial data, i.e. open orders and recent required minutes ohlcv
        assetrefresh = false
        refreshtimestamp = Dates.now(Dates.UTC)
        println("starting trading core loop")
        while !assetrefresh && continuetrading
            if profileinit
                @profile continuetrading = coretradeloop(cache)
            else
                continuetrading = coretradeloop(cache)
                # profileinit = true
            end
        end
        # if !backtest && (Dates.now(Dates.UTC)-refreshtimestamp > Dates.Minute(12*60))
        #     # TODO the read ohlcv data shall be from time to time appended to the historic data and in cache data to be shortend (= cleanup)
        # end
        reportliquidity(cache, nothing)
        total, free, locked = totalusdtliquidity(cache)
        println("liquidity portfolio total free: $(round(free;digits=3)) USDT, locked: $(round(locked;digits=3)) USDT, total: $(round(total;digits=3)) USDT")
    end
    writetradelogs(cache)
    println("traded $(keys(cache.bd))")
    println("finished trading core loop")
    # Profile.print()
    global_logger(defaultlogger)
    close(cache.messagelog)
end


end  # module
