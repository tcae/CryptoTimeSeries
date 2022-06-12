# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
## problem statement

This module shall automatically follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.


All history data will be collected but a fixed subset **`historysubset`** will be used for training, evaluation and test. Such data is OHLCV data of a fixed set of crypto currencies that have proven to show sufficient liquidity.
"""
module Trade

using Dates, DataFrames, JSON, Profile
using EnvConfig, Ohlcv, Classify, CryptoXch, Assets, Features

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

minimumdayusdtvolume = 10000000  # per day = 6944 per minute

mutable struct BaseInfo
    assetfree
    assetlocked
    currentix
    features
end

"""
trade cache contains all required data to support the tarde loop
"""
mutable struct Cache
    backtestchunk  # 0 if no backtest or in case of backtest nbr of minutes to process until next reload
    usdtfree
    usdtlocked
    classify!  # function to classsify trade chances
    bd  # dict[base] of BaseInfo
    tradechances  # ::Classify.TradeChances001
    openorders  # DataFrame with cols: base, orderId, price, origQty, executedQty, status, timeInForce, type, side, logtime
    orderlog  # DataFrame with cols like openorders
    # transactionlog  # DataFrame
    Cache(backtestchunk, free, locked) = new(backtestchunk, free, locked, Classify.traderules001!, Dict(), nothing, CryptoXch.orderdataframe([], Dates.now(UTC)), CryptoXch.orderdataframe([], Dates.now(UTC)))
end

makerfee = 0.075 / 100
takerfee = 0.075 / 100
tradingfee = max(makerfee, takerfee)

ohlcvdf(cache, base) = Ohlcv.dataframe(Features.ohlcv(cache.bd[base].features))
backtest(cache) = cache.backtestchunk > 0

function freelocked(portfoliodf, base)
    portfoliodf = portfoliodf[portfoliodf.base .== base, :]
    if size(portfoliodf, 1) > 0
        return portfoliodf.free[begin], portfoliodf.locked[begin]
    else
        return 0.0, 0.0
    end
end

function preparetradecache(backtestchunk)
    # TODO read not only assets but also open orders and assign them to cache to be considered in the trade loop
    usdtdf = CryptoXch.getUSDTmarket()
    pdf = CryptoXch.portfolio(usdtdf)
    free, locked = freelocked(pdf, "usdt")
    cache = Cache(backtestchunk, free, locked)  # no need to cache focusbases because they will be implicitly stored via keys(bd)
    usdtdf = usdtdf[usdtdf.quotevolume24h .> 10000000, :]
    focusbases = usdtdf.base
    if backtest(cache)
        if EnvConfig.configmode == EnvConfig.test
            initialperiod = Dates.Minute(Classify.requiredminutes + 4*60)
            # 3*requiredminutes for 1*  to calc features, 1* to build trade history, 1* to check Trade loop
        else
            initialperiod = Dates.Month(6)
            # initialperiod = Dates.Year(4)
        end
        enddt = DateTime("2022-04-02T01:00:00")  # fix to get reproducible results
    else
        initialperiod = Dates.Minute(Classify.requiredminutes)
        # 2* because 1* for collecting breakouts to compare and 1* to build up std of largest window
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)  # don't use ceil because that includes a potentially partial running minute
    end
    startdt = enddt - initialperiod
    @assert startdt < enddt
    startdt = floor(startdt, Dates.Minute)
    for base in focusbases
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.read!(ohlcv)
        origlen = size(ohlcv.df, 1)
        ohlcv.df = ohlcv.df[enddt .>= ohlcv.df.opentime .>= startdt, :]
        println("cutting $base ohlcv from $origlen to $(size(ohlcv.df, 1)) minutes ($(EnvConfig.timestr(startdt)) - $(EnvConfig.timestr(enddt)))")
        # CryptoXch.cryptoupdate!(ohlcv, startdt, enddt)  # not required because loadassets will already update
        if size(Ohlcv.dataframe(ohlcv), 1) < Classify.requiredminutes
            @warn "insufficient ohlcv data returned for" base receivedminutes=size(Ohlcv.dataframe(ohlcv), 1) requiredminutes=Classify.requiredminutes
            continue
        end
        currentix = backtest(cache) ? Classify.requiredminutes : lastindex(Ohlcv.dataframe(ohlcv), 1)
        free, locked = freelocked(pdf, base)
        cache.bd[base] = BaseInfo(free, locked, currentix, Features.Features002(ohlcv, 1, Classify.requiredminutes+cache.backtestchunk))
        # @info "preparetradecache ohlcv.df=$(size(ohlcv.df, 1))  features: firstix=$(cache.bd[base].features.firstix) lastix=$(cache.bd[base].features.lastix)"
    end
    # no need to cache assets because they are implicitly stored via keys(bd)
    reportliquidity(cache, nothing)
    reportliquidity(cache, "usdt")
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

"""
append most recent ohlcv data as well as corresponding features
returns `true` if successful appended else `false`
"""
function appendmostrecent!(cache::Cache, base)
    global count = 0
    continuetrading = false
    df = ohlcvdf(cache, base)
    if backtest(cache)
        cache.bd[base].currentix += 1
        if cache.bd[base].currentix > cache.bd[base].features.lastix
            # calculate next chunk of features
            lastix = min(cache.bd[base].currentix + cache.backtestchunk - 1, lastindex(df, 1))
            firstix =  max(min(cache.bd[base].currentix, lastix) - Classify.requiredminutes, firstindex(df, 1))
            cache.bd[base].features.update(cache.bd[base].features, firstix, lastix)
            # @info "appendmostrecent! ohlcv.df=$(size(cache.bd[base].features.ohlcv.df, 1))  features: firstix=$(cache.bd[base].features.firstix) lastix=$(cache.bd[base].features.lastix)"
            count+= 1
        end
        if cache.bd[base].currentix <= cache.bd[base].features.lastix
            continuetrading = true
            if (cache.bd[base].currentix % 1000) == 0
                @info "continue at ix=$(cache.bd[base].currentix) < size=$(size(ohlcvdf(cache, base), 1)) backtestchunk=$(cache.backtestchunk) continue=$continuetrading"
            end
        else
            continuetrading = false
            @info "stop trading loop due to backtest ohlcv for $base exhausted - count = $count"
        end
    else  # production
        lastdt = df.opentime[end]
        enddt = sleepuntilnextminute(lastdt)
        # startdt = enddt - Dates.Minute(Features.requiredminutes)
        startdt = df.opentime[begin]  # stay with start until tradeloop cleanup
        currentix = lastindex(df, 1)
        CryptoXch.cryptoupdate!(Features.ohlcv(cache.bd[base].features), startdt, enddt)
        df = ohlcvdf(cache, base)
        println("extended from $lastdt to $enddt -> check df: $(df.opentime[begin]) - $(df.opentime[end]) size=$(size(df,1))")
        # ! TODO implement error handling
        @assert lastdt == df.opentime[currentix]  # make sure begin wasn't cut
        cache.bd[base].currentix = lastindex(df, 1)
        cache.bd[base].features.update(cache.bd[base].features, cache.bd[base].currentix - Classify.requiredminutes, cache.bd[base].currentix)
        if lastdt < df.opentime[end]
            continuetrading = true
        else
            continuetrading = false
            @info "stop trading loop due to no reloading progress for $base"
        end
    end
    return continuetrading
end

"Returns the cumulated portfolio liquidity in USDT as (total, free, locked)"
function totalusdtliquidity(cache)
    usdtfree = cache.usdtfree
    usdtlocked = cache.usdtlocked
    for (b, binfo) in cache.bd
        df = Ohlcv.dataframe(Features.ohlcv(binfo.features))
        currentix = binfo.currentix > size(df, 1) ? size(df, 1) : binfo.currentix
        usdtfree += binfo.assetfree * df.close[currentix]
        usdtlocked += binfo.assetlocked * df.close[currentix]
    end
    usdttotal = usdtfree + usdtlocked
    return usdttotal, usdtfree, usdtlocked
end

"Returns the asset liquidity in USDT as (total, free, locked)"
function usdtliquidity(cache, base)
    if base == "usdt"
        free = cache.usdtfree
        locked = cache.usdtlocked
    else
        df = Ohlcv.dataframe(Features.ohlcv(cache.bd[base].features))
        currentix = cache.bd[base].currentix > size(df, 1) ? size(df, 1) : cache.bd[base].currentix
        lastprice = df.close[currentix]
        free = cache.bd[base].assetfree * lastprice
        locked = cache.bd[base].assetlocked * lastprice
    end
    total = free + locked
    return total, free, locked
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

function closeorder!(cache, order, side, status)
    if backtest(cache)
        if status == "FILLED"
            order = (;order..., executedQty=order.origQty)
            # order.executedQty = order.origQty
        end
    else
        order = (;order..., executedQty=order.origQty)
        # order.executedQty = order.origQty
        # TODO implement production
        # TODO read order and fill executedQty and confirm FILLED (if not CANCELED) and read order fills (=transactions) to get the right fee
    end
    df = Ohlcv.dataframe(cache.bd[order.base].features.ohlcv)
    opentime = df[!, :opentime]
    timeix = cache.bd[order.base].currentix
    # @info "close $side order #$(order.orderId) of $(order.origQty) $(order.base) (executed $(order.executedQty) $(order.base)) because $status at $(round(order.price;digits=3))USDT on ix:$(timeix) / $(EnvConfig.timestr(opentime[timeix]))"
    # reportliquidity(cache, order.base)
    order = (;order..., status = status)
    # order.status = status
    executedusdt = CryptoXch.floorbase("usdt", order.executedQty * order.price)
    if (side == "BUY")
        orderusdt = CryptoXch.floorbase("usdt", order.origQty * order.price)
        if order.executedQty > 0
            Classify.registerbuy!(cache.tradechances, timeix, order.price, order.orderId, cache.bd[order.base].features)
        end
        if !isapprox(cache.usdtlocked, orderusdt)
            if (cache.usdtlocked < orderusdt)
                @warn "locked $(cache.usdtlocked) USDT insufficient to fill buy order #$(order.orderId) of $(order.origQty) $(order.base) (== $orderusdt USDT)"
            end
        end
        cache.usdtlocked = cache.usdtlocked < orderusdt ? 0.0 : cache.usdtlocked - orderusdt
        cache.usdtfree += (orderusdt - executedusdt)
        cache.bd[order.base].assetfree += CryptoXch.floorbase(order.base, order.executedQty * (1 - tradingfee)) #! TODO minus fee or fee is deducted from BNB
    elseif (side == "SELL")
        if !isapprox(cache.bd[order.base].assetlocked, order.origQty)
            if cache.bd[order.base].assetlocked < order.origQty
                @warn "locked $(cache.bd[order.base].assetlocked) $(order.base) insufficient to fill sell order #$(order.orderId) of $(order.origQty) $(order.base)"
            end
        end
        cache.bd[order.base].assetlocked = cache.bd[order.base].assetlocked < order.origQty ? 0.0 : cache.bd[order.base].assetlocked - order.origQty
        cache.bd[order.base].assetfree += order.origQty - order.executedQty
        cache.usdtfree += CryptoXch.floorbase("usdt", executedusdt * (1 - tradingfee)) #! TODO minus fee or fee is deducted from BNB
    end
    cache.openorders = filter(row -> !(row.orderId == order.orderId), cache.openorders)
    push!(cache.orderlog, order)
    Classify.deletetradechanceoforder!(cache.tradechances, order.orderId)
    totalusdt, _, _ = totalusdtliquidity(cache)
    @info "close $side order #$(order.orderId) of $(order.origQty) $(order.base) (executed $(order.executedQty) $(order.base)) because $status at $(round(order.price;digits=3))USDT on ix:$(timeix) / $(EnvConfig.timestr(opentime[timeix]))  new total USDT = $(round(totalusdt;digits=3))"
end

function buyqty(cache, base)
    usdttotal, usdtfree, _ = totalusdtliquidity(cache)
    usdtmin = 20.0
    orderportfoliopercentage = 2.0 / 100
    usdtqty = max((usdttotal * orderportfoliopercentage), usdtmin)
    df = ohlcvdf(cache, base)
    baseqty = usdtqty / df.close[cache.bd[base].currentix]
    return baseqty
end

function neworder!(cache, base, price, qty, side, status, tc)
    qty = CryptoXch.floorbase(base, qty)  # see also minimum order granularity of xchange
    if backtest(cache)
        orderid = Dates.value(convert(Millisecond, Dates.now())) # unique id in backtest
        while !isnothing(Classify.tradechanceoforder(cache.tradechances, orderid))
            orderid = Dates.value(convert(Millisecond, Dates.now())) # unique id in backtest
        end
        order = (
            base = base,
            orderId = orderid, # unique id in backtest
            price = price,
            origQty = qty,
            executedQty = 0.0,
            status = status,
            timeInForce = "GTC",
            type = "LIMIT",
            side = side,
            logtime = Dates.now(UTC)
        )
    else
        # TODO implement production
        # create new one with corrected price
        # use create order response to fill order record
        # check balances and whether it matches with fills of transactions including fees
    end
    # reportliquidity(cache, base)
    priceusdt = CryptoXch.roundbase("usdt", order.price)
    # @info "open $(order.side) order #$(order.orderId) of $(order.origQty) $(order.base) because $(order.status) at $priceusdt USDT tc: $tc"
    if side == "BUY"
        orderusdt = CryptoXch.roundbase("usdt", order.origQty * order.price)
        if orderusdt > cache.usdtfree
            @warn "BUY order $(order.origQty) $(order.base)=$orderusdt USDT has insufficient free coverage $(cache.usdtfree) USDT - order #$(order.orderId) is not executed"
        else
            cache.usdtfree -= orderusdt
            cache.usdtlocked += orderusdt
            push!(cache.openorders, order)
            Classify.registerbuyorder!(cache.tradechances, orderid, tc)
        end
    elseif side == "SELL"
        if order.origQty > cache.bd[order.base].assetfree
            @warn "SELL order $(order.origQty) $(order.base) has insufficient free coverage $(cache.bd[order.base].assetfree) $(order.base) - order #$(order.orderId) is not executed"
        else
            cache.bd[order.base].assetfree  -= order.origQty
            cache.bd[order.base].assetlocked  += order.origQty
            push!(cache.openorders, order)
            Classify.registersellorder!(cache.tradechances, orderid, tc)
        end
    end
end

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
    # TODO update TradeLog -> append to csv
    for order in copy.(eachrow(cache.openorders))
        tc = Classify.tradechanceoforder(cache.tradechances, order.orderId)
        if isnothing(tc)
            @warn "no tradechance found for $(order.side) order #$(order.orderId)" tc
            continue
        end
        if order.side == "SELL"
            df = ohlcvdf(cache, order.base)
            if df.high[cache.bd[order.base].currentix] > order.price
                closeorder!(cache, order, "SELL", "FILLED")
            elseif Classify.significantsellpricechange(tc, order.price)  # including emergency sells
                closeorder!(cache, order, "SELL", "CANCELED")
                neworder!(cache, order.base, tc.sellprice, order.origQty, "SELL", "NEW", tc)
                #? order.origQty remains unchanged?
            end
        end
        if order.side == "BUY"
            df = ohlcvdf(cache, order.base)
            if df.low[cache.bd[order.base].currentix] < order.price
                closeorder!(cache, order, "BUY", "FILLED")
                # buy order is closed, now issue a sell order
                qty = cache.bd[order.base].assetfree
                neworder!(cache, order.base, tc.sellprice, qty, "SELL", "NEW", tc)
                # getorder and check that it is filled
            else
                if order.base in keys(cache.tradechances.basedict)
                    # buy price not reached but other new buy chance available - hence, cancel old buy order and use new chance
                    newtc = cache.tradechances.basedict[order.base]
                    closeorder!(cache, order, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order
                    neworder!(cache, order.base, newtc.buyprice, buyqty(cache, order.base), "BUY", "NEW", newtc)
                    Classify.deletenewbuychanceofbase!(cache.tradechances, order.base)
                elseif tc.probability < 0.5
                    closeorder!(cache, order, "BUY", "CANCELED")
                    # buy order is closed because of low chance of buy success
                elseif Classify.significantbuypricechange(tc, order.price)
                    closeorder!(cache, order, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order with adapted price
                    neworder!(cache, order.base, tc.buyprice, buyqty(cache, order.base), "BUY", "NEW", tc)
                end
            end
        end
    end
    # check remaining new base buy chances
    for (base, tc) in cache.tradechances.basedict
        usdttotal, _, _ = usdtliquidity(cache, base)
        if usdttotal < CryptoXch.minimumquotevolume
            # no new order if asset is already in possession
            neworder!(cache, base, tc.buyprice, buyqty(cache, base), "BUY", "NEW", tc)
            Classify.deletenewbuychanceofbase!(cache.tradechances, base)
        end
    end
end

# function performancecheck()

# end

function coretradeloop(cache)
    global count = 1
    continuetrading = false
    for base in keys(cache.bd)
        continuetrading = appendmostrecent!(cache, base)
        if continuetrading
            cache.tradechances = cache.classify!(cache.tradechances, cache.bd[base].features, cache.bd[base].currentix)
        else
            assetrefresh = true
            break
        end
    end
    if continuetrading
        trade!(cache)
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
function tradeloop(backtestchunk)
    # TODO add hooks to enable coupling to the cockpit visualization
    profileinit = false
    # Profile.clear()
    continuetrading = true
    while continuetrading
        cache = preparetradecache(backtestchunk)
        assetrefresh = false
        refreshtimestamp = Dates.now(Dates.UTC)
        println("strting trading core loop")
        while !assetrefresh && continuetrading
            if profileinit
                @profile continuetrading = coretradeloop(cache)
            else
                continuetrading = coretradeloop(cache)
                # profileinit = true
            end
        end
        # if !backtest && (Dates.now(Dates.UTC)-refreshtimestamp > Dates.Minute(12*60))
        #     # TODO the read ohlcv data shall be from time to time appended to the historic data
        # end
        reportliquidity(cache, nothing)
    end
    println("finished trading core loop")
    # Profile.print()
end

end  # module
