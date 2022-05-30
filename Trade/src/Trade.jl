# using Pkg;
# Pkg.add(["Dates", "DataFrames"])

"""
## problem statement

This module shall automatically follow the most profitable crypto currecncy at Binance, buy when an uptrend starts and sell when it ends.


All history data will be collected but a fixed subset **`historysubset`** will be used for training, evaluation and test. Such data is OHLCV data of a fixed set of crypto currencies that have proven to show sufficient liquidity.
"""
module Trade

using Dates, DataFrames, JSON
using EnvConfig, Ohlcv, Classify, CryptoXch, Assets, Features

@enum OrderType buylongmarket buylonglimit selllongmarket selllonglimit

# cancelled by trader, rejected by exchange, order change = cancelled+new order opened
@enum OrderStatus opened cancelled rejected closed

minimumdayusdtvolume = 10000000  # per day = 6944 per minute

mutable struct BaseInfo
    assetfree
    assetlocked
    lastix
    features
end

"""
trade cache contains all required data to support the tarde loop
"""
mutable struct Cache
    backtest::Bool
    usdtfree
    usdtlocked
    classify!  # function to classsify trade chances
    bd  # dict[base] of BaseInfo
    tradechances  # ::Classify.TradeChances001
    openorders  # DataFrame with cols: base, orderId, price, origQty, executedQty, status, timeInForce, type, side, logtime
    orderlog  # DataFrame with cols like openorders
    # transactionlog  # DataFrame
    Cache(backtest, free, locked) = new(backtest, free, locked, Classify.traderules001!, Dict(), nothing, CryptoXch.orderdataframe([], Dates.now(UTC)), CryptoXch.orderdataframe([], Dates.now(UTC)))
end

makerfee = 0.075 / 100
takerfee = 0.075 / 100
tradingfee = max(makerfee, takerfee)

ohlcvdf(cache, base) = Ohlcv.dataframe(Features.ohlcv(cache.bd[base].features))

function freelocked(portfoliodf, base)
    portfoliodf = portfoliodf[portfoliodf.base .== base, :]
    if size(portfoliodf, 1) > 0
        return portfoliodf.free[begin], portfoliodf.locked[begin]
    else
        return 0.0, 0.0
    end
end

function preparetradecache(backtest)
    # TODO read not only assets but also open orders and assign them to cache to be considered in the trade loop
    usdtdf = CryptoXch.getUSDTmarket()
    pdf = CryptoXch.portfolio(usdtdf)
    usdtdf = usdtdf[usdtdf.quotevolume24h .> 10000000, :]
    focusbases = usdtdf.base
    if backtest
        if EnvConfig.configmode == EnvConfig.test
            initialperiod = Dates.Minute(100 + Features.requiredminutes)
        else
            initialperiod = Dates.Year(4)
        end
        enddt = DateTime("2022-04-02T01:00:00")  # fix to get reproducible results
    else
        initialperiod = Dates.Minute(Features.requiredminutes)
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)  # don't use ceil because that includes a potentially partial running minute
    end
    startdt = enddt - initialperiod
    @assert startdt < enddt
    startdt = floor(startdt, Dates.Minute)
    free, locked = freelocked(pdf, "usdt")
    cache = Cache(backtest, free, locked)  # no need to cache focusbases because they will be implicitly stored via keys(bd)
    for base in focusbases
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.read!(ohlcv)
        origlen = size(ohlcv.df, 1)
        ohlcv.df = ohlcv.df[enddt .>= ohlcv.df.opentime .>= startdt, :]
        println("cutting ohlcv from $origlen to $(size(ohlcv.df)) minutes")
        # CryptoXch.cryptoupdate!(ohlcv, startdt, enddt)  # not required because loadassets will already update
        if size(Ohlcv.dataframe(ohlcv), 1) < Features.requiredminutes
            @warn "insufficient ohlcv data returned for" base receivedminutes=size(Ohlcv.dataframe(ohlcv), 1) requiredminutes=Features.requiredminutes
            continue
        end
        lastix = backtest ? Features.requiredminutes : size(Ohlcv.dataframe(ohlcv), 1)
        free, locked = freelocked(pdf, base)
        cache.bd[base] = BaseInfo(free, locked, lastix, Features.Features002(ohlcv))
    end
    # no need to cache assets because they are implicitly stored via keys(bd)
    return cache
end

"""
append most recent ohlcv data as well as corresponding features
returns `true` if successful appended else `false`
"""
function appendmostrecent!(cache::Cache, base)
    if cache.backtest
        cache.bd[base].lastix += 1
        return cache.bd[base].lastix <= size(ohlcvdf(cache, base), 1)
    else  # production
        df = ohlcvdf(cache, base)
        lastdt = df.opentime[end]
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
        # startdt = enddt - Dates.Minute(Features.requiredminutes)
        startdt = df.opentime[begin]  # stay with start to prevent invalidating extremeix
        lastix = size(df, 1)
        CryptoXch.cryptoupdate!(Features.ohlcv(cache.bd[base].features), startdt, enddt)
        df = ohlcvdf(cache, base)
        println("extended from $lastdt to $enddt -> check df: $(df.opentime[begin]) - $(df.opentime[end]) size=$(size(df,1))")
        # ! TODO impleement error handling
        @assert lastdt == df.opentime[lastix]  # make sure begin wasn't cut
        cache.bd[base].lastix = size(df, 1)
        cache.bd[base].features.update(cache.bd[base].features)
        return cache.bd[base].lastix  > lastix
    end
end

significantsellpricechange(tc, orderprice) = abs(tc.sellprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.1
significantbuypricechange(tc, orderprice) = abs(tc.buyprice - orderprice) / abs(tc.sellprice - tc.buyprice) > 0.1

"Returns the cumulated portfolio liquidity in USDT as (total, free, locked)"
function totalliquidity(cache)
    usdtfree = cache.usdtfree
    usdtlocked = cache.usdtlocked
    for (b, binfo) in cache.bd
        df = Ohlcv.dataframe(Features.ohlcv(binfo.features))
        usdtfree += binfo.assetfree * df.close[binfo.lastix]
        usdtlocked += binfo.assetlocked * df.close[binfo.lastix]
    end
    usdttotal = usdtfree + usdtlocked
    return usdttotal, usdtfree, usdtlocked
end

"Returns the asset liquidity in USDT as (total, free, locked)"
function assetliquidity(cache, base)
    if base == "usdt"
        free = cache.usdtfree
        locked = cache.usdtlocked
    else
        df = Ohlcv.dataframe(Features.ohlcv(cache.bd[base].features))
        lastprice = df.close[cache.bd[base].lastix]
        free = cache.bd[base].assetfree * lastprice
        locked = cache.bd[base].assetlocked * lastprice
    end
    total = free + locked
    return total, free, locked
end

function reportliquidity(cache, base)
    usdttotal, _, _ = totalliquidity(cache)
    @info "liquidity: $(cache.bd[base].assetlocked) $(base), $(cache.usdtfree + cache.usdtlocked) USDT, portfolio: $usdttotal USDT"
end

function closeorder!(cache, order, orderix, side, status)
    if cache.backtest
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
    deleteat!(cache.openorders, orderix)
    order = (;order..., status = status)
    # order.status = status
    if order.executedQty > 0.0
        if (side == "BUY")
            if (cache.usdtlocked < order.executedQty * order.price)
                @warn " locked $(cache.usdtlocked) USDT insufficient to fill buy order $(order.executedQty * order.price) $(order.base)"
                cache.usdtlocked = 0.0
            else
                cache.usdtlocked -= order.executedQty * order.price
            end
            cache.bd[order.base].assetfree += order.executedQty * (1 - tradingfee) #! TODO minus fee or fee is deducted from BNB
        elseif (side == "SELL")
            if cache.bd[order.base].assetlocked < order.executedQty
                @warn " locked $(cache.bd[order.base].assetlocked) $(order.base) insufficient to fill sell order $(order.executedQty) $(order.base)"
                cache.bd[order.base].assetlocked = 0.0
            else
                cache.bd[order.base].assetlocked -= order.executedQty
            end
            cache.usdtfree += order.executedQty * order.price * (1 - tradingfee) #! TODO minus fee or fee is deducted from BNB
        end
    end
    @info "close $side order $(order.executedQty) $(order.base) because $status at $(round(order.price;digits=3))USDT "
    push!(cache.orderlog, order)
    Classify.deletetradechanceoforder!(cache.tradechances, order.orderId)
end

function buyqty(cache, base)
    usdttotal, usdtfree, _ = totalliquidity(cache)
    usdtmin = 20.0
    orderportfoliopercentage = 2.0 / 100
    usdtqty = max((usdttotal * orderportfoliopercentage), usdtmin)
    df = ohlcvdf(cache, base)
    baseqty = usdtqty / df.close[cache.bd[base].lastix]
    return baseqty
end

function neworder!(cache, base, price, qty, side, status, tc)
    qty = round(qty; digits=3)  # see also minimum order granularity of xchange
    if cache.backtest
        orderid = Dates.value(convert(Millisecond, Dates.now())) # unique id in backtest
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
    @info "open $(order.side) order $(order.origQty) $(order.base) because $(order.status) at $(round(order.price;digits=3))USDT "
    push!(cache.openorders, order)
    if side == "BUY"
        if (order.origQty * order.price) > cache.usdtfree
            @warn "BUY order $(order.origQty) $(order.base)=$(round(order.origQty * order.price;digits=3)) USDT has insufficient free coverage $(cache.usdtfree) USDT - order is not executed"
        else
            Classify.registerbuyorder!(cache.tradechances, orderid, tc)
        end
    elseif side == "SELL"
        if order.origQty > cache.bd[order.base].assetfree
            @warn "SELL order $(order.origQty) $(order.base) has insufficient free coverage $(cache.bd[order.base].assetfree) $(order.base) - order is not executed"
        else
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
    println("$(length(cache.tradechances)) trade chances")
    # TODO update TradeLog -> append to csv
    for (orderix, order) in enumerate(copy.(eachrow(cache.openorders)))
        tc = Classify.tradechanceoforder(cache.tradechances, order.orderId)
        if isnothing(tc)
            @warn "no tradechance found for order $(order.orderId)" tc
            continue
        end
        if order.side == "SELL"
            df = ohlcvdf(cache, order.base)
            if df.high[cache.bd[order.base].lastix] > order.price
                closeorder!(cache, order, orderix, "SELL", "FILLED")
            elseif significantsellpricechange(tc, order.price)  # including emergency sells
                closeorder!(cache, order, orderix, "SELL", "CANCELED")
                # buy order is closed, now issue a sell order
                neworder!(cache, order.base, tc.sellprice, order.origQty, "SELL", "NEW", tc)
                #? order.origQty remains unchanged?
            end
        end
        if order.side == "BUY"
            df = ohlcvdf(cache, order.base)
            if df.low[cache.bd[order.base].lastix] < order.price
                closeorder!(cache, order, orderix, "BUY", "FILLED")
                # buy order is closed, now issue a sell order
                qty = cache.bd[order.base].assetfree
                neworder!(cache, order.base, tc.sellprice, qty, "SELL", "NEW", tc)
                # getorder and check that it is filled
            else
                if order.base in cache.tradechances.basedict
                    newtc = cache.tradechances.basedict[order.base]
                    closeorder!(cache, order, orderix, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order
                    neworder!(cache, order.base, tc.buyprice, buyqty(cache, order.base), "BUY", "NEW", tc)
                    Classify.deletenewbuychanceofbase!(cache.tradechances, order.base)
                elseif significantbuypricechange(tc, order.price) || (tc.probability < 0.5)
                    closeorder!(cache, order, orderix, "BUY", "CANCELED")
                    # buy order is closed, now issue a new buy order
                    neworder!(cache, order.base, tc.buyprice, buyqty(cache, order.base), "BUY", "NEW", tc)
                end
            end
        end
    end
    # check remaining new base buy chances
    for (base, tc) in cache.tradechances.basedict
        usdttotal, _, _ = assetliquidity(cache, base)
        if usdttotal < CryptoXch.minimumquotevolume
            # no new order if asset is already in possession
            neworder!(cache, base, tc.buyprice, buyqty(cache, base), "BUY", "NEW", tc)
            Classify.deletenewbuychanceofbase!(cache.tradechances, base)
        end
    end
end

# function performancecheck()

# end

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop(backtest=true)
    # TODO add hooks to enable coupling to the cockpit visualization
    continuetrading = true
    while continuetrading
        cache = preparetradecache(backtest)
        noassetrefresh = true
        refreshtimestamp = Dates.now(Dates.UTC)
        while noassetrefresh
            for base in keys(cache.bd)
                continuetrading = appendmostrecent!(cache, base)
                if continuetrading
                    cache.tradechances = cache.classify!(cache.tradechances, cache.bd[base].features, cache.bd[base].lastix)
                else
                    noassetrefresh = false
                    break
                end
            end
            trade!(cache)
        end
        # if !backtest && (Dates.now(Dates.UTC)-refreshtimestamp > Dates.Minute(12*60))
        #     # TODO the read ohlcv data shall be from time to time appended to the historic data
        # end
    end
end

end  # module
