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

struct TradeLog
    orderid::Int64  # Dates.datetime2epochms(Dates.now(Dates.UTC))
    base::String
    ordertype::OrderType
    orderstatus::OrderStatus
    price  # opened = limit price | nothing, closed = average price, cancelled/rejected = nothing
    basevolume  # opened = base volume, closed = actual (partial) base volume, cancelled/rejected = nothing
    timestamp::Dates.DateTime
end

"""
trade cache contains all required data to support the tarde loop
"""
mutable struct Cache
    classify!
    features::Features.Features002
    lastix
    Cache(features, lastix) = new(Classify.traderules001!, features, lastix)
end

ohlcvdf(cache) = Ohlcv.dataframe(Features.ohlcv(cache.features))

function preparetradecache(backtest)
    if backtest
        bases = EnvConfig.trainingbases
        if EnvConfig.configmode == EnvConfig.test
            initialperiod = Dates.Minute(100 + Features.requiredminutes)
        else
            initialperiod = Dates.Year(4)
        end
        enddt = DateTime("2022-04-02T01:00:00")  # fix to get reproducible results
    else
        assets = Assets.loadassets()
        bases = assets.df.base
        initialperiod = Dates.Minute(Features.requiredminutes + 100)
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)  # don't use ceil because that includes a potentially partial running minute
    end
    startdt = enddt - initialperiod
    @assert startdt < enddt
    startdt = floor(startdt, Dates.Minute)
    cache = Dict()
    for base in bases
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
        cache[base] = Cache(Features.Features002(ohlcv), lastix)
    end
    return cache
end

"""
append most recent ohlcv data as well as corresponding features
returns `true` if successful appended else `false`
"""
function appendmostrecent!(cache::Cache, backtest)
    if backtest
        cache.lastix += 1
        return cache.lastix <= size(ohlcvdf(cache), 1)
    else  # production
        df = ohlcvdf(cache)
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
        CryptoXch.cryptoupdate!(Features.ohlcv(cache.features), startdt, enddt)
        df = ohlcvdf(cache)
        println("extended from $lastdt to $enddt -> check df: $(df.opentime[begin]) - $(df.opentime[end]) size=$(size(df,1))")
        # ! TODO impleement error handling
        @assert lastdt == df.opentime[lastix]  # make sure begin wasn't cut
        cache.lastix = size(df, 1)
        cache.features.update(cache.features)
        return cache.lastix  > lastix
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
    - start with max 2 active trades per base
"""
function trade!(tradechances, caches)
    println("$(length(tradechances)) trade chances")
    #  ! TODO implementation
    # TODO update TradeLog -> append to csv
    for key in keys(caches)
        println("key: $key lastix=$(caches[key].lastix)")
    end
    for (tix, tc) in enumerate(tradechances)
        ix = caches[tc.base].lastix
        f2 = caches[tc.base].features
        ixinrange = ix <= size(f2.ohlcv.df, 1)
        println("trade: $tc")
        del = []
        if ixinrange
            if (tc.buyix > 0) && (f2.ohlcv.df.high[ix] > tc.sellprice)
                push!(del, tix)
            end
            deleteat!(tradechances, del)
            Classify.registerbuy!(tradechances, tc.base, ix, f2.ohlcv.df.low[ix], f2)
            # tc.buyix = caches[tc.base].lastix
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
        caches = preparetradecache(backtest)
        noassetrefresh = true
        refreshtimestamp = Dates.now(Dates.UTC)
        tc = nothing
        while noassetrefresh
            for base in keys(caches)
                continuetrading = appendmostrecent!(caches[base], backtest)
                if continuetrading
                    tc = caches[base].classify!(tc, caches[base].features, caches[base].lastix)
                else
                    noassetrefresh = false
                    break
                end
            end
            trade!(tc, caches)
        end
        # if !backtest && (Dates.now(Dates.UTC)-refreshtimestamp > Dates.Minute(12*60))
        #     # ! TODO the read ohlcv data shall be from time to time appended to the historic data
        # end
    end
end

end  # module
