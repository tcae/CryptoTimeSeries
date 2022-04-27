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

getfeatures = Features.getfeatures002
classifier = Classify.traderules001


mutable struct TradeCache
    features
    classifier
    chance::Classify.TradeChance
end



function preparetradecache(backtest)
    if backtest
        bases = EnvConfig.trainingbases
        initialperiod = Dates.Year(4)
        enddt = DateTime("2022-04-02T01:00:00")  # fix to get reproducible results
    else
        assets = Assets.loadassets()
        bases = assets.df.base
        initialperiod = Dates.Minute(Features.requiredminutes)
        enddt = floor(Dates.now(Dates.UTC), Dates.Minute)  # don't use ceil because that includes a potentially partial running minute
    end
    startdt = enddt - initialperiod
    @assert startdt < enddt
    startdt = floor(startdt, Dates.Minute)
    tradecache = Dict()
    for base in bases
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.read!(ohlcv)
        if !backtest
            origlen = size(ohlcv.df, 1)
            ohlcv.df = ohlcv.df[ohlcv.df.opentime .>= startdt, :]
            println("cutting ohlcv from $origlen to $(size(ohlcv.df)) minutes")
        end
        # CryptoXch.cryptoupdate!(ohlcv, startdt, enddt)  # not required because loadassets will already update
        if size(Ohlcv.dataframe(ohlcv), 1) < Features.requiredminutes
            @warn "insufficient ohlcv data returned for" base receivedminutes=size(Ohlcv.dataframe(ohlcv), 1) requiredminutes=Features.requiredminutes
            continue
        end
        features = getfeatures(ohlcv)
        emptytc = Classify.TradeChance(base, 0.0, 0.0, 0.0)
        tradecache[base] = TradeCache(features, classifier, emptytc)
    end
    return tradecache
end

"""
append most recent ohlcv data as well as corresponding features
"""
function appendmostrecent!(tc::TradeCache)
    #  ! TODO implementation

end

"""
- selects the trades to be executed and places orders
- corrects orders that are not yet executed
- cancels orders that are not yet executed and where sufficient gain seems unlikely
"""
function trade!(tradecache)
    #  ! TODO implementation

end

"""
**`tradeloop`** has to
+ get new exchange data (preferably non blocking)
+ evaluate new exchange data and derive trade signals
+ place new orders (preferably non blocking)
+ follow up on open orders (preferably non blocking)

"""
function tradeloop(backtest=true)
    while true  # endless loop to refresh assets from time to time
        tradecache = preparetradecache(backtest)
        noassetrefresh = true
        refreshtimestamp = Dates.now(Dates.UTC)
        lastix = maximum(values(Features.regressionwindows001))
        while noassetrefresh
            for base in keys(tradecache)
                if backtest
                    lastix += 1
                else
                    appendmostrecent!(tradecache[base])
                    lastix = Features.mostrecentix(tradecache[base].features)
                end
                tc.chance = Classify.tradechance(tc.features, lastix)
            end
            trade!(tradecache)
        end
        if !backtest && (Dates.now(Dates.UTC)-refreshtimestamp > Dates.Minute(12*60))
            # ! TODO the read ohlcv data shall be from time to time appended to the historic data
        end
    end
end


end  # module