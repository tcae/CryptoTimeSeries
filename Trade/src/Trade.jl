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
        assets = Assets.loadassets(minutesperiod=Dates.Week(4))
        bases = assets.df.base
        minrequiredminutes = maximum(values(Features.regressionwindows001))
        initialperiod = Dates.Minute(minrequiredminutes+1)
        enddt = Dates.now(Dates.UTC)
        enddt = floor(enddt, Dates.Minute)  # don't use ceil because that includes a potentially partial running minute
    end
    startdt = enddt - initialperiod
    @assert startdt < enddt
    startdt = floor(startdt, Dates.Minute)
    tradecache = TradeCache[]
    for base in bases
        ohlcv = CryptoXch.cryptodownload(base, "1m", startdt, enddt)
        df = Ohlcv.dataframe(ohlcv)
        df = df[startdt .< df.opentime .<= enddt, :]
        if size(df, 1) == 0
            @warn "unexpected empty ohlcv data returned for" base
            continue
        end
        features = Features.getfeatures(ohlcv)
        emptytc = Classify.TradeChance(base, 0.0, 0.0, 0.0)
        classifier = Classify.loadclassifier(base, features)
        push!(tradecache, TradeCache(features, emptytc))
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
    while true  # endless loop to refresh assets from time
        tradecache = preparetradecache(backtest)
        noassetrefresh = true
        lastix = maximum(values(Features.regressionwindows001))
        while noassetrefresh
            for tc in tradecache
                if backtest
                    lastix += 1
                else
                    appendmostrecent!(tc)
                    lastix = size(tc.features.fdf, 1)
                end
                tc.chance = Classify.tradechance(tc.features, lastix)
            end
            trade!(tradecache)
        end
        # ! TODO the read ohlcv data shall be from time to time appended to the historic data
    end
end


end  # module