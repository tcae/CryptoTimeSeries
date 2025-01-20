module TradeUnitTest

using Test, Dates, Logging, LoggingExtras, DataFrames
using EnvConfig, Trade, Classify, Ohlcv, CryptoXch

function prepcache()
    EnvConfig.init(test)  # not production as this would result in real orders
    CryptoXch.verbosity = 3
    Classify.verbosity = 2
    # Ohlcv.verbosity = 2
    Trade.verbosity = 2
    # EnvConfig.init(production)
    enddt = DateTime("2025-01-05T11:19:00")
    startdt = enddt - Day(30)  # DateTime("2025-01-03T16:09:00")
    coins = ["ADA"]
    xc=CryptoXch.XchCache( startdt=startdt, enddt=enddt)
    cache = Trade.tradeselection!(Trade.TradeCache(xc=xc), coins, assetonly=true)
    return cache
end

function tradeamounttest(cache)

end

cache = prepcache()
tradeamounttest(cache)

end