include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/cryptoxch.jl")

"""
    This module maintains promising or tagged assets to monitor. Promising means that the asset matches promising criteria.
    Tagged means that it is monitored for other reasons, e.g. because the asset is part of a portfolio.

    The reason why an asset is monitored is logged with timestamp and creator (algorithms or individuals).
    Day and minute OHLCV data are updated.
    The asset can receive **predictions** within a given **time period** from algorithms or individuals by:

    - assigning *increase*, *neutral*, *decrease*
    - target price
    - target +-% from current price

    Prediction algorithms are identified by name. Individuals are identified by name.
"""
module Assets

using Dates, DataFrames, Logging
using ..Config, ..Ohlcv, ..CryptoXch


function cryptolistdownload(cryptolist)
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Year(1)
    for base in cryptolist
        CryptoXch.cryptodownload(base, "1d", startdt, enddt)
    end

    startdt = enddt - Dates.Week(4)
    for base in cryptolist
        CryptoXch.cryptodownload(base, "1m", startdt, enddt)
    end
end

function cryptomarketdownload()
    mdf = CryptoXch.getmarket()
    cryptolistdownload(mdf.base)
end

end
