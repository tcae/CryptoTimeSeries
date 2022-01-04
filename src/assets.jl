include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/cryptoxch.jl")

"""
This module maintains promising or tagged assets to monitor. Promising means that the asset matches promising criteria.
Tagged means that it is monitored for other reasons, e.g. because the asset is part of a portfolio.

The reason why an asset is monitored is logged with timestamp and creator (algorithms or individuals).
Day and minute OHLCV data are updated.

"""
module Assets

using Dates, DataFrames, Logging, JDF
using ..Config, ..Ohlcv, ..CryptoXch


"""
AssetData.df holds the following persistent columns:

- base = symbol of traded asset
- xch = name of exchange, e.g. for crypto binance or for stocks and options nasdaq or aex
- manual = manually selected
- automatic = automatic (algorithmic) selected
- portfolio = base is part of current portfolio

"""
mutable struct AssetData
    df::DataFrames.DataFrame
end

"Returns an empty dataframe with all persistent columns"
function emptyassetdataframe()::DataFrames.DataFrame
    # df = DataFrame(base=String[], xch=String[], manual=Bool[], automatic=Bool[], portfolio=Bool[])
    df = DataFrame()
    return df
end

"manually selected assets"
function manualselect()
    if Config.configmode == production
        return [
            "btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx", "zrx", "bch",
            "etc", "link", "ada", "matic", "xtz", "zil", "omg", "xlm", "zec",
            "tfuel", "theta", "ont", "vet", "iost"]
    else  # Config.configmode == test
        return ["sinus"]
    end
end
manualignore = ["usdt", "tusd", "busd", "usdc"]
minimumquotevolume = 10000000

function automaticselect(usdtdf)
    bases = [usdtdf[ix, :base] for ix in 1:size(usdtdf, 1) if usdtdf[ix, :quotevolume24h] > minimumquotevolume]
    bases = [base for base in bases if !(base in manualignore)]
    # println("#=$(length(bases)) automaticselect check1: $bases")

    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Year(1)
    deletebases = [false for _ in bases]
    for (ix, base) in enumerate(bases)
        ohlcv = CryptoXch.cryptodownload(base, "1d", startdt, enddt)
        if (size(Ohlcv.dataframe(ohlcv), 1) < 30)  # no data for the last 30 days
            deletebases[ix] = true
        else
            odf = Ohlcv.dataframe(ohlcv)
            quotevolume = odf.basevolume .* odf.close
            deletebases[ix] = !all([quotevolume[end-ix] > minimumquotevolume for ix in 0:29])
        end
    end
    deleteat!(bases, deletebases)
    # println("#=$(length(bases)) automaticselect check2: $bases")
    return [lowercase(base) for base in bases]
end

function portfolioselect(usdtdf)
    portfolio = CryptoXch.balances()
    # [println("locked: $(d["locked"]), free: $(d["free"]), asset: $(d["asset"])") for d in portfolio]
    basevolume = [parse(Float32, d["free"]) + parse(Float32, d["locked"]) for d in portfolio]
    bases = [lowercase(d["asset"]) for d in portfolio]
    bases = [base for base in bases if !(base in manualignore)]
    usdtbases = usdtdf.base
    deletebases = fill(true, (size(bases)))  # by default remove all bases that are not found in usdtbases
    for (ix, base) in enumerate(bases)
        ubix = findfirst(x -> x == base, usdtbases)
        if !(ubix === nothing)
            deletebases[ix] = (usdtdf[ubix, :lastprice] * basevolume[ix]) < 10  # remove all base symbols with a USD value <10
        end
    end
    deleteat!(bases, deletebases)
    return bases
end

dataframe(ad::AssetData) = ad.df

function setdataframe!(ad::AssetData, df)
    ad.df = df
end

function loadassets()::AssetData
    usdtdf = CryptoXch.getUSDTmarket()
    # println("#=$(length((usdtdf.base))) loadassets check1")  # : $(usdtdf.bases)
    ad = AssetData(emptyassetdataframe())
    portfolio = Set(portfolioselect(usdtdf))
    # println("#=$(length(portfolio)) loadassets portfolio: $(portfolio)")
    manual = Set(manualselect())
    # println("#=$(length(manual)) loadassets manual: $(manual)")
    automatic = Set(automaticselect(usdtdf))
    # println("#=$(length(automatic)) loadassets automatic: $(automatic)")

    missingdayklines = setdiff(union(portfolio, manual), automatic)
    # println("#=$(length(missingdayklines)) loadassets missingdayklines: $(missingdayklines)")
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Year(1)
    for base in missingdayklines
        ohlcv = CryptoXch.cryptodownload(base, "1d", startdt, enddt)
    end

    startdt = enddt - Dates.Week(4)
    allbases = union(portfolio, manual, automatic)
    # println("#=$(length(allbases)) loadassets allbases: $(allbases)")
    for base in allbases
        CryptoXch.cryptodownload(base, "1m", startdt, enddt)
    end
    ad.df[:, :base] = [base for base in allbases]
    ad.df[:, :manual] .= false
    ad.df[in.(ad.df[!,:base], Ref([base for base in manual])), :manual] .= true
    ad.df[:, :automatic] .= false
    ad.df[in.(ad.df[!,:base], Ref([base for base in automatic])), :automatic] .= true
    ad.df[:, :portfolio] .= false
    ad.df[in.(ad.df[!,:base], Ref([base for base in portfolio])), :portfolio] .= true
    ad.df[:, :xch] .= CryptoXch.defaultcryptoexchange
    return ad
end


mnemonic() = "AssetData_v1"
save_cols = [:base, :manual, :automatic, :portfolio, :xch]

function write(ad::AssetData)
    mnm = mnemonic()
    filename = Config.datafile(mnm)
    # println(filename)
    JDF.savejdf(filename, ad.df[!, save_cols])  # without :pivot
end

function read()::AssetData
    mnm = mnemonic()
    filename = Config.datafile(mnm)
    df = emptyassetdataframe()
    # println(filename)
    if isdir(filename)
        try
            df = DataFrame(JDF.loadjdf(filename))
        catch e
            Logging.@warn "exception $e detected"
        end
    end
    ad = AssetData(df)
    return ad
end

function delete(ad::AssetData)
    mnm = mnemonic()
    filename = Config.datafile(mnm)
    # println(filename)
    if isdir(filename)
        rm(filename; force=true, recursive=true)
    end
end


end
