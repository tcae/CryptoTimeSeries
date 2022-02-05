include("../src/cryptoxch.jl")

"""
This module maintains promising or tagged assets to monitor. Promising means that the asset matches promising criteria.
Tagged means that it is monitored for other reasons, e.g. because the asset is part of a portfolio.

The reason why an asset is monitored is logged with timestamp and creator (algorithms or individuals).
Day and minute OHLCV data are updated.

"""
module Assets

using Dates, DataFrames, Logging, JDF
using ..EnvConfig, ..Ohlcv, ..CryptoXch


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
    df = DataFrame()
    return df
end

"manually selected assets"
manualselect() = return EnvConfig.bases
manualignore = ["usdt", "tusd", "busd", "usdc", "eur", "btt", "ust"]
minimumquotevolume = 10000000

function automaticselect(usdtdf, dayssperiod)
    bases = [usdtdf[ix, :base] for ix in 1:size(usdtdf, 1) if usdtdf[ix, :quotevolume24h] > minimumquotevolume]
    bases = [base for base in bases if !(base in manualignore)]

    enddt = Dates.now(Dates.UTC)
    startdt = enddt - dayssperiod
    deletebases = [false for _ in bases]
    for (ix, base) in enumerate(bases)
        ohlcv = CryptoXch.cryptodownload(base, "1d", startdt, enddt)
        if (size(Ohlcv.dataframe(ohlcv), 1) < 30)  # no data for the last 30 days
            deletebases[ix] = true
        else
            odf = Ohlcv.dataframe(ohlcv)
            quotevolume = odf.basevolume .* odf.close
            deletebases[ix] = !all([quotevolume[end-ix] > minimumquotevolume for ix in 0:29])  # check criteria also for last 30days
        end
    end
    deleteat!(bases, deletebases)
    return [lowercase(base) for base in bases]
end

function portfolioselect(usdtdf)
    bases = EnvConfig.bases
    if EnvConfig.configmode == EnvConfig.production
        portfolio = CryptoXch.balances()
        # [println("locked: $(d["locked"]), free: $(d["free"]), asset: $(d["asset"])") for d in portfolio]
        basevolume = [parse(Float32, d["free"]) + parse(Float32, d["locked"]) for d in portfolio]
        bases = [lowercase(d["asset"]) for d in portfolio]
        @assert length(basevolume) == length(bases)
        usdtbases = usdtdf.base
        deletebases = fill(true, (size(bases)))  # by default remove all bases that are not found in usdtbases
        for (ix, base) in enumerate(bases)
            ubix = findfirst(x -> x == base, usdtbases)
            if !(ubix === nothing)
                deletebases[ix] = (usdtdf[ubix, :lastprice] * basevolume[ix]) < 10  # remove all base symbols with a USD value <10
            # else
            #     Logging.@warn "portfolio base $base not found in USDT market bases"
            #  there are some: USDT but also currencies not tradable in USDT
            end
            if base in manualignore  # independent of portfolio volume, e.g. USDT
                deletebases[ix] = true
            end
        end
        deleteat!(bases, deletebases)
    end
    return bases
end

dataframe(ad::AssetData) = ad.df

function setdataframe!(ad::AssetData, df)
    ad.df = df
end

mnemonic() = "AssetData_v1"
savecols = [:base, :manual, :automatic, :portfolio, :xch, :update, :quotevolume24h, :priceChangePercent]

function write(ad::AssetData)
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
    println("writing asset data to $filename")
    JDF.savejdf(filename, ad.df[!, savecols])  # without :pivot
end

function read()::AssetData
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
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
    filename = EnvConfig.datafile(mnm)
    # println(filename)
    if isdir(filename)
        rm(filename; force=true, recursive=true)
    end
end

function loadassets(;dayssperiod=Dates.Year(4), minutesperiod=Dates.Week(4))::AssetData
    usdtdf = CryptoXch.getUSDTmarket()
    portfolio = Set(portfolioselect(usdtdf))
    println("#=$(length(portfolio)) loadassets portfolio: $(portfolio)")
    manual = Set(manualselect())
    automatic = Set(automaticselect(usdtdf, dayssperiod))

    allbases = union(portfolio, manual, automatic)
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - dayssperiod
    for base in allbases
        ohlcv = CryptoXch.cryptodownload(base, "1d", floor(startdt, Dates.Day), ceil(enddt, Dates.Day))
    end

    startdt = enddt - minutesperiod
    for base in allbases
        CryptoXch.cryptodownload(base, "1m", floor(startdt, Dates.Minute), ceil(enddt, Dates.Minute))
    end
    ad = AssetData(emptyassetdataframe())
    ad.df[:, :base] = [base for base in allbases]
    ad.df[:, :manual] = [ad.df[ix, :base] in manual ? true : false for ix in 1:size(ad.df, 1)]
    ad.df[:, :automatic] = [ad.df[ix, :base] in automatic ? true : false for ix in 1:size(ad.df, 1)]
    ad.df[:, :portfolio] = [ad.df[ix, :base] in portfolio ? true : false for ix in 1:size(ad.df, 1)]
    ad.df[:, :xch] .= CryptoXch.defaultcryptoexchange
    ad.df[:, :update] .= Dates.format(enddt,"yyyy-mm-dd HH:MM")
    sort!(ad.df, [:base])
    sort!(usdtdf, [:base])
    ad.df[:, :quotevolume24h] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :quotevolume24h]
    ad.df[:, :priceChangePercent] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :priceChangePercent]

    write(ad)
    return ad
end



end
