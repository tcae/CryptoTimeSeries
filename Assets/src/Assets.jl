"""
This module maintains promising or tagged assets to monitor. Promising means that the asset matches promising criteria.
Tagged means that it is monitored for other reasons, e.g. because the asset is part of a portfolio.

The reason why an asset is monitored is logged with timestamp and creator (algorithms or individuals).
Day and minute OHLCV data are updated.

"""
module Assets

using Dates, DataFrames, Logging, JDF, Statistics
using EnvConfig, Ohlcv, CryptoXch, Features

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info, e.g. number of steps in rowix
"""
verbosity = 2

"""
AssetData.basedf holds the following persistent columns:

- base = symbol of traded asset
- manual = manually selected
- automatic = automatic (algorithmic) selected
- portfolio = base is part of current portfolio
- basevolume

"""
mutable struct AssetData
    basedf::AbstractDataFrame
    xc::CryptoXch.XchCache
    usdtvolume
    AssetData(basedf=emptyassetdataframe(), xc=CryptoXch.XchCache(true), usdtvolume=100.0) = new(basedf, xc, usdtvolume)
end

function Base.show(io::IO, assets::AssetData)
    println("assets basedf len = $(size(assets.basedf)) columns: $(names(assets.basedf)) - usdtvolume=$(assets.usdtvolume)")
end

"Returns an empty dataframe with all persistent columns"
function emptyassetdataframe()::DataFrames.DataFrame
    df = DataFrame()
    return df
end

manualselect() = EnvConfig.bases
minimumdayquotevolume = 2 * 1000000 # was lowered to 2 million with in many volatile coins

dataframe(ad::AssetData) = ad.basedf

function setdataframe!(ad::AssetData, df)
    ad.basedf = df
end

mnemonic() = "AssetData_v1"
savecols = [:base, :manual, :automatic, :portfolio, :update, :quotevolume24h_M, :pricechangepercent]

function write(ad::AssetData)
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
    if size(ad.basedf, 1) > 0
        (verbosity == 2) && println("$(EnvConfig.timestr(maximum(ad.basedf[!, :update]))) writing asset data of $(size(ad.basedf, 1)) base candidates to $filename")
        JDF.savejdf(filename, ad.basedf[!, savecols])  # without :pivot
    else
        @warn "missing asset data to write to $filename"
    end
end

function read!(ad::AssetData)
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
    df = emptyassetdataframe()
    (verbosity == 3) && println("$(EnvConfig.now()) loading asset info from $filename")
    if isdir(filename)
        try
            df = DataFrame(JDF.loadjdf(filename))
        catch e
            Logging.@warn "exception $e detected"
        end
    end
    ad.basedf = df
    (verbosity == 3) && println("$(EnvConfig.now()) $df")
    return ad
end

function delete(ad::AssetData)
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
    (verbosity == 3) && println("$(EnvConfig.now()) deleting asset data of $filename")
    if isdir(filename)
        rm(filename; force=true, recursive=true)
    end
end

function checkedok(ohlcv)
    return size(ohlcv.df, 1) > (11 * 24 * 60)  # data for at least 11 days
end

function loadassets!(ad::AssetData, neededbases=EnvConfig.bases)::AssetData
    enddt = Dates.now(Dates.UTC)
    usdtdf = CryptoXch.getUSDTmarket(ad.xc)
    manual = neededbases
    portfolio = CryptoXch.assetbases(ad.xc)
    automatic = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :basecoin]
    allbases = union(automatic, manual, portfolio)
    allbases = setdiff(allbases, CryptoXch.baseignore)
    (verbosity == 3) && println("usdt bases: $(automatic)")
    (verbosity == 3) && println("portfolio: $(portfolio)")
    (verbosity == 3) && println("manual: $(manual)")
    (verbosity == 3) && println("allbases: $(allbases)")
    # CryptoXch.downloadupdate!(ad.xc, allbases, enddt, Dates.Year(10))

    removebases = []
    count = length(allbases)
    for (ix, base) in enumerate(allbases)
        # break
        (verbosity >= 2) && print("\r$(EnvConfig.now()) start updating $base ($ix of $count)      ")
        startdt = enddt - Dates.Year(10)
        ohlcv = CryptoXch.cryptodownload(ad.xc, base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        if checkedok(ohlcv)
            Ohlcv.write(ohlcv)
        else
            #remove from allbases
            push!(removebases, ohlcv.base)
        end
    end
    allbases = setdiff(allbases, removebases)
    usdtdf = filter(r -> r.basecoin in allbases, usdtdf)
    checkbases = filter(!(el -> el in usdtdf.basecoin), allbases)
    if length(checkbases) > 0
        @warn "unexpected bases missing in USDT bases" checkbases
    end
    allbases = usdtdf[!, :basecoin]
    usdtvolume = 100.0  # dummy 100 USDT
    ad = AssetData(emptyassetdataframe(), ad.xc, usdtvolume)
    ad.basedf[:, :base] = [base for base in allbases]
    ad.basedf[:, :manual] = [ad.basedf[ix, :base] in manual ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :automatic] = [ad.basedf[ix, :base] in automatic ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :portfolio] = [ad.basedf[ix, :base] in portfolio ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :basevolume] .= 0.0
    # TODO add portfolio basevolume
    ad.basedf[:, :update] .= enddt  # Dates.format(enddt,"yyyy-mm-ddTHH:MM")
    sort!(ad.basedf, [:base])
    sort!(usdtdf, [:basecoin])
    ad.basedf[:, :quotevolume24h_M] = usdtdf[in.(usdtdf[!,:basecoin], Ref([base for base in ad.basedf[!, :base]])), :quotevolume24h] / 1000000
    ad.basedf[:, :pricechangepercent] = usdtdf[in.(usdtdf[!,:basecoin], Ref([base for base in ad.basedf[!, :base]])), :pricechangepercent]
    #TODO add column with std in percent for 24h regression

    write(ad)
    return ad
end



end
