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
minimumdayquotevolume = 1000000

"deprecated"
function automaticselect(usdtdf, volumecheckdays, minimumdayquotevolume)
    bases = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :base]
    volumecheckminutes = volumecheckdays * 24 * 60
    deletebases = [false for _ in bases]
    for (ix, base) in enumerate(bases)
        ohlcv = Ohlcv.defaultohlcv(base)
        Ohlcv.setinterval!(ohlcv, "1m")
        Ohlcv.read!(ohlcv)
        olddf = Ohlcv.dataframe(ohlcv)
        if size(olddf, 1) >= volumecheckminutes
            odf = Ohlcv.dataframe(ohlcv)
            quotevolume = odf.basevolume .* odf.close
            minimumminutequotevolume = minimumdayquotevolume / 24 / 60
            medianminutequotevolume = median(quotevolume)
            deletebases[ix] = medianminutequotevolume < minimumminutequotevolume
        else
            # insufficient data for base
            deletebases[ix] = true
        end
    end
    deleteat!(bases, deletebases)
    return [lowercase(base) for base in bases]
end

"deprecated"
function portfolioselect(ad::AssetData, usdtdf)
    pdf = DataFrame()
    portfolio = CryptoXch.balances(ad.xc)
    # [println("locked: $(d["locked"]), free: $(d["free"]), asset: $(d["asset"])") for d in portfolio]
    # pdf[:, :basevolume] = [parse(Float32, d["WalletBalance"]) + parse(Float32, d["locked"]) for d in portfolio]
    pdf[:, :basevolume] = [parse(Float32, d["walletBalance"]) for d in portfolio]
    pdf[:, :base] = [lowercase(d["coin"]) for d in portfolio]
    pdf = filter(r -> r.base in usdtdf.base, pdf)
    pdf[:, :usdt] .= 0.0
    pusdtdf = filter(r -> r.base in pdf.base, usdtdf)
    sort!(pdf, [:base])
    sort!(pusdtdf, [:base])
    @assert length(pdf.base) == length(pusdtdf.base)
    for ix in 1:length(pdf.base)
        @assert pdf[ix, :base] == pusdtdf[ix, :base]
        pdf[ix, :usdt] = pusdtdf[ix, :lastprice] * pdf[ix, :basevolume]
    end
    pdf = pdf[pdf.usdt .>= 10, :]
    return pdf
end

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
        println("writing asset data to $filename")
        JDF.savejdf(filename, ad.basedf[!, savecols])  # without :pivot
    else
        @warn "missing asset data to write to $filename"
    end
end

function read!(ad::AssetData)
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
    ad.basedf = df
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

function checkedok(ohlcv)
    return size(ohlcv.df, 1) > (11 * 24 * 60)  # data for at least 11 days
end

function loadassets!(ad::AssetData)::AssetData
    enddt = Dates.now(Dates.UTC)
    usdtdf = CryptoXch.getUSDTmarket(ad.xc)
    manual = manualselect()
    portfolio = CryptoXch.assetbases(ad.xc)
    automatic = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :basecoin]
    allbases = union(automatic, manual, portfolio)
    allbases = setdiff(allbases, CryptoXch.baseignore)
    println("usdt bases: $(automatic)")
    println("portfolio: $(portfolio)")
    println("manual: $(manual)")
    println("allbases: $(allbases)")
    # CryptoXch.downloadupdate!(ad.xc, allbases, enddt, Dates.Year(10))

    count = length(allbases)
    for (ix, base) in enumerate(allbases)
        # break
        println()
        println("$(EnvConfig.now()) start updating $base ($ix of $count)")
        startdt = enddt - Dates.Year(10)
        ohlcv = CryptoXch.cryptodownload(ad.xc, base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        if checkedok(ohlcv)
            Ohlcv.write(ohlcv)
        else
            #remove from allbases
            allbases = setdiff(allbases, [ohlcv.base])
        end
    end

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
    ad.basedf[:, :update] .= Dates.format(enddt,"yyyy-mm-dd HH:MM")
    # println("ad.basedf")
    # println(ad.basedf)
    sort!(ad.basedf, [:base])
    # println("usdtdf")
    # println(usdtdf)
    sort!(usdtdf, [:basecoin])
    ad.basedf[:, :quotevolume24h_M] = usdtdf[in.(usdtdf[!,:basecoin], Ref([base for base in ad.basedf[!, :base]])), :quotevolume24h] / 1000000
    ad.basedf[:, :pricechangepercent] = usdtdf[in.(usdtdf[!,:basecoin], Ref([base for base in ad.basedf[!, :base]])), :pricechangepercent]
    #TODO add column with std in percent for 24h regression

    write(ad)
    return ad
end



end
