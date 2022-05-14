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
- xch = name of exchange, e.g. for crypto binance or for stocks and options nasdaq or aex
- manual = manually selected
- automatic = automatic (algorithmic) selected
- portfolio = base is part of current portfolio
- basevolume

"""
struct AssetData
    basedf::DataFrames.DataFrame
    usdtvolume
    backtest
end

"Returns an empty dataframe with all persistent columns"
function emptyassetdataframe()::DataFrames.DataFrame
    df = DataFrame()
    return df
end

manualselect() = EnvConfig.bases
minimumdayquotevolume = 10000000  # per day = 6944 per minute

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

function portfolioselect(usdtdf)
    if EnvConfig.configmode == EnvConfig.production
        portfolio = CryptoXch.balances()
        # [println("locked: $(d["locked"]), free: $(d["free"]), asset: $(d["asset"])") for d in portfolio]
        pdf = DataFrame()
        pdf[:, :basevolume] = [parse(Float32, d["free"]) + parse(Float32, d["locked"]) for d in portfolio]
        pdf[:, :base] = [lowercase(d["asset"]) for d in portfolio]
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
    else
        return []
    end
end

dataframe(ad::AssetData) = ad.basedf

function setdataframe!(ad::AssetData, df)
    ad.basedf = df
end

mnemonic() = "AssetData_v1"
savecols = [:base, :manual, :automatic, :portfolio, :xch, :update, :quotevolume24h, :priceChangePercent]

function write(ad::AssetData)
    mnm = mnemonic()
    filename = EnvConfig.datafile(mnm)
    println("writing asset data to $filename")
    JDF.savejdf(filename, ad.basedf[!, savecols])  # without :pivot
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

function loadassets(backtest=false)::AssetData
    enddt = Dates.now(Dates.UTC)
    usdtdf = CryptoXch.getUSDTmarket()
    manual = Set(manualselect())
    portfoliodf = Set(portfolioselect(usdtdf))
    portfolio = portfoliodf[!, :portfolio]
    # println("portfolio len=$(length(portfolio)) - $portfolio")
    bases = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :base]
    # println("lastdayvolume OK USDT len=$(length(bases)) - $bases")
    bases = union(Set(bases), manual, portfolio)
    # println("union1 len=$(length(bases)) - $bases")
    CryptoXch.downloadupdate!(bases, enddt, Dates.Year(4))
    automatic = Set(automaticselect(usdtdf, 30, minimumdayquotevolume))  # will use just downloaded updates
    allbases = union(portfolio, manual, automatic)
    # println("allbases len=$(length(allbases)) - $allbases")
    usdtdf = filter(r -> r.base in allbases, usdtdf)
    checkbases = filter(!(el -> el in usdtdf.base), allbases)
    if length(checkbases) > 0
        @warn "unexpected bases missing in USDT bases" checkbases
    end
    allbases = filter(el -> el in usdtdf.base, allbases)
    ad = AssetData(emptyassetdataframe(), backtest)
    ad.basedf[:, :base] = [base for base in allbases]
    ad.basedf[:, :manual] = [ad.basedf[ix, :base] in manual ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :automatic] = [ad.basedf[ix, :base] in automatic ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :portfolio] = [ad.basedf[ix, :base] in portfolio ? true : false for ix in 1:size(ad.basedf, 1)]
    ad.basedf[:, :xch] .= CryptoXch.defaultcryptoexchange
    ad.basedf[:, :basevolume] .= 0.0
    # !TODO add portfolio basevolume
    ad.basedf[:, :update] .= Dates.format(enddt,"yyyy-mm-dd HH:MM")
    # println("ad.basedf")
    # println(ad.basedf)
    sort!(ad.basedf, [:base])
    # println("usdtdf")
    # println(usdtdf)
    sort!(usdtdf, [:base])
    ad.basedf[:, :quotevolume24h] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.basedf[!, :base]])), :quotevolume24h]
    ad.basedf[:, :priceChangePercent] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.basedf[!, :base]])), :priceChangePercent]

    write(ad)
    return ad
end



end
