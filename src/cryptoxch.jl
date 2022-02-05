# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# Pkg.add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])

include("../src/MyBinance.jl")
include("../src/testohlcv.jl")


module CryptoXch
using Dates, DataFrames, DataAPI, JDF, CSV, Logging
using ..MyBinance, ..EnvConfig, ..Ohlcv, ..TestOhlcv
import ..Ohlcv: intervalperiod

function klines2jdict(jsonkline)
    Dict(
        :opentime => Dates.unix2datetime(jsonkline[1]/1000),
        :open => parse(Float32, jsonkline[2]),
        :high => parse(Float32, jsonkline[3]),
        :low => parse(Float32, jsonkline[4]),
        :close => parse(Float32, jsonkline[5]),
        :basevolume => parse(Float32, jsonkline[6]),
        :closetime => Dates.unix2datetime(jsonkline[7]/1000),
        :quotevolume => parse(Float32, jsonkline[8]),
        :nbrtrades => Int32(jsonkline[9]),
        :takerbuybasevolume => parse(Float32, jsonkline[10]),
        :takerbuyquotevolume => parse(Float32, jsonkline[11]),
        :ignore => jsonkline[12]
    )
end

function klines2jdf(jsonkline)
    df = DataFrames.DataFrame()
    if ismissing(jsonkline)
        df = DataFrame(
            opentime=Float32[],
            open=Float32[],
            high=Float32[],
            low=Float32[],
            close=Float32[],
            basevolume=Float32[]
            # quotevolume=Float32[]
            )
    else
        len = length(jsonkline)
        df[:, :opentime] = [Dates.unix2datetime(jsonkline[ix][1]/1000) for ix in 1:len]
        df[:, :open] = [parse(Float32, jsonkline[ix][2]) for ix in 1:len]
        df[:, :high] = [parse(Float32, jsonkline[ix][3]) for ix in 1:len]
        df[:, :low] = [parse(Float32, jsonkline[ix][4]) for ix in 1:len]
        df[:, :close] = [parse(Float32, jsonkline[ix][5]) for ix in 1:len]
        df[:, :basevolume] = [parse(Float32, jsonkline[ix][6]) for ix in 1:len]
        # df.quotevolume = [parse(Float32, jsonkline[ix][8]) for ix in 1:len]
    end
    return df
end

defaultcryptoexchange = "binance"

"""
Requests base/USDT from start until end (both including) in interval frequency but maximum 1000 entries

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
"""
function ohlcfromexchange(base::String, startdt::DateTime, enddt::DateTime=Dates.now(), interval="1m", cryptoquote=EnvConfig.cryptoquote)
    rstatus = 0
    df = nothing
    if EnvConfig.configmode == EnvConfig.test
        println("ohlcfromexchange test mode: $base, $startdt, $enddt, $interval, $cryptoquote")
        if base in EnvConfig.bases
            rstatus, df = TestOhlcv.testdataframe(base, startdt, enddt, interval, cryptoquote)
        else
            df = Ohlcv.defaultohlcvdataframe()
            rstatus = 112
            @warn "$base is an unknown base for EnvConfig.test mode"
        end
    else
        try
            symbol = uppercase(base*cryptoquote)
            # println("symbol=$symbol start=$startdt end=$enddt")
            rstatus, arr = MyBinance.getKlines(symbol; startDateTime=startdt, endDateTime=enddt, interval=interval)
            # println(typeof(r))
            # show(r)
            # arr = MyBinance.r2j(r.body)
            df = klines2jdf(arr)
            # return Dict(:status => r.status, :headers => r.headers, :body => df, :version => r.version, :request => r.request)
        catch e
            Logging.@warn "exception $e detected"
            df = klines2jdf(missing)
        end
    end
    return rstatus, df
end

function getlastminutesdata()
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Minute(7)
    stat, res = ohlcfromexchange("BTCUSDT", startdt, enddt)
    # display(nrow(res))
    println("getlastminutesdata $stat")
    display(last(res, 3))
    # display(first(res, 3))
    enddt = Dates.now(Dates.UTC)
    stat, res2 = ohlcfromexchange("BTCUSDT", enddt - Dates.Second(1), enddt)
    println("getlastminutesdata $stat")
    # display(res)
    display(nrow(res2))
    display(last(res2, 3))
    println("dates equal? $(res[end, :opentime]==res2[end, :opentime])")
    # display(first(res2, 3))
    # display(res[:body][1:3, :])
    # display(res[:body][end-3:end, :])
end

"""
Requests base/USDT from start until end (both including) in interval frequency

Kline/Candlestick chart intervals (m -> minutes; h -> hours; d -> days; w -> weeks; M -> months):
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

time gaps will not be filled
"""
function gethistoryohlcv(base::String, startdt::DateTime, enddt::DateTime=Dates.now(Dates.UTC), interval="1m")
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    startdt = floor(startdt, intervalperiod(interval))
    enddt = floor(enddt, intervalperiod(interval))
    println("requesting from $startdt until $enddt $(ceil(enddt - startdt, intervalperiod(interval)) + intervalperiod(interval)) $base OHLCV from binance")

    notreachedenddate = true
    df = Ohlcv.defaultohlcvdataframe()
    lastdt = startdt - Dates.Minute(1)  # make sure lastdt break condition is not true
    while notreachedenddate
        stat, res = ohlcfromexchange(base, startdt, enddt, interval)
        # display(nrow(res))
        # display(first(res, 3))
        # display(last(res, 3))
        if stat != 200  # == NOT OK
            Logging.@warn "HTTP binance klines request NOT OK returning status $stat"
            break
        end
        if size(res, 1) == 0
            Logging.@warn "no $base $interval data returned by last ohlcv read from $startdt until $enddt"
            break
        end
        notreachedenddate = (res[end, :opentime] < enddt)
        if res[end, :opentime] <= lastdt
            # no progress since last ohlcv  read
            Logging.@warn "no progress since last ohlcv read"
            break
        end
        lastdt = res[end, :opentime]
        # println("$(Dates.now()) read $(nrow(res)) $base from $startdt until $lastdt")
        startdt = floor(lastdt, intervalperiod(interval))
        while (size(df,1) > 0) && (res[begin, :opentime] <= df[end, :opentime])  # replace last row with updated data
            deleteat!(df, size(df, 1))
        end
        if (size(res, 1) > 0) && (names(df) == names(res))
            df = vcat(df, res)
        else
            Logging.@error "vcat data frames names not matching df: $(names(df)) - res: $(names(res))"
            break
        end
    end
    # display(nrow(df))
    # display(first(df, 3))
    # display(last(df, 3))
    return df
end

function cryptodownload(base, interval, startdt, enddt)
    ohlcv = Ohlcv.defaultohlcv(base)
    Ohlcv.setinterval!(ohlcv, interval)
    println("Requesting $base $interval intervals from $startdt until $enddt")
    if enddt <= startdt
        Logging.@warn "Invalid datetime range: end datetime $enddt <= start datetime $startdt"
        return ohlcv
    end
    Ohlcv.read!(ohlcv)
    olddf = Ohlcv.dataframe(ohlcv)
    if size(olddf, 1) > 0  # there is already data available
        if startdt < olddf[begin, :opentime]
            # correct enddt in each case (gap between new and old range or range overlap) to avoid time range gaps
            tmpdt = olddf[begin, :opentime] - intervalperiod(interval)
            # get data of a timerange before the already available data
            newdf = gethistoryohlcv(base, startdt, tmpdt, interval)
            if size(newdf, 1) > 0
                if names(olddf) == names(newdf)
                    olddf = vcat(newdf, olddf)
                else
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
        if enddt > olddf[end, :opentime]
            tmpdt = olddf[end, :opentime]  # update last data row
            newdf = gethistoryohlcv(base, tmpdt, enddt, interval)
            if size(newdf, 1) > 0
                while (size(olddf, 1) > 0) && (newdf[begin, :opentime] <= olddf[end, :opentime])  # replace last row with updated data
                    deleteat!(olddf, size(olddf, 1))
                end
                if names(olddf) == names(newdf)
                    olddf = vcat(olddf, newdf)
                else
                    Logging.@error "vcat data frames names not matching df: $(names(olddf)) - res: $(names(newdf))"
                end
            end
            Ohlcv.setdataframe!(ohlcv, olddf)
        end
    else
        newdf = gethistoryohlcv(base, startdt, enddt, interval)
        Ohlcv.setdataframe!(ohlcv, newdf)
    end
    Ohlcv.write(ohlcv)
    return ohlcv
end

"""
Returns a dataframe with 24h values of all USDT quote bases with the following columns:

- base
- quotevolume24h
- lastprice

"""
function getUSDTmarket()
    df = DataFrames.DataFrame()
    if EnvConfig.configmode == EnvConfig.test
        df.base = EnvConfig.bases
        df.quotevolume24h = [15000000 for ix in 1:size(df,1)]
        df.lastprice = [100 for ix in 1:size(df,1)]
        df.priceChangePercent = [5.0 for ix in 1:size(df,1)]
    else
        symbols = MyBinance.getAllPrices()
        len = length(symbols)
        values = MyBinance.get24HR()
        quotesymbol = uppercase(EnvConfig.cryptoquote)
        basesix = [(ix,
            parse(Float32, values[ix]["quoteVolume"]),
            parse(Float32, values[ix]["lastPrice"]),
            parse(Float32, values[ix]["priceChangePercent"]))
            for ix in 1:len if endswith(symbols[ix]["symbol"], quotesymbol)]
        # minvolbasesix = [(ix, quotevol) for (ix, quotevol) in basesix if quotevol > minquotevolume]

        df = DataFrames.DataFrame()
        quotelen = length(quotesymbol)
        df.base = [lowercase(symbols[ix]["symbol"][1:end-quotelen]) for (ix, _, _, _) in basesix]
        df.quotevolume24h = [qv for (_, qv, _, _) in basesix]
        df.lastprice = [lp for (_, _, lp, _) in basesix]
        df.priceChangePercent = [pcp for (_, _, _, pcp) in basesix]
    end
    return df
end

function balances()
    portfolio = MyBinance.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
    # println(portfolio)
    # [println("locked: $(d["locked"]), free: $(d["free"]), asset: $(d["asset"])  all: $d") for d in portfolio]
    return portfolio
end

end  # of module