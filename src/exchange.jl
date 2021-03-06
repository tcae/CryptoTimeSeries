using DrWatson
@quickactivate "CryptoTimeSeries"

# using Pkg;
# Pkg.add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))

include("../src/Binance.jl")
include("../src/env_config.jl")

module Exchange
using Dates, DataFrames, DataAPI
using JDF, CSV
using ..MyBinance, ..Config

function klines2jdict(jsonkline)
    Dict(
        :opentime => Dates.unix2datetime(jsonkline[1]/1000),
        :open => parse(Float32, jsonkline[2]),
        :high => parse(Float32, jsonkline[3]),
        :low => parse(Float32, jsonkline[4]),
        :close => parse(Float32, jsonkline[5]),
        :volume => parse(Float32, jsonkline[6]),
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
            volume=Float32[]
            # quotevolume=Float32[]
            )
    else
        len = length(jsonkline)
        df.opentime = [Dates.unix2datetime(jsonkline[ix][1]/1000) for ix in 1:len]
        df.open = [parse(Float32, jsonkline[ix][2]) for ix in 1:len]
        df.high = [parse(Float32, jsonkline[ix][3]) for ix in 1:len]
        df.low = [parse(Float32, jsonkline[ix][4]) for ix in 1:len]
        df.close = [parse(Float32, jsonkline[ix][5]) for ix in 1:len]
        df.volume = [parse(Float32, jsonkline[ix][6]) for ix in 1:len]
        # df.quotevolume = [parse(Float32, jsonkline[ix][8]) for ix in 1:len]
    end
    return df
end


"""
Requests base/USDT from start until end (both including) in minute frequency
"""
function ohlcfromexchange(base::String, startdt::DateTime, enddt::DateTime=Dates.now())
    try
        symbol = uppercase(base) * "USDT"
        # println("symbol=$symbol start=$startdt end=$enddt")
        arr = MyBinance.getKlines(symbol; startDateTime=startdt, endDateTime=enddt, interval="1m")
        # println(typeof(r))
        # show(r)
        # arr = MyBinance.r2j(r.body)
        df = klines2jdf(arr)
        # return Dict(:status => r.status, :headers => r.headers, :body => df, :version => r.version, :request => r.request)
    catch e
        println("exception $e detected")
        df = klines2jdf(missing)
    end

end

function getlastminutesdata()
    enddt = Dates.now(Dates.UTC)
    startdt = enddt - Dates.Minute(7)
    res = ohlcfromexchange("BTCUSDT", startdt, enddt)
    # display(nrow(res))
    display(last(res, 3))
    # display(first(res, 3))
    enddt = Dates.now(Dates.UTC)
    res2 = ohlcfromexchange("BTCUSDT", enddt - Dates.Second(1), enddt)
    # display(res)
    display(nrow(res2))
    display(last(res2, 3))
    println("dates equal? $(res[end, :opentime]==res2[end, :opentime])")
    # display(first(res2, 3))
    # display(res[:body][1:3, :])
    # display(res[:body][end-3:end, :])
end

function gethistoryohlcv(symbol, startdt, enddt)
    # startdt = DateTime("2020-08-11T22:45:00")
    # enddt = DateTime("2020-08-12T22:49:00")
    notreachedenddate = true
    df = nothing
    while notreachedenddate
        res = ohlcfromexchange(symbol, startdt, enddt)
        # display(nrow(res))
        # display(first(res, 3))
        # display(last(res, 3))
        notreachedenddate = !(res[end, :opentime] == enddt)
        startdt = res[end, :opentime] + Dates.Minute(1)
        df = isnothing(df) ? res : vcat(df, res; cols=:orderequal)
    end
    # display(nrow(df))
    # display(first(df, 3))
    # display(last(df, 3))
    return df
end

function balances()
    MyBinance.balances(Config.authorization.key, Config.authorization.secret)
end

end  # of module