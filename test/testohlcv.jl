using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/ohlcv.jl")
# include(srcdir("ohlcv.jl"))

"""
Produces test ohlcv data pattern
"""
module TestOhlcv

using Dates, DataFrames, Plots, PlotlyBase
using Test
using ..Config
using ..Ohlcv

function sinedata(periodminutes, periods, offset=0)
    price = 200
    volumeconst = 100
    amplitude = 0.007  # 0.5% of price
    firstutc = DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s")
    firstutc = round(firstutc, Dates.Minute)
    # lastutc = round(lastutc, Dates.Minute)
    # first is the reference point to reproduce the pattern
    # minutes = Int((Dates.Minute(lastutc - firstutc) + Dates.Minute(1)) / Dates.Minute(1))
    # display(minutes)
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    minutes = periodminutes * periods
    x = [(m + offset) * pi * 2 / periodminutes for m in 1:minutes]
    # x = [m * pi / (minutes/2) for m in 1:minutes]
    y = sin.(x)
    variation = -cos.(x) .* 0.01
    # display(y)
    timestamp = [firstutc + Dates.Minute(m) for m in 1:minutes]
    # display(timestamp)
    # open =   (y / 4)
    # high =   (y / 2)
    # low =    (y / 2)
    # close =  (y / 4)
    open =  price .* (y .* amplitude .+ 1 .+ variation ./ 4)
    high =  price .* (y .* amplitude .+ 1 .- 0.01 ./ 2)
    low =   price .* (y .* amplitude .+ 1 .+ 0.01 ./ 2)
    close = price .* (y .* amplitude .+ 1 .- variation ./ 4)
    volume = (1.1 .- abs.(y)) .* volumeconst
    df = DataFrame(open=open, high=high, low=low, close=close, volume=volume, timestamp=timestamp)
    ohlcv = Ohlcv.OhlcvData(df, "testsine")
    ohlcv = Ohlcv.addpivot!(ohlcv)
    return ohlcv
end

function doublesinedata(periodminutes, periods)
    price = 200
    volumeconst = 100
    amplitude = 0.007  # 0.5% of price
    firstutc = DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s")
    firstutc = round(firstutc, Dates.Minute)
    # lastutc = round(lastutc, Dates.Minute)
    # first is the reference point to reproduce the pattern
    # minutes = Int((Dates.Minute(lastutc - firstutc) + Dates.Minute(1)) / Dates.Minute(1))
    # display(minutes)
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    minutes = periodminutes * periods
    x = [m * pi * 2 / periodminutes for m in 1:minutes]
    y = (sin.(x) + sin.(2 * x)) / 2
    variation = -cos.(x) .* 0.01
    # display(y)
    timestamp = [firstutc + Dates.Minute(m) for m in 1:minutes]
    # display(timestamp)
    # open =   (y / 4)
    # high =   (y / 2)
    # low =    (y / 2)
    # close =  (y / 4)
    open =  price .* (y .* amplitude .+ 1 .+ variation ./ 4)
    high =  price .* (y .* amplitude .+ 1 .- 0.01 ./ 2)
    low =   price .* (y .* amplitude .+ 1 .+ 0.01 ./ 2)
    close = price .* (y .* amplitude .+ 1 .- variation ./ 4)
    volume = (1.1 .- abs.(y)) .* volumeconst
    df = DataFrame(open=open, high=high, low=low, close=close, volume=volume, timestamp=timestamp)
    ohlcv = Ohlcv.OhlcvData(df, "doubletestsine")
    ohlcv = Ohlcv.addpivot!(ohlcv)
    return ohlcv
end


Config.init(test)

function sinedata_test()
    ohlcv = sinedata(20, 3)
    display(ohlcv.df)
end

"""
plotly()
df = TestOhlcv.sinedata(
    DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s"),
    DateTime("2019-01-03 01:11:28:121", "y-m-d H:M:S:s"))
# show(df)
plot(df.timestamp, [df.open, df.high])


@testset "Ohlcv tests" begin


# @test Ohlcv.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)[7] == 0.310714285714285

end  # of testset
"""

end  # TestOhlcv
