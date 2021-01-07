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

function sinedata(firstutc::DateTime, lastutc::DateTime)
    price = 200
    volume = 100
    amplitude = price * 0.04  # 2% of price
    variation = 0.01
    firstutc = round(firstutc, Dates.Minute)
    lastutc = round(lastutc, Dates.Minute)
    # first is the reference point to reproduce the pattern
    minutes = Int((Dates.Minute(lastutc - firstutc) + Dates.Minute(1)) / Dates.Minute(1))
    # display(minutes)
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    x = [m * pi / 180 for m in 1:minutes]
    y = sin.(x)
    # display(y)
    timestamp = [firstutc + Dates.Minute(m) for m in 1:minutes]
    # display(timestamp)
    open =   (y / 4)
    high =   (y / 2)
    low =    (y / 2)
    close =  (y / 4)
    # open =   (y * amplitude / 2 + price + (abs.(y) - 1) * variation * price / 4),
    # high =   (y * amplitude / 2 + price - (abs.(y) - 1) * variation * price / 2),
    # low =    (y * amplitude / 2 + price + (abs.(y) - 1) * variation * price / 2),
    # close =  (y * amplitude / 2 + price - (abs.(y) - 1) * variation * price / 4),
    volume = -1 * (abs.(y) .- 1) * volume
    df = DataFrame(open=open, high=high, low=low, close=close, volume=volume, timestamp=timestamp)
    return df
end


Config.init(test)

function sinedata_test()
    df = sinedata(
        DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s"),
        DateTime("2019-01-03 01:11:28:121", "y-m-d H:M:S:s"))
    display(df)
end

plotly()
df = TestOhlcv.sinedata(
    DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s"),
    DateTime("2019-01-03 01:11:28:121", "y-m-d H:M:S:s"))
# show(df)
plot(df.timestamp, [df.open, df.high])


@testset "Ohlcv tests" begin


# @test Ohlcv.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)[7] == 0.310714285714285

end  # of testset

end  # TestOhlcv
