include("../src/ohlcv.jl")
# include(srcdir("ohlcv.jl"))
# using Plots

"""
Produces test ohlcv data pattern
"""
module TestOhlcv

using Dates, DataFrames
using ..Config
using ..Ohlcv

"""
Returns cumulative sine function samples by adding sines on each other described by parameters given as a tuple (periodsamples, offset, amplitude).
The parameter samples defines the length of the returned functioni samples and level the zero level of the function.
"""
function sinesamples(samples, level, sineparams)
    y = zeros(samples) .+ level
    x = collect(1:samples) .- 1
    for (periodsamples, offset, amplitude) in sineparams
        println("sinedata: periodsamples=$periodsamples, offset=$offset, level=$level, amplitude=$amplitude")
        # show(DataFrame(x=x, y=y))
        @. y += sin((x + offset) * 2 * pi / (periodsamples)) * amplitude
    end
    return x, y
end

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
    df = DataFrame(opentime=timestamp, open=open, high=high, low=low, close=close, basevolume=volume)
    ohlcv = Ohlcv.defaultohlcv("testsine")
    Ohlcv.setdataframe!(ohlcv, df)
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
    df = DataFrame(opentime=timestamp, open=open, high=high, low=low, close=close, basevolume=volume)
    ohlcv = Ohlcv.defaultohlcv("doubletestsine")
    Ohlcv.setdataframe!(ohlcv, df)
    return ohlcv
end


Config.init(test)

function sinedata_test()
    ohlcv = sinedata(20, 3)
    # display(ohlcv.df)
end




end  # TestOhlcv

# plotly()
# x = 1:10; y = rand(10); # These are the plotting data
# plot(x,y, label="my label")
show("hallo")
ohlcv = TestOhlcv.sinedata(120, 3)
# df = TestOhlcv.sinedata(
#     DateTime("2019-01-02 01:11:28:121", "y-m-d H:M:S:s"),
#     DateTime("2019-01-03 01:11:28:121", "y-m-d H:M:S:s"))
# display(ohlcv.df)
# show(ohlcv.df)
# Plots.plot(ohlcv.df.timestamp, [ohlcv.df.open, ohlcv.df.high])

