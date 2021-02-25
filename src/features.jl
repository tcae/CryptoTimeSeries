using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/ohlcv.jl")

import Pkg; Pkg.add(["RollingFunctions"])

module Features

using Dates, DataFrames, RollingFunctions, Statistics
using ..Config, ..Ohlcv

"""
Returns the index of the next extreme **after** gradient changed sign or after it was zero.
"""
function nextextremeindex(regressions, startindex)
    reglen = length(regressions)
    extremeindex = 0
    @assert (startindex > 0)
    if regressions[startindex] > 0
        while (startindex < reglen) && (regressions[startindex+1] > 0)
            startindex += 1
        end
    else  # regressions[startindex] <= 0
        while (startindex < reglen) && (regressions[startindex+1] <= 0)
            startindex += 1
        end
    end
    if startindex < reglen  # then extreme detected
        extremeindex = startindex + 1
        # else end of array and no extreme detected, which is signalled by returned index 0
    end
    return extremeindex
end


function rollingregression(price, windowsize)::Array{Union{Missing, Float32}}
    sum_x = sum(1:windowsize)
    sum_x_squared = sum((1:windowsize).^2)
    sum_xy = rolling(sum,price,windowsize,collect(1:windowsize))
    sum_y = rolling(sum,price,windowsize)
    b = ((windowsize*sum_xy) - (sum_x*sum_y))/(windowsize*sum_x_squared - sum_x^2)
    # c = [fill(missing,windowsize-1);b]
    c = [fill(0.0,windowsize-1);b]  # fill with leading zeros instead of missing
end

function normrollingregression(price, windowsize)
    return rollingregression(price, windowsize) ./ price
end

function relativevolume(volumes, shortwindow::Int, largewindow::Int)
    large = rollmedian(volumes, largewindow)
    largelen = size(large, 1)
    short = rollmedian(volumes, shortwindow)
    shortlen = size(short, 1)
    short = @view short[shortlen - largelen + 1: shortlen]
    # println("short=$short, large=$large, short/large=$(short./large)")
    return short ./ large
end

"""
4 rolling features providing the current price distance and the time distance to the last maximum and minimum
"""
function lastextremes(prices, regressions)::DataFrames.DataFrame
    tmax = 1
    tmin = 2
    pmax = 3
    pmin = 4
    lastmaxix = 1
    lastminix = 1
    dist = zeros(Float32, 4, size(regressions,1))
    for ix in 2:size(regressions,1)
        lastminix = (regressions[ix-1] < 0) && (regressions[ix] >= 0) ? ix - 1 : lastminix
        lastmaxix = (regressions[ix-1] > 0) && (regressions[ix] <= 0) ? ix - 1 : lastmaxix
        dist[pmax, ix] = (prices[lastmaxix] - prices[ix]) / prices[ix]  # normalized to last price
        dist[tmax, ix] = ix - lastmaxix
        dist[pmin, ix] = (prices[lastminix] - prices[ix]) / prices[ix]  # normalized to last price
        dist[tmin, ix] = ix - lastminix
    end
    df = DataFrame(
        pricemax = dist[pmax, :], timemax = dist[tmax, :],
        pricemin = dist[pmin, :], timemin = dist[tmin, :])
    return df
end

"""
2 rolling features providing the last forward looking relative gain and the last forward looking relative loss.
The returned dataframe contains the columns `lastgain` and `lastloss`
"""
function lastgainloss(prices, regressions)::DataFrames.DataFrame
    gainix = 1  # const
    lossix = 2  # const
    lastmaxix = [1, 1]
    lastminix = [1, 1]
    gainloss = zeros(Float32, 2, size(regressions,1))
    for ix in 2:size(regressions,1)
        if (regressions[ix-1] <= 0) && (regressions[ix] > 0)
            lastminix[1] = lastminix[2]
            lastminix[2] = ix - 1
        end
        if (regressions[ix-1] >= 0) && (regressions[ix] < 0)
            lastmaxix[1] = lastmaxix[2]
            lastmaxix[2] = ix - 1
        end
        if lastmaxix[2] > lastminix[2]  # first loss then gain -> same minimum, different maxima
            gainloss[gainix, ix] = (prices[lastmaxix[2]] - prices[lastminix[2]]) / prices[lastminix[2]]
            gainloss[lossix, ix] = (prices[lastminix[2]] - prices[lastmaxix[1]]) / prices[lastmaxix[1]]
        else  # first gain then loss -> same maximum, different mimima
            gainloss[gainix, ix] = (prices[lastmaxix[2]] - prices[lastminix[1]]) / prices[lastminix[1]]
            gainloss[lossix, ix] = (prices[lastminix[2]] - prices[lastmaxix[2]]) / prices[lastmaxix[2]]
        end
    end
    df = DataFrame(lastgain = gainloss[gainix, :], lastloss = gainloss[lossix, :])
    return df
end

"""
1 rolling feature providing the regression acceleration history, i.e. >1 (<-1) if the regression gradient is montonically increasing (decreasing)
"""
function regressionaccelerationhistory(regressions)
    acchistory = zeros(Float32, 1, size(regressions,1))
    for ix in 2:size(regressions,1)
        acceleration = regressions[ix] - regressions[ix-1]
        if acceleration > 0
            if acchistory[ix-1] > 0
                acchistory[ix] = acchistory[ix-1] + acceleration
            else
                acchistory[ix] = acceleration
            end
        elseif acceleration < 0
            if acchistory[ix-1] < 0
                acchistory[ix] = acchistory[ix-1] + acceleration
            else
                acchistory[ix] = acceleration
            end
        end  # else regressions[ix] == regressions[ix-1] => stay with acchistory[ix] = 0
    end
    return acchistory
end


"""
self.config = {
    "dpiv": [  # dpiv == difference of subsequent pivot prices
        {"suffix": "5m", "minutes": 5, "periods": 12}],
    # "range": [
    #    {"suffix": "5m", "minutes": 5, "periods": 12},  # ! to be checked - does not sem to be a good feature
    #    # ! better: % high-now and % low-now for aggregations 30min, 1h, 2h, 4h
    #    {"suffix": "1h", "minutes": 60, "periods": 4}],
    "regr": [  # regr == regression
        {"suffix": "5m", "minutes": 5},
        {"suffix": "15m", "minutes": 15},
        {"suffix": "30m", "minutes": 30},
        {"suffix": "1h", "minutes": 60},
        {"suffix": "2h", "minutes": 2*60},
        {"suffix": "4h", "minutes": 4*60},
        {"suffix": "12h", "minutes": 12*60},
        {"suffix": "24h", "minutes": 24*60},
        {"suffix": "3d", "minutes": 3*24*60},
        {"suffix": "9d", "minutes": 9*24*60}],
    "vol": [
        # {"suffix": "5m12h", "minutes": 5, "norm_minutes": 12*60},
        {"suffix": "5m1h", "minutes": 5, "norm_minutes": 60}]
"""
function f4condagg!(ohlcv::Ohlcv.OhlcvData)
    df::DataFrames.DataFrame = ohlcv.df
    df.regr5m = normrollingregression(df.pivot, 5)
    df.regr15m = normrollingregression(df.pivot, 15)
    df.regr30m = normrollingregression(df.pivot, 30)
    df.regr1h = normrollingregression(df.pivot, 60)
    df.regr2h = normrollingregression(df.pivot, 2*60)
    df.regr4h = normrollingregression(df.pivot, 4*6)
    df.regr12h = normrollingregression(df.pivot, 12*60)
    df.regr24h = normrollingregression(df.pivot, 24*60)
    df.regr3d = normrollingregression(df.pivot, 3*24*60)
    df.regr9d = normrollingregression(df.pivot, 9*24*60)
    df.vol5m1h = relativevolume(df.volume, 5, 60)
    df.vol1h1d = relativevolume(df.volume, 60, 24*60)
end

function features1(ohlcv::Ohlcv.OhlcvData)
    df::DataFrames.DataFrame = ohlcv.df
    features = zeros(Float32, (6,size(df.pivot, 1)))
    regr = normrollingregression(df.pivot, 5)
    dfgl = lastgainloss(df.pivot, regr)
    features[1,:] = regr
    features[2,:] = dfgl.lastgain
    features[3,:] = dfgl.lastloss
    features[4,:] = normrollingregression(df.pivot, 15)
    features[5,:] = normrollingregression(df.pivot, 30)
    features[6,:] = normrollingregression(df.pivot, 60)
    return features
end

# ============================
# stuff below are cross checks

function normrollingregression2(price, windowsize)::Array{Union{Missing, Float32}}
    regressionnorm = zeros(size(price, 1))
    for ix in windowsize:size(price,1)
        pricenorm = price / price[ix]
        regressionnorm[ix] = Features.rollingregression(pricenorm, windowsize)[ix]
    end
    return regressionnorm
end

# =====================
# stuff below is unused

function hellotest(config::Dict{})
    config["function"] == !hellotest ? println("this is hellotest") : println("this is NOT hellotest")
    display(config)
end

"""
config represents the parametized feature generatioin configurations - on hold for now as it seems unnecessary complicated. Instead a functioin per feature configuration is implemented.
"""
config = Dict("F4CondAgg" => [Dict("function" => hellotest,
                                    "column" => :pivot,
                                    "window" => 3),
                                Dict("function" => hellotest,
                                    "column" => :pivot,
                                    "window" => 5)])

function executeconfig()
    fct = executeconfig  # dummy assignment to declare function variable
    for (key, features) in config
        println(key)
        println(features)
        for feat in features
            println(feat)
            for (fkey, fvalue) in feat
                println("$fkey $(typeof(fkey))")
                println("$fvalue $(typeof(fvalue))")
                if fkey == "function"
                    fct = fvalue
                end
            end
            fct(feat)
        end
    end
end

end  # module