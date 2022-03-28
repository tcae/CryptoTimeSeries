# import Pkg
# Pkg.add(["RollingFunctions"])

include("../src/ohlcv.jl")

"""
Provides features to classify the most price development.
Currently advertised features are in Feature001Set

"""
module Features

# using Dates, DataFrames
import RollingFunctions: rollmedian, runmedian, rolling
import DataFrames: DataFrame, Statistics
using Combinatorics
using Logging
using ..EnvConfig, ..Ohlcv

mutable struct Feature001Set
    df::DataFrame
    featuremask::Vector{String}
    base::String
    qte::String  # instead of quote because *quote* is a Julia keyword
    xch::String  # exchange - also implies whether the asset type is crypto or stocks
    interval::String
end

indexinrange(index, last) = 0 < index <= last
nextindex(forward, index) = forward ? index + 1 : index - 1
up(slope) = slope > 0
downorflat(slope) = slope <= 0

"""
- returns index of next regression extreme and index of gradient inflection index
    - regression extreme index: in case of uphill **after** slope > 0
    - regression extreme index: in case of downhill **after** slope <= 0
- **next** index means
    - in case `forward`: next higher index
    - in case of NOT `forward`: next lower index
- it is expected that the relevant actual price maximum is between gradient inflection index and regression extreme
>> it turned out that the inflection is not reliable
>> search from regression extreme back to last price extreme for global extreme yields better results
"""
function extremeregressionindex(regressions, startindex; forward)
    reglen = length(regressions)
    extremeindex = 0
    maxabsgradix = startindex
    @assert indexinrange(startindex, reglen) "index: $startindex  len: $reglen"
    if regressions[startindex] > 0
        while indexinrange(startindex, reglen) && up(regressions[startindex])
            maxabsgradix = regressions[maxabsgradix] > regressions[startindex] ? maxabsgradix : startindex
            startindex = nextindex(forward, startindex)
        end
    else  # regressions[startindex] <= 0
        while indexinrange(startindex, reglen) && downorflat(regressions[startindex])
            maxabsgradix = regressions[maxabsgradix] < regressions[startindex] ? maxabsgradix : startindex
            startindex = nextindex(forward, startindex)
        end
    end
    if indexinrange(startindex, reglen)  # then extreme detected
        extremeindex = startindex
        # else end of array and no extreme detected, which is signalled by returned index 0
    end
    return extremeindex, maxabsgradix
end

newifbetterequal(old, new, maxsearch) = maxsearch ? new >= old : new <= old
newifbetter(old, new, maxsearch) = maxsearch ? new > old : new < old

"""
- returns index of nearest overall price extreme between `startindex` and `endindex`
- searches backwards if `startindex` > `endindex` else searches forward
- search for maximum if `maxsearch` else search for minimum

"""
function extremepriceindex(prices, startindex, endindex, maxsearch)
    @assert !(prices === nothing) && (size(prices, 1) > 0) "prices nothing == $(prices === nothing) or length == 0"
    plen = length(prices)
    extremeindex = startindex
    @assert indexinrange(startindex, plen)  "index: $startindex  len: $plen"
    @assert indexinrange(endindex, plen)  "index: $endindex  len: $plen"
    forward = startindex < endindex
    while forward ? startindex <= endindex : startindex >= endindex
        extremeindex = newifbetterequal(prices[extremeindex], prices[startindex], maxsearch) ? startindex : extremeindex
        startindex = nextindex(forward, startindex)
    end
    return extremeindex
end

"""
- returns index of nearest local price extreme between `startindex` and `endindex`
- searches backwards if `startindex` > `endindex` else searches forward
- search for maximum if `maxsearch` else search for minimum

"""
function nextlocalextremepriceindex(prices, startindex, endindex, maxsearch)
    @assert !(prices === nothing) && (size(prices, 1) > 0) "prices nothing == $(prices === nothing) or length == 0"
    plen = length(prices)
    extremeindex = startindex
    @assert indexinrange(startindex, plen)  "index: $startindex  len: $plen"
    @assert indexinrange(endindex, plen)  "index: $endindex  len: $plen"
    forward = startindex < endindex
    startindex = nextindex(forward, startindex)
    while forward ? startindex <= endindex : startindex >= endindex
        if newifbetterequal(prices[extremeindex], prices[startindex], maxsearch)
            extremeindex =  startindex
            startindex = nextindex(forward, startindex)
        else  # extreme passed
            return extremeindex
        end
    end
    return extremeindex
end

gain(prices, baseix, testix) = (baseix > testix ? prices[baseix] - prices[testix] : prices[testix] - prices[baseix]) / abs(prices[baseix])

function fillwithextremeix(distancesix, startix, endix)
    while startix < endix
        distancesix[startix] = endix
        startix += 1
    end
    return endix
end

"""
Returns the index vector to the next significant extreme.

- A maximum is considered *significant* if the gain between the last significant minimum and the next maximum >= `mingainpct`.
- A minimum is considered *significant* if the loss between the last significant maximum and the next minimum <= `minlosspct`.
- mingainpct and minlosspct are relative values related to the last significant extreme (extreme-last_extreme)/last_extreme
- If there is no next significant maximum or minimum then the distanceix == 0

maxix[2], minix[2] are updated with the latest improvement. They are reset after the counterslope is closed as significant
"""
function nextpeakindices(prices, mingainpct, minlosspct)
    distancesix = zeros(Int32, length(prices))  # 0 indicates as default no significant extremum
    minix = [1, 1, 1]
    maxix = [1, 1, 1]
    pix = 1
    plen = length(prices)
    maxix[1] = nextlocalextremepriceindex(prices, 1, plen, true)
    maxix[2] = prices[maxix[2]] < prices[maxix[1]] ? maxix[1] : maxix[2]
    minix[1] = nextlocalextremepriceindex(prices, 1, plen, false)
    minix[2] = prices[minix[2]] > prices[minix[1]] ? minix[1] : minix[2]
    while pix <= plen
        if minix[1] > maxix[1]  # last time minimum fund -> now find maximum
            maxix[1] = nextlocalextremepriceindex(prices, minix[1], plen, true)
            maxix[2] = prices[maxix[2]] < prices[maxix[1]] ? maxix[1] : maxix[2]
        elseif minix[1] < maxix[1]  # last time maximum fund -> now find minimum
            minix[1] = nextlocalextremepriceindex(prices, maxix[1], plen, false)
            minix[2] = prices[minix[2]] > prices[minix[1]] ? minix[1] : minix[2]
        else  # no further extreme should be end of prices array
            if !(minix[1] == maxix[1] == plen)
                @warn "unexpected !(minix[1] == maxix[1] == plen)" minix[1] maxix[1] plen pix
            end
        end
        if maxix[2] > minix[2]  # gain
            if gain(prices, minix[2], maxix[2]) >= mingainpct
                # as soon as gain exceeds threshold, no minimum improvement anymore possible
                if gain(prices, maxix[3], minix[2]) <= minlosspct
                    pix = fillwithextremeix(distancesix, maxix[3], minix[2])  # write loss indices
                end
                pix = minix[3] = minix[2]  # no minimum improvement anymore possible
                minix[2] = maxix[2]  # reset to follow new minix[1] improvements
            end
        elseif maxix[2] < minix[2]  # loss
            if gain(prices, maxix[2], minix[2]) <= minlosspct
                # as soon as loss exceeds threshold, no maximum improvement anymore possible
                if gain(prices, minix[3], maxix[2]) >= mingainpct
                    pix = fillwithextremeix(distancesix, minix[3], maxix[2])  # write gain indices
                end
                pix = maxix[3] = maxix[2]  # no maximum improvement anymore possible
                maxix[2] = minix[2]  # reset to follow new maxix[1] improvements
            end
        end
        if (maxix[1] == plen) || (minix[1] == plen)  # finish
            if (maxix[2] > minix[3]) && (gain(prices, minix[3], maxix[2]) >= mingainpct)
                pix = fillwithextremeix(distancesix, minix[3], maxix[2])  # write loss indices
            end
            if (maxix[3] < minix[2]) && (gain(prices, maxix[3], minix[2]) <= minlosspct)
                pix = fillwithextremeix(distancesix, maxix[3], minix[2])  # write gain indices
            end
            break
        end
    end
    distances = [ (distancesix[ix] == 0 ? 0.0 : prices[distancesix[ix]] - prices[ix] ) for ix in 1:length(prices)]
    return distances, distancesix
end

function smoothdistance(prices, lastpeakix, currentix, nextpeakix)
    # if !(0 < lastpeakix <= currentix <= nextpeakix)
    #     @warn "unexpected distancesregressionpeak index sequence" lastpeakix currentix nextpeakix
    # end
    grad = nextpeakix > lastpeakix ? (prices[nextpeakix]  - prices[lastpeakix]) / (nextpeakix  - lastpeakix) : 0.0
    smoothprice = prices[lastpeakix] + grad * (currentix - lastpeakix)
    return prices[nextpeakix] - smoothprice
end

"""
- returns distances of a smoothed price (straight price line between extremes) to next extreme
- distances will be negative if the next extreme is a minimum and positive if it is a maximum
- the extreme is determined by slope sign change of the regression gradients given in `regressions`
- from this regression extreme the peak is search backwards thereby skipping all local extrema that are insignifant for that regression window
- for debugging purposes 2 further index arrays are returned: with regression extreme indices and with price extreme indices

"""
function distancesregressionpeak(prices, regressions)
    @assert !(prices === nothing) && (size(prices, 1) > 0) "prices nothing == $(prices === nothing) or length == 0"
    @assert !(regressions === nothing) && (size(regressions, 1) > 0) "regressions nothing == $(regressions === nothing) or length == 0"
    @assert size(prices, 1) == size(regressions, 1) "size(prices) $(size(prices, 1)) != size(regressions) $(size(regressions, 1))"
    @assert length(size(prices)) == 1 "length(size(prices)) $(length(size(prices))) != 1"
    distances = zeros(Float32, length(prices))
    regressionix = zeros(Int32, length(prices))
    priceix = zeros(Int32, length(prices))
    plen = length(prices)
    gix = pix = rix = 0
    lastpix = 1
    for cix in 1:plen
        if pix <= cix
            maxsearch = regressions[cix] > 0
            if rix < cix
                rix, gix = extremeregressionindex(regressions, cix; forward=true)
                rix = rix == 0 ? plen : rix
            elseif rix < plen  # rix >= cix
                maxsearch = regressions[rix] > 0
                rix, gix = extremeregressionindex(regressions, rix; forward=true)
                rix = rix == 0 ? plen : rix
            end
            lastpix = pix > lastpix ? pix : lastpix
            # pix = extremepriceindex(prices, rix, gix, maxsearch)
            pix = extremepriceindex(prices, rix, cix, maxsearch)
        end
        distances[cix] = smoothdistance(prices, lastpix, cix, pix)  # use staright line between extremes to calculate distance
        # distances[cix] = prices[pix] - prices[cix]  # calculate the distance to the actual price which may be instable
        regressionix[cix] = rix
        priceix[cix] = pix
    end
    @assert all([0<priceix[i]<=plen for i in 1:plen]) priceix
    @assert all([0<regressionix[i]<=plen for i in 1:plen]) regressionix
    return distances, regressionix, priceix
end

"""
Returns the index of the next extreme **after** gradient changed sign or after it was zero.
"""
function nextextremeindex(regressions, startindex)
    reglen = length(regressions)
    extremeindex = 0
    @assert (startindex > 0) && (startindex <= reglen)
    if regressions[startindex] > 0
        while (startindex <= reglen) && (regressions[startindex] > 0)
            startindex += 1
        end
    else  # regressions[startindex] <= 0
        while (startindex <= reglen) && (regressions[startindex] <= 0)
            startindex += 1
        end
    end
    if startindex <= reglen  # then extreme detected
        extremeindex = startindex
        # else end of array and no extreme detected, which is signalled by returned index 0
    end
    return extremeindex
end

"""
Returns the index of the previous extreme **after** gradient changed sign or after it was zero.
"""
function prevextremeindex(regressions, startindex)
    reglen = length(regressions)
    extremeindex = 0
    @assert (startindex > 0) && (startindex <= reglen)
    if regressions[startindex] > 0
        while (startindex > 0) && (regressions[startindex] > 0)
            startindex -= 1
        end
    else  # regressions[startindex] <= 0
        while (startindex > 0) && (regressions[startindex] <= 0)
            startindex -= 1
        end
    end
    if startindex > 0  # then extreme detected
        extremeindex = startindex
        # else end of array and no extreme detected, which is signalled by returned index 0
    end
    return extremeindex
end

"""
Calculates `window` y coordinates of a regression line given by the last y of the line `regry` and the gradient `grad`.
Returns the y coordinates as a vector of size `window` of equidistant x coordinates. The x distances shall match grad.
"""
regressiony(regry, grad, window) = [regry - grad * (window - 1), regry]

"""
 This implementation ignores index and assumes an equidistant x values.
 y is a one dimensional array.

 - Regression Equation(y) = a + bx
 - Gradient(b) = (NΣXY - (ΣX)(ΣY)) / (NΣ(X^2) - (ΣX)^2)
 - Intercept(a) = (ΣY - b(ΣX)) / N
 - used from https://www.easycalculation.com/statistics/learn-regression.php

 returns 2 one dimensioinal arrays: gradient and regression_y
 gradient are the gradients per minute (= x equidistant)
 regression_y are the regression line last points

 k(x) = 0.310714 * x + 2.54286 is the linear regression of [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
 """
function rollingregression(y, windowsize)::Tuple{Array{Float32,1},Array{Float32,1}}
    sum_x = sum(1:windowsize)
    sum_x_squared = sum((1:windowsize).^2)
    sum_xy = rolling(sum, y, windowsize,collect(1:windowsize))
    sum_y = rolling(sum, y, windowsize)
    gradient = ((windowsize * sum_xy) - (sum_x * sum_y))/(windowsize * sum_x_squared - sum_x^2)
    intercept = (sum_y - gradient*(sum_x)) / windowsize
    regression_y = [[intercept[1] + gradient[1] * i for i in 1:(windowsize-1)]; intercept + (gradient .* windowsize)]
    gradient = [fill(gradient[1], windowsize-1);gradient]  # fill with first gradient instead of missing
    return regression_y, gradient
end


"""

For each x:

- expand regression to the length of window size
- subtract regression from y to remove trend within window
- calculate std and mean on resulting trend free data for just 1 x

Returns a tuple of vectors for each x calculated calculated back using the last `window` `y` data

- standard deviation of `y` minus rolling regression as given in `regr_y` and `grad`
- mean of last `window` `y` minus rolling regression as given in `regr_y` and `grad`
- x distance from regression line of last `window` points
"""
function rollingregressionstd(y, regr_y, grad, window)
    @assert size(y, 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(y, 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    normy = zeros(size(y, 1))
    std = zeros(size(y, 1))
    mean = zeros(size(y, 1))
    for ix1 in size(y, 1):-1:1
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            normy[ix2] = y[ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
        end
        mean[ix1] = Statistics.mean(normy[ix2min:ix1])
        std[ix1] = Statistics.stdm(normy[ix2min:ix1], mean[ix1])
        std[1] = 0  # not avoid NaN
    end
    return std, mean, normy
end

""" don't use - this is a version for debug reasons """
function rollingregressionstdxt(y, regr_y, grad, window)
    @assert size(y, 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(y, 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    normy = zeros(size(y, 1), window)
    std = zeros(size(y, 1))
    mean = zeros(size(y, 1))
    for ix1 in size(y, 1):-1:1
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            @assert 0<ix2<=window "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), window: $window"
            @assert 0<ix1<=size(y, 1) "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), size(y, 1): $(size(y, 1))"
            ix3 = ix2-ix2min+1
            ny = y[ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
            # Logging.@info "check" ix1 ix2 ix2min ix3 ny
            normy[ix1, ix3] = ny
        end
        mean[ix1] = Statistics.mean(normy[ix1, 1:ix1])
        std[ix1] = Statistics.stdm(normy[ix1, 1:ix1], mean[ix1])
        std[1] = 0  # not avoid NaN
    end
    Logging.@info "check" normy y regr_y grad std mean
    return std, mean, normy
end

function relativevolume(volumes, shortwindow::Int, largewindow::Int)
    # large = rollmedian(volumes, largewindow)
    # largelen = size(large, 1)
    # short = rollmedian(volumes, shortwindow)
    # shortlen = size(short, 1)
    # short = @view short[shortlen - largelen + 1: shortlen]
        large = runmedian(volumes, largewindow)
        largelen = size(large, 1)
        short = runmedian(volumes, shortwindow)
        shortlen = size(short, 1)
    # println("short=$short, large=$large, short/large=$(short./large)")
    return short ./ large
end

"""
4 rolling features providing the current price distance and the time distance to the last maximum and minimum
"""
function lastextremes(prices, regressions)::DataFrame
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
function lastgainloss(prices, regressions)::DataFrame
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
Reduces the column of the fullfeatures dataframe to the machine learning relevant columns that are part of mlfeaturenames.
"""
function mlfeatures(fullfeatures, mlfeaturenames)
    features = DataFrame()
    for feature in mlfeaturenames
        features[:, feature] = fullfeatures[!, feature]
    end
    return features
end

"""
Returns a dataframe of linear and combined features used for machine learning (ml)

- features are the linear features for machine learning
- polynomialconnect is the polynomial degree of feature combination via multiplication (usually 1, 2 or 3)

polynomialconnect example 3rd degree 4 features
a b c d
ab ac ad bc bd cd
abc abd acd bcd

"""
function polynomialfeatures!(features, polynomialconnect)
    featurenames = names(features)  # copy or ref to names? - copy is required
    for poly in 2:polynomialconnect
        for pcf in Combinatorics.combinations(featurenames, poly)
            combiname = ""
            combifeature = fill(1.0f0, (size(features, 1)))
            for feature in pcf
                combiname = combiname == "" ? feature : combiname * "*" * feature
                combifeature .*= features[:, feature]
            end
            features[:, combiname] = combifeature
        end
    end
    return features
end

function mlfeatures_test()
    fnames = ["a", "b", "c", "x", "d"]
    fnamesselect = ["a", "b", "c", "d"]
    df = DataFrame()
    for f in fnames
        df[:, f] = [1.0f0, 2.0f0]
    end
    # println("input: $df")
    df = mlfeatures(df, fnamesselect)
    df = polynomialfeatures!(df, 3)
    # println("mlfeatures output: $df")
    return all(v->(v==2.0), df[2, 1:4]) && all(v->(v==4.0), df[2, 5:10]) && all(v->(v==8.0), df[2, 11:14])
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

regressionwindows001 = Dict("5m" => 5, "15m" => 15, "1h" => 1*60, "4h" => 4*60, "12h" => 12*60, "1d" => 24*60, "3d" => 3*24*60, "10d" => 10*24*60)

sortedregressionwindowkeys001 = [d[1] for d in sort(collect(regressionwindows001), by = x -> x[2])]

"""
Properties at various rolling windows calculated on df data with ohlcv + pilot columns:

- per regression window
    - gradient of pivot regression line
    - standard deviation of regression normalized distribution
    - difference of last pivot price to regression line

"""
function features001set(pivot::Vector{Float32})
    featuremask::Vector{String} = []
    fdf = DataFrame()
    for wk in sortedregressionwindowkeys001  # wk = window key
        ws = regressionwindows001[wk]  # ws = window size in minutes
        fdf[:, "regry$wk"], fdf[:, "grad$wk"] = rollingregression(pivot, ws)
        stdnotrend, _, fdf[:, "regrdiff$wk"] = Features.rollingregressionstd(pivot, fdf[!, "regry$wk"], fdf[!, "grad$wk"], ws)
        fdf[:, "2xstd$wk"] = stdnotrend .* 2.0  # also to be used as normalization reference
        append!(featuremask, ["grad$wk", "regrdiff$wk", "2xstd$wk"])
    end
    return fdf, featuremask
end

"""
Properties at various rolling windows calculated on df data with ohlcv + pilot columns:

- per regression window
    - gradient of pivot regression line
    - standard deviation of regression normalized distribution
    - difference of last pivot price to regression line
- ration 4hour/9day volume to detect mid term rising volume
- ration 5minute/4hour volume to detect short term rising volume

"""
function features001set(indf::DataFrame)
    fdf, featuremask = features001set(indf.pivot)
    fdf[:, "4h/9dvol"] = relativevolume(indf[!, :basevolume], 4*60, 9*24*60)
    fdf[:, "5m/4hvol"] = relativevolume(indf[!, :basevolume], 5, 4*60)
    append!(featuremask, ["4h/9dvol", "5m/4hvol"])
    return fdf, featuremask
end

"""
Properties at various rolling windows calculated on 1minute OHLCV data:

- per regression window
    - gradient of pivot regression line
    - standard deviation of regression normalized distribution
    - difference of last pivot price to regression line
- ration 4hour/9day volume to detect mid term rising volume
- ration 5minute/4hour volume to detect short term rising volume

"""
function features001set(ohlcv::OhlcvData)
    indf = ohlcv.df
    pivot = Ohlcv.pivot!(indf)
    fdf, featuremask = features001set(indf)
    return Feature001Set(fdf, featuremask, ohlcv.base, ohlcv.qte, ohlcv.xch, ohlcv.interval)
end

end  # module
