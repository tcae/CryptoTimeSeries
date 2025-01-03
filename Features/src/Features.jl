"""
Provides features to classify the most price development.
Currently advertised features are in Features001

"""
module Features

# using Dates, DataFrames
import RollingFunctions: rollmedian, runmedian, rolling
using RollingFunctions
using DataFrames, Statistics, Indicators, JDF
using Combinatorics, Dates
using Logging
using EnvConfig, Ohlcv

#region abstract-features

"Defines the features interface that shall be provided by all feature implementations."
abstract type AbstractFeatures <: EnvConfig.AbstractConfiguration end

"Adds a coin with OhlcvData to the feature generation. It will remove any previously added ohlcv and corresponding features."
function setbase!(features::AbstractFeatures, ohlcv::Ohlcv.OhlcvData) end

"Removes any previously added ohlcv and corresponding the features."
function removebase!(features::AbstractFeatures) end

"Returns the OhlcvData reference of features"
function ohlcv(features::AbstractFeatures) end

"Returns a dataframe view that matches the f5 features time range of the last added ohlcv"
function ohlcvdfview(features::AbstractFeatures) end

"Provides a description that characterizes the features"
function describe(features::AbstractFeatures)::String end

"Defines the number of minutes history required to provide the first suitable feature set"
requiredminutes(features::AbstractFeatures) = 0

"Add newer features to match the recent timeline of ohlcv[firstix:lastix] with the newest ohlcv datapoints, i.e. datapoints newer than last(features)"
function supplement!(features::AbstractFeatures) end

"returns a dataframe with an opentime column followed by feature columns"
function features(features::AbstractFeatures, firstix::Integer, lastix::Integer) end
function features(features::AbstractFeatures, firstix::DateTime, lastix::DateTime) end

"Cuts the features time range to match the ohlcv time range that was used to derive the features"
function timerangecut!(features::AbstractFeatures) end

#endregion abstract-features

#region FeatureUtilities

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

periodlabels(p) = typeof(p) <: Integer ? (p%(24*60) == 0 ? ("$(round(Int, p/(24*60)))d") : (p%60 == 0 ? "$(round(Int, p/60))h" : "$(p)m")) : p

indexinrange(index, start, last) = start <= index <= last
nextindex(forward, index) = forward ? index + 1 : index - 1
uporflat(slope) = slope >= 0
downorflat(slope) = slope <= 0

relativedayofyear(date::DateTime)::Float32 = round(Dates.dayofyear(date) / 365.0, digits=4)
relativedayofweek(date::DateTime)::Float32 = round(Dates.dayofweek(date) / 7.0, digits=4)
relativeminuteofday(date::DateTime)::Float32 = round(Dates.Minute(round(date, Dates.Minute(1)) - DateTime(Date(date))).value / 1440.0, digits=4)
relativetimedict = Dict(
    "relminuteofday" => relativeminuteofday,
    "reldayofweek" => relativedayofweek,
    "reldayofyear" => relativedayofyear)

"""
- returns index of next regression extreme or last (if forward) / first index in case of no extreme
    - regression extreme index: in case of uphill **after** slope >= 0, i.e. index of first downhill grad as positive number to indicate maximum
    - regression extreme index: in case of downhill **after** slope <= 0, i.e. index of first uphill grad as negative number to indicate minimum
    - if there is any slope then teh last index is considered an extreme

"""
function extremeregressionindex(regressions, startindex; forward)
    startindex= abs(startindex)
    regend = lastindex(regressions)
    regstart = firstindex(regressions)
    extremeindex = 0
    @assert indexinrange(startindex, regstart, regend) "index: $startindex  len: $regend"
    upwards = regressions[startindex] > 0
    downwards = regressions[startindex] < 0
    while indexinrange(startindex, regstart, regend) && regressions[startindex] == 0
        startindex = nextindex(forward, startindex)
    end
    if indexinrange(startindex, regstart, regend)
        upwards = regressions[startindex] > 0
        downwards = regressions[startindex] < 0
    else
        upwards = downwards = false
    end
    if upwards
        while indexinrange(startindex, regstart, regend) && uporflat(regressions[startindex])
            startindex = nextindex(forward, startindex)
        end
    elseif downwards
        while indexinrange(startindex, regstart, regend) && downorflat(regressions[startindex])
            startindex = nextindex(forward, startindex)
        end
    end
    if indexinrange(startindex, regstart, regend)  # then extreme detected
        extremeindex = forward ? (downwards ? -startindex : startindex) : (downwards ? startindex : -startindex)
    else
        extremeindex = forward ? (downwards ? -regend : regend) : (downwards ? regstart : -regstart)
    end
    return extremeindex
end

"""
Returns an index array of regression extremes.
The index is >0 in case of a maximum and <0 in case of a minimum. The index is the absolute number.
The index array is either created if not present or is extended.

    - extremeix will be amended or created and will also be returned
    - regressiongradients are the gradients of the regression lines calculated for each price
    - startindex is the index within regressiongradients where the search for extremes shall start
    - if forward (default) the extremeix ix will be appended otherwise added to the start
    - the first in case of forward == false or the last in case of forward == true will always be added

"""
function regressionextremesix!(extremeix::Union{Nothing, Vector{T}}, regressiongradients, startindex=firstindex(regressiongradients); forward=true) where {T<:Integer}
    regend = lastindex(regressiongradients)
    regstart = firstindex(regressiongradients)
    @assert indexinrange(startindex, regstart, regend) "index: $startindex  len: $regend"
    xix = extremeregressionindex(regressiongradients, startindex; forward)
    if isnothing(extremeix)
        extremeix = Int32[]
    elseif forward  # clean up extremeix that xix connects right - especially removes last element that was not an extreme but an end of array
        ix = lastindex(extremeix)
        while (length(extremeix) > 0) && ((abs(extremeix[ix]) >= abs(xix)) || (sign(extremeix[ix]) == sign(xix)))
            if ix == firstindex(extremeix)
                extremeix = Int32[]
                break
            else
                pop!(extremeix)
            end
            ix = lastindex(extremeix)
        end
    else  # backward == !forward - clean up extremeix that xix connects right - especially removes first element that was not an extreme but a start of array
        ix = firstindex(extremeix)
        while (length(extremeix) > 0) && ((abs(extremeix[ix]) <= abs(xix)) || (sign(extremeix[ix]) == sign(xix)))
            if ix == lastindex(extremeix)
                extremeix = Int32[]
                break
            else
                deleteat!(extremeix, 1)
            end
            ix = firstindex(extremeix)  # should be the same as before
        end
    end
    while true
        if forward
            @assert (length(extremeix) == 0) || (length(extremeix) > 0) && ((abs(extremeix[end]) >= abs(xix)) || (sign(extremeix[end]) != sign(xix))) "inconsistency: extremeix[end]=$(length(extremeix) > 0 ? extremeix[end] : "[]") xix=$xix"
            extremeix = push!(extremeix, xix)
        else
            @assert (length(extremeix) == 0) || (length(extremeix) > 0) && ((abs(first(extremeix)) <= abs(xix)) || (sign(first(extremeix)) != sign(xix))) "inconsistency: first(extremeix)=$(length(extremeix) > 0 ? first(extremeix) : "[]") xix=$xix"
            extremeix = pushfirst!(extremeix, xix)
        end
        arrayend = forward ? abs(xix) == lastindex(regressiongradients) : abs(xix) == firstindex(regressiongradients)
        arrayend ? break : false
        xix = extremeregressionindex(regressiongradients, xix; forward)
    end
    # println("regressionextremesix!: f2 extremeix=$extremeix")
    return extremeix
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
    pend = lastindex(prices)
    pstart = firstindex(prices)
    extremeindex = startindex
    @assert indexinrange(startindex, pstart, pend)  "index: $startindex  len: $pend"
    @assert indexinrange(endindex, pstart, pend)  "index: $endindex  len: $pend"
    forward = startindex < endindex
    while forward ? startindex <= endindex : startindex >= endindex
        extremeindex = newifbetter(prices[extremeindex], prices[startindex], maxsearch) ? startindex : extremeindex
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
    pend = lastindex(prices)
    pstart = firstindex(prices)
    extremeindex = startindex
    @assert indexinrange(startindex, pstart, pend)  "index: $startindex  len: $pend"
    @assert indexinrange(endindex, pstart, pend)  "index: $endindex  len: $pend"
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
function nextpeakindices(prices::Vector{T}, mingainpct, minlosspct) where {T<:Real}
    distancesix = zeros(Int32, length(prices))  # 0 indicates as default no significant extremum
    minix = [1, 1, 1]
    maxix = [1, 1, 1]
    pix = 1
    pend = lastindex(prices)
    maxix[1] = nextlocalextremepriceindex(prices, 1, pend, true)
    maxix[2] = prices[maxix[2]] < prices[maxix[1]] ? maxix[1] : maxix[2]
    minix[1] = nextlocalextremepriceindex(prices, 1, pend, false)
    minix[2] = prices[minix[2]] > prices[minix[1]] ? minix[1] : minix[2]
    while pix <= pend
        if minix[1] > maxix[1]  # last time minimum fund -> now find maximum
            maxix[1] = nextlocalextremepriceindex(prices, minix[1], pend, true)
            maxix[2] = prices[maxix[2]] < prices[maxix[1]] ? maxix[1] : maxix[2]
        elseif minix[1] < maxix[1]  # last time maximum fund -> now find minimum
            minix[1] = nextlocalextremepriceindex(prices, maxix[1], pend, false)
            minix[2] = prices[minix[2]] > prices[minix[1]] ? minix[1] : minix[2]
        else  # no further extreme should be end of prices array
            if !(minix[1] == maxix[1] == pend)
                @warn "unexpected !(minix[1] == maxix[1] == pend)" minix[1] maxix[1] pend pix
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
        if (maxix[1] == pend) || (minix[1] == pend)  # finish
            if (maxix[2] > minix[3]) && (gain(prices, minix[3], maxix[2]) >= mingainpct)
                pix = fillwithextremeix(distancesix, minix[3], maxix[2])  # write loss indices
            end
            if (maxix[3] < minix[2]) && (gain(prices, maxix[3], minix[2]) <= minlosspct)
                pix = fillwithextremeix(distancesix, maxix[3], minix[2])  # write gain indices
            end
            break
        end
    end
    distances = [ (distancesix[ix] == 0 ? T(0.0) : prices[distancesix[ix]] - prices[ix] ) for ix in eachindex(prices)]  # 1:length(prices)]
    return distances, distancesix
end

"returns the price difference of current price to next peak price based on a straight line aaproach between last peak and next peak "
function smoothdistance(prices, lastpeakix, currentix, nextpeakix)
    # if !(0 < lastpeakix <= currentix <= nextpeakix)
    #     @warn "unexpected pricediffregressionpeak index sequence" lastpeakix currentix nextpeakix
    # end
    grad = nextpeakix > lastpeakix ? (prices[nextpeakix]  - prices[lastpeakix]) / (nextpeakix  - lastpeakix) : 0.0
    smoothprice = prices[lastpeakix] + grad * (currentix - lastpeakix)
    return prices[nextpeakix] - smoothprice
end

maxsearch(regressionextremeindex) = regressionextremeindex > 0

"""
- if `smoothing == true` returns pricediffs of a smoothed price (straight price line between extremes) to next extreme price
- if `smoothing == false` returns pricediffs of current price to next extreme price
- pricediffs will be negative if the next extreme is a minimum and positive if it is a maximum
- the extreme is determined by slope sign change of the regression gradients given in `regressions`
- from this regression extreme the peak is search backwards thereby skipping all local extrema that are insignifant for that regression window
- 2 further index arrays are returned: with regression extreme indices and with price extreme indices

"""
function pricediffregressionpeak(prices, regressiongradients; smoothing=true) #! deprecated
    #! deprecated
    @error "Features.pricediffregressionpeak is deprecated and replaced by Targets.peaksbeforeregressiontargets"
    return


    @assert !(prices === nothing) && (size(prices, 1) > 0) "prices nothing == $(prices === nothing) or length == 0"
    @assert !(regressiongradients === nothing) && (size(regressiongradients, 1) > 0) "regressions nothing == $(regressiongradients === nothing) or length == 0"
    @assert size(prices, 1) == size(regressiongradients, 1) "size(prices) $(size(prices, 1)) != size(regressions) $(size(regressiongradients, 1))"
    pricediffs = zeros(Float32, length(prices))
    regressionix = zeros(Int32, length(prices))
    priceix = zeros(Int32, length(prices))
    pend = lastindex(prices)
    lastpix = pix = rix = firstindex(prices)
    rix = firstindex(regressiongradients)
    for cix in eachindex(prices)
        if (abs(pix) <= cix)
            if abs(rix) == lastindex(regressiongradients)
                pix = -sign(rix) * lastindex(prices)  # pix backwwards search alerady done
            else
                rix = extremeregressionindex(regressiongradients, rix; forward=true)
                pix = sign(rix) * extremepriceindex(prices, abs(rix), min((cix), abs(rix)), maxsearch(rix))  # search back from extreme rix to current index cix
            end
            lastpix = abs(pix) > lastpix ? abs(pix) : lastpix
        end
        if smoothing
            pricediffs[cix] = smoothdistance(prices, lastpix, cix, abs(pix))  # use straight line between extremes to calculate distance
        else
            pricediffs[cix] = prices[abs(pix)]  - prices[cix]
        end
        regressionix[cix] = rix
        priceix[cix] = pix
    end
    @assert all([0 < abs(priceix[i]) <= pend for i in 1:pend]) priceix
    @assert all([0 < abs(regressionix[i]) <= pend for i in 1:pend]) regressionix
    # println("pricediffs=$pricediffs")
    # println("regressionix=$regressionix")
    # println("priceix=$priceix")
    return pricediffs, regressionix, priceix
end

"""
Returns the index of the next extreme **after** gradient changed sign or after it was zero.
"""
function nextextremeindex(regressions, startindex)
    regend = lastindex(regressions)
    extremeindex = 0
    @assert (startindex > 0) && (startindex <= regend)
    if regressions[startindex] > 0
        while (startindex <= regend) && (regressions[startindex] > 0)
            startindex += 1
        end
    else  # regressions[startindex] <= 0
        while (startindex <= regend) && (regressions[startindex] <= 0)
            startindex += 1
        end
    end
    if startindex <= regend  # then extreme detected
        extremeindex = startindex
        # else end of array and no extreme detected, which is signalled by returned index 0
    end
    return extremeindex
end

"""
Returns the index of the previous extreme **after** gradient changed sign or after it was zero.
"""
function prevextremeindex(regressions, startindex)
    regend = lastindex(regressions)
    extremeindex = 0
    @assert (startindex > 0) && (startindex <= regend)
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
 This implementation ignores index and assumes an equidistant x values.
 y is a one dimensional array.

 - Regression Equation(y) = a + bx
 - Gradient(b) = (NΣXY - (ΣX)(ΣY)) / (NΣ(X^2) - (ΣX)^2) (y increase per x unit)
 - Intercept(a) = (ΣY - b(ΣX)) / N
 - used from https://www.easycalculation.com/statistics/learn-regression.php

 returns 2 one dimensioinal arrays: gradient and regression_y that start at max(1, startindex-windowsize)
 gradient are the gradients per minute (= x equidistant)
 regression_y are the regression line last points

 k(x) = 0.310714 * x + 2.54286 is the linear regression of [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
 """
 function rollingregression(y, windowsize)
    sum_x = sum(1:windowsize)
    sum_x_squared = sum((1:windowsize).^2)
    sum_xy = rolling(sum, y, windowsize,collect(1:windowsize))
    sum_y = rolling(sum, y, windowsize)
    gradient = ((windowsize * sum_xy) - (sum_x * sum_y))/(windowsize * sum_x_squared - sum_x^2)
    intercept = (sum_y - gradient*(sum_x)) / windowsize
    regression_y = [[intercept[1] + gradient[1] * i for i in 1:(windowsize-1)]; intercept + (gradient .* windowsize)]
    gradient = [fill(gradient[1], windowsize-1);gradient]  # fill with first gradient instead of missing
    return regression_y[windowsize:end], gradient[windowsize:end]
end

"""
Acts like rollingregression(y, windowsize) but starts calculation at *startindex-windowsize+1*.
In order to get only the regression_y, gradient without padding use the subvectors *[windowsize:end]*
"""
function rollingregression(y, windowsize, startindex)
    startindex = max(1, startindex-windowsize+1)
    suby = y[startindex:end]
    sum_x = sum(1:windowsize)
    sum_x_squared = sum((1:windowsize).^2)
    sum_xy = rolling(sum, suby, windowsize,collect(1:windowsize))
    sum_y = rolling(sum, suby, windowsize)
    gradient = ((windowsize * sum_xy) - (sum_x * sum_y))/(windowsize * sum_x_squared - sum_x^2)
    intercept = (sum_y - gradient*(sum_x)) / windowsize
    regression_y = [[intercept[1] + gradient[1] * i for i in 1:(windowsize-1)]; intercept + (gradient .* windowsize)]
    gradient = [fill(gradient[1], windowsize-1);gradient]  # fill with first gradient instead of missing
    return regression_y[windowsize:end], gradient[windowsize:end]
end

"""
calculates and appends missing `length(y) - length(regressions)` *regression_y, gradient* elements that correpond to the last elements of *y*
"""
function rollingregression!(regression_y, gradient, y, windowsize)
    if (length(y) > 0)
        startindex = isnothing(regression_y) ? 1 : (length(regression_y) < length(y)) ? length(regression_y)+1 : length(y)
        regnew, gradnew = rollingregression(y, windowsize, startindex)
        if isnothing(regression_y) || isnothing(gradient)
            regression_y = regnew
            gradient = gradnew
        elseif length(regression_y) < length(y)  # only change regression_y and gradient if it makes sense
            @assert size(regression_y, 1) == size(gradient, 1)
            startindex = min(startindex, windowsize)
            regression_y = append!(regression_y, regnew[startindex:end])
            gradient = append!(gradient, gradnew[startindex:end])
        else
            @warn "nothing to append when length(regression_y) >= length(y)" length(regression_y) length(y)
        end
    end
    return regression_y, gradient
end

function normrollingregression(price, windowsize)
    return rollingregression(price, windowsize) ./ price
end


"""

For each x:

- expand regression to the length of window size
- subtract regression from y to remove trend within window
- calculate std and mean on resulting trend free data for just 1 x

Returns a tuple of vectors for each x calculated calculated back using the last `window` `y` data

- standard deviation of `y` minus rolling regression as given in `regr_y` and `grad`
- mean of last `window` `y` minus rolling regression as given in `regr_y` and `grad`
- y distance from regression line of last `window` points
"""
function rollingregressionstd(y, regr_y, grad, window)
    @assert size(y, 1) >= window  > 0 "$(size(y, 1)), $window"
    @assert (size(y, 1) - window + 1) == size(regr_y, 1) == size(grad, 1) > 0 "size(y, 1)=$(size(y, 1)), window=$window, size(regr_y, 1)=$(size(regr_y, 1)), size(grad, 1)=$(size(grad, 1))"
    normy = similar(y)
    std = similar(y)
    mean = similar(y)
    # normy .= 0
    # std .= 0
    # mean .= 0
    for ix1 in size(y, 1):-1:window
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            normy[ix2] = y[ix2] - (regr_y[ix1-window+1] - grad[ix1-window+1] * (ix1 - ix2))
        end
        mean[ix1] = Statistics.mean(normy[ix2min:ix1])
        std[ix1] = Statistics.stdm(normy[ix2min:ix1], mean[ix1])
    end
    std[1] = 0  # not avoid NaN
    return std[window:end], mean[window:end], normy[window:end]
end

"""
Acts like rollingregressionstd(y, regr_y, grad, window) but starts calculation at *startindex-windowsize+1*.
In order to get only the std, mean, normy without padding use the subvectors *[windowsize:end]*
"""
function rollingregressionstd(y, regr_y, grad, window, startindex)
    @assert size(y, 1) >= window  > 0 "$(size(y, 1)), $window"
    @assert (size(y, 1) - window + 1) == size(regr_y, 1) == size(grad, 1) > 0 "size(y, 1)=$(size(y, 1)), window=$window, size(regr_y, 1)=$(size(regr_y, 1)), size(grad, 1)=$(size(grad, 1))"
    starty = max(1, startindex-window+1)
    offset = starty - 1
    normy = similar(y[starty:end])
    std = similar(normy)
    mean = similar(normy)
    # normy .= 0
    # std .= 0
    # mean .= 0
    for ix1 in size(y, 1):-1:max(startindex, window)
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            normy[ix2-offset] = y[ix2] - (regr_y[ix1-window+1] - grad[ix1-window+1] * (ix1 - ix2))
        end
        ix3 = ix2min-offset
        ix4 = ix1-offset
        mean[ix4] = Statistics.mean(normy[ix3:ix4])
        std[ix4] = Statistics.stdm(normy[ix3:ix4], mean[ix4])
    end
    std[1] = 0  # not avoid NaN
    startindex = max(startindex, window)
    return std[startindex:end], mean[startindex:end], normy[startindex:end]
end

""" don't use - this is a version for debug reasons """
function rollingregressionstdxt(y, regr_y, grad, window)
    @assert size(y, 1) >= window  > 0 "$(size(y, 1)), $window"
    @assert (size(y, 1) - window + 1) == size(regr_y, 1) == size(grad, 1) > 0 "size(y, 1)=$(size(y, 1)), window=$window, size(regr_y, 1)=$(size(regr_y, 1)), size(grad, 1)=$(size(grad, 1))"
    normy = similar(y, (size(y, 1), window))
    std = similar(y)
    mean = similar(y)
    for ix1 in size(y, 1):-1:window
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            # @assert 0<ix2<=window "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), window: $window"
            # @assert 0<ix1<=size(y, 1) "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), size(y, 1): $(size(y, 1))"
            ix3 = ix2-ix2min+1
            ny = y[ix2] - (regr_y[ix1-window+1] - grad[ix1-window+1] * (ix1 - ix2))
            # Logging.@info "check" ix1 ix2 ix2min ix3 ny
            normy[ix1, ix3] = ny
        end
        mean[ix1] = Statistics.mean(normy[ix1, 1:min(window, ix1)])
        std[ix1] = Statistics.stdm(normy[ix1, 1:min(window, ix1)], mean[ix1])
        std[1] = 0  # not avoid NaN
    end
    # Logging.@info "check" normy y regr_y grad std mean
    return std[window:end], mean[window:end], normy[window:end]
end

"""

For each x starting at *startindex-windowsize+1*: (starindex is related to ymv)

- expand regression to the length of window size
- subtract regression from y to remove trend within window
- calculate std and mean on resulting trend free data for just 1 x

Returns a std vector of length `length(regr_y) - startindex + 1` for each x calculated back using the last `window` `ymv[*]` data

In multiple vectors *mv* version, ymv is an array of ymv vectors all of the same length like regr_y and grad

- standard deviation of `ymv` vectors minus rolling regression as given in `regr_y` and `grad`

In order to get only the std without padding use the subvector *[windowsize:end]*
"""
function rollingregressionstdmv(ymv, regr_y, grad, window, ymvstartindex)
    ymvstartindex = max(ymvstartindex, window)
    @assert size(ymv, 1) > 0
    @assert size(ymv[1], 1) >= window > 0 "size(ymv[1], 1)=$(size(ymv[1], 1)), ymvstartindex=$ymvstartindex, window=$window"
    @assert (size(ymv[1], 1) - ymvstartindex + 1) == size(regr_y, 1) == size(grad, 1) > 0 "size(ymv[1], 1)=$(size(ymv[1], 1)), window=$window, size(regr_y, 1)=$(size(regr_y, 1)), size(grad, 1)=$(size(grad, 1)) ymvstartindex=$ymvstartindex"
    ymvlen = size(ymv, 1) # number of ohlc == 4
    normy = repeat(similar([ymv[1][1]], window * ymvlen))
    std = similar(regr_y)
    for ix1 in ymvstartindex:lastindex(ymv[1])
        ix2min = max(1, ix1 - window + 1)
        thiswindow = ix1 - ix2min + 1
        normy .= 0.0
        normyix = 1
        for ymvix in 1:ymvlen
            for ix2 in (ix1 - window + 1):ix1
                # println("normyix=$normyix length(normy)=$(length(normy)) ix1=$ix1 ymvix=$ymvix ix2=$ix2")
                normy[normyix] = ymv[ymvix][ix2] - (regr_y[ix1-ymvstartindex+1] - grad[ix1-ymvstartindex+1] * (ix1 - ix2))
                normyix += 1
            end
        end
        normylen = ymvlen*thiswindow
        ymvmean = Statistics.mean(normy[1:normylen])
        std[ix1-ymvstartindex+1] = Statistics.stdm(normy[1:normylen], ymvmean)
    end
    std[1] = isnan(std[1]) ? 0 : std[1]
    return std
end

"""
calculates and appends missing `length(y) - length(std)` *std, mean, normy* elements that correpond to the last elements of *ymv*
"""
function rollingregressionstdmv!(std, ymv, regr_y, grad, window)
    @warn "broken regression test"
    @assert size(ymv, 1) > 0
    @assert size(ymv[1],1)-window+1 == size(regr_y,1) == size(grad,1) "$(size(ymv[1],1))-$window+1 == $(size(regr_y,1)) == $(size(grad,1))"
    ymvlen = size(ymv, 1)
    if (length(regr_y) > 0)
        startindex = isnothing(std) ? 1 : (length(std) < length(regr_y)) ? length(std)+1 : length(regr_y)
        stdnew = rollingregressionstdmv(ymv, regr_y, grad, window, startindex)
        if isnothing(std)
            std = stdnew
        elseif length(std) < length(regr_y)  # only change regression_y and gradient if it makes sense
            # std = append!(std, stdnew[startindex:end])
            std = append!(std, stdnew)
            @assert length(std) == length(regr_y) "length(std)=$(length(std)) == length(regr_y)=$(length(regr_y))"
        else
            @warn "nothing to append when length(std) >= length(y)" length(std) ymvlen
        end
    end
    return std
end

"""
Returns the rolling median of std over requiredminutes starting at startindex.
If window > 0 then the window length is subtracted from requiredminutes because it is considered in the first std

This function is deprecated as it did not result in better results than using std
"""
function rollingmedianstd!(medianstd, std, requiredminutes, startindex, window=1)
    if window > requiredminutes
        @warn "rollingmedianstd! window=$window > requiredminutes=$requiredminutes"
        window = requiredminutes
    end
    if isnothing(medianstd) || (startindex <= 1)
        # @info "rollingmedianstd! startindex=$startindex length(std)=$(length(std)) requiredminutes=$requiredminutes window=$window"
        medianstd = runmedian(std, requiredminutes-window+1)
    else
        # @info "rollingmedianstd! startindex=$startindex length(medianstd)=$(length(medianstd)) length(std)=$(length(std)) requiredminutes=$requiredminutes window=$window"
        calcstart = max(1, startindex-requiredminutes+window)
        tmpstd = runmedian(std[calcstart:end], requiredminutes-window+1)
        endoffset = length(std) - startindex
        append!(medianstd, tmpstd[end-endoffset:end])
    end
    @assert length(std) == length(medianstd)
    return medianstd
end

# 1000, 100, 10, si = 991, ci = 891, l(t)=110
# t 110 - 10

function correctunworkablevolume(vv::Vector{T})::Vector{T} where {T <: AbstractFloat}
    return map(x -> x <= 0.0 ? 0.0001 : x, vv)
end

function relativevolume(volumes, shortwindow, largewindow)
    # large = rollmedian(volumes, largewindow)
    # largelen = size(large, 1)
    # short = rollmedian(volumes, shortwindow)
    # shortlen = size(short, 1)
    # short = @view short[shortlen - largelen + 1: shortlen]
    large = runmedian(volumes, largewindow)
    large = correctunworkablevolume(large)
    # largelen = size(large, 1)
    short = runmedian(volumes, shortwindow)
    # shortlen = size(short, 1)
    # println("short=$short, large=$large, short/large=$(short./large)")
    return short ./ large
end

"""
4 rolling features providing the current price distance and the time distance to the last maximum and minimum
"""
function lastextremes(prices, regressions)::AbstractDataFrame
    tmax = 1
    tmin = 2
    pmax = 3
    pmin = 4
    lastmaxix = 1
    lastminix = 1
    dist = zeros(Float32, 4, size(regressions,1))
    for ix in Iterators.drop(eachindex(regressions), 1)  # 2:size(regressions,1)
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
function lastgainloss(prices, regressions)::AbstractDataFrame
    gainix = 1  # const
    lossix = 2  # const
    lastmaxix = [1, 1]
    lastminix = [1, 1]
    gainloss = zeros(Float32, 2, size(regressions,1))
    for ix in Iterators.drop(eachindex(regressions), 1)  # 2:size(regressions,1)
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
function polynomialfeatures!(features::AbstractDataFrame, polynomialconnect::Int)
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
    for ix in Iterators.drop(eachindex(regressions), 1)  # 2:size(regressions,1)
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

#endregion FeatureUtilities

#region Features001
regressionwindows001 = Dict("5m" => 5, "15m" => 15, "1h" => 1*60, "4h" => 4*60, "12h" => 12*60, "1d" => 24*60, "3d" => 3*24*60, "10d" => 10*24*60)

sortedregressionwindowkeys001 = [d[1] for d in sort(collect(regressionwindows001), by = x -> x[2])]

" Features001 is no longer used and replaced by Features002 "
struct Features001
    fdf::AbstractDataFrame
    featuremask::Vector{String}
    ohlcv::OhlcvData
end


"""
Properties at various rolling windows calculated on df data with ohlcv + pilot columns:

- per regression window
    - gradient of pivot regression line
    - standard deviation of regression normalized distribution
    - difference of last pivot price to regression line

"""
function getfeatures001(pivot::Vector{<:AbstractFloat})
    featuremask::Vector{String} = []
    fdf = DataFrame()
    for wk in sortedregressionwindowkeys001  # wk = window key
        ws = regressionwindows001[wk]  # ws = window size in minutes
        fdf[:, "regry$wk"], fdf[:, "grad$wk"] = rollingregression(pivot, ws)
        stdnotrend, _, fdf[:, "regrdiff$wk"] = rollingregressionstd(pivot, fdf[!, "regry$wk"], fdf[!, "grad$wk"], ws)
        fdf[:, "2xstd$wk"] = stdnotrend .* 2.0  # also to be used as normalization reference
        append!(featuremask, ["grad$wk", "regrdiff$wk", "2xstd$wk"])
    end
    return fdf, featuremask
end

"""
- getfeatures001(pivot::Vector{Float32})
- ration 4hour/9day volume to detect mid term rising volume
- ration 5minute/4hour volume to detect short term rising volume
"""
function getfeatures001(ohlcvdf::AbstractDataFrame)
    fdf, featuremask = getfeatures001(ohlcvdf.pivot)
    fdf[:, "4h/9dvol"] = relativevolume(ohlcvdf[!, :basevolume], 4*60, 9*24*60)
    fdf[:, "5m/4hvol"] = relativevolume(ohlcvdf[!, :basevolume], 5, 4*60)
    append!(featuremask, ["4h/9dvol", "5m/4hvol"])
    return fdf, featuremask
end

function getfeatures001(ohlcv::OhlcvData)
    ohlcvdf = ohlcv.df
    fdf, featuremask = getfeatures001(ohlcvdf)
    return Features001(fdf, featuremask, ohlcv)
end

"""
 Update featureset001 to match ohlcv.
 constraint: either featureset001 is nothing or empty or it has to be appended
 but no need to add before an existing start
"""
function getfeatures!(features::Features001, ohlcv::OhlcvData)
    # TODO
    @error "to be done"
    return features
end


#endregion Features001

#region Features005

# Features005 is based on Features004. To reuse the F4 cache files, those features are always calculated to keep the saved cache complete.

regressionwindows005 = [5, 15, 60, 4*60, 12*60, 24*60, 3*24*60, 10*24*60]
regressionproperties = ["grad", "std", "regry"]
savedregressionwindows005 = [60, 4*60, 12*60, 24*60, 3*24*60, 10*24*60] # == Features.regressionwindows004
savedregressionproperties = ["grad", "std", "regry"]
minmaxproperties = ["mindist", "maxdist"]
regressionfeaturespec005(regrwindows=regressionwindows005, regrprops=regressionproperties) = [join(["rw", rw, rp], "_") for rw in regrwindows for rp in regrprops]
minmaxfeaturespec005(mmwindows=regressionwindows005, mmprops=minmaxproperties) = [join(["mm", mmw, rp], "_") for mmw in mmwindows for rp in mmprops]
volumefeaturespec005(shortlongtuplevector=[(5, 4*60), (60, 3*24*60)]) = [join(["rv", sw, lw], "_") for (sw, lw) in shortlongtuplevector]

"Concatenates the 3 feature specifications. In case one of the specs is not requested then add an empty vector [] instead"
featurespecification005(regressionfeaturespec=regressionfeaturespec005(), minmaxfeaturespec=minmaxfeaturespec005(), volumefeaturespec=volumefeaturespec005()) = vcat(regressionfeaturespec, minmaxfeaturespec, volumefeaturespec)

"Checks validity of requested configs and returns dataframe with analysis"
function configdf(requestedconfigs, requiredminutes)
    savecols = names(emptycachef005())
    df = DataFrame()
    allconfigs = union(requestedconfigs, savecols)
    configix = 1
    while configix <= length(allconfigs)
        config = allconfigs[configix]
        cfg = split(config, "_")
        valid = false
        first = cfg[1]
        second = third = ""
        if !(first in ["opentime", "rw", "mm", "rv"])
            @error "skipping $config with unknown feature type $(first) (requestedconfigs=$(requestedconfigs))"
        else
            if (first in ["rw", "mm", "rv"]) && length(cfg) != 3
                @error "$config has not 3 elements cfg=$cfg (requestedconfigs=$(requestedconfigs))"
            elseif (first in ["opentime"]) && length(cfg) != 1
                @error "$config has not 1 element cfg=$cfg (requestedconfigs=$(requestedconfigs))"
            else
                if first == "rw"
                    second = parse(Int, cfg[2])
                    if second in regressionwindows005
                        third = cfg[3]
                        if third in regressionproperties
                            if third in ["std", "regry"]
                                grd = join(["rw", string(cfg[2]), "grad"], "_")
                                if !(grd in allconfigs) push!(allconfigs, grd) end
                            end
                            if third in ["std", "grad"]
                                rgr = join(["rw", string(cfg[2]), "regry"], "_")
                                if !(rgr in allconfigs) push!(allconfigs, rgr) end
                            end
                            valid = true
                        else
                            @error "skipping $config due to regression property=$(third) not in $regressionproperties"
                        end
                    else
                        @error "skipping $config due to regression window=$(second) not in $regressionwindows005 (requestedconfigs=$(requestedconfigs))"
                    end
                elseif first == "mm"
                    second = parse(Int, cfg[2])
                    if 0 < second <= requiredminutes
                        third = cfg[3]
                        if third in minmaxproperties
                            valid = true
                        else
                            @error "skipping $config due to min/max property=$(third) not in $minmaxproperties (requestedconfigs=$(requestedconfigs))"
                        end
                    else
                        @error "skipping $config due to minmax window=$(second) not within [1:$regressionwindows005] (requestedconfigs=$(requestedconfigs))"
                    end
                elseif first == "rv"
                    second = parse(Int, cfg[2])
                    third = parse(Int, cfg[3])
                    if 0 < second <= requiredminutes
                        if 0 < third <= requiredminutes
                            if second < third
                                valid = true
                            else
                                @error "skipping $config due to wrong short_long sequence: shortwin=$second < longwin=$third (requestedconfigs=$(requestedconfigs))"
                            end
                        else
                            @error "skipping $config due to longwin=$(third) not within [1:$regressionwindows005] (requestedconfigs=$(requestedconfigs))"
                        end
                    else
                        @error "skipping $config due to shortwin=$(second) not within [1:$regressionwindows005] (requestedconfigs=$(requestedconfigs))"
                    end
                elseif first == "opentime"
                    valid = true
                else
                    @error "$config contains no known feature"
                end
            end
        end
        push!(df, (config = config, first = first, second = second, third = third, valid = valid, save = config in savecols, requested = config in requestedconfigs), promote=true)
        configix += 1
    end
    # (verbosity >= 3) && println("Features005 cfgdf: $df")
    return df
end


"""
Features005 provides a configurable feature set with the following config choices provided as vector of Strings with elements:
- "rw" followed by *minutes* (e.g. "rw_60") to denote a 60 minute regression window with the following `_` concatenated options: "grad", "std", "regry".
  Regression window minutes are constraint by the set of *regressionwindows005*
    - grad = gradient of regression line
    - std = standard deviation of regression line
    - regry = y position of regression line end
- "mm" followed by *minutes* (e.g. "mm_60") to denote a 60 minute min/max window with the following `_` concatenated options: "maxdist", "mindist".
    - maxdist = relative distance to max value of minmaxwindows
    - mindist = relative distance to min value of minmaxwindows
- "rv" to denote a relative median volume with the following `_` concatenated option: shortwindow followed by longwindow in minutes,
  e.g. "rv_5_60" provides 1 vector with the ratio of the volume median of shortwindow / longwindow
    - shortwindow = minutes of short window median
    - longwindow = minutes of long window median
Each tuple element will result in a feature vector of that name

Example:
```
Features005(["rw_60_regry", "rw_60_grad", "mm_30_maxdist", "mm_15_mindist", "rv_5_60"])
```

Struct property notes:
- cfgdf: dataframe of the validated configuration
  - configcols: vector of column names of the requested configuration, which is subset of names(f5.fdf)
- ohlcv: reference ohlcv data that are used as basis for feature calculation
- fdf: features data frame with union of configured columns and Features004 columns

"""
mutable struct Features005 <: AbstractFeatures
    requiredminutes  # determined by requiredminutes or firstix whatever is larger
    cfgdf::DataFrame
    ohlcv::Union{Nothing, Ohlcv.OhlcvData}
    fdf::Union{Nothing, DataFrame}
    latestloadeddt::Union{Nothing, DateTime}
    complete::Bool # init set it to true, timerangecut to set it to false
    ohlcvoffset::Union{Nothing, Int}
    function Features005(config::Vector)
        requiredminutes = maximum(regressionwindows005)
        @assert length(config) > 0
        cfgdf = configdf(config, requiredminutes)
        @assert all(cfgdf[!, :valid]) "invalid config: $cfgdf"
        f5 = new(requiredminutes, cfgdf, nothing, nothing, nothing, false, nothing)
        return f5
    end
end

function setbase!(f5::Features005, ohlcv::Ohlcv.OhlcvData; usecache=false)
    f5.ohlcv = ohlcv
    f5.fdf = usecache ? read!(f5) : emptycachef005()
    f5.latestloadeddt = size(f5.fdf, 1) > 0 ? f5.fdf[end, :opentime] : nothing
    f5.complete = true
    f5.ohlcvoffset = nothing
    featureoffset!(f5)
    supplement!(f5)
end

function removebase!(f5::Features005)
    f5.ohlcv = nothing
    f5.fdf = nothing
    f5.latestloadeddt = nothing
    f5.complete = false
    f5.ohlcvoffset = nothing
end


ohlcvix(f5::Features005, featureix) = featureix + f5.ohlcvoffset
featureix(f5::Features005, ohlcvix) = ohlcvix - f5.ohlcvoffset

function featureoffset!(f5::Features005)
    odf = Ohlcv.dataframe(f5.ohlcv)
    if (size(odf, 1) > 0) && (size(f5.fdf, 1)> 0)
        f5.ohlcvoffset = Ohlcv.rowix(odf[!,:opentime], f5.fdf[begin, :opentime]) - 1
        @assert odf[ohlcvix(f5, firstindex(f5.fdf[!, :opentime])), :opentime] == f5.fdf[begin, :opentime] "f5.ohlcvoffset=$(f5.ohlcvoffset) odf[ohlcvix(f5, 1), :opentime]=$(odf[ohlcvix(f5, 1), :opentime]) f5.fdf[begin, :opentime]=$(f5.fdf[begin, :opentime])"
    else
        f5.ohlcvoffset = nothing
    end
end

ohlcvdfview(f5::Features005) = isnothing(f5.ohlcv) ? nothing : view(Ohlcv.dataframe(f5.ohlcv), Ohlcv.rowix(f5.fdf[!, :opentime], f5.fdf[begin, :opentime]):Ohlcv.rowix(f5.fdf[!, :opentime], f5.fdf[end, :opentime]), :)
ohlcv(f5::Features005) = f5.ohlcv

requiredminutes(f5::Features005) = f5.requiredminutes

function emptycachef005()
    df = DataFrame()
    df[:, "opentime"] = DateTime[]
    for rw in savedregressionwindows005   # share data with Features004 and only regressionwindows004 subset of regressionwindows005 in cache
        for rp in savedregressionproperties
            df[:, join(["rw", string(rw), rp], "_")] = Float32[]
        end
    end
    return df
end
describe(f5::Features005)::String = "Features005($(uppercase(f5.ohlcv.base)))" * join(names(f5.fdf), "+")
mnemonic(f5::Features005) = uppercase(f5.ohlcv.base) * "_" * uppercase(f5.ohlcv.quotecoin) * "_" * "_F4"  #* share data with Features004

function file(f5::Features005)
    mnm = mnemonic(f5)
    filename = EnvConfig.datafile(mnm, "Features004")  # share data with Features004
    if isdir(filename)
        return (filename=filename, existing=true)
    else
        return (filename=filename, existing=false)
    end
end

function write(f5::Features005)
    if EnvConfig.configmode == production
        if !isnothing(f5.fdf) && !isnothing(f5.ohlcv)
            fn = file(f5)
            if !f5.complete || (size(f5.fdf, 1) == 0)
                (verbosity >= 3) && println("$(EnvConfig.now()) f5 not written due to incomplete data=$(f5.complete) or empty data: size(f5.fdf)=$(size(f5.fdf))")
                return
            end
            if !isnothing(f5.latestloadeddt) && (f5.latestloadeddt >= f5.fdf[end, :opentime])
                (verbosity >= 3) && println("$(EnvConfig.now()) f5 not written due to already stored data:$(f5.latestloadeddt) >= $(f5.fdf[end, :opentime])")
                return
            end
            df = DataFrame()
            for configrow in eachrow(f5.cfgdf)
                if configrow.save
                    savecol = configrow.first == "opentime" ? "opentime" : join([string(configrow.second), string(configrow.third)], "_")
                    df[:, savecol] = f5.fdf[!, configrow.config]
                end
            end
            try
                JDF.savejdf(fn.filename, df[!, :])
                (verbosity >= 2) && println("$(EnvConfig.now()) saved F5 data of $(f5.ohlcv.base) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows to $(fn.filename)")
                f5.latestloadeddt = size(f5.fdf, 1) > 0 ? f5.fdf[end, :opentime] : nothing
            catch e
                Logging.@error "exception $e detected when writing $(fn.filename)"
            end
        else
            (verbosity >= 2) && isnothing(f5.fdf) && println("no features found in F5 - nothing to write")
            (verbosity >= 2) && isnothing(f5.ohlcv) && println("no ohlcv found in F5 - missing base info required to write")
        end
    else
        (verbosity >= 2) && println("no F5.write() if EnvConfig.configmode != production to prevent mixing testnet data with real canned data")
    end
end

function read!(f5::Features005)::DataFrame
    if isnothing(f5.ohlcv)
        (verbosity >= 2)  && println("no ohlcv found in F5 - missing base info required to read")
        return emptycachef005()
    end
    fn = file(f5)
    # try
    f5.complete = true
    if fn.existing
            (verbosity >= 3) && println("$(EnvConfig.now()) start loading F5 data of $(f5.ohlcv.base) from $(fn.filename)")
            df = DataFrame(JDF.loadjdf(fn.filename))
            (verbosity >= 2) && println("$(EnvConfig.now()) loaded F5 data of $(f5.ohlcv.base) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows from $(fn.filename)")
            if size(df, 1) == 0
                (verbosity >= 2) && println("$(EnvConfig.now()) no data loaded from $(fn.filename)")
                df = emptycachef005()
            else
                startix = Ohlcv.dataframe(f5.ohlcv)[begin, :opentime] <= df[begin, :opentime] <= Ohlcv.dataframe(f5.ohlcv)[end, :opentime] ? firstindex(df[!, :opentime]) : nothing
                endix = Ohlcv.dataframe(f5.ohlcv)[begin, :opentime] <= df[end, :opentime] <= Ohlcv.dataframe(f5.ohlcv)[end, :opentime] ? lastindex(df[!, :opentime]) : nothing
                if !isnothing(startix)
                    if isnothing(endix) # df[begin] is inside and df[end] is outside ohlcv time range
                        endix = Ohlcv.rowix(df[!, :opentime], Ohlcv.dataframe(f5.ohlcv)[end, :opentime])
                        df = df[startix:endix, :]
                    # else no df change
                    end
                else
                    if !isnothing(endix) # df[begin] is outside and df[end] is inside ohlcv time range
                        startix = Ohlcv.rowix(df[!, :opentime], Ohlcv.dataframe(f5.ohlcv)[begin, :opentime])
                        df = df[startix:endix, :]
                    else # both df[begin] and df[end] are outside ohlcv time range
                        df = emptycachef005()
                    end
                end
            end
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(fn.filename)")
            df = emptycachef005()
        end
        f5.fdf = df
        # (verbosity >= 3) && println("$(EnvConfig.now()) Features005.read! after loading and adapting $f5")
        # (verbosity >= 3) && println("$(EnvConfig.now()) Features005.read! names(df)=$(names(df))")
        # catch e
    #     Logging.@warn "exception $e detected"
    # end
    return df
end

function delete(f5::Features005)
    if isnothing(f5.ohlcv)
        (verbosity >= 2)  && println("no ohlcv found in F5 - missing base info required to delete file")
        return emptycachef005()
    end
    fn = file(f5)
    if fn.existing
        rm(fn.filename; force=true, recursive=true)
        (verbosity >= 2) && println("$(EnvConfig.now()) deleted $(fn.filename)")
    end
end


function supplement!(f5::Features005)
    if isnothing(f5.ohlcv)
        (verbosity >= 2)  && println("no ohlcv found in F5 - nothing to supplement")
        return emptycachef005()
    end
    # replace the existing fdf by a new one
    Ohlcv.pivot!(f5.ohlcv)
    odf = Ohlcv.dataframe(f5.ohlcv)
    reqmin = requiredminutes(f5) # all dataframes to start at the same time to make performance comparable and f5 handling easier
    fdf = DataFrame()
    if size(f5.fdf, 1) > 0
        odfstartix = Ohlcv.rowix(odf[!, "opentime"], f5.fdf[end, "opentime"]) + 1 # ohlcv index of last feature datetime
        odfstartix = odfstartix > lastindex(odf[!, "opentime"]) ? nothing : odfstartix  # nothing = most recent ohlcv data have matching features
        newlen = isnothing(odfstartix) ? 0 : lastindex(odf[!, "opentime"]) - odfstartix + 1
        cols = names(f5.fdf)
    else
        odfstartix = nothing
        newlen = length(odf[!, "opentime"]) - reqmin + 1
        cols = []
    end
    for configrow in eachrow(f5.cfgdf)
        if configrow.config in names(fdf)
            # (verbosity >= 3) && println("$configrow already processed together with other column")
            continue
        end
        if configrow.first == "rw"
            fc = f5.cfgdf
            regryrow = first(f5.cfgdf[(fc[!, :first] .== "rw") .&& (fc[!, :second] .== configrow.second) .&& (fc[!, :third] .== "regry"), :])
            gradrow = first(f5.cfgdf[(fc[!, :first] .== "rw") .&& (fc[!, :second] .== configrow.second) .&& (fc[!, :third] .== "grad"), :])
            win = configrow.second
            std = nothing
            if regryrow.config in cols
                if !isnothing(odfstartix)
                    regry, grad = rollingregression(odf[!, :pivot], win, odfstartix)
                    @assert length(regry) == newlen "length(regry)=$(length(regry)) != newlen=$newlen"
                    regry = vcat(f5.fdf[!, regryrow.config], regry)
                    grad = vcat(f5.fdf[!, gradrow.config], grad)
                else
                    regry = f5.fdf[!, regryrow.config]
                    grad = f5.fdf[!, gradrow.config]
                end
            else
                regry, grad = rollingregression(odf[!, :pivot], win, reqmin)
            end
            fdf[:, regryrow.config] = regry
            fdf[:, gradrow.config] = grad
            finddf = f5.cfgdf[(fc[!, :first] .== "rw") .&& (fc[!, :second] .== configrow.second) .&& (fc[!, :third] .== "std"), :]
            if size(finddf, 1) > 0
                stdrow = first(finddf)
                if stdrow.config in cols
                    if !isnothing(odfstartix)
                        std = rollingregressionstdmv([odf[!, :open], odf[!, :high], odf[!, :low], odf[!, :close]], regry, grad, win, odfstartix)
                        @assert length(std) == newlen "length(std)=$(length(std)) != newlen=$newlen"
                        std = vcat(f5.fdf[!, stdrow.config], std)
                    else
                        std = f5.fdf[!, stdrow.config]
                    end
                else
                    std = rollingregressionstdmv([odf[!, :open], odf[!, :high], odf[!, :low], odf[!, :close]], regry, grad, win, reqmin)
                end
                fdf[:, stdrow.config] = std
            end

        elseif configrow.first == "mm"
            win = configrow.second
            if (configrow.third == "mindist")
                if (configrow.config in cols)
                    if !isnothing(odfstartix)
                        minvec = RollingFunctions.rollmin(odf[odfstartix-win+1:end, :low], win)
                        mindist = (odf[odfstartix:end, :pivot] .- minvec) ./ minvec
                        @assert length(mindist) == newlen "length(mindist)=$(length(mindist)) != newlen=$newlen"
                        mindist = vcat(f5.fdf[!, configrow.config], mindist)
                    else
                        mindist = f5.fdf[!, configrow.config]
                    end
                else
                    minvec = RollingFunctions.rollmin(odf[reqmin-win+1:end, :low], win)
                    mindist = (odf[reqmin:end, :pivot] .- minvec) ./ minvec
                end
                fdf[:, configrow.config] = mindist
            end
            if (configrow.third == "maxdist")
                if (configrow.config in cols)
                    if !isnothing(odfstartix)
                        maxvec = RollingFunctions.rollmax(odf[odfstartix-win+1:end, :high], win)
                        maxdist = (odf[odfstartix:end, :pivot] .- maxvec) ./ maxvec
                        @assert length(maxdist) == newlen "length(maxdist)=$(length(maxdist)) != newlen=$newlen"
                        maxdist = vcat(f5.fdf[!, configrow.config], maxdist)
                    else
                        maxdist = f5.fdf[!, configrow.config]
                    end
                else
                    maxvec = RollingFunctions.rollmax(odf[reqmin-win+1:end, :high], win)
                    maxdist = (odf[reqmin:end, :pivot] .- maxvec) ./ maxvec
                end
                fdf[:, configrow.config] = maxdist
            end
        elseif configrow.first == "rv"
            shortwin = configrow.second
            longwin = configrow.third
            if configrow.config in cols
                if !isnothing(odfstartix)
                    rvvec = relativevolume(odf[odfstartix-longwin+1:end, :pivot], shortwin, longwin)[end-newlen+1:end]
                    @assert length(rvvec) == newlen "length(rvvec)=$(length(rvvec)) != newlen=$newlen"
                    rvvec = vcat(f5.fdf[!, configrow.config], rvvec)
                else
                    rvvec = f5.fdf[!, configrow.config]
                end
            else
                rvvec = relativevolume(odf[reqmin-longwin+1:end, :pivot], shortwin, longwin)[end-lastindex(odf[!, "opentime"])+reqmin:end]
            end
            fdf[:, configrow.config] = rvvec
        elseif configrow.first == "opentime"
            if !isnothing(odfstartix)
                @assert configrow.config in cols "configrow=$configrow in cols=$cols"
                ot = odf[odfstartix:end, :opentime]
                @assert length(ot) == newlen "length(ot)=$(length(ot)) != newlen=$newlen"
                ot = vcat(f5.fdf[!, configrow.config], ot)
            else
                ot = odf[reqmin:end, configrow.config]
            end
            fdf[:, configrow.config] = ot
        else
            @error "unexpected configrow=$(string(configrow))"
        end
    end
    f5.fdf = fdf
    featureoffset!(f5)
end

function timerangecut!(f5::Features005)
    if isnothing(f5.ohlcv) || isnothing(f5.fdf)
        (verbosity >= 2) && isnothing(f5.fdf) && println("no features found in F5 - no time range to cut")
        (verbosity >= 2) && isnothing(f5.ohlcv) && println("no ohlcv found in F5 - missing ohlcv reference to cut time range")
        return
    end
    startdt = Ohlcv.dataframe(f5.ohlcv)[begin, :opentime]
    startix = Ohlcv.rowix(f5.fdf[!, :opentime], startdt)
    enddt = Ohlcv.dataframe(f5.ohlcv)[end, :opentime]
    endix = Ohlcv.rowix(f5.fdf[!, :opentime], enddt)
    f5.fdf = f5.fdf[startix:endix, :]
    f5.complete = false
    featureoffset!(f5)
end

function features(f5::Features005, firstix::Integer=firstindex(f5.fdf[!, :opentime]), lastix::Integer=lastindex(f5.fdf[!, :opentime]))
    if isnothing(f5.fdf)
        (verbosity >= 2) && isnothing(f5.fdf) && println("no features found in F5")
        return nothing
    end
    @assert !isnothing(firstix) && (firstindex(f5.fdf[!, :opentime]) <= firstix <= lastix <= lastindex(f5.fdf[!, :opentime])) "firstindex=$(firstindex(f5.fdf[!, :opentime])) <= firstix=$firstix <= lastix=$lastix <= lastindex=$(lastindex(f5.fdf[!, :opentime]))"
    cols = f5.cfgdf[f5.cfgdf[!, :requested], :config]
    return f5.fdf[firstix:lastix, cols]
end

function features(f5::Features005, startdt::DateTime, enddt::DateTime)
    if isnothing(f5.fdf)
        (verbosity >= 2) && isnothing(f5.fdf) && println("no features found in F5")
        return nothing
    end
    return features(f5, Ohlcv.rowix(f5.fdf[!, :opentime], startdt), Ohlcv.rowix(f5.fdf[!, :opentime], enddt))
end

grad(f5::Features005, regrminutes) = f5.fdf[!, join(["rw", regrminutes, "grad"], "_")]
regry(f5::Features005, regrminutes) = f5.fdf[!, join(["rw", regrminutes, "regry"], "_")]
std(f5::Features005, regrminutes) = f5.fdf[!, join(["rw", regrminutes, "std"], "_")]
mindist(f5::Features005, mmminutes) = f5.fdf[!, join(["mm", mmminutes, "mindist"], "_")]
maxdist(f5::Features005, mmminutes) = f5.fdf[!, join(["mm", mmminutes, "maxdist"], "_")]
relativevolume(f5::Features005, shortlongtuple) = f5.fdf[!, join(["rv", shortlongtuple[1], shortlongtuple[2]], "_")]
regrwindows(f5::Features005) = keys(f4.rw)
opentime(f5::Features005) = isnothing(f5.fdf) ? [] : f5.fdf[!, :opentime]

function Base.show(io::IO, f5::Features005)
    println(io, "Features005 requiredminutes=$(f5.requiredminutes) base=$(f5.ohlcv.base), size(fdf)=$(size(f5.fdf)) $(size(f5.fdf, 1) > 0 ? "from $(f5.fdf[begin, :opentime]) to $(f5.fdf[end, :opentime]) " : "no time range ")requiredminutes=$(f5.requiredminutes) latestloadeddt=$(f5.latestloadeddt) complete=$(f5.complete) ohlcvoffset=$(f5.ohlcvoffset)")
    # (verbosity >= 3) && println(io, "Features005 cfgdf=$(f5.cfgdf)")
    # (verbosity >= 2) && println(io, "Features005 config=$(f5.cfgdf[!, :config])")
    println(io, "Features005 ohlcv=$(f5.ohlcv)")
end

#endregion Features005

#region Features004
"Features004 is a simplified subset of Features002 without regression extremes and relative volume but with save and read functions and implemented as DataFrame"

regressionwindows004 = [60, 4*60, 12*60, 24*60, 3*24*60, 10*24*60]
regressionwindows004dict = Dict("1h" => 1*60, "4h" => 4*60, "12h" => 12*60, "1d" => 24*60, "3d" => 3*24*60, "10d" => 10*24*60)

"""
Provides per regressionwindow gradient, regression line price, standard deviation.
"""
mutable struct Features004 <: AbstractFeatures
    basecoin::String
    quotecoin::String
    ohlcvoffset
    rw::Dict{Integer, AbstractDataFrame}  # keys: regression window in minutes, values: dataframe with columns :opentime, :regry, :grad, :std
    latestloadeddt  # nothing or latest DateTime of loaded data
    # opentime::Vector{DateTime}
    # grad::Vector{Float32} # rolling regression gradients; length == ohlcv - requiredminutes
    # regry::Vector{Float32}  # rolling regression price; length == ohlcv - requiredminutes
    # std::Vector{Float32}  # standard deviation of regression window; length == ohlcv - requiredminutes
    Features004(basecoin::String, quotecoin::String) = new(basecoin, quotecoin, nothing, Dict(), nothing)
end

"Provides Features004 of the given ohlcv within the requested time range. Canned data will be read and supplemented with calculated data."
function Features004(ohlcv; firstix=firstindex(ohlcv.df.opentime), lastix=lastindex(ohlcv.df.opentime), regrwindows=regressionwindows004, usecache=false)::Union{Nothing, Features004}
    startix = maxregrwindow = maximum(regrwindows)  # all dataframes to start at the same time to make performance comparable and f4 handling easier
    f4 = Features004(ohlcv.base, ohlcv.quotecoin)
    df = Ohlcv.dataframe(ohlcv)
    if !(firstindex(df[!, :opentime]) <= startix <= lastix <= lastindex(df[!, :opentime])) || ((lastix - (max(firstix, maxregrwindow)-maxregrwindow)) < maxregrwindow)
        (verbosity >= 2) && @warn "$(ohlcv.base): $(firstindex(df[!, :opentime])) <= $startix <= $lastix <= $(lastindex(df[!, :opentime])); size(dfv, 1)=$((lastix - (max(firstix, maxregrwindow)-maxregrwindow))) < maxregrwindow=$maxregrwindow"
        return nothing
    end
    dfv = view(df, (max(firstix, maxregrwindow)-maxregrwindow+1):lastix, :)
    if usecache
        f4 = read!(f4, dfv[startix, :opentime], dfv[end, :opentime])
    end
    for window in regrwindows
        if !(window in keys(f4.rw))
            f4.rw[window] = DataFrame(opentime=DateTime[], regry=Float32[], grad=Float32[], std=Float32[])
        end
    end
    return isnothing(supplement!(f4, ohlcv; firstix=firstix, lastix=lastix)) ? nothing : f4
end

#TODO add regression test
mnemonic(f4::Features004) = uppercase(f4.basecoin) * "_" * uppercase(f4.quotecoin) * "_" * "_F4"
ohlcvix(f4::Features004, featureix) = featureix + f4.ohlcvoffset
featureix(f4::Features004, ohlcvix) = ohlcvix - f4.ohlcvoffset

function consistent(f4::Features004, ohlcv::Ohlcv.OhlcvData)
    checkok = true
    if ohlcv.base != f4.basecoin
        @warn "bases of ohlcv=$(ohlcv.base) != f4=$(f4.basecoin)"
        checkok = false
    end
    for (rw, rwdf) in f4.rw
        if rwdf[end,:opentime] != ohlcv.df[end, :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][end,:opentime]=$(rwdf[end,:opentime]) != ohlcv.df[end, :opentime]=$(ohlcv.df[end, :opentime])"
            checkok = false
        end
        if rwdf[begin,:opentime] - Minute(requiredminutes(f4)-1) != ohlcv.df[begin, :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][begin,:opentime]-requiredminutes(f4)+1=$(rwdf[begin,:opentime]-Minute(requiredminutes(f4)-1)) != ohlcv.df[begin, :opentime]=$(ohlcv.df[begin, :opentime])"
            checkok = false
        end
        if isnothing(f4.ohlcvoffset)
            @warn "isnothing(f4.ohlcvoffset) of $(ohlcv.base) "
            checkok = false
        elseif rwdf[begin,:opentime] != ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime]
            @warn "f4 of $(ohlcv.base) rw[$rw][begin,:opentime]=$(rwdf[begin,:opentime]) != ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime]=$(ohlcv.df[ohlcvix(f4, firstindex(rwdf[!,:opentime])), :opentime])"
            checkok = false
        end
    end
    startequal = all([rwdf[begin,:opentime] == first(values(f4.rw))[begin,:opentime] for (rw, rwdf) in f4.rw])
    if !startequal
        @warn "different F4 start dates: $([(regr=rw, startdt=rwdf[begin,:opentime]) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    endequal = all([rwdf[end,:opentime] == first(values(f4.rw))[end,:opentime] for (rw, rwdf) in f4.rw])
    if !endequal
        @warn "different F4 end dates: $([(regr=rw, enddt=rwdf[end,:opentime]) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    lengthequal = all([size(rwdf, 1) == size(first(values(f4.rw)), 1) for (rw, rwdf) in f4.rw])
    if !lengthequal
        @warn "different F4 data length: $([(regr=rw, length=size(rwdf, 1)) for (rw, rwdf) in f4.rw])"
        checkok = false
    end
    return checkok
end

function featureoffset!(f4::Features004, ohlcv::Ohlcv.OhlcvData)
    f4.ohlcvoffset = nothing
    if (length(f4.rw) > 0) && (size(first(values(f4.rw)), 1) > 0) && (size(ohlcv.df, 1) > 0)
        fix = nothing
        f4df = first(values(f4.rw))
        # fix = (ohlcv.df[begin,:opentime] <= f4df[begin, :opentime]) && (ohlcv.df[end,:opentime] >= f4df[begin, :opentime]) ? firstindex(f4df[!, :opentime]) : (ohlcv.df[begin,:opentime] <= f4df[end, :opentime] && ohlcv.df[end,:opentime] >= f4df[end, :opentime] ?  lastindex(f4df[!, :opentime]) : nothing)
        if ohlcv.df[begin,:opentime] <= f4df[begin, :opentime] <= ohlcv.df[end,:opentime]
            fix = firstindex(f4df[!, :opentime])
        elseif ohlcv.df[begin,:opentime] <= f4df[end, :opentime] <= ohlcv.df[end,:opentime]
            fix = lastindex(f4df[!, :opentime])
        end
        if !isnothing(fix)
            oix = Ohlcv.rowix(ohlcv.df[!,:opentime], f4df[fix, :opentime])
            if f4df[fix, :opentime] == ohlcv.df[oix, :opentime]
                f4.ohlcvoffset = oix - fix
            end
        else
            (verbosity >= 3) && @warn "could not calc $(ohlcv.base) f4offset ohlcv.begin=$(ohlcv.df[begin,:opentime]), ohlcv.end=$(ohlcv.df[end,:opentime]), f4df.begin=$(f4df[begin, :opentime]), f4df.end=$(f4df[end, :opentime])"
        end
    end
    return f4.ohlcvoffset
end

function _equaltimes(f4)
    times = [(df[begin, :opentime], size(df, 1), df[end, :opentime]) for df in values(f4.rw)]
    if length(times) > 0
        if all([times[1] == t for t in times])
            return true
        else
            times = [(regr=regr, first=df[begin, :opentime], length=size(df, 1), last=df[end, :opentime]) for (regr, df) in f4.rw]
            @warn "F4 dataframes not equal: $times"
            return false
        end
    else
        @warn "F4 dataframes missing"
        return false
    end
end

function _join(f4)
    df = DataFrame()
    for (regr, rdf) in f4.rw
        for cname in names(rdf)
            if cname == "opentime"
                df[:, cname] = rdf[!, cname]
            else
                df[:, join([string(regr), cname], "_")] = rdf[!, cname]
            end
        end
    end
    return df
end

function timerangecut!(f4::Features004, startdt, enddt)
    (length(f4.rw) == 0) && @warn "empty f4 for $(f4.basecoin)"
    for (regr, rdf) in f4.rw
        if isnothing(rdf) || (size(rdf, 1) == 0)
            @warn "unexpected missing f4 data $f4"
            return
        end
        startdt = isnothing(startdt) ? rdf[begin, :opentime] : startdt
        startix = Ohlcv.rowix(rdf[!, :opentime], startdt)
        enddt = isnothing(enddt) ? rdf[end, :opentime] : enddt
        endix = Ohlcv.rowix(rdf[!, :opentime], enddt)
        f4.rw[regr] = rdf[startix:endix, :]
        # println("startdt=$startdt enddt=$enddt size(rdf)=$(size(rdf)) rdf=$(describe(rdf, :all)) ")
        # if !isnothing(startdt) && !isnothing(enddt)
        #     subset!(rdf, :opentime => t -> floor(startdt, Minute(1)) .<= t .<= floor(enddt, Minute(1)))
        # elseif !isnothing(startdt)
        #     subset!(rdf, :opentime => t -> floor(startdt, Minute(1)) .<= t)
        # elseif !isnothing(enddt)
        #     subset!(rdf, :opentime => t -> t .<= floor(enddt, Minute(1)))
        # end
    end
end

function _split!(f4, df)
    @assert length(f4.rw) == 0
    ot = nothing
    for cname in names(df)
        if cname == "opentime"
            ot = df[!, cname]
        else
            cnamevec = split(cname, "_")
            if length(cnamevec) != 2
                @error "unexpected f4.rw dataframe column name: $cnamevec"
            end
            regr = parse(Int, cnamevec[1])
            if !(regr in keys(f4.rw))
                f4.rw[regr] = DataFrame()
            end
            f4.rw[regr][:, cnamevec[2]] = df[!, cname]
        end
    end
    for (regr, rdf) in f4.rw
        rdf[:, "opentime"] = ot
    end
    return f4
end

function file(f4::Features004)
    mnm = mnemonic(f4)
    filename = EnvConfig.datafile(mnm, "Features004")
    if isdir(filename)
        return (filename=filename, existing=true)
    else
        return (filename=filename, existing=false)
    end
end

function write(f4::Features004)
    if EnvConfig.configmode == production
        @assert _equaltimes(f4)
        df = _join(f4)
        if !isnothing(f4.latestloadeddt) && (f4.latestloadeddt >= df[end, :opentime])
            (verbosity >= 3) && println("$(EnvConfig.now()) F4 not written due to missing supplementations of already stored data")
            return
        end
        fn = file(f4)
        try
            JDF.savejdf(fn.filename, df[!, :])
            (verbosity >= 2) && println("$(EnvConfig.now()) saved F4 data of $(f4.basecoin) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows to $(fn.filename)")
        catch e
            Logging.@error "exception $e detected when writing $(fn.filename)"
        end
    else
        (verbosity >= 2) && println("no F4.write() if EnvConfig.configmode != production to prevent mixing testnet data with real canned data")
    end
end

read!(f4::Features004)::Features004 = read!(f4, nothing, nothing)

function read!(f4::Features004, startdt, enddt)::Features004
    fn = file(f4)
    # try
        if fn.existing
            (verbosity >= 3) && println("$(EnvConfig.now()) start loading F4 data of $(f4.basecoin) from $(fn.filename)")
            df = DataFrame(JDF.loadjdf(fn.filename))
            startdt = isnothing(startdt) ? df[begin, :opentime] : startdt
            enddt = isnothing(enddt) ? df[end, :opentime] : enddt
            (verbosity >= 2) && println("$(EnvConfig.now()) loaded F4 data of $(f4.basecoin) from $(df[1, :opentime]) until $(df[end, :opentime]) with $(size(df, 1)) rows from $(fn.filename)")
            if (size(df, 1) > 0) && (startdt <= df[end, :opentime]) && (enddt >= df[begin, :opentime])
                f4.latestloadeddt = df[end, :opentime]
                startix = Ohlcv.rowix(df[!, :opentime], startdt)
                startix = df[startix, :opentime] < startdt ? min(lastindex(df[!, :opentime]), startix+1) : startix
                endix = Ohlcv.rowix(df[!, :opentime], enddt)
                endix = df[endix, :opentime] > enddt ? max(firstindex(df[!, :opentime]), endix-1) : endix
                df = df[startix:endix, :]
                f4 = _split!(f4, df)
            end
        else
            (verbosity >= 2) && println("$(EnvConfig.now()) no data found for $(fn.filename)")
        end
    # catch e
    #     Logging.@warn "exception $e detected"
    # end
    return f4
end

function delete(f4::Features004)
    fn = file(f4)
    if fn.existing
        rm(fn.filename; force=true, recursive=true)
    end
end

mutable struct Features004Files
    filenames
    Features004Files() = new(nothing)
end

# function Base.iterate(of::OhlcvFiles, state=1)
# end

function Base.iterate(f4f::Features004Files, state=1)
    if isnothing(f4f.filenames)
        allff = readdir(EnvConfig.datafolderpath("Features004"), join=false, sort=false)
        fileixlist = findall(f -> endswith(f, "_F4.jdf"), allff)
        f4f.filenames = [allff[ix] for ix in fileixlist]
        if length(f4f.filenames) > 0
            state = firstindex(f4f.filenames)
        else
            return nothing
        end
    end
    if state > lastindex(f4f.filenames)
        return nothing
    end
    # fn = split(of.filenames[state], "/")[end]
    fnparts = split(f4f.filenames[state], "_")
    # return (basecoin=fnparts[1], quotecoin=fnparts[2], interval=fnparts[3]), state+1
    basecoin=fnparts[1]
    quotecoin=fnparts[2]
    f4 = Features.Features004(String(basecoin), String(quotecoin))
    read!(f4, nothing, nothing)
    return f4, state+1
end

"Supplements Features004 with the newest ohlcv datapoints, i.e. datapoints newer than last(f4)"
function supplement!(f4::Features004, ohlcv::Ohlcv.OhlcvData; firstix=firstindex(ohlcv.df[!, :opentime]), lastix=lastindex(ohlcv.df[!, :opentime]))
    usecache = (length(f4.rw) > 0) && (size(first(values(f4.rw)), 1) > 0)
    Ohlcv.pivot!(ohlcv)
    df = Ohlcv.dataframe(ohlcv)
    startix = maxregrwindow = maximum(regrwindows(f4))  # all dataframes to start at the same time to make performance comparable and f4 handling easier
    if !(firstindex(df[!, :opentime]) <= startix <= lastix <= lastindex(df[!, :opentime])) || ((lastix - (max(firstix, maxregrwindow)-maxregrwindow)) < maxregrwindow)
        @warn "$(firstindex(df[!, :opentime])) <= $startix <= $lastix <= $(lastindex(df[!, :opentime])); size(dfv, 1)=$((lastix - (max(firstix, maxregrwindow)-maxregrwindow))) < maxregrwindow=$maxregrwindow"
        return nothing
    end
    dfv = view(df, (max(firstix, maxregrwindow)-maxregrwindow+1):lastix, :)
    pivot = dfv.pivot
    startafterix = endbeforeix = nothing
    if usecache
        ot = dfv[!, "opentime"]
        otstored = first(values(f4.rw))[!, "opentime"]
        endbeforeix = Ohlcv.rowix(ot, otstored[begin]) - 1
        endbeforeix = endbeforeix < startix ? nothing : endbeforeix
        startafterix = Ohlcv.rowix(ot, otstored[end]) + 1
        startafterix = startafterix > lastindex(ot) ? nothing : startafterix
    end
    for window in regrwindows(f4)
        if usecache && (window in keys(f4.rw)) && (size(f4.rw[window], 1) > 0)
            if !isnothing(endbeforeix)
                (verbosity >= 3) && println("$(EnvConfig.now()) F4 endbeforeix=$endbeforeix with window=$window and startix=$startix for $(endbeforeix-startix+1) rows")
                regry, grad = rollingregression(pivot[begin:endbeforeix], window, startix)
                std = rollingregressionstdmv([dfv.open[begin:endbeforeix], dfv.high[begin:endbeforeix], dfv.low[begin:endbeforeix], dfv.close[begin:endbeforeix]], regry, grad, window, startix)
                dft = DataFrame(opentime=dfv[startix:endbeforeix, :opentime], regry=regry, grad=grad, std=std)
                prepend!(f4.rw[window], dft)
            end
            if !isnothing(startafterix)
                (verbosity >= 3) && println("$(EnvConfig.now()) F4 startafterix=$startafterix with window=$window for $(length(pivot)-startix+1) rows")
                regry, grad = rollingregression(pivot, window, startafterix)
                std = rollingregressionstdmv([dfv.open, dfv.high, dfv.low, dfv.close], regry, grad, window, startafterix)
                dft = DataFrame(opentime=dfv[startafterix:end, :opentime], regry=regry, grad=grad, std=std)
                append!(f4.rw[window], dft)
            end
        else
            (verbosity >= 3) && println("$(EnvConfig.now()) F4 full calc startix=$startix with window=$window for $(length(pivot)-startix+1) rows")
            regry, grad = rollingregression(pivot, window, startix)
            std = rollingregressionstdmv([dfv.open, dfv.high, dfv.low, dfv.close], regry, grad, window, startix)
            f4.rw[window] = DataFrame(opentime=dfv[startix:end, :opentime], regry=regry, grad=grad, std=std)
        end
    end
    return isnothing(featureoffset!(f4, ohlcv)) ? nothing : f4
end

requiredminutes(f4::Features004) = maximum(regrwindows(f4))

function Base.show(io::IO, f4::Features004)
    print(io, "Features004 base=$(f4.basecoin) quote=$(f4.quotecoin) $(["$regr:size=$(size(df))" for (regr,df) in f4.rw])")
    # for (key, value) in f4.rw
    #     println(io, "Features004 base=$(f4.basecoin), regr key: $key of size=$(size(value))")
    #     (verbosity >= 3) && println(io, describe(value, :first, :last, :min, :mean, :max, :nuniqueall, :nnonmissing, :nmissing, :eltype))
    # end
end

grad(f4::Features004, regrminutes) =     f4.rw[regrminutes][!, :grad]
regry(f4::Features004, regrminutes) =    f4.rw[regrminutes][!, :regry]
std(f4::Features004, regrminutes) =      f4.rw[regrminutes][!, :std]
opentime(f4::Features004, regrminutes) = f4.rw[regrminutes][!, :opentime]
opentime(f4::Features004) = first(f4.rw)[2][!, :opentime]  # opentime array from all rw members shall start and end equally
regrwindows(f4::Features004) = keys(f4.rw)

function features(f4::Features004, firstix, lastix)::AbstractDataFrame
    #TODO
    @error "to be implemented"
end

#endregion Features004

#region Features002
"Features002 is a feature set used in trading strategy"

regressionwindows002 = [5, 15, 60, 4*60, 12*60, 24*60, 3*24*60, 10*24*60]
relativevolumes002 = [(1, 60), (5, 4*60)]  # , (4*60, 9*24*60)] # TODO should be part of Features002

mutable struct Features002Regr
    grad::Vector{Float32} # rolling regression gradients; length == ohlcv - requiredminutes
    regry::Vector{Float32}  # rolling regression price; length == ohlcv - requiredminutes
    std::Vector{Float32}  # standard deviation of regression window; length == ohlcv - requiredminutes
    xtrmix::Vector{Int32}  # indices of extremes (<0 if min, >0 if max); length << ohlcv
    # medianstd::Vector{Float32}  # deprecated; median standard deviation over requiredminutes; length == ohlcv - requiredminutes
end

"""
Provides per regressionwindow gradient, regression line price, standard deviation, indices of regression extremes.
A dictionary of relative volumes shortwindow/longwindow that can be accessed by (shortwindow, longwindow) tuple as key.
These tuples are defined as module variable `relativevolumes002`.
Features are generated starting `firstix` until and including `lastix`. `requiredminutes` indicates the number of required minutes necessary to calculate the features.
"""
mutable struct Features002
    ohlcv::OhlcvData
    regr::Dict  # dict with regression minutes as key -> value is Features002Regr
    relvol::Dict  # dict of (shortwindowlength, longwindowlength) key and relative volume vector Vector{Float32} values
    update  # function to update features due to extended ohlcv
    firstix  # features start at firstix of ohlcv.df
    lastix  # features end at lastix of ohlcv.df
    requiredminutes
end

function Features002(ohlcv; firstix=firstindex(ohlcv.df.opentime), lastix=lastindex(ohlcv.df.opentime), regrwindows=regressionwindows002)::Features002
    df = Ohlcv.dataframe(ohlcv)
    reqmin = requiredminutes(regrwindows)
    @assert size(df, 1) >= reqmin "size(df, 1)=$(size(df, 1)) >= reqmin=$reqmin"
    lastix = lastix > lastindex(df, 1) ? lastindex(df, 1) : lastix
    firstix = firstix < (firstindex(df, 1) + reqmin - 1) ? (firstindex(df, 1) + reqmin - 1) : firstix
    @assert firstix <= lastix "firstix=$firstix <= lastix=$lastix ohlcv=$ohlcv"
    # maxfirstix = max((lastix - reqmin + 1), firstindex(df, 1))
    # firstix = firstix < reqmin ? reqmin : firstix
    # firstix = firstix > maxfirstix ? maxfirstix : firstix
    return Features002(ohlcv, getfeatures002regr(ohlcv, firstix, lastix, regrwindows), getrelvolumes002(ohlcv, firstix, lastix), getfeatures002!, firstix, lastix, reqmin)
end

requiredminutes(f2::Features002) = f2.requiredminutes
requiredminutes(regr::Vector{<:Integer}=regressionwindows002) = maximum(regr)

function Base.show(io::IO, features::Features002Regr)
    println(io, "- gradients: size=$(size(features.grad)) max=$(maximum(features.grad)) median=$(Statistics.median(features.grad)) min=$(minimum(features.grad))")
    println(io, "- regression y: size=$(size(features.regry)) max=$(maximum(features.regry)) median=$(Statistics.median(features.regry)) min=$(minimum(features.regry))")
    println(io, "- std deviation: size=$(size(features.std)) max=$(maximum(features.std)) median=$(Statistics.median(features.std)) min=$(minimum(features.std))")
    # print(io, "- median std deviation: size=$(size(features.medianstd)) max=$(maximum(features.medianstd)) median=$(Statistics.median(features.medianstd)) min=$(minimum(features.medianstd))")
    println(io, "- extreme indices: size=$(size(features.xtrmix)) #maxima=$(length(filter(r -> r > 0, features.xtrmix))) #minima=$(length(filter(r -> r < 0, features.xtrmix)))")
end

function Base.show(io::IO, features::Features002)
    println(io, "Features002 firstix=$(features.firstix), lastix=$(features.lastix)")
    println(io, features.ohlcv)
    for (key, value) in features.regr
        println(io, "regr key: $key")
        println(io, value)
    end
end

ohlcv(features::Features002) = features.ohlcv

featureix(f2::Features002, ohlcvix) = ohlcvix - f2.firstix + 1
ohlcvix(f2::Features002, featureix) = featureix + f2.firstix - 1

firstix(f2::Features002) = f2.firstix
grad(f2::Features002, regrminutes) =  f2.regr[regrminutes].grad[ohlcvix(f2, 1):end]
regry(f2::Features002, regrminutes) = f2.regr[regrminutes].regry[ohlcvix(f2, 1):end]
std(f2::Features002, regrminutes) =   f2.regr[regrminutes].std[ohlcvix(f2, 1):end]
ohlcvdataframe(f2::Features002) = Ohlcv.dataframe(f2.ohlcv)[ohlcvix(f2, 1):end, :]
opentime(f2::Features002) = Ohlcv.dataframe(f2.ohlcv)[ohlcvix(f2, 1):end, :opentime]
regrwindows(f2::Features002) = keys(f2.regr)

"""
In general don't call this function directly but via Feature002 constructor `Features.Features002(ohlcv)`
"""
function getfeatures002(ohlcv::OhlcvData, firstix=firstindex(ohlcv.df[!, :opentime]), lastix=lastindex(ohlcv.df[!, :opentime]))
    f2 = Features002(ohlcv; firstix=firstix, lastix=lastix)
    return f2
end

"""
Is called by Features002 constructor and returns a Dict of (short, long) => relative volume Float32 vector
"""
function getrelvolumes002(ohlcv::OhlcvData, firstix, lastix)::Dict
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    relvols = [relativevolume(ohlcvdf[((lastix-firstix) > l ? firstix : max((lastix-l+1), 1)):lastix, :basevolume], s, l) for (s, l) in relativevolumes002]
    rvd = Dict(zip(relativevolumes002, relvols))
    return rvd
end

"""
Is called by Features002 constructor and returns a Dict of regression calculatioins for all time windows of *regressionwindows002*
"""
function getfeatures002regr(ohlcv::OhlcvData, firstix, lastix, regrwindows)::Dict
    # println("getfeatures002 init")
    Ohlcv.pivot!(ohlcv)
    df = Ohlcv.dataframe(ohlcv)
    regr = Dict()
    for window in regrwindows
        dfv = view(df, (firstix-window+1):lastix, :)
        startix = window
        open = dfv.open
        high = dfv.high
        low = dfv.low
        close = dfv.close
        ymv = [open, high, low, close]
        pivot = dfv.pivot
        # println("firstix=$firstix lastix=$lastix size(df)=$(size(df)) size(dfv)=$(size(dfv))")
        @assert firstindex(dfv[!, :opentime]) <= startix <= lastindex(dfv[!, :opentime]) "$(firstindex(dfv[!, :opentime])) <= $startix <= $lastix <= $(lastindex(dfv[!, :opentime]))"
        regry, grad = rollingregression(pivot, window, startix)
        std = rollingregressionstdmv(ymv, regry, grad, window, startix) # startix is related to ymv
        xtrmix = regressionextremesix!(nothing, grad)
        regr[window] = Features002Regr(grad, regry, std, xtrmix)
    end
    return regr
end

"""
Appends features from firstix of ohlcv until and including lastix of ohlcv
but only if f2.lastix < lastindex(f2.ohlcv.pivot) because otherwise the features are already there

Features before firstix are deleted to save memory
"""
function getfeatures002!(f2::Features002, firstix=f2.firstix, lastix=lastindex(f2.ohlcv.df[!, :opentime]))
    # println("getfeatures002!")
    df = Ohlcv.dataframe(f2.ohlcv)
    if f2.lastix >= lastindex(df, 1)
        return f2
    end
    lastix = lastix > lastindex(df, 1) ? lastindex(df, 1) : lastix
    maxfirstix = max((lastix - requiredminutes(f2) + 1), firstindex(df, 1))
    firstix = firstix > maxfirstix ? maxfirstix : firstix

    pivot = df[!, :pivot][firstix:lastix]
    open = df.open[firstix:lastix]
    high = df.high[firstix:lastix]
    low = df.low[firstix:lastix]
    close = df.close[firstix:lastix]
    ymv = [open, high, low, close]
    if (f2.lastix >= firstix >= f2.firstix) && (lastix >= f2.lastix)
        firstfeatureix = featureix(f2, firstix)  # firstix - f2.firstix + 1
        for window in keys(f2.regr)
            fr = f2.regr[window]
            if firstfeatureix > 1  # cut start of available features to save memory
                fr.regry = fr.regry[firstfeatureix:end]
                fr.grad = fr.grad[firstfeatureix:end]
                fr.std = fr.std[firstfeatureix:end]
                ## fr.xtrmix is not cut but should not hurt due to number(extremes) << length(ohlcv)
                #! getrelvolumes002 volumes not yet cut
            end
            if lastix > f2.lastix
                regry, grad = rollingregression!(fr.regry, fr.grad, pivot, window)
                fr.std = rollingregressionstdmv!(fr.std, ymv, regry, grad, window)
                fr.xtrmix = regressionextremesix!(fr.xtrmix, fr.grad, fr.xtrmix[end]) # search forward from last extreme
                 #! getrelvolumes002 volumes not yet added
                else
                @warn "getfeatures002! nothing to add because lastix == f2.lastix"
            end
            @assert length(pivot) == length(fr.grad) == length(fr.regry) == length(fr.std)
        end
    else
        @info "getfeatures002! no reuse of previous calculations: f2.firstix=$(f2.firstix) f2.lastix=$(f2.lastix) firstix=$firstix lastix=$lastix"
        for window in keys(f2.regr)
            regry, grad = rollingregression(pivot, window)
            std = rollingregressionstdmv(ymv, regry, grad, window, 1)
            xtrmix = regressionextremesix!(nothing, grad, 1)
            f2.regr[window] = Features002Regr(grad, regry, std, xtrmix)
            @assert length(pivot) == length(grad) == length(regry) == length(std)
        end
    end
    f2.firstix = firstix
    f2.lastix = lastix

    for window in keys(f2.regr)
        @assert firstindex(f2.regr[window].std) == featureix(f2, firstix)
        @assert lastindex(f2.regr[window].std) == featureix(f2, lastix)
    end

    return f2
end

#endregion Features002

function getfeatures(ohlcv::OhlcvData)
    return getfeatures001(ohlcv)
    # return getfeatures002(ohlcv)
end

#region Features003

"""
- maxlookback = number of features (e.g. regression windows) to concatenate as feature vector
- features are availble from ohlcv index `f3.firstix = f3.f2.firstix + f3.maxlookback *` maxlength(feature) onwards
- maxlength(feature) = e.g. number of minutes of longest regression window
"""
mutable struct Features003
    f2::Features002
    maxlookback  # number of regression windows to concatenate as feature vector
    firstix
    regrwindow
    function Features003(f2, maxlookback)
        firstix = requiredminutes(f2, maxlookback)
        f3regrwindow = keys(featureslookback01)
        new(f2, maxlookback, firstix, f3regrwindow)
    end
end

(Features003)(ohlcv, regrwindows, maxlookback; firstix=firstindex(ohlcv.df.opentime), lastix=lastindex(ohlcv.df.opentime)) =
    Features003(Features002(ohlcv; firstix=firstix, lastix=lastix, regrwindows=regrwindows), maxlookback)

# requiredminutes(f2::Features002, maxlookback) = f2.firstix + requiredminutes(f2) * (maxlookback)
requiredminutes(f2::Features002, maxlookback) = f2.firstix + 24 * 60 * (maxlookback)  #! hard coded concatenation of regression windows up to 1 day
requiredminutes(f3::Features003) = requiredminutes(f3.f2, f3.maxlookback)

function Base.show(io::IO, features::Features003)
    println(io::IO, "Features003 lookback periods=$(features.maxlookback)")
    println(io::IO, features.f2)
end

featureix(f3::Features003, ohlcvix) = ohlcvix - f3.firstix + 1
ohlcvix(f3::Features003, featureix) = featureix + f3.firstix - 1

firstix(f3::Features003) = f3.firstix
grad(f3, regrminutes) =  f3.f2.regr[regrminutes].grad[ohlcvix(f3, 1):end]
regry(f3, regrminutes) = f3.f2.regr[regrminutes].regry[ohlcvix(f3, 1):end]
std(f3, regrminutes) =   f3.f2.regr[regrminutes].std[ohlcvix(f3, 1):end]
ohlcvdataframe(f3) = Ohlcv.dataframe(f3.f2.ohlcv)[ohlcvix(f3, 1):end, :]
opentime(f3) = Ohlcv.dataframe(f3.f2.ohlcv)[ohlcvix(f3, 1):end, :opentime]

"""
Return a DataFrame column `df[firstix-lookback:lastix-lookback, col]` that reprents the `lookback` predecessor rows of that col.
Assumes that predecessors have lower row indices than the successor rows, i.e. newer values are appended at the end of the `df`.
If lookback refers to elements out side of df the `fill` be used. If `fill` is `nothing` then df[begin, col] is used.
"""
function lookbackrow!(rdf::Union{DataFrame, Nothing}, df::AbstractDataFrame, col::String,lookback, firstix=1, lastix=size(df,1); fill=nothing)
    dfl = size(df,1)
    @assert lookback >= 0 "lookback ($lookback) >= 0"
    @assert dfl >= 1 "size(df,1) == $dfl < 1"
    @assert 1 <= firstix <= dfl "1 <= firstix ($firstix) <= dfl ($dfl)"
    @assert 1 <= lastix <= dfl "1 <= lastix ($lastix) <= dfl ($dfl)"
    @assert firstix <= lastix "firstix ($firstix) <= lastix ($lastix)"
    if isnothing(rdf)
        rdf = DataFrame()
        rdfl = 0
    else
        rdfl = size(rdf,1)
        @assert rdfl == (lastix - firstix + 1)  "rdfl ($rdfl) == (lastix ($lastix) - firstix ($firstix) + 1)"
    end
    # ixstr = lookback > 0 ? string(lookback, pad=2, base=10) : ""
    ixstr = string(lookback, pad=2, base=10)
    configrow = col*ixstr
    fix = max(1, firstix - lookback)
    lix = lastix - lookback
    fill = isnothing(fill) ? df[fix, col] : fill
    fillcount = firstix > lookback ? 0 : 1 - (firstix - lookback)
    fillarr = similar(df[!, col], fillcount)
    fillarr .= fill
    if lix >= fix # copy lookback vector part
        if fillcount > 0
            rdf[!, configrow] = vcat(fillarr, df[fix:lix, col])
        else
            rdf[!, configrow] = df[fix:lix, col]
        end
    else
        rdf[!, configrow] = fillarr
    end
    return rdf, configrow
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on f2.ohlcv time windows.
The feature vectors covering the ohlcv time range from f2.startix until f2.lastix.
Each feature vector is composed of:

- The most recent `lookbackperiods` x f2.ohlcv time window ohlc relative values. If `lookbackperiods = 0` then only the most recent ohlc relative values.
  - (OHLC- pivot) / pivot

"""
function deltaOhlc!(fvecdf::Union{DataFrame, Nothing}, f3::Features.Features003; normalize=true::Bool)::AbstractDataFrame
    #! not yet tested
    ohlcvdf = Ohlcv.dataframe(f3.f2.ohlcv)
    fvecdf = nothing
    df = DataFrame()
    ofix = f3.firstix - f3.maxlookback
    @assert ofix > 0
    fix = f3.firstix
    lix = f3.f2.lastix
    for col in ["open", "high", "low", "close"]
        deltacol = col * "-p"
        df[!,deltacol] = ohlcvdf[ofix:lix, col] .- ohlcvdf[ofix:lix, :pivot] #TODO inefficient if long vectors and f3.firstix close to end
        # println("size(df)=$(size(df))  size(ohlcvdf)=$(size(ohlcvdf))  col=$col  ofix=$ofix  fix=$fix  lix=$lix  ")
        for lookback in 0:f3.maxlookback
            fvecdf, colname = Features.lookbackrow!(fvecdf, df, deltacol, lookback, 1+f3.maxlookback)
            @assert (lix-fix+1) == size(fvecdf, 1) "(lix-fix+1) == size(fvecdf, 1)  ($lix-$fix+1) == $(size(fvecdf, 1))"
            if normalize
                fvecdf[!, colname] = fvecdf[!, colname] ./ ohlcvdf[fix:lix, :pivot]
            end
        end
    end
    return fvecdf
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on f3.f2.ohlcv time windows.
The feature vectors covering the ohlcv time range from f3.f2.startix + f3.maxlookback until f3.f2.lastix.
Each feature vector is composed of:

- Most recent `lookback` periods regression features for time windows as provided in `regrwindows` (e.g `[15, 60, 4*60]`). If `lookback = 0` then only the most recent regression features.
  - `grad`: Regression gradient
  - `disty`: Y distance from regression line   (not any longer related to std because std can be zero: / (2 * std deviation))
  - both grad and disty are normalized as to pivot price if `normalize=true`
"""
function regressionfeatures!(fvecdf::Union{DataFrame, Nothing}, f3::Features.Features003; regrwindows::Vector{<:Integer}, lookback, normalize=true::Bool)::AbstractDataFrame
    # debug = true
    @assert f3.maxlookback >= lookback
    fvecdf = isnothing(fvecdf) ? DataFrame() : fvecdf
    ohlcvdf = Ohlcv.dataframe(f3.f2.ohlcv)
    # debug ? println("f3.firstix=$(f3.firstix) size(ohlcvdf)=$(size(ohlcvdf))") : 0
    # debug ? println("fvecdf = $fvecdf") : 0
    for ix in 0:lookback
        ixstr = lookback > 0 ? string(ix, pad=2, base=10) : ""
        for regrwindow in regrwindows
            if !(regrwindow in keys(f3.f2.regr))
                @warn "skipping regrwindow because it is not in f3.f2.regr" regrwindow keys(f3.f2.regr)
                continue
            end
            ixoffset = ix * regrwindow
            firstfix = featureix(f3.f2, f3.firstix - ixoffset)
            lastfix = featureix(f3.f2, f3.f2.lastix - ixoffset)
            firstoix = f3.firstix - ixoffset
            lastoix = f3.f2.lastix - ixoffset
            colname = "grad" * Features.periodlabels(regrwindow) * ixstr
            # debug ? println("fvecdf = $(size(fvecdf)) $(names(fvecdf)) firstfix=$firstfix lastfix=$lastfix firstoix=$firstoix lastoix=$lastoix") : 0
            # debug ? println("grad = $(size(f3.f2.regr[regrwindow].grad)) ") : 0
            fvecdf[!, colname] = f3.f2.regr[regrwindow].grad[firstfix:lastfix]
            if normalize
                fvecdf[!, colname] = fvecdf[!, colname] ./ ohlcvdf[f3.firstix:f3.f2.lastix, :pivot]
            end
            colname = "disty" * Features.periodlabels(regrwindow) * ixstr
            fvecdf[!, colname] = ohlcvdf[firstoix:lastoix, :pivot] .- f3.f2.regr[regrwindow].regry[firstfix:lastfix]
            # println("f3.f2.regr[regrwindow($(regrwindow))].std: min=$(minimum(f3.f2.regr[regrwindow].std)) mean=$(Statistics.mean(f3.f2.regr[regrwindow].std)) max=$(maximum(f3.f2.regr[regrwindow].std))")
            # println("f3.f2.regr[regrwindow($(regrwindow))].regry: min=$(minimum(f3.f2.regr[regrwindow].regry)) mean=$(Statistics.mean(f3.f2.regr[regrwindow].regry)) max=$(maximum(f3.f2.regr[regrwindow].regry))")
            # println("ohlcvdf[firstoix:lastoix, :pivot]: min=$(minimum(ohlcvdf[firstoix:lastoix, :pivot])) mean=$(Statistics.mean(ohlcvdf[firstoix:lastoix, :pivot])) max=$(maximum(ohlcvdf[firstoix:lastoix, :pivot]))")
            # println("pivot - regr[regrwindow($(regrwindow))].regry: min=$(minimum(fvecdf[!, colname])) mean=$(Statistics.mean(fvecdf[!, colname])) max=$(maximum(fvecdf[!, colname]))")
            #* no longer related to std because std can be zero: fvecdf[!, colname] = fvecdf[!, colname] ./ f3.f2.regr[regrwindow].std[firstfix:lastfix]
            # println("(pivot - regr[regrwindow($(regrwindow))].regry)/std: min=$(minimum(fvecdf[!, colname])) mean=$(Statistics.mean(fvecdf[!, colname])) max=$(maximum(fvecdf[!, colname]))")
            # debug ? fvecdf[!, "firstix" * Features.periodlabels(regrwindow) * ixstr] = [ix for ix in firstfix:lastfix] : 0
            if normalize
                fvecdf[!, colname] = fvecdf[!, colname] ./ ohlcvdf[f3.firstix:f3.f2.lastix, :pivot]
            end
        end
    end
    return fvecdf
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on f2.ohlcv time windows.
The feature vectors covering the ohlcv time range from f2.startix until f2.lastix.
Each feature vector is composed of:

- Relative median volume 1m/1h (just 1 element)
"""
function relativevolume!(fvecdf::Union{DataFrame, Nothing}, f3::Features.Features003, shortvol, longvol)::AbstractDataFrame
    fvecdf = isnothing(fvecdf) ? DataFrame() : fvecdf
    ohlcvdf = Ohlcv.dataframe(f3.f2.ohlcv)
    colname = Features.periodlabels(shortvol) * "/" * Features.periodlabels(longvol) * "vol"
    fvecdf[!, colname] = relativevolume(ohlcvdf[f3.firstix:f3.f2.lastix, :basevolume], shortvol, longvol)
    return fvecdf
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on f2.ohlcv time windows.
The feature vectors covering the ohlcv time range from f2.startix until f2.lastix.
Each feature vector is composed of:

- Relative date/time according to keyword as mapped in Features.relativetimedict
  - "relminuteofday" => relativeminuteofday
  - "reldayofweek" => relativedayofweek
  - "reldayofyear" => relativedayofyear

"""
function relativetime!(fvecdf::Union{DataFrame, Nothing}, f3::Features.Features003, relativedatetime)::AbstractDataFrame
    ohlcvdf = Ohlcv.dataframe(f3.f2.ohlcv)
    @assert !isnothing(ohlcvdf)
    fvecdf = isnothing(fvecdf) ? DataFrame() : fvecdf
    fvecdf[!, relativedatetime] = map(relativetimedict[relativedatetime], ohlcvdf[f3.firstix:f3.f2.lastix, :opentime])
    return fvecdf
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on 1 minute time windows.
The feature vectors covering the ohlcv time range from f2.startix until f2.lastix.
Each feature vector is composed of:

- The most recent 12x 1 minute time window ohlc relative values
  - (OHLC- pivot) / pivot
- Most recent regression features for time windows 15m,1h,4h, 12h, 1d
  - Regression gradient
  - Y distance from regression line / std deviation
- Relative median volume 1m/1h
- Relative minute of the day
"""
function features12x1m01(f3::Features003)
    # println(f3)
    fvecdf = deltaOhlc!(nothing, f3; normalize=true)
    # println("deltaOhlc!: size(fvecdf)=$(size(fvecdf))")
    fvecdf = regressionfeatures!(fvecdf, f3; regrwindows=[15, 60, 4*60, 12*60, 24*60], lookback=0, normalize=true)
    # println("regressionfeatures!: size(fvecdf)=$(size(fvecdf))")
    fvecdf = relativevolume!(fvecdf, f3, 1, 60)
    fvecdf = relativetime!(fvecdf, f3, "relminuteofday")
    return fvecdf, f3
end

function features12x1m01(f2::Features002, lookbackperiods=11)
    f3 = Features003(f2, lookbackperiods)
    return features12x1m01(f3)
end

function features12x1m01(ohlcv::Ohlcv.OhlcvData, lookbackperiods=11)
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    @assert !isnothing(ohlcvdf)
    @assert Ohlcv.interval(ohlcv) == "1m"
    f2 = Features.Features002(ohlcv)
    @assert (size(ohlcvdf, 1) >= f2.lastix) "size(ohlcvdf)=$(size(ohlcvdf, 1)) < f2.lastix"
    @assert (f2.firstix <= f2.lastix)
    return features12x1m01(f2, lookbackperiods)
end

"""
Returns a DataFrame with feature vectors in each row. Each feature vector is based on 1 minute time windows.
The feature vectors covering the ohlcv time range from f2.startix until f2.lastix.
Each feature vector is composed of:

- The most recent `lookbackperiods` + 1  `focusregrwindow` minutes time window ohlc relative values
  - (OHLC- pivot) / pivot
- Most recent regression features for time windows in minutes, e.g. [15, 60, 4*60, 12*60, 24*60]
  - Regression gradient
  - Y distance from regression line / std deviation
- Relative median volume short time window/long time window
- Relative datetime
"""
function regressionfeatures01(f3::Features003, focusregrwindow, regrwindows, shortvol, longvol, reltime)
    fvecdf = regressionfeatures!(nothing, f3; regrwindows=regrwindows, lookback=0, normalize=true)
    fvecdf = regressionfeatures!(fvecdf, f3; regrwindows=[focusregrwindow], lookback=f3.maxlookback, normalize=true)
    fvecdf = relativevolume!(fvecdf, f3, shortvol, longvol)
    fvecdf = relativetime!(fvecdf, f3, reltime)
    return fvecdf, f3
end

function regressionfeatures01(f2::Features002, lookbackperiods, focusregrwindow, regrwindows, shortvol, longvol, reltime)
    f3 = Features003(f2, lookbackperiods)
    return regressionfeatures01(f3, focusregrwindow, regrwindows, shortvol, longvol, reltime)
end

function regressionfeatures01(ohlcv::Ohlcv.OhlcvData, lookbackperiods, focusregrwindow, regrwindows, shortvol, longvol, reltime)
    ohlcvdf = Ohlcv.dataframe(ohlcv)
    @assert !isnothing(ohlcvdf)
    @assert Ohlcv.interval(ohlcv) == "1m"
    f2 = Features.Features002(ohlcv)
    @assert (size(ohlcvdf, 1) >= f2.lastix) "size(ohlcvdf)=$(size(ohlcvdf, 1)) < f2.lastix"
    @assert (f2.firstix <= f2.lastix)
    return regressionfeatures01(f2, lookbackperiods, focusregrwindow, regrwindows, shortvol, longvol, reltime)
end

features12x5m01(f2::Features002) =  regressionfeatures01(f2,  11, 5,     [15, 60,   4*60,  12*60,   24*60],    5,     4*60,     "relminuteofday")
features12x15m01(f2::Features002) = regressionfeatures01(f2,  11, 15,    [5,  60,   4*60,  12*60,   24*60],    15,    12*60,    "relminuteofday")
features12x1h01(f2::Features002) =  regressionfeatures01(f2,  11, 60,    [5,  15,   4*60,  12*60,   24*60],    60,    24*60,    "reldayofweek")
features12x4h01(f2::Features002) =  regressionfeatures01(f2,  11, 4*60,  [15, 60,   12*60, 24*60,   3*24*60],  4*60,  3*24*60,  "reldayofweek")
features12x12h01(f2::Features002) = regressionfeatures01(f2,  11, 12*60, [60, 4*60, 24*60, 3*24*60, 10*24*60], 12*60, 7*24*60,  "reldayofyear")
features12x1d01(f2::Features002) =  regressionfeatures01(f2,  11, 24*60, [60, 4*60, 12*60, 3*24*60, 10*24*60], 24*60, 14*24*60, "reldayofyear")

features5m01(f3::Features003) =  regressionfeatures01(f3, 5,     [15, 60,   4*60,  12*60,   24*60],    5,     4*60,     "relminuteofday")
features15m01(f3::Features003) = regressionfeatures01(f3, 15,    [5,  60,   4*60,  12*60,   24*60],    15,    12*60,    "relminuteofday")
features1h01(f3::Features003) =  regressionfeatures01(f3, 60,    [5,  15,   4*60,  12*60,   24*60],    60,    24*60,    "reldayofweek")
features4h01(f3::Features003) =  regressionfeatures01(f3, 4*60,  [15, 60,   12*60, 24*60,   3*24*60],  4*60,  3*24*60,  "reldayofweek")
features12h01(f3::Features003) = regressionfeatures01(f3, 12*60, [60, 4*60, 24*60, 3*24*60, 10*24*60], 12*60, 7*24*60,  "reldayofyear")
features1d01(f3::Features003) =  regressionfeatures01(f3, 24*60, [60, 4*60, 12*60, 3*24*60, 10*24*60], 24*60, 14*24*60, "reldayofyear")

featureslookback01 = Dict()
featureslookback01[5] = features5m01
featureslookback01[15] = features15m01
featureslookback01[60] = features1h01
featureslookback01[4*60] = features4h01
featureslookback01[12*60] = features12h01
featureslookback01[24*60] = features1d01
#endregion Features003


end  # module

