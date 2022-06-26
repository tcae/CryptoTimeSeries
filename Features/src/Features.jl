"""
Provides features to classify the most price development.
Currently advertised features are in Features001

"""
module Features

# using Dates, DataFrames
import RollingFunctions: rollmedian, runmedian, rolling
import DataFrames: DataFrame, Statistics
using Combinatorics
using Logging
using EnvConfig, Ohlcv

struct Features001
    fdf::DataFrame
    featuremask::Vector{String}
    ohlcv::OhlcvData
end

regressionwindows002 = [5, 15, 60, 4*60, 12*60, 24*60]  # , 3*24*60, 10*24*60]
requiredminutes = maximum(regressionwindows002)

periodlabels(p) = p%(24*60) == 0 ? "$(round(Int, p/(24*60)))d" : p%60 == 0 ? "$(round(Int, p/60))h" : "$(p)m"

mutable struct Features002Regr
    grad::Vector{Float32} # rolling regression gradients - length == ohlcv
    regry::Vector{Float32}  # rolling regression price - length == ohlcv
    std::Vector{Float32}  # standard deviation of regression window - length == ohlcv
    # xtrmix::Vector{Int32}  # indices of extremes (<0 if min, >0 if max) - length <= ohlcv
    # breakoutix::Vector{Int32}  # indices of extremes (<0 if min, >0 if max) - length <= ohlcv
    medianstd::Vector{Float32}  # median standard deviation over requiredminutes
end

mutable struct Features002
    ohlcv::OhlcvData
    regr::Dict  # dict with regression minutes as key -> value is Features002Regr
    # fdf::DataFrame  # cache of features
    update  # function to update features due to extended ohlcv
    firstix  # features start at firstix of ohlcv.df
    lastix  # features end at lastix of ohlcv.df
    function Features002(ohlcv, firstix=firstindex(ohlcv.df.opentime), lastix=lastindex(ohlcv.df.opentime))
        df = Ohlcv.dataframe(ohlcv)
        lastix = lastix > lastindex(df, 1) ? lastix = lastindex(df, 1) : lastix
        maxfirstix = max((lastix - requiredminutes + 1), firstindex(df, 1))
        firstix = firstix > maxfirstix ? maxfirstix : firstix
        new(ohlcv, getfeatures002(ohlcv, firstix, lastix), getfeatures002!, firstix, lastix)
    end
end


function Base.show(io::IO, features::Features002Regr)
    println(io::IO, "- gradients: size=$(size(features.grad)) max=$(maximum(features.grad)) median=$(Statistics.median(features.grad)) min=$(minimum(features.grad))")
    println(io::IO, "- regression y: size=$(size(features.regry)) max=$(maximum(features.regry)) median=$(Statistics.median(features.regry)) min=$(minimum(features.regry))")
    println(io::IO, "- std deviation: size=$(size(features.std)) max=$(maximum(features.std)) median=$(Statistics.median(features.std)) min=$(minimum(features.std))")
    print(io::IO, "- median std deviation: size=$(size(features.medianstd)) max=$(maximum(features.medianstd)) median=$(Statistics.median(features.medianstd)) min=$(minimum(features.medianstd))")
    # println(io::IO, "- extreme indices: size=$(size(features.xtrmix)) #maxima=$(length(filter(r -> r > 0, features.xtrmix))) #minima=$(length(filter(r -> r < 0, features.xtrmix)))")
    # print(io::IO, "- breakoutix indices: size=$(size(features.breakoutix)) #maxima=$(length(filter(r -> r > 0, features.breakoutix))) #minima=$(length(filter(r -> r < 0, features.breakoutix)))")
end

function Base.show(io::IO, features::Features002)
    println(io::IO, features.ohlcv)
    for (key, value) in features.regr
        println(io::IO, "regr key: $key")
        println(io::IO, value)
    end
end

ohlcv(features::Features002) = features.ohlcv
indexinrange(index, last) = 0 < index <= last
nextindex(forward, index) = forward ? index + 1 : index - 1
up(slope) = slope > 0
downorflat(slope) = slope <= 0


"""
- returns index of next regression extreme (or 0 if no extreme)
    - regression extreme index: in case of uphill **after** slope > 0
    - regression extreme index: in case of downhill **after** slope <= 0

"""
function extremeregressionindex(regressions, startindex; forward)
    reglen = length(regressions)
    extremeindex = 0
    @assert indexinrange(startindex, reglen) "index: $startindex  len: $reglen"
    if regressions[startindex] > 0
        while indexinrange(startindex, reglen) && up(regressions[startindex])
            startindex = nextindex(forward, startindex)
        end
    else  # regressions[startindex] <= 0
        while indexinrange(startindex, reglen) && downorflat(regressions[startindex])
            startindex = nextindex(forward, startindex)
        end
    end
    if indexinrange(startindex, reglen)  # then extreme detected
        if forward
            extremeindex = startindex
        elseif startindex < reglen
            extremeindex = startindex
        end
        # else end of array and no extreme detected, which is signalled by returned index 0
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

"""
function regressionextremesix!(extremeix, regressiongradients, startindex; forward=true)
    @assert startindex > 0
    @assert !isnothing(regressiongradients)
    if startindex > length(regressiongradients)
        @warn "unepected startindex beyond length of search vector *regressiongradients*" startindex length(regressiongradients)
        return extremeix
    end
    if isnothing(extremeix)
        extremeix = Int32[]
    end
    xix = extremeregressionindex(regressiongradients, startindex; forward)
    while xix != 0
        if forward
            if (length(extremeix) > 0) && (abs(extremeix[end]) >= xix)
                @warn "inconsistency: abs(extremeix[end]) >= next xtreme ix" extremeix[end] xix
            end
            extremeix = regressiongradients[xix] > 0 ? push!(extremeix, -xix) : push!(extremeix, xix)
        else
            if (length(extremeix) > 0) && (abs(extremeix[1]) <= xix)
                @warn "inconsistency: extremeix[end] <= next xtreme ix" extremeix[end] xix
            end
            extremeix = regressiongradients[xix] > 0 ? pushfirst!(extremeix, xix) : pushfirst!(extremeix, -xix)
        end
        xix = extremeregressionindex(regressiongradients, xix; forward)
    end
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
                rix = extremeregressionindex(regressions, cix; forward=true)
                rix = rix == 0 ? plen : rix
            elseif rix < plen  # rix >= cix
                # extreme price index pix between cix and last regression extreme rix was found and distances filled
                # look for next regression extreme starting from last regression extreme
                maxsearch = regressions[rix] > 0
                rix = extremeregressionindex(regressions, rix; forward=true)
                rix = rix == 0 ? plen : rix
            end
            lastpix = pix > lastpix ? pix : lastpix
            # - it was expected that the relevant actual price maximum is between gradient inflection index and regression extreme
            # >> it turned out that the inflection is not reliable
            # >> search from regression extreme back to last price extreme for global extreme yields better results
            pix = extremepriceindex(prices, rix, cix, maxsearch)  # search back from extreme rix to current index cix
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
    return regression_y, gradient
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
    return regression_y, gradient
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
    @assert size(y, 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(y, 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    normy = similar(y)
    std = similar(y)
    mean = similar(y)
    # normy .= 0
    # std .= 0
    # mean .= 0
    for ix1 in size(y, 1):-1:1
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            normy[ix2] = y[ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
        end
        mean[ix1] = Statistics.mean(normy[ix2min:ix1])
        std[ix1] = Statistics.stdm(normy[ix2min:ix1], mean[ix1])
    end
    std[1] = 0  # not avoid NaN
    return std, mean, normy
end

"""
Acts like rollingregressionstd(y, regr_y, grad, window) but starts calculation at *startindex-windowsize+1*.
In order to get only the std, mean, normy without padding use the subvectors *[windowsize:end]*
"""
function rollingregressionstd(y, regr_y, grad, window, startindex)
    @assert size(y, 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(y, 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    starty = max(1, startindex-window+1)
    offset = starty - 1
    normy = similar(y[starty:end])
    std = similar(normy)
    mean = similar(normy)
    # normy .= 0
    # std .= 0
    # mean .= 0
    for ix1 in size(y, 1):-1:startindex
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            normy[ix2-offset] = y[ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
        end
        ix3 = ix2min-offset
        ix4 = ix1-offset
        mean[ix4] = Statistics.mean(normy[ix3:ix4])
        std[ix4] = Statistics.stdm(normy[ix3:ix4], mean[ix4])
    end
    std[1] = 0  # not avoid NaN
    return std, mean, normy
end

"""
calculates and appends missing `length(y) - length(std)` *std, mean, normy* elements that correpond to the last elements of *y*
"""
function rollingregressionstd!(std, y, regr_y, grad, window)
    @assert size(y) == size(regr_y) == size(grad) "$(size(y)) == $(size(regr_y)) == $(size(grad))"
    if (length(y) > 0)
        startindex = isnothing(std) ? 1 : (length(std) < length(y)) ? length(std)+1 : length(y)
        stdnew, mean, normy = rollingregressionstd(y, regr_y, grad, window, startindex)
        if isnothing(std) || isnothing(mean) || isnothing(normy)
            std = stdnew
        elseif length(std) < length(y)  # only change regression_y and gradient if it makes sense
            startindex = min(startindex, window)
            std = append!(std, stdnew[startindex:end])
        else
            @warn "nothing to append when length(std) >= length(y)" length(std) length(y)
        end
    end
    return std, mean[startindex:end], normy[startindex:end]
end

    """ don't use - this is a version for debug reasons """
function rollingregressionstdxt(y, regr_y, grad, window)
    @assert size(y, 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(y, 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    normy = similar(y, (size(y, 1), window))
    std = similar(y)
    mean = similar(y)
    for ix1 in size(y, 1):-1:1
        ix2min = max(1, ix1 - window + 1)
        for ix2 in ix2min:ix1
            # @assert 0<ix2<=window "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), window: $window"
            # @assert 0<ix1<=size(y, 1) "ix1: $ix1, ix2: $ix2, ix2-ix2min+1: $(ix2-ix2min+1), size(y, 1): $(size(y, 1))"
            ix3 = ix2-ix2min+1
            ny = y[ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
            # Logging.@info "check" ix1 ix2 ix2min ix3 ny
            normy[ix1, ix3] = ny
        end
        mean[ix1] = Statistics.mean(normy[ix1, 1:min(window, ix1)])
        std[ix1] = Statistics.stdm(normy[ix1, 1:min(window, ix1)], mean[ix1])
        std[1] = 0  # not avoid NaN
    end
    # Logging.@info "check" normy y regr_y grad std mean
    return std, mean, normy
end

"""

For each x starting at *startindex-windowsize+1*:

- expand regression to the length of window size
- subtract regression from y to remove trend within window
- calculate std and mean on resulting trend free data for just 1 x

Returns a std vector of length `length(regr_y) - startindex + 1` for each x calculated back using the last `window` `ymv[*]` data

In multiple vectors *mv* version, ymv is an array of ymv vectors all of the same length like regr_y and grad

- standard deviation of `ymv` vectors minus rolling regression as given in `regr_y` and `grad`

In order to get only the std without padding use the subvector *[windowsize:end]*
"""
function rollingregressionstdmv(ymv, regr_y, grad, window, startindex)
    @assert size(ymv, 1) > 0
    @assert size(ymv[1], 1) == size(regr_y, 1) == size(grad, 1) >= window > 0 "$(size(ymv[1], 1)), $(size(regr_y, 1)), $(size(grad, 1)), $window"
    ymvlen = size(ymv, 1)
    normy = repeat(similar([ymv[1][1]], window * ymvlen))
    std = similar(regr_y[startindex:end])
    for ix1 in startindex:size(regr_y, 1)
        ix2min = max(1, ix1 - window + 1)
        thiswindow = ix1 - ix2min + 1
        for ymvix in 1:ymvlen
            for ix2 in ix2min:ix1
                normy[ix2-ix2min+1 + (ymvix-1)*thiswindow] = ymv[ymvix][ix2] - (regr_y[ix1] - grad[ix1] * (ix1 - ix2))
            end
        end
        normylen = ymvlen*thiswindow
        ymvmean = Statistics.mean(normy[1:normylen])
        std[ix1-startindex+1] = Statistics.stdm(normy[1:normylen], ymvmean)
    end
    std[1] = isnan(std[1]) ? 0 : std[1]
    return std
end

"""
calculates and appends missing `length(y) - length(std)` *std, mean, normy* elements that correpond to the last elements of *y*
"""
function rollingregressionstdmv!(std, ymv, regr_y, grad, window)
    @assert size(ymv, 1) > 0
    @assert size(ymv[1]) == size(regr_y) == size(grad) "$(size(ymv)) == $(size(regr_y)) == $(size(grad))"
    ymvlen = size(ymv, 1)
    if (length(regr_y) > 0)
        startindex = isnothing(std) ? 1 : (length(std) < length(regr_y)) ? length(std)+1 : length(regr_y)
        stdnew = rollingregressionstdmv(ymv, regr_y, grad, window, startindex)
        if isnothing(std)
            std = stdnew
        elseif length(std) < length(regr_y)  # only change regression_y and gradient if it makes sense
            # std = append!(std, stdnew[startindex:end])
            std = append!(std, stdnew)
            @assert length(std) == length(regr_y)
        else
            @warn "nothing to append when length(std) >= length(y)" length(std) ymvlen
        end
    end
    return std
end

"""
Returns the rolling median of std over requiredminutes starting at startindex.
If window > 0 then the window length is subtracted from requiredminutes because it is considered in the first std
"""
function rollingmedianstd!(medianstd, std, requiredminutes, startindex, window=1)
    medianstd = copy(std)
    return medianstd

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

function relativevolume(volumes, shortwindow, largewindow)
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
function getfeatures001(ohlcvdf::DataFrame)
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

# function last3extremes(pivot, regrgrad)
#     l1xtrm = similar(pivot); l2xtrm = similar(pivot); l3xtrm = similar(pivot)
#     regressiongains = zeros(Float32, pricelen)
#     # may be it is beneficial for later incremental addition during production to use indices in order to recognize what was filled
#     # work consistently backward to treat initial fill and incremental fill alike
#     return l1xtrm, l2xtrm, l3xtrm
# end

featureix(f2::Features002, ohlcvix) = ohlcvix - f2.firstix + 1

"""
In general don't call this function directly but via Feature002 constructor `Features.Features002(ohlcv)`
"""
function getfeatures002(ohlcv::OhlcvData, firstix, lastix)
    # println("getfeatures002 init")
    df = Ohlcv.dataframe(ohlcv)
    pivot = Ohlcv.pivot!(ohlcv)[firstix:lastix]
    open = df.open[firstix:lastix]
    high = df.high[firstix:lastix]
    low = df.low[firstix:lastix]
    close = df.close[firstix:lastix]
    ymv = [open, high, low, close]
    @assert length(pivot) >= (lastix - firstix + 1) >= requiredminutes "length(pivot): $(length(pivot)) >= $(lastix - firstix + 1) >= $requiredminutes"
    @assert firstindex(ohlcv.df[!, :opentime]) <= firstix <= lastix <= lastindex(ohlcv.df[!, :opentime]) "$(firstindex(ohlcv.df[!, :opentime])) <= $firstix <= $lastix <= $(lastindex(ohlcv.df[!, :opentime]))"
    regr = Dict()
    for window in regressionwindows002
        regry, grad = rollingregression(pivot, window)
        std = rollingregressionstdmv(ymv, regry, grad, window, 1)
        medianstd = rollingmedianstd!(nothing, std, requiredminutes, 1, window)
        regr[window] = Features002Regr(grad, regry, std, medianstd)
    end
    return regr
end

"""
Appends features if length(f2.ohlcv.pivot) > length(f2.regr[x].grad)
"""
function getfeatures002!(f2::Features002, firstix=f2.firstix, lastix=lastindex(f2.ohlcv.df[!, :opentime]))
    # println("getfeatures002!")
    df = Ohlcv.dataframe(f2.ohlcv)
    if f2.lastix >= lastindex(df, 1)
        return f2
    end
    lastix = lastix > lastindex(df, 1) ? lastix = lastindex(df, 1) : lastix
    maxfirstix = max((lastix - requiredminutes + 1), firstindex(df, 1))
    firstix = firstix > maxfirstix ? maxfirstix : firstix

    pivot = df[!, :pivot][firstix:lastix]
    open = df.open[firstix:lastix]
    high = df.high[firstix:lastix]
    low = df.low[firstix:lastix]
    close = df.close[firstix:lastix]
    ymv = [open, high, low, close]
    if (f2.lastix >= firstix >= f2.firstix) && (lastix >= f2.lastix)
        firstfeatureix = firstix - f2.firstix + 1
        for window in keys(f2.regr)
            fr = f2.regr[window]
            if firstfeatureix > 1  # cut start of available features
                fr.regry = fr.regry[firstfeatureix:end]
                fr.grad = fr.grad[firstfeatureix:end]
                fr.std = fr.std[firstfeatureix:end]
                fr.medianstd = fr.medianstd[firstfeatureix:end]
            end
            if lastix > f2.lastix
                regry, grad = rollingregression!(fr.regry, fr.grad, pivot, window)
                fr.std = rollingregressionstdmv!(fr.std, ymv, regry, grad, window)
                fr.medianstd = rollingmedianstd!(fr.medianstd, fr.std, requiredminutes, length(fr.medianstd)+1, window)
            else
                @warn "getfeatures002! nothing to add because lastix == f2.lastix"
            end
            @assert length(pivot) == length(fr.grad) == length(fr.regry) == length(fr.std)
        end
    else
        @info "getfeatures002! no reuse of previous calculations: f2.firstix=$(f2.firstix) f2.lastix=$(f2.lastix) firstix=$firstix lastix=$lastix"
        for window in regressionwindows002
            regry, grad = rollingregression(pivot, window)
            std = rollingregressionstdmv(ymv, regry, grad, window, 1)
            medianstd = rollingmedianstd!(nothing, std, requiredminutes, 1, window)
            f2.regr[window] = Features002Regr(grad, regry, std, medianstd)
            @assert length(pivot) == length(grad) == length(regry) == length(std)
        end
    end
    f2.firstix = firstix
    f2.lastix = lastix

    for window in keys(f2.regr)
        @assert firstindex(f2.regr[window].std) == featureix(f2, firstix)
        @assert lastindex(f2.regr[window].std) == featureix(f2, lastix)
        @assert firstindex(f2.regr[window].medianstd) == featureix(f2, firstix)
        @assert lastindex(f2.regr[window].medianstd) == featureix(f2, lastix) "lastix=$lastix f2.firstix=$(f2.firstix) lastindex(f2.regr[$window].medianstd)=$(lastindex(f2.regr[window].medianstd)) len(std)=$(length(f2.regr[window].std)) len(std)=$(length(f2.regr[window].medianstd))"
    end

    return f2
end

function mostrecentix(f2::Features002)
    ix = size(f2.ohlcv.df, 1)
end

function getfeatures(ohlcv::OhlcvData)
    return getfeatures001(ohlcv)
    # return getfeatures002(ohlcv)
end


"""
 Update featureset001 to match ohlcv.
 constraint: either featureset001 is nothing or empty or it has to be appended
 but no need to add before an existing start
"""
function getfeatures!(features::Features001, ohlcv::OhlcvData)
    # TODO to be done
    return features
end

end  # module

