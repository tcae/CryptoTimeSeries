"""
**! predicitions will be merged with targets.jl**
The asset can receive **predictions** within a given **time period** from algorithms or individuals by:

- assigning *increase*, *neutral*, *decrease*
- target price
- target +-% from current price

Prediction algorithms are identified by name. Individuals are identified by name.

"""
module Targets

using EnvConfig, Ohlcv, Features

# labellevels = ["short", "close", "hold", "long"]
labellevels = ["close", "hold", "long"]

strongsell=-2; sell=-1; hold=0; buy=1; strongbuy=2
gainthreshold = 0.01  # 1% gain threshold
lossthreshold = -0.005  # loss threshold to signal strong sell
gainlikelihoodthreshold = 0.7  # 50% probability threshold to meet or exceed gain/loss threshold

"""
Go back in index look for a more actual price extreme than the one from horizontal regression.
If regression of buyix is >0 then look back for a maximum else it is a falling slope then look back for a minimum.
"""
function absmaxindex(prices, regressions, buyix, sellix)
    comparison = regressions[buyix] > 0 ? (>) : (<)
    maxsellix = sellix
    while (sellix > buyix)
        sellix -= 1
        if comparison(prices[sellix], prices[maxsellix])
            maxsellix = sellix
        end
    end
    return maxsellix
end

"""
Returns a Float32 vector equal in size to prices/regressions that provides for each price the gain to the next extreme of 'regressions'.
"""
function gain2nextextreme(prices, regressions)
    pricelen = length(prices)
    @assert pricelen == length(regressions)
    regressiongains = zeros(Float32, pricelen)
    runix = startix = 1
    while startix < pricelen
        endix = Features.nextextremeindex(regressions, startix)
        if endix == 0  # no extreme but the array end
            absmaxix = endix = pricelen
        else
            absmaxix = absmaxindex(prices, regressions, startix, endix)
        end
        while runix <= absmaxix
            regressiongains[runix] = Ohlcv.relativegain(prices, runix, absmaxix)
            runix += 1
        end
        startix = endix
    end
    return regressiongains
end

"""
Labels Sell when tradegradientthresholds endregression is reached (does not consider the real ramp down in each case).
Labels Buy when either previous waves are high enough and minimum reached or tradegradientthresholds startregr is reached.
prices shall be the ohlcv.pivot prices array.
"""
function singleregressionlabels(prices, regressionminutes)  # TODO UNIT TEST TO BE ADDED
    regressions = Features.normrollingregression(prices, regressionminutes)
    startregr, endregr, _, _ = Targets.tradegradientthresholds(prices, regressions)
    # println("startregr=$startregr  endregr=$endregr")
    endregr = max(endregr, 0)
    df = Features.lastgainloss(prices, regressions)
    # println("lastgain=$(df.lastgain)")
    # println("lastloss=$(df.lastloss)")
    regressiongains = gain2nextextreme(prices, regressions)
    priceslen = size(prices, 1)
    @assert priceslen == size(df.lastgain, 1) == size(df.lastloss, 1) == size(regressions, 1) == size(regressiongains, 1)
    targets = zeros(Int8, size(prices))
    lastgain = lastlastgain = 0.0
    for ix in 1:size(prices, 1)
        if regressiongains[ix] >= gainthreshold  # buy
            stronglastgain = (lastlastgain >= gainthreshold) && (lastlastgain >= -lastgain) && (regressions[ix] > 0)
            # TODO investigate on training data whhether the stronglastgain criteria is valid
            if (regressions[ix] > startregr) || stronglastgain
                targets[ix] = strongbuy
            else
                targets[ix] = buy
            end
        elseif regressiongains[ix] <= lossthreshold
            targets[ix] = strongsell
        else  # gainthreshold <= regressions[ix] <= lossthreshold
            if (regressions[ix] < endregr)
                targets[ix] = sell
            else
                targets[ix] = hold
            end
        end
        if (ix < priceslen) && (sign(regressiongains[ix]) != sign(regressiongains[ix+1]))
            lastlastgain = lastgain
            lastgain = regressiongains[ix+1]
        end
    end
    return targets
end

"""
Labels Buy from the actual price minimum after the regression minimum.
Label Sell from the actual price maximum before the regression maximum.
"""
function regressionlabels2(prices, regressions)
    # prepare lookup tables
    @assert size(prices) == size(regressions) "sizes of prices and regressions shall match"
    targets = zeros(Int8, size(prices))
    targets .= sell  # identified buy and hold labels will correct sell
    firstix = 1
    while isequal(regressions[firstix], missing)
        firstix += 1
    end
    ixbuyend = 0
    for ix in length(regressions):-1:firstix  # against timeline
        if regressions[ix] <= 0  # falling slope
            if ixbuyend != 0  # last index that is a potential buy of rising slope
                ixbuystart = ix + 1 # valid index of minimum
                for ix2 in ixbuystart+1:ixbuyend
                    # from regression minimum forward search for price minimum
                    if prices[ix2] < prices[ixbuystart]
                        ixbuystart = ix2
                    end
                end
                while ixbuystart <= ixbuyend  # label slope
                    if Ohlcv.relativegain(prices, ixbuystart, ixbuyend) >= (2*gainthreshold)
                        targets[ixbuystart] = strongbuy
                    elseif Ohlcv.relativegain(prices, ixbuystart, ixbuyend) >= gainthreshold
                        targets[ixbuystart] = buy
                    else
                        targets[ixbuystart] = hold
                    end
                    ixbuystart += 1
                end
                ixbuyend = 0
            end
            # falling slope processing
            targets[ix] = strongsell
        else  # regressions[ix] > 0  # rising slope
            if (ixbuyend == 0)
                # start at the end of the rising slope at the regression maximum to search backward for the price maximum
                ixbuyend = ix
            else
                if prices[ix] > prices[ixbuyend]
                    ixbuyend = ix
                    # ixbuyend is last index that is a potential buy
                end
            end
        end
    end
    return targets
end

"""
Labels Sell when actual price maximum before regression maximum is reached
Labels Buy when either previous waves are high enough and minimum reached or tradegradientthresholds startregr is reached.
"""
function regressionlabels3(prices, regressions)
    # prepare lookup tables
    @assert size(prices) == size(regressions) "sizes of prices and regressions shall match"
    startregr::Float32, endregr::Float32 = Targets.tradegradientthresholds(prices, regressions)
    # println("startregr=$startregr  endregr=$endregr")
    df = Features.lastgainloss(prices, regressions)
    # println("lastgain=$(df.lastgain)")
    # println("lastloss=$(df.lastloss)")
    @assert size(prices) == size(df.lastgain) == size(df.lastloss)
    targets = zeros(Int8, size(prices))
    targets .= sell  # identified buy and hold labels will correct sell
    firstix = 1
    while isequal(regressions[firstix], missing)
        firstix += 1
    end
    ixbuyend = 0
    for ix in length(regressions):-1:firstix  # against timeline
        if regressions[ix] <= 0  # falling slope
            if ixbuyend != 0  # last index that is a potential buy of rising slope
                ixbuystart = ix + 1 # valid index of minimum
                stronglastgainloss = (df.lastgain[ixbuyend] >= gainthreshold) && (df.lastgain[ixbuyend] >= df.lastloss[ixbuyend])
                while (regressions[ixbuystart] < startregr) && !stronglastgainloss && (ixbuystart < ixbuyend) && (Ohlcv.relativegain(prices, ixbuystart, ixbuyend) >= gainthreshold)
                    # from minimum forward until steep enough slope detected
                    targets[ixbuystart] = buy
                    ixbuystart += 1
                end
                while ixbuystart <= ixbuyend  # label slope
                    if Ohlcv.relativegain(prices, ixbuystart, ixbuyend) >= gainthreshold
                        targets[ixbuystart] = strongbuy
                    else
                        targets[ixbuystart] = hold
                    end
                    ixbuystart += 1
                end
                ixbuyend = 0
            end
            # falling slope processing
            targets[ix] = strongsell
        else  # regressions[ix] > 0  # rising slope
            if (ixbuyend == 0)
                # start at the end of the rising slope at the regression maximum to search backward for the price maximum
                ixbuyend = ix
            else
                if prices[ix] > prices[ixbuyend]
                    ixbuyend = ix
                    # ixbuyend is last index that is a potential buy
                end
            end
        end
    end
    return targets
end

"""
Returns a Float32 vector equal in size to prices/regressions that provides for each price the gain to the next extreme of 'regressions'.
The index of the next extreme is also provided as an Int array
"""
function rollingnextextreme(prices, regressions)  # TODO missing unit test
    pricelen = length(prices)
    @assert pricelen == length(regressions)
    gain = zeros(Float32, pricelen)
    xix = zeros(Int32, pricelen)
    ix = xixnext = 1
    while ix < pricelen
        xixnext = Features.nextextremeindex(regressions, xixnext)
        xixnext = xixnext == 0 ? pricelen : xixnext
        while ix <= xixnext
            xix[ix] = xixnext
            gain[ix] = (prices[xixnext] - prices[ix]) / prices[ix]
            ix += 1
        end
    end
    return gain, xix
end

"""
Buy if regression(5) is at start up slope and next extreme or any longer regression has a regression price gain > gainthreshold before having a regression price loss < lossthreshold.
Sell if regression(5) is at start down slope and next extreme or any longer regression has a regression loss < lossthreshold before having a regression price gain > gainthreshold.
If there is no regression gainthreshold or lossthreshold exceeded then sell.

- `prices` is an array of not yet normalized pivot prices.
- `regressionminutes` is an array of regression windows in minutes.

Returns a dict of intermediate and final results consiting of arrays of same length as prices with the following keys:
- `target` = array of integers denoting the ideal labels, which is the result of this function
- *regression minutes* = dict of intermediate results belonging to *regression minutes*
    - `gradient` = array of gradients of *regression minutes*
    - `price` = array of regression prices (not actual prices) of *regression minutes*
    - `xix` = index of next regression extreme of *regression minutes*
    - `gain` = regression price gain as a ratio to current regression price to next extreme of *regression minutes*
"""
function targets4(prices, regressionminutes)  # TODO missing unit test
    sort!(regressionminutes)
    result = Dict()
    result[:target] = zeros(Int8, length(prices))
    for rmin in regressionminutes
        result[rmin] = Dict()
        result[rmin][:price], result[rmin][:gradient] = Features.rollingregression(prices, rmin)
        result[rmin][:gain], result[rmin][:xix] = rollingnextextreme(result[rmin][:price], result[rmin][:gradient])
    end
    for ix in 1:length(prices)
        for rmin in regressionminutes
            result[:target][ix] = sell  # in doubt sell #! result[:target] is overridden with each new rmin
            if result[rmin][:gain][ix] > gainthreshold
                result[:target][ix] = buy
                break
            elseif result[rmin][:gain][ix] < lossthreshold
                result[:target][ix] = sell
                break
            elseif result[rmin][:gain][ix] > 0
                result[:target][ix] = hold
                # no break because a buy or sell may come with larger regression windows
            end
        end
    end
    return result
end

struct LabelThresholds
    longbuy
    longhold
    shorthold
    shortbuy
end

"""
- buy long at more than 3% gain potential from current price
- hold long above 0.01% gain potential from current price
- close long position below 0.01% gain potential from current price

- buy short at or lower than -3% loss potential from current price
- hold short below -0.01% loss potential from current price
- close short position above -0.01% loss potential from current price
"""
defaultlabelthresholds = LabelThresholds(0.03, 0.0001, -0.0001, -0.03)

"""
Because the trade signals are not independent classes but an ordered set of actions, this function returns the labels that correspond to specific thresholds:

- The folllowing invariant is assumed: `longbuy > longclose >= 0 >= shortclose > shortbuy`
- a gain shall be above `longbuy` threshold for a buy (long buy) signal
- bought assets shall be held (but not bought) if the remaining gain is above `closelong` threshold
- bought assets shall be closed if the remaining gain is below `closelong` threshold
- a loss shall be below `shortbuy` for a sell (short buy) signal
- sold (short buy) assets shall be held if the remaining loss is below `closeshort`
- sold (short buy) assets shall be closed if the remaining loss is above `closeshort`
- all thresholds are relative gain values: if backwardrelative then the relative gain is calculated with the target price otherwise with the current price

"""
function getlabels(relativedist, labelthresholds)
    lt = labelthresholds
    rd = relativedist
    labels = [(rd > lt.longbuy ? "longbuy" : (rd > lt.longhold ? "longhold" :
              (lt.shortbuy > rd ? "shortbuy" : (lt.shorthold > rd ? "shorthold" : "close")))) for rd in relativedist]
    # labels = [(rd > lt.longbuy ? "buy" : (lt.shortclose > rd ? "close" : "hold")) for rd in relativedist]
    return labels
end

function relativedistances(prices, pricediffs, priceix, backwardrelative=true)
    if backwardrelative
        relativedist = [(priceix[ix] == 0 ? 0.0 : pricediffs[ix] / prices[priceix[ix]]) for ix in 1:size(prices, 1)]
    else
        relativedist = pricediffs ./ prices
    end
end

function continuousdistancelabels(prices, labelthresholds)
    pricediffs, priceix = Features.nextpeakindices(prices, labelthresholds.longbuy, labelthresholds.shortbuy)
    relativedist = relativedistances(prices, pricediffs, priceix, true)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

"""
- returns pricediffs of current price to next extreme price
- pricediffs will be negative if the next extreme is a minimum and positive if it is a maximum
- the extreme is determined by slope sign change of the regression gradients given in `regressions`
- from this regression extreme the peak is search backwards thereby skipping all local extrema that are insignifant for that regression window
- for debugging purposes 2 further index arrays are returned: with regression extreme indices and with price extreme indices

"""
function continuousdistancelabels(prices, regressiongradients, labelthresholds)
    pricediffs, regressionix, priceix = Features.pricediffregressionpeak(prices, regressiongradients; smoothing=false)
    relativedist = relativedistances(prices, pricediffs, priceix, true)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, regressionix, priceix
end

"""
- Returns the index within `relativedistarray` of the best regression.
    + an entry of 0 means that none of the regressions meets the required amplitude requirement specified in `requiredrelativeamplitude`
- `relativedistarray` is a sorted array of relativedist arrays as provided by `continuousdistancelabels`
    + the arrays are sorted according to regressionwindow starting with the shortest regressionwindow and ending with the longest
    + each array entry contains an array of relative pricediffs of the price extremes
    + each array has the same length
"""
function bestregression(relativedistarray, requiredrelativeamplitude)
    requiredrelativeamplitude = abs(requiredrelativeamplitude)
    plen = 0
    if (size(relativedistarray, 1) > 0) && (size(relativedistarray[1], 1) > 0)
        plen = size(relativedistarray[1], 1)
    else
        @warn "empty relativedistarray"
        return []
    end
    bestregr = zeros(Int32, plen)
    for xpix in eachindex(relativedistarray)  # 1:size(relativedistarray, 1)
        @assert plen == size(relativedistarray[xpix])
        dist = relativedistarray[xpix][1]
        for ix in 1:plen
            if (dist * relativedistarray[xpix][ix]) <= 0  # distance sign change --> new distance between extremes
                dist = relativedistarray[xpix][ix]
            end
            if (bestregr[ix] == 0) && (abs(dist) >= requiredrelativeamplitude)
                bestregr[ix] = xpix
            end
        end
    end
    return bestregr
end

end  # module
