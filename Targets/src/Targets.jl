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
using DataFrames, Logging

"returns all possible labels:"
possiblelabels() = ["longbuy", "longhold", "close", "shorthold", "shortbuy"]


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

"""
- defines the relative transaction thresholds
    - buy long at more than *longbuy* gain potential from current price
    - hold long above *longhold* gain potential from current price
    - close long position below *longhold* gain potential from current price
    - buy short at or lower than *shortbuy* loss potential from current price
    - hold short below *shorthold* loss potential from current price
    - close short position above *shorthold* loss potential from current price
- Targets.defaultlabelthresholds provides default thresholds
"""
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
function getlabels(relativedist, labelthresholds::LabelThresholds)
    @assert all([lab in ["longbuy", "longhold", "shortbuy", "shorthold", "close"] for lab in possiblelabels()])
    lt = labelthresholds
    rd = relativedist
    labels = [(rd > lt.longbuy ? "longbuy" : (rd > lt.longhold ? "longhold" :
              (lt.shortbuy > rd ? "shortbuy" : (lt.shorthold > rd ? "shorthold" : "close")))) for rd in relativedist]
    # labels = [(rd > lt.longbuy ? "buy" : (lt.shortclose > rd ? "close" : "hold")) for rd in relativedist]
    return labels
end

function relativedistances(prices::Vector{T}, pricediffs, priceix, backwardrelative=true) where {T<:Real}
    if backwardrelative
        relativedist = [(priceix[ix] == 0 ? T(0.0) : pricediffs[ix] / prices[priceix[ix]]) for ix in 1:size(prices, 1)]
    else
        relativedist = pricediffs ./ prices
    end
end

function continuousdistancelabels(prices, labelthresholds::LabelThresholds)
    pricediffs, priceix = Features.nextpeakindices(prices, labelthresholds.longbuy, labelthresholds.shortbuy)
    relativedist = relativedistances(prices, pricediffs, priceix, true)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

mutable struct Dists
    pricediffs
    regressionix
    priceix
    relativedist
    # Dists(prices, pricediffs, regressionix, priceix) = new(pricediffs, regressionix, priceix, relativedistances(prices, pricediffs, priceix, false))
end

function (Dists)(prices::Vector{T}, regressiongradients::Vector{Vector{T}}) ::Vector{Dists} where {T<:Real}
    d = Array{Dists}(undef, length(regressiongradients))
    for ix in eachindex(d)
        pricediffs, regressionix, priceix = Features.pricediffregressionpeak(prices, regressiongradients[ix]; smoothing=false)
        relativedist = relativedistances(prices, pricediffs, priceix, false)
        d[ix] = Dists(pricediffs, regressionix, priceix, relativedist)
    end
    return d
end

"""
- returns for each index of prices
    - labels based on given thresholds that are compared with relative pricediffs
    - relative pricediffs related to the index under consideration (answering: what can be gained/lost from the current ix)
    - absolute pricediffs of current price to next extreme price
    - for debugging purposes 2 further index arrays are returned: with regression extreme indices and with price extreme indices
    - pricediffs will be negative if the next extreme is a minimum and positive if it is a maximum
- the extreme is determined by slope sign change of the regression gradients given in `regressions`
- from this regression extreme the peak is search backwards thereby skipping all local extrema that are insignifant for that regression window
- for multiple regressiongradients the next extreme exceeding long/short buy thresholds is used

"""
function continuousdistancelabels(prices::Vector{T}, regressiongradients::Vector{Vector{T}}, labelthresholds::LabelThresholds) where {T<:Real}
    # for rg in regressiongradients
    #     println(prices)
    #     println(Features.pricediffregressionpeak(prices, rg; smoothing=false)...)
    # end
    dists = (Dists)(prices, regressiongradients)
    df = DataFrames.DataFrame()
    df[!, "prices"] = prices
    for i in eachindex(dists)
        df[!, "grad" * string(i, pad=2, base=10)] = regressiongradients[i]
        df[!, "pricediffs" * string(i, pad=2, base=10)] = dists[i].pricediffs
        df[!, "regressionix" * string(i, pad=2, base=10)] = dists[i].regressionix
        df[!, "priceix" * string(i, pad=2, base=10)] = dists[i].priceix
        df[!, "relativedist" * string(i, pad=2, base=10)] = dists[i].relativedist
    end
    result = Dists(zero(dists[1].pricediffs), zero(dists[1].regressionix), zero(dists[1].priceix), zero(dists[1].relativedist))
    rix = nothing
    for pix in eachindex(prices)
        nix = nothing
        for dix in eachindex(dists)
            if (dists[dix].relativedist[pix] > labelthresholds.longbuy) || (dists[dix].relativedist[pix] < labelthresholds.shortbuy)
                # dix exceeds threshold => rix is assigned to dix if priceix is closer or rix undefined
                if !isnothing(rix)
                    if sign(dists[dix].priceix[pix]) == sign(dists[rix].priceix[pix])
                        if (abs(dists[dix].priceix[pix]) < abs(dists[rix].priceix[pix])) &&  #dix peak is nearer than rix peak
                           (abs(dists[dix].relativedist[pix]) > (abs(dists[rix].relativedist[pix]) * 0.5))  # dix peak has at least half the gain of rix
                           rix = dix  # take over
                           # else rix remains unchanged
                        end
                    else  # buy peak in opposite direction
                        if (abs(dists[dix].priceix[pix]) < abs(dists[rix].priceix[pix]))  #dix peak is nearer than rix peak => take over and close
                            rix = dix
                            # else wait with take over until slope changes because short term slope is not in favor of gain => wait for better buy gain
                            # this also ensures that the open buy position will get close signals (i.e. close or opposite hold or opposite buy)
                        end
                    end
                else  # no rix assigned yet => take over
                    rix = dix
                end
            end
            # nix gets next extrema
            nix = !isnothing(nix) && (abs(dists[dix].priceix[pix]) > abs(dists[nix].priceix[pix])) ? nix : dix
        end
        if isnothing(rix)  # fallback to nix if rix still undefined
            result.pricediffs[pix] = dists[nix].pricediffs[pix]
            result.regressionix[pix] = dists[nix].regressionix[pix]
            result.priceix[pix] = dists[nix].priceix[pix]
            result.relativedist[pix] = dists[nix].relativedist[pix]
        else
            result.pricediffs[pix] = dists[rix].pricediffs[pix]
            result.regressionix[pix] = dists[rix].regressionix[pix]
            result.priceix[pix] = dists[rix].priceix[pix]
            result.relativedist[pix] = dists[rix].relativedist[pix]
        end
    end
    labels = getlabels(result.relativedist, labelthresholds)
    df[!, "result-pricediffs"] = result.pricediffs
    df[!, "result-regressionix"] = result.regressionix
    df[!, "result-priceix"] = result.priceix
    df[!, "result-relativedist"] = result.relativedist
    df[!, "result-labels"] = labels
    # println("result-pricediffs=$(result.pricediffs)")
    # println("result-regressionix=$(result.regressionix)")
    # println("result-priceix=$(result.priceix)")
    # println("result-relativedist=$(result.relativedist)")
return labels, result.relativedist, result.pricediffs, result.regressionix, result.priceix, df
end

"""
- returns for each index of prices
    - labels based on given thresholds that are compared with relative pricediffs
    - relative pricediffs related to the index under consideration (answering: what can be gained/lost from the current ix)
    - absolute pricediffs of current price to next extreme price
    - for debugging purposes 2 further index arrays are returned: with regression extreme indices and with price extreme indices
    - pricediffs will be negative if the next extreme is a minimum and positive if it is a maximum
- the extreme is determined by slope sign change of the regression gradients given in `regressions`
- from this regression extreme the peak is search backwards thereby skipping all local extrema that are insignifant for that regression window

"""
function continuousdistancelabels(prices::Vector{T}, regressiongradients::Vector{T}, labelthresholds::LabelThresholds) where {T<:Real}
    pricediffs, regressionix, priceix = Features.pricediffregressionpeak(prices, regressiongradients; smoothing=false)
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, regressionix, priceix
end

"Default relative transaction penalty of 1% on all transactions for fee and time lag, i.e. each open and each close, that is subtracted from gain."
defaultrelativetransactionpenalty = 0.01

function calcgaindetails(prices, xtrmix, relativetransactionpenalty)
    gain = zeros(Float64, length(xtrmix))
    lastix = firstindex(xtrmix)
    for ix in eachindex(xtrmix)
        gain[ix] = ix == lastix ? 0 : 0 #TODO
    end
end

mutable struct Gains
    pricediffs
    regressionix
    priceix
    relativedist
    # Dists(prices, pricediffs, regressionix, priceix) = new(pricediffs, regressionix, priceix, relativedistances(prices, pricediffs, priceix, false))
end

function (Dists)(prices::Vector{T}, regressiongradients::Vector{Vector{T}}) ::Vector{Dists} where {T<:Real}
    d = Array{Dists}(undef, length(regressiongradients))
    for ix in eachindex(d)
        pricediffs, regressionix, priceix = Features.pricediffregressionpeak(prices, regressiongradients[ix]; smoothing=false)
        relativedist = relativedistances(prices, pricediffs, priceix, false)
        d[ix] = Dists(pricediffs, regressionix, priceix, relativedist)
    end
    return d
end

mutable struct PriceExtremeCombi
    peakix  # vector of signed indices (positive for maxima, negative for minima)
    ix  # running index within peakix
    anchorix  # index of begin of peak sequence under assessment
    gain  # current cumulated gain since anchorix
    function PriceExtremeCombi(prices, regressionextremesix=Int64[])
        regr = regressionextremesix  # fill array of price extremes by backward search from regression extremes
        extremesix = [sign(regr(rix)) * Features.extremepriceindex(prices, abs(regr(rix)), abs(regr(rix-1)), (regr(rix) > 0)) for rix in firstindex(regr)+1:lastindex(regr)]
        return new(extremesix, firstindex(extremesix)+1, firstindex(extremesix), 0.0)
    end
end

function gain(prices, ix1, ix2, labelthresholds::LabelThresholds, relativetransactionpenalty)
    @assert labelthresholds.longbuy > 2 * relativetransactionpenalty
    @assert abs(labelthresholds.shortbuy) > 2 * relativetransactionpenalty
    g = Ohlcv.relativegain(prices, abs(ix1), abs(ix2))
    if g > 0
        g = g >= labelthresholds.longbuy ? g - relativetransactionpenalty : 0.0  # consider fee and time lag
    else
        g = g <= labelthresholds.shortbuy ? abs(g) - relativetransactionpenalty : 0.0  # consider fee and time lag
    end
    return g
end

"""
- Returns an array of best prices index extremes, an array of corresponding performance and a DataFrame with debug info
- relativetransactionpenalty (default=1%) is subtracted from the gain first at open and second at close
"""
function bestregressiontargetcombi(f2::Features.Features002, labelthresholds::LabelThresholds=defaultlabelthresholds, relativetransactionpenalty=defaultrelativetransactionpenalty)
    #! unit test to be added
    sortedxtrmix = sort([(k, length(f2.regr[k].xtrmix)) for k in keys(f2.regr)], by= x -> x[2], rev=true)  # sorted tuple of (key, length of xtremes),long arrays first
    prices = Ohlcv.dataframe(f2.ohlcv).Ohlcv.pivot
    @assert !isnothing(prices)
    combi = PriceExtremeCombi(f2.regr[sortedxtrmix[begin][1]].xtrmix)
    logdf = DataFrame([:combikey, :combipeakix, :combixix, :singlekey, :singlepeakix, :singlexix], [Int32, Int32, Int32, Int32, Int32])
    for six in (firstindex(sortedxtrmix)+1):lastindex(sortedxtrmix)
        newcombi = PriceExtremeCombi(prices)
        single = PriceExtremeCombi(f2.regr[sortedxtrmix[six][1]].xtrmix)
        while (combi.ix <= lastindex(combi.peakix)) || (single.ix <= lastindex(single.peakix))
            combi.gain = gain(prices, combi.peakix[combi.ix-1], combi.peakix[combi.ix], labelthresholds, relativetransactionpenalty) + combi.gain  # accumulate gain
            single.gain = gain(prices, single.peakix[single.ix-1], single.peakix[single.ix], labelthresholds, relativetransactionpenalty) + single.gain  # accumulate gain
            if abs(single.peakix[single.ix]) <= abs(combi.peakix[combi.ix])  # then the higher frequent combi catched up with the next extreme of single
                if combi.gain >= single.gain
                    if (lastindex(newcombi.peakix) > 0) && (abs(newcombi.peakix[end]) >= abs(combi.peakix[combi.anchorix]))  # one index overlap?
                        newcombi.peakix = vcat(newcombi.peakix, combi.peakix[combi.anchorix+1:combi.ix])
                    else
                        newcombi.peakix = vcat(newcombi.peakix, combi.peakix[combi.anchorix:combi.ix])
                    end
                else
                    if (lastindex(newcombi.peakix) > 0) && (abs(newcombi.peakix[end]) >= abs(single.peakix[single.anchorix]))  # one index overlap?
                        newcombi.peakix = vcat(newcombi.peakix, single.peakix[single.anchorix+1:single.ix])
                    else
                        newcombi.peakix = vcat(newcombi.peakix, single.peakix[single.anchorix:single.ix])
                    end
                end
                combi.gain = single.gain = 0.0
            end
            if abs(single.peakix[single.ix]) <= abs(combi.peakix[combi.ix])  # then the higher frequent combi catched up with the next extreme of single
                push!(logdf, (sortedxtrmix[six-1][1], combi.peakix[combi.ix], combi.ix, sortedxtrmix[six][1], single.peakix[single.ix], single.ix))
            end
            combi.anchorix = combi.ix
            combi.ix += 1
            single.anchorix = single.ix
            single.ix += 1
        end
    end
    if size(logdf, 1) > 0
        @warn "lower frequent extremes are not always a full subset of higher frequent extremes" logdf
    end
    return sortedxtrmix
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
