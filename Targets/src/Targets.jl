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
using DataFrames, Logging, Dates, SortingAlgorithms

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
    longbuy::Float32
    longhold::Float32
    shorthold::Float32
    shortbuy::Float32
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
    labels = [(rd >= lt.longbuy ? "longbuy" : (rd > lt.longhold ? "longhold" :
              (lt.shortbuy >= rd ? "shortbuy" : (lt.shorthold > rd ? "shorthold" : "close")))) for rd in relativedist]
    return labels
end

function relativedistances(prices::Vector{T}, pricediffs, priceix, backwardrelative=true) where {T<:Real}
    if backwardrelative
        relativedist = [(priceix[ix] == 0 ? T(0.0) : pricediffs[ix] / prices[abs(priceix[ix])]) for ix in 1:size(prices, 1)]
    else
        relativedist = pricediffs ./ prices
    end
end

function continuousdistancelabels(prices, labelthresholds::LabelThresholds)
    pricediffs, priceix = Features.nextpeakindices(prices, labelthresholds.longbuy, labelthresholds.shortbuy)
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

mutable struct Dists #! deprecated
    pricediffs
    regressionix
    priceix
    relativedist
    # Dists(prices, pricediffs, regressionix, priceix) = new(pricediffs, regressionix, priceix, relativedistances(prices, pricediffs, priceix, false))
end

function (Dists)(prices::Vector{T}, regressiongradients::Vector{Vector{T}}) ::Vector{Dists} where {T<:Real}
    #! deprecated
    @error "Targets.continuousdistancelabels is deprecated and replaced by Targets.bestregressiontargetcombi"
    return

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
function continuousdistancelabels(prices::Vector{T}, regressiongradients::Vector{Vector{T}}, labelthresholds::LabelThresholds) where {T<:AbstractFloat}
    #! deprecated
    @error "Targets.continuousdistancelabels is deprecated and replaced by Targets.bestregressiontargetcombi"
    return

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
function continuousdistancelabels(prices::Vector{T}, regressiongradients::Vector{T}, labelthresholds::LabelThresholds) where {T<:AbstractFloat}
    #! deprecated
    @error "Targets.continuousdistancelabels is deprecated and replaced by Targets.bestregressiontargetcombi"
    return

    pricediffs, regressionix, priceix = Features.pricediffregressionpeak(prices, regressiongradients; smoothing=false)
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, regressionix, priceix
end

mutable struct PriceExtremeCombi
    peakix  # vector of signed indices (positive for maxima, negative for minima)
    regrxtrmix
    ix  # running index within peakix
    anchorix  # index of begin of peak sequence under assessment
    gain  # current cumulated gain since anchorix
    regrwindow
    rw
    function PriceExtremeCombi(f2, regrwindow)
        if isnothing(f2)
            extremesix = Int64[]
            regr = Int64[]
            rw = String[]
        else
            prices = Ohlcv.dataframe(f2.ohlcv).pivot
            regr = f2.regr[regrwindow].xtrmix
            regr = [sign(regr[rix]) * Features.ohlcvix(f2, abs(regr[rix])) for rix in eachindex(regr)]  # translate to price index
            # println("PriceExtremeCombi regr=$regr")
            # fill array of price extremes by backward search from regression extremes
            extremesix = [sign(regr[rix]) * Features.extremepriceindex(prices, rix == lastindex(regr) ? abs(regr[rix]) : abs(regr[rix])-1 , rix == firstindex(regr) ? f2.firstix : abs(regr[rix-1]), (regr[rix] > 0)) for rix in eachindex(regr)]
            extremesix = Int64[]
            for rix in eachindex(regr)
                startrix = rix == lastindex(regr) ? abs(regr[rix]) : abs(regr[rix])-1
                endrix = rix == firstindex(regr) ? f2.firstix : abs(regr[rix-1])
                maxsearch = regr[rix] > 0
                peakindex = sign(regr[rix]) * Features.extremepriceindex(prices, startrix, endrix, maxsearch)
                @assert length(extremesix) > 0 ? abs(peakindex) > abs(last(extremesix)) : true
                push!(extremesix, peakindex)
            end
            if abs(last(extremesix)) < abs(last(regr))  # add end peak to have the end slope considered in gain
                push!(extremesix, -last(regr))
                push!(regr, -last(regr))
            end
            if abs(first(extremesix)) > firstindex(prices) # add start peak to have the first slope considered in gain
                pushfirst!(extremesix, -sign(first(extremesix)) * firstindex(prices))
                pushfirst!(regr, first(extremesix))
            end
            # println("PriceExtremeCombi extremesix=$extremesix")
            rw = repeat([string(regrwindow)], length(extremesix))
        end
        return new(extremesix, regr, firstindex(extremesix), firstindex(extremesix), 0.0, regrwindow, rw)
    end
end

function Base.show(io::IO, pec::PriceExtremeCombi)
    println(io, "PriceExtremeCombi(regrwindow=$(pec.regrwindow)) len(peakix)=$(length(pec.peakix)) ix=$(pec.ix) anchorix=$(pec.anchorix)")
end

function gain(prices::Vector{T}, ix1::K, ix2::K, labelthresholds::LabelThresholds) where {T<:AbstractFloat, K<:Integer}
    g = Ohlcv.relativegain(prices, abs(ix1), abs(ix2))
    if g > 0
        g = g >= labelthresholds.longbuy ? g : 0.0
    else
        g = g <= labelthresholds.shortbuy ? abs(g) : 0.0
    end
    return g
end

function gain(prices::Vector{T}, peakix::Vector{K}, labelthresholds::LabelThresholds) where {T<:AbstractFloat, K<:Integer}
    g = 0.0
    for i in eachindex(peakix)
        if i > firstindex(peakix)
            g += gain(prices, peakix[i-1], peakix[i], labelthresholds)
        end
    end
    return g
end

"""
- may change append.peakix[append.anchorix]
"""
function adaptappendsegmentconnection!(append, newcombi)
    if length(newcombi.peakix) == 0
        return
    end
    skip = 0
    while (abs(newcombi.peakix[end]) >= abs(append.peakix[append.anchorix+skip]) || (sign(newcombi.peakix[end]) == sign(append.peakix[append.anchorix+skip]))) && (append.anchorix+skip < append.ix)
        skip += 1
    end
    if (skip > 0)
        append.anchorix = append.anchorix + skip - 1
        append.peakix[append.anchorix] = newcombi.peakix[end]
    end
end

function copyover!(newcombi::PriceExtremeCombi, append::PriceExtremeCombi)
    eix = lastindex(newcombi.peakix)
    overlap = (length(newcombi.peakix) > 0) && (abs(newcombi.peakix[end]) >= abs(append.peakix[append.anchorix])) ? 1 : 0  # one index overlap?
    newcombi.peakix = vcat(newcombi.peakix, append.peakix[append.anchorix+overlap:append.ix])
    newcombi.regrxtrmix = vcat(newcombi.regrxtrmix, append.regrxtrmix[append.anchorix+overlap:append.ix])
    newcombi.rw = vcat(newcombi.rw, append.rw[append.anchorix+overlap:append.ix])
    # println("append.ix=$(append.ix) append.anchorix=$(append.anchorix) append.regrwindow=$(append.regrwindow) overlap=$overlap")
    # println("newcombi.peakix=$(newcombi.peakix) append.peakix=$(append.peakix)")
    # println("newcombi.regrxtrmix=$(newcombi.regrxtrmix) append.regrxtrmix=$(append.regrxtrmix)")
    # println("newcombi.rw=$(newcombi.rw) append.rw=$(append.rw)")
    if (eix > 0) && (eix < lastindex(newcombi.peakix))
        @assert sign(newcombi.peakix[eix]) != sign(newcombi.peakix[eix+1])
    end
    return newcombi
end

function reset!(pec::PriceExtremeCombi)
    pec.ix = firstindex(pec.peakix)
    pec.anchorix = firstindex(pec.peakix)
end

"merges the ix position and returns the better of the 2 PriceExtremeCombi"
function segmentgain(prices, pec::PriceExtremeCombi, labelthresholds::LabelThresholds=defaultlabelthresholds)
    g = 0.0
    for ix in (pec.anchorix+1):pec.ix
        g = gain(prices, pec.peakix[ix-1], pec.peakix[ix], labelthresholds) + g  # accumulate gain
    end
    return g
end

function nextextremepair!(combi, single)
    while (single.ix < lastindex(single.peakix)) && ((abs(single.peakix[single.ix]) <= abs(combi.peakix[combi.ix])) || (single.anchorix == single.ix))
        single.ix += 1
    end
    while (combi.ix < lastindex(combi.peakix)) && (abs(single.peakix[single.ix]) > abs(combi.peakix[combi.ix]))
        # println("bestregressiontargetcombi $six combi.ix=$(combi.ix)lastindex(combi.peakix)=$(lastindex(combi.peakix)) single.ix=$(single.ix) lastindex(single.peakix)=$(lastindex(single.peakix)) single.peakix[single.ix]=$(single.peakix[single.ix]) combi.peakix[combi.ix]=$(combi.peakix[combi.ix]) combi.peakix[combi.ix+1]=$(combi.peakix[combi.ix+1])")
        combi.ix += 1
    end
    # now combi catched up with the next extreme of single
    while (single.ix < lastindex(single.peakix)) && (abs(single.peakix[single.ix]) <= abs(combi.peakix[combi.ix]))
        if abs(single.peakix[single.ix+1]) > abs(combi.peakix[combi.ix])
            break
        else
            single.ix += 1
        end
    end
end

mutable struct ToporderElem
    pix   # peak index
    rwarr # signed regression window array (negative = minimum, positive = maximum)
end

mutable struct GraphElem
    pix   # peak (=extreme) index
    gain  # relative gain if above threshold otherwise 0.0
    rw    # signed regression window (negative = minimum, positive = maximum)
end

" Returns peakarr with each extreme index only once for a maximum and once for a minimum in sorted order as tuple of (positive extreme index, weight=Inf, array of signed regression minutes)"
function reducedpeakarrold(peakarr::Vector{ToporderElem}, startix)  # peakarr = [(extreme index, signed regression minutes),()], startix = index within peakarr
    if isnothing(startix) || startix < firstindex(peakarr)
        return vcat([ToporderElem(0, [0])], reducedpeakarr(peakarr, firstindex(peakarr)))
    end
    if startix > lastindex(peakarr)
        return [ToporderElem(last(peakarr).pix+1000000, [0])]
    end
    ix = startix
    pix = peakarr[startix].pix
    while (ix < lastindex(peakarr)) && (pix == peakarr[ix].pix)
        ix += 1
    end
    # create arrays of signed regression minutes from consecutive tuples with same pix
    rwarrmax = [first(peakarr[i].rwarr) for i in startix:ix if (pix == peakarr[i].pix) && (sign(first(peakarr[i].rwarr)) == 1)]
    rwarrmin = [first(peakarr[i].rwarr) for i in startix:ix if (pix == peakarr[i].pix) && (sign(first(peakarr[i].rwarr)) == -1)]
    ix = startix == lastindex(peakarr) ? ix+1 : ix
    if length(rwarrmax) > 0
        if length(rwarrmin) > 0
            peaktuplearr = vcat([ToporderElem(pix, rwarrmax)], [ToporderElem(-pix, rwarrmin)], reducedpeakarr(peakarr, ix))
        else
            peaktuplearr = vcat([ToporderElem(pix, rwarrmax)], reducedpeakarr(peakarr, ix))
        end
    else
        if length(rwarrmin) > 0
            peaktuplearr = vcat([ToporderElem(-pix, rwarrmin)], reducedpeakarr(peakarr, ix))
        end
    end
    return peaktuplearr
end

" Returns peakarr with each extreme index only once for a maximum and once for a minimum in sorted order as tuple of (positive extreme index, weight=Inf, array of signed regression minutes)"
function reducepeakarr(peakarr::Vector{ToporderElem}, startix)  # peakarr = [(extreme index, signed regression minutes),()], startix = index within peakarr
    reducedpeakarr = [ToporderElem(0, [0])]
    startix = firstindex(peakarr)
    while startix <=lastindex(peakarr)
        endix = startix
        pix = peakarr[startix].pix
        while (endix < lastindex(peakarr)) && (pix == peakarr[endix].pix)
            endix += 1
        end
        # create arrays of signed regression minutes from consecutive tuples with same pix
        rwarrmax = [first(peakarr[i].rwarr) for i in startix:endix if (pix == peakarr[i].pix) && (sign(first(peakarr[i].rwarr)) == 1)]
        if length(rwarrmax) > 0
            push!(reducedpeakarr, ToporderElem(pix, rwarrmax))
        end
        rwarrmin = [first(peakarr[i].rwarr) for i in startix:endix if (pix == peakarr[i].pix) && (sign(first(peakarr[i].rwarr)) == -1)]
        if length(rwarrmin) > 0
            push!(reducedpeakarr, ToporderElem(-pix, rwarrmin))
        end
        startix += length(rwarrmax) + length(rwarrmin)
        @assert length(rwarrmax) + length(rwarrmin) > 0 "peakarr=$peakarr startix=$startix endix=$endix"
    end
    ToporderElem(last(reducedpeakarr).pix+1000000, [0])
    return reducedpeakarr
end

graphelemcomparison(a, b) = (a.pix == b.pix) && (a.rw == b.rw)

notinarray(array, element) = isnothing(findfirst(x -> graphelemcomparison(x, element), array))

function peaksuccessors(rwset::Set, peaktuplearr::Vector{ToporderElem}, startix, prices, labelthresholds::LabelThresholds)
    tuplearr = []
    pix = abs(peaktuplearr[startix].pix)
    @assert all(x -> x == sign(first(peaktuplearr[startix].rwarr)), sign.(peaktuplearr[startix].rwarr))  # all of same sign?
    pixsign = sign(first(peaktuplearr[startix].rwarr))
    rws = copy(rwset)
    ix = startix + 1
    while (ix <= lastindex(peaktuplearr)) && (!isempty(rws))
        if pix < abs(peaktuplearr[ix].pix)
            for rw in peaktuplearr[ix].rwarr
                if ((abs(rw) in rws) || (rw == 0)) && (pixsign != sign(rw))  # pixsign == 0 will be connected to maxima and minima
                    delete!(rws, abs(rw))
                    if  (ix < firstindex(peaktuplearr)) || (ix > lastindex(peaktuplearr)) ||
                        (pix < firstindex(prices)) || (pix > lastindex(prices)) ||
                        (peaktuplearr[ix].pix < firstindex(prices)) || (peaktuplearr[ix].pix > lastindex(prices))
                        g = 0.000001  # very small gain to incentive reproducable behavior - it also supports longer instead of shorter paths
                    else
                        g = gain(prices, pix, peaktuplearr[ix].pix, labelthresholds)
                    end
                    push!(tuplearr, GraphElem(peaktuplearr[ix].pix, g, rw))
                end
            end
        end
        ix += 1
    end
    return tuplearr  # of format (positive prices index of extreme, gain, signed regression minutes depending of max/min)
end

function computebestpath(graph, toporder::Vector{ToporderElem}, startpix, maxgain)
    # vertex is an extreme index (a.k.a pix)
    distances = Dict(pix => 0.0 for pix in keys(graph))
    predecessors = Dict{Integer, Union{Nothing, ToporderElem}}(pix => nothing for pix in keys(graph))
    distances[startpix] = maxgain

    for vertex in toporder
        for ge in graph[vertex.pix]
            if distances[vertex.pix] + ge.gain > distances[ge.pix]
                distances[ge.pix] = distances[vertex.pix] + ge.gain
                predecessors[ge.pix] = vertex
            end
        end
    end
    return distances, predecessors
end

function tracepath(predecessors::Dict{Integer, Union{Nothing, Targets.ToporderElem}}, startvertex::Targets.ToporderElem, lastvertex::Targets.ToporderElem)
    path = []
    current_vertex = lastvertex
    while !isnothing(current_vertex) && (current_vertex.pix != startvertex.pix)
        pushfirst!(path, current_vertex)
        current_vertex = predecessors[current_vertex.pix]
    end
    pushfirst!(path, startvertex)
    return path
end

" Returns an array of best prices index extremes, an array of corresponding performance and a DataFrame with debug info "
function bestregressiontargetcombi2(f2::Features.Features002, labelthresholds::LabelThresholds=defaultlabelthresholds)
    debug = false
    @debug "Targets.bestregressiontargetcombi2" begin
        debug = true
    end
    maxgain = 100000.0
    prices = Ohlcv.dataframe(f2.ohlcv).pivot
    @assert !isnothing(prices)
    # pec = Dict(rw => PriceExtremeCombi(f2, rw) for rw in keys(f2.regr))
    pec = Dict()
    for rw in keys(f2.regr)
        pec[rw] = PriceExtremeCombi(f2, rw)
    end

    peakarr = sort([ToporderElem(abs(pix), [sign(pix) * rw]) for rw in keys(pec) for pix in pec[rw].peakix], by = x -> x.pix, rev=false)
    # peakarr contains now a sorted list of tuple of (positive extreme indices, signed regression minutes depending of max/min)
    # multiple equal peak indices can be in peakarr from different regression windows

    reducedpeakarr = reducepeakarr(peakarr, nothing)
    # now reducedpeakarr has each extreme index only once in sorted order as tuple of (positive extreme index, weight=Inf, array of signed regression minutes)

    rwset = Set([rw for rw in keys(pec)])
    peakgraph = Dict()
    for ix in eachindex(reducedpeakarr)
        pix = reducedpeakarr[ix].pix
        psucc = peaksuccessors(rwset, reducedpeakarr, ix, prices, labelthresholds)
        peakgraph[pix] = psucc
    end
    _, predecessors = computebestpath(peakgraph, reducedpeakarr, first(reducedpeakarr).pix, maxgain)
    bestpath = tracepath(predecessors, first(reducedpeakarr), last(reducedpeakarr))
    @debug "ToporderElem {pix = peak index in prices, rwarr = regressions windows} - = minimum, + = maximum" bestpath
    peakix = [toe.pix for toe in bestpath if firstindex(prices) <= abs(toe.pix) <= lastindex(prices)]
    if debug
        for (rw, pec) in pec
            pec.anchorix = firstindex(pec.peakix)
            pec.ix = lastindex(pec.peakix)
            pec.gain = segmentgain(prices, pec, labelthresholds)
        end
        g = gain(prices, peakix, labelthresholds)
        println("gains of regression window: $([(rw, p.gain) for (rw, p) in pec]) combi gain: $g")
    end
    return peakix
end

function continuousdistancelabels2(f2::Features.Features002, labelthresholds::LabelThresholds=defaultlabelthresholds)
    peakix = bestregressiontargetcombi2(f2, labelthresholds)
    labels, relativedist, pricediffs, priceix = ohlcvlabels(Ohlcv.dataframe(f2.ohlcv).pivot, peakix, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

" Returns an array of best prices index extremes and an arry of regressionextremes "
function bestregressiontargetcombi(f2::Features.Features002, labelthresholds::LabelThresholds=defaultlabelthresholds)
    debug = false
    @debug "Targets.bestregressiontargetcombi" begin
        debug = true
    end
    sortedxtrmix = sort([(k, length(f2.regr[k].xtrmix)) for k in keys(f2.regr)], by= x -> x[2], rev=true)  # sorted tuple of (key, length of xtremes),long arrays first
    prices = Ohlcv.dataframe(f2.ohlcv).pivot
    @assert !isnothing(prices)
    combi = PriceExtremeCombi(f2, first(sortedxtrmix)[1])
    debug ? gains = [("single+" * Features.periodlabels(combi.regrwindow), gain(prices, combi.peakix, labelthresholds))] : 0
    debug ? df = DataFrame() : 0
    debug ? df[:, "single+" * Features.periodlabels(combi.regrwindow)] = vcat(combi.peakix, repeat([0], length(combi.peakix))) : 0
    debug ? df[:, "sregrxtrmix+" * Features.periodlabels(combi.regrwindow)] = vcat(combi.regrxtrmix, repeat([0], length(df[!, 1]) - length(combi.regrxtrmix))) : 0
    debug ? df[:, "sregrwin+" * Features.periodlabels(combi.regrwindow)] = vcat(combi.rw, repeat([0], length(df[!, 1]) - length(combi.rw))) : 0
    for six in (firstindex(sortedxtrmix)+1):lastindex(sortedxtrmix)
        newcombi = PriceExtremeCombi(nothing, "combi+" * Features.periodlabels(sortedxtrmix[six][1]))
        single = PriceExtremeCombi(f2, sortedxtrmix[six][1])
        debug ? push!(gains, ("single+" * Features.periodlabels(single.regrwindow), gain(prices, single.peakix, labelthresholds))) : 0
        reset!(combi)
        debug ? df[:, "single+" * Features.periodlabels(single.regrwindow)] = vcat(single.peakix, repeat([0], length(df[!, 1]) - length(single.peakix))) : 0
        debug ? df[:, "sregrxtrmix+" * Features.periodlabels(single.regrwindow)] = vcat(single.regrxtrmix, repeat([0], length(df[!, 1]) - length(single.regrxtrmix))) : 0
        debug ? df[:, "sregrwin+" * Features.periodlabels(single.regrwindow)] = vcat(single.rw, repeat([0], length(df[!, 1]) - length(single.rw))) : 0
        while (combi.ix < lastindex(combi.peakix)) && (single.ix < lastindex(single.peakix))
            nextextremepair!(combi, single)
            @assert (combi.ix == lastindex(combi.peakix)) || (abs(single.peakix[single.ix]) <= abs(combi.peakix[combi.ix]))
            # now single catched up with combi and staying just one ix behind
            adaptappendsegmentconnection!(single, newcombi)
            adaptappendsegmentconnection!(combi, newcombi)
            # println("bestregressiontargetcombi2 $six combi.ix=$(combi.ix)lastindex(combi.peakix)=$(lastindex(combi.peakix)) single.ix=$(single.ix) lastindex(single.peakix)=$(lastindex(single.peakix)) ")

            combigain = segmentgain(prices, combi, labelthresholds)
            singlegain = segmentgain(prices, single, labelthresholds)
            if (combigain != 0.0f0) || (singlegain != 0.0f0)  # prefer better gain
                newcombi = combigain >= singlegain ? copyover!(newcombi, combi) : copyover!(newcombi, single)
            else  # if both gains == 0.0 then prefer higher frequency of peaks
                newcombi = (combi.ix - combi.anchorix) > (single.ix - single.anchorix) ? copyover!(newcombi, combi) : copyover!(newcombi, single)
            end
            combi.anchorix = combi.ix
            single.anchorix = single.ix
        end
        # println("bestregressiontargetcombi3 $six combi.ix=$(combi.ix)lastindex(combi.peakix)=$(lastindex(combi.peakix)) single.ix=$(single.ix) lastindex(single.peakix)=$(lastindex(single.peakix)) ")
        if combi.ix <= lastindex(combi.peakix)
            # println("combi finish combi=$combi")
            @assert single.ix == lastindex(single.peakix)
            combi.ix = lastindex(combi.peakix)
            copyover!(newcombi, combi)
        elseif single.ix <= lastindex(single.peakix)
            # println("combi finish combi=$combi")
            @assert combi.ix == lastindex(combi.peakix)
            single.ix = lastindex(single.peakix)
            copyover!(newcombi, single)
        else
            # matching last element of combi and single
        end
        @debug combi, single, newcombi
        combi = newcombi
        debug ? push!(gains, (string(combi.regrwindow), gain(prices, combi.peakix, labelthresholds))) : 0
        debug ? df[:, "combi+" * Features.periodlabels(sortedxtrmix[six][1])] = vcat(combi.peakix, repeat([0], length(df[!, 1]) - length(combi.peakix))) : 0
        debug ? df[:, "cregrxtrmix+" * Features.periodlabels(sortedxtrmix[six][1])] = vcat(combi.regrxtrmix, repeat([0], length(df[!, 1]) - length(combi.regrxtrmix))) : 0
        debug ? df[:, "cregrwin+" * Features.periodlabels(sortedxtrmix[six][1])] = vcat(combi.rw, repeat([0], length(df[!, 1]) - length(combi.rw))) : 0
        debug ? df[:, "prices+" * Features.periodlabels(sortedxtrmix[six][1])] = vcat([prices[abs(i)] for i in combi.peakix], repeat([0], length(df[!, 1]) - length(combi.peakix))) : 0
        # debug ? println(combi, single, newcombi) : 0
        @assert all([(sign(combi.peakix[i]) == -sign(combi.peakix[i-1])) for i in eachindex(combi.peakix) if (i > firstindex(combi.peakix)) && (combi.peakix[i] != 0)])
    end
    @debug "" gains
    # @debug "" df
    debug ? println(df) : 0
    return combi.peakix, combi.regrxtrmix
end

function ohlcvlabels(prices::Vector{T}, pricepeakix::Vector{S}, labelthresholds::LabelThresholds=defaultlabelthresholds) where {T<:AbstractFloat, S<:Integer}
    pricediffs = zeros(Float32, length(prices))
    priceix = zeros(Int32, length(prices))
    pix = firstindex(pricepeakix)
    for ix in eachindex(prices)
        if (ix >= abs(pricepeakix[pix]) && (pix < lastindex(pricepeakix)))
            pix = pix + 1
            priceix[ix] = pricepeakix[pix]
        elseif (pix <= lastindex(pricepeakix)) && (ix <= abs(pricepeakix[pix]))
            priceix[ix] = pricepeakix[pix]
        else
            priceix[ix] = -sign(pricepeakix[pix]) * lastindex(priceix)
        end
        pricediffs[ix] = prices[abs(priceix[ix])]  - prices[ix]
        # println("ix=$ix prices[ix]=$(prices[ix]) priceix[ix]=$(priceix[ix]) prices[abs(priceix[ix])]=$(prices[abs(priceix[ix])])")
    end
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

function continuousdistancelabels(f2::Features.Features002, labelthresholds::LabelThresholds=defaultlabelthresholds)
    peakix, _ = bestregressiontargetcombi(f2, labelthresholds)
    labels, relativedist, pricediffs, priceix = ohlcvlabels(Ohlcv.dataframe(f2.ohlcv).pivot, peakix, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end

function fakef2fromarrays(prices::Vector{T}, regressiongradients::Vector{Vector{T}}) where {T<:AbstractFloat}
    ohlcv = Ohlcv.defaultohlcv("BTC")
    Ohlcv.setdataframe!(ohlcv, DataFrame(opentime=zeros(DateTime, length(prices)), open=prices, high=prices, low=prices, close=prices, basevolume=prices, pivot=prices))
    regr = Dict()
    for window in eachindex(regressiongradients)
        regr[window] = Features.Features002Regr(regressiongradients[window], [1.0f0], [1.0f0], Features.regressionextremesix!(nothing, regressiongradients[window], 1))
    end
    f2 = Features.Features002(ohlcv, regr, Dict(1 =>[1.0f0]), fakef2fromarrays, firstindex(prices), lastindex(prices), 0)
    return f2
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

"Function to perform topological sort on a graph that is described via a dict"
function topological_sort(graph)
    visited = Set()
    top_order = []

    function dfs(vertex)
        if vertex in visited
            return
        end
        push!(visited, vertex)
        for neighbor in graph[vertex]
            dfs(neighbor[1])
        end
        push!(top_order, vertex)
    end

    for vertex in keys(graph)
        dfs(vertex)
    end

    return reverse(top_order)
end

"""
Finding the best path in an acyclic directed graph is a well-defined problem. If the graph is acyclic, it means that there are no cycles (i.e., no closed loops) in the graph. One common algorithm used to find the best path in such a graph is the topological sort algorithm.

Here are the general steps to compute the best path in an acyclic directed graph:

    Topological Sort:
        Perform a topological sort of the vertices in the graph. Topological sorting is a linear ordering of the vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.

    Initialize Distances:
        Initialize the distances from the source vertex to all other vertices to a large value (infinity for practical purposes) except for the source vertex itself, which is initialized to 0.

    Relaxation:
        For each vertex u in the topologically sorted order, relax all the adjacent vertices v of u. Relaxation involves checking if the path through u to v is shorter than the current known path to v. If it is, update the distance to v.

    Shortest Path:
        After the topological sort and relaxation steps, the distances to all vertices from the source vertex will represent the shortest path.

Function to compute the best path and predecessors in an acyclic directed graph
"""
function compute_best_path_test(graph, source)
    top_order = topological_sort(graph)
    distances = Dict(vertex => Inf for vertex in keys(graph))
    predecessors = Dict{String, Union{Nothing, String}}(vertex => nothing for vertex in keys(graph))
    distances[source] = 0

    for vertex in top_order
        for (neighbor, weight) in graph[vertex]
            if distances[vertex] + weight < distances[neighbor]
                distances[neighbor] = distances[vertex] + weight
                predecessors[neighbor] = vertex
            end
        end
    end

    return distances, predecessors
end

# Function to trace the best path from source to target
function trace_path_test(predecessors, source, target)
    path = []
    current_vertex = target
    while current_vertex != source
        pushfirst!(path, current_vertex)
        current_vertex = predecessors[current_vertex]
    end
    pushfirst!(path, source)
    return path
end

function test_trace_path()
    # Example usage:
    graph = Dict(
        "A" => [("B", 1), ("C", 2)],
        "B" => [("D", 3)],
        "C" => [("D", 3)],
        "D" => []
    )

    source_vertex = "A"
    target_vertex = "D"

    distances, predecessors = compute_best_path_test(graph, source_vertex)
    best_path = trace_path_test(predecessors, source_vertex, target_vertex)

    println("Distances from $source_vertex: ", distances)
    println("Best path from $source_vertex to $target_vertex: ", best_path)
end

end  # module

# Targets.pathtest()
