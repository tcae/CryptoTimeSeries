using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/env_config.jl")
include("../src/features.jl")

module Targets

import Pkg; Pkg.add(["JDF", "RollingFunctions"])
using DataFrames, Statistics
using ..Config, ..Features

strongsell=-2; sell=-1; hold=0; buy=1; strongbuy=2
gainthreshold = 0.01  # 1% gain threshold
lossthreshold = 0.005  # loss threshold to signal strong sell

"""
returns the relative forward/backward looking gain
"""
function relativegain(prices, firstix, secondix; forward=true)
    if forward
        gain = (prices[secondix] - prices[firstix]) / prices[firstix]
        # println("forward gain(prices[$firstix]= $(prices[firstix]) prices[$secondix]= $(prices[secondix]))=$gain")
    else
        gain = (prices[secondix] - prices[firstix]) / prices[secondix]
        # println("backward gain(prices[$firstix]= $(prices[firstix]) prices[$secondix]= $(prices[secondix]))=$gain")
    end
    return gain
end

"""
Assumption: the probability for a successful trade signal is higher at a certain slope threshold but it is unclear where this threshold is.
This function shall collect all buy/sell gradients versus all of their potential counter sell/buy gradients versus the gain between them as a histogram
to identify the slope where a majority of gains is above 1% gain.
To achieve this the gradients as well as the gaps need to be split into buckets.
"""
function gradientgaphistogram(prices, regressions, regbuckets=20, gainrange = 0.1)

    # prepare lookup tables
    @assert size(prices) == size(regressions) "sizes of prices and regressions shall match"
    # gainrange = 0.1  # 10% = 0.1 considered OK to cover -5% .. +5% gain range
    # gainrange = 0.02  # 2% = 0.02 considered OK to decided about gradients for target label
    gainstep = 0.01
    gainborders = [g for g in (-gainrange/2):gainstep:(gainrange/2)]
    # println("gainborders=$gainborders  length(gainborders)=$(length(gainborders))")
    regstep = 1 / (regbuckets - 1)
    # println("regstep=$regstep")
    regprobs = [rs for rs in (regstep/2):regstep:(1-regstep/2)]
    # println("regprobs=$regprobs")
    regquantiles = Statistics.quantile(skipmissing(regressions), regprobs)  # quantile border considers < vs. >=
    # println("regquantiles=$regquantiles  length(regquantiles)=$(length(regquantiles))")
    @assert regbuckets == (length(regquantiles) + 1 )

    histo = zeros(Int32, (regbuckets, regbuckets, length(gainborders)+1))
    gradpairgains = zeros(Union{Missing, Float32}, (regbuckets, regbuckets))
    gradpairgains .= missing  # init gradient pair gains
    endslopeix = 0


    #=
    gainborders and regquantiles are vectors to map value to array index and search function (Base.Sort.searchsortedfirst/searchsortedlast) to search index of a given gain or gradient.
    histo = 3 dim array: dim 1 for start gradient, dim 2 for end gradient, dim 3 for gain
    =#
    for sampleix in 2:length(regressions)
        if !(isequal(regressions[sampleix], missing) || isequal(regressions[sampleix-1], missing))
            if (regressions[sampleix] >= 0)
                sampleregix = searchsortedlast(regquantiles, regressions[sampleix]) + 1  # index of regression
                # println("sampleregix($sampleix, $(regressions[sampleix]))=$sampleregix")
                if (regressions[sampleix-1] < 0)  # minimum detected
                    gradpairgains .= missing  # init gradient pair gains
                    endslopeix = sampleix+1
                    while (endslopeix <= length(regressions)) && (regressions[endslopeix] >= 0) # no check required for missing
                        endslopeix += 1
                    end
                    endslopeix > length(regressions) ? break : true # for sampleix loop due to no more maximum
                    endslopeix -= 1
                end
                for ix in endslopeix:-1:(sampleix+1)
                    endregix = searchsortedlast(regquantiles, regressions[ix]) + 1  # index of regression
                    if isequal(gradpairgains[sampleregix, endregix], missing)
                        gain = relativegain(prices, sampleix, ix, forward=true)
                        gradpairgains[sampleregix, endregix] = gain  # overwrite any previous gain during this ascend
                        # println("gradpairgains[$sampleregix, $endregix] = $gain")
                    end
                end
            else  # (regressions[sampleix] < 0)
                if (regressions[sampleix-1] >= 0)  # maximum detected
                    # display(gradpairgains)
                    for startix in 1:regbuckets
                        for endix in 1:regbuckets
                            if !isequal(gradpairgains[startix, endix], missing)
                                gain = gradpairgains[startix, endix]
                                gainix = searchsortedlast(gainborders, gain) + 1
                                histo[startix, endix, gainix] += 1
                            end
                        end
                    end
                end
            end
        end
    end

    return (histo, regquantiles, gainborders)
end

function gradientthresholdlikelihoods(histo, regquantiles, gainborders, threshold=gainthreshold)
    regbuckets = length(regquantiles) + 1
    gainbuckets = length(gainborders) + 1
    @assert size(histo) == (regbuckets, regbuckets, gainbuckets)
    likelihoods = zeros(Float32, (regbuckets, regbuckets))
    gain1pctix = searchsortedlast(gainborders, threshold) + 1  # ix with gain >= threshold
    allsum = sum(histo)
    for startix in 1:regbuckets
        for endix in 1:regbuckets
            likelihoods[startix, endix] = sum(histo[startix, endix, gain1pctix:end]) / allsum
        end
    end
    return likelihoods
end

function tradegradientthresholds(prices, regressions, regbuckets=20, threshold=gainthreshold)
    histo, regquantiles, gainborders = Targets.gradientgaphistogram(prices, regressions, regbuckets)
    lh = Targets.gradientthresholdlikelihoods(histo, regquantiles, gainborders, threshold)
    max, ix = findmax(lh)
    startregr = ix[1]>1 ? regquantiles[ix[1]-1] : regquantiles[1]
    endregr = ix[2]>1 ? regquantiles[ix[2]-1] : regquantiles[1]
    return (startregr, endregr)
end

function regressionlabels1(prices, regressions)
    # prepare lookup tables
    @assert size(prices) == size(regressions) "sizes of prices and regressions shall match"
    startregr::Float32, endregr::Float32 = Targets.tradegradientthresholds(prices, regressions)
    # println("startregr=$startregr  endregr=$endregr")
    df = Features.lastgainloss(prices, regressions)
    # println("lastgain=$(df.lastgain)")
    # println("lastloss=$(df.lastloss)")
    @assert size(prices) == size(df.lastgain) == size(df.lastloss)
    targets = zeros(Int8, size(prices))
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
                while (regressions[ixbuystart] < startregr) && !stronglastgainloss && (ixbuystart < ixbuyend) && (relativegain(prices, ixbuystart, ixbuyend) >= gainthreshold)
                    # from minimum forward until steep enough slope detected
                    targets[ixbuystart] = buy
                    ixbuystart += 1
                end
                while ixbuystart <= ixbuyend  # label slope
                    if relativegain(prices, ixbuystart, ixbuyend) >= gainthreshold
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
        end
        if regressions[ix] > 0  # rising slope
            if (ixbuyend == 0)
                if (regressions[ix] < endregr)
                    # from maximum backward until steep enough slope detected
                    targets[ix] = sell
                else
                    # ixbuyend is last index that is a potential buy
                    ixbuyend = ix
                end
            end
        end
    end
    return targets
end

"""
    Targets are based on regressioin info, i.e. the gain between 2 horizontal regressions with at least 1% gain.
    However the underlining trend can already start before the horizontal regression is reached if the regressioin window is larger to smooth high frequent volatility.
    Therefore BUY shall be signaled before the minimum is reached as soon as the regression is monotonically increasing through the minimum.
    Correspondingly SELL shall be signaled before the maximum is reached as soon as the regression is monotonically decreasing through the maximum.

    Another aspect is that a signal label should represent an observable pattern, i.e. not solely based on a future pattern.
    Either the regression line exceeds a threshold that makes further increases more probable or recent maxima indicate likelihood to reach such level again.
    These are pre-requisite observables to label an increase as BUY tagrget.
"""
function targets(prices, regressions)

end

end  # module
