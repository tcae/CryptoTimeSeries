"""
Provides strategies to translate classifier predictions into trading actions.

- classifier shall provide
  - score acceptance thresholds for each label
  - mingain and maxgain per sample as limit inidications
  - maxwindow of predictions
  - maxpredictionlabel, maxscore, set, symbol (coin) per sample
- 
"""
module TradingStrategy

using DataFrames, Dates
using EnvConfig, Ohlcv, Features, Targets

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

"Defines the single symbol trading interface that shall be provided by all trading strategy implementations."
abstract type AbstractSingleSymbolTrading <: EnvConfig.AbstractConfiguration end

"Defines the multi symbol trading interface that shall be provided by all trading strategy implementations."
abstract type AbstractMultiSymbolTrading <: EnvConfig.AbstractConfiguration end

"Adds a coin with OhlcvData to the target generation. Each coin can only have 1 associated data set."
function setbase!(targets::AbstractSingleSymbolTrading, ohlcv::Ohlcv.OhlcvData) error("not implemented") end

function _mapleft!(d, tl, tlix, tll)
    for ix in tlix:-1:firstindex(tll)
        if tll[ix] in keys(d)
            d[tl] = d[tll[ix]] # mapped
            break
        end
    end
end

function _mapright!(d, tl, tlix, tll)
    for ix in tlix:lastindex(tll)
        if tll[ix] in keys(d)
            d[tl] = d[tll[ix]] # mapped
            break
        end
    end
end

"""
Transforms a named tuple of trade label names with score thresholds and returns a dict of it, which is subsequently used to look up a corresponding score threshold.  
Any missing trade label that is found in the predictions is mapped to a neighbor trade label.
"""
function unused_scoreacceptnt2dict(scorethreshold)::Dict
    d = Dict() # Dict([(tl, nothing) for tl in Targets.tradelabels()])
    (verbosity >= 3) && println(scorethreshold)
    tls = keys(scorethreshold) 
    for tlsix in eachindex(tls)
        d[Targets.tradelabel(string(tls[tlsix]))] = scorethreshold[tls[tlsix]]
        (verbosity >= 3) && println("d[Targets.tradelabel(string(tls[tlsix]))=$(Targets.tradelabel(string(tls[tlsix])))] = scorethreshold[tls[tlsix]]=$(scorethreshold[tls[tlsix]])")
    end
    (verbosity >= 3) && println("dict of named tuple: $d")
    # map missing values to provided values
    tll = Targets.tradelabels()
    tli = Int.(tll)
    for tl in tll
        if tl == ignore
            continue
        end
        if !(tl in keys(d))
            tlix = findfirst(x -> x == tl, tll)
            if Int(tl) < 0 # short
                _mapright!(d, tl, tlix, tll)
                if !(tl in keys(d))
                    _mapleft!(d, tl, tlix, tll)
                end
            else # long
                _mapleft!(d, tl, tlix, tll)
                if !(tl in keys(d))
                    _mapright!(d, tl, tlix, tll)
                end
            end
        end
    end
    (verbosity >= 3) && println("extended dict of named tuple including missing trade label score thresholds: $d")
    return d
end

"""
Transforms a named tuple of trade label names with score thresholds and returns a dict of it, which is subsequently used to look up a corresponding score threshold.  
Any missing trade label that is found in the predictions will result in a dict key error.
"""
function scorethresholdnt2dict(scorethreshold)::Dict
    d = Dict() # Dict([(tl, nothing) for tl in Targets.tradelabels()])
    (verbosity >= 4) && println(scorethreshold)
    tls = keys(scorethreshold) 
    for tlsix in eachindex(tls)
        d[Targets.tradelabel(string(tls[tlsix]))] = scorethreshold[tls[tlsix]]
        (verbosity >= 4) && println("d[Targets.tradelabel(string(tls[tlsix]))=$(Targets.tradelabel(string(tls[tlsix])))] = scorethreshold[tls[tlsix]]=$(scorethreshold[tls[tlsix]])")
    end
    (verbosity >= 4) && println(d)
    return d
end

# println("scorethresholdnt2dict((longbuy=0.8, allclose=0.5, shortbuy=0.7)) = $(scorethresholdnt2dict((longbuy=0.8,  allclose=0.5, shortbuy=0.7)))")

"""
This struct is used to convey input data to the selected trading startegy and to receive trade action instructions.  
Considers the price development of one symbol at a specific minute.  
The absolute amount is not determined by trading strategy but a factor of an unknown amount.

  - as input a short/long buy/close order can be open
    - if within limit price range it is being executed
    - it may be canceled
    - the limit may be changed
  - as output
    - a new short/long buy/close limit order may be created
    - a new short/long buy/close market order may be created
  
"""
mutable struct TradeAction
    cancelrunningorder::Bool
    orderlabel::Union{TradeLabel, Nothing} # used: longbuy, longclose, shortclose, shortbuy
    orderlimit::Union{Float32, Nothing} # order limit or nothing if no limit order placed
    amountfactor::Float32
end

mutable struct GainSegment
    algorithm  # algorithm function to be applied
    openthreshold::Float32 # score threshold
    closethreshold::Float32 # score threshold
    # scorethreshold::Dict # score threshold dict
    maxwindow::Integer # maxwindow of sample window considered for the prediction
    endix::Integer # is the index of the last analyzed row of scores/labels/predictionsdf
    gaindf::DataFrame
    makerfee::Float32
    takerfee::Float32
    scores::Union{AbstractVector, Nothing}
    labels::Union{AbstractVector, Nothing}
    predictionsdf::Union{AbstractDataFrame, Nothing}
    buyix::Integer # buy index of last open gain segment that is not yet saved in gainsdf
    lastix::Integer # last inspected row index of last open gain segment that is not yet saved in gainsdf
    trend::TrendPhase # trend of last 
    limit::Union{Float32, Nothing} # order limit or nothing if no limit order placed
    buygain::Float32 # relative gain compared to current price for limit order
    sellgain::Float32 # relative gain compared to current price for limit order
    buyprice::Union{Float32, Nothing} # buy price of open gain segment
    ordertype::TradeLabel # used: longbuy, longclose, allclose (default), shortclose, shortbuy
    assettype::TrendPhase # indicates whether current assets are not present (flat), short (down), long (up)
    function GainSegment(;maxwindow::Integer, openthreshold, closethreshold, algorithm=algorithm02!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0)
        return new(algorithm, openthreshold, closethreshold, maxwindow, 0, DataFrame(), makerfee, takerfee, nothing, nothing, nothing, 0, 0, flat, nothing, 0.01f0, 0.005f0, nothing, allclose, flat)
    end
end

function reset!(gs::GainSegment)
    gs.predictionsdf = nothing
    gs.scores = nothing
    gs.labels = nothing
    gs.endix = gs.lastix = gs.buyix = 0
    gs.trend = flat
    gs.limit = nothing
    gs.buyprice = nothing
    gs.ordertype = allclose
    gs.assettype = flat
    gs.gaindf = DataFrame()
end

function isopengainsegment(gs::GainSegment)
    return (gs.buyix > 0)
end

function calcgain!(gs::GainSegment, sellix::Integer, sellprice::Float32=gs.predictionsdf[sellix, :close])
    if isopengainsegment(gs)
        @assert !isnothing(gs.buyprice)  "inconsistency: gs.buyix=$(gs.buyix), gs.buyprice=$(gs.buyprice)"
        starttime = gs.predictionsdf[gs.buyix, :opentime]
        ixtime = gs.predictionsdf[sellix, :opentime]
        @assert gs.assettype in [up, down] "if gs.trend=$(gs.assettype) == flat then buyix=$(gs.buyix) == 0"
        if gs.assettype == up
            # long sell
            startprice = gs.buyprice # * (1 + gs.makerfee)
            ixprice = sellprice # * (1 - gs.makerfee)
            gain = (ixprice - startprice) / startprice
        elseif gs.assettype == down
            # short sell
            startprice = gs.buyprice # * (1 - gs.makerfee)
            ixprice = sellprice # * (1 + gs.makerfee)
            gain = -(ixprice - startprice) / startprice  # down trend -> negative price diff -> gain shall be positive for a short trade
        end
        push!(gs.gaindf, (trend=gs.trend, samplecount=(sellix-gs.buyix+1), minutes=Minute(ixtime-starttime).value + 1, gain=gain, gainfee=(gain - 2 * gs.makerfee), startdt=starttime, enddt=ixtime, startix=gs.buyix, endix=sellix))
        gs.buyix = 0  # indicates closure of gain segment
    end
end

"""
Consumes the next prediction results after endix for all remaining predictions and calculates the corresponding gain segments that are stored in gaindf.  
Hold as long as trend is supported above closethreshold by hold, buy, strongbuy, otherwise close
"""
function algorithm01!(gs::GainSegment, lastix)
    # @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    for ix in (gs.endix+1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        thistrend = flat
        thisbuyix = 0
        if (gs.trend == up) && (gs.labels[ix] in [longhold, longbuy, longstrongbuy]) && (gs.scores[ix] > gs.closethreshold)
            thistrend = up
        elseif (gs.labels[ix] in [longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif (gs.trend == down) && (gs.labels[ix] in [shorthold, shortbuy, shortstrongbuy]) && (gs.scores[ix] > gs.closethreshold)
            thistrend = down
        elseif (gs.labels[ix] in [shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if gs.trend != thistrend
            calcgain!(gs, ix)
            gs.buyix = thisbuyix
            gs.trend = thistrend
            if isopengainsegment(gs)
                gs.buyprice = gs.predictionsdf[gs.buyix, :close]
                gs.assettype = thistrend
            else
                gs.buyprice = nothing
                gs.assettype = flat
            end
        end
    end
    gs.endix = lastix
end

"""
Consumes the next prediction results after endix for all remaining predictions and calculates the corresponding gain segments that are stored in gaindf.  
Hold as long as trend is not broken by opposite buy or close above openthreshold.
"""
function algorithm02!(gs::GainSegment, lastix)
    # @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    for ix in (gs.endix+1):lastix
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        thistrend = gs.trend
        thisbuyix = 0
        if (gs.trend in [up]) && (gs.labels[ix] in [allclose, longstrongclose, longclose]) && (gs.scores[ix] > gs.openthreshold)
            thistrend = flat
        elseif (gs.trend in [down]) && (gs.labels[ix] in [shortclose, shortstrongclose, allclose]) && (gs.scores[ix] > gs.openthreshold)
            thistrend = flat
        elseif (gs.labels[ix] in [longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif (gs.labels[ix] in [shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end
        if gs.trend != thistrend
            calcgain!(gs, ix)
            gs.buyix = thisbuyix
            gs.trend = thistrend
            if isopengainsegment(gs)
                gs.buyprice = gs.predictionsdf[gs.buyix, :close]
                gs.assettype = thistrend
            else
                gs.buyprice = nothing
                gs.assettype = flat
            end
        end
    end
    gs.endix = lastix
end

function processclosedorder!(gs::GainSegment, ix::Integer)
    if !isnothing(gs.limit) && (gs.predictionsdf[gs.buyix, :low] <= gs.limit <= gs.predictionsdf[gs.buyix, :high]) 
        # assume trade took place
        if gs.assettype in [up, down]
            # opposite trend asset was sold together with asset buy trade
            calcgain!(gs, ix, gs.limit)
        end
        if gs.ordertype in [longbuy, shortbuy]
            if gs.ordertype == longbuy
                if gs.assettype == down
                    # short trend asset was sold together with long asset buy trade
                    calcgain!(gs, ix, gs.limit)
                end
                gs.assettype = up
            elseif gs.ordertype == shortbuy
                if gs.assettype == up
                    # long trend asset was sold together with short asset buy trade
                    calcgain!(gs, ix, gs.limit)
                end
                gs.assettype = down
            end
            gs.buyprice = gs.limit
            gs.buyix = ix
        else
            if gs.ordertype in [longclose, shortclose]
                @assert gs.assettype in [up, down]
                calcgain!(gs, ix, gs.limit)
            else
                @assert gs.ordertype in [longbuy, shortbuy, longclose, shortclose] "unexpected gs.ordertype=$(gs.ordertype) although gs.limit=$(gs.limit)"
            end
            gs.assettype = flat
            gs.buyprice = nothing
            gs.buyix = 0
        end
        gs.limit = nothing # order is closed
    end
end

"""
Consumes the next prediction results after endix for all remaining predictions and calculates the corresponding gain segments that are stored in gaindf.  
Buy will buy at current price +- buygain.  
Hold as long as trend is not broken by opposite buy.  
A close will reduce the initial sel limit of current price +- sellgain selllimit by 5% with each non trend supporting sample.
"""
function algorithm03!(gs::GainSegment, lastix)
    ta = TradeAction(false, nothing, nothing, 1f0)
    limitreduction = 1/20 
    # @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    for ix in (gs.endix+1):lastix
        # first check whether the limit order is executed, if so, set trend and buyix and selllimit
        ta = processclosedorder!(gs, ix)
        #TODO decide how long a still open limit order should stay and what limit corrections are required
        # limitorder can stay open (not yet filled, gs.limit != nothing) => check adjustment of limit
        # no limitorder in place => check whether to set a limitorder
        """
        (short/long) types of limitorder:  
        - buy due to buy signal
        - continuelimitorder due to previous buy signal but not now (and no other buy signal), i.e. adjust limit
        - sell due to sell signal
        """
        thistrend = gs.trend # assume trend continues and correct this assumption if required
        thisbuyix = 0
        if (gs.trend == up) && (gs.labels[ix] in [allclose, longstrongclose, longclose]) && (gs.scores[ix] > gs.openthreshold) && (gs.assettype == up)  
            # either modify close limit of running order or create new close order for long position
            ta.orderlabel = gs.ordertype = longclose
            ta.orderlimit = gs.limit = max(gs.predictionsdf[ix, :close], gs.limit * (1f0 - limitreduction))
            thistrend = flat
        elseif (gs.trend == down) && (gs.labels[ix] in [shortclose, shortstrongclose, allclose]) && (gs.scores[ix] > gs.openthreshold) && (gs.assettype == down)
            # either modify close limit of running order or create new close order for long position
            ta.orderlabel = gs.ordertype = shortclose
            ta.orderlimit = gs.limit = min(gs.predictionsdf[ix, :close], gs.limit * (1f0 + limitreduction))
            thistrend = flat
        elseif (gs.labels[ix] in [longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            if !isnothing(gs.limit) && (gs.assettype = down)
                # cancel shortclose order and add amount to longbuy order
                ta.cancelrunningorder = true
                ta.amountfactor = 2 # amount to sell long assets + amount to buy short assets
            end
            ta.orderlimit = gs.limit = gs.predictionsdf[gs.buyix, :close] * (1f0 - gs.buygain)
            ta.orderlabel = gs.ordertype = longbuy
            # thisbuyix = ix
            # thistrend = up
        elseif (gs.labels[ix] in [shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.openthreshold)
            if !isnothing(gs.limit) && (gs.assettype = up)
                # cancel longclose order and add amount to shortbuy order
                ta.cancelrunningorder = true
                ta.amountfactor = 2 # amount to sell long assets + amount to buy short assets
            end
            ta.orderlimit = gs.limit = gs.predictionsdf[gs.buyix, :close] * (1f0 + gs.buygain)
            ta.orderlabel = gs.ordertype = shortbuy
            # thisbuyix = ix
            # thistrend = down
        else
            if gs.assettype == up
                
                if gs.ordertype == longclose
                    ta.orderlimit = gs.limit = max(gs.predictionsdf[ix, :close], gs.limit * (1f0 - limitreduction))
                elseif gs.ordertype == longclose
                    ta.orderlimit = gs.limit = max(gs.predictionsdf[ix, :close], gs.limit * (1f0 - limitreduction))
                elseif gs.ordertype == longclose
                    ta.orderlimit = gs.limit = max(gs.predictionsdf[ix, :close], gs.limit * (1f0 - limitreduction))
                else
                    @assert gs.ordertype in [longclose, longbuy, shortclose, shortbuy]
                end
                ta.orderlabel = gs.ordertype
            end
        end
        if gs.trend != thistrend
            calcgain!(gs, ix)
            gs.buyix = thisbuyix
            gs.trend = thistrend
        end
    end
    gs.endix = lastix
end

"""
Returns a dataframe with gains of buy/sell pair actions until lastix. If `forcegain` == true then add a row that forces to calculate the last open gain segment.
The returned dataframe has the following columns:
- trend (up, flat, down)  
- samplecount of gain segment  
- minutes of gain segment  
- gain of gain segment
- startdt and enddt :: DateTime of gain segment
- startix, endix within predictionsdf rows of gain segment

Input is 
-  predictionsdf::AbstractDataFrame with the following colums
    - label as TradeLabel
    - score as Float32 prediction score
    - target as TradeLabel
    - mingain as expected % below pivot price to set a trade limit
    - maxgain as expected % above pivot price to set a trade limit
    - downminutes as expected down trend time in minutes
    - upminutes as expected up trend time in minutes
    - lastextrememinutes as minutes since last extreme
    - rangeid as indicator which samples belong to a sequence
    - opentime, high, low, close, pivot (samples of the same range are expected to be in opentime sorting order)
    - set as CategoricalVector with levels train, test, eval
    - coin as CategoricalVector
"""
function getgains(gs::GainSegment, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, forcegain::Bool; lastix=lastindex(scores), openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
    @assert firstindex(scores) == firstindex(labels) == firstindex(predictionsdf, 1) "firstindex(scores)=$(firstindex(scores)) == firstindex(labels)=$(firstindex(labels)) == firstindex(predictionsdf, 1)=$(firstindex(predictionsdf, 1))"
    gs.openthreshold = openthreshold
    gs.closethreshold = closethreshold
    gs.predictionsdf = predictionsdf
    gs.scores = scores
    gs.labels = labels
    gs.algorithm(gs, lastix)
    if forcegain
        calcgain!(gs, lastix)
    end
    return gs.gaindf
end

#region deprecated-legacy-copy
function addgainadmin!(gdf, coin, sampleset, predicted, rangeid, openthreshold, closethreshold)
    gdf[!, :coin] = fill(coin, size(gdf, 1))
    gdf[!, :set] = fill(sampleset, size(gdf, 1))
    gdf[!, :predicted] = fill(predicted, size(gdf, 1))
    gdf[!, :rangeid] = fill(rangeid, size(gdf, 1))
    gdf[!, :openthreshold] = fill(openthreshold, size(gdf, 1))
    gdf[!, :closethreshold] = fill(closethreshold, size(gdf, 1))
end

"""
Returns a dataframe with gains of buy/sell pair actions. At the end of a range a sell is enforced.
The returned dataframe has the following columns:
  - trend (up, flat, down)
  - samplecount of gain segment
  - minutes of gain segment
  - gain of gain segment
  - startdt and enddt :: DateTime of gain segment
  - startix, endix within predictionsdf rows of gain segment

Input is 
  -  predictionsdf::AbstractDataFrame with the following colums
    - label as TradeLabel
    - score as Float32 prediction score
    - target as TradeLabel
    - mingain as expected % below pivot price to set a trade limit
    - maxgain as expected % above pivot price to set a trade limit
    - downminutes as expected down trend time in minutes
    - upminutes as expected up trend time in minutes
    - lastextrememinutes as minutes since last extreme
    - rangeid as indicator which samples belong to a sequence
    - opentime, high, low, close, pivot (samples of the same range are expected to be in opentime sorting order)
    - set as CategoricalVector with levels train, test, eval
    - coin as CategoricalVector
  - maxwindow of sample window considered for the prediction
  - scorethresholds as named tuple with acceptance score; trade label names that are not included are ignored
"""
function getgainsdf(predictionsdf::AbstractDataFrame; maxwindow, scorethresholds=(longbuy=0.6, allclose=0.6, shortbuy=0.6))
    if isnothing(predictionsdf) || (size(predictionsdf, 1) == 0)
        return nothing
    end
    @assert maxwindow > 0
    ranges = unique(predictionsdf[!, :rangeid])
    for rngix in eachindex(ranges)
        rng = ranges[rngix]
        resultsview = @view predictionsdf[predictionsdf[!, :rangeid] .== rng, :]
        (verbosity >= 2) && print("$(EnvConfig.now()) calculating gains for range ($rngix/$(length(ranges))) $rng                             \r")
        (verbosity >= 3) && println()
        if size(resultsview, 1) == 0 
            (verbosity >= 2) && println("\n$(EnvConfig.now()) WARNING: unexpected empty resultsview for rangeid $rng")
            continue
        end
        # @assert issorted(resultsview[!, :opentime]) "unexpected unsorted opentime in resultsview for rangeid $rng"
        # @assert all(resultsview[begin, :set] .== resultsview[!, :set]) "Unexpected different sets $(unique(resultsview[!, :set])) in same range $rng"

        for scorethreshold in scorethresholds
            gdf = _getgainsdf(resultsview, resultsview[!, :score], resultsview[!, :label], scorethreshold)
            if size(gdf, 1) > 0
                addgainadmin!(gdf, resultsview[begin, :coin], resultsview[begin, :set], true, rng, openthreshold, closethreshold)
                gaindf = isnothing(gaindf) ? gdf : append!(gaindf, gdf)
            end
        end
        gdf = _getgainsdf(resultsview, fill(1f0, size(resultsview, 1)), resultsview[!, :target], 0.9f0, 0.9f0)
        addgainadmin!(gdf, resultsview[begin, :coin], resultsview[begin, :set], false, rng, 0.9f0, 0.9f0)
        gaindf = isnothing(gaindf) ? gdf : vcat(gaindf, gdf)

    end
    # println("describe(gaindf)=$(describe(gaindf)), size(gaindf)=$(size(gaindf))")
    gaindf = gaindf[.!ismissing.(gaindf[!, :set]), :] # exclude gaps between set partitions
    if size(gaindf, 1) > 0
        sort!(gaindf, [:coin, :predicted, :openthreshold, :closethreshold, :startdt])
        EnvConfig.savedf(gaindf, gainsfilename())
    end
    (verbosity >= 2) && println("$(EnvConfig.now()) calculated gains for $(length(ranges)) ranges")
    return gaindf
end
#endregion deprecated-legacy-copy

end # module
