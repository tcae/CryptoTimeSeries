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

mutable struct GainSegment
    algorithm  # algorithm function to be applied
    openthreshold::AbstractFloat # score threshold
    closethreshold::AbstractFloat # score threshold
    # scorethreshold::Dict # score threshold dict
    maxwindow::Integer # maxwindow of sample window considered for the prediction
    endix::Integer # is the index of the last analyzed row of scores/labels/predictionsdf
    gaindf::DataFrame
    makerfee::AbstractFloat
    takerfee::AbstractFloat
    scores::AbstractVector
    labels::AbstractVector
    predictionsdf::AbstractDataFrame
    buyix::Integer # buy index of last open gain segment that is not yet saved in gainsdf
    lastix::Integer # last inspected row index of last open gain segment that is not yet saved in gainsdf
    trend::TrendPhase # trend of last 
    buylimit::AbstractFloat
    selllimit::AbstractFloat
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
function GainSegment(predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, maxwindow::Integer, openthreshold, closethreshold; algorithm=algorithm01!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0)
    @assert length(scores) == length(labels) == size(predictionsdf, 1) > 0 "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1)) > 0"
    @assert firstindex(scores) == firstindex(labels) == firstindex(predictionsdf, 1) "firstindex(scores)=$(firstindex(scores)) == firstindex(labels)=$(firstindex(labels)) == firstindex(predictionsdf, 1)=$(firstindex(predictionsdf, 1))"
    return GainSegment(algorithm, openthreshold, closethreshold, maxwindow, firstindex(scores)-1, DataFrame(), makerfee, takerfee, scores, labels, predictionsdf, 0, 0, flat, 0f0, 0f0)
end

"""
Consumes the next sample row and updates the gain segment. Returns the gain of the gain segment, which is `nothing` as long as the position is not closed.
"""
function algorithm01!(gs::GainSegment)
    for ix in (gs.endix+1):lastindex(gs.scores)
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
            if gs.buyix > 0 # there is an open gain segment -> close it
                starttime = gs.predictionsdf[gs.buyix, :opentime]
                ixtime = gs.predictionsdf[ix, :opentime]
                @assert gs.trend in [up, down] "if gs.trend=$(gs.trend) == flat then buyix=$(gs.buyix) == 0"
                if gs.trend == up
                    # long sell
                    startprice = gs.predictionsdf[gs.buyix, :close] * (1 + gs.makerfee)
                    ixprice = gs.predictionsdf[ix, :close] * (1 - gs.makerfee)
                    gain = (ixprice - startprice) / startprice
                elseif gs.trend == down
                    # short sell
                    startprice = gs.predictionsdf[gs.buyix, :close] * (1 - gs.makerfee)
                    ixprice = gs.predictionsdf[ix, :close] * (1 + gs.makerfee)
                    gain = -(ixprice - startprice) / startprice  # down trend -> negative price diff -> gain shall be positive for a short trade
                end
                push!(gs.gaindf, (trend=gs.trend, samplecount=(ix-gs.buyix+1), minutes=Minute(ixtime-starttime).value + 1, gain=gain, startdt=starttime, enddt=ixtime, startix=gs.buyix, endix=ix))
                gs.buyix = 0  # indicates closure of gain segment
            end
            gs.buyix = thisbuyix
            gs.trend = thistrend
        end
    end
end

"""
Consumes the next sample row and updates the gain segment. Returns the gain of the gain segment, which is `nothing` as long as the position is not closed.
"""
function algorithm02!(gs::GainSegment)
    gainrow = nothing
    for ix in (gs.endix+1):lastindex(gs.scores)
        #TODO first check whether the limit order is executed, if so, set trend and buyix and selllimit
        #TODO decide how long the limit order should stay and what limit corrections are required
        if gs.buyix > gs.endix   # there is an open position => looking for sell
            @assert gs.endix < gs.buyix <= lastindex(gs.scores) "gs.endix=$(gs.endix) < gs.buyix=$(gs.buyix) <= lastindex(gs.scores)=$(lastindex(gs.scores))"
            starttime = gs.predictionsdf[gs.buyix, :opentime]
            ixtime = gs.predictionsdf[ix, :opentime]
            if (gs.trend == up) && (gs.labels[ix] in [longstrongclose, longclose, allclose, shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.scorethreshold[gs.labels[ix]])
                # long sell signal
                startprice = gs.predictionsdf[gs.buyix, :close] * (1 + gs.makerfee)
                ixprice = gs.predictionsdf[ix, :close] * (1 - gs.makerfee)
                gain = (ixprice - startprice) / startprice
            elseif (gs.trend == down) && (gs.labels[ix] in [shortstrongclose, shortclose, allclose, longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.scorethreshold[gs.labels[ix]])
                # short sell signal
                startprice = gs.predictionsdf[gs.buyix, :close] * (1 - gs.makerfee)
                ixprice = gs.predictionsdf[ix, :close] * (1 + gs.makerfee)
                gain = -(ixprice - startprice) / startprice  # down trend -> negative price diff -> gain shall be positive for a short trade
            else
                gain = nothing  # gain segment not closed
            end
            if !isnothing(gain)
                gainrow =(trend=gs.trend, samplecount=(ix-gs.buyix+1), minutes=Minute(ixtime-starttime).value + 1, gain=gain, startdt=starttime, enddt=ixtime, startix=gs.buyix, endix=ix)
                push!(gs.gaindf, gainrow)
                gs.buyix = 0  # indicates closure of gain segment
                gs.trend = flat
            end
        end
        if gs.buyix == 0  # no open gain segment, check whether to open a gain segment
            if (gs.labels[ix] in [longbuy, longstrongbuy]) && (gs.scores[ix] >= gs.scorethreshold[gs.labels[ix]])
                # long buy signal
                gs.trend = up
                gs.buyix = ix
            elseif (gs.labels[ix] in [shortbuy, shortstrongbuy]) && (gs.scores[ix] >= gs.scorethreshold[gs.labels[ix]])
                # short buy signal
                gs.trend = down
                gs.buyix = ix
            end # all other labels are ignored without an open position
        end
    end
    # if (size(gs.gaindf, 1) == 0)
    #     starttime = gs.predictionsdf[begin, :opentime]
    #     endtime = gs.predictionsdf[end, :opentime]
    #     gainrow =(trend=flat, samplecount=(length(gs.scores)), minutes=Minute(endtime-starttime).value + 1, gain=0f0, startdt=starttime, enddt=endtime, startix=firstindex(gs.scores), endix=lastindex(gs.scores))
    #     push!(gs.gaindf, gainrow)
    # end 
end

function getgains(gs::GainSegment)
    gs.algorithm(gs)
    return gs.gaindf
end

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

end # module
