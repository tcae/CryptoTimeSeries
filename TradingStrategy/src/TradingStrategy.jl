"""
Minimal TradingStrategy module retained for Trade integration and TrendDetector workflows.

This module intentionally keeps only the API surface required by:
- Trade/src/Trade.jl
- scripts/TrendDetector.jl
- scripts/tradereal.jl
- scripts/tradesim.jl
"""
module TradingStrategy

using DataFrames, Dates
using EnvConfig, Targets

"""Return the normalized config-scoped subfolder used for persisted trade artifacts."""
function tradesfolder(stem::AbstractString="gains")::String
    normalized = replace(normpath(splitext(String(stem))[1]), '\\' => '/')
    return startswith(normalized, "trades/") || (normalized == "trades") ? normalized : joinpath("trades", normalized)
end

"""Return the aggregate storage key used for one persisted trade artifact."""
tradesaggregate(stem::AbstractString="gains") = joinpath("trades", splitext(basename(String(stem)))[1] * "_all")

"""Return the per-coin storage key used for one persisted trade artifact."""
tradefilename(coin::AbstractString; stem::AbstractString="gains") = joinpath(tradesfolder(stem), uppercase(strip(String(coin))))

"""Persist a trades dataframe into config-scoped storage, plus optional aggregate copy."""
function savetrades(tradedf::AbstractDataFrame; stem::AbstractString="gains", include_aggregate::Bool=true)
    if size(tradedf, 1) == 0
        return String[]
    end
    @assert :coin in propertynames(tradedf) "tradedf must contain a :coin column; names=$(names(tradedf))"

    paths = String[]
    coins = unique(String.(tradedf[!, :coin]))
    for coin in coins
        coindf = DataFrame(tradedf[String.(tradedf[!, :coin]) .== coin, :])
        if size(coindf, 1) > 0
            push!(paths, EnvConfig.savedf(coindf, tradefilename(coin; stem=stem)))
        end
    end
    if include_aggregate
        push!(paths, EnvConfig.savedf(DataFrame(tradedf), tradesaggregate(stem)))
    end
    return paths
end

"""Load persisted trades, preferring aggregate cache over per-coin fragments."""
function loadtrades(; stem::AbstractString="gains")
    aggregate = EnvConfig.readdf(tradesaggregate(stem))
    if !isnothing(aggregate) && (size(aggregate, 1) > 0)
        return DataFrame(aggregate)
    end

    folderpath = normpath(joinpath(EnvConfig.logfolder(), tradesfolder(stem)))
    isdir(folderpath) || return DataFrame()

    parts = DataFrame[]
    for entry in readdir(folderpath; join=false, sort=true)
        name = splitext(entry)[1]
        piece = EnvConfig.readdf(name; folderpath=folderpath)
        if !isnothing(piece) && (size(piece, 1) > 0)
            push!(parts, DataFrame(piece))
        end
    end
    return isempty(parts) ? DataFrame() : reduce(vcat, parts; cols=:union)
end

"""Load persisted trades for one specific coin."""
function loadtrades(coin::AbstractString; stem::AbstractString="gains")
    tradedf = EnvConfig.readdf(tradefilename(coin; stem=stem))
    return isnothing(tradedf) ? DataFrame() : DataFrame(tradedf)
end

@inline islongopenlabel(label::TradeLabel) = (label == longbuy) || (label == longstrongbuy)
@inline isshortopenlabel(label::TradeLabel) = (label == shortbuy) || (label == shortstrongbuy)
@inline islongholdoropenlabel(label::TradeLabel) = (label == longhold) || islongopenlabel(label)
@inline isshortholdoropenlabel(label::TradeLabel) = (label == shorthold) || isshortopenlabel(label)
@inline islongcloselabel(label::TradeLabel) = (label == allclose) || (label == longstrongclose) || (label == longclose)
@inline isshortcloselabel(label::TradeLabel) = (label == shortclose) || (label == shortstrongclose) || (label == allclose)

"""
Lane state for open/close intent and realized-entry bookkeeping.

- `label`: directional open/close intent or `ignore`
- `closeprice`: active close limit (or 0f0 when absent)
- `openprice`: intended/realized entry anchor
- `openix`: realized entry bar index (0 means not realized)
"""
mutable struct TradeAction
    label::Union{TradeLabel, Nothing}
    closeprice::Float32
    openprice::Float32
    openix::Integer
    function TradeAction(label::Union{TradeLabel, Nothing}=ignore, closeprice=0f0, openprice=0f0, openix=0)
        ta = new(label, closeprice, openprice, openix)
        isopen(ta)
        return ta
    end
end

"""Return whether a trade action currently carries an active limit order."""
function isopen(ta::TradeAction)
    if ta.closeprice > 0f0
        @assert (ta.label != ignore) && (ta.closeprice > 0f0) && (ta.openprice > 0f0) "(ta.label != ignore) && (ta.closeprice > 0f0) && (ta.openprice > 0f0); got label=$(ta.label), closeprice=$(ta.closeprice), openprice=$(ta.openprice)"
    end
    return ta.closeprice > 0f0
end

"""Clear all order state from a trade action."""
function removeorder!(ta::TradeAction)
    ta.label = ignore
    ta.closeprice = 0f0
    ta.openprice = 0f0
    ta.openix = 0
    return ta
end

@inline _islongopenaction(label) = label in [longbuy, longstrongbuy]
@inline _isshortopenaction(label) = label in [shortbuy, shortstrongbuy]

"""Clear one lane state completely."""
function _clearactionlane!(ta::TradeAction)
    ta.label = ignore
    ta.closeprice = 0f0
    ta.openprice = 0f0
    ta.openix = 0
    return ta
end

"""Clear only open-intent labels while preserving close guidance."""
function _clearopenintent!(ta::TradeAction)
    if ta.label in [longbuy, longstrongbuy, shortbuy, shortstrongbuy]
        ta.label = ignore
    end
    return ta
end

@inline _lanehascloseguidance(ta::TradeAction) = (ta.openprice > 0f0) && (ta.closeprice > 0f0)
@inline _price_in_bar(price::Float32, low::Real, high::Real) = (Float32(low) <= price) && (price <= Float32(high))

"""Return an empty gain dataframe with the canonical Trade-consumed schema."""
function emptygaindf()::DataFrame
    return DataFrame(
        trend=TrendPhase[],
        samplecount=Int[],
        minutes=Int[],
        gain=Float32[],
        gainfee=Float32[],
        startdt=DateTime[],
        enddt=DateTime[],
        startix=Int[],
        endix=Int[],
    )
end

"""
State container for gain-based strategy execution over one symbol.

The lane model keeps long and short guidance independently in `longta` and `shortta`.
"""
mutable struct GainSegment
    algorithm
    openthreshold::Float32
    closethreshold::Float32
    maxwindow::Integer
    endix::Integer
    gaindf::DataFrame
    makerfee::Float32
    takerfee::Float32
    scores::Union{AbstractVector, Nothing}
    labels::Union{AbstractVector, Nothing}
    predictionsdf::Union{AbstractDataFrame, Nothing}
    lastix::Integer
    buygain::Float32
    sellgain::Float32
    limitreduction::Float32
    longta::TradeAction
    shortta::TradeAction
    trace::Union{Nothing, DataFrame}
    tracecontext::Union{Nothing, String}
    lane_reconciliation::Union{Nothing, NamedTuple}
    function GainSegment(;maxwindow::Integer=4*60, openthreshold=0.6, closethreshold=0.5, algorithm=gain_reversal!, makerfee::AbstractFloat=0f0, takerfee::AbstractFloat=0f0, buygain::AbstractFloat=0.001f0, sellgain::AbstractFloat=0.01f0, limitreduction::AbstractFloat=0f0)
        return new(algorithm, openthreshold, closethreshold, maxwindow, 0, emptygaindf(), makerfee, takerfee, nothing, nothing, nothing, 0, buygain, sellgain, limitreduction, TradeAction(), TradeAction(), nothing, nothing, nothing)
    end
end

"""Set execution reconciliation hints for lane synchronization."""
function setreconciliation!(gs::GainSegment;
    long_open_qty::Real=0f0,
    long_avg_entry::Real=0f0,
    long_open_ix::Integer=0,
    short_open_qty::Real=0f0,
    short_avg_entry::Real=0f0,
    short_open_ix::Integer=0,
)
    gs.lane_reconciliation = (
        long_open_qty=Float32(max(0f0, Float32(long_open_qty))),
        long_avg_entry=Float32(max(0f0, Float32(long_avg_entry))),
        long_open_ix=Int(long_open_ix),
        short_open_qty=Float32(max(0f0, Float32(short_open_qty))),
        short_avg_entry=Float32(max(0f0, Float32(short_avg_entry))),
        short_open_ix=Int(short_open_ix),
    )
    return gs
end

"""Clear previously configured reconciliation hints."""
function clearreconciliation!(gs::GainSegment)
    gs.lane_reconciliation = nothing
    return gs
end

"""Apply reconciliation hints to lane state without altering directional intent."""
function _apply_reconciliation_to_lanes!(gs::GainSegment)
    isnothing(gs.lane_reconciliation) && return gs
    rc = gs.lane_reconciliation

    if rc.long_open_qty > 0f0
        entry = rc.long_avg_entry > 0f0 ? rc.long_avg_entry : max(gs.longta.openprice, 0f0)
        if entry > 0f0
            gs.longta.openprice = entry
            if rc.long_open_ix > 0
                gs.longta.openix = rc.long_open_ix
            end
            if gs.longta.closeprice <= 0f0
                gs.longta.closeprice = entry * (1f0 + gs.sellgain)
            end
        end
    end

    if rc.short_open_qty > 0f0
        entry = rc.short_avg_entry > 0f0 ? rc.short_avg_entry : max(gs.shortta.openprice, 0f0)
        if entry > 0f0
            gs.shortta.openprice = entry
            if rc.short_open_ix > 0
                gs.shortta.openix = rc.short_open_ix
            end
            if gs.shortta.closeprice <= 0f0
                gs.shortta.closeprice = entry * (1f0 - gs.sellgain)
            end
        end
    end

    return gs
end

"""Apply reconciliation hints and return the updated gain segment."""
synclanes!(gs::GainSegment) = _apply_reconciliation_to_lanes!(gs)

"""Reset one gain segment instance to initial runtime state."""
function reset!(gs::GainSegment)
    gs.predictionsdf = nothing
    gs.scores = nothing
    gs.labels = nothing
    gs.endix = gs.lastix = 0
    gs.gaindf = emptygaindf()
    removeorder!(gs.longta)
    removeorder!(gs.shortta)
    clearreconciliation!(gs)
    if !isnothing(gs.trace)
        empty!(gs.trace)
    end
    return gs
end

"""Return whether any lane currently carries close guidance."""
isopensegment(gs::GainSegment) = _lanehascloseguidance(gs.longta) || _lanehascloseguidance(gs.shortta)

"""Infer trend direction from lane price relation."""
assettrend(ta::TradeAction) = ta.closeprice > ta.openprice ? up : (ta.closeprice < ta.openprice ? down : flat)

@inline function _actiontrend(label::TradeLabel)
    if label in [longbuy, longstrongbuy, longclose, longstrongclose]
        return up
    elseif label in [shortbuy, shortstrongbuy, shortclose, shortstrongclose]
        return down
    end
    return flat
end

"""Close all open lanes at one sell index and optional price."""
function calcgain!(gs::GainSegment, sellix::Integer, sellprice::Float32=gs.predictionsdf[sellix, :close])
    if _lanehascloseguidance(gs.longta) && (gs.longta.openix > 0)
        calcgain!(gs, gs.longta, sellix, sellprice)
        _clearactionlane!(gs.longta)
    end
    if _lanehascloseguidance(gs.shortta) && (gs.shortta.openix > 0)
        calcgain!(gs, gs.shortta, sellix, sellprice)
        _clearactionlane!(gs.shortta)
    end
    return gs
end

"""Record one realized gain segment from one lane action."""
function calcgain!(gs::GainSegment, ta::TradeAction, sellix::Integer, sellprice::Float32)
    if (ta.openix > 0) && (ta.openprice > 0f0)
        starttime = gs.predictionsdf[ta.openix, :opentime]
        ixtime = gs.predictionsdf[sellix, :opentime]
        trend = _actiontrend(ta.label)
        if trend == flat
            trend = assettrend(ta)
        end

        if trend == up
            gain = (sellprice - ta.openprice) / ta.openprice
        elseif trend == down
            gain = -(sellprice - ta.openprice) / ta.openprice
        else
            return gs
        end

        minutes = Int(div(Dates.value(ixtime - starttime), 60000)) + 1
        push!(gs.gaindf, (trend, (sellix - ta.openix + 1), minutes, gain, (gain - 2f0 * gs.makerfee), starttime, ixtime, ta.openix, sellix))
        ta.openix = 0
    end
    return gs
end

"""Open/hold/close strategy that closes when hold support drops below threshold."""
function gain_open_close!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat

    @inbounds for ix in (gs.endix + 1):lastix
        label = labels[ix]
        score = scores[ix]
        thistrend = lasttrend
        thisbuyix = 0

        if (lasttrend == up) && islongholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = up
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif (lasttrend == down) && isshortholdoropenlabel(label) && (score > gs.closethreshold)
            thistrend = down
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end

        if lasttrend != thistrend
            calcgain!(gs, ix)
            if thistrend == up
                gs.longta.label = ignore
                gs.longta.closeprice = closecol[ix] * (1f0 + gs.sellgain)
                gs.longta.openprice = closecol[ix]
                gs.longta.openix = thisbuyix
                _clearactionlane!(gs.shortta)
            elseif thistrend == down
                gs.shortta.label = ignore
                gs.shortta.closeprice = closecol[ix] * (1f0 - gs.sellgain)
                gs.shortta.openprice = closecol[ix]
                gs.shortta.openix = thisbuyix
                _clearactionlane!(gs.longta)
            else
                _clearactionlane!(gs.longta)
                _clearactionlane!(gs.shortta)
            end
            lasttrend = thistrend
        end
    end

    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""Reversal strategy that closes on explicit opposite/close labels above threshold."""
function gain_reversal!(gs::GainSegment, lastix)
    labels = gs.labels
    scores = gs.scores
    closecol = gs.predictionsdf[!, :close]
    lasttrend = flat

    @inbounds for ix in (gs.endix + 1):lastix
        label = labels[ix]
        score = scores[ix]
        thistrend = lasttrend
        thisbuyix = 0

        if (lasttrend == up) && islongcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif (lasttrend == down) && isshortcloselabel(label) && (score > gs.openthreshold)
            thistrend = flat
        elseif islongopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = up
        elseif isshortopenlabel(label) && (score >= gs.openthreshold)
            thisbuyix = ix
            thistrend = down
        end

        if lasttrend != thistrend
            calcgain!(gs, ix)
            if thistrend == up
                gs.longta.label = ignore
                gs.longta.closeprice = closecol[ix] * (1f0 + gs.sellgain)
                gs.longta.openprice = closecol[ix]
                gs.longta.openix = thisbuyix
                _clearactionlane!(gs.shortta)
            elseif thistrend == down
                gs.shortta.label = ignore
                gs.shortta.closeprice = closecol[ix] * (1f0 - gs.sellgain)
                gs.shortta.openprice = closecol[ix]
                gs.shortta.openix = thisbuyix
                _clearactionlane!(gs.longta)
            else
                _clearactionlane!(gs.longta)
                _clearactionlane!(gs.shortta)
            end
            lasttrend = thistrend
        end
    end

    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""Handle lane fills for one candle and update realized gain state."""
function processclosedorder!(gs::GainSegment, ix::Integer, ta::TradeAction)
    if (ta.openprice > 0f0) && (ta.closeprice > 0f0)
        if (ta.label in [longbuy, longstrongbuy, shortbuy, shortstrongbuy])
            if (ta.openix <= 0) && _price_in_bar(Float32(ta.openprice), gs.predictionsdf[ix, :low], gs.predictionsdf[ix, :high])
                ta.openix = ix
            end
            if (ta.openix > 0) && _price_in_bar(Float32(ta.closeprice), gs.predictionsdf[ix, :low], gs.predictionsdf[ix, :high])
                calcgain!(gs, ta, ix, ta.closeprice)
            end
        elseif ta.label in [longclose, longstrongclose]
            if ta.closeprice <= gs.predictionsdf[ix, :high]
                calcgain!(gs, ta, ix, ta.closeprice)
                _clearactionlane!(ta)
            end
        elseif ta.label in [shortclose, shortstrongclose]
            if gs.predictionsdf[ix, :low] <= ta.closeprice
                calcgain!(gs, ta, ix, ta.closeprice)
                _clearactionlane!(ta)
            end
        end
    end
end

"""Adjust lane open/close intents by one prediction candle for the limit-reversal strategy."""
function reachgainuntilreversal!(longta::TradeAction, shortta::TradeAction, label::TradeLabel, score, high, low, close, openthreshold, buygain, sellgain, limitreduction)
    _ = high
    _ = low
    branch = "none"

    longsignal = (label in [longbuy, longstrongbuy]) && (score >= openthreshold)
    shortsignal = (label in [shortbuy, shortstrongbuy]) && (score >= openthreshold)

    !longsignal && _clearopenintent!(longta)
    !shortsignal && _clearopenintent!(shortta)

    if longsignal
        branch = "long_signal"
        if _lanehascloseguidance(shortta)
            shortta.closeprice = close * (1f0 - buygain)
        end
        longta.openprice = close * (1f0 - buygain)
        longta.label = label == longstrongbuy ? longstrongbuy : longbuy
        longta.closeprice = close * (1f0 + sellgain)
    elseif shortsignal
        branch = "short_signal"
        if _lanehascloseguidance(longta)
            longta.closeprice = close * (1f0 + buygain)
        end
        shortta.openprice = close * (1f0 + buygain)
        shortta.label = label == shortstrongbuy ? shortstrongbuy : shortbuy
        shortta.closeprice = close * (1f0 - sellgain)
    else
        branch = "reduce_limit"
        if _lanehascloseguidance(longta) && (longta.closeprice > longta.openprice)
            longta.closeprice = max(close, longta.closeprice * (1f0 - sellgain * limitreduction))
        end
        if _lanehascloseguidance(shortta) && (shortta.closeprice < shortta.openprice)
            shortta.closeprice = min(close, shortta.closeprice * (1f0 + sellgain * limitreduction))
        end
    end

    return branch
end

"""Limit-reversal strategy with lane-based independent open/close guidance."""
function gain_limit_reversal!(gs::GainSegment, lastix)
    for ix in (gs.endix + 1):lastix
        processclosedorder!(gs, ix, gs.longta)
        processclosedorder!(gs, ix, gs.shortta)
        reachgainuntilreversal!(
            gs.longta,
            gs.shortta,
            gs.labels[ix],
            gs.scores[ix],
            gs.predictionsdf[ix, :high],
            gs.predictionsdf[ix, :low],
            gs.predictionsdf[ix, :close],
            gs.openthreshold,
            gs.buygain,
            gs.sellgain,
            gs.limitreduction,
        )
    end
    gs.endix = lastix
    _apply_reconciliation_to_lanes!(gs)
    return gs
end

"""
Run one gain algorithm pass and optionally force-close open lanes at `lastix`.

The input `predictionsdf` must provide columns `:opentime`, `:high`, `:low`, `:close`.
"""
function getgains(gs::GainSegment, predictionsdf::AbstractDataFrame, scores::AbstractVector, labels::AbstractVector, forcegain::Bool; lastix=lastindex(scores), openthreshold=gs.openthreshold, closethreshold=gs.closethreshold)
    @assert length(scores) == length(labels) == size(predictionsdf, 1) "length(scores)=$(length(scores)) == length(labels)=$(length(labels)) == size(predictionsdf, 1)=$(size(predictionsdf, 1))"

    gs.openthreshold = Float32(openthreshold)
    gs.closethreshold = Float32(closethreshold)
    gs.predictionsdf = predictionsdf
    gs.scores = scores
    gs.labels = labels
    gs.algorithm(gs, lastix)
    _apply_reconciliation_to_lanes!(gs)

    if forcegain
        calcgain!(gs, lastix)
        _apply_reconciliation_to_lanes!(gs)
    end
    return gs.gaindf
end

end # module
