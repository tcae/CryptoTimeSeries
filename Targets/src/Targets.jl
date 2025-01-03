"""
**! predicitions will be merged with targets.jl**
The asset can receive **predictions** within a given **time period** from algorithms or individuals by:

- assigning *increase*, *neutral*, *decrease*
- target price
- target +-% from current price

Prediction algorithms are identified by name. Individuals are identified by name.

"""
module Targets

using EnvConfig, Ohlcv, TestOhlcv, Features
using DataFrames, Dates, Logging, CategoricalArrays
export TradeLabel, shortstrongbuy, shortbuy, shorthold, shortclose, shortstrongclose, longshortclose, longstrongclose, longclose, longhold, longbuy, longstrongbuy

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1


"""
returns all possible labels (don't change sequence because index is used as class id). "longshortclose" is default.
"""
const tradelabels = ["ignore", "longbuy", "longhold", "longshortclose", "shorthold", "shortbuy", "longclose", "shortclose", "longstrongbuy", "shortstrongbuy", "longstrongclose", "shortstrongclose"]
@enum TradeLabel shortstrongbuy=-5 shortbuy=-4 shorthold=-3 shortclose=-2 shortstrongclose =-1 longshortclose=0 longstrongclose=1 longclose=2 longhold=3 longbuy=4 longstrongbuy=5

"Defines the targets interface that shall be provided by all target implementations. Ohlcv is provided at init and maintained as internal reference."
abstract type AbstractTargets <: EnvConfig.AbstractConfiguration end

"Adds a coin with OhlcvData to the target generation. Each coin can only have 1 associated data set."
function setbase!(targets::AbstractTargets, ohlcv::Ohlcv.OhlcvData) end

"Removes the targets of a basecoin."
function removebase!(targets::AbstractTargets) end

"Add newer targets to match the recent timeline of ohlcv with the newest ohlcv datapoints, i.e. datapoints newer than last(features)"
function supplement!(targets::AbstractTargets) end

"Provides the class label of the target class."
function labels(targets::AbstractTargets, firstix::Integer, lastix::Integer) end
function labels(targets::AbstractTargets, firstix::DateTime, lastix::DateTime) end

"Provide the relative gain of the current price compared to the target price in the future. (currentprice - targetprice) / currentprice"
function relativegain(targets::AbstractTargets, firstix::Integer, lastix::Integer) end
function relativegain(targets::AbstractTargets, firstix::DateTime, lastix::DateTime) end

"Provides a description that characterizes the features"
describe(targets::AbstractTargets) = "$(typeof(targets))"

"""
Calculates the start y coordinate of a straight regression line given by the last y of the line `regry`, the gradient `grad` and the length `window`.
"""
startregry(eregry, grad, window) = eregry - grad * (window - 1)

"Returns the relative gain of the given regression relative to start y if `forward` (default) otherwise relative to given end regry"
function relativegain(endregry, grad, window; relativefee::AbstractFloat=0f0, forward=true)
    sregry = startregry(endregry, grad, window)
    return Targets.relativegain(sregry, endregry; relativefee=relativefee, forward=forward)
end
relativedaygain(f4::Features.Features004, regrminutes::Integer, featuresix::Integer) = relativegain(regry(f4, regrminutes)[featuresix], grad(f4, regrminutes)[featuresix], 24*60)

"""
- returns the relative forward/backward looking gain
- deducts relativefee from both values
"""
function relativegain(startvalue::AbstractFloat, endvalue::AbstractFloat; relativefee::AbstractFloat=0f0, forward::Bool=true)
    startvalue = startvalue * (1 + relativefee)
    endvalue = endvalue * (1 - relativefee)
    if forward
        gain = (endvalue - startvalue) / startvalue
    else
        gain = (endvalue - startvalue) / endvalue
    end
    return gain
end

function relativegain(values::AbstractVector{T}, baseix::Integer, gainix::Integer; relativefee::AbstractFloat=0f0) where {T<:AbstractFloat}
    if baseix > gainix
        return relativegain(values[gainix], values[baseix]; relativefee=relativefee, forward=false)
    else
        return relativegain(values[baseix], values[gainix]; relativefee=relativefee, forward=true)
    end
end


function labeldistribution(targets::AbstractVector{T}, labels=unique(targets)) where T<:AbstractString
    cattargets = categorical(targets; levels=labels, compress=true)
    return labeldistribution(cattargets)
end

function labeldistribution(targets::CategoricalArray)
    labels = levels(targets)
    cnt = zeros(Int, length(labels))
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    for tl in targets
        cnt[levelcode(tl)] += 1
    end
    targetcount = size(targets, 1)
    dist = [(labels[i], round(cnt[i] / targetcount*100, digits=1)) for i in eachindex(labels)]
    println("target label distribution in %: ", dist)
    return dist
end


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
- defines the relative transaction thresholds
    - longbuy long at more than *longbuy* gain potential from current price
    - longhold long above *longhold* gain potential from current price
    - close long position below *longhold* gain potential from current price
    - longbuy short at or lower than *shortbuy* loss potential from current price
    - longhold short below *shorthold* loss potential from current price
    - close short position above *shorthold* loss potential from current price
- Targets.defaultlabelthresholds provides default thresholds
"""
struct LabelThresholds
    longbuy::Float32
    longhold::Float32
    shorthold::Float32
    shortbuy::Float32
end
LabelThresholds(;longbuy, longhold, shorthold, shortbuy) = LabelThresholds(longbuy, longhold, shorthold, shortbuy)
thresholds(lt::LabelThresholds)::NamedTuple = (longbuy=lt.longbuy, longhold=lt.longhold, shorthold=lt.shorthold, shortbuy=lt.shortbuy)
thresholds(lt::NamedTuple)::LabelThresholds = LabelThresholds(lt.longbuy, lt.longhold, lt.shorthold, lt.shortbuy)

defaultlabelthresholds = LabelThresholds(0.03, 0.005, -0.005, -0.03)

"""
Because the trade signals are not independent classes but an ordered set of actions, this function returns the labels that correspond to specific thresholds:

- The folllowing invariant is assumed: `longbuy > longhold >= 0 >= shorthold > shortbuy`
- a gain shall be above `longbuy` threshold for a longbuy (long longbuy) signal
- bought assets shall be held (but not bought) if the remaining gain is above `longhold` threshold
- bought assets shall be closed if the remaining gain is below `longhold` threshold
- a loss shall be below `shortbuy` for a longclose (short longbuy) signal
- sold (short longbuy) assets shall be held if the remaining loss is below `shorthold`
- sold (short longbuy) assets shall be closed if the remaining loss is above `shorthold`
- all thresholds are relative gain values: if backwardrelative then the relative gain is calculated with the target price otherwise with the current price

"""
function getlabels(relativedist, labelthresholds::LabelThresholds)
    lt = labelthresholds
    rd = relativedist
    lastbuy = "nobuy"

    function settradestate(newstate)
        if newstate == "longhold"
            if lastbuy != "longbuy"
                lastbuy = "nobuy"
                newstate = "ignore"
            end
        elseif newstate == "longbuy"
            lastbuy = "longbuy"
        elseif newstate == "shortbuy"
            lastbuy = "shortbuy"
        elseif newstate == "shorthold"
            if lastbuy != "shortbuy"
                lastbuy = "nobuy"
                newstate = "ignore"
            end
        elseif newstate == "longshortclose"
            if !(lastbuy in ["shortbuy", "longbuy"])
                newstate = "ignore"
            elseif lastbuy == "longbuy"
                newstate = "longclose"
            elseif lastbuy == "shortbuy"
                newstate = "shortclose"
            end
        else
            @error "unexpected newstate=$newstate"
        end
        return newstate
    end

    labels = [(rd >= lt.longbuy ? settradestate("longbuy") :
                (rd >= lt.longhold ? settradestate("longhold") :
                    (rd <= lt.shortbuy ? settradestate("shortbuy") :
                        (rd <= lt.shorthold ? settradestate("shorthold") :
                            settradestate("longshortclose"))))) for rd in relativedist]
    return labels
end

function relativedistances(prices::Vector{T}, pricediffs, priceix, backwardrelative=true) where {T<:Real}
    if backwardrelative
        relativedist = [(priceix[ix] == 0 ? T(0.0) : pricediffs[ix] / prices[abs(priceix[ix])]) for ix in 1:size(prices, 1)]
    else
        relativedist = pricediffs ./ prices
    end
end

function continuousdistancelabels(prices::Vector{T}, labelthresholds::LabelThresholds) where T<:AbstractFloat
    pricediffs, priceix = Features.nextpeakindices(prices, labelthresholds.longbuy, labelthresholds.shortbuy)
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, labelthresholds)
    return labels, relativedist, pricediffs, priceix
end


mutable struct PriceExtreme
    peakix  # vector of signed indices (positive for maxima, negative for minima)
    gain  # current cumulated gain since anchorix
    labelthresholds::LabelThresholds
    function PriceExtreme(f2, regrwindow, labelthresholds::LabelThresholds=defaultlabelthresholds)
        if isnothing(f2)
            extremesix = Int64[]
        else
            prices = Ohlcv.dataframe(f2.ohlcv).pivot
            regr = f2.regr[regrwindow].xtrmix
            regr = [sign(regr[rix]) * Features.ohlcvix(f2, abs(regr[rix])) for rix in eachindex(regr)]  # translate to price index
            # println("PriceExtreme regr=$regr")
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
        end
        return new(extremesix, 0.0, labelthresholds)
    end
end

function Base.show(io::IO, pe::PriceExtreme)
    println(io, "PriceExtreme len(peakix)=$(length(pe.peakix)) gain=$(pe.gain) labelthresholds=$(pe.labelthresholds)")
end

function gain(prices::Vector{T}, ix1::K, ix2::K, labelthresholds::LabelThresholds) where {T<:AbstractFloat, K<:Integer}
    g = Targets.relativegain(prices, abs(ix1), abs(ix2))
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

"""
Returns a Dict{regression window => PriceExtreme} of best prices index extremes
including the special regressionwindow key "combi" that represents the best combination of extremes across the various regression windows.
- `regrwinarr` may contain a subset of available regressionwindow minutes, e.g. `[5, 15]`, to reduce the best target combination to those.
"""
function peaksbeforeregressiontargets(f2::Features.Features002; labelthresholds::LabelThresholds=defaultlabelthresholds, regrwinarr=nothing)::Dict
    debug = false
    @debug "Targets.peaksbeforeregressiontargets" begin
        debug = true
    end
    maxgain = 100000.0
    prices = Ohlcv.dataframe(f2.ohlcv).pivot
    @assert !isnothing(prices)
    # pe = Dict(rw => PriceExtreme(f2, rw, labelthresholds) for rw in keys(f2.regr))
    pe = Dict()
    for rw in keys(f2.regr)
        if isnothing(regrwinarr) || (rw in regrwinarr)
            pe[rw] = PriceExtreme(f2, rw, labelthresholds)
        end
    end

    peakarr = sort([ToporderElem(abs(pix), [sign(pix) * rw]) for rw in keys(pe) for pix in pe[rw].peakix], by = x -> x.pix, rev=false)
    # peakarr contains now a sorted list of tuple of (positive extreme indices, signed regression minutes depending of max/min)
    # multiple equal peak indices can be in peakarr from different regression windows

    reducedpeakarr = reducepeakarr(peakarr, nothing)
    # now reducedpeakarr has each extreme index only once in sorted order as tuple of (positive extreme index, weight=Inf, array of signed regression minutes)

    rwset = Set([rw for rw in keys(pe)])
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
    for (rw, p) in pe
        p.gain = gain(prices, p.peakix, labelthresholds)
    end
    pe["combi"] = PriceExtreme(nothing, "combi", labelthresholds)
    pe["combi"].peakix = peakix
    pe["combi"].gain = gain(prices, peakix, labelthresholds)
    return pe
end

function continuousdistancelabels(f2::Features.Features002; labelthresholds::LabelThresholds=defaultlabelthresholds, regrwinarr=nothing)
    pe = peaksbeforeregressiontargets(f2; labelthresholds=labelthresholds, regrwinarr=regrwinarr)
    @debug "gains of regression window: $([(rw, p.gain) for (rw, p) in pe])"
    labels, relativedist, pricediffs, priceix = ohlcvlabels(Ohlcv.dataframe(f2.ohlcv).pivot, pe["combi"])
    return labels, relativedist, pricediffs, priceix
end

function ohlcvlabels(prices::Vector{T}, pe::PriceExtreme) where {T<:AbstractFloat}
    pricediffs = zeros(Float32, length(prices))
    priceix = zeros(Int32, length(prices))
    pix = firstindex(pe.peakix)
    for ix in eachindex(prices)  # prepare priceix with the next relevant peak index within prices
        if (ix >= abs(pe.peakix[pix]) && (pix < lastindex(pe.peakix)))
            pix = pix + 1
            priceix[ix] = pe.peakix[pix]
        elseif (pix <= lastindex(pe.peakix)) && (ix <= abs(pe.peakix[pix]))
            priceix[ix] = pe.peakix[pix]
        else  # ix > abs(pe.peakix[pix])
            priceix[ix] = -sign(pe.peakix[pix]) * lastindex(priceix)
        end
        pricediffs[ix] = prices[abs(priceix[ix])]  - prices[ix]
    end
    relativedist = relativedistances(prices, pricediffs, priceix, false)
    labels = getlabels(relativedist, pe.labelthresholds)
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


function loaddata(ohlcv, labelthresholds)
    println("loaddata ohlcv")
    println("$(EnvConfig.now()) start generating features002")
    f2 = Features.Features002(ohlcv)
    println("$(EnvConfig.now()) start generating features003")
    lookbackperiods = 11  # == using the last 12 concatenated regression windows
    f3 = Features.Features003(f2, lookbackperiods)
    println("$(EnvConfig.now()) start generating peaksbeforeregressiontargets")
    pe = Targets.peaksbeforeregressiontargets(f2; labelthresholds=labelthresholds, regrwinarr=nothing)
    return f3, pe
end

emptyfdgdf() = DataFrame(Dict(:opentime=>DateTime[], :pivot=>Float32[], :label=>String[], :longbuy=>Bool[], :longhold=>Bool[], :shortbuy=>Bool[], :shorthold=>Bool[], :minix=>UInt32[], :maxix=>UInt32[], :minreldiff=>Float32[], :maxreldiff=>Float32[]))

"""
Provides 4 independent binary targets
- longbuy if label threshold `thres.longbuy` is exceeded within the next `window` minutes
- longhold if label threshold `thres.longhold` is not undercut within the next `window` minutes
- shorthold if label threshold `thres.shorthold` is not exceeded within the next `window` minutes
- shortbuy if label threshold `thres.shortbuy` is undercut within the next `window` minutes
"""
mutable struct FixedDistanceGain <: AbstractTargets
    window::Int # in minutes
    thres::LabelThresholds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function FixedDistanceGain(window, thres)
        fdg = new(window, thres, nothing, nothing)
        return fdg
    end
end

function setbase!(fdg::FixedDistanceGain, ohlcv::Ohlcv.OhlcvData)
    fdg.ohlcv = ohlcv
    fdg.df = DataFrame()
    supplement!(fdg)
end

function removebase!(fdg::FixedDistanceGain)
    fdg.ohlcv = nothing
    fdg.df = nothing
end

function supplement!(fdg::FixedDistanceGain)
    if isnothing(fdg.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in fdg - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(fdg.ohlcv)
    piv = odf[!, :pivot]
    if length(piv) > 0
        size(fdg.df, 1) > 0 && @assert fdg.df[begin, :opentime] == odf[begin, :opentime] "$(fdg.df[begin, :opentime]) != $(odf[begin, :opentime])"
        startix = max(firstindex(piv), firstindex(piv) + size(fdg.df, 1) - fdg.window + 1)
        # len(piv) = 10, len(fdg)=5, window = 3 --> startix = 1+5-3+1=4
        # recalc the last (window-1) rows due to new ohlcv max in last window element
        pivnew = view( piv, startix:lastindex(piv))
        pvlen = length(pivnew)
        if pvlen > 0
            deltalen = length(piv) - pvlen
            dfnew = DataFrame()
            maxix = zeros(UInt32, pvlen)
            minix = zeros(UInt32, pvlen)
            maxreldiff = zeros(Float32, pvlen)
            minreldiff = zeros(Float32, pvlen)
            for ix in eachindex(pivnew)
                maxix[ix] = ix == lastindex(pivnew) ? ix : argmax(view(pivnew, ix+1:min(ix+1 + fdg.window - 1, lastindex(pivnew)))) + ix
                minix[ix] = ix == lastindex(pivnew) ? ix : argmin(view(pivnew, ix+1:min(ix+1 + fdg.window - 1, lastindex(pivnew)))) + ix
                minreldiff[ix] = (pivnew[minix[ix]] - pivnew[ix]) / pivnew[ix]
                maxreldiff[ix] = (pivnew[maxix[ix]] - pivnew[ix]) / pivnew[ix]
            end
            dfnew[!, :minreldiff] = minreldiff
            dfnew[!, :maxreldiff] = maxreldiff
            dfnew[!, :longbuy] = dfnew[!, :maxreldiff] .>= fdg.thres.longbuy
            dfnew[!, :longhold] = dfnew[!, :minreldiff] .>= fdg.thres.longhold
            dfnew[!, :shortbuy] = dfnew[!, :minreldiff] .<= fdg.thres.shortbuy
            dfnew[!, :shorthold] = dfnew[!, :maxreldiff] .<= fdg.thres.shorthold
            dfnew[!, :label] .= "ignore"
            for ix in eachindex(pivnew)
                if (maxix[ix] < minix[ix]) || (dfnew[ix, :maxreldiff] > 0f0)
                    if dfnew[ix, :longbuy]
                        dfnew[ix, :label] = "longbuy"
                    elseif (ix > firstindex(pivnew)) && (dfnew[ix-1, :label] in ["longbuy", "longhold"])  # longhold can only follow longbuy or longhold
                        dfnew[ix, :label] = dfnew[ix, :longhold] ? "longhold" : "longclose"
                    end
                end
                if (maxix[ix] > minix[ix]) || (dfnew[ix, :minreldiff] < 0f0)
                    if dfnew[ix, :shortbuy]
                        dfnew[ix, :label] = "shortbuy"
                    elseif (ix > firstindex(pivnew)) && (dfnew[ix-1, :label] in ["shortbuy", "shorthold"])  # shorthold can only follow shortbuy or shorthold
                        dfnew[ix, :label] = dfnew[ix, :shorthold] ? "shorthold" : "shortclose"
                    end
                end
            end
            dfnew[!, :opentime] = odf[startix:lastindex(piv), :opentime]
            dfnew[!, :maxix] = maxix .+ deltalen
            dfnew[!, :minix] = minix .+ deltalen
            dfnew[!, :pivot] = pivnew
        end
        fdg.df = deltalen > 0 ? vcat(fdg.df[begin:startix-1, :], dfnew) : dfnew
    end
end

function timerangecut!(fdg::FixedDistanceGain)
    if isnothing(fdg.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in fdg - no time range to cut")
        return
    end
    # cut at start requires maxix correction, cut at end requires recalculation of last window elements
    startdt = Ohlcv.dataframe(fdg.ohlcv)[begin, :opentime]
    startix = Ohlcv.rowix(fdg.df[!, :opentime], startdt)
    startdeltaix = startix - firstindex(fdg.df[!, :opentime])
    enddt = Ohlcv.dataframe(fdg.ohlcv)[end, :opentime]
    endix = Ohlcv.rowix(fdg.df[!, :opentime], enddt)
    enddeltaix = lastindex(fdg.df[!, :opentime]) - endix
    endix = enddeltaix > 0 ? endix - fdg.window + 1 : endix
    fdg.df = fdg.df[startix:endix, :]
    if startdeltaix > 0
        fdg.df[!, :maxix] .-= startdeltaix
    end
    supplement!(fdg)
end

describe(fdg::FixedDistanceGain) = "$(typeof(fdg))_$(fdg.window)_label-thresholds=(longbuy=$(fdg.thres.longbuy),longhold=$(fdg.thres.longhold),shorthold=$(fdg.thres.shorthold),shortbuy=$(fdg.thres.shortbuy))"
firstrowix(fdg::FixedDistanceGain)::Int = isnothing(fdg.df) ? 1 : (size(fdg.df, 1) > 0 ? firstindex(fdg.df[!, 1]) : 1)
lastrowix(fdg::FixedDistanceGain)::Int = isnothing(fdg.df) ? 0 : (size(fdg.df, 1) > 0 ? lastindex(fdg.df[!, 1]) : 0)

"returns a dataframe with binary volume columns :longbuy, :longhold, :shorthold, :shortbuy"
function df(fdg::FixedDistanceGain, firstix::Integer=firstrowix(fdg), lastix::Integer=lastrowix(fdg))::AbstractDataFrame
    return isnothing(fdg.df) ? emptyfdgdf() : view(fdg.df, firstix:lastix, :)
end

df(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = df(fdg, Ohlcv.rowix(fdg.df[!, :opentime], startdt), Ohlcv.rowix(fdg.df[!, :opentime], enddt))
longbuybinarytargets(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = [lb ? "longbuy" : "longclose" for lb in df(fdg, startdt, enddt)[!, :longbuy]]

function labels(fdg::FixedDistanceGain, firstix::Integer=firstrowix(fdg), lastix::Integer=lastrowix(fdg))::AbstractVector
    return isnothing(fdg.df) ? [] : fdg.df[firstix:lastix, :label]
end

labels(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = labels(fdg, Ohlcv.rowix(fdg.df[!, :opentime], startdt), Ohlcv.rowix(fdg.df[!, :opentime], enddt))

function relativegain(fdg::FixedDistanceGain, firstix::Integer=firstrowix(fdg), lastix::Integer=lastrowix(fdg))::AbstractVector
    return isnothing(fdg.df) ? [] : fdg.df[firstix:lastix, :maxreldiff]
end

relativegain(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = relativegain(fdg, Ohlcv.rowix(fdg.df[!, :opentime], startdt), Ohlcv.rowix(fdg.df[!, :opentime], enddt))

function relativeloss(fdg::FixedDistanceGain, firstix::Integer=firstrowix(fdg), lastix::Integer=lastrowix(fdg))::AbstractVector
    return isnothing(fdg.df) ? [] : fdg.df[firstix:lastix, :minreldiff]
end

relativeloss(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = relativegain(fdg, Ohlcv.rowix(fdg.df[!, :opentime], startdt), Ohlcv.rowix(fdg.df[!, :opentime], enddt))

function Base.show(io::IO, fdg::FixedDistanceGain)
    println(io, "FixedDistanceGain fdg base=$(fdg.ohlcv.base) window=$(fdg.window) label thresholds=$(thresholds(fdg.thres)) size(df)=$(size(fdg.df)) $(size(fdg.df, 1) > 0 ? "from $(fdg.df[begin, :opentime]) to $(fdg.df[end, :opentime]) " : "no time range ")")
    # (verbosity >= 3) && println(io, "Features005 cfgdf=$(f5.cfgdf)")
    # (verbosity >= 2) && println(io, "Features005 config=$(f5.cfgdf[!, :config])")
    println(io, "FixedDistanceGain ohlcv=$(fdg.ohlcv)")
end


end  # module

