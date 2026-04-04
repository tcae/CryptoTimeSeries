"""
**! predicitions will be merged with targets.jl**
The asset can receive **predictions** within a given **time period** from algorithms or individuals by:

- assigning *increase*, *flat*, *decrease*
- target price
- target +-% from current price

Prediction algorithms are identified by name. Individuals are identified by name.

"""
module Targets

using EnvConfig, Ohlcv, TestOhlcv, Features
using DataFrames, Dates, Logging, CategoricalArrays
export TradeLabel, shortstrongbuy, shortbuy, shorthold, shortclose, shortstrongclose, allclose, longstrongclose, longclose, longhold, longbuy, longstrongbuy, ignore
export TrendPhase, down, flat, up
using Test

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 1

# Trend01 removed. Trend04 is the supported trend target implementation.
# Trend02 removed. Trend04 is the supported trend target implementation.

"""
returns all possible labels. "allclose" is default.
"""
@enum TradeLabel shortstrongbuy=-5 shortbuy=-4 shorthold=-3 shortclose=-2 shortstrongclose=-1 allclose=0 longstrongclose=1 longclose=2 longhold=3 longbuy=4 longstrongbuy=5 ignore=9
uniquelabels() = [lbl for lbl in instances(TradeLabel)]
tradelabelstrings(labels::AbstractVector{TradeLabel}=uniquelabels()) = string.(labels)
tradelabel(str::AbstractString, labels::AbstractVector{TradeLabel}=uniquelabels()) = labels[findfirst(x -> string(x) == str, labels)]
tradelabelix(tl::TradeLabel, labels::AbstractVector{TradeLabel}=uniquelabels()) = findfirst(x -> x == tl, labels)
tradelabelix(str::AbstractString, labels::AbstractVector{TradeLabel}=uniquelabels()) = findfirst(x -> string(x) == str, labels)
tradelabelcode(tl::TradeLabel) = Int8(tl)

@enum TrendPhase down=-1 flat=0 up=1 choppy=2

#region AbstractTargets
"Defines the targets interface that shall be provided by all target implementations. Ohlcv is provided at init and maintained as internal reference."
abstract type AbstractTargets <: EnvConfig.AbstractConfiguration end

"Adds a coin with OhlcvData to the target generation. Each coin can only have 1 associated data set."
function setbase!(targets::AbstractTargets, ohlcv::Ohlcv.OhlcvData) error("not implemented") end

"Removes the targets of a basecoin."
function removebase!(targets::AbstractTargets) error("not implemented") end

"Add newer targets to match the recent timeline of ohlcv with the newest ohlcv datapoints, i.e. datapoints newer than last(features)"
function supplement!(targets::AbstractTargets) error("not implemented") end

"Provides a description that characterizes the features"
describe(targets::AbstractTargets) = "$(typeof(targets))"

"""
Returns a vector with all supported  
- labels ::TradeLabel for classifiers
- valuelabels ::AbstractString for regressors
"""
function uniquelabels(targets::AbstractTargets)::AbstractVector error("not implemented") end

#classifier functions:
"provides a target label Bool vector of the given label"
function labelbinarytargets(targets::AbstractTargets, label::TradeLabel, firstix::Integer, lastix::Integer) error("not implemented") end
function labelbinarytargets(targets::AbstractTargets, label::TradeLabel, startdt::DateTime, enddt::DateTime) error("not implemented") end

"provides a relative gain Float32 vector of the given label"
function labelrelativegain(targets::AbstractTargets, label::TradeLabel, firstix::Integer, lastix::Integer) error("not implemented") end
function labelrelativegain(targets::AbstractTargets, label::TradeLabel, startdt::DateTime, enddt::DateTime) error("not implemented") end

"Provides a vector of class labels ::String of the target class per sample."
function labels(targets::AbstractTargets, firstix::Integer, lastix::Integer) error("not implemented") end
function labels(targets::AbstractTargets, startdt::DateTime, enddt::DateTime) error("not implemented") end

"""
Provide a vector of relative gains of the current price compared to the target price in the future. (currentprice - targetprice) / currentprice,
which provide a means to adapt a regressor
"""
function relativegain(targets::AbstractTargets, firstix::Integer, lastix::Integer) error("not implemented") end
function relativegain(targets::AbstractTargets, startdt::DateTime, enddt::DateTime) error("not implemented") end

#regressor functions:
"""
Returns a dataframe with columns for each valuelabel and rows for each sample. The values are the target values for the regression task.
The column names are equal to uniquelabels()
"""
function labelvalues(targets::AbstractTargets, firstix::Integer, lastix::Integer)::AbstractDataFrame error("not implemented") end
function labelvalues(targets::AbstractTargets, startdt::DateTime, enddt::DateTime)::AbstractDataFrame error("not implemented") end
    
function crosscheck(trd::AbstractTargets)::Vector{String} return String[] end
function crosscheck(trd::AbstractTargets, labels::AbstractVector{<:TradeLabel}, pivots::AbstractVector{<:AbstractFloat})::Vector{String} return String[] end

#region AbstractTargets

function labeldistribution(targets::CategoricalArray)
    labels = levels(targets)
    cnt = zeros(Int, length(labels))
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    for tl in targets
        cnt[levelcode(tl)] += 1
    end
    targetcount = size(targets, 1)
    dist = [(labels[i], round(cnt[i] / targetcount*100, digits=1)) for i in eachindex(labels)]
    (verbosity >= 3) && println("target label distribution in %: ", dist)
    return dist
end

labeldistribution(targets::AbstractVector{T}) where {T<:AbstractString} = labeldistribution(categorical(targets; levels=unique(targets), compress=true))

function labeldistribution(targets::AbstractVector{T}) where {T<:TradeLabel}
    cnt = Dict()
    # println("before oversampling: $(Distributions.fit(UnivariateFinite, categorical(traintargets))))")
    for tl in targets
        if tl in keys(cnt)
            cnt[tl] += 1
        else
            cnt[tl] = 1
        end
    end
    targetcount = size(targets, 1)
    dist = [(lab, round(labcnt / targetcount*100, digits=1)) for (lab, labcnt) in cnt]
    (verbosity >= 3) && println("target label distribution in %: ", dist)
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
LabelThresholds(;longbuy, longhold, shorthold, shortbuy) = LabelThresholds(longbuy, longhold, shorthold, shortbuy)
thresholds(lt::LabelThresholds)::NamedTuple = (longbuy=lt.longbuy, longhold=lt.longhold, shorthold=lt.shorthold, shortbuy=lt.shortbuy)
thresholds(lt::NamedTuple)::LabelThresholds = LabelThresholds(lt.longbuy, lt.longhold, lt.shorthold, lt.shortbuy)

defaultlabelthresholds = LabelThresholds(0.03, 0.005, -0.005, -0.03)

"""
Because the trade signals are not independent classes but an ordered set of actions, this function returns the labels that correspond to specific thresholds:

- The folllowing invariant is assumed: `longbuy > longhold >= 0 >= shorthold > shortbuy`
- a long gain shall be above `longbuy` threshold for a longbuy signal
- bought assets shall be held (but not bought) if the remaining gain is above `longhold` threshold
- bought assets shall be closed if the remaining gain is below `longhold` threshold
- a short gain shall be below `shortbuy` for a shortbuy signal
- borrowed (shortbuy) assets shall be held if the remaining loss is below `shorthold`
- borrowed (shortbuy) assets shall be closed if the remaining loss is above `shorthold`
- all thresholds are relative gain values: if backward relative then the relative gain is calculated with the target price otherwise with the current price

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
        elseif newstate == "allclose"
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
                            settradestate("allclose"))))) for rd in relativedist]
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
    g = Features.relativegain(prices, abs(ix1), abs(ix2))
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

#region FixedDistanceGain
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

uniquelabels(fdg::FixedDistanceGain) = [string(tl) for tl in Targets.uniquelabels()]

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

relativeloss(fdg::FixedDistanceGain, startdt::DateTime, enddt::DateTime) = relativeloss(fdg, Ohlcv.rowix(fdg.df[!, :opentime], startdt), Ohlcv.rowix(fdg.df[!, :opentime], enddt))

function Base.show(io::IO, fdg::FixedDistanceGain)
    println(io, "FixedDistanceGain fdg base=$(isnothing(fdg.ohlcv) ? "no ohlcv base" : fdg.ohlcv.base) window=$(fdg.window) label thresholds=$(thresholds(fdg.thres)) size(df)=$(size(fdg.df)) $(isnothing(fdg.df) ? "no df" : size(fdg.df, 1) > 0 ? "from $(isnothing(fdg.df) ? "no df" : fdg.df[begin, :opentime]) to $(isnothing(fdg.df) ? "no df" : fdg.df[end, :opentime]) " : "no time range ")")
    # (verbosity >= 3) && println(io, "Features005 cfgdf=$(f5.cfgdf)")
    # (verbosity >= 2) && println(io, "Features005 config=$(f5.cfgdf[!, :config])")
    println(io, "FixedDistanceGain ohlcv=$(fdg.ohlcv)")
end
#endregion FixedDistanceGain

#region Trend04

const _trend04diag_enabled = Ref(false)
const _trend04diag_counts = Dict{String, Int}()

"""
Enable or disable Trend04 diagnostics.

When enabled, Trend04 internals collect lightweight counters about hold candidate
acceptance/rejection paths to support root-cause analysis.
"""
function enable_trend04_diagnostics!(enabled::Bool=true)
    _trend04diag_enabled[] = enabled
    return enabled
end

"""Reset Trend04 diagnostic counters."""
function reset_trend04_diagnostics!()
    empty!(_trend04diag_counts)
    return _trend04diag_counts
end

"""Return a copy of Trend04 diagnostic counters."""
trend04_diagnostics() = Dict(_trend04diag_counts)

function _trend04diaginc!(key::AbstractString)
    if _trend04diag_enabled[]
        _trend04diag_counts[key] = get(_trend04diag_counts, key, 0) + 1
    end
    return nothing
end


"""
Provides mutual exclusive targets as well as their relative gain
- minwindow is the minimum number of minutes a trend should have
- maxwindow is the maximum number of history minutes to detect a trend with given thresholds 
- required condition: 0 <= minwindow < maxwindow
- thres provides the hold and buy thresholds for long and short trends with required condition: thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy
- a sample sequence is labeled as longbuy/shortbuy trend range if:
  - it exceeds a price difference of the longbuy/shortbuy threshold
  - exceeds minwindow samples to ignore spikes above buy threshold in opposite direction
  - exceeds the buy threshold within <= maxwindow samples
  - no opposite trend is part of such sample sequence
  - samples within a trend range with a lower price for long / higher price for short than the first trend range sample are restablishing the trend range with that very sample as new start trend range
  - not exceeding the last trend range sample (higher for long / lower for short) within maxwindow samples breaks the buy trend, i.e. the last valid trend range extreme (max for long / min for short) closes the trend range
  - the price difference before the last trend range sample is relevant if it continues the previous trend, otherwise it is not relevant for the current trend
- a sample sequence is labeled as longhold/shorthold trend if:
  - it exceeds a price difference of the longhold/shorthold threshold but stays below the longbuy/shortbuy threshold
  - is an extension to a buy sequence
  - because it is a buy trend extension, it does not need to exceed minwindow samples
  - exceeds the hold threshold within <= maxwindow samples
  - samples within a hold trend range with a lower price for long / higher price for short than the start trend range sample are breaking the trend range, i.e. the last valid trend range extreme (max for long / min for short) closes the trend range
  - not exceeding the last hold trend range sample (higher for long / lower for short) within maxwindow samples breaks the hold trend, i.e. the last valid trend range extreme (max for long / min for short) closes the trend range
  - the price difference before the last hold trend range sample is relevant if it continues the previous hold trend, otherwise it is not relevant for the current trend
- all other sample sequences are labeled as allclose
"""
mutable struct Trend04 <: AbstractTargets
    minwindow::Int # in minutes
    maxwindow::Int # in minutes
    thres::LabelThresholds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function Trend04(minwindow, maxwindow, thres)
        @assert 0 <= minwindow < maxwindow "condition violated: 0 <= minwindow=$(minwindow) < maxwindow=$(maxwindow)"
        @assert thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy "condition violated: thres.shortbuy=$(thres.shortbuy) <= thres.shorthold=$(thres.shorthold) <= thres.longhold=$(thres.longhold) <= fdg.thres.longbuy=$(thres.longbuy)"
        trd = new(minwindow, maxwindow, thres, nothing, nothing)
        return trd
    end
end

function setbase!(trd::Trend04, ohlcv::Ohlcv.OhlcvData)
    trd.ohlcv = ohlcv
    trd.df = DataFrame()
    supplement!(trd)
end

function removebase!(trd::Trend04)
    trd.ohlcv = nothing
    trd.df = nothing
end

function _fillsegment!(labels, relix, endix)
    startix = relix[endix]
    if (endix > firstindex(labels)) && (labels[endix - 1] == labels[endix]) && (relix[endix - 1] == relix[endix])
        return
    end
    if labels[endix] == longhold
        if labels[startix] != longhold
            startix = startix + 1
        end
    elseif labels[endix] == shorthold
        if labels[startix] != shorthold
            startix = startix + 1
        end
    end

    for ix in startix:(endix-1)
        labels[ix] = labels[endix]
        relix[ix] = relix[endix]
    end
end

"forward looking relative difference from ix to endix"
_reldiff(piv, ix, endix) = (piv[endix] - piv[ix]) / piv[ix]

"true if a long continuation from the same anchor is monotonic at endix"

"""
Updates trd.df[maxbackix:endix, [:label, :relix]] based on the backtrace to:
- find a connecting buy segment until maxbackix
  - to connect a close segment
  - to connect a hold segment
  - to extend the buy segment
- establish a new buy segment if it exceeds thresholds and span more or equal minwindow samples
- fall back to default close segment
"""
function _filltrendanchor!(trd::Trend04, maxbackix, endix)
    _trend04diaginc!("calls.filltrendanchor")
    piv = Ohlcv.dataframe(trd.ohlcv)[!, :pivot]
    labels = trd.df[!, :label]
    relix = trd.df[!, :relix]
    @assert length(piv) == length(labels) == length(relix)  "length mismatch: length(piv)=$(length(piv)) must equal length(labels)=$(length(labels)) and length(relix)=$(length(relix))"
    (maxbackix == endix) && return 
    #* relix contains row indices of global pivot and not of pivnew
    bestlongix = bestshortix = maxbackcloseix = endix
    labels[endix] = allclose
    for ix in (endix-1):-1:maxbackix
        if (labels[endix] == labels[ix] == labels[ix+1] == allclose) && ((endix - ix) >= trd.minwindow)
            maxbackcloseix = ix
            bestlongix = (_reldiff(piv, bestlongix, endix) < _reldiff(piv, ix, endix)) ? ix : bestlongix
            bestshortix = (_reldiff(piv, bestshortix, endix) > _reldiff(piv, ix, endix)) ? ix : bestshortix
        end
        if labels[ix] in [longbuy, longhold]
            if _reldiff(piv, ix, endix) < 0 # price decline, break backtrace but check if hold, shortbuy or allclose
                anchorix = (labels[ix] == longbuy) ? relix[ix] : relix[relix[ix]] # single or double-deref to bestlongix
                holdrelix = (labels[ix] == longbuy) ? ix : relix[ix] # ixbuy: last buy bar, preserved by _fillsegment! for hold
                # Hold semantic: price is expected to continue rising but small dips (within the
                # shorthold tolerance from the last buy peak) are acceptable. Use the buy peak
                # (holdrelix) as reference and shorthold as dip-tolerance threshold.
                holdcheckix = holdrelix
                _trend04diaginc!("cand.longhold.reversal")
                    if _reldiff(piv, holdcheckix, endix) >= trd.thres.shorthold &&
                        (minimum(view(piv, ix:endix)) >= piv[holdrelix] * (1 + trd.thres.shorthold))
                    _trend04diaginc!("acc.longhold.reversal")
                    labels[endix] = longhold
                    relix[endix] = holdrelix
                    _fillsegment!(labels, relix, endix)
                else
                    _trend04diaginc!("rej.longhold.reversal.threshold")
                    # Anchor reversal at local high so a short range starts at its segment extreme.
                    searchrange = (ix + 1):endix
                    _, maxrelix = findmax(view(piv, searchrange))
                    newix = first(searchrange) + maxrelix - 1
                    opposite_span = endix - newix + 1
                    opposite_short_ok = (_reldiff(piv, newix, endix) <= trd.thres.shortbuy) && (piv[endix] == minimum(view(piv, newix:endix)))
                    if (opposite_span >= trd.minwindow) && opposite_short_ok
                        labels[endix] = shortbuy
                        relix[endix] = newix 
                        _fillsegment!(labels, relix, endix)
                    elseif opposite_short_ok && (opposite_span < trd.minwindow)
                        # Micro short peak from long trend: explicit transient class (diagnostics)
                        # and keep longhold continuity if entry-anchor hold is still valid.
                        _trend04diaginc!("transient.micro.shortpeak.from_long")
                        if _reldiff(piv, holdrelix, endix) >= trd.thres.shorthold
                            _trend04diaginc!("acc.longhold.from_micro_shortpeak")
                            labels[endix] = longhold
                            relix[endix] = holdrelix
                            _fillsegment!(labels, relix, endix)
                        else
                            labels[endix] = allclose
                            relix[endix] = newix
                        end
                    else # price declined but not enough for a shortbuy, so it is for now no long trend anymore
                        labels[endix] = allclose
                        relix[endix] = newix 
                    end
                end
                break # anchor found at hold/opposite trend, stop backtrace
            else # pdiff >=0 , i.e. continuation of long trend either hold or buy
                if labels[ix] == longhold
                    # Double-dereference: relix[ix]=ixbuy (last buy bar before hold),
                    # relix[ixbuy]=bestlongix (original entry anchor). This ensures the
                    # threshold check uses the cumulative gain from entry, not from the peak.
                    anchorix = relix[relix[ix]]
                    ixbuy = relix[ix]  # last buy bar = the peak of the buy phase
                    # A promoted longbuy sub-segment starts at the current hold sample `ix`,
                    # so both the buy threshold and minwindow must be met from that local
                    # restart point rather than from an older buy anchor.
                    localbuyreldiff = _reldiff(piv, ix, endix)
                    localbuyspan = endix - ix + 1
                    islocalstartlow = piv[ix] == minimum(view(piv, ix:endix))
                    # Hold continues as long as price stays within shorthold dip-tolerance of
                    # the last buy peak (ixbuy). Use the same reference and threshold as the
                    # reversal path for consistency.
                    holdcheckix = relix[ix]  # ixbuy: last buy bar = buy peak
                    _trend04diaginc!("cand.longhold.from_longhold")
                    # Compare vs last buy bar (the peak), not just the previous hold bar.
                    # This prevents small bounces within the declining hold phase from
                    # triggering a new buy sub-segment spuriously.
                    # Use strict > to avoid equal-amplitude periodic signals creating spurious buys.
                    isnewhigh = piv[endix] > piv[ixbuy]
                    if isnewhigh && islocalstartlow && (localbuyspan >= trd.minwindow) && (_reldiff(piv, anchorix, endix) >= trd.thres.longbuy) && (localbuyreldiff >= trd.thres.longbuy)
                        _trend04diaginc!("rej.longhold.promoted_longbuy")
                        labels[endix] = longbuy
                        relix[endix] = ix # buy sub-segment starts after last hold bar, preserving earlier hold bars
                        _fillsegment!(labels, relix, endix)
                          elseif (!isnewhigh) && (_reldiff(piv, holdcheckix, endix) >= trd.thres.shorthold) &&
                                       (minimum(view(piv, ix:endix)) >= piv[holdcheckix] * (1 + trd.thres.shorthold))
                        _trend04diaginc!("acc.longhold.from_longhold")
                        labels[endix] = longhold
                        relix[endix] = relix[ix] # extend hold segment by anchor take over because there is no sell buy transaction for the hold phase extension
                        _fillsegment!(labels, relix, endix)
                    else
                        _trend04diaginc!("rej.longhold.from_longhold.threshold")
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix
                        _fillsegment!(labels, relix, endix)
                    end
                else # labels[ix] == longbuy
                    anchorix = relix[ix]  # relix[lastbuybar] = bestlongix (original entry anchor)
                    isnewhigh = piv[endix] >= piv[ix]  # only needs to exceed last directional extreme, not intermediate allclose bars
                    isanchormin = piv[anchorix] == minimum(view(piv, anchorix:endix))
                    _trend04diaginc!("cand.longhold.from_longbuy")
                    if isnewhigh && isanchormin && (_reldiff(piv, anchorix, endix) >= trd.thres.longbuy) # minwindow condition is ensured in continuation case
                        _trend04diaginc!("rej.longhold.continues_longbuy")
                        labels[endix] = longbuy 
                        relix[endix] = relix[ix] # extend segment; there is no sell-buy transaction for the buy phase extension
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, ix, endix) >= trd.thres.longhold)
                        _trend04diaginc!("acc.longhold.from_longbuy")
                        labels[endix] = longhold
                        relix[endix] = ix # the hold phase starts after the previous buy sample; _fillsegment! skips ix itself (last buy bar)
                        _fillsegment!(labels, relix, endix)
                    else # price not above hold threshold from entry anchor, so no long trend anymore
                        _trend04diaginc!("rej.longhold.from_longbuy.threshold")
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix 
                        _fillsegment!(labels, relix, endix)
                    end
                end
                break # anchor found at continuation, stop backtrace
            end
        elseif labels[ix] in [shortbuy, shorthold]
            if _reldiff(piv, ix, endix) > 0 # price increase, break backtrace but check if hold, longbuy or allclose
                anchorix = (labels[ix] == shortbuy) ? relix[ix] : relix[relix[ix]] # single or double-deref to bestshortix
                holdrelix = (labels[ix] == shortbuy) ? ix : relix[ix] # ixshortbuy: last shortbuy bar, preserved by _fillsegment! for hold
                # Hold semantic: price is expected to continue falling but small recoveries
                # (within longhold tolerance above the last shortbuy trough) are acceptable.
                # Use the sell trough (holdrelix) as reference and longhold as rally-tolerance.
                holdcheckix = holdrelix
                _trend04diaginc!("cand.shorthold.reversal")
                    if _reldiff(piv, holdcheckix, endix) <= trd.thres.longhold &&
                        (maximum(view(piv, ix:endix)) <= piv[holdrelix] * (1 + trd.thres.longhold))
                    _trend04diaginc!("acc.shorthold.reversal")
                    labels[endix] = shorthold
                    relix[endix] = holdrelix
                    _fillsegment!(labels, relix, endix)
                else
                    _trend04diaginc!("rej.shorthold.reversal.threshold")
                    # Anchor reversal at local low so a long range starts at its segment extreme.
                    searchrange = (ix + 1):endix
                    _, minrelix = findmin(view(piv, searchrange))
                    newix = first(searchrange) + minrelix - 1
                    opposite_span = endix - newix + 1
                    opposite_long_ok = (_reldiff(piv, newix, endix) >= trd.thres.longbuy) && (piv[endix] == maximum(view(piv, newix:endix)))
                    if (opposite_span >= trd.minwindow) && opposite_long_ok
                        labels[endix] = longbuy
                        relix[endix] = newix 
                        _fillsegment!(labels, relix, endix)
                    elseif opposite_long_ok && (opposite_span < trd.minwindow)
                        # Micro long peak from short trend: explicit transient class (diagnostics)
                        # and keep shorthold continuity if entry-anchor hold is still valid.
                        _trend04diaginc!("transient.micro.longpeak.from_short")
                        if _reldiff(piv, holdrelix, endix) <= trd.thres.longhold
                            _trend04diaginc!("acc.shorthold.from_micro_longpeak")
                            labels[endix] = shorthold
                            relix[endix] = holdrelix
                            _fillsegment!(labels, relix, endix)
                        else
                            labels[endix] = allclose
                            relix[endix] = newix
                        end
                    else # price increased but not enough for a longbuy, so it is for now no short trend anymore
                        labels[endix] = allclose
                        relix[endix] = newix  
                    end
                end # end else (reversal)
                break # anchor found at hold/opposite trend, stop backtrace
            else # pdiff <=0 , i.e. continuation of short trend either hold or buy
                if labels[ix] == shorthold
                    # Double-dereference: relix[ix]=ixshortbuy (last shortbuy bar before hold),
                    # relix[ixshortbuy]=bestshortix (original entry anchor). This ensures the
                    # threshold check uses the cumulative gain from entry, not from the trough.
                    anchorix = relix[relix[ix]]
                    ixshortbuy = relix[ix]  # last shortbuy bar = the trough of the buy phase
                    # A promoted shortbuy sub-segment starts at the current hold sample `ix`,
                    # so both the buy threshold and minwindow must be met from that local
                    # restart point rather than from an older buy anchor.
                    localbuyreldiff = _reldiff(piv, ix, endix)
                    localbuyspan = endix - ix + 1
                    islocalstarthigh = piv[ix] == maximum(view(piv, ix:endix))
                    # Hold continues as long as price stays within longhold rally-tolerance of
                    # the last shortbuy trough (ixshortbuy). Use the same reference and threshold
                    # as the reversal path for consistency.
                    holdcheckix = relix[ix]  # ixshortbuy: last shortbuy bar = sell trough
                    _trend04diaginc!("cand.shorthold.from_shorthold")
                    # Compare vs last shortbuy bar (the trough), not just the previous hold bar.
                    # This prevents small reversals within the recovering hold phase from
                    # triggering a new shortbuy sub-segment spuriously.
                    # Use strict < to avoid equal-amplitude periodic signals creating spurious buys.
                    isnewlow = piv[endix] < piv[ixshortbuy]
                    if isnewlow && islocalstarthigh && (localbuyspan >= trd.minwindow) && (_reldiff(piv, anchorix, endix) <= trd.thres.shortbuy) && (localbuyreldiff <= trd.thres.shortbuy)
                        _trend04diaginc!("rej.shorthold.promoted_shortbuy")
                        labels[endix] = shortbuy
                        relix[endix] = ix # buy sub-segment starts after last hold bar, preserving earlier short hold bars
                        _fillsegment!(labels, relix, endix)
                          elseif (!isnewlow) && (_reldiff(piv, holdcheckix, endix) <= trd.thres.longhold) &&
                                       (maximum(view(piv, ix:endix)) <= piv[holdcheckix] * (1 + trd.thres.longhold))
                        _trend04diaginc!("acc.shorthold.from_shorthold")
                        labels[endix] = shorthold
                        relix[endix] = relix[ix] # extend hold segment by anchor take over because there is no sell buy transaction for the hold phase extension
                        _fillsegment!(labels, relix, endix)
                    else
                        _trend04diaginc!("rej.shorthold.from_shorthold.threshold")
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix
                        _fillsegment!(labels, relix, endix)
                    end
                else # labels[ix] == shortbuy
                    anchorix = relix[ix]  # relix[lastshortbuybar] = bestshortix (original entry anchor)
                    isnewlow = piv[endix] <= piv[ix]  # only needs to exceed last directional extreme, not intermediate allclose bars
                    isanchormax = piv[anchorix] == maximum(view(piv, anchorix:endix))
                    _trend04diaginc!("cand.shorthold.from_shortbuy")
                    if isnewlow && isanchormax && (_reldiff(piv, anchorix, endix) <= trd.thres.shortbuy) # minwindow condition is ensured in continuation case
                        _trend04diaginc!("rej.shorthold.continues_shortbuy")
                        labels[endix] = shortbuy
                        relix[endix] = relix[ix] # extend segment; there is no sell-buy transaction for the buy phase extension
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, ix, endix) <= trd.thres.shorthold)
                        _trend04diaginc!("acc.shorthold.from_shortbuy")
                        labels[endix] = shorthold
                        relix[endix] = ix # the hold phase starts after the previous shortbuy sample; _fillsegment! skips ix itself (last shortbuy bar)
                        _fillsegment!(labels, relix, endix)
                    else # price not below hold threshold from entry anchor, so no short trend anymore
                        _trend04diaginc!("rej.shorthold.from_shortbuy.threshold")
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix 
                        _fillsegment!(labels, relix, endix)
                    end
                end
                break # anchor found at continuation, stop backtrace
            end
        else
            @assert labels[ix] == allclose "unexpected label at ix=$(ix): labels[ix]=$(labels[ix]) expected=allclose=$(allclose)"
        end
    end
    if labels[endix] == allclose
        if (bestlongix < endix) || (bestshortix < endix) # it is ensured that (endix - bestix +1) >= minwindow for bestix because it is only updated if it is the case
            longgain = bestlongix < endix ? _reldiff(piv, bestlongix, endix) : -Inf
            shortgain = bestshortix < endix ? _reldiff(piv, bestshortix, endix) : Inf
            longcandidate = (longgain >= trd.thres.longbuy) &&
                            ((endix - bestlongix + 1) <= trd.maxwindow) &&
                            (piv[bestlongix] == minimum(view(piv, bestlongix:endix)))
            shortcandidate = (shortgain <= trd.thres.shortbuy) &&
                             ((endix - bestshortix + 1) <= trd.maxwindow) &&
                             (piv[bestshortix] == maximum(view(piv, bestshortix:endix)))

            if longcandidate && (!shortcandidate || (longgain >= abs(shortgain)))
                if piv[endix] == maximum(view(piv, bestlongix:endix))
                    labels[endix] = longbuy
                    relix[endix] = bestlongix 
                    _fillsegment!(labels, relix, endix)
                else
                    labels[endix] = allclose
                    relix[endix] = maxbackcloseix
                end
            elseif shortcandidate
                if piv[endix] == minimum(view(piv, bestshortix:endix))
                    labels[endix] = shortbuy
                    relix[endix] = bestshortix 
                    _fillsegment!(labels, relix, endix)
                else
                    labels[endix] = allclose
                    relix[endix] = maxbackcloseix
                end
            else
                relix[endix] = maxbackcloseix
            end
        else # no buy trend established, so it is allclose with best anchor as reference for next samples
            relix[endix] = maxbackcloseix  # is not used
        end
    end
    _trend04diaginc!("label." * string(labels[endix]))
end

function supplement!(trd::Trend04)
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(trd.ohlcv)
    piv = odf[!, :pivot]
    ot = odf[!, :opentime]
    if length(piv) > 0
        if size(trd.df, 1) > 0
            @assert trd.df[begin, :opentime] == ot[begin] "$(trd.df[begin, :opentime]) != $(ot[begin])"
        end
        startix = firstindex(piv) + size(trd.df, 1)
        dfnewlen = length(piv) - size(trd.df, 1)
        if dfnewlen > 0
            dfnew = DataFrame(relix=zeros(Int, dfnewlen), label=fill(allclose, dfnewlen), opentime=ot[startix:end])
            trd.df = vcat(trd.df, dfnew)
            lpct = -1
            for ix in startix:lastindex(trd.df, 1)
                pct = round(ix / lastindex(trd.df, 1) * 100)
                (verbosity >= 2) && (pct != lpct) && print("processed: $(pct)% ($ix of $(lastindex(trd.df, 1))))\r")
                # print("processed: $(pct)% ($ix of $(lastindex(trd.df, 1))))\r")
                lpct = pct
                maxbackix = max(firstindex(trd.df, 1), ix - trd.maxwindow + 1)
                _filltrendanchor!(trd, maxbackix, ix)
            end
            (verbosity >= 3) && println()
            @assert all(in.(trd.df[!, :label], Ref([longbuy, longhold, shortbuy, shorthold, allclose]))) "unique(labels)=$(unique(trd.df[!, :label]))"
        end
    end
end

uniquelabels(trd::Trend04) = [longbuy, longhold, shortbuy, shorthold, allclose]

function timerangecut!(trd::Trend04)
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - no time range to cut")
        return
    end
    # cut at start requires maxix correction, cut at end requires recalculation of last maxwindow elements
    startdt = Ohlcv.dataframe(trd.ohlcv)[begin, :opentime]
    startix = Ohlcv.rowix(trd.df[!, :opentime], startdt)
    startdeltaix = startix - firstindex(trd.df[!, :opentime])
    enddt = Ohlcv.dataframe(trd.ohlcv)[end, :opentime]
    endix = Ohlcv.rowix(trd.df[!, :opentime], enddt)
    enddeltaix = lastindex(trd.df[!, :opentime]) - endix
    endix = enddeltaix > 0 ? endix - trd.maxwindow + 1 : endix
    trd.df = trd.df[startix:endix, :]
    if startdeltaix > 0
        trd.df[!, :relix] .-= startdeltaix
    end
    supplement!(trd)
end

describe(trd::Trend04) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_maxwindow=$(trd.maxwindow)_minwindow=$(trd.minwindow)_thresholds=(longbuy=$(trd.thres.longbuy)_longhold=$(trd.thres.longhold)_shorthold=$(trd.thres.shorthold)_shortbuy=$(trd.thres.shortbuy))"
firstrowix(trd::Trend04)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Trend04)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

# df(trd::Trend04, startdt::DateTime, enddt::DateTime) = df(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))
# longbuybinarytargets(trd::Trend04, startdt::DateTime, enddt::DateTime) = [lb ? "longbuy" : "longclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]
# shortbuybinarytargets(trd::Trend04, startdt::DateTime, enddt::DateTime) = [lb ? "shortbuy" : "shortclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]

labelbinarytargets(trd::Trend04, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labels(trd, firstix, lastix) .== label
labelbinarytargets(trd::Trend04, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelbinarytargets(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labelrelativegain(trd::Trend04, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labelbinarytargets(trd, label, firstix, lastix) .* relativegain(trd, firstix, lastix)
labelrelativegain(trd::Trend04, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelrelativegain(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labels(trd::Trend04, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :label)
labels(trd::Trend04, startdt::DateTime, enddt::DateTime) = labels(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

function relativegain(trd::Trend04, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector 
    if isnothing(trd.df) || isnothing(trd.ohlcv)
        return zeros(Float32, 0)
    end
    piv = Ohlcv.dataframe(trd.ohlcv)[!, :pivot]
    relix = trd.df[!, :relix]
    @assert length(piv) == length(relix) "length mismatch: length(piv)=$(length(piv)) must equal length(relix)=$(length(relix))"
    reldiff = [_reldiff(piv, relix[ix], ix) for ix in firstix:lastix]
    return reldiff
end

relativegain(trd::Trend04, startdt::DateTime, enddt::DateTime) = relativegain(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

function Base.show(io::IO, trd::Trend04)
    println(io, "Trend04 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base) maxwindow=$(trd.maxwindow) label thresholds=$(thresholds(trd.thres)) $(isnothing(trd.df) ? "no df" : "from $(trd.df[begin, :opentime]) to $(trd.df[end, :opentime]) ")")
    println(io, "Trend04 ohlcv=$(trd.ohlcv)")
end

"""
    _max_consecutive_run(pivots, fromix, toix, refval, cmpfn) -> Int

Return the length of the longest consecutive run of indices `i in fromix:toix`
where `cmpfn(pivots[i], refval)` is true.  Used by `crosscheck(::Trend04, ...)`
to measure spike/plateau widths rather than distances from segment boundaries.
"""
function _max_consecutive_run(pivots::AbstractVector{<:AbstractFloat}, fromix::Int, toix::Int, refval::AbstractFloat, cmpfn)::Int
    max_run = 0
    cur_run = 0
    for i in fromix:toix
        if cmpfn(pivots[i], refval)
            cur_run += 1
            max_run = max(max_run, cur_run)
        else
            cur_run = 0
        end
    end
    return max_run
end

"""
Crosschecks a `Trend04` label vector against the Trend04 constraints and returns
list of detailed violations.

The function checks, for each labeled range:
- labels must be part of `uniquelabels(trd)`
- hold segments must be preceded by a buy segment of the same direction
- segment start/end price delta and span must satisfy label-specific thresholds
    and constraints
- long/short ranges must not continue for `maxwindow` samples without a new
  direction-consistent extreme (record high for long, record low for short)

Returns `Vector{String}` where an empty vector means valid.
"""
function crosscheck(trd::Trend04, labels::AbstractVector{<:TradeLabel}, pivots::AbstractVector{<:AbstractFloat})::Vector{String}
    issues = String[]
    n = length(labels)

    if n != length(pivots)
        push!(issues, "length mismatch: length(labels)=$(n) must equal length(pivots)=$(length(pivots))")
        return issues
    end
    if n == 0
        push!(issues, "empty inputs: length(labels)=0 and length(pivots)=0")
        return issues
    end

    allowed = Set(uniquelabels(trd))
    longlabels = Set((longbuy, longhold))
    shortlabels = Set((shortbuy, shorthold))
    longbuylabels = Set((longbuy,))
    shortbuylabels = Set((shortbuy,))

    # Hold labels are intentionally excluded from :long/:short direction for extreme checks:
    # hold represents a declining/flat continuation phase, not a new directional entry,
    # so start/end extreme invariants do not apply to hold sub-segments.
    _direction(lbl::TradeLabel) = lbl in longbuylabels ? :long : (lbl in shortbuylabels ? :short : :close)

    for ix in eachindex(labels)
        if !(labels[ix] in allowed)
            push!(issues, "invalid label at ix=$(ix): label=$(labels[ix]) allowed=$(collect(allowed))")
        end
    end

    # Build contiguous directional ranges: long/short/allclose
    # Also split on relix changes within the same direction: two adjacent directional
    # sub-segments that share a label but have different relix values are independent
    # segments (different buy anchors) and must be validated separately.
    relix_arr = (!isnothing(trd.df) && size(trd.df, 1) == n) ? trd.df[!, :relix] : nothing
    segments = Tuple{Int, Int, Symbol}[] # trend start, trend end, trend direction
    segstart = firstindex(labels)
    segdir = _direction(labels[segstart])
    for ix in (firstindex(labels)+1):lastindex(labels)
        d = _direction(labels[ix])
        relix_changed = (!isnothing(relix_arr) && d != :close && relix_arr[ix] != relix_arr[ix - 1])
        if d != segdir || relix_changed
            push!(segments, (segstart, ix - 1, segdir))
            segstart = ix
            segdir = d
        end
    end
    push!(segments, (segstart, lastindex(labels), segdir))

    # Hold segments must be preceded by a buy segment of the same direction.
    # This checks contiguous label segments, not coarse directional ranges.
    labelsegments = Tuple{Int, Int, TradeLabel}[]
    lstart = firstindex(labels)
    llbl = labels[lstart]
    for ix in (firstindex(labels)+1):lastindex(labels)
        if labels[ix] != llbl
            push!(labelsegments, (lstart, ix - 1, llbl))
            lstart = ix
            llbl = labels[ix]
        end
    end
    push!(labelsegments, (lstart, lastindex(labels), llbl))

    for segix in eachindex(labelsegments)
        lss, lse, lbl = labelsegments[segix]
        if lbl == longhold
            if (segix == firstindex(labelsegments)) || (labelsegments[segix - 1][3] != longbuy)
                push!(issues, "long hold segment at $(lss):$(lse) must be preceded by a longbuy segment")
            end
        elseif lbl == shorthold
            if (segix == firstindex(labelsegments)) || (labelsegments[segix - 1][3] != shortbuy)
                push!(issues, "short hold segment at $(lss):$(lse) must be preceded by a shortbuy segment")
            end
        end
    end

    # Segment-level threshold and span checks using extrema-consistent
    # relative gain/loss within each segment (not only start/end).
    six = firstindex(segments)
    ss, se, dir = segments[six]
    for (lss, lse, lbl) in labelsegments
        if lbl == allclose
            continue
        end
        while (se < lse) && (six < lastindex(segments))
            six += 1
            ss, se, dir = segments[six]
        end
        
        withinsegmentoffset = ss < lss ? 1 : 0 # use last sample of previous labelsegment if this is in the same segment
        reldiff = _reldiff(pivots, lss-withinsegmentoffset, lse)
        span = lse - lss + 1

        if lbl == longbuy
            # For the first sub-segment of a coarse buy segment, use the actual entry anchor
            # (relix[lss] = the valley/bestlongix) for both span and threshold checks.
            # This avoids false violations when the sub-segment starts mid-valley due to
            # allclose gaps between the anchor bar and the first labeled buy bar.
            if ss == lss && !isnothing(relix_arr)
                buy_anchor = relix_arr[lss]
                anchor_span = lse - buy_anchor + 1
                anchor_reldiff = _reldiff(pivots, buy_anchor, lse)
                if anchor_span < trd.minwindow
                    push!(issues, "longbuy segment $(lss):$(lse) violates minwindow: span=$(anchor_span) < minwindow=$(trd.minwindow)")
                end
                if anchor_reldiff < trd.thres.longbuy
                    push!(issues, "longbuy segment $(lss):$(lse) violates threshold: reldiff=$(anchor_reldiff) < longbuy=$(trd.thres.longbuy)")
                end
            else
                if (ss == lss) && (span < trd.minwindow)
                    push!(issues, "longbuy segment $(lss):$(lse) violates minwindow: span=$(span) < minwindow=$(trd.minwindow)")
                end
                if reldiff < trd.thres.longbuy
                    push!(issues, "longbuy segment $(lss):$(lse) violates threshold: reldiff=$(reldiff) < longbuy=$(trd.thres.longbuy)")
                end
            end
        elseif lbl == shortbuy
            # Symmetric to longbuy: use actual entry anchor for first sub-segment checks.
            if ss == lss && !isnothing(relix_arr)
                buy_anchor = relix_arr[lss]
                anchor_span = lse - buy_anchor + 1
                anchor_reldiff = _reldiff(pivots, buy_anchor, lse)
                if anchor_span < trd.minwindow
                    push!(issues, "shortbuy segment $(lss):$(lse) violates minwindow: span=$(anchor_span) < minwindow=$(trd.minwindow)")
                end
                if anchor_reldiff > trd.thres.shortbuy
                    push!(issues, "shortbuy segment $(lss):$(lse) violates threshold: reldiff=$(anchor_reldiff) > shortbuy=$(trd.thres.shortbuy)")
                end
            else
                if (ss == lss) && (span < trd.minwindow)
                    push!(issues, "shortbuy segment $(lss):$(lse) violates minwindow: span=$(span) < minwindow=$(trd.minwindow)")
                end
                if reldiff > trd.thres.shortbuy
                    push!(issues, "shortbuy segment $(lss):$(lse) violates threshold: reldiff=$(reldiff) > shortbuy=$(trd.thres.shortbuy)")
                end
            end
        elseif lbl == longhold
            # Use cumulative gain from original buy entry anchor (double-dereference through ixbuy).
            # Hold semantic: price expected to continue rising; small dips up to |shorthold| below
            # the last buy peak are acceptable. Use the END bar's relix (single-deref: ixbuy = last
            # buy peak) since multi-step hold extensions can have different anchors at start vs end.
            hold_anchor = (!isnothing(relix_arr)) ? relix_arr[lse] : max(firstindex(pivots), lss - 1)
            reldiff_hold = _reldiff(pivots, hold_anchor, lse)
            # Hold is valid as long as price hasn't dipped more than |shorthold| below the buy peak.
            if reldiff_hold < trd.thres.shorthold
                push!(issues, "longhold segment $(lss):$(lse) violates hold threshold: reldiff=$(reldiff_hold) < longhold=$(trd.thres.longhold)")
            end
        elseif lbl == shorthold
            # Symmetric to longhold: use the END bar's relix (single-deref: ixshortbuy = sell
            # trough); rally from trough must not exceed longhold%.
            hold_anchor = (!isnothing(relix_arr)) ? relix_arr[lse] : max(firstindex(pivots), lss - 1)
            reldiff_hold = _reldiff(pivots, hold_anchor, lse)
            # Hold is valid as long as price hasn't rallied more than longhold above the sell trough.
            if reldiff_hold > trd.thres.longhold
                push!(issues, "shorthold segment $(lss):$(lse) violates hold threshold: reldiff=$(reldiff_hold) > shorthold=$(trd.thres.shorthold)")
            end
        end
    end

    # Range continuation must break when no new directional extreme is reached within maxwindow.
    for (ss, se, dir) in segments
        if dir == :long
            segminix = ss + argmin(view(pivots, ss:se)) - 1
            if segminix != ss
                if _max_consecutive_run(pivots, ss+1, se, pivots[ss], <) >= trd.minwindow
                    push!(issues, "long range $(ss):$(se) must start at a segment low extreme: start value=$(pivots[ss]), segment min=$(pivots[segminix]) at ix=$(segminix), minwindow=$(trd.minwindow)")
                end
            end

            lastrecordix = ss
            recordval = pivots[ss]
            for ix in ss:se
                if pivots[ix] > recordval
                    recordval = pivots[ix]
                    lastrecordix = ix
                end
                if (ix - lastrecordix) >= trd.maxwindow
                    push!(issues, "long range $(ss):$(se) violates maxwindow continuation at ix=$(ix): last record high at ix=$(lastrecordix), maxwindow=$(trd.maxwindow)")
                    break
                end
            end
            if lastrecordix != se
                if _max_consecutive_run(pivots, ss, se-1, pivots[se], >) >= trd.minwindow
                    push!(issues, "long range $(ss):$(se) must end at a segment high extreme: last record high at ix=$(lastrecordix), range end=$(se), minwindow=$(trd.minwindow)")
                end
            end
        elseif dir == :short
            segmaxix = ss + argmax(view(pivots, ss:se)) - 1
            if segmaxix != ss
                if _max_consecutive_run(pivots, ss+1, se, pivots[ss], >) >= trd.minwindow
                    push!(issues, "short range $(ss):$(se) must start at a segment high extreme: start value=$(pivots[ss]), segment max=$(pivots[segmaxix]) at ix=$(segmaxix), minwindow=$(trd.minwindow)")
                end
            end

            lastrecordix = ss
            recordval = pivots[ss]
            for ix in ss:se
                if pivots[ix] < recordval
                    recordval = pivots[ix]
                    lastrecordix = ix
                end
                if (ix - lastrecordix) >= trd.maxwindow
                    push!(issues, "short range $(ss):$(se) violates maxwindow continuation at ix=$(ix): last record low at ix=$(lastrecordix), maxwindow=$(trd.maxwindow)")
                    break
                end
            end
            if lastrecordix != se
                if _max_consecutive_run(pivots, ss, se-1, pivots[se], <) >= trd.minwindow
                    push!(issues, "short range $(ss):$(se) must end at a segment low extreme: last record low at ix=$(lastrecordix), range end=$(se), minwindow=$(trd.minwindow)")
                end
            end
        end
    end

    return issues
end

"""
Convenience overload that crosschecks labels from `trd.df` against pivot prices
from `trd.ohlcv`.

Returns `Vector{String}` where an empty vector means valid.
"""
function crosscheck(trd::Trend04)::Vector{String}
    issues = String[]
    if isnothing(trd.ohlcv)
        push!(issues, "missing ohlcv in Trend04: setbase! must be called before crosscheck(trd)")
    end
    if isnothing(trd.df)
        push!(issues, "missing df in Trend04: setbase! or supplement! must populate labels before crosscheck(trd)")
    end
    if !isempty(issues)
        return issues
    end

    pivots = Ohlcv.dataframe(trd.ohlcv)[!, :pivot]
    labels = trd.df[!, :label]
    return crosscheck(trd, labels, pivots)
end
#endregion Trend04


#region Bounds01

"""
Provides an lowtarget and hightarget within the coming `window` samples that indicate the upper and lower price targets, which can be used as limits for trade orders.
For convenience, provides an hightarget and lowertarget estimation within the coming `window` samples based on center and width.
"""
mutable struct Bounds01 <: AbstractTargets
    window::Int # in minutes
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing} # columns: opentime, lowtarget, hightarget
    function Bounds01(window)
        @assert 0 < window  "condition violated: 0 < window=$(window) "
        trd = new(window, nothing, nothing)
        return trd
    end
end

function setbase!(trd::Bounds01, ohlcv::Ohlcv.OhlcvData)
    @assert size(Ohlcv.dataframe(ohlcv), 1) > 0 "condition violated: ohlcv=$(ohlcv) has no data rows"
    trd.ohlcv = ohlcv
    trd.df = DataFrame()
    supplement!(trd)
end

function removebase!(trd::Bounds01)
    trd.ohlcv = nothing
    trd.df = nothing
end


function supplement!(trd::Bounds01)
    if isnothing(trd.ohlcv) 
        (verbosity >= 2) && println("no ohlcv found in trd - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(trd.ohlcv)
    piv = odf[!, :pivot]
    if length(piv) > 0
        if size(trd.df, 1) > 0
            @assert trd.df[begin, :opentime] == odf[begin, :opentime] "$(trd.df[begin, :opentime]) != $(odf[begin, :opentime])"
            startix = firstindex(piv) + size(trd.df, 1)
        else
            startix = firstindex(piv)
        end
        
        odfview = view(odf, startix:lastindex(odf, 1), :)
        
        if size(odfview, 1) > 0
            dfnew = DataFrame()
            dfnew[!, :hightarget] = Features.rollingmax(odfview[!, :high], trd.window)
            dfnew[!, :lowtarget] = Features.rollingmin(odfview[!, :low], trd.window)
            dfnew[!, :opentime] = odfview[!, :opentime]
        end
        
        trd.df = vcat(trd.df, dfnew)
        @assert size(trd.df, 1) == length(piv) "size(trd.df, 1)=$(size(trd.df, 1)) should match length(piv)=$(length(piv))"
    end
end

uniquelabels(trd::Bounds01) = ["lowtarget", "hightarget"] 

function timerangecut!(trd::Bounds01)
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - no time range to cut")
        return
    end
    # cut at start requires maxix correction, cut at end requires recalculation of last maxwindow elements
    startdt = Ohlcv.dataframe(trd.ohlcv)[begin, :opentime]
    startix = Ohlcv.rowix(trd.df[!, :opentime], startdt)
    enddt = Ohlcv.dataframe(trd.ohlcv)[end, :opentime]
    endix = Ohlcv.rowix(trd.df[!, :opentime], enddt)
    trd.df = trd.df[startix:endix, :]
end

"""
Returns a DataFrame with the columns hightarget and lowtarget indicating the expected price range targets.
If `relpricediff` is true, the targets are relative price differences to the corresponding pivot price, otherwise absolute price targets.
"""
function lowhigh(trd::Bounds01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd); relpricediff=true)::AbstractDataFrame 
    df1 = labelvalues(trd, firstix, lastix)
    if relpricediff
        df = DataFrame()
        piv = Ohlcv.dataframe(trd.ohlcv)[!, :pivot]
        df[!, :hightarget] = (df1[!, :hightarget] - piv[firstix:lastix]) ./ piv[firstix:lastix]
        df[!, :lowtarget] = (df1[!, :lowtarget] - piv[firstix:lastix]) ./ piv[firstix:lastix]
    else
        df = df1
    end
    return df
end

lowhigh(trd::Bounds01, startdt::DateTime, enddt::DateTime; relpricediff=true)::AbstractDataFrame = lowhigh(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt); relpricediff=relpricediff)

"""
Returns a DataFrame with the columns centertarget and widthtarget indicating the expected price range targets.
If `relpricediff` is true, the targets are relative price differences to the corresponding pivot price, otherwise absolute price targets.
"""
function centerwidth(trd::Bounds01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd); relpricediff=true)::AbstractDataFrame
    df_abs = lowhigh(trd, firstix, lastix; relpricediff=false)
    if relpricediff
        piv = Ohlcv.dataframe(trd.ohlcv)[firstix:lastix, :pivot]
        return lowhigh2centerwidth(df_abs[!, :lowtarget], df_abs[!, :hightarget], piv)
    else
        return lowhigh2centerwidth(df_abs[!, :lowtarget], df_abs[!, :hightarget])
    end
end
centerwidth(trd::Bounds01, startdt::DateTime, enddt::DateTime; relpricediff=true)::AbstractDataFrame = centerwidth(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt); relpricediff=relpricediff)

labelvalues(trd::Bounds01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractDataFrame = view(trd.df, firstix:lastix, [:lowtarget, :hightarget])
labelvalues(trd::Bounds01, startdt::DateTime, enddt::DateTime)::AbstractDataFrame = labelvalues(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

describe(trd::Bounds01) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_window=$(trd.window)"
firstrowix(trd::Bounds01)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Bounds01)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

function Base.show(io::IO, trd::Bounds01)
    println(io, "Bounds01 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base), window=$(trd.window) $(isnothing(trd.df) ? "no df" : "from $(trd.df[begin, :opentime]) to $(trd.df[end, :opentime]) ")")
    println(io, "Bounds01 ohlcv=$(trd.ohlcv)")
end

"""
Returns a DataFrame with columns :lowtarget and :hightarget based on the absolute center and absolute width values of the input vectors.
If base is provided, the center and width values are interpreted as relative price differences to the corresponding base price, otherwise absolute price targets.
"""
function centerwidth2lowhigh(center::AbstractVector{<:Real}, width::AbstractVector{<:Real}, base::Union{Nothing, AbstractVector{<:Real}}=nothing)::DataFrame
    @assert length(center) == length(width) "condition violated: length(center)=$(length(center)) should match length(width)=$(length(width))"
    @assert all(width .>= 0) "condition violated: all(width .>= 0) should hold but found width[1:10]=$(width[begin:begin+10]), length(width)=$(length(width))"
    @assert all(center .>= 0) "condition violated: all(center .>= 0) should hold but found center[1:10]=$(center[begin:begin+10]), length(center)=$(length(center))"
    if !isnothing(base)
        @assert length(center) == length(base) "condition violated: length(center)=$(length(center)) should match length(base)=$(length(base))"
        center = base .* (1 .+ center)
        width = base .* width
    end
    lowtarget = clamp.(center .- width ./ 2, 0f0, Inf32)
    hightarget = clamp.(center .+ width ./ 2, 0f0, Inf32)
    if !isnothing(base)
        lowtarget = (lowtarget .- base) ./ base
        hightarget = (hightarget .- base) ./ base
    end
    return DataFrame(lowtarget=lowtarget, hightarget=hightarget)
end

"""
Returns a DataFrame with columns :centertarget and :widthtarget based on the absolute lowtarget and absolute hightarget values of the input vectors"
If `relpricediff` is true, the targets are relative price differences to the corresponding pivot price, otherwise absolute price targets.
"""
function lowhigh2centerwidth(lowtarget::AbstractVector{<:Real}, hightarget::AbstractVector{<:Real}, base::Union{Nothing, AbstractVector{<:Real}}=nothing)::DataFrame
    @assert length(lowtarget) == length(hightarget) "condition violated: length(lowtarget)=$(length(lowtarget)) should match length(hightarget)=$(length(hightarget))"
    @assert all(0 .<= lowtarget .<= hightarget) "condition violated: all(lowtarget .<= hightarget) should hold but found lowtarget=$(lowtarget) and hightarget=$(hightarget)"
    center = (lowtarget .+ hightarget) ./ 2
    width = clamp.(hightarget .- lowtarget, 0f0, Inf32)
    if !isnothing(base)
        @assert length(center) == length(base) "condition violated: length(center)=$(length(center)) should match length(base)=$(length(base))"
        center = (center .- base) ./ base
        width = width ./ base
    end
    return DataFrame(centertarget=center, widthtarget=width)
end

#endregion Bounds01

end  # module

