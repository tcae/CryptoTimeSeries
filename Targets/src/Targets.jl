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

#region classifier functions
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
#endregion classifier functions

#region regressor functions
"""
Returns a dataframe with columns for each valuelabel and rows for each sample. The values are the target values for the regression task.
The column names are equal to uniquelabels()
"""
function labelvalues(targets::AbstractTargets, firstix::Integer, lastix::Integer)::AbstractDataFrame error("not implemented") end
function labelvalues(targets::AbstractTargets, startdt::DateTime, enddt::DateTime)::AbstractDataFrame error("not implemented") end
    
function crosscheck(trd::AbstractTargets)::Vector{String} return String[] end
function crosscheck(trd::AbstractTargets, labels::AbstractVector{<:TradeLabel}, pivots::AbstractVector{<:AbstractFloat})::Vector{String} return String[] end

#endregion regressor functions

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

#region Trend02

"""
Provides mutual exclusive targets as well as their relative gain
- minwindow is the minimum number of minutes a trend should have
- maxwindow is the maximum number of history minutes to detect a trend with given thresholds 
- required condition: 0 <= minwindow < maxwindow
- thres provides the hold and buy thresholds for long and short trends with required condition: thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy
- a sample sequence is labeled as longbuy/shortbuy trend range if:
  - it exceeds a price difference of the longbuy/shortbuy threshold
  - exceeds minwindow samples
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
mutable struct Trend02 <: AbstractTargets
    minwindow::Int # in minutes
    maxwindow::Int # in minutes
    thres::LabelThresholds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function Trend02(minwindow, maxwindow, thres)
        @assert 0 <= minwindow < maxwindow "condition violated: 0 <= minwindow=$(minwindow) < maxwindow=$(maxwindow)"
        @assert thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy "condition violated: thres.shortbuy=$(thres.shortbuy) <= thres.shorthold=$(thres.shorthold) <= thres.longhold=$(thres.longhold) <= fdg.thres.longbuy=$(thres.longbuy)"
        trd = new(minwindow, maxwindow, thres, nothing, nothing)
        return trd
    end
end

function setbase!(trd::Trend02, ohlcv::Ohlcv.OhlcvData)
    trd.ohlcv = ohlcv
    trd.df = DataFrame()
    supplement!(trd)
end

function removebase!(trd::Trend02)
    trd.ohlcv = nothing
    trd.df = nothing
end

"""
For sample at endix, find the best anchor in [startix, endix-1]:
- Best = maximum absolute price difference in the dominant direction
- Scan in reverse and stop at the nearest qualifying opposite trend
- Return allclose if resulting trend span is shorter than minwindow
- Returns: (label, anchorix, reldiff)
  - label: longbuy if pdiff >= longbuy_threshold, shortbuy if pdiff <= shortbuy_threshold, else allclose
  - anchorix: index of anchor sample (or endix if no trend found)
  - reldiff: (piv[endix] - piv[anchorix]) / piv[anchorix]
"""
function _findtrendanchor(trd::Trend02, piv, startix, endix, labels, _relix)
    currentprice = piv[endix]
    bestanchor = endix
    bestpdiff = 0f0

    for ix in (endix-1):-1:startix
        # Once a trend direction exists, stop at the nearest opposite tentative trend.
        # Important: this check uses forward-pass tentative labels only; it does not
        # verify opposite-span >= minwindow here.
        if bestanchor != endix
            opposite = bestpdiff > 0 ? (shortbuy, shorthold) : (longbuy, longhold)
            if labels[ix] in opposite
                break
            end
        end

        pdiff = (currentprice - piv[ix]) / piv[ix]

        # Track best positive trend (long)
        if pdiff > bestpdiff && pdiff >= trd.thres.longbuy
            bestanchor = ix
            bestpdiff = pdiff
        end
        # Track best negative trend (short) if more extreme than best positive
        if pdiff < bestpdiff && pdiff <= trd.thres.shortbuy
            bestanchor = ix
            bestpdiff = pdiff
        end
    end

    if bestanchor == endix
        return allclose, endix, 0f0
    end

    if (endix - bestanchor + 1) < trd.minwindow
        return allclose, endix, 0f0
    end

    label = bestpdiff > 0 ? longbuy : shortbuy
    return label, bestanchor, bestpdiff
end

function supplement!(trd::Trend02)
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(trd.ohlcv)
    piv = odf[!, :pivot]
    if length(piv) > 0
        if size(trd.df, 1) > 0
            @assert trd.df[begin, :opentime] == odf[begin, :opentime] "$(trd.df[begin, :opentime]) != $(odf[begin, :opentime])"
            startix = max(firstindex(piv), firstindex(piv) + size(trd.df, 1) - trd.maxwindow + 1)  # replace the last (trd.maxwindow -1) samples of trd.df
            # why replacing the last trd.maxwindow samples? because newer samples lead to conslusion of a trend that was not there before and short trend may not yet removed
        else
            startix = firstindex(piv)
        end
        
        pivnew = view(piv, startix:lastindex(piv))
        pvlen = length(pivnew)
        
        if pvlen > 0
            deltalen = length(piv) - pvlen
            
            # Forward pass: find best trend anchor for each sample
            relix = zeros(Int, pvlen)  # global index of trend anchor (in piv)
            lastextremeix = zeros(Int, pvlen)  # global index of last valid trend-range extreme
            reldiff = zeros(Float32, pvlen)  # relative difference to anchor
            labels = fill(allclose, pvlen)  # tentative labels

            # Seed overlap from existing analysis so continuing trends keep their original anchor.
            overlap = min(size(trd.df, 1), trd.maxwindow - 1)
            if overlap > 0
                labels[1:overlap] = trd.df[end-overlap+1:end, :label]
                relix[1:overlap] = trd.df[end-overlap+1:end, :relix]
                reldiff[1:overlap] = Float32.(trd.df[end-overlap+1:end, :reldiff])

                # Reconstruct last valid trend-range extremes for the seeded overlap.
                for ix in 1:overlap
                    currentglobal = deltalen + ix
                    if labels[ix] in [longbuy, longhold]
                        if (ix == 1) || !(labels[ix-1] in [longbuy, longhold])
                            lastextremeix[ix] = currentglobal
                        else
                            prevextreme = lastextremeix[ix-1]
                            lastextremeix[ix] = piv[currentglobal] >= piv[prevextreme] ? currentglobal : prevextreme
                        end
                    elseif labels[ix] in [shortbuy, shorthold]
                        if (ix == 1) || !(labels[ix-1] in [shortbuy, shorthold])
                            lastextremeix[ix] = currentglobal
                        else
                            prevextreme = lastextremeix[ix-1]
                            lastextremeix[ix] = piv[currentglobal] <= piv[prevextreme] ? currentglobal : prevextreme
                        end
                    else
                        lastextremeix[ix] = currentglobal
                    end
                end
            end
            
            # Forward pass with anchor-based continuation:
            # - Buy trends continue while buy threshold against anchor is met.
            # - Hold trends continue while hold threshold against anchor is met.
            # - If price crosses anchor in opposite direction, trend is re-established
            #   with current sample as new anchor (label allclose at this sample).
            for ix in (firstindex(pivnew) + overlap):lastindex(pivnew)
                if ix > firstindex(pivnew) && (labels[ix-1] in [longbuy, longhold, shortbuy, shorthold])
                    prevlabel = labels[ix-1]
                    prevanchor = relix[ix-1]
                    prevextreme = lastextremeix[ix-1]
                    currentglobal = deltalen + ix
                    if (firstindex(piv) <= prevanchor < currentglobal)
                        span = currentglobal - prevanchor + 1
                        extreldiff = (piv[currentglobal] - piv[prevanchor]) / piv[prevanchor]

                        if prevlabel in [longbuy, longhold]
                            # Lower price than anchor re-establishes long trend anchor.
                            if piv[currentglobal] <= piv[prevanchor]
                                labels[ix] = allclose
                                relix[ix] = currentglobal
                                lastextremeix[ix] = currentglobal
                                reldiff[ix] = 0f0
                                continue
                            end

                            reachednewextreme = piv[currentglobal] > piv[prevextreme]
                            lastextremeix[ix] = reachednewextreme ? currentglobal : prevextreme

                            if !reachednewextreme && ((currentglobal - prevextreme + 1) > trd.maxwindow)
                                labels[ix] = allclose
                                relix[ix] = currentglobal
                                lastextremeix[ix] = currentglobal
                                reldiff[ix] = 0f0
                                continue
                            elseif (span >= trd.minwindow) && (extreldiff >= trd.thres.longbuy)
                                labels[ix] = longbuy
                                relix[ix] = prevanchor
                                reldiff[ix] = extreldiff
                                continue
                            elseif extreldiff >= trd.thres.longhold
                                labels[ix] = longhold
                                relix[ix] = prevanchor
                                reldiff[ix] = extreldiff
                                continue
                            end
                        else
                            # Higher price than anchor re-establishes short trend anchor.
                            if piv[currentglobal] >= piv[prevanchor]
                                labels[ix] = allclose
                                relix[ix] = currentglobal
                                lastextremeix[ix] = currentglobal
                                reldiff[ix] = 0f0
                                continue
                            end

                            reachednewextreme = piv[currentglobal] < piv[prevextreme]
                            lastextremeix[ix] = reachednewextreme ? currentglobal : prevextreme

                            if !reachednewextreme && ((currentglobal - prevextreme + 1) > trd.maxwindow)
                                labels[ix] = allclose
                                relix[ix] = currentglobal
                                lastextremeix[ix] = currentglobal
                                reldiff[ix] = 0f0
                                continue
                            elseif (span >= trd.minwindow) && (extreldiff <= trd.thres.shortbuy)
                                labels[ix] = shortbuy
                                relix[ix] = prevanchor
                                reldiff[ix] = extreldiff
                                continue
                            elseif extreldiff <= trd.thres.shorthold
                                labels[ix] = shorthold
                                relix[ix] = prevanchor
                                reldiff[ix] = extreldiff
                                continue
                            end
                        end
                    end
                end

                # If continuing a trend, ignore samples before previous anchor.
                anchorbound = firstindex(pivnew)
                if ix > firstindex(pivnew) && (labels[ix-1] in [longbuy, longhold, shortbuy, shorthold])
                    anchorbound = max(anchorbound, relix[ix-1] - deltalen)
                end
                startback = max(firstindex(pivnew), ix - trd.maxwindow + 1, anchorbound)
                label, anchorix, relg = _findtrendanchor(trd, pivnew, startback, ix, labels, relix)
                labels[ix] = label
                relix[ix] = anchorix + deltalen
                lastextremeix[ix] = deltalen + ix
                reldiff[ix] = relg
            end
            
            if verbosity >= 3
                dfnew = DataFrame()
                dfnew[!, :tmprelix] = copy(relix)
                dfnew[!, :tmpreldiff] = copy(reldiff)
                dfnew[!, :tmplabel] = copy(labels)
            end
            
            # Backward pass: mark confirmed buy ranges and derive hold labels
            # from anchor-relative gain where buy threshold is not met.
            finalabels = copy(labels)
            prebackrelix = copy(relix)
            
            ix = lastindex(pivnew)
            while ix >= firstindex(pivnew)
                if labels[ix] in [longbuy, shortbuy]
                    currentlabel = labels[ix]
                    anchorglobal = relix[ix]
                    anchorix = anchorglobal - deltalen

                    if (firstindex(piv) <= anchorglobal < (deltalen + ix)) &&
                       (((deltalen + ix) - anchorglobal + 1) >= trd.minwindow)
                        startlocal = max(firstindex(pivnew), anchorix)
                        # If overlap recalculation starts in the middle of an already
                        # established trend for the same anchor, carry forward the
                        # buy-started state from the previous sample.
                        buystarted = false
                        buystartglobal = 0
                        runextreme = piv[anchorglobal]
                        segmin = piv[startlocal]
                        segmax = piv[startlocal]
                        if (startlocal > firstindex(pivnew)) && (relix[startlocal - 1] == anchorglobal)
                            if currentlabel == longbuy
                                buystarted = labels[startlocal - 1] in [longbuy, longhold]
                            else
                                buystarted = labels[startlocal - 1] in [shortbuy, shorthold]
                            end
                            buystartglobal = buystarted ? (deltalen + startlocal - 1) : 0
                        end
                        for labelix in startlocal:ix
                            currentglobal = deltalen + labelix

                            # Preserve explicit forward re-establish transition points
                            # (active trend on previous sample, then anchor reset to self).
                            isreestablish = (labelix > firstindex(pivnew)) &&
                                            (labels[labelix] == allclose) &&
                                            (labels[labelix - 1] in [longbuy, longhold, shortbuy, shorthold]) &&
                                            (prebackrelix[labelix] == currentglobal)
                            if isreestablish
                                finalabels[labelix] = allclose
                                relix[labelix] = currentglobal
                                reldiff[labelix] = 0f0
                                buystarted = false
                                buystartglobal = 0
                                continue
                            end

                            relix[labelix] = anchorglobal
                            span = currentglobal - anchorglobal + 1
                            rd = (piv[currentglobal] - piv[anchorglobal]) / piv[anchorglobal]
                            locallongrd = (piv[currentglobal] - segmin) / segmin
                            localshortrd = (piv[currentglobal] - segmax) / segmax
                            reldiff[labelix] = rd

                            if currentlabel == longbuy
                                isnewextreme = piv[currentglobal] >= runextreme
                                runextreme = max(runextreme, piv[currentglobal])
                                retrace = (labelix > startlocal) && (piv[currentglobal] < piv[currentglobal - 1])
                                prevlong = (labelix > startlocal) && (finalabels[labelix - 1] in [longbuy, longhold])
                                if (span >= trd.minwindow) && (rd >= trd.thres.longbuy) && (locallongrd >= trd.thres.longbuy)
                                    if retrace
                                        finalabels[labelix] = (buystarted && prevlong && (locallongrd >= trd.thres.longhold)) ? longhold : allclose
                                    elseif isnewextreme
                                        finalabels[labelix] = longbuy
                                        buystarted = true
                                        buystartglobal = buystartglobal == 0 ? currentglobal : buystartglobal
                                    elseif buystarted && prevlong && (locallongrd >= trd.thres.longhold)
                                        finalabels[labelix] = longhold
                                    else
                                        finalabels[labelix] = allclose
                                    end
                                elseif buystarted && (buystartglobal > 0) && (currentglobal > buystartglobal) && (locallongrd >= trd.thres.longhold)
                                    finalabels[labelix] = (!retrace && prevlong) ? longhold : allclose
                                else
                                    finalabels[labelix] = allclose
                                end
                            else
                                isnewextreme = piv[currentglobal] <= runextreme
                                runextreme = min(runextreme, piv[currentglobal])
                                retrace = (labelix > startlocal) && (piv[currentglobal] > piv[currentglobal - 1])
                                prevshort = (labelix > startlocal) && (finalabels[labelix - 1] in [shortbuy, shorthold])
                                if (span >= trd.minwindow) && (rd <= trd.thres.shortbuy) && (localshortrd <= trd.thres.shortbuy)
                                    if retrace
                                        finalabels[labelix] = (buystarted && prevshort && (localshortrd <= trd.thres.shorthold)) ? shorthold : allclose
                                    elseif isnewextreme
                                        finalabels[labelix] = shortbuy
                                        buystarted = true
                                        buystartglobal = buystartglobal == 0 ? currentglobal : buystartglobal
                                    elseif buystarted && prevshort && (localshortrd <= trd.thres.shorthold)
                                        finalabels[labelix] = shorthold
                                    else
                                        finalabels[labelix] = allclose
                                    end
                                elseif buystarted && (buystartglobal > 0) && (currentglobal > buystartglobal) && (localshortrd <= trd.thres.shorthold)
                                    finalabels[labelix] = (!retrace && prevshort) ? shorthold : allclose
                                else
                                    finalabels[labelix] = allclose
                                end
                            end

                            segmin = min(segmin, piv[currentglobal])
                            segmax = max(segmax, piv[currentglobal])
                        end
                    end

                    ix = anchorix - 1
                else
                    ix -= 1
                end
            end
            
            labels = finalabels
            
            if verbosity >= 3
                dfnew[!, :tmp2label] = copy(labels)
            end
            
            # Assemble output dataframe
            dfnew = DataFrame()
            dfnew[!, :relix] = relix
            dfnew[!, :reldiff] = reldiff
            dfnew[!, :label] = labels
            dfnew[!, :opentime] = odf[startix:lastindex(piv), :opentime]
            
            @assert all(in.(labels, Ref([longbuy, longhold, shortbuy, shorthold, allclose]))) "unique(labels)=$(unique(labels))"
        end
        
        trd.df = deltalen > 0 ? vcat(trd.df[begin:startix-1, :], dfnew) : dfnew
    end
end

uniquelabels(trd::Trend02) = [longbuy, longhold, shortbuy, shorthold, allclose]

function timerangecut!(trd::Trend02)
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

describe(trd::Trend02) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_maxwindow=$(trd.maxwindow)_minwindow=$(trd.minwindow)_thresholds=(longbuy=$(trd.thres.longbuy)_longhold=$(trd.thres.longhold)_shorthold=$(trd.thres.shorthold)_shortbuy=$(trd.thres.shortbuy))"
firstrowix(trd::Trend02)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Trend02)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

# df(trd::Trend02, startdt::DateTime, enddt::DateTime) = df(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))
# longbuybinarytargets(trd::Trend02, startdt::DateTime, enddt::DateTime) = [lb ? "longbuy" : "longclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]
# shortbuybinarytargets(trd::Trend02, startdt::DateTime, enddt::DateTime) = [lb ? "shortbuy" : "shortclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]

labelbinarytargets(trd::Trend02, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labels(trd, firstix, lastix) .== label
labelbinarytargets(trd::Trend02, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelbinarytargets(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labelrelativegain(trd::Trend02, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labelbinarytargets(trd, label, firstix, lastix) .* relativegain(trd, firstix, lastix)
labelrelativegain(trd::Trend02, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelrelativegain(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labels(trd::Trend02, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :label)
labels(trd::Trend02, startdt::DateTime, enddt::DateTime) = labels(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

relativegain(trd::Trend02, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :reldiff)
relativegain(trd::Trend02, startdt::DateTime, enddt::DateTime) = relativegain(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

function Base.show(io::IO, trd::Trend02)
    println(io, "Trend02 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base) maxwindow=$(trd.maxwindow) label thresholds=$(thresholds(trd.thres)) $(isnothing(trd.df) ? "no df" : "from $(trd.df[begin, :opentime]) to $(trd.df[end, :opentime]) ")")
    println(io, "Trend02 ohlcv=$(trd.ohlcv)")
end

"""
Crosschecks a `Trend02` label vector against the Trend02 constraints and returns
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
function crosscheck(trd::Trend02, labels::AbstractVector{<:TradeLabel}, pivots::AbstractVector{<:AbstractFloat})::Vector{String}
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

    _direction(lbl::TradeLabel) = lbl in longlabels ? :long : (lbl in shortlabels ? :short : :close)

    for ix in eachindex(labels)
        if !(labels[ix] in allowed)
            push!(issues, "invalid label at ix=$(ix): label=$(labels[ix]) allowed=$(collect(allowed))")
        end
    end

    # Build contiguous directional ranges: long/short/allclose
    segments = Tuple{Int, Int, Symbol}[]
    segstart = firstindex(labels)
    segdir = _direction(labels[segstart])
    for ix in (firstindex(labels)+1):lastindex(labels)
        d = _direction(labels[ix])
        if d != segdir
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
        s, e, lbl = labelsegments[segix]
        if lbl == longhold
            if (segix == firstindex(labelsegments)) || (labelsegments[segix - 1][3] != longbuy)
                push!(issues, "long hold segment at $(s):$(e) must be preceded by a longbuy segment")
            end
        elseif lbl == shorthold
            if (segix == firstindex(labelsegments)) || (labelsegments[segix - 1][3] != shortbuy)
                push!(issues, "short hold segment at $(s):$(e) must be preceded by a shortbuy segment")
            end
        end
    end

    # Segment-level threshold and span checks using extrema-consistent
    # relative gain/loss within each segment (not only start/end).
    for (s, e, lbl) in labelsegments
        if lbl == allclose
            continue
        end

        span = e - s + 1
        bestlong = -Inf32
        bestshort = Inf32
        if span >= 2
            minp = Float32(pivots[s])
            maxp = Float32(pivots[s])
            for ix in (s + 1):e
                p = Float32(pivots[ix])
                longrd = (p - minp) / minp
                shortrd = (p - maxp) / maxp
                bestlong = max(bestlong, longrd)
                bestshort = min(bestshort, shortrd)
                minp = min(minp, p)
                maxp = max(maxp, p)
            end
        end

        if lbl == longbuy
            if span < trd.minwindow
                push!(issues, "longbuy segment $(s):$(e) violates minwindow: span=$(span) < minwindow=$(trd.minwindow)")
            end
            if bestlong < trd.thres.longbuy
                push!(issues, "longbuy segment $(s):$(e) violates threshold: bestrd=$(bestlong) < longbuy=$(trd.thres.longbuy)")
            end
        elseif lbl == shortbuy
            if span < trd.minwindow
                push!(issues, "shortbuy segment $(s):$(e) violates minwindow: span=$(span) < minwindow=$(trd.minwindow)")
            end
            if bestshort > trd.thres.shortbuy
                push!(issues, "shortbuy segment $(s):$(e) violates threshold: bestrd=$(bestshort) > shortbuy=$(trd.thres.shortbuy)")
            end
        elseif lbl == longhold
            if bestlong < trd.thres.longhold
                push!(issues, "longhold segment $(s):$(e) violates hold threshold: bestrd=$(bestlong) < longhold=$(trd.thres.longhold)")
            end
            if bestlong >= trd.thres.longbuy
                push!(issues, "longhold segment $(s):$(e) violates buy exclusion: bestrd=$(bestlong) >= longbuy=$(trd.thres.longbuy)")
            end
        elseif lbl == shorthold
            if bestshort > trd.thres.shorthold
                push!(issues, "shorthold segment $(s):$(e) violates hold threshold: bestrd=$(bestshort) > shorthold=$(trd.thres.shorthold)")
            end
            if bestshort <= trd.thres.shortbuy
                push!(issues, "shorthold segment $(s):$(e) violates buy exclusion: bestrd=$(bestshort) <= shortbuy=$(trd.thres.shortbuy)")
            end
        end
    end

    # Range continuation must break when no new directional extreme is reached within maxwindow.
    for (s, e, d) in segments
        if d == :long
            if pivots[s] != minimum(view(pivots, s:e))
                push!(issues, "long range $(s):$(e) must start at a segment low extreme: start value=$(pivots[s]), segment min=$(minimum(view(pivots, s:e)))")
            end

            # Every full maxwindow slice inside a long range must contain a longbuy
            # confirmation relative to its slice start.
            if (e - s + 1) >= trd.maxwindow
                for wstart in s:(e - trd.maxwindow + 1)
                    wend = wstart + trd.maxwindow - 1
                    hasbuy = any(((pivots[ix] - pivots[wstart]) / pivots[wstart]) >= trd.thres.longbuy for ix in (wstart+1):wend)
                    if !hasbuy
                        push!(issues, "long range $(s):$(e) lacks longbuy confirmation within maxwindow=$(trd.maxwindow) for window $(wstart):$(wend)")
                        break
                    end
                end
            end

            lastrecordix = s
            recordval = pivots[s]
            for ix in s:e
                if pivots[ix] > recordval
                    recordval = pivots[ix]
                    lastrecordix = ix
                end
                if (ix - lastrecordix) >= trd.maxwindow
                    push!(issues, "long range $(s):$(e) violates maxwindow continuation at ix=$(ix): last record high at ix=$(lastrecordix), maxwindow=$(trd.maxwindow)")
                    break
                end
            end
            if lastrecordix != e
                push!(issues, "long range $(s):$(e) must end at a segment high extreme: last record high at ix=$(lastrecordix), range end=$(e)")
            end
        elseif d == :short
            if pivots[s] != maximum(view(pivots, s:e))
                push!(issues, "short range $(s):$(e) must start at a segment high extreme: start value=$(pivots[s]), segment max=$(maximum(view(pivots, s:e)))")
            end

            # Every full maxwindow slice inside a short range must contain a shortbuy
            # confirmation relative to its slice start.
            if (e - s + 1) >= trd.maxwindow
                for wstart in s:(e - trd.maxwindow + 1)
                    wend = wstart + trd.maxwindow - 1
                    hasbuy = any(((pivots[ix] - pivots[wstart]) / pivots[wstart]) <= trd.thres.shortbuy for ix in (wstart+1):wend)
                    if !hasbuy
                        push!(issues, "short range $(s):$(e) lacks shortbuy confirmation within maxwindow=$(trd.maxwindow) for window $(wstart):$(wend)")
                        break
                    end
                end
            end

            lastrecordix = s
            recordval = pivots[s]
            for ix in s:e
                if pivots[ix] < recordval
                    recordval = pivots[ix]
                    lastrecordix = ix
                end
                if (ix - lastrecordix) >= trd.maxwindow
                    push!(issues, "short range $(s):$(e) violates maxwindow continuation at ix=$(ix): last record low at ix=$(lastrecordix), maxwindow=$(trd.maxwindow)")
                    break
                end
            end
            if lastrecordix != e
                push!(issues, "short range $(s):$(e) must end at a segment low extreme: last record low at ix=$(lastrecordix), range end=$(e)")
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
function crosscheck(trd::Trend02)::Vector{String}
    issues = String[]
    if isnothing(trd.ohlcv)
        push!(issues, "missing ohlcv in Trend02: setbase! must be called before crosscheck(trd)")
    end
    if isnothing(trd.df)
        push!(issues, "missing df in Trend02: setbase! or supplement! must populate labels before crosscheck(trd)")
    end
    if !isempty(issues)
        return issues
    end

    pivots = Ohlcv.dataframe(trd.ohlcv)[!, :pivot]
    labels = trd.df[!, :label]
    return crosscheck(trd, labels, pivots)
end
#endregion Trend02

#region Trend04

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
                # Check cumulative gain from original entry anchor: if still above hold threshold,
                # the position remains profitable and we stay in hold rather than reversing.
                anchorix = (labels[ix] == longbuy) ? relix[ix] : relix[relix[ix]] # single or double-deref to bestlongix
                holdrelix = (labels[ix] == longbuy) ? ix : relix[ix] # ixbuy: last buy bar, preserved by _fillsegment! for hold
                # Hold continuation should be measured from the last buy extreme once we are
                # already in hold mode, otherwise the original entry anchor can keep a stale
                # hold active for too long and suppress natural direction changes.
                holdcheckix = holdrelix
                if _reldiff(piv, holdcheckix, endix) >= trd.thres.longhold
                    labels[endix] = longhold
                    relix[endix] = holdrelix
                    _fillsegment!(labels, relix, endix)
                else
                    # Anchor reversal at local high so a short range starts at its segment extreme.
                    searchrange = (ix + 1):endix
                    _, maxrelix = findmax(view(piv, searchrange))
                    newix = first(searchrange) + maxrelix - 1
                    if ((endix - newix + 1) >= trd.minwindow) && (_reldiff(piv, newix, endix) <= trd.thres.shortbuy) && (piv[endix] == minimum(view(piv, newix:endix)))
                        labels[endix] = shortbuy
                        relix[endix] = newix 
                        _fillsegment!(labels, relix, endix)
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
                    holdcheckix = relix[ix]
                    # Compare vs last buy bar (the peak), not just the previous hold bar.
                    # This prevents small bounces within the declining hold phase from
                    # triggering a new buy sub-segment spuriously.
                    # Use strict > to avoid equal-amplitude periodic signals creating spurious buys.
                    isnewhigh = piv[endix] > piv[ixbuy]
                    if isnewhigh && islocalstartlow && (localbuyspan >= trd.minwindow) && (_reldiff(piv, anchorix, endix) >= trd.thres.longbuy) && (localbuyreldiff >= trd.thres.longbuy)
                        labels[endix] = longbuy
                        relix[endix] = ix # buy sub-segment starts after last hold bar, preserving earlier hold bars
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, holdcheckix, endix) >= trd.thres.longhold) # use entry anchor only while it is not stale; otherwise use latest buy extreme
                        labels[endix] = longhold
                        relix[endix] = relix[ix] # extend hold segment by anchor take over because there is no sell buy transaction for the hold phase extension
                        _fillsegment!(labels, relix, endix)
                    else
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix
                        _fillsegment!(labels, relix, endix)
                    end
                else # labels[ix] == longbuy
                    anchorix = relix[ix]  # relix[lastbuybar] = bestlongix (original entry anchor)
                    isnewhigh = piv[endix] >= piv[ix]  # only needs to exceed last directional extreme, not intermediate allclose bars
                    isanchormin = piv[anchorix] == minimum(view(piv, anchorix:endix))
                    if isnewhigh && isanchormin && (_reldiff(piv, anchorix, endix) >= trd.thres.longbuy) # minwindow condition is ensured in continuation case
                        labels[endix] = longbuy 
                        relix[endix] = relix[ix] # extend segment; there is no sell-buy transaction for the buy phase extension
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, ix, endix) >= trd.thres.longhold)
                        labels[endix] = longhold
                        relix[endix] = ix # the hold phase starts after the previous buy sample; _fillsegment! skips ix itself (last buy bar)
                        _fillsegment!(labels, relix, endix)
                    else # price not above hold threshold from entry anchor, so no long trend anymore
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
                # Check cumulative gain from original entry anchor: if still above hold threshold,
                # the position remains profitable and we stay in hold rather than reversing.
                anchorix = (labels[ix] == shortbuy) ? relix[ix] : relix[relix[ix]] # single or double-deref to bestshortix
                holdrelix = (labels[ix] == shortbuy) ? ix : relix[ix] # ixshortbuy: last shortbuy bar, preserved by _fillsegment! for hold
                # Hold continuation should be measured from the last shortbuy extreme once we
                # are already in hold mode, otherwise the original entry anchor can keep a
                # stale hold active for too long and suppress natural direction changes.
                holdcheckix = holdrelix
                if _reldiff(piv, holdcheckix, endix) <= trd.thres.shorthold
                    labels[endix] = shorthold
                    relix[endix] = holdrelix
                    _fillsegment!(labels, relix, endix)
                else
                    # Anchor reversal at local low so a long range starts at its segment extreme.
                    searchrange = (ix + 1):endix
                    _, minrelix = findmin(view(piv, searchrange))
                    newix = first(searchrange) + minrelix - 1
                if ((endix - newix + 1) >= trd.minwindow) && (_reldiff(piv, newix, endix) >= trd.thres.longbuy) && (piv[endix] == maximum(view(piv, newix:endix)))
                    labels[endix] = longbuy
                    relix[endix] = newix 
                    _fillsegment!(labels, relix, endix)
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
                    holdcheckix = relix[ix]
                    # Compare vs last shortbuy bar (the trough), not just the previous hold bar.
                    # This prevents small reversals within the recovering hold phase from
                    # triggering a new shortbuy sub-segment spuriously.
                    # Use strict < to avoid equal-amplitude periodic signals creating spurious buys.
                    isnewlow = piv[endix] < piv[ixshortbuy]
                    if isnewlow && islocalstarthigh && (localbuyspan >= trd.minwindow) && (_reldiff(piv, anchorix, endix) <= trd.thres.shortbuy) && (localbuyreldiff <= trd.thres.shortbuy)
                        labels[endix] = shortbuy
                        relix[endix] = ix # buy sub-segment starts after last hold bar, preserving earlier short hold bars
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, holdcheckix, endix) <= trd.thres.shorthold) # use entry anchor only while it is not stale; otherwise use latest shortbuy extreme
                        labels[endix] = shorthold
                        relix[endix] = relix[ix] # extend hold segment by anchor take over because there is no sell buy transaction for the hold phase extension
                        _fillsegment!(labels, relix, endix)
                    else
                        newix = ix + 1 # new segment starts one minute later after the end point of the previous segment
                        labels[endix] = allclose
                        relix[endix] = newix
                        _fillsegment!(labels, relix, endix)
                    end
                else # labels[ix] == shortbuy
                    anchorix = relix[ix]  # relix[lastshortbuybar] = bestshortix (original entry anchor)
                    isnewlow = piv[endix] <= piv[ix]  # only needs to exceed last directional extreme, not intermediate allclose bars
                    isanchormax = piv[anchorix] == maximum(view(piv, anchorix:endix))
                    if isnewlow && isanchormax && (_reldiff(piv, anchorix, endix) <= trd.thres.shortbuy) # minwindow condition is ensured in continuation case
                        labels[endix] = shortbuy
                        relix[endix] = relix[ix] # extend segment; there is no sell-buy transaction for the buy phase extension
                        _fillsegment!(labels, relix, endix)
                    elseif (_reldiff(piv, ix, endix) <= trd.thres.shorthold)
                        labels[endix] = shorthold
                        relix[endix] = ix # the hold phase starts after the previous shortbuy sample; _fillsegment! skips ix itself (last shortbuy bar)
                        _fillsegment!(labels, relix, endix)
                    else # price not below hold threshold from entry anchor, so no short trend anymore
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
            println()
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
            # Hold can appear during price decline, so measuring from the last buy bar gives a
            # negative reldiff; measuring from entry correctly reflects retained position value.
            hold_anchor = (!isnothing(relix_arr)) ? relix_arr[relix_arr[lss]] : lss - 1
            reldiff_hold = _reldiff(pivots, hold_anchor, lse)
            # Hold = "position is still profitable at hold level: gain from entry >= hold threshold".
            # No buy exclusion check: hold is valid even when cumulative gain >= buy threshold,
            # because during the declining phase price may still be above buy level from entry.
            if reldiff_hold < trd.thres.longhold
                push!(issues, "longhold segment $(lss):$(lse) violates hold threshold: reldiff=$(reldiff_hold) < longhold=$(trd.thres.longhold)")
            end
        elseif lbl == shorthold
            # Symmetric to longhold: use cumulative gain from original short entry anchor.
            hold_anchor = (!isnothing(relix_arr)) ? relix_arr[relix_arr[lss]] : lss - 1
            reldiff_hold = _reldiff(pivots, hold_anchor, lse)
            # Hold = "position is still profitable at hold level: gain from entry <= shorthold threshold".
            # No buy exclusion check: hold is valid even when cumulative gain <= shortbuy threshold.
            if reldiff_hold > trd.thres.shorthold
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

#region Trend01

"""
Provides 2 independent binary targets as wellas their relative gain
- minwindow is the minimum number of minutes a trend should have
- maxwindow is the maximum number of history minutes to detect a trend with given thresholds 
- required condition: 0 <= minwindow < maxwindow <= 4*60
- longbuy label for all samples exceeding `thres.longhold` if subsequently threshold `thres.longbuy` is exceeded within the next `maxwindow` minutes
- shortbuy label for all samples undercutting `thres.shorthold` if subsequently threshold `thres.shortbuy` is undercut within the next `maxwindow` minutes
"""
mutable struct Trend01 <: AbstractTargets
    minwindow::Int # in minutes
    maxwindow::Int # in minutes
    thres::LabelThresholds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function Trend01(minwindow, maxwindow, thres)
        @assert 0 <= minwindow < maxwindow <= 4*60 "condition violated: 0 <= minwindow=$(minwindow) < maxwindow=$(maxwindow) <= 4*60"
        @assert thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy "condition violated: thres.shortbuy=$(thres.shortbuy) <= thres.shorthold=$(thres.shorthold) <= thres.longhold=$(thres.longhold) <= fdg.thres.longbuy=$(thres.longbuy)"
        trd = new(minwindow, maxwindow, thres, nothing, nothing)
        return trd
    end
end

function setbase!(trd::Trend01, ohlcv::Ohlcv.OhlcvData)
    trd.ohlcv = ohlcv
    trd.df = DataFrame()
    supplement!(trd)
end

function removebase!(trd::Trend01)
    trd.ohlcv = nothing
    trd.df = nothing
end

function _removeshorttrends!(trd::Trend01, labels)
    startix = firstindex(labels)
    lblbefore = lbl = labels[startix]
    for ix in eachindex(labels)
        if lbl != labels[ix]
            if (lbl in [shortbuy, longbuy]) && ((ix - startix) < trd.minwindow)
                # too short trend detected -> to be removed
                if (verbosity >= 4) 
                    @warn "$lbl short trend $startix:$(ix-1) lblbefore=$lblbefore labels[$ix]=$(labels[ix])"
                end
                if lblbefore == labels[ix]
                    for repairix in startix:(ix-1)
                        labels[repairix] = lblbefore
                    end
                else
                    for repairix in startix:(ix-1)
                        labels[repairix] = allclose
                    end
                end
            end
            lblbefore = lbl
            lbl = labels[ix]
            startix = ix
        end
    end
end

"""
- trend has to be steep enough to be confirmed
- initial state is allclose
  - as long as hold threshold is no exceeded the resix anchor and the label is not changed but rd is updated -> if isnothing(resix) case
- hold threshold exceeded causes label, resix, rd to be updated -> label== hold case
  - subsequent backtraces are limited to resix of hold asnew anchor and rd has to built up again
  - rd may built up to exceed buy threshold in a one minute jump then a buy label will be applied but only until the next hold label anchor, i.e. we will hardly see a buy label if the threshold of hold and buy is differently
  - rd stays above hold threshold but below buy threshold => will be overwritten in backpropagation by allclose
  - the current implementation works only for hold = buy, which prevents another rd step to take and enables buy sequences
- exceeding buy threshold causes label, resix, rd to be updated -> label== buy case
  - the buy label is honored in the backpropagation
  - not exceeding hold/buy => hold on to anchor
  - if not exceededing hold/buy is longer than maxwindow then allclose will be applied in the backpropagation
  - once a buy label is set and thereby a new anchor established, a new trend need to be built up - it will not continue an already established trend due to the backtrace limitation to the new anchor

Consequences:
- trends are shorter than intended and missing the newer part that does not exceed again teh required buy threshold
- allclose establishes only 
  - where flat parts exceed maxwindow
  - where an established trend continues but does not exceed again the buy threshold
"""
function _trendinrange(trd::Trend01, piv, startix, endix, lastlabel)
    label = resix = reldiff = nothing
    currentprice = piv[endix]
    for ix in endix-1:-1:startix
        rd = (currentprice - piv[ix]) / piv[ix]
        if isnothing(resix)
            if rd >= trd.thres.longhold
                resix = ix
                reldiff = rd
                label = longhold
            elseif rd <= trd.thres.shorthold
                resix = ix
                reldiff = rd
                label = shorthold
            end
        elseif rd > reldiff >= trd.thres.longhold # no check of label but reldiff guards long trend => reduces allclose candidates
            resix = ix #! this causes to limit the backtrace analysis to resix although it hasonly reached hold level
            reldiff = rd
        elseif rd < reldiff <= trd.thres.shorthold # no check of label but reldiff guards short trend => reduces allclose candidates
            resix = ix #! this causes to limit the backtrace analysis to resix although it hasonly reached hold level
            reldiff = rd
        end
    end
    if isnothing(resix)
        resix = startix # at least backtrace is not further limited, i.e. allclose may be overwritten with later insight
        reldiff = (currentprice - piv[startix]) / piv[startix]
        label = lastlabel #* if flat the last label will be prolonged - even if it is a longbuy and price goes down but above shorthold it would receive a longbuy
        #*! if after a long trend the longbuy is prolonged but later teh price falls below shorthold the trend would be limited because it only looks until the last buy
        #* if hold threshold is not exceed it just holds on to the anchor with relx and label, thereby giving more samples to compile a threshold diff
    elseif label == longhold
        #TODO a check can be added testing the length of the trend and only marking the label a buy if the minwindow is met or exceeded
        label = reldiff >= trd.thres.longbuy ? longbuy : longhold
    elseif label == shorthold
        #TODO a check can be added testing the length of the trend and only marking the label a buy if the minwindow is met or exceeded
        label = reldiff <= trd.thres.shortbuy ? shortbuy : shorthold
    end
    return label, resix, reldiff
end

"""
because prices can be very volatile no assumption on continous price development can be done and trend need to be checked for each maxwindow move

- trend trigger index to be identified -> breaks opposite trend continue to check whether and where it establishes
- trend established to be identified
- algo:
  - reverse loop starting from current ix until 
    - trend trigger (hold threshold) is identified or 
    - trend confirmation of previous trigger (buy threshold) or 
    - start of maxwindow reached or 
    - the trend index of the previous opposite trend is reached
  - note 1) trend trigger index or 2) trend confirmation index (only the one that was previously triggered) or 3) start maxwindow index if 1) and 2) don't apply
  - startix denotes the start of the maxwindow or the trend index of the predecessor opposite trend (whichever is larger)
  - triggerix denotes the index of the first trend trigger
  - triggeredtrend identifies whether a short or long trend is triggered
  - confirmix denotes the trend confirmation index

  what is a trend and when is it broken?
  - establish by exceeding buy threshold against an achor reference, don't deviate from most extreme point of trend by more than hold thrshold
"""
function supplement!(trd::Trend01)
    #TODO TrendDetector001.AdaTest() showed only targets of longbuy and shortbuy and no sample with target allclose, which should be checked
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(trd.ohlcv)
    piv = odf[!, :pivot]
    if length(piv) > 0
        if size(trd.df, 1) > 0
            @assert trd.df[begin, :opentime] == odf[begin, :opentime] "$(trd.df[begin, :opentime]) != $(odf[begin, :opentime])"
            startix = max(firstindex(piv), firstindex(piv) + size(trd.df, 1) - trd.maxwindow + 1)  # replace the last (trd.maxwindow -1) samples of trd.df
            #* why replacing the last trd.maxwindow samples? because newer samples lead to conslusion of a trend that was not there before and short trend may not yet removed
            #TODO in order to refresh also trd.maxwindow history, 2 * trd.maxwindow samples need to be inspected that should be part of pivnew
        else
            startix = firstindex(piv)
        end
        # len(piv) = 10, len(trd)=5, maxwindow = 3 --> startix = 1+5-3+1=4
        # recalc the last (maxwindow-1) rows due to new ohlcv max in last maxwindow element
        pivnew = view( piv, startix:lastindex(piv))
        pvlen = length(pivnew)
        if pvlen > 0
            deltalen = length(piv) - pvlen
            dfnew = DataFrame()
            relix = zeros(UInt32, pvlen)
            reldiff = zeros(Float32, pvlen)
            labels = fill(allclose, pvlen)
            lastix = firstindex(pivnew)
            relix[lastix] = firstindex(pivnew)
            for ix in eachindex(pivnew)
                windowstartix = max(ix - trd.maxwindow + 1, relix[lastix]) # _trendinrange can only look back until the next buy label
                labels[ix], relix[ix], reldiff[ix] = _trendinrange(trd, pivnew, windowstartix, ix, labels[lastix])
                lastix = ix
            end
            if verbosity >= 3
                dfnew[!, :tmprelix] = copy(relix) .+ deltalen
                dfnew[!, :tmpreldiff] = copy(reldiff)
                dfnew[!, :tmplabel] = copy(labels)
            end
            trdix = lastindex(pivnew)
            trdlabel = allclose
            for ix in reverse(eachindex(pivnew))  # now propagade [longbuy, shortbuy] across their trend
                if (ix > trdix) && (labels[ix] != trdlabel)
                    relix[ix] = trdix
                    reldiff[ix] = (pivnew[ix] - pivnew[trdix]) / pivnew[trdix]
                    labels[ix] = trdlabel   
                elseif labels[ix] in [longbuy, shortbuy]
                    trdix = relix[ix]
                    trdlabel = labels[ix] 
                else
                    labels[ix] = allclose # also overwriting shorthold, longhold
                    trdix = relix[ix]
                    trdlabel = labels[ix] 
                end
            end
            if verbosity >= 3
                dfnew[!, :tmp2label] = copy(labels)
            end
            _removeshorttrends!(trd, labels)
            dfnew[!, :relix] = relix .+ deltalen
            dfnew[!, :reldiff] = reldiff
            dfnew[!, :label] = labels
            dfnew[!, :opentime] = odf[startix:lastindex(piv), :opentime]
        end
        trd.df = deltalen > 0 ? vcat(trd.df[begin:startix-1, :], dfnew) : dfnew
    end
end

uniquelabels(trd::Trend01) = [longbuy, longhold, shortbuy, shorthold, allclose]

function timerangecut!(trd::Trend01)
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

describe(trd::Trend01) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_maxwindow=$(trd.maxwindow)_minwindow=$(trd.minwindow)_thresholds=(longbuy=$(trd.thres.longbuy)_longhold=$(trd.thres.longhold)_shorthold=$(trd.thres.shorthold)_shortbuy=$(trd.thres.shortbuy))"
firstrowix(trd::Trend01)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Trend01)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

# df(trd::Trend01, startdt::DateTime, enddt::DateTime) = df(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))
# longbuybinarytargets(trd::Trend01, startdt::DateTime, enddt::DateTime) = [lb ? "longbuy" : "longclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]
# shortbuybinarytargets(trd::Trend01, startdt::DateTime, enddt::DateTime) = [lb ? "shortbuy" : "shortclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]

labelbinarytargets(trd::Trend01, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labels(trd, firstix, lastix) .== label
labelbinarytargets(trd::Trend01, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelbinarytargets(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labelrelativegain(trd::Trend01, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labelbinarytargets(trd, label, firstix, lastix) .* relativegain(trd, firstix, lastix)
labelrelativegain(trd::Trend01, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelrelativegain(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labels(trd::Trend01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :label)
labels(trd::Trend01, startdt::DateTime, enddt::DateTime) = labels(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

relativegain(trd::Trend01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :reldiff)
relativegain(trd::Trend01, startdt::DateTime, enddt::DateTime) = relativegain(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

function Base.show(io::IO, trd::Trend01)
    println(io, "Trend01 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base) maxwindow=$(trd.maxwindow) label thresholds=$(thresholds(trd.thres)) $(isnothing(trd.df) ? "no df" : size(trd.df, 1) > 0 ? "from $(isnothing(trd.df) ? "no df" : trd.df[begin, :opentime]) to $(isnothing(trd.df) ? "no df" : trd.df[end, :opentime]) " : "no time range ")")
    # (verbosity >= 3) && println(io, "Features005 cfgdf=$(f5.cfgdf)")
    # (verbosity >= 2) && println(io, "Features005 config=$(f5.cfgdf[!, :config])")
    println(io, "Trend01 ohlcv=$(trd.ohlcv)")
end
#endregion Trend01

#region Bounds01

"""
Provides an highbound and lowerbound estimation within the coming `window` samples.  
If `relpricediff` is true, the bounds are relative price differences to the current pivot price, otherwise absolute price bounds.
"""
mutable struct Bounds01 <: AbstractTargets
    window::Int # in minutes
    relpricediff::Bool # if true, label thresholds are relative price differences, otherwise absolute price bounds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function Bounds01(window; relpricediff::Bool=true)
        @assert 0 < window  "condition violated: 0 < window=$(window) "
        trd = new(window, relpricediff, nothing, nothing)
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
            maxb = Features.rollingmax(odfview[!, :high], trd.window)
            minb = Features.rollingmin(odfview[!, :low], trd.window)
            if trd.relpricediff
                maxb = (maxb - piv) ./ piv
                minb = (minb - piv) ./ piv
            end

            # Assemble output dataframe
            dfnew = DataFrame()
            dfnew[!, :highbound] = maxb
            dfnew[!, :lowbound] = minb
            dfnew[!, :opentime] = odfview[!, :opentime]
        end
        
        trd.df = size(trd.df, 1) > 0 ? vcat(trd.df, dfnew) : dfnew
        @assert size(trd.df, 1) == length(piv) "size(trd.df, 1)=$(size(trd.df, 1)) should match length(piv)=$(length(piv))"
    end
end

uniquelabels(trd::Bounds01) = ["lowbound", "highbound"] 

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

lowboundhighbound(trd::Bounds01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractDataFrame = labelvalues(trd, firstix, lastix)
lowboundhighbound(trd::Bounds01, startdt::DateTime, enddt::DateTime)::AbstractDataFrame = labelvalues(trd, startdt, enddt)

labelvalues(trd::Bounds01, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractDataFrame = view(trd.df, firstix:lastix, [:lowbound, :highbound])
labelvalues(trd::Bounds01, startdt::DateTime, enddt::DateTime)::AbstractDataFrame = labelvalues(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

describe(trd::Bounds01) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_window=$(trd.window)_relpricediff=$(trd.relpricediff)"
firstrowix(trd::Bounds01)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Bounds01)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

function Base.show(io::IO, trd::Bounds01)
    println(io, "Bounds01 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base), relpricediff=$(trd.relpricediff), window=$(trd.window) $(isnothing(trd.df) ? "no df" : "from $(trd.df[begin, :opentime]) to $(trd.df[end, :opentime]) ")")
    println(io, "Bounds01 ohlcv=$(trd.ohlcv)")
end

"returns a DataFrame with columns :lowbound and :highbound based on the center and width values of the input vectors"
function centerwidth2lowhigh(center::AbstractVector{<:Real}, width::AbstractVector{<:Real})
    @assert length(center) == length(width) "condition violated: length(center)=$(length(center)) should match length(width)=$(length(width))"
    lowbound = clamp.(center .- width ./ 2, 0f0, Inf32)
    highbound = clamp.(center .+ width ./ 2, 0f0, Inf32)
    return DataFrame(lowbound=lowbound, highbound=highbound)
end

"returns a DataFrame with columns :center and :width based on the lowbound and highbound values of the input vectors"
function lowhigh2centerwidth(lowbound::AbstractVector{<:Real}, highbound::AbstractVector{<:Real})::AbstractDataFrame
    @assert length(lowbound) == length(highbound) "condition violated: length(lowbound)=$(length(lowbound)) should match length(highbound)=$(length(highbound))"
    center = (lowbound .+ highbound) ./ 2
    width = clamp.(highbound .- lowbound, 0f0, Inf32)
    return DataFrame(center=centertarget, width=widthtarget)
end

#endregion Bounds01

#region Trend03

"""
Provides the following mutual exclusive targets as well as their relative gain:
- maxwindow is the maximum number of history minutes to detect a trend with given thresholds 
- required condition: 0 <= maxwindow <= 4*60
- longbuy if label threshold `thres.longbuy` is met within the next `maxwindow` minutes and no undercut of current price before target threshold sample. All samples in between become longhold but they may be promoted to longbuy when they are the current sample
- shortbuy if label threshold `thres.shortbuy` is met within the next `maxwindow` minutes and no exceed of current price before target threshold sample. All samples in between become shorthold but they may be promoted to shortbuy when they are the current sample
- allclose if no trend is established within the next `maxwindow` minutes 
"""
mutable struct Trend03 <: AbstractTargets
    minwindow::Int # in minutes
    maxwindow::Int # in minutes
    thres::LabelThresholds
    ohlcv::Union{OhlcvData, Nothing}
    df::Union{DataFrame, Nothing}
    function Trend03(maxwindow, thres)
        @assert 0 <= maxwindow <= 4*60 "condition violated: 0 <= maxwindow=$(maxwindow) <= 4*60"
        @assert thres.shortbuy <= thres.shorthold <= thres.longhold <= thres.longbuy "condition violated: thres.shortbuy=$(thres.shortbuy) <= thres.shorthold=$(thres.shorthold) <= thres.longhold=$(thres.longhold) <= fdg.thres.longbuy=$(thres.longbuy)"
        trd = new(maxwindow, thres, nothing, nothing)
        return trd
    end
end

function setbase!(trd::Trend03, ohlcv::Ohlcv.OhlcvData)
    trd.ohlcv = ohlcv
    trd.df = DataFrame()
    supplement!(trd)
end

function removebase!(trd::Trend03)
    trd.ohlcv = nothing
    trd.df = nothing
end

"""
because prices can be very volatile no assumption on continous price development can be done and trend need to be checked for each maxwindow move

- samples are checked in sequence according timeline. The current sample under investigation is called focus sample, which is initially `allclose` but may be revised when processing later focus samples.
- starting from the focus sample prices, previous samples up to maximum of maxwindow minutes are assessed concerning a label change. 
- The first sample with a price lower than (current price - longbuy threshold) is a longbuy if there is no sample in between with a higher price than the focus price and no sample in between with a lower price than the longbuy candidate sample. 
- From that longbuy sample also all previous samples are longbuy samples if they fullfill these criteria.
- Samples between the closest identified longbuy and the focus sample with a price difference larger than longhold threshold are longhold, which may be promoted to longbuy when processing later focus samples.
- The same criteria but with opposite price difference sign are applicable for shortbuy, shorthold.
"""
function supplement!(trd::Trend03)
    if isnothing(trd.ohlcv)
        (verbosity >= 2) && println("no ohlcv found in trd - nothing to supplement")
        return
    end
    odf = Ohlcv.dataframe(trd.ohlcv)
    piv = odf[!, :pivot]
    if length(piv) > 0
        @assert 0 <= size(trd.df, 1) <= length(piv)
        @assert (size(trd.df, 1) > 0 ? trd.df[begin, :opentime] == odf[begin, :opentime] : true) "$(trd.df[begin, :opentime]) != $(odf[begin, :opentime])"
        flen = length(piv) - size(trd.df, 1)
        startix = size(trd.df, 1) + 1
        filldf = DataFrame(label=fill(allclose, flen), maxgain=fill(0f0, flen), mingain=fill(0f0, flen), opentime=odf[startix:end, :opentime])
        trd.df = size(trd.df, 1) > 0 ? vcat(trd.df, filldf) : filldf
        @assert size(trd.df, 1) == length(piv) "size(trd.df, 1)=$(size(trd.df, 1)) != length(piv)=$(length(piv))"
        @assert trd.df[end, :opentime] == odf[end, :opentime] "$(trd.df[end, :opentime]) != $(odf[end, :opentime])"
        minix = startix
        for focusix in startix:lastindex(piv) # iterate with focus sample forward across the to be added samples
            thistrend = flat
            lastpdiff = 0f0
            buyix = nothing
            maxbackix = max(firstindex(piv), focusix-trd.maxwindow)
            for assessix in (focusix-1):-1:maxbackix # for each focus look backwards for a potential trend that emerged
                pdiff = piv[focusix] - piv[assessix]
                thistrend = (thistrend == flat) && (pdiff != 0f0) ? (pdiff < 0f0 ? down : up) : thistrend
                if thistrend == up
                    if pdiff < 0f0 # assess price moves above focus price, which breaks the up trend
                        break
                    else
                        if (pdiff >= lastpdiff)
                            lastpdiff = pdiff
                            if (pdiff > trd.thres.longbuy) 
                                @assert !(trd.df[assessix, :label] in [shortbuy, shorthold]) "expecting different label than $(trd.df[assessix, :label])"
                                if trd.df[assessix, :label] == longbuy
                                    break # no need to go further
                                end
                                buyix = assessix
                            end
                        end
                    end
                elseif thistrend == down
                    if pdiff > 0f0 # passess price moves below focus price, which breaks the down trend
                        break
                    else
                        if (pdiff <= lastpdiff)
                            lastpdiff = pdiff
                            if (pdiff < trd.thres.shortbuy) 
                                @assert !(trd.df[assessix, :label] in [longbuy, longhold]) "expecting different label than $(trd.df[assessix, :label])"
                                if trd.df[assessix, :label] == shortbuy
                                    break # no need to go further
                                end
                                buyix = assessix
                            end
                        end
                    end
                end
            end
            if !isnothing(buyix)
                for assessix in buyix:(focusix-1) # now adjust label from buyix sample to focus sample
                    pdiff = piv[focusix] - piv[assessix]
                    if (pdiff > trd.thres.longbuy) 
                        trd.df[assessix, :label] = longbuy
                    elseif (pdiff > trd.thres.longhold) 
                        trd.df[assessix, :label] = longhold
                    elseif (pdiff < trd.thres.shortbuy) 
                        trd.df[assessix, :label] = shortbuy
                    elseif (pdiff < trd.thres.shorthold) 
                        trd.df[assessix, :label] = shorthold
                    else
                        trd.df[assessix, :label] = allclose
                    end
                end
            end
            minix = isnothing(buyix) ? minix : min(minix, buyix)
        end
        trendlastix = minix - 1
        debugix = 0
        for assessix in minix:lastindex(piv)
            if trendlastix < assessix
                thistrend = (trd.df[assessix, :label] in [longbuy, longhold]) ? up : ((trd.df[assessix, :label] in [shortbuy, shorthold]) ? down : flat)
                for tix in (assessix):lastindex(piv)
                    debugix = tix
                    if (thistrend == up)  && (trd.df[tix, :label] in [longbuy, longhold, allclose])
                        trendlastix = tix
                    elseif (thistrend == down) && (trd.df[tix, :label] in [shortbuy, shorthold, allclose])
                        trendlastix = tix
                    elseif (thistrend == flat) && (trd.df[tix, :label] in [allclose])
                        trendlastix = tix
                    else
                        # println("ERROR: thistrend=$(string(thistrend)), tix=$tix, trd.df[tix, :label]=$(trd.df[tix, :label]), assessix=$assessix, lastindex(piv)=$(lastindex(piv)), size(trd.df, 1)=$(size(trd.df, 1))")
                        break
                    end
                end
                @assert assessix <= trendlastix <= lastindex(piv) "ERROR: thistrend=$(string(thistrend)), trendlastix=$trendlastix, trd.df[assessix, :label]=$(trd.df[assessix, :label]), trd.df[debugix=$debugix, :label]=$(trd.df[debugix, :label]), assessix=$assessix, lastindex(piv)=$(lastindex(piv)), size(trd.df, 1)=$(size(trd.df, 1))"
            end
            trd.df[assessix, :maxgain] = (maximum(odf[assessix:trendlastix, :high]) - piv[assessix]) / piv[assessix]
            trd.df[assessix, :mingain] = (minimum(odf[assessix:trendlastix, :low]) - piv[assessix]) / piv[assessix]
        end
    end
end

uniquelabels(trd::Trend03) = [longbuy, longhold, shortbuy, shorthold, allclose]

function timerangecut!(trd::Trend03)
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
    @assert size(trd.df, 1) == size(Ohlcv.dataframe(trd.ohlcv), 1)
end

describe(trd::Trend03) = "$(typeof(trd))_$(isnothing(trd.ohlcv) ? "Base?" : trd.ohlcv.base)_maxwindow=$(trd.maxwindow)_thresholds=(longbuy=$(trd.thres.longbuy)_longhold=$(trd.thres.longhold)_shorthold=$(trd.thres.shorthold)_shortbuy=$(trd.thres.shortbuy))"
firstrowix(trd::Trend03)::Int = isnothing(trd.df) ? 1 : (size(trd.df, 1) > 0 ? firstindex(trd.df[!, 1]) : 1)
lastrowix(trd::Trend03)::Int = isnothing(trd.df) ? 0 : (size(trd.df, 1) > 0 ? lastindex(trd.df[!, 1]) : 0)

# df(trd::Trend03, startdt::DateTime, enddt::DateTime) = df(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))
# longbuybinarytargets(trd::Trend03, startdt::DateTime, enddt::DateTime) = [lb ? "longbuy" : "longclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]
# shortbuybinarytargets(trd::Trend03, startdt::DateTime, enddt::DateTime) = [lb ? "shortbuy" : "shortclose" for lb in df(trd, startdt, enddt)[!, :longbuy]]

labelbinarytargets(trd::Trend03, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labels(trd, firstix, lastix) .== label
labelbinarytargets(trd::Trend03, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelbinarytargets(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labelrelativegain(trd::Trend03, label::TradeLabel, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd)) = labelbinarytargets(trd, label, firstix, lastix) .* relativegain(trd, firstix, lastix)
labelrelativegain(trd::Trend03, label::TradeLabel, startdt::DateTime, enddt::DateTime) = labelrelativegain(trd, label, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

labels(trd::Trend03, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :label)
labels(trd::Trend03, startdt::DateTime, enddt::DateTime) = labels(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

relativegain(trd::Trend03, firstix::Integer=firstrowix(trd), lastix::Integer=lastrowix(trd))::AbstractVector = isnothing(trd.df) ? [] : view(trd.df, firstix:lastix, :reldiff)
relativegain(trd::Trend03, startdt::DateTime, enddt::DateTime) = relativegain(trd, Ohlcv.rowix(trd.df[!, :opentime], startdt), Ohlcv.rowix(trd.df[!, :opentime], enddt))

function Base.show(io::IO, trd::Trend03)
    println(io, "Trend03 targets base=$(isnothing(trd.ohlcv) ? "no ohlcv base" : trd.ohlcv.base) maxwindow=$(trd.maxwindow) label thresholds=$(thresholds(trd.thres)) $(isnothing(trd.df) ? "no df" : size(trd.df, 1) > 0 ? "from $(isnothing(trd.df) ? "no df" : trd.df[begin, :opentime]) to $(isnothing(trd.df) ? "no df" : trd.df[end, :opentime]) " : "no time range ")")
    # (verbosity >= 3) && println(io, "Features005 cfgdf=$(f5.cfgdf)")
    # (verbosity >= 2) && println(io, "Features005 config=$(f5.cfgdf[!, :config])")
    println(io, "Trend03 ohlcv=$(trd.ohlcv)")
end
#endregion Trend03

end  # module

