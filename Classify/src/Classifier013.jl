
mutable struct BaseClassifier013
    ohlcv::Ohlcv.OhlcvData
    f5::Union{Nothing, Features.Features005}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier013(ohlcv::Ohlcv.OhlcvData, f5=Features.Features005(Features.featurespecification005(Features.regressionfeaturespec005(),[],[])))
        Features.setbase!(f5, ohlcv, usecache=true)
        cl = isnothing(f5) ? nothing : new(ohlcv, f5, 0)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier013)
    println(io, "BaseClassifier013[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f5=$(!isnothing(bc.f5)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier013)
    if !isnothing(bc.f5)
        Features.write(bc.f5)
    end
end

supplement!(bc::BaseClassifier013) = Features.supplement!(bc.f5)

const REGRESSIONWINDOW013 = Int32[rw for rw in Features.regressionwindows005 if 12*60 <= rw <= 24*60]
const LONGGAINTHRESHOLD013 = Float32[0.02f0, 0.04f0, 1f0]
const SHORTGAINTHRESHOLD013 = Float32[-0.02f0, -0.04f0, -1f0]
const LONGSELLTHRESHOLD013 = Float32[0.01f0]
const SHORTSELLTHRESHOLD013 = Float32[-0.01f0]
const SELLTHRESHOLDFACTOR013 = Float32[0.25f0]
const OPTPARAMS013 = Dict(
    "regrwindow" => REGRESSIONWINDOW013,
    "longgainthreshold" => LONGGAINTHRESHOLD013,
    "shortgainthreshold" => SHORTGAINTHRESHOLD013,
    "longsellthreshold" => LONGSELLTHRESHOLD013,
    "shortsellthreshold" => SHORTSELLTHRESHOLD013,
    "sellthresholdfactor" => SELLTHRESHOLDFACTOR013
)

"""
Classifier013 idea
- focus regression line is the one that exceeds a gain threshold over its regression line length and the gain of the next smaller (over focus length) is higher than focus gain ==> longbuy
- longclose if gain is decreased to gain threshold / x or pivot crosses regression line 
- longclose criteria in case of portfolio assets and new start (i.e. no known focus regression): calc focus regression -> if none then longclose otherwise follow longclose criteria above

"""
mutable struct Classifier013 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier013}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    function Classifier013(optparams=OPTPARAMS013)
        cl = new(Dict(), DataFrame(), optparams, nothing)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier013, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier013(ohlcv)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier013)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier013)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier013)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier013)::Integer =  maximum(Features.regressionwindows005)


function advice(cl::Classifier013, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier013"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end

function advice(cl::Classifier013, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f5, ohlcvix)
    regrwindow = nothing
    if !isnothing(investment)
        if configureclassifier!(cl, base, investment.configid)
            regrwindow = cfg.regrwindow
        else
            @error "cannot configure string(cl) with configid from $investment"
        end
    end
    lastgrad = nothing
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], 0f0, 1f0, nothing)
    if !isnothing(regrwindow) # check longclose condition
        regry = Features.regry(bc.f5, regrwindow)[fix]
        grad = Features.grad(bc.f5, regrwindow)[fix]
        gain = Features.relativegain(regry, grad, regrwindow, forward=false)
        ta.hourlygain = Features.relativegain(regry, grad, 60, forward=false)
        if investment.tradelabel in [longhold, longbuy, longstrongbuy]
            if (gain < cfg.longgainthreshold * cfg.sellthresholdfactor) && (piv[ohlcvix] >= regry * (1 + cfg.longsellthreshold))
                ta.tradelabel = longclose
            else
                ta.tradelabel = longhold
            end
        elseif investment.tradelabel in [shortstrongbuy, shortbuy, shorthold]
            if (gain > cfg.shortgainthreshold * cfg.sellthresholdfactor) && (piv[ohlcvix] <= regry * (1 + cfg.shortsellthreshold))
                ta.tradelabel = shortclose
            else
                ta.tradelabel = shorthold
            end
        end
    end
    longbuyenabled = shortbuyenabled = true
    for rw in Features.regressionwindows005
        regry = Features.regry(bc.f5, rw)[fix]
        grad = Features.grad(bc.f5, rw)[fix]
        gain = Features.relativegain(regry, grad, rw, forward=false)
        if gain < 0f0
            longbuyenabled = false  # no long buy if a smaller regression window has negative gradient
        elseif gain > 0f0
            shortbuyenabled = false  # no short buy if a smaller regression window has positive gradient
        end
        if !isnothing(regrwindow) && (rw > regrwindow) # no focus on longer term windows
            break
        elseif gain >= cfg.longgainthreshold
            # if longbuyenabled && (isnothing(lastgrad) || (lastgrad >= grad))  # check that trend is still there and not a short term peak that is flatten out
                # longbuy condition in place - ay be with shorter rw as established regrwindow
                bc.cfgid = configurationid(cl, (cfg..., regrwindow=rw))
                regrwindow = rw
                ta.hourlygain = Features.relativegain(regry, grad, 60, forward=false)
                ta.tradelabel = longbuy
                break
            # end
        elseif gain <= cfg.shortgainthreshold
            # if shortbuyenabled && (isnothing(lastgrad) || (lastgrad <= grad))  # check that trend is still there and not a short term peak that is flatten out
                # shortbuy condition in place - may be with shorter rw as established regrwindow
                bc.cfgid = configurationid(cl, (cfg..., regrwindow=rw))
                regrwindow = rw
                ta.hourlygain = Features.relativegain(regry, grad, 60, forward=false)
                ta.tradelabel = shortbuy
                break
            # end
        end
        lastgrad = grad
    end
    return ta
end

configurationid4base(cl::Classifier013, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier013, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier013"
        return false
    end
end

