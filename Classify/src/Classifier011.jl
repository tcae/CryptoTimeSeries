
mutable struct BaseClassifier011
    ohlcv::Ohlcv.OhlcvData
    f4::Union{Nothing, Features.Features004}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier011(ohlcv::Ohlcv.OhlcvData, cfgid, f4=Features.Features004(ohlcv, usecache=true))
        cl = isnothing(f4) ? nothing : new(ohlcv, f4, cfgid)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier011)
    println(io, "BaseClassifier011[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f4=$(!isnothing(bc.f4)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier011)
    if !isnothing(bc.f4)
        Features.write(bc.f4)
    end
end

supplement!(bc::BaseClassifier011) = Features.supplement!(bc.f4, bc.ohlcv)

const REGRWINDOW011 = Int16[60]
const LONGTRENDTHRESHOLD011 = Float32[1f0] # , 0.04f0, 0.06f0, 1f0]  # 1f0 == switch off long trend following
const SHORTTRENDTHRESHOLD011 = Float32[-1f0] # , -0.04f0, -0.06f0, -1f0]  # -1f0 == switch off short trend following
const VOLATILITYBUYTHRESHOLD011 = Float32[-0.005f0]
const VOLATILITYSELLTHRESHOLD011 = Float32[0.005f0]
const VOLATILITYSHORTTHRESHOLD011 = Float32[-1f0] # 0f0, -1f0]  # -1f0 == switch off volatility short investments
const VOLATILITYLONGTHRESHOLD011 = Float32[-0.005f0]  # 1f0 == switch off volatility long investments
const OPTPARAMS011 = Dict(
    "regrwindow" => REGRWINDOW011,
    "longtrendthreshold" => LONGTRENDTHRESHOLD011,
    "shorttrendthreshold" => SHORTTRENDTHRESHOLD011,
    "volatilitybuythreshold" => VOLATILITYBUYTHRESHOLD011,   # symetric for long and short
    "volatilitysellthreshold" => VOLATILITYSELLTHRESHOLD011, # symetric for long and short
    "volatilityshortthreshold" => VOLATILITYSHORTTHRESHOLD011,
    "volatilitylongthreshold" => VOLATILITYLONGTHRESHOLD011
)

"""
Classifier011 idea
- leverage long and short volatility trades in times of flat slopes
- follow the regression long and short if their gradient exceeds thresholds
- use fixed regression window
"""
mutable struct Classifier011 <: AbstractClassifier  #* is also used as ohlcv and f4 anchor in cryptocockpit
    bc::Dict{AbstractString, BaseClassifier011}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    defaultcfgid
    function Classifier011(optparams=OPTPARAMS011)
        cl = new(Dict(), DataFrame(), optparams, nothing, 1)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier011, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier011(ohlcv, cl.defaultcfgid)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier011)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier011)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier011)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier011)::Integer =  maximum(Features.regressionwindows004)


function advice(cl::Classifier011, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier011"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end


#TODO implement batchadvice - see Trade
#TODO implement self evaluation on regular basis - transparent for Trade?
function advice(cl::Classifier011, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f4, ohlcvix)
    regry = Features.regry(bc.f4, cfg.regrwindow)[fix]
    grad = Features.grad(bc.f4, cfg.regrwindow)[fix]
    ra = Features.relativegain(regry, grad, cfg.regrwindow, forward=false)
    volatilitydownprice = regry * (1 + cfg.volatilitybuythreshold)
    volatilityupprice = regry * (1 + cfg.volatilitysellthreshold)
    ta = TradeAdvice(cl, bc.cfgid, allclose, 1f0, base, piv[ohlcvix], Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], Features.relativegain(regry, grad, 60, forward=false), 1f0, nothing)

    if ((piv[ohlcvix] < volatilitydownprice) && (ra >= cfg.volatilitylongthreshold)) || (ra >= cfg.longtrendthreshold)
        ta.tradelabel = longbuy
    elseif (piv[ohlcvix-1] >= volatilityupprice) && (ra < cfg.longtrendthreshold)
        ta.tradelabel = longclose
    elseif ((piv[ohlcvix] > volatilityupprice) && (ra <= cfg.volatilityshortthreshold)) || (ra <= cfg.shorttrendthreshold)
        ta.tradelabel = shortbuy
    elseif (piv[ohlcvix-1] <= volatilitydownprice) && (ra > cfg.shorttrendthreshold)
        ta.tradelabel = shortclose
    elseif (ra >= cfg.volatilitylongthreshold)
        ta.tradelabel = longhold
    elseif (ra <= cfg.volatilityshortthreshold)
        ta.tradelabel = shorthold
    else
        ta.tradelabel = allclose
    end
    return ta
end

configurationid4base(cl::Classifier011, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier011, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier011"
        return false
    end
end

function configureclassifier!(cl::Classifier011, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end
