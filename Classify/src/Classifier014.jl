"""
idea: stay close volatility crossing to leverage overall swing as well as volatility, considering that a swing is often followed buy volatile ending

- focus regression is the largest one with touchpoints in first and second half of the regression line
- each rergression line has its own volatility buy and sell amplitudes
- buy when 
    - buy volatility amplitude is reached
    - gain over focus regression length exceeds threshold AND next smaller regression gradient is equal or greater, which prevents buy when smaller regression flattens out
- sell when
  - volatility is reached and gain over focus regression length does not exceed threshold
"""
mutable struct BaseClassifier014
    ohlcv::Ohlcv.OhlcvData
    f5::Union{Nothing, Features.Features005}
    "cfgid: id to retrieve configuration parameters; uninitialized == 0"
    cfgid::Int16
    function BaseClassifier014(ohlcv::Ohlcv.OhlcvData, f5=Features.Features005(Features.featurespecification005(Features.regressionfeaturespec005(),[],[])))
        Features.setbase!(f5, ohlcv, usecache=true)
        cl = isnothing(f5) ? nothing : new(ohlcv, f5, 0)
        return cl
    end
end

function Base.show(io::IO, bc::BaseClassifier014)
    println(io, "BaseClassifier014[$(bc.ohlcv.base)]: ohlcv.ix=$(bc.ohlcv.ix),  ohlcv length=$(size(bc.ohlcv.df,1)), has f5=$(!isnothing(bc.f5)), cfgid=$(bc.cfgid)")
end

function writetargetsfeatures(bc::BaseClassifier014)
    if !isnothing(bc.f5)
        Features.write(bc.f5)
    end
end

supplement!(bc::BaseClassifier014) = Features.supplement!(bc.f5)

const REGRESSIONWINDOW014 = Int32[rw for rw in Features.regressionwindows005 if 4*60 <= rw <= 24*60]
const LONGTRENDTHRESHOLD014 = Float32[0.02f0]  # 1f0 == switch off long trend following
const SHORTTRENDTHRESHOLD014 = Float32[-0.02f0]  # -1f0 == switch off short trend following
const VOLATILITYBUYTHRESHOLD014 = Float32[-0.01f0, -0.02f0]
const VOLATILITYSELLTHRESHOLD014 = Float32[0.01f0, 0.02f0]
const VOLATILITYSHORTTHRESHOLD014 = Float32[0f0]  # -1f0 == switch off volatility short investments
const VOLATILITYLONGTHRESHOLD014 = Float32[0f0]  # 1f0 == switch off volatility long investments
const AUTOSWITCH014 = Int32[1, 2, 3]  # swithc between regrwindows 1=stay at 24h, 2= switch between 24h and 12h, 3= switch between 24h, 12h, 4h 
const OPTPARAMS014 = Dict(
    "regrwindow" => REGRESSIONWINDOW014,
    "longtrendthreshold" => LONGTRENDTHRESHOLD014,
    "shorttrendthreshold" => SHORTTRENDTHRESHOLD014,
    "volatilitybuythreshold" => VOLATILITYBUYTHRESHOLD014,   # symetric for long and short
    "volatilitysellthreshold" => VOLATILITYSELLTHRESHOLD014, # symetric for long and short
    "volatilityshortthreshold" => VOLATILITYSHORTTHRESHOLD014,
    "volatilitylongthreshold" => VOLATILITYLONGTHRESHOLD014,
    "autoswitch" => AUTOSWITCH014
)

"""
Classifier014 idea
- focus regression line is the one that exceeds a gain threshold over its regression line length and the gain of the next smaller (over focus length) is higher than focus gain ==> longbuy
- longclose if gain is decreased to gain threshold / x or pivot crosses regression line 
- longclose criteria in case of portfolio assets and new start (i.e. no known focus regression): calc focus regression -> if none then longclose otherwise follow longclose criteria above

"""
mutable struct Classifier014 <: AbstractClassifier
    bc::Dict{AbstractString, BaseClassifier014}
    cfg::Union{Nothing, DataFrame}  # configurations
    optparams::Dict  # maps parameter name strings to vectors of valid parameter values to be evaluated
    dbgdf::Union{Nothing, DataFrame} # debug DataFrame if required
    defaultcfgid
    function Classifier014(optparams=OPTPARAMS014)
        cl = new(Dict(), DataFrame(), optparams, DataFrame(), 1)
        readconfigurations!(cl, optparams)
        @assert !isnothing(cl.cfg)
        return cl
    end
end

function addbase!(cl::Classifier014, ohlcv::Ohlcv.OhlcvData)
    bc = BaseClassifier014(ohlcv)
    if isnothing(bc)
        @error "$(typeof(cl)): Failed to add $ohlcv"
    else
        cl.bc[ohlcv.base] = bc
    end
end

function supplement!(cl::Classifier014)
    for bcl in values(cl.bc)
        supplement!(bcl)
    end
end

function writetargetsfeatures(cl::Classifier014)
    for bcl in values(cl.bc)
        writetargetsfeatures(bcl)
    end
end

# requiredminutes(cl::Classifier014)::Integer =  isnothing(cl.cfg) || (size(cl.cfg, 1) == 0) ? requiredminutes() : max(maximum(cl.cfg[!,:regrwindow]),maximum(cl.cfg[!,:trendwindow])) + 1
requiredminutes(cl::Classifier014)::Integer =  maximum(Features.regressionwindows005)


function advice(cl::Classifier014, base::AbstractString, dt::DateTime; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    bc = ohlcv = nothing
    if base in keys(cl.bc)
        bc = cl.bc[base]
        ohlcv = bc.ohlcv
    else
        (verbosity >= 2) && @warn "$base not found in Classifier014"
        return nothing
    end
    oix = Ohlcv.rowix(ohlcv, dt)
    return advice(cl, ohlcv, oix, investment=investment)
end

"checks whether all regression lines in rws were touched or crossed in the first half as well as the second half of the regression line"
function regressiontouched(piv, pix, f5, fix, rws)
    res = Dict()
    checkminutes = 24*60
    halftime = round(Int, checkminutes / 2)
    if (pix < checkminutes) || (fix < checkminutes)
        [res[rw] = (rw == rws[end]) for rw in rws]  # no history => stay with longest regression window
    else
        for rw in rws
            regry = Features.regry(f5, rw)
            c1 = c2 = 0
            dl = nothing

            function _check(c, dl, ix)
                d = piv[pix-ix] - regry[fix-ix]
                if isnothing(dl) ? (d == 0) : (dl * d <= 0)
                    # change and on different sides of regression line
                    c += 1
                    dl = d
                end
                return c, dl
            end

            for ix in 0:halftime-1
                c2, dl = _check(c2, dl, ix)
            end
            for ix in halftime:checkminutes-1
                c1, dl = _check(c1, dl, ix)
            end
            res[rw] = ((c1>0) && (c2>0))
        end
    end
    return res
end

function advice(cl::Classifier014, ohlcv::Ohlcv.OhlcvData, ohlcvix=ohlcv.ix; investment::Union{Nothing, TradeAdvice}=nothing)::Union{Nothing, TradeAdvice}
    if ohlcvix < requiredminutes(cl)
        return nothing
    end
    base = ohlcv.base
    bc = cl.bc[base]
    cfg = configuration(cl, bc.cfgid)
    piv = Ohlcv.pivot!(ohlcv)
    fix = Features.featureix(bc.f5, ohlcvix)
    touched = regressiontouched(piv, ohlcvix, bc.f5, fix, REGRESSIONWINDOW014)
    regrwindow = REGRESSIONWINDOW014[end]
    for rw in reverse(REGRESSIONWINDOW014)
        if touched[rw]
            regrwindow = rw
            break
        end
    end
    regry = Features.regry(bc.f5, regrwindow)[fix]
    grad = Features.grad(bc.f5, regrwindow)[fix]
    ra = Features.relativegain(regry, grad, regrwindow, forward=false)
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
    else
        ta.tradelabel = shorthold
    end
    push!(cl.dbgdf, (otime=Ohlcv.dataframe(ohlcv)[ohlcvix, :opentime], tradelabel=ta.tradelabel, cfgid=bc.cfgid, rw=regrwindow, cfg...))
    return ta
end

configurationid4base(cl::Classifier014, base::AbstractString)::Integer = base in keys(cl.bc) ? cl.bc[base].cfgid : 0

function configureclassifier!(cl::Classifier014, base::AbstractString, configid::Integer)
    if base in keys(cl.bc)
        cl.bc[base].cfgid = configid
        return true
    else
        @error "cannot find $base in Classifier014"
        return false
    end
end

function configureclassifier!(cl::Classifier014, configid::Integer, updatedbases::Bool)
    cl.defaultcfgid = configid
    if updatedbases
        for base in keys(cl.bc)
            cl.bc[base].cfgid = configid
        end
    end
end

