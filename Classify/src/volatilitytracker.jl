
module VolatilityTracker

using DataFrames, Dates, Statistics, CSV, Logging, JDF
using EnvConfig, TestOhlcv, Ohlcv, Features, Targets

const FEE = 0.08 / 100 # VIP1 Bybit taker fee (worst case)

function trackregression!(tradedf, f2::Features.Features002; asset, trendminutes, buygainthreshold, sellgainthreshold, regrwindow, buygap, sellgap, selfmonitor, stdcheck, maxconcurrentbuy)
    longlastok = shortlastok = true
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    longopenix = Int32[]
    shortopenix = Int32[]
    # gap = 5  # 5 minute gap between trades
    longcloseix = -sellgap
    shortcloseix = -sellgap
    ctrades = otrades = otok = ootok = cltok = cstok = 0
    trendgainlong = zeros(trendminutes)
    trendgainshort = zeros(trendminutes)
    drawdownminutes = 30 * 24 * 60
    drawdown30daysshort = zeros(drawdownminutes)
    drawdown30dayslong = zeros(drawdownminutes)
    #TODO maxnumber of concurrent positions (should be length of openix arrays)
    #TODO monitor shortgain against long gain

    function closetrades!(openix, currentix, longshort, handleall)
        ctrades += 1
        cumgain = 0.0
        while length(openix) > 0
            # handle open long positions
            tl = currentix - openix[begin]
            if longshort == "long"
                gain = Targets.relativegain(piv, openix[begin], currentix, relativefee=FEE)
                longlastok = (gain >= selfmonitor)
                handleall = longlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=currentix, tradelen=tl, gain=gain, trendminutes=trendminutes, buygainthreshold=buygainthreshold, sellgainthreshold=sellgainthreshold, buygap=buygap, sellgap=sellgap, selfmonitor=selfmonitor, stdcheck=stdcheck, maxconcurrentbuy=maxconcurrentbuy, drawdown30dayslong=sum(drawdown30dayslong), drawdown30daysshort=sum(drawdown30daysshort)))
                cltok += 1
            else
                gain = -Targets.relativegain(piv, openix[begin], currentix, relativefee=FEE)
                shortlastok = (gain >= selfmonitor)
                handleall = shortlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=currentix, tradelen=tl, gain=gain, trendminutes=trendminutes, buygainthreshold=buygainthreshold, sellgainthreshold=sellgainthreshold, buygap=buygap, sellgap=sellgap, selfmonitor=selfmonitor, stdcheck=stdcheck, maxconcurrentbuy=maxconcurrentbuy, drawdown30dayslong=sum(drawdown30dayslong), drawdown30daysshort=sum(drawdown30daysshort)))
                cstok += 1
            end
            cumgain += gain
            openix = deleteat!(openix, 1)
            if !handleall
                break
            end
        end
        return cumgain
    end

    function opentrades!(openix, currentix)
        otrades += 1
        if !stdcheck || ((f2.regr[regrwindow].std[currentix] / f2.regr[regrwindow].regry[currentix]) > buygainthreshold)
            if ((length(openix) == 0) || ((currentix - openix[end]) >= buygap)) && ((maxconcurrentbuy == 0) || (maxconcurrentbuy > length(openix)))
                push!(openix, currentix)
                ootok += 1
            end
            otok += 1
        end
        return openix
    end

    function preparerollinggain!(gainvector, index)
        len = length(gainvector)
        if len > 0
            gainvector[index % len + 1] = 0.0  # will be updated in that cycle
        end
    end

    rollinggainupdate!(gainvector, index, gain) = (length(gainvector) > 0 ? gainvector[index % length(gainvector) + 1] = gain : 0.0)

    function sufficientgain(openix, currentix, longshort, sellgainthreshold)
        if length(openix) > 0
            gain = longshort == "long" ? Targets.relativegain(piv, openix[begin], currentix, relativefee=FEE) : -Targets.relativegain(piv, openix[begin], currentix, relativefee=FEE)
            return gain >= sellgainthreshold
        else
            return false
        end
    end

    #* idea: adapt gap with direction gradient
    for ix in eachindex(piv)
        preparerollinggain!(drawdown30daysshort, ix)
        preparerollinggain!(drawdown30dayslong, ix)
        preparerollinggain!(trendgainshort, ix)
        preparerollinggain!(trendgainlong, ix)
        if sellgainthreshold >= 0 ? piv[ix] > f2.regr[regrwindow].regry[ix] * (1 + sellgainthreshold) : sufficientgain(longopenix, ix, "long", -sellgainthreshold)
            if (ix - longcloseix) >= sellgap
                gain = closetrades!(longopenix, ix, "long", false)
                rollinggainupdate!(drawdown30dayslong, ix, gain)
                rollinggainupdate!(trendgainlong, ix, gain)
                longcloseix = ix
            end
        end
        if piv[ix] > f2.regr[regrwindow].regry[ix] * (1 + buygainthreshold)
            if shortlastok && (trendminutes > 0 ? sum(trendgainlong) < buygainthreshold : true)
                opentrades!(shortopenix, ix)
            end
        end
        if sellgainthreshold >= 0 ? piv[ix] < f2.regr[regrwindow].regry[ix] * (1 - sellgainthreshold) : sufficientgain(shortopenix, ix, "short", -sellgainthreshold)
            if (ix - shortcloseix) >= sellgap
                gain = closetrades!(shortopenix, ix, "short", false)
                rollinggainupdate!(drawdown30daysshort, ix, gain)
                rollinggainupdate!(trendgainshort, ix, gain)
                shortcloseix = ix
            end
        end
        if piv[ix] < f2.regr[regrwindow].regry[ix] * (1 - buygainthreshold)
            if longlastok && (trendminutes > 0 ? sum(trendgainshort) > -buygainthreshold : true)
                opentrades!(longopenix, ix)
            end
        end

    end
    # println("trackregression!: $asset, $regrwindow, $gainthreshold, $gap, $selfmonitor size(tradedf)=$(size(tradedf))")
    # println("ctrades=$ctrades otrades=$otrades otok=$otok ootok=$ootok cltok=$cltok cstok=$cstok")
    return tradedf
end

function trackohlc(ohlcv::Ohlcv.OhlcvData, rkeys, trendrwfactors, buygainthresholds, sellgainthresholds, buygaps, sellgaps, selfmonitorset, stdcheckset, maxconcurrentbuyset)
    tdf = DataFrame()
    asset = Ohlcv.basecoin(ohlcv)
    println("$(EnvConfig.now()) calculating F002 features for $asset")
    f2 = Features.Features002(ohlcv)
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        # println("$(EnvConfig.now()) processing regression window $rw")
        # println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        if kix != firstindex(rkeys)
            for buygainthreshold in buygainthresholds
                for sellgainthreshold in sellgainthresholds
                    for buygap in buygaps
                        for sellgap in sellgaps
                            for selfmonitor in selfmonitorset
                                for stdcheck in stdcheckset
                                    for maxconcurrentbuy in maxconcurrentbuyset
                                        for trendrwfactor in trendrwfactors
                                            trendminutes = rw * trendrwfactor
                                            # trendrw = trenddircontrol ? (kix == lastindex(rkeys) ? 0 : rkeys[kix+1]) : 0
                                            println("$(EnvConfig.now()) assessing performance for asset=$asset, trendminutes=$trendminutes, buygainthreshold=$buygainthreshold, sellgainthreshold=$sellgainthreshold, regrwindow=$rw, buygap=$buygap, sellgap=$sellgap, selfmonitor=$selfmonitor, stdcheck=$stdcheck, maxconcurrentbuy=$maxconcurrentbuy")
                                            tdf = trackregression!(tdf, f2, asset=asset, trendminutes=trendminutes, buygainthreshold=buygainthreshold, sellgainthreshold=sellgainthreshold, regrwindow=rw, buygap=buygap, sellgap=sellgap, selfmonitor=selfmonitor, stdcheck=stdcheck, maxconcurrentbuy=maxconcurrentbuy)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return tdf
end

function minmaxaccugain(nti)
    sumgain = maxgain = mingain = 0.0
    for trade in nti
        sumgain += trade.gain
        maxgain = sumgain > maxgain ? sumgain : maxgain
        mingain = sumgain < mingain ? sumgain : mingain
    end
    return round(maxgain, digits=3), round(mingain, digits=3)
end

function kpi(tradedf, asset, regr, longshort, trendrw, buygainthreshold, sellgainthreshold, buygap, sellgap, selfmonitor, stdcheck, maxconcurrentbuy)
    # println("kpi: $asset, $regr, $longshort, $gainthreshold, $buygap, $sellgap, $selfmonitor")
    # println(describe(tradedf, :all))
    cumgain=0.0; meangain=0.0; mediangain=0.0; maxgain=0.0; mingain=0.0; count=0; mediantl=0.0; maxtl=0.0; drawdown30dayslong=0.0; drawdown30daysshort=0.0
    if size(tradedf, 1) > 0
        maxgain, mingain = minmaxaccugain(Tables.namedtupleiterator(sort(tradedf, :closeix, view=true)))
        gainvec = tradedf[!, :gain]
        tlvec = tradedf[!, :tradelen]
        cumgain = round(sum(gainvec), digits=3)
        count = length(gainvec)
        meangain = round(mean(gainvec), digits=3)
        mediangain = round(median(gainvec), digits=3)
        mediantl = round(median(tlvec), digits=0)
        maxtl = round(maximum(tlvec), digits=0)
        drawdown30dayslong = round(minimum(tradedf[!, :drawdown30dayslong]), digits=3)
        drawdown30daysshort = round(minimum(tradedf[!, :drawdown30daysshort]), digits=3)
    end
    return (asset=asset, regr=regr, longshort=longshort, buygap=buygap, sellgap=sellgap, trendrw=trendrw, buygainthreshold=buygainthreshold, sellgainthreshold=sellgainthreshold, selfmonitor=selfmonitor, stdcheck=stdcheck, maxconcurrentbuy=maxconcurrentbuy, cumgain=cumgain, meangain=meangain, mediangain=mediangain, maxgain=maxgain, mingain=mingain, count=count, mediantradelen=mediantl, maxtradelen=maxtl, drawdown30dayslong=drawdown30dayslong, drawdown30daysshort=drawdown30daysshort)
end

function calckpi!(kpidf, tradedf, assets, rkeys, trendrwfactors, buygainthresholds, sellgainthresholds, buygaps, sellgaps, selfmonitorset, stdcheckset, maxconcurrentbuyset, longshortset)
    for asset in assets
        for kix in eachindex(rkeys)
            rw = rkeys[kix]
            if kix != firstindex(rkeys)
                for buygainthreshold in buygainthresholds
                    for sellgainthreshold in sellgainthresholds
                        for buygap in buygaps
                            for sellgap in sellgaps
                                for selfmonitor in selfmonitorset
                                    for stdcheck in stdcheckset
                                        for longshort in longshortset
                                            for maxconcurrentbuy in maxconcurrentbuyset
                                                for trendrwfactor in trendrwfactors
                                                    trendminutes = rw * trendrwfactor
                                                    # trendrw = trenddircontrol ? (kix == lastindex(rkeys) ? continue : rkeys[kix+1]) : 0
                                                    # subdf = filter(row -> (row.asset == asset) && (row.regr == rw) && (row.longshort == longshort) && (row.gainthreshold == gainthreshold) && (row.gap==gap) && (row.selfmonitor==selfmonitor), tradedf, view=true)
                                                    subdf = subset(tradedf, :asset => x -> x .== asset, :regr => x -> x .== rw, :longshort => x -> x .== longshort, :trendminutes => x -> x .== trendminutes, :buygainthreshold => x -> x .== buygainthreshold, :sellgainthreshold => x -> x .== sellgainthreshold, :buygap => x -> x .== buygap, :sellgap => x -> x .== sellgap, :selfmonitor => x -> x .== selfmonitor, :stdcheck => x -> x .== stdcheck, :maxconcurrentbuy => x -> x .== maxconcurrentbuy, view=true)
                                                    println("$(EnvConfig.now()) calculating kpi for asset=$asset, regrwindow=$rw, longshort=$longshort, trendminutes=$trendminutes, buygainthreshold=$buygainthreshold, sellgainthreshold=$sellgainthreshold, buygap=$buygap, sellgap=$sellgap, selfmonitor=$selfmonitor, stdcheck=$stdcheck, maxconcurrentbuy=$maxconcurrentbuy")
                                                    push!(kpidf, kpi(subdf, asset, rw, longshort, trendminutes, buygainthreshold, sellgainthreshold, buygap, sellgap, selfmonitor, stdcheck, maxconcurrentbuy))
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return kpidf
end

const VOLATILITYSTUDYKPIFILE = "volatilitykpi.csv"
const VOLATILITYSTUDYTRADEFILE = "volatilitytrades.jdf"

function loadstudy()
    df = DataFrame()
    try
        # kpifilename = EnvConfig.logpath(VOLATILITYSTUDYKPIFILE)
        tradefilename = EnvConfig.logpath(VOLATILITYSTUDYTRADEFILE)
        if isdir(tradefilename)
            df = DataFrame(JDF.loadjdf(tradefilename))
            println("loaded trade data of assets $(unique(df[!, :asset])) with size $(size(df)) from $tradefilename")
        else
            println("no data found for $tradefilename")
        end
    catch e
        Logging.@warn "exception $e detected"
    end
    return df
end

function savestudy(tradedf)
    tradefilename = EnvConfig.logpath(VOLATILITYSTUDYTRADEFILE)
    EnvConfig.savebackup(tradefilename)
    try
        JDF.savejdf(tradefilename, tradedf)
        println("$(EnvConfig.now()) saved tradedf as $tradefilename with size $(size(tradedf, 1)) of assets $(unique(tradedf[!, :asset]))")
    catch e
        Logging.@warn "$(EnvConfig.now()) exception $e detected when saving $tradefilename with df size=$(size(tradedf))"
    end
end

function trackasset(bases, startdt=nothing, period=nothing)
    kpidf = DataFrame()
    tradedf = loadstudy()
    loadedbases = []
    if size(tradedf, 1) > 0
        loadedbases = unique(tradedf[!, :asset])
        rkeys = unique(tradedf[!, :regr])
        println("loaded already studied assets $loadedbases")
    else
    end
    rkeys = sort([rw for rw in Features.regressionwindows002])  #  if 200 < rw
    buygainthresholds = [0.01, 0.02]  # [0.02]  #
    sellgainthresholds = [-0.02, -0.01, 0.01]  # [0.02]  #
    buygaps = [60]
    sellgaps = [60]
    selfmonitorset = [-Inf]  # [-Inf, 0.0]
    longshortset = ["long"]  # , "short"]
    maxconcurrentbuyset = [2, 20]
    trendrwfactors = [0]  # , 1, 2, 4]
    stdcheckset = [false]
    for base in bases
        if !(base in loadedbases)
            ohlcv = Ohlcv.defaultohlcv(base)
            Ohlcv.read!(ohlcv)
            if !isnothing(startdt) && !isnothing(period)
                enddt = startdt + period
                Ohlcv.timerangecut!(ohlcv, startdt, enddt)
                # subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
            end
            println("loaded $ohlcv")
            tdf = trackohlc(ohlcv, rkeys, trendrwfactors, buygainthresholds, sellgainthresholds, buygaps, sellgaps, selfmonitorset, stdcheckset, maxconcurrentbuyset)
            if size(tdf, 1) > 0
                tradedf = vcat(tradedf, tdf)
            else
                @warn "empty data set of trackohlc for $base"
            end
            savestudy(tradedf)
        else
            println("skipping $base because it was already studied and is loaded")
        end
    end
    calckpi!(kpidf, tradedf, bases, rkeys, trendrwfactors, buygainthresholds, sellgainthresholds, buygaps, sellgaps, selfmonitorset, stdcheckset, maxconcurrentbuyset, longshortset)
    println(kpidf)
    kpifilename = EnvConfig.logpath(VOLATILITYSTUDYKPIFILE)
    EnvConfig.savebackup(kpifilename)
    CSV.write(kpifilename, kpidf, decimal=',', delim=';')  # decimal as , to consume with European locale
    println("done")
end

end  #of module

using EnvConfig, Dates, Ohlcv
Ohlcv.verbosity = 3
# startdt = DateTime("2022-05-15T22:54:00")
enddt = DateTime("2024-06-18T13:29:00")
period = Month(1)
startdt = enddt - period
EnvConfig.init(production)
# EnvConfig.setlogpath("230802-240107_VolatilityTracker")
EnvConfig.setlogpath("2424-3_TrendawareVolatilityTracker")
VolatilityTracker.trackasset(["BTC"])
# VolatilityTracker.trackasset(["BTC"], startdt, period)
# VolatilityTracker.trackasset(["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"])
# VolatilityTracker.trackasset(["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"], DateTime("2024-06-12T22:54:00"), Year(3))
# VolatilityTracker.trackasset(["MATIC", "BTC"], startdt, period)
