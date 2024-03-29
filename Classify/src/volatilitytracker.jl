
module VolatilityTracker

using DataFrames, Dates, Statistics, CSV, Logging, JDF
using EnvConfig, TestOhlcv, Ohlcv, Features

function trackregression!(tradedf, f2::Features.Features002; asset, trendminutes, gainthreshold, regrwindow, gap, selfmonitor, stdcheck)
    longlastok = shortlastok = true
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    longopenix = Int32[]
    shortopenix = Int32[]
    # gap = 5  # 5 minute gap between trades
    longcloseix = -gap
    shortcloseix = -gap
    ctrades = otrades = otok = ootok = cltok = cstok = trendlongstop = trendshortstop = 0
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
                gain = Ohlcv.relativegain(piv, openix[begin], currentix)
                longlastok = (gain >= selfmonitor)
                handleall = longlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=currentix, tradelen=tl, gain=gain, trendminutes=trendminutes, gainthreshold=gainthreshold, gap=gap, selfmonitor=selfmonitor, stdcheck=stdcheck, drawdown30dayslong=sum(drawdown30dayslong), drawdown30daysshort=sum(drawdown30daysshort)))
                cltok += 1
            else
                gain = -Ohlcv.relativegain(piv, openix[begin], currentix)
                shortlastok = (gain >= selfmonitor)
                handleall = shortlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=currentix, tradelen=tl, gain=gain, trendminutes=trendminutes, gainthreshold=gainthreshold, gap=gap, selfmonitor=selfmonitor, stdcheck=stdcheck, drawdown30dayslong=sum(drawdown30dayslong), drawdown30daysshort=sum(drawdown30daysshort)))
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
        if !stdcheck || ((f2.regr[regrwindow].std[currentix] / f2.regr[regrwindow].regry[currentix]) > gainthreshold)
            if ((length(openix) == 0) || ((currentix - openix[end]) >= gap))
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

    #* idea: adapt gap with direction gradient
    for ix in eachindex(piv)
        preparerollinggain!(drawdown30daysshort, ix)
        preparerollinggain!(drawdown30dayslong, ix)
        preparerollinggain!(trendgainshort, ix)
        preparerollinggain!(trendgainlong, ix)
        if piv[ix] > f2.regr[regrwindow].regry[ix] * (1 + gainthreshold)  # + f2.regr[regrwindow].std[ix]
            if (ix - longcloseix) >= gap
                gain = closetrades!(longopenix, ix, "long", false)
                rollinggainupdate!(drawdown30dayslong, ix, gain)
                rollinggainupdate!(trendgainlong, ix, gain)
                longcloseix = ix
            end
            if shortlastok && (trendminutes > 0 ? sum(trendgainlong) < gainthreshold : true)
                opentrades!(shortopenix, ix)
            else
                trendshortstop += 1
            end
        elseif piv[ix] < f2.regr[regrwindow].regry[ix] * (1 - gainthreshold)  # - f2.regr[regrwindow].std[ix]
            if (ix - shortcloseix) >= gap
                gain = closetrades!(shortopenix, ix, "short", false)
                rollinggainupdate!(drawdown30daysshort, ix, gain)
                rollinggainupdate!(trendgainshort, ix, gain)
                shortcloseix = ix
            end
            if longlastok && (trendminutes > 0 ? sum(trendgainshort) > -gainthreshold : true)
                opentrades!(longopenix, ix)
            else
                trendlongstop += 1
            end
        end

    end
    # println("trackregression!: $asset, $regrwindow, $gainthreshold, $gap, $selfmonitor size(tradedf)=$(size(tradedf))")
    # println("ctrades=$ctrades otrades=$otrades otok=$otok ootok=$ootok cltok=$cltok cstok=$cstok")
    # println("trendrw=$trendrw: NOT(trendgain > -gainthreshold=-$gainthreshold) -> don't open long triggered $trendlongstop times")
    # println("trendrw=$trendrw: NOT(trendgain < gainthreshold=$gainthreshold) -> don't open short triggered $trendshortstop times")
    return tradedf
end

function trackohlc(ohlcv::Ohlcv.OhlcvData, rkeys, trendrwfactors, gainthresholds, gaps, selfmonitorset, stdcheckset)
    tdf = DataFrame()
    asset = Ohlcv.basecoin(ohlcv)
    println("$(EnvConfig.now()) calculating F002 features for $asset")
    f2 = Features.Features002(ohlcv)
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        # println("$(EnvConfig.now()) processing regression window $rw")
        # println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        if kix != firstindex(rkeys)
            for gainthreshold in gainthresholds
                for gap in gaps
                    for selfmonitor in selfmonitorset
                        for stdcheck in stdcheckset
                            for trendrwfactor in trendrwfactors
                                trendminutes = rw * trendrwfactor
                                # trendrw = trenddircontrol ? (kix == lastindex(rkeys) ? 0 : rkeys[kix+1]) : 0
                                # println("$(EnvConfig.now()) analyzing volatility performace of regression window $rw, gainthreshold=$gainthreshold, gap=$gap")
                                println("$(EnvConfig.now()) assessing performance for asset=$asset, trendminutes=$trendminutes, gainthreshold=$gainthreshold, regrwindow=$rw, gap=$gap, selfmonitor=$selfmonitor, stdcheck=$stdcheck")
                                tdf = trackregression!(tdf, f2, asset=asset, trendminutes=trendminutes, gainthreshold=gainthreshold, regrwindow=rw, gap=gap, selfmonitor=selfmonitor, stdcheck=stdcheck)
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
    return round(maxgain*100, digits=3), round(mingain*100, digits=3)
end

function kpi(tradedf, asset, regr, longshort, trendrw, gainthreshold, gap, selfmonitor, stdcheck)
    # println("kpi: $asset, $regr, $longshort, $gainthreshold, $gap, $selfmonitor")
    # println(describe(tradedf, :all))
    cumgain=0.0; meangain=0.0; mediangain=0.0; maxgain=0.0; mingain=0.0; count=0; mediantl=0.0; maxtl=0.0; drawdown30dayslong=0.0; drawdown30daysshort=0.0
    if size(tradedf, 1) > 0
        maxgain, mingain = minmaxaccugain(Tables.namedtupleiterator(sort(tradedf, :closeix, view=true)))
        gainvec = tradedf[!, :gain]
        tlvec = tradedf[!, :tradelen]
        cumgain = round(sum(gainvec)*100, digits=3)
        count = length(gainvec)
        meangain = round(mean(gainvec)*100, digits=3)
        mediangain = round(median(gainvec)*100, digits=3)
        mediantl = round(median(tlvec), digits=0)
        maxtl = round(maximum(tlvec), digits=0)
        drawdown30dayslong = round(minimum(tradedf[!, :drawdown30dayslong]), digits=3)
        drawdown30daysshort = round(minimum(tradedf[!, :drawdown30daysshort]), digits=3)
    end
    return (asset=asset, regr=regr, longshort=longshort, gap=gap, trendrw=trendrw, gainthreshold=gainthreshold*100, selfmonitor=selfmonitor, stdcheck=stdcheck, cumgain=cumgain, meangain=meangain, mediangain=mediangain, maxgain=maxgain, mingain=mingain, count=count, mediantradelen=mediantl, maxtradelen=maxtl, drawdown30dayslong=drawdown30dayslong, drawdown30daysshort=drawdown30daysshort)
end

function calckpi!(kpidf, tradedf, assets, rkeys, trendrwfactors, gainthresholds, gaps, selfmonitorset, stdcheckset, longshortset)
    for asset in assets
        for kix in eachindex(rkeys)
            rw = rkeys[kix]
            if kix != firstindex(rkeys)
                for gainthreshold in gainthresholds
                    for gap in gaps
                        for selfmonitor in selfmonitorset
                            for stdcheck in stdcheckset
                                for longshort in longshortset
                                    for trendrwfactor in trendrwfactors
                                        trendminutes = rw * trendrwfactor
                                        # trendrw = trenddircontrol ? (kix == lastindex(rkeys) ? continue : rkeys[kix+1]) : 0
                                        # subdf = filter(row -> (row.asset == asset) && (row.regr == rw) && (row.longshort == longshort) && (row.gainthreshold == gainthreshold) && (row.gap==gap) && (row.selfmonitor==selfmonitor), tradedf, view=true)
                                        subdf = subset(tradedf, :asset => x -> x .== asset, :regr => x -> x .== rw, :longshort => x -> x .== longshort, :trendminutes => x -> x .== trendminutes, :gainthreshold => x -> x .== gainthreshold, :gap => x -> x .== gap, :selfmonitor => x -> x .== selfmonitor, :stdcheck => x -> x .== stdcheck, view=true)
                                        println("$(EnvConfig.now()) calculating kpi for asset=$asset, regrwindow=$rw, longshort=$longshort, trendminutes=$trendminutes, gainthreshold=$gainthreshold, gap=$gap, selfmonitor=$selfmonitor, stdcheck=$stdcheck")
                                        push!(kpidf, kpi(subdf, asset, rw, longshort, trendminutes, gainthreshold, gap, selfmonitor, stdcheck))
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

function bestcombi(tradedf, asset, comparewindow, longshort, gainthreshold, gap, selfmonitor)
    # println("bestcombi: $asset, $longshort, $gainthreshold, $gap, $selfmonitor")
    rwset = sort(unique(tradedf[!, :regr]), rev=true)
    bestgain = Dict()  # with openix as key
    tdf = sort(tradedf, :closeix, view=true)
    dfopenix = sort(tradedf.openix)
    lastoix = first(dfopenix) - 1
    for rw in rwset
        rtdf = subset(tdf, :regr => x -> x .== rw, view=true)
        dfgain = rtdf.gain
        dfcloseix = rtdf.closeix
        six = eix = firstindex(dfcloseix)
        for oix in eachindex(dfopenix)
            if (oix > firstindex(dfopenix)) && (dfopenix[oix] == dfopenix[oix-1])
                continue  # gain for this openix was already generated
            end
            while (eix < lastindex(dfcloseix)) && (dfopenix[eix+1] < dfopenix[oix])
                eix += 1
            end
            while dfcloseix[six] < (dfcloseix[eix] - comparewindow)
                six += 1
            end
            rwgain = sum(dfgain[six:eix])
            if (lastoix != dfopenix[oix]) || (bestgain[dfopenix[oix]].gain < rwgain)
                bestgain[dfopenix[oix]] = (rw=rw, gain=rwgain)
                lastoix = dfopenix[oix]
            end
        end
    end
    combitrades = DataFrame()
    tdf = sort(tradedf, :openix, view=true)
    dfopenix = tradedf.openix
    oix = first(dfopenix) - 1
    for dix in eachindex(dfopenix)
        if oix != dfopenix[dix]
            oix = dfopenix[dix]
            bestrw = bestgain[oix].rw
        end
        if tdf[dix, :regr] == bestrw
            push!(combitrades, tdf[dix, :])
        end
    end
    return kpi(combitrades, asset, 0, longshort, gainthreshold, gap, selfmonitor)
end

function assesscombi!(kpidf, tradedf, assets, comparewindows, gainthresholds, gaps, selfmonitorset, longshortset)
    for asset in assets
        for gainthreshold in gainthresholds
            for gap in gaps
                for selfmonitor in selfmonitorset
                    for longshort in longshortset
                        for comparewindow in comparewindows
                            # subdf = filter(row -> (row.asset == asset) && (row.regr == rw) && (row.longshort == longshort) && (row.gainthreshold == gainthreshold) && (row.gap==gap) && (row.selfmonitor==selfmonitor), tradedf, view=true)
                            subdf = subset(tradedf, :asset => x -> x .== asset, :longshort => x -> x .== longshort, :gainthreshold => x -> x .== gainthreshold, :gap => x -> x .== gap, :selfmonitor => x -> x .== selfmonitor, view=true)
                            push!(kpidf, bestcombi(subdf, asset, comparewindow, longshort, gainthreshold, gap, selfmonitor))
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
    EnvConfig.checkbackup(tradefilename)
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
    gainthresholds = [0.005, 0.01, 0.02]  # [0.02]  #
    gaps = [2]
    selfmonitorset = [-Inf]  # [-Inf, 0.0]
    longshortset = ["long"]  # , "short"]
    trendrwfactors = [0]  # , 1, 2, 4]
    stdcheckset = [true, false]
    for base in bases
        if !(base in loadedbases)
            ohlcv = Ohlcv.defaultohlcv(base)
            Ohlcv.read!(ohlcv)
            if !isnothing(startdt) && !isnothing(period)
                enddt = startdt + period
                subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
            end
            println("loaded $ohlcv")
            tdf = trackohlc(ohlcv, rkeys, trendrwfactors, gainthresholds, gaps, selfmonitorset, stdcheckset)
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
    calckpi!(kpidf, tradedf, bases, rkeys, trendrwfactors, gainthresholds, gaps, selfmonitorset, stdcheckset, longshortset)
    comparewindows = [60 * 24]  # minutes
    # assesscombi!(kpidf, tradedf, bases, comparewindows, gainthresholds, gaps, selfmonitorset, longshortset)
    println(kpidf)
    kpifilename = EnvConfig.logpath(VOLATILITYSTUDYKPIFILE)
    EnvConfig.checkbackup(kpifilename)
    CSV.write(kpifilename, kpidf, decimal=',', delim=';')  # decimal as , to consume with European locale
    println("done")
end

end  #of module

using EnvConfig, Dates
# startdt = DateTime("2022-05-15T22:54:00")
enddt = DateTime("2024-02-18T13:29:00")
period = Month(12)
startdt = enddt - period
EnvConfig.init(production)
# EnvConfig.setlogpath("230802-240107_VolatilityTracker")
EnvConfig.setlogpath("2407-7_TrendawareVolatilityTracker")
# VolatilityTracker.trackasset(["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"])
# VolatilityTracker.trackasset(["BTC", "ETC", "XRP", "GMT", "PEOPLE", "SOL", "APEX", "MATIC", "OMG"], DateTime("2023-08-02T22:54:00"), Dates.Day(400))
VolatilityTracker.trackasset(["MATIC", "BTC"], startdt, period)
