
module VolatilityTracker

using DataFrames, Dates, Statistics, CSV, Logging, JDF
using EnvConfig, TestOhlcv, Ohlcv, Features

function trackregression!(tradedf, f2::Features.Features002; asset, gainthreshold, regrwindow, gap, selfmonitor)
    longlastok = shortlastok = true
    piv = Ohlcv.dataframe(Features.ohlcv(f2)).pivot[f2.firstix:f2.lastix]
    longopenix = Int32[]
    shortopenix = Int32[]
    # gap = 5  # 5 minute gap between trades
    longcloseix = -gap
    shortcloseix = -gap
    ctrades = otrades = otok = ootok = cltok = cstok = 0

    function closetrades!(openix, closeix, longshort, handleall)
        ctrades += 1
        while length(openix) > 0
            # handle open long positions
            tl = closeix - openix[begin]
            if longshort == "long"
                gain = Ohlcv.relativegain(piv, openix[begin], closeix)
                longlastok = (gain >= selfmonitor)
                handleall = longlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=closeix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, selfmonitor=selfmonitor))
                cltok += 1
            else
                gain = -Ohlcv.relativegain(piv, openix[begin], closeix)
                shortlastok = (gain >= selfmonitor)
                handleall = shortlastok ? handleall : true
                push!(tradedf, (asset=asset, regr=regrwindow, longshort=longshort, openix=openix[begin], closeix=closeix, tradelen=tl, gain=gain, gainthreshold=gainthreshold, gap=gap, selfmonitor=selfmonitor))
                cstok += 1
            end
            openix = deleteat!(openix, 1)
            if !handleall
                break
            end
        end
        return openix
    end

    function opentrades!(openix, closeix)
        otrades += 1
        if ((f2.regr[regrwindow].std[closeix] / f2.regr[regrwindow].regry[closeix]) > gainthreshold)
            if ((length(openix) == 0) || ((closeix - openix[end]) >= gap))
                push!(openix, closeix)
                ootok += 1
            end
            otok += 1
        end
        return openix
    end

    #* idea: adapt gap with direction gradient
    for ix in eachindex(piv)
        absgradgain = f2.regr[regrwindow].regry[ix] - Features.startregry(f2.regr[regrwindow].regry[ix], f2.regr[regrwindow].grad[ix], regrwindow)
        # if piv[ix] > f2.regr[regrwindow].regry[ix] + (f2.regr[regrwindow].std[ix] > absgradgain ? f2.regr[regrwindow].std[ix] : absgradgain)
        if piv[ix] > f2.regr[regrwindow].regry[ix] + f2.regr[regrwindow].std[ix] + absgradgain
            if (ix - longcloseix) >= gap
                longopenix = closetrades!(longopenix, ix, "long", false)
                longcloseix = ix
            end
            if shortlastok
                opentrades!(shortopenix, ix)
            end
        # elseif piv[ix] < f2.regr[regrwindow].regry[ix] + (-f2.regr[regrwindow].std[ix] < absgradgain ? -f2.regr[regrwindow].std[ix] : absgradgain)
        elseif piv[ix] < f2.regr[regrwindow].regry[ix] - f2.regr[regrwindow].std[ix] + absgradgain
            if (ix - shortcloseix) >= gap
                shortopenix = closetrades!(shortopenix, ix, "short", false)
                shortcloseix = ix
            end
            if longlastok
                opentrades!(longopenix, ix)
            end
        end
    end
    println("trackregression!: $asset, $regrwindow, $gainthreshold, $gap, $selfmonitor size(tradedf)=$(size(tradedf))")
    println("ctrades=$ctrades otrades=$otrades otok=$otok ootok=$ootok cltok=$cltok cstok=$cstok")
    return tradedf
end

function trackohlc(ohlcv::Ohlcv.OhlcvData, rkeys, gainthresholds, gaps, selfmonitorset)
    tdf = DataFrame()
    println("$(EnvConfig.now()) calculating F002 features")
    f2 = Features.Features002(ohlcv)
    asset = Ohlcv.basesymbol(Features.ohlcv(f2))
    for kix in eachindex(rkeys)
        rw = rkeys[kix]
        println("$(EnvConfig.now()) processing regression window $rw")
        # println(describe(DataFrame(grad=f2.regr[rw].grad, regry=f2.regr[rw].regry, std=f2.regr[rw].std), :all))
        if kix != firstindex(rkeys)
            for gainthreshold in gainthresholds
                for gap in gaps
                    for selfmonitor in selfmonitorset
                        # println("$(EnvConfig.now()) analyzing volatility performace of regression window $rw, gainthreshold=$gainthreshold, gap=$gap")
                        tdf = trackregression!(tdf, f2, asset=asset, gainthreshold=gainthreshold, regrwindow=rw, gap=gap, selfmonitor=selfmonitor)
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

function kpi(tradedf, asset, regr, longshort, gainthreshold, gap, selfmonitor)
    println("kpi: $asset, $regr, $longshort, $gainthreshold, $gap, $selfmonitor")
    println(describe(tradedf, :all))
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
        return (asset=asset, regr=regr, longshort=longshort, gap=gap, gainthreshold=gainthreshold*100, selfmonitor=selfmonitor, cumgain=cumgain, meangain=meangain, mediangain=mediangain, maxgain=maxgain, mingain=mingain, count=count, mediantradelen=mediantl, maxtradelen=maxtl)
    else
        return (asset=asset, regr=regr, longshort=longshort, gap=gap, gainthreshold=gainthreshold*100, selfmonitor=selfmonitor, cumgain=0.0, meangain=0.0, mediangain=0.0, maxgain=0.0, mingain=0.0, count=0, mediantradelen=0.0, maxtradelen=0.0)
    end
end

function calckpi!(kpidf, tradedf, assets, rkeys, gainthresholds, gaps, selfmonitorset, longshortset)
    for asset in assets
        for kix in eachindex(rkeys)
            rw = rkeys[kix]
            if kix != firstindex(rkeys)
                for gainthreshold in gainthresholds
                    for gap in gaps
                        for selfmonitor in selfmonitorset
                            for longshort in longshortset
                                # subdf = filter(row -> (row.asset == asset) && (row.regr == rw) && (row.longshort == longshort) && (row.gainthreshold == gainthreshold) && (row.gap==gap) && (row.selfmonitor==selfmonitor), tradedf, view=true)
                                subdf = subset(tradedf, :asset => x -> x .== asset, :regr => x -> x .== rw, :longshort => x -> x .== longshort, :gainthreshold => x -> x .== gainthreshold, :gap => x -> x .== gap, :selfmonitor => x -> x .== selfmonitor, view=true)
                                push!(kpidf, kpi(subdf, asset, rw, longshort, gainthreshold, gap, selfmonitor))
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
    println("bestcombi: $asset, $longshort, $gainthreshold, $gap, $selfmonitor")
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
        println("saved tradedf as $tradefilename with size $(size(tradedf, 1)) of assets $(unique(tradedf[!, :asset]))")
    catch e
        Logging.@warn "exception $e detected when saving $tradefilename with df size=$(size(tradedf))"
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
    rkeys = sort([rw for rw in Features.regressionwindows002])
    gainthresholds = [0.005, 0.01, 0.02]
    gaps = [2, 5]
    selfmonitorset = [-Inf, 0.0]
    longshortset = ["long"]  # ["long", "short"]
    for base in bases
        if !(base in loadedbases)
            ohlcv = Ohlcv.defaultohlcv(base)
            Ohlcv.read!(ohlcv)
            if !isnothing(startdt) && !isnothing(period)
                enddt = startdt + period
                subset!(ohlcv.df, :opentime => t -> startdt .<= t .<= enddt)
            end
            println("loaded $ohlcv")
            tdf = trackohlc(ohlcv, rkeys, gainthresholds, gaps, selfmonitorset)
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
    calckpi!(kpidf, tradedf, bases, rkeys, gainthresholds, gaps, selfmonitorset, longshortset)
    comparewindows = [60 * 24]  # minutes
    assesscombi!(kpidf, tradedf, bases, comparewindows, gainthresholds, gaps, selfmonitorset, longshortset)
    println(kpidf)
    kpifilename = EnvConfig.logpath(VOLATILITYSTUDYKPIFILE)
    EnvConfig.checkbackup(kpifilename)
    CSV.write(kpifilename, kpidf, decimal=',', delim=';')  # decimal as , to consume with European locale
    println("done")
end

end  #of module

using EnvConfig, Dates

EnvConfig.init(production)
EnvConfig.setlogpath("CombiVolatilityTracker")
# EnvConfig.setlogpath("TestVolatilityTracker")
VolatilityTracker.trackasset(["BTC", "ETC"])
# VolatilityTracker.trackasset(["BTC"], DateTime("2023-08-02T22:54:00"), Dates.Day(40))
