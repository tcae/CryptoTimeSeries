module liquidcoinstest

using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, TestOhlcv, DataFrames

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 2
Ohlcv.verbosity = 1
EnvConfig.init(production)

minquotevol=15000f0
accumulate=5
checkperiod=24*60
startthreshold=0.01
stopthreshold=0.25
minliquidminutes=24*60*10
startdistance=24*60*10

function checkconstructedcase(minquotevol, accumulate, checkperiod, startthreshold, stopthreshold, minliquidminutes, startdistance)
    longenough = 5 * max(checkperiod, minliquidminutes, startdistance)
    p = fill(minquotevol/10f0, longenough)
    v = fill(11f0, longenough)
    ot = [DateTime("2023-02-17T13:30:00") + Minute(ix) for ix in 1:longenough]
    startnok = round(Int, checkperiod * startthreshold)
    stopnok = round(Int, checkperiod * stopthreshold)
    testnok1 = 1:100
    v[testnok1] .= 0.1f0 #* test late range start

    testnok2 = checkperiod + testnok1[end] + 1 : checkperiod + testnok1[end] + 1 + stopnok + accumulate - 1
    v[testnok2] .= 0.1f0 #* test gap range in the middle
    (verbosity >= 3) && println("v[$testnok2] = $(v[testnok2]) range=$(testnok2[end] - testnok2[begin] + 1))")
    expectrange1 = (testnok1[end] + 1 + checkperiod - startnok):(testnok2[end] - 1)
    expectrange2 = (testnok2[end] + 1 + checkperiod - startnok):longenough

    #TODO test gap before end and before minliquidminutes is reached

    odf = DataFrame(opentime=ot, open=p, high=p, low=p, close=p, pivot=p, basevolume=v)
    ohlcv = Ohlcv.defaultohlcv("VOLTEST")
    Ohlcv.setdataframe!(ohlcv, odf)
    (verbosity >= 3) && println("expecting startof first range at testnok1[end]=$(testnok1[end]) + 1 + checkperiod=$(checkperiod) - startnok=$(startnok) = $(testnok1[end] + 1 + checkperiod - startnok) until testnok2[end] = $(testnok2[end])")
    (verbosity >= 3) && println("expecting startof second range at testnok2[end]=$(testnok2[end]) + 1 + checkperiod=$(checkperiod) - startnok=$(startnok) = $(testnok2[end] + 1 + checkperiod - startnok) until longenough=$longenough")
    (verbosity >= 3) && println("checkperiod=$checkperiod, testnok1=$testnok1, startnok=$startnok, stopnok=$stopnok, testnok2[begin]=$(testnok2[begin]), testnok2[end]=$(testnok2[end]) $(ohlcv) $(describe(Ohlcv.dataframe(ohlcv)))")

    rv = Ohlcv.liquiditycheck(ohlcv; minquotevol=minquotevol, accumulate=accumulate, checkperiod=checkperiod, startthreshold=startthreshold, stopthreshold=stopthreshold, minliquidminutes=minliquidminutes, startdistance)
    if (verbosity >= 3)
        println("liquid coin test of $(ohlcv.base), length=$(size(Ohlcv.dataframe(ohlcv), 1))")
        ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
        for ix in rv
            period = ot[ix[end]] - ot[ix[begin]]
            days = round(period, Day)
            # hours = round(period-days, Hour)
            # minutes = round(period - days - hours, Minute)
            println("$(ohlcv.base): range = $ix, $days, $(ot[ix[begin]]) - $(ot[ix[end]])")
        end
    end
    @test length(rv) > 0
    if length(rv) > 1
        @test expectrange1[begin] == rv[1][begin]
        @test expectrange1[end] == rv[1][end]
        @test expectrange2[begin] == rv[2][begin]
        @test expectrange2[end] == rv[2][end]
    elseif length(rv) > 0
        @test expectrange2[begin] == rv[1][begin]
        @test expectrange2[end] == rv[1][end]
    end
end

function getbtc()
    # startdt = DateTime("2023-02-17T13:30:00")
    # enddt = startdt + Hour(20) - Minute(1) 
    ohlcv = Ohlcv.defaultohlcv("BTC")
    if Ohlcv.file(ohlcv).existing
        Ohlcv.read!(ohlcv)
    end
    return ohlcv
end

function liquidcheck(ohlcv, minliquidminutes)
    rv = Ohlcv.liquiditycheck(ohlcv; minquotevol=minquotevol, accumulate=accumulate, checkperiod=checkperiod, startthreshold=startthreshold, stopthreshold=stopthreshold, minliquidminutes=minliquidminutes)
    (verbosity >= 2) && println("liquid coin test of $(ohlcv.base), length=$(size(Ohlcv.dataframe(ohlcv), 1))")
    ot = Ohlcv.dataframe(ohlcv)[!, :opentime]
    for ix in rv
        period = ot[ix[end]] - ot[ix[begin]]
        days = round(period, Day)
        # hours = round(period-days, Hour)
        # minutes = round(period - days - hours, Minute)
        (verbosity >= 2) && println("$(ohlcv.base): range = $ix, $days, $(ot[ix[begin]]) - $(ot[ix[end]])")
    end
end

@testset "liquid coins tests" begin
    minliquidminutes = 1
    startdistance=0
    checkconstructedcase(minquotevol, accumulate, checkperiod, startthreshold, stopthreshold, minliquidminutes, startdistance)

    minliquidminutes = checkperiod
    startdistance=0
    checkconstructedcase(minquotevol, accumulate, checkperiod, startthreshold, stopthreshold, minliquidminutes, startdistance)

    minliquidminutes=24*60*10
        # ohlcv = getbtc()
    # liquidcheck(ohlcv)
    count = 0
    for ohlcv in Ohlcv.OhlcvFiles()
        rv = Ohlcv.liquiditycheck(ohlcv)
        println("$count: liquid coin test of $(ohlcv.base), length=$(size(Ohlcv.dataframe(ohlcv), 1)), length(rv)=$(length(rv)), rv=$rv")
        # liquidcheck(ohlcv, minliquidminutes)
        # break
    end

end # testset
end