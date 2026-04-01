using Dates, DataFrames  # , Logging, LoggingFacilities, NamedArrays
using Test
using Logging, LoggingExtras
using EnvConfig, Features, Targets, TestOhlcv, Ohlcv

# with_logger(TimestampTransformerLogger(current_logger(), BeginningMessageLocation();
#                                               format = "yyyy-mm-dd HH:MM:SSz")) do
Targets.verbosity = 1

EnvConfig.init(test)
# EnvConfig.init(production)
println("\nconfig mode = $(EnvConfig.configmode)")

function minitrenddisturbance!(ohlcv, startix, minutes, amplitude, delta)
    for ix in startix:(startix+minutes-1)
        ohlcv.df[ix, :pivot] = ohlcv.df[startix, :pivot] * (1 + amplitude + (ix-1) * delta)
    end
end

function noshorttrends(trd)
    lbl = trd.df[1, :label]
    startix = 1
    for ix in eachindex(trd.df[!, :label])
        if lbl != trd.df.label[ix]
            if (lbl in [longbuy, shortbuy]) && ((ix-startix) < trd.minwindow)
                if !((startix == 40) && (ix == 42))
                    (Targets.verbosity >= 3) && println("too small $(string(lbl)) trend detected at $startix:$(ix-1)")
                    return false
                end
            end
            startix = ix
            lbl = trd.df[ix, :label]
        end
    end
    return true
end

@testset "Targets::Trend01 tests" begin
    startdt = DateTime("2025-02-17T13:30:00")
    enddt = startdt + Hour(6)
    # EnvConfig.init(production)
    ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt, "1m")
    Ohlcv.timerangecut!(ohlcv, startdt, enddt)
    # println(describe(ohlcv.df, :all))

    # trd = Targets.Trend01(30, Targets.defaultlabelthresholds)
    thres = Targets.LabelThresholds(longbuy=0.11, longhold=0.01, shorthold=-0.01, shortbuy=-0.11)
    trd = Targets.Trend01(3, 30, thres)
    minitrenddisturbance!(ohlcv, 100, 4, 0.9 * thres.shortbuy, 0.01 * thres.shortbuy)
    minitrenddisturbance!(ohlcv, 110, 4, 1.1 * thres.shortbuy, 0.01 * thres.shortbuy)
    minitrenddisturbance!(ohlcv, 40, 2, 0.9 * thres.longbuy, 0.01 * thres.longbuy)
    minitrenddisturbance!(ohlcv, 150, 4, 1.1 * thres.longbuy, 0.01 * thres.longbuy)
    Targets.setbase!(trd, ohlcv)

    # println("trade labels: $(Targets.uniquelabels(trd))")
    # println(Targets.df(trd, DateTime("2023-02-17T13:31:00"), DateTime("2023-02-17T13:39:00")))
    # println(trd.df)
    # println(describe(trd.df, :all))

    # println("baseline trd")
    # println(Targets.relativegain(trd))
    # println(Targets.labels(trd))


    ixcheck = [firstindex(ohlcv.df[!, :opentime]) <= trd.df[ix, :relix] <= ix for ix in eachindex(trd.df[!,:relix])]
    gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :relix], :pivot]) / ohlcv.df[trd.df[ix, :relix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    if (Targets.verbosity >= 3)
        tmp2labels = :tmp2label in names(trd.df) ? trd.df[!, :tmp2label] : fill(allclose, size(trd.df, 1))
        tmprelix = :tmprelix in names(trd.df) ? trd.df[!, :tmprelix] : copy(trd.df[!, :relix])
        tmpreldiff = :tmpreldiff in names(trd.df) ? trd.df[!, :tmpreldiff] : copy(trd.df[!, :reldiff])
        tmplabels = :tmplabel in names(trd.df) ? trd.df[!, :tmplabel] : copy(trd.df[!, :label])
        df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, relix=trd.df.relix, reldiff=trd.df.reldiff, 
            labels=Targets.labels(trd), tmp2labels=tmp2labels, tmprelix=tmprelix, tmpreldiff=tmpreldiff, tmplabels=tmplabels, ixcheck=ixcheck, 
            relativegain=Targets.relativegain(trd), gaincheck=gaincheck, gainchecktest=(gaincheck .== Targets.relativegain(trd)),
            longbuybinarytargets=Targets.labelbinarytargets(trd, longbuy), longbuyrelativegain=Targets.labelrelativegain(trd, longbuy), 
            shortbuybinarytargets=Targets.labelbinarytargets(trd, shortbuy), shortbuyrelativegain=Targets.labelrelativegain(trd, shortbuy)
            ))
        println(df)
    end
    @test noshorttrends(trd)
    @test all(ixcheck)
    @test all(gaincheck .== Targets.relativegain(trd))

    # cut time range at start and end
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt-Minute(5))
    Targets.timerangecut!(trd)
    # println("timerangecut $trd")
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, maxix=trd.df.maxix, relativegain=Targets.relativegain(trd), labels=Targets.labels(trd)))
    # println(df)
    # ixcheck = [ix - trd.window < trd.df[ix, :minix] <= ix for ix in eachindex(trd.df[!,:maxix])]
    # @test all(ixcheck)
    # gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :maxix], :pivot]) / ohlcv.df[trd.df[ix, :maxix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    # @test all(gaincheck .== Targets.relativegain(trd))

    # extend 5 minutes
    ohlcvxt = TestOhlcv.testohlcv("SINE", startdt+Minute(5), enddt+Minute(5), "1m")
    ohlcv.df = ohlcvxt.df
    Ohlcv.timerangecut!(ohlcv, startdt+Minute(5), enddt+Minute(5))
    Targets.supplement!(trd)
    # ixcheck = [ix <= trd.df[ix, :maxix] <= ix + trd.window for ix in eachindex(trd.df[!, :maxix])]
    # @test all(ixcheck)
    # gaincheck = [(ohlcv.df[ix, :pivot] .- ohlcv.df[trd.df[ix, :maxix], :pivot]) / ohlcv.df[trd.df[ix, :maxix], :pivot] for ix in eachindex(ohlcv.df[!, :opentime])]
    # @test all(gaincheck .== Targets.relativegain(trd))
    # df = DataFrame((opentime=ohlcv.df.opentime, pivot=ohlcv.df.pivot, targettime=trd.df.opentime, maxix=trd.df.maxix, relativegain=Targets.relativegain(trd), labels=Targets.labels(trd), gaincheck=gaincheck, equal=(gaincheck .== Targets.relativegain(trd))))
    # println(df)

    @test length(Targets.relativegain(trd)) > 0
    Targets.removebase!(trd)
    @test length(Targets.relativegain(trd)) == 0
    ret="end"
end  # of testset

return 