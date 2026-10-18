using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, Targets, TestOhlcv

@testset "TargetStore tests" begin
    EnvConfig.init(test)
    oldcoinspath = EnvConfig.activecoinspath
    testfolder = joinpath(EnvConfig.debugfilesfolder, "targetstore_test")
    rm(normpath(joinpath(dirname(EnvConfig.defaultlogfilespath), testfolder)); force=true, recursive=true)
    EnvConfig.setcoinspath!(testfolder)
    try
        startdt = DateTime("2025-02-17T13:30:00")
        enddt = startdt + Hour(20) - Minute(1)
        ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
        odf = Ohlcv.dataframe(ohlcv)
        thres = Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01))

        # Trend04 store equals direct label calculation except the incomplete look-ahead tail
        trd = Targets.Trend04(10, 60, thres)
        tdf = Targets.updatetargets!(ohlcv, "TTEST04", Targets.Trend04(10, 60, thres); confighash="h04")
        Targets.setbase!(trd, ohlcv)
        direct = Targets.labels(trd, odf[begin, :opentime], odf[end, :opentime])
        complete = size(odf, 1) - Targets.lookaheadminutes(trd)
        @test tdf[!, :opentime] == odf[!, :opentime]
        @test tdf[1:complete, :label] == Targets.tradelabelcode.(direct[1:complete])
        @test all(tdf[(complete + 1):end, :label] .== Targets.tradelabelcode(Targets.ignore))
        @test EnvConfig.tablemetadata("TTEST04"; folderpath=Targets.targetsfolder("SINE", ohlcv.quotecoin))["labels"] == join(string.(Targets.uniquelabels(trd)), ",")
        @test_throws AssertionError Targets.readtargets("SINE", ohlcv.quotecoin, "TTEST04"; confighash="other")
        @test isnothing(Targets.readtargets("SINE", ohlcv.quotecoin, "TMISSING"; confighash="h04"))

        # TrendRegression is calculated from its own regression features
        trr = Targets.TrendRegression(15, 0.005, -0.005)
        rdf = Targets.updatetargets!(ohlcv, "TTESTRR", trr; confighash="hrr")
        @test size(rdf, 1) == size(odf, 1)
        @test Set(Targets.TradeLabel.(rdf[!, :label])) ⊆ Set(vcat(Targets.uniquelabels(trr), [Targets.ignore]))
        @test isnothing(trr.f6)

        # extending a store with new OHLCV rows equals a calculation from scratch
        shortohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt - Hour(5))
        t05factory = () -> Targets.Trend05(leadregr=5, supportregr=15, trendregr=60, triggerdist=0.005f0, targetgain=0.005f0)
        t05df = Targets.updatetargets!(ohlcv, "TTEST05", t05factory(); confighash="h05")
        t05labels = Targets.TradeLabel.(t05df[!, :label])
        @test any(==(Targets.longopen), t05labels) && any(==(Targets.shortopen), t05labels)
        @test t05labels[end] == Targets.ignore
        for (id, trdfactory, fullid, fullhash) in (("TEXT04", () -> Targets.Trend04(10, 60, thres), "TTEST04", "h04"), ("TEXTRR", () -> Targets.TrendRegression(15, 0.005, -0.005), "TTESTRR", "hrr"), ("TEXT05", t05factory, "TTEST05", "h05"))
            Targets.updatetargets!(shortohlcv, id, trdfactory(); confighash="hext")
            extdf = Targets.updatetargets!(ohlcv, id, trdfactory(); confighash="hext")
            fulldf = Targets.readtargets("SINE", ohlcv.quotecoin, fullid; confighash=fullhash)
            @test extdf[!, :opentime] == fulldf[!, :opentime]
            @test extdf[!, :label] == fulldf[!, :label]
        end
        # minute by minute extension as in live trading
        liveohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt - Minute(30))
        Targets.updatetargets!(liveohlcv, "TLIVE05", t05factory(); confighash="hlive")
        for minutes in 29:-1:0
            Targets.updatetargets!(TestOhlcv.testohlcv("SINE", startdt, enddt - Minute(minutes)), "TLIVE05", t05factory(); confighash="hlive")
        end
        @test Targets.readtargets("SINE", ohlcv.quotecoin, "TLIVE05"; confighash="hlive")[!, :label] == t05df[!, :label]

        @test Targets.configdescription(Targets.Trend04(10, 60, thres)) != Targets.configdescription(Targets.Trend04(10, 120, thres))
        @test Targets.configdescription(Targets.TrendRegression(15, 0.005, -0.005)) == Targets.configdescription(trr)
        @test Targets.regressionfeatures(trd) === nothing
        @test Features.requestedcolumns(Targets.regressionfeatures(Targets.Trend05())) == ["rg+5+0", "rg+15+0", "rg+240+0"]
    finally
        EnvConfig.setcoinspath!(relpath(oldcoinspath, dirname(EnvConfig.defaultlogfilespath)))
    end
end
