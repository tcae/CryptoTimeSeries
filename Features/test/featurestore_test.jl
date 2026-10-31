using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, Features, TestOhlcv

function featurestoretestf6()
    f6 = Features.Features006()
    Features.addgrad!(f6, window=15, offset=0)
    Features.addnormgrad!(f6, window=15, offset=3)
    Features.addstd!(f6, window=10, offset=5)
    Features.addregry!(f6, window=60, offset=0)
    Features.addmaxdist!(f6, window=30, offset=0)
    Features.addrelvol!(f6, short=5, long=60)
    return f6
end

@testset "internal strategy regression dependencies" begin
    EnvConfig.init(test)
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        oldpath = EnvConfig.coinspath()
        EnvConfig.setcoinspath!(folder)
        try
            startdt = DateTime("2025-02-17T13:30:00")
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(120))
            baseline = featurestoretestf6()
            augmented = featurestoretestf6()
            cols = Features.requestedcolumns(baseline)
            description = Features.configdescription(baseline)
            count = Features.featurecount(baseline)
            required = copy(augmented.required)
            Features.require_regression!(augmented; window=15)
            @test augmented.required == required
            Features.require_regression!(augmented; window=180)
            dependencies = copy(augmented.required)
            Features.require_regression!(augmented; window=180)
            @test augmented.required == dependencies
            @test Features.requiredminutes(augmented) == 180
            @test augmented.maxoffset == baseline.maxoffset
            @test Features.requestedcolumns(augmented) == cols
            @test Features.featurecount(augmented) == count
            @test Features.configdescription(augmented) == description
            @test_throws AssertionError Features.require_regression!(augmented; window=1)
            @test_throws AssertionError Features.require_regression!(augmented; window=14401)
            ry, rg = Features.rollingregression(Ohlcv.dataframe(ohlcv).pivot, 180)
            Features.setbase!(baseline, ohlcv; usecache=false)
            Features.setbase!(augmented, ohlcv; usecache=true)
            @test Features.features(augmented) ≈ Features.features(baseline)
            @test augmented.fdfno[!, "ry+180"] ≈ ry
            @test augmented.fdfno[!, "rg+180"] ≈ rg
            @test isnothing(Features.readregression("SINE", ohlcv.quotecoin, 180))
            @test_throws AssertionError Features.require_regression!(augmented; window=240)

            Features.updateregression!(ohlcv, 180)
            path = joinpath(Features.regressionfolder("SINE", ohlcv.quotecoin), "regr180.arrow")
            bytes = read(path)
            timestamp = stat(path).mtime
            cached = featurestoretestf6()
            Features.require_regression!(cached; window=180)
            Features.setbase!(cached, Ohlcv.ohlcvview(ohlcv, 60:100); usecache=true)
            @test Features.regressionat(cached, 180, Ohlcv.dataframe(ohlcv)[100, :opentime]).regry ≈ ry[100]
            @test read(path) == bytes
            @test Features.regressionat(augmented, 180, startdt + Minute(120)).regry ≈ ry[end]

            extended = TestOhlcv.testohlcv("SINE", startdt, startdt + Minute(125))
            Ohlcv.setdataframe!(ohlcv, DataFrame(Ohlcv.dataframe(extended)))
            Features.supplement!(augmented)
            @test read(path) == bytes
            @test stat(path).mtime == timestamp
            @test Features.regressionat(augmented, 180, startdt + Minute(125)) isa NamedTuple
            @test nrow(Features.readregression("SINE", ohlcv.quotecoin, 180)) < nrow(Ohlcv.dataframe(ohlcv))
        finally
            EnvConfig.setcoinspath!(oldpath)
        end
    end
end

@testset "FeatureStore tests" begin
    EnvConfig.init(test)
    oldcoinspath = EnvConfig.activecoinspath
    testfolder = joinpath(EnvConfig.debugfilesfolder, "featurestore_test")
    rm(normpath(joinpath(dirname(EnvConfig.defaultlogfilespath), testfolder)); force=true, recursive=true)
    EnvConfig.setcoinspath!(testfolder)
    try
        startdt = DateTime("2025-02-17T13:30:00")
        enddt = startdt + Hour(20) - Minute(1)
        ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
        odf = Ohlcv.dataframe(ohlcv)

        # regression store equals direct calculation and incremental extension equals full calculation
        shortohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt - Hour(2))
        Features.updateregression!(shortohlcv, 15; std=true)
        @test size(Features.readregression("SINE", ohlcv.quotecoin, 15), 1) == size(Ohlcv.dataframe(shortohlcv), 1)
        Features.updateregression!(ohlcv, 15; std=true)
        regr = Features.readregression("SINE", ohlcv.quotecoin, 15)
        regrstd = Features.readregression("SINE", ohlcv.quotecoin, 15; std=true)
        ry, rg = Features.rollingregression(odf[!, :pivot], 15)
        rs = Features.rollingregressionstdmv([odf[!, :open], odf[!, :high], odf[!, :low], odf[!, :close]], ry, rg, 15)
        @test regr[!, :opentime] == odf[!, :opentime]
        @test regr[!, :regry] ≈ ry
        @test regr[!, :grad] ≈ rg
        @test regrstd[!, :std] ≈ rs
        @test isnothing(Features.readregression("SINE", ohlcv.quotecoin, 77))

        # Features006 with regression store equals Features006 without cache
        f6nocache = featurestoretestf6()
        Features.setbase!(f6nocache, ohlcv, usecache=false)
        for window in first(Features.regressionwindows(f6nocache))
            Features.updateregression!(ohlcv, window; std=true)
        end
        f6cache = featurestoretestf6()
        Features.setbase!(f6cache, ohlcv, usecache=true)
        @test f6cache.fdf[!, :opentime] == f6nocache.fdf[!, :opentime]
        for col in Features.requestedcolumns(f6nocache)
            @test f6cache.fdf[!, col] ≈ f6nocache.fdf[!, col]
        end

        # feature store drops warm-up rows and keeps requested column order
        f6 = featurestoretestf6()
        fdf = Features.updatefeatures!(ohlcv, "FTEST", f6; confighash="hash1")
        @test names(fdf) == vcat(["opentime"], Features.requestedcolumns(f6))
        @test size(fdf, 1) == size(odf, 1) - Features.requiredminutes(f6) + 1
        @test fdf[begin, :opentime] == odf[Features.requiredminutes(f6), :opentime]
        @test fdf[end, :opentime] == odf[end, :opentime]
        rowoffset = size(f6nocache.fdf, 1) - size(fdf, 1)
        for col in Features.requestedcolumns(f6)
            @test fdf[!, col] ≈ f6nocache.fdf[(rowoffset + 1):end, col]
        end
        @test Features.configdescription(f6) == "Features006(" * join(Features.requestedcolumns(f6), ",") * ";rv=positive-median-if-zero,cap60;nrg=rg/meanpivot)"
        @test size(Features.readfeatures("SINE", ohlcv.quotecoin, "FTEST"; confighash="hash1"), 1) == size(fdf, 1)
        @test_throws AssertionError Features.readfeatures("SINE", ohlcv.quotecoin, "FTEST"; confighash="hash2")
        @test isnothing(Features.readfeatures("SINE", ohlcv.quotecoin, "FMISSING"; confighash="hash1"))
        @test_throws AssertionError Features.updatefeatures!(Ohlcv.ohlcvview(ohlcv, 1:100), "FVIEW", featurestoretestf6(); confighash="hash1")

        # extending a store with new OHLCV rows equals a calculation from scratch
        Features.updatefeatures!(TestOhlcv.testohlcv("SINE", startdt, enddt - Hour(3)), "FEXT", featurestoretestf6(); confighash="hash1")
        extdf = Features.updatefeatures!(ohlcv, "FEXT", featurestoretestf6(); confighash="hash1")
        @test extdf[!, :opentime] == fdf[!, :opentime]
        for col in Features.requestedcolumns(f6)
            @test extdf[!, col] ≈ fdf[!, col]
        end

        # live trading extends Features006 minute by minute; the result equals a calculation from scratch
        n0 = size(odf, 1) - 300
        live = Ohlcv.defaultohlcv("SINE")
        Ohlcv.setdataframe!(live, DataFrame(odf[1:n0, :]))
        f6live = featurestoretestf6()
        Features.setbase!(f6live, live, usecache=false)
        for k in (n0 + 1):size(odf, 1)
            Ohlcv.setdataframe!(live, DataFrame(odf[1:k, :]))
            Features.supplement!(f6live)
        end
        for col in Features.requestedcolumns(f6live)
            @test f6live.fdf[!, col] ≈ f6nocache.fdf[!, col]
        end
    finally
        EnvConfig.setcoinspath!(relpath(oldcoinspath, dirname(EnvConfig.defaultlogfilespath)))
    end
end
