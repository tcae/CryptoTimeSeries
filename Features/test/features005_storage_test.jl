@testset "Features005 independent cache" begin
    @test !isdefined(Features, :Features004)
    oldpath = EnvConfig.activecoinspath
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        try
            EnvConfig.setcoinspath!(folder)
            config = Features.Features005(["rw_60_grad"])
            required = Features.requiredminutes(config)
            ohlcv = Ohlcv.defaultohlcv("SINE", "1m", required + 5)
            startdt = DateTime("2025-02-01")
            ohlcv.df.opentime = [startdt + Minute(ix-1) for ix in 1:nrow(ohlcv.df)]
            config.ohlcv = ohlcv
            df = Features.emptycachef005()
            for ix in required:nrow(ohlcv.df)
                push!(df, vcat([ohlcv.df[ix, :opentime]], fill(1f0, ncol(df)-1)))
            end
            config.fdf = df
            config.complete = true
            @test endswith(Features.mnemonic(config), "_F5")
            @test 60 in Features.regrwindows(config)
            @test !Features.file(config).existing
            Features.write(config)
            path = Features.file(config).filename
            @test basename(path) == "f5.arrow"
            @test Features.file(config).existing
            @test Features.read!(config) == df
            @test config.latestloadeddt == df[end, :opentime]

            shortened = Ohlcv.copy(ohlcv)
            Ohlcv.setdataframe!(shortened, DataFrame(ohlcv.df[1:end-2, :]))
            config.ohlcv = shortened
            @test Features.read!(config) == df[1:end-2, :]

            shifted = Ohlcv.copy(ohlcv)
            shifted.df.opentime .+= Day(1)
            config.ohlcv = shifted
            @test isempty(Features.read!(config))
            config.ohlcv = ohlcv
            broken = copy(df)
            broken[2, :opentime] += Minute(1)
            EnvConfig.savedf(broken, "f5"; folderpath=dirname(path), format=:arrow)
            @test_throws AssertionError Features.read!(config)
            Features.delete(config)
            @test !Features.file(config).existing
            @test isempty(Features.read!(config))

            unbased = Features.Features005(["rw_60_grad"])
            @test_throws AssertionError Features.file(unbased)
            @test_throws AssertionError Features.read!(unbased)
            @test_throws AssertionError Features.write(unbased)
        finally
            EnvConfig.activecoinspath = oldpath
        end
    end
end
