module Jdf2ArrowTests

using Test, Dates, DataFrames, EnvConfig, Ohlcv
include("../scripts/Jdf2Arrow.jl")

@testset "legacy OHLCV conversion to Bybit" begin
    @test Jdf2Arrow.DEFAULT_SHARED_ARTIFACTS == ["ohlcv"]
    @test_throws AssertionError Jdf2Arrow.convert_shared_data(; artifacts=["unsupported"])
    @test_throws AssertionError Jdf2Arrow.crosscheck_shared_data(; artifacts=["unsupported"])
    oldpath = EnvConfig.activecoinspath
    oldlogs = EnvConfig.defaultlogfilespath
    oldtrading = EnvConfig.tradingfolder
    oldmode = EnvConfig.configmode
    olddata = EnvConfig.datafolder
    oldauth = EnvConfig.authorization
    oldbases = EnvConfig.bases
    oldtrainingbases = EnvConfig.trainingbases
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        try
            EnvConfig.defaultlogfilespath = joinpath(folder, "logs")
            EnvConfig.tradingfolder = folder
            EnvConfig.setcoinspath!("KrakenSpot")
            sourcefolder = joinpath(folder, "Features", "OHLCV")
            startdt = DateTime("2025-02-01")
            df = DataFrame(opentime=[startdt, startdt+Minute(1)],
                open=Float32[1, 2], high=Float32[2, 3], low=Float32[0.5, 1],
                close=Float32[1.5, 2.5], basevolume=Float32[10, 20])
            source = EnvConfig.savedf(df, "SINE_USD_1m"; folderpath=sourcefolder, format=:jdf)
            summary = Jdf2Arrow.convert_shared_data(; bases=["SINE"], savesummary=false)
            @test EnvConfig.coinspath() == joinpath(folder, "Bybit")
            @test summary.status == ["converted"]
            target = joinpath(folder, "Bybit", "SINE-USD", "ohlcv.arrow")
            @test summary.output == [target]
            @test EnvConfig.readtable("ohlcv"; folderpath=dirname(target), format=:arrow) == df
            @test isdir(source)
            bytes = read(target)
            @test_logs (:error, "legacy OHLCV conversion failed") begin
                repeated = Jdf2Arrow.convert_shared_data(; bases=["SINE"], savesummary=false)
                @test repeated.status == ["error"]
            end
            @test read(target) == bytes
            EnvConfig.setcoinspath!("KrakenSpot")
            checked = Jdf2Arrow.crosscheck_shared_data(; bases=["SINE"], savesummary=false)
            @test checked.output == [target]
            @test checked.counterpart_exists == [true]
            @test !ispath(joinpath(folder, "coins"))
        finally
            EnvConfig.activecoinspath = oldpath
            EnvConfig.defaultlogfilespath = oldlogs
            EnvConfig.tradingfolder = oldtrading
            EnvConfig.configmode = oldmode
            EnvConfig.datafolder = olddata
            EnvConfig.authorization = oldauth
            EnvConfig.bases = oldbases
            EnvConfig.trainingbases = oldtrainingbases
        end
    end
end

end
