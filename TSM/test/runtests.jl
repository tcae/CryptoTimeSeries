using Test
using Dates
using DataFrames
using CategoricalArrays
using Targets
using TSM

@testset "TSM trades schema" begin
    df = DataFrame(opentime=[DateTime(2026, 1, 1)])
    TSM.ensuretradeschema!(df)

    @test :config in propertynames(df)
    @test :tsmstate in propertynames(df)
    @test df[!, :config] isa CategoricalVector
    @test df[!, :tsmstate] isa CategoricalVector
    @test String(df[1, :config]) == "none"
    @test String(df[1, :tsmstate]) == "none"

    TSM.settrades_config!(df, 1, "cfg-1")
    TSM.settrades_tsmstate!(df, 1, "armed")
    @test String(TSM.gettrades_config(df, 1)) == "cfg-1"
    @test String(TSM.gettrades_tsmstate(df, 1)) == "armed"

    TSM.settrades_label!(df, 1, longopen)
    @test TSM.gettrades_label(df, 1) == longopen
end

@testset "TSM normalizes legacy label column" begin
    df = DataFrame(
        opentime=[DateTime(2026, 1, 1), DateTime(2026, 1, 1, 0, 1)],
        label=categorical(["longbuy", "shortopen"]),
        close=Float32[1f0, 1f0],
        high=Float32[1f0, 1f0],
        low=Float32[1f0, 1f0],
    )

    tsm = TSM.TsmCache()
    TSM.settrades!(tsm, "BTCUSDT", df)
    tdf = TSM.trades(tsm, "BTCUSDT")

    @test eltype(tdf[!, :label]) == TradeLabel
    @test tdf[1, :label] == longopen
    @test tdf[2, :label] == shortopen

    TSM.settrades_label!(tdf, 1, shortclose)
    @test tdf[1, :label] == shortclose
end

include("trades_schema_contract_test.jl")
include("compilegainsdf_test.jl")