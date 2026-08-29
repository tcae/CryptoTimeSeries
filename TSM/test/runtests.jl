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

@testset "TSM uncompressed id categoricals" begin
    tsm = TSM.TsmCache()
    tdf = TSM.trades(tsm, "BTCUSDT")
    @test eltype(CategoricalArrays.refs(tdf[!, :lo_id])) == UInt32
    TSM.ensuretradesrow!(tsm, "BTC", "USDT", DateTime(2026, 1, 1))
    tdf = TSM.trades(tsm, "BTCUSDT")
    @test String(tdf[1, :tsmstate]) == "sync"
end

@testset "TSM checkpoint resume helpers" begin
    checkpoint = DataFrame(
        opentime=[DateTime(2026, 1, 1) + Minute(i) for i in 0:3],
        close=Float32[1f0, 2f0, 3f0, 4f0],
        tsmstate=categorical(["xch", "xch", "request", "none"]),
    )
    @test TSM.lastcheckpointedrowindex(checkpoint) == 3

    fresh = DataFrame(
        opentime=checkpoint[!, :opentime],
        close=Float32[0f0, 0f0, 0f0, 0f0],
        tsmstate=categorical(fill("none", 4)),
    )
    TSM.restorecheckpointrows!(fresh, checkpoint, 2)
    @test fresh[1, :close] == 1f0
    @test fresh[2, :close] == 2f0
    @test fresh[3, :close] == 0f0
    @test String(fresh[1, :tsmstate]) == "xch"
    @test String(fresh[3, :tsmstate]) == "none"
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
include("settrades_ownership_test.jl")
include("compilegainsdf_test.jl")