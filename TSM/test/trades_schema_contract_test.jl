module TsmTradesSchemaContractTest
using Test
using Dates
using DataFrames
using CategoricalArrays

using Targets, TSM

@testset "TSM trades schema contract" begin
    startdt = DateTime("2022-01-01T01:00:00")
    tsm = TSM.TsmCache()
    tdf = TSM.trades(tsm, "BTC", "USDT")

    required = Set([
        :opentime, :lastopentrade, :pair, :config, :tsmstate, :label, :score,
        :lo_limit, :lc_limit, :so_limit, :sc_limit,
        :lo_id, :lo_status, :lol_filled, :lol_pavg, :lo_msg,
        :lc_id, :lc_status, :lcl_filled, :lcl_pavg, :lc_msg,
        :so_id, :so_status, :sol_filled, :sol_pavg, :so_msg,
        :sc_id, :sc_status, :scl_filled, :scl_pavg, :sc_msg,
        :lp_amount, :sp_amount, :close, :high, :low, :maintmargin,
        :equity, :balance, :freemargin, :freequote,
    ])

    got = Set(Symbol.(names(tdf)))
    @test required ⊆ got
    @test !(:coin in got)
    @test tdf[!, :pair] isa CategoricalVector
    @test tdf[!, :config] isa CategoricalVector
    @test tdf[!, :tsmstate] isa CategoricalVector
    @test tdf[!, :lo_status] isa CategoricalVector
    @test tdf[!, :lc_status] isa CategoricalVector
    @test tdf[!, :so_status] isa CategoricalVector
    @test tdf[!, :sc_status] isa CategoricalVector
    @test tdf[!, :lo_msg] isa CategoricalVector
    @test tdf[!, :lc_msg] isa CategoricalVector
    @test tdf[!, :so_msg] isa CategoricalVector
    @test tdf[!, :sc_msg] isa CategoricalVector
    @test !any(ismissing, tdf[!, :lo_msg])
    @test !any(ismissing, tdf[!, :lc_msg])
    @test !any(ismissing, tdf[!, :so_msg])
    @test !any(ismissing, tdf[!, :sc_msg])
    @test all(String(v) == "none" for v in tdf[!, :lo_msg])
    @test all(String(v) == "none" for v in tdf[!, :lc_msg])
    @test all(String(v) == "none" for v in tdf[!, :so_msg])
    @test all(String(v) == "none" for v in tdf[!, :sc_msg])
    @test eltype(tdf[!, :label]) == TradeLabel
    @test eltype(tdf[!, :lo_limit]) == Float32
    @test eltype(tdf[!, :lc_limit]) == Float32
    @test eltype(tdf[!, :so_limit]) == Float32
    @test eltype(tdf[!, :sc_limit]) == Float32
    @test eltype(tdf[!, :lp_amount]) == Float32
    @test eltype(tdf[!, :sp_amount]) == Float32

    defaultdf = DataFrame(opentime=[startdt])
    TSM.tradingstrategy_tradesdf_label(defaultdf)
    @test defaultdf[1, :label] == ignore

    defaultlimitsdf = DataFrame(opentime=[startdt], pair=["BTCUSDT"])
    for contributor in TSM.tradingstrategy_tradesdf_contributors()
        contributor(defaultlimitsdf)
    end
    @test defaultlimitsdf[1, :lo_limit] == 0f0
    @test defaultlimitsdf[1, :lc_limit] == 0f0
    @test defaultlimitsdf[1, :so_limit] == 0f0
    @test defaultlimitsdf[1, :sc_limit] == 0f0

    push!(tdf, (opentime=startdt, pair="BTCUSDT", label=ignore, lastopentrade=missing, lp_amount=0f0, sp_amount=0f0); cols=:subset)
    TSM.xch_tradesdf_msg(tdf, longopen)
    TSM.xch_tradesdf_msg(tdf, longclose)
    TSM.xch_tradesdf_msg(tdf, shortopen)
    TSM.xch_tradesdf_msg(tdf, shortclose)
    @test nrow(tdf) == 1
    @test tdf[1, :label] == ignore
    @test !ismissing(tdf[1, :pair])
    @test String(tdf[1, :pair]) == "BTCUSDT"
    @test tdf[1, :lp_amount] == 0f0
    @test tdf[1, :sp_amount] == 0f0
    @test ismissing(tdf[1, :lo_msg]) || String(tdf[1, :lo_msg]) == "none"
    @test ismissing(tdf[1, :lc_msg]) || String(tdf[1, :lc_msg]) == "none"
    @test ismissing(tdf[1, :so_msg]) || String(tdf[1, :so_msg]) == "none"
    @test ismissing(tdf[1, :sc_msg]) || String(tdf[1, :sc_msg]) == "none"

    tdf[1, :label] = longopen
    @test tdf[1, :label] == longopen
end

end