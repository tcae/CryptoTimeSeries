module XchTradesSchemaContractTest
using Test
using Dates
using DataFrames
using CategoricalArrays

using EnvConfig, Xch

const HAS_TRADINGSTRATEGY = try
    @eval using TradingStrategy
    true
catch
    false
end

@testset "Xch trades schema contract" begin
    if !HAS_TRADINGSTRATEGY
        @info "Skipping Xch trades schema contract test because TradingStrategy is not available in this test environment"
        return
    end

    EnvConfig.init(test)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(1)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), TradingStrategy.tradesdf_contributors()))

    tdf = Xch.trades(xc, "BTC", EnvConfig.pairquote)

    required = Set([
        :opentime, :lastopentrade, :pair, :tradelabel, :labelscore,
        :longopenlimit, :longcloselimit, :shortopenlimit, :shortcloselimit,
        :longid, :longstatus, :longunfilled, :longpriceavg, :longmsgid,
        :shortid, :shortstatus, :shortunfilled, :shortpriceavg, :shortmsgid,
        :postype, :posleverage, :posamount, :quoteprice, :maintmargin,
        :equity, :balance, :freemargin, :freequote,
    ])

    got = Set(Symbol.(names(tdf)))
    @test required ⊆ got
    @test !(:coin in got)
    @test tdf[!, :pair] isa CategoricalVector
    @test eltype(tdf[!, :tradelabel]) == Union{Missing, typeof(TradingStrategy.longopen)}

    push!(tdf, (opentime=startdt, lastopentrade=missing); cols=:subset)
    @test nrow(tdf) == 1
    @test ismissing(tdf[1, :tradelabel])
    @test ismissing(tdf[1, :pair])

    tdf[1, :tradelabel] = TradingStrategy.longopen
    @test tdf[1, :tradelabel] == TradingStrategy.longopen
end

end
