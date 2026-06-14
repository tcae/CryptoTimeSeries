module XchTradesSchemaContractTest
using Test
using Dates
using DataFrames

using EnvConfig, Xch

@testset "Xch trades schema contract" begin
    EnvConfig.init(test)
    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(1)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)

    tdf = Xch.trades(xc, "BTC", EnvConfig.pairquote)

    expected = Set([
        :opentime, :lastopentrade, :pair, :coin, :tradelabel, :labelscore,
        :longleverage, :longamount, :shortleverage, :shortamount,
        :longopenlimit, :longcloselimit, :shortopenlimit, :shortcloselimit,
        :longid, :longstatus, :longunfilled, :longpriceavg, :longmsgid,
        :shortid, :shortstatus, :shortunfilled, :shortpriceavg, :shortmsgid,
        :postype, :posleverage, :posamount, :quoteprice, :maintmargin,
        :equity, :balance, :freemargin, :freequote,
    ])

    got = Set(Symbol.(names(tdf)))
    @test expected == got
    @test !(:label in got)

    push!(tdf, (opentime=startdt, lastopentrade=missing); cols=:subset)
    @test nrow(tdf) == 1
    @test ismissing(tdf[1, :tradelabel])

    tdf[1, :tradelabel] = "longopen"
    @test tdf[1, :tradelabel] == "longopen"
end

end
