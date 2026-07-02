module XchTradesSchemaContractTest
using Test
using Dates
using DataFrames
using CategoricalArrays

using EnvConfig, Xch, Targets

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
        :longid, :longstatus, :longunfilled, :longpriceavg, :longmsg,
        :shortid, :shortstatus, :shortunfilled, :shortpriceavg, :shortmsg,
        :postype, :posamount, :quoteprice, :maintmargin,
        :equity, :balance, :freemargin, :freequote,
    ])

    got = Set(Symbol.(names(tdf)))
    @test required ⊆ got
    @test !(:coin in got)
    @test tdf[!, :pair] isa CategoricalVector
    @test tdf[!, :longstatus] isa CategoricalVector
    @test tdf[!, :shortstatus] isa CategoricalVector
    @test tdf[!, :longmsg] isa CategoricalVector
    @test tdf[!, :shortmsg] isa CategoricalVector
    @test !any(ismissing, tdf[!, :longmsg])
    @test !any(ismissing, tdf[!, :shortmsg])
    @test all(String(v) == "none" for v in tdf[!, :longmsg])
    @test all(String(v) == "none" for v in tdf[!, :shortmsg])
    @test eltype(tdf[!, :tradelabel]) == Targets.TradeLabel
    @test eltype(tdf[!, :longopenlimit]) == Float32
    @test eltype(tdf[!, :longcloselimit]) == Float32
    @test eltype(tdf[!, :shortopenlimit]) == Float32
    @test eltype(tdf[!, :shortcloselimit]) == Float32
    @test eltype(tdf[!, :postype]) == Targets.TrendPhase

    defaultdf = DataFrame(opentime=[startdt])
    TradingStrategy.tradesdf_tradelabel(defaultdf)
    @test defaultdf[1, :tradelabel] == Targets.ignore

    defaultlimitsdf = DataFrame(opentime=[startdt], pair=["BTCUSDT"])
    for contributor in TradingStrategy.tradesdf_contributors()
        contributor(defaultlimitsdf)
    end
    @test defaultlimitsdf[1, :longopenlimit] == 0f0
    @test defaultlimitsdf[1, :longcloselimit] == 0f0
    @test defaultlimitsdf[1, :shortopenlimit] == 0f0
    @test defaultlimitsdf[1, :shortcloselimit] == 0f0

    push!(tdf, (opentime=startdt, pair="BTCUSDT", tradelabel=Targets.ignore, lastopentrade=missing, postype=Targets.flat); cols=:subset)
    Xch.tradesdf_longmsg(tdf)
    Xch.tradesdf_shortmsg(tdf)
    @test nrow(tdf) == 1
    @test tdf[1, :tradelabel] == Targets.ignore
    @test !ismissing(tdf[1, :pair])
    @test String(tdf[1, :pair]) == "BTCUSDT"
    @test tdf[1, :postype] == Targets.flat
    @test String(tdf[1, :longmsg]) == "none"
    @test String(tdf[1, :shortmsg]) == "none"

    tdf[1, :tradelabel] = Targets.longopen
    @test tdf[1, :tradelabel] == Targets.longopen
end

end
