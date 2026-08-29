module TradeBudgetAllocationTest
using Test
using Dates
using DataFrames

using EnvConfig, Trade, TradingStrategy, Xch, TSM, Targets

# `trade!` divides the available quote budget across all open signals before assigning any
# order, so the sum of assigned orders can never exceed the account budget. Nothing else
# covers that path: every other Trade test runs with trademode=notrade, which returns before
# the algorithm is invoked.

const QUOTE = EnvConfig.pairquote
const DT = DateTime(2026, 3, 2, 12, 0)

"Algorithm stub that pins the plugin contract: replay and live must both pass column handles."
function contract_algorithm!(cfg::TradingStrategy.StrategyConfig, cols::TSM.TradesColumns, ix::Integer)
    @assert ix >= 1 "unexpected row index $(ix)"
    return nothing
end

"Build a TradeCache in buysell mode plus the per-base trades rows `trade!` consumes."
function build_case(bases::Vector{String}; equity::Float32, closeprice::Float32, label=longopen, minorderquote::Float32=10f0, maxbudgetquote=nothing)
    xc = Xch.XchCache(startdt=DT, enddt=DT)
    TSM.ensuretradesschema!(xc.tsm, TSM.tradesdf_all_contributors())

    strategy = TradingStrategy.StrategyConfig(algorithm=contract_algorithm!)
    tc = Trade.TradeCache(xc=xc, strategy=strategy, trademode=Trade.buysell, stoplosspct=0.05)
    tc.cfg = DataFrame(basecoin=bases, openenabled=fill(true, length(bases)), closeenabled=fill(true, length(bases)))
    tc.mc[:minorderquote] = minorderquote
    tc.mc[:maxbudgetquote] = maxbudgetquote

    tradesdfdict = Dict{String, NamedTuple}()
    for base in bases
        entry = TSM.ensuretradesrow!(xc.tsm, base, QUOTE, DT)
        tdf, rowix = entry.tradesdf, entry.rowix
        tdf[rowix, :label] = label
        tdf[rowix, :close] = closeprice
        tdf[rowix, :equity] = equity
        tdf[rowix, :freequote] = equity
        tdf[rowix, :freemargin] = equity
        tradesdfdict[base] = (tradesdf=tdf, rowix=rowix)
    end
    return tc, tradesdfdict
end

"Total quote value of the open orders `trade!` assigned across all bases."
function assigned_openquote(tradesdfdict::Dict, lane::Symbol)
    total = 0f0
    for entry in values(tradesdfdict)
        row = entry.tradesdf[entry.rowix, :]
        total += (lane === :long ? row.lo_amount : row.so_amount) * row.close
    end
    return total
end

@testset "trade! divides the budget across open signals without overrun" begin
    equity = 900f0
    closeprice = 3f0
    bases = ["AAA", "BBB", "CCC"]
    tc, tradesdfdict = build_case(bases; equity=equity, closeprice=closeprice)

    Trade.trade!(tc, tradesdfdict)

    assigned = assigned_openquote(tradesdfdict, :long)
    @test assigned <= equity + 1f-3
    # equal division: three open signals share the budget
    for base in bases
        entry = tradesdfdict[base]
        @test isapprox(entry.tradesdf[entry.rowix, :lo_amount] * closeprice, equity / length(bases); rtol=1f-3)
    end
end

@testset "trade! caps total assignment at maxbudgetquote" begin
    equity = 900f0
    budget = 300f0
    closeprice = 3f0
    bases = ["AAA", "BBB", "CCC"]
    tc, tradesdfdict = build_case(bases; equity=equity, closeprice=closeprice, maxbudgetquote=budget)

    Trade.trade!(tc, tradesdfdict)

    @test assigned_openquote(tradesdfdict, :long) <= budget + 1f-3
end

@testset "trade! assigns nothing when the account cannot fund a minimum order" begin
    closeprice = 3f0
    bases = ["AAA"]
    tc, tradesdfdict = build_case(bases; equity=1f0, closeprice=closeprice, minorderquote=10f0)

    Trade.trade!(tc, tradesdfdict)

    entry = tradesdfdict["AAA"]
    @test entry.tradesdf[entry.rowix, :lo_amount] == 0f0
    @test entry.tradesdf[entry.rowix, :label] == ignore
end

@testset "trade! sizes short opens from the same budget" begin
    equity = 600f0
    closeprice = 2f0
    bases = ["AAA", "BBB"]
    tc, tradesdfdict = build_case(bases; equity=equity, closeprice=closeprice, label=shortopen)

    Trade.trade!(tc, tradesdfdict)

    @test assigned_openquote(tradesdfdict, :short) <= equity + 1f-3
    for base in bases
        entry = tradesdfdict[base]
        @test isapprox(entry.tradesdf[entry.rowix, :so_amount] * closeprice, equity / length(bases); rtol=1f-3)
    end
end

end # module
