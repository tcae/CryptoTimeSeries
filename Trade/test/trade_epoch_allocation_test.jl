module TradeEpochAllocationTest
using Test
using Dates
using DataFrames

using EnvConfig, Trade, TradingStrategy, Xch, TSM, Targets

# `tradeselection!` fixes the pair set, so it is where each pair's Trades rows for the
# coming epoch are allocated. Growing the frame per tick instead copies it and invalidates
# any held TradesColumns, so these tests pin the epoch length policy and the allocation.

@testset "epoch spans to the next reload time" begin
    tc = Trade.TradeCache(xc=Xch.XchCache(), trademode=Trade.notrade, stoplosspct=0.05)
    tc.mc[:reloadtimes] = [Time("04:00:00"), Time("16:00:00")]

    @test Trade._tradeselection_epochminutes(tc, DateTime(2026, 5, 1, 3, 0)) == 60
    @test Trade._tradeselection_epochminutes(tc, DateTime(2026, 5, 1, 15, 30)) == 30
end

@testset "epoch wraps to the next day after the last reload time" begin
    tc = Trade.TradeCache(xc=Xch.XchCache(), trademode=Trade.notrade, stoplosspct=0.05)
    tc.mc[:reloadtimes] = [Time("04:00:00")]

    # 06:00 is past the only reload time, so the next one is 04:00 tomorrow
    @test Trade._tradeselection_epochminutes(tc, DateTime(2026, 5, 1, 6, 0)) == 22 * 60
end

@testset "epoch defaults to a day without a reload schedule" begin
    tc = Trade.TradeCache(xc=Xch.XchCache(), trademode=Trade.notrade, stoplosspct=0.05)
    tc.mc[:reloadtimes] = Time[]

    @test Trade._tradeselection_epochminutes(tc, DateTime(2026, 5, 1, 6, 0)) == 24 * 60
end

@testset "epoch length is always positive" begin
    tc = Trade.TradeCache(xc=Xch.XchCache(), trademode=Trade.notrade, stoplosspct=0.05)
    tc.mc[:reloadtimes] = [Time("04:00:00")]

    # exactly at the reload time the next boundary is a full day away, never zero
    @test Trade._tradeselection_epochminutes(tc, DateTime(2026, 5, 1, 4, 0)) >= 1
end

end # module
