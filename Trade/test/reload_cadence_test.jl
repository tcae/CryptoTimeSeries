using Test
using Dates
using Trade
using CryptoXch

@testset "trade reload cadence" begin
    ts = DateTime("2026-05-18T04:00:30")
    tc = Trade.TradeCache(xc=CryptoXch.XchCache(startdt=ts, enddt=ts), trademode=Trade.notrade)

    tc.xc.currentdt = ts
    tc.mc[:reloadtimes] = [Time("04:00:00")]
    tc.mc[:last_traderefresh_dt] = nothing

    @test Trade._should_refresh_tradeselection(tc)

    Trade._mark_tradeselection_refreshed!(tc)
    @test !Trade._should_refresh_tradeselection(tc)

    tc.xc.currentdt = ts + Minute(1)
    @test !Trade._should_refresh_tradeselection(tc)

    tc.xc.currentdt = ts + Day(1)
    @test Trade._should_refresh_tradeselection(tc)
end
