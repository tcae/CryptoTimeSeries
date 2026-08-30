module TradeLoopProgressTest

using Dates, Test, DataFrames
using EnvConfig, Trade, Xch

@testset "Trade loop progresses across day boundaries" begin
    EnvConfig.init(EnvConfig.test)
    startdt = DateTime("2025-01-01T23:59:00")
    enddt = startdt + Minute(2)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    cache = Trade.TradeCache(xc=xc, trademode=Trade.notrade, stoplosspct=0.05f0)
    cache.cfg = DataFrame(basecoin=String[], pair=String[], openenabled=Bool[], closeenabled=Bool[])

    Trade.run_backtest!(cache; skip_init=true)

    @test cache.xc.currentdt === nothing
end

end