module TradeBybitGuardrailTest
using Test
using EnvConfig, Xch, Trade

EnvConfig.init(test)

@testset "Trade Bybit guardrail" begin
    cache = Trade.TradeCache()
    cache.xc.mc[:simmode] = Xch.nosimulation

    Xch.setsymbolinfocache!(cache.xc, "BTCUSDT", (
        symbol="BTCUSDT",
        status="Trading",
        basecoin="BTC",
        quotecoin="USDT",
        ticksize=0.01f0,
        baseprecision=0.00001f0,
        quoteprecision=0.01f0,
        minbaseqty=0.00001f0,
        minquoteqty=1.0f0,
    ))

    Xch.setrole!(cache.xc, Xch.data_exchange, Xch.EXCHANGE_KRAKENSPOT)
    Xch.setrole!(cache.xc, Xch.trade_exchange_spot, Xch.EXCHANGE_BYBIT)
    cache.xc.routecaches[Xch.EXCHANGE_KRAKENSPOT] = Xch.KrakenSpot.KrakenSpotCache(autoloadexchangeinfo=false, publickey="", secretkey="")

    err = try
        Xch.createopenorder(cache.xc, "BTC"; limitprice=1.0, basequantity=1.0, maker=true, configside=:long)
        nothing
    catch e
        e
    end

    @test err isa ErrorException
    @test occursin("configured as data-only", sprint(showerror, err))
end

end