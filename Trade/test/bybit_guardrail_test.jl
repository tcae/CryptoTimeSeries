module TradeBybitGuardrailTest
using Test
using EnvConfig, Xch, Trade

EnvConfig.init(test)

@testset "Trade Bybit guardrail" begin
    cache = Trade.TradeCache()
    cache.xc.mc[:simmode] = Xch.nosimulation

    Xch.setrole!(cache.xc, Xch.data_exchange, Xch.EXCHANGE_KRAKENSPOT)
    Xch.setrole!(cache.xc, Xch.trade_exchange_spot, Xch.EXCHANGE_BYBIT)

    err = try
        Xch.createbuyorder(cache.xc, "BTC"; limitprice=1.0, basequantity=1.0, maker=true)
        nothing
    catch e
        e
    end

    @test err isa ErrorException
    @test occursin("configured as data-only", sprint(showerror, err))
end

end