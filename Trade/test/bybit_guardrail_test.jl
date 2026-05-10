module TradeBybitGuardrailTest
using Test
using EnvConfig, CryptoXch, Trade

EnvConfig.init(test)

@testset "Trade Bybit guardrail" begin
    cache = Trade.TradeCache()
    cache.xc.mc[:simmode] = CryptoXch.nosimulation

    CryptoXch.setrole!(cache.xc, CryptoXch.data_exchange, CryptoXch.EXCHANGE_KRAKENSPOT)
    CryptoXch.setrole!(cache.xc, CryptoXch.trade_exchange_spot, CryptoXch.EXCHANGE_BYBIT)

    err = try
        CryptoXch.createbuyorder(cache.xc, "BTC"; limitprice=1.0, basequantity=1.0, maker=true)
        nothing
    catch e
        e
    end

    @test err isa ErrorException
    @test occursin("configured as data-only", sprint(showerror, err))
end

end