using Test
using TradingStrategy
using Targets

@testset "gain_limit_reversal direction" begin
    @testset "short signal with open longclose mirrors short entry buyprice" begin
        sellta = TradingStrategy.TradeAction(longclose, 110f0, 100f0, 1)
        buyta = TradingStrategy.TradeAction()

        TradingStrategy.reachgainuntilreversal!(
            sellta,
            buyta,
            shortbuy,
            0.9f0,
            101f0,
            99f0,
            100f0,
            0.6f0,
            0.001f0,
            0.01f0,
            0f0,
        )

        @test isapprox(sellta.orderlimit, 100.1f0; atol=1f-4)
    end

    @testset "long signal with open shortclose mirrors long entry buyprice" begin
        sellta = TradingStrategy.TradeAction(shortclose, 90f0, 100f0, 1)
        buyta = TradingStrategy.TradeAction()

        TradingStrategy.reachgainuntilreversal!(
            sellta,
            buyta,
            longbuy,
            0.9f0,
            101f0,
            99f0,
            100f0,
            0.6f0,
            0.001f0,
            0.01f0,
            0f0,
        )

        @test isapprox(sellta.orderlimit, 99.9f0; atol=1f-4)
    end

    @testset "new long intent uses -buygain entry and +sellgain exit target" begin
        sellta = TradingStrategy.TradeAction()
        buyta = TradingStrategy.TradeAction()

        TradingStrategy.reachgainuntilreversal!(
            sellta,
            buyta,
            longbuy,
            0.9f0,
            101f0,
            99f0,
            100f0,
            0.6f0,
            0.001f0,
            0.01f0,
            0f0,
        )

        @test buyta.orderlabel == longbuy
        @test buyta.buyprice == 99.9f0
        @test buyta.orderlimit == 101f0
    end
end
