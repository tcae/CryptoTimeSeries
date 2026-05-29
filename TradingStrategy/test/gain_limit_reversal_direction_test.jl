using Test
using TradingStrategy
using Targets

@testset "gain_limit_reversal direction" begin
    @testset "short signal with open longclose mirrors short entry openprice" begin
        longta = TradingStrategy.TradeAction(longclose, 110f0, 100f0, 1)
        shortta = TradingStrategy.TradeAction()

        TradingStrategy.reachgainuntilreversal!(
            longta,
            shortta,
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

        @test isapprox(longta.closeprice, 100.1f0; atol=1f-4)
    end

    @testset "long signal with open shortclose mirrors long entry openprice" begin
        longta = TradingStrategy.TradeAction()
        shortta = TradingStrategy.TradeAction(shortclose, 90f0, 100f0, 1)

        TradingStrategy.reachgainuntilreversal!(
            longta,
            shortta,
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

        @test isapprox(shortta.closeprice, 99.9f0; atol=1f-4)
    end

    @testset "new long intent uses -buygain entry and +sellgain exit target" begin
        longta = TradingStrategy.TradeAction()
        shortta = TradingStrategy.TradeAction()

        TradingStrategy.reachgainuntilreversal!(
            longta,
            shortta,
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

        @test longta.label == longbuy
        @test longta.openprice == 99.9f0
        @test longta.closeprice == 101f0
    end
end
