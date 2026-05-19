using Test
using Dates
using DataFrames
using Trade
using TradingStrategy
using Targets

@testset "strategy advice handling" begin
    tc = Trade.TradeCache()

    @testset "same-minute reversal expansion" begin
        ta = Trade.StrategyAdvice(
            classifier=tc.cl,
            configid=1,
            tradelabel=longbuy,
            relativeamount=1f0,
            base="BTC",
            price=nothing,
            datetime=DateTime("2026-05-18T00:00:00"),
            hourlygain=0.2f0,
            probability=0.9f0,
            investmentid=nothing,
            source=:getgainsalgo,
            allowreversal=true,
        )

        assets = DataFrame(
            coin=["BTC", "USDT"],
            free=Float32[0f0, 1000f0],
            borrowed=Float32[1f0, 0f0],
        )

        expanded = Trade._expand_reversal_advice(ta, assets)
        @test length(expanded) == 2
        @test expanded[1].tradelabel == shortclose
        @test expanded[2].tradelabel == longbuy

        assets_no_short = DataFrame(
            coin=["BTC", "USDT"],
            free=Float32[0f0, 1000f0],
            borrowed=Float32[0f0, 0f0],
        )
        expanded_no_reversal = Trade._expand_reversal_advice(ta, assets_no_short)
        @test length(expanded_no_reversal) == 1
        @test expanded_no_reversal[1].tradelabel == longbuy
    end

    @testset "getgains limit price extraction" begin
        gs = TradingStrategy.GainSegment(; algorithm=TradingStrategy.gain_limit_reversal!)
        gs.buyta = TradingStrategy.TradeAction(longbuy, 101f0, 100f0, 1)
        gs.sellta = TradingStrategy.TradeAction()
        @test Trade._getgainsalgo_limitprice(gs, longbuy) == 100f0

        gs.buyta = TradingStrategy.TradeAction()
        gs.sellta = TradingStrategy.TradeAction(longclose, 110f0, 100f0, 1)
        @test Trade._getgainsalgo_limitprice(gs, longclose) == 110f0
    end

    @testset "USDTmsg uses net free quote balance" begin
        oldquote = EnvConfig.cryptoquote
        try
            EnvConfig.cryptoquote = "USDT"
            assets = DataFrame(
                coin=["BTC", "USDT"],
                free=Float32[0f0, 108f0],
                borrowed=Float32[0.1f0, 0f0],
                usdtprice=Float32[100f0, 1f0],
                usdtvalue=Float32[-10f0, 108f0],
            )

            @test Trade.USDTmsg(assets) == "USDT: total=98, free=100%"
        finally
            EnvConfig.cryptoquote = oldquote
        end
    end
end
