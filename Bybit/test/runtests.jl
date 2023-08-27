using Bybit, EnvConfig, Test, Dates

EnvConfig.init(production)

@testset "CryptoXch tests" begin

    @test Bybit.serverTime() > DateTime("2023-08-18T20:07:54.209")

    acc = Bybit.account(EnvConfig.authorization.key, EnvConfig.authorization.secret)
    @test size([k for k in keys(acc)], 1) == 7

    xchinf = Bybit.getExchangeInfo()
    @test typeof(xchinf) == Vector{Any}
    @test size(xchinf) > (400,)
    @test typeof(xchinf[1]) == Dict{String, Any}
    @test size(xchinf) > (400,)
    @test size([k for k in keys(xchinf[1])], 1) == 8
    @test length([xchdict["symbol"] for xchdict in xchinf if xchdict["symbol"] == "BTCUSDT"]) == 1

    dayresult = Bybit.get24HR("BTCUSDT")
    @test size([k for k in keys(dayresult)], 1) == 13
    @test dayresult["symbol"] == "BTCUSDT"

    klines = Bybit.getKlines("BTCUSDT")
    @test typeof(klines) == Vector{Any}
    @test typeof(klines[1]) == Vector{Any}
    @test size(klines[1]) == (7,)
    @test size(klines) == (1000,)

end

Bybit.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
# order = Bybit.createOrder("BTCUSDT", )
# res = Bybit.openOrders(nothing, EnvConfig.authorization.key, EnvConfig.authorization.secret)
# res = Bybit.openOrders("CHZUSDT", EnvConfig.authorization.key, EnvConfig.authorization.secret)
# println(res)
# for o in res
#     for dictentry in o
#         println(dictentry)
#     end
#     println("=")
# end
