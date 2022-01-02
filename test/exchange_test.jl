
using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/exchange.jl")

module ExchangeTest
using Dates, DataFrames
using Test

using ..Ohlcv, ..Config
using ..Exchange

function balances_test()
    result = Exchange.balances()
    display(result)
    display(Config.bases)
    display(Config.trainingbases)
    display(Config.datapath)
end

# Config.init(test)
Config.init(production)
# balances_test()

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")
# res = Binance.getKlines("BTCUSDT"; startDateTime=startdt, endDateTime=enddt, interval="1m")
# display(res)
# display(last(res[:body], 3))
# display(first(res[:body], 3))
# display(res[:body][1:3, :])
# display(res[:body][end-3:end, :])

# Binance.wsKlineStreams(cb, ["BTCUSDT", "XRPUSDT"])

function gethistoryohlcv_test()
    startdt = DateTime("2020-08-11T22:45:00")
    enddt = DateTime("2020-08-12T22:49:00")
    df = Exchange.gethistoryohlcv("btc", startdt, enddt)
    # display(first(df, 2))
    # display(last(df, 2))
    # println("saved btc from $(df[1, :opentime]) until $(df[end, :opentime])")
    return df
end

@testset "Exchange tests" begin
    startdt = DateTime("2020-08-11T22:45:00")
    enddt = DateTime("2020-08-12T22:49:00")
    df = Exchange.gethistoryohlcv("btc", startdt, enddt, "1m")
    @test names(df) == ["opentime", "open", "high", "low", "close", "volume"]
    @test nrow(df) == 1445

    ohlcv1 = Ohlcv.OhlcvData(df, "btc", "1m")
    Ohlcv.addpivot!(ohlcv1)
    Ohlcv.write(ohlcv1)
    ohlcv2 = Ohlcv.read("btc", "1m")
    @test ohlcv1.df == ohlcv2.df
    @test ohlcv1.base == ohlcv2.base
    # println(first(ohlcv1.df,3))
    # println(first(ohlcv2.df,3))

    df = Exchange.klines2jdf(missing)
    @test names(df) == ["opentime", "open", "high", "low", "close", "volume"]
    @test nrow(df) == 0

end
println(Exchange.getmarket())

end  # module