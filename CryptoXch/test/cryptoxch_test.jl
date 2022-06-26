
module CryptoXchTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, CryptoXch

function balances_test()
    result = CryptoXch.balances()
    display(result)
    display(EnvConfig.bases)
    display(EnvConfig.trainingbases)
    display(EnvConfig.datapath)
end

# EnvConfig.init(test)
EnvConfig.init(production)
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
    df = CryptoXch.gethistoryohlcv("btc", startdt, enddt)
    # display(first(df, 2))
    # display(last(df, 2))
    # println("saved btc from $(df[1, :opentime]) until $(df[end, :opentime])")
    return df
end

function addstartgapbtcdownload()
    startdt = DateTime("2022-01-02T22:40:03")
    enddt = DateTime("2022-01-02T22:41:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function appendgapbtcdownload()
    startdt = DateTime("2022-01-02T22:53:03")
    enddt = DateTime("2022-01-02T22:55:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function appendoverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:47:03")
    enddt = DateTime("2022-01-02T22:51:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function addfulloverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:44:01")
    enddt = DateTime("2022-01-02T22:50:45")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function addstartoverlapbtcdownload()
    startdt = DateTime("2022-01-02T22:43:03")
    enddt = DateTime("2022-01-02T22:47:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end

function initialbtcdownload()
    startdt = DateTime("2022-01-02T22:45:03")
    enddt = DateTime("2022-01-02T22:49:35")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", startdt, enddt)
    return ohlcv
end



@testset "CryptoXch tests" begin

    df = CryptoXch.klines2jdf(missing)
    @test nrow(df) == 0
    mdf = CryptoXch.getUSDTmarket()
    # println(mdf)
    @test names(mdf) == ["base", "qte", "quotevolume24h", "pricechangepercent", "lastprice"]
    @test nrow(mdf) > 10

    EnvConfig.init(EnvConfig.test)
    ohlcv = Ohlcv.defaultohlcv("btc")
    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:45:03"), DateTime("2022-01-02T22:49:35"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5
    @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume", "pivot"]

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:45:01"), DateTime("2022-01-02T22:49:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:48:01"), DateTime("2022-01-02T22:51:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:53:03"), DateTime("2022-01-02T22:55:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 13

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:40:03"), DateTime("2022-01-02T22:41:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 16

    ohlcv = CryptoXch.cryptodownload("btc", "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 20

    ohlcv1 = Ohlcv.copy(ohlcv)
    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 4

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:47:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 5

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:49:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 7

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:42:00"), DateTime("2022-01-02T22:49:45"))
    # does not add anything for DateTime("2022-01-02T22:42:03")
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 8

    CryptoXch.cryptoupdate!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    @test size(Ohlcv.dataframe(Ohlcv.read!(ohlcv)), 1) == 0

    @test CryptoXch.onlyconfiguredsymbols("BTCUSDT")
    @test !CryptoXch.onlyconfiguredsymbols("BTCBNB")
    @test !CryptoXch.onlyconfiguredsymbols("EURUSDT")
end


end  # module