
include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/cryptoxch.jl")

module CryptoXchTest
using Dates, DataFrames
using Test

using ..Ohlcv, ..Config, ..CryptoXch

function balances_test()
    result = CryptoXch.balances()
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
    startdt = DateTime("2022-01-02T22:45:01")
    enddt = DateTime("2022-01-02T22:49:45")
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

function PrepareTest()
    Config.init(Config.test)
    ohlcv = Ohlcv.defaultohlcv("btc")
    mnm = Ohlcv.mnemonic(ohlcv)
    filename = Config.datafile(mnm)
    Ohlcv.delete(ohlcv)
    return filename
end


@testset "CryptoXch tests" begin
    startdt = DateTime("2020-08-11T22:45:00")
    enddt = DateTime("2020-08-12T22:49:00")
    df = CryptoXch.gethistoryohlcv("btc", startdt, enddt, "1m")
    @test names(df) == ["opentime", "open", "high", "low", "close", "basevolume"]
    @test nrow(df) == 1445

    ohlcv1 = Ohlcv.defaultohlcv("btc")
    Ohlcv.setdataframe!(ohlcv1, df)
    # println(first(ohlcv1.df,3))
    Ohlcv.write(ohlcv1)
    ohlcv2 = Ohlcv.defaultohlcv("btc")
    ohlcv2 = Ohlcv.read!(ohlcv2)
    # println(first(ohlcv2.df,3))
    @test ohlcv1.df == ohlcv2.df
    @test ohlcv1.base == ohlcv2.base

    df = CryptoXch.klines2jdf(missing)
    @test names(df) == ["opentime", "open", "high", "low", "close", "basevolume"]
    @test nrow(df) == 0
    mdf = CryptoXch.getmarket()
    # println(mdf)
    @test names(mdf) == ["base", "quotevolume24h"]
    @test nrow(mdf) > 10

    @test PrepareTest() == "/home/tor/crypto/TestFeatures/btc_usdt_binance_1m_OHLCV.jdf"
    ohlcv = initialbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5
    @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume"]

    ohlcv = addfulloverlapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5

    ohlcv = appendoverlapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    ohlcv = addstartoverlapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    ohlcv = appendgapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    ohlcv = addstartgapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    ohlcv = addfulloverlapbtcdownload()
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

end

end  # module