
include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/cryptoxch.jl")
include("../src/assets.jl")

module AssetsTest
using Dates, DataFrames
using Test

using ..Ohlcv, ..Config, ..Assets

Config.init(production)
# ad1 = Assets.loadassets()
# Assets.write(ad1)
ad1 = Assets.read()
ad2 = Assets.read()

# println(ad1.df)
# println(ad2.df)
println("ad1 == ad2? $(ad1.df==ad2.df)")

# @testset "CryptoXch tests" begin
#     startdt = DateTime("2020-08-11T22:45:00")
#     enddt = DateTime("2020-08-12T22:49:00")
#     df = CryptoXch.gethistoryohlcv("btc", startdt, enddt, "1m")
#     @test names(df) == ["opentime", "open", "high", "low", "close", "basevolume"]
#     @test nrow(df) == 1445

#     ohlcv1 = Ohlcv.defaultohlcv("btc")
#     Ohlcv.setdataframe!(ohlcv1, df)
#     # println(first(ohlcv1.df,3))
#     Ohlcv.write(ohlcv1)
#     ohlcv2 = Ohlcv.defaultohlcv("btc")
#     ohlcv2 = Ohlcv.read!(ohlcv2)
#     # println(first(ohlcv2.df,3))
#     @test ohlcv1.df == ohlcv2.df
#     @test ohlcv1.base == ohlcv2.base

#     df = CryptoXch.klines2jdf(missing)
#     @test names(df) == ["opentime", "open", "high", "low", "close", "basevolume"]
#     @test nrow(df) == 0
#     mdf = CryptoXch.getmarket()
#     # println(mdf)
#     @test names(mdf) == ["base", "quotevolume24h"]
#     @test nrow(mdf) > 10

#     @test PrepareTest() == "/home/tor/crypto/TestFeatures/btc_usdt_binance_1m_OHLCV.jdf"
#     ohlcv = initialbtcdownload()
#     # println(Ohlcv.dataframe(ohlcv))
#     @test size(Ohlcv.dataframe(ohlcv), 1) == 5
#     @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume"]



# end

end  # module