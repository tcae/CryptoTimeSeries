
include("../src/env_config.jl")
include("../src/ohlcv.jl")
include("../src/cryptoxch.jl")
include("../src/assets.jl")

module AssetsTest
using Dates, DataFrames
using Test

using ..Ohlcv, ..Config, ..Assets, ..CryptoXch

Config.init(production)
# ad2 = Assets.read()
# sort!(ad2.df, [:base])
# usdtdf = CryptoXch.getUSDTmarket()
# sort!(usdtdf, [:base])
# # println("usdtdf")
# println(first(usdtdf, 10))
# ad = ad2
# # ad2.df.quotevolume24h = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad2.df.base])), :quotevolume24h]
# ad.df[:, :quotevolume24h] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :quotevolume24h]
# ad.df[:, :priceChangePercent] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :priceChangePercent]

# println(Assets.portfolioselect(CryptoXch.getUSDTmarket()))
@testset "Assets tests" begin
    # ad1 = Assets.read()
    ad1 = Assets.loadassets()
    # @test size(ad1.df, 1) > 0
    nsyms = Symbol.(names(ad1.df))
    @test all([col in nsyms for col in Assets.savecols])
    Assets.write(ad1)
    ad2 = Assets.read()
    @test ad1.df==ad2.df
    # println("ad2.df")
    # println(ad2.df)

end

end  # module
