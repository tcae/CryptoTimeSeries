
module AssetsTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, Assets, CryptoXch

EnvConfig.init(production)
# ad2 = Assets.read!(Assets.AssetData())
# sort!(ad2.df, [:base])
# usdtdf = CryptoXch.getUSDTmarket()
# sort!(usdtdf, [:base])
# # println("usdtdf")
# println(first(usdtdf, 10))
# ad = ad2
# # ad2.df.quotevolume24h = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad2.df.base])), :quotevolume24h]
# ad.df[:, :quotevolume24h] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :quotevolume24h]
# ad.df[:, :pricechangepercent] = usdtdf[in.(usdtdf[!,:base], Ref([base for base in ad.df[!, :base]])), :pricechangepercent]

# println(Assets.portfolioselect(CryptoXch.getUSDTmarket()))
@testset "Assets tests" begin
    ad1 = Assets.read!(Assets.AssetData())
    # ad1 = Assets.loadassets!(Assets.AssetData())
    # @test size(ad1.df, 1) > 0
    nsyms = Symbol.(names(ad1.basedf))
    @test all([col in nsyms for col in Assets.savecols])
    Assets.write(ad1)
    ad2 = Assets.read!(Assets.AssetData())
    @test ad1.basedf==ad2.basedf
    # println("ad2.df")
    # println(ad2.df)

end

end  # module
