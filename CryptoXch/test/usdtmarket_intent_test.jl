module CryptoXchUsdtMarketIntentTest

using Test
using Dates
using DataFrames
using EnvConfig, CryptoXch, Bybit

EnvConfig.init(EnvConfig.test)
EnvConfig.cryptoquote = "USDT"

@testset "CryptoXch USDT market intent APIs" begin
    dt = DateTime("2025-05-01T12:00:00")
    xc = CryptoXch.XchCache(startdt=dt, enddt=dt, exchange=CryptoXch.EXCHANGE_BYBITSIM)
    CryptoXch.setcurrenttime!(xc, dt)

    if isnothing(findfirst(==("AAPLXUSDT"), xc.bc.syminfodf[!, :symbol]))
        push!(xc.bc.syminfodf, (
            symbol="AAPLXUSDT",
            status="Trading",
            basecoin="AAPLX",
            quotecoin="USDT",
            ticksize=0.01f0,
            baseprecision=1f-4,
            quoteprecision=0.01f0,
            minbaseqty=1f-4,
            minquoteqty=1f0,
            innovation=0,
        ))
    end

    Bybit.seedportfolio!(xc.bc, "USDT", 100000.0)
    Bybit.seedportfolio!(xc.bc, "BTC", 0.5)

    balancesdf = CryptoXch.balances(xc; ignoresmallvolume=false)
    requestedbases = [String(c) for c in balancesdf[!, :coin] if c != EnvConfig.cryptoquote]

    screeningdf = CryptoXch.screeningUSDTmarket(xc; dt=dt)
    @test nrow(screeningdf) > 0
    @test !("AAPLX" in Set(String.(screeningdf[!, :basecoin])))

    valuationdf = CryptoXch.valuationUSDTmarket(xc, requestedbases; dt=dt)
    compatvaluation = CryptoXch.getUSDTmarket(xc; dt=dt, requestedbases=requestedbases)

    @test nrow(valuationdf) > 0

    valuationcoins = Set(String.(valuationdf[!, :basecoin]))
    @test valuationcoins == Set(requestedbases)
    @test !("AAPLX" in valuationcoins)
    @test Set(String.(compatvaluation[!, :basecoin])) == valuationcoins

    portfolio = CryptoXch.portfolio!(xc, balancesdf, nothing; ignoresmallvolume=false)
    @test "BTC" in Set(String.(portfolio[!, :coin]))
    @test all(ismissing.(portfolio[!, :usdtprice]) .== false)
end

end