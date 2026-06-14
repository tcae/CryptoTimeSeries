module XchUsdtMarketIntentTest

using Test
using Dates
using DataFrames
using EnvConfig, Xch, Bybit

EnvConfig.init(EnvConfig.test)
EnvConfig.setpairquote!("USDT")

@testset "Xch USDT market intent APIs" begin
    dt = DateTime("2025-05-01T12:00:00")
    xc = Xch.XchCache(startdt=dt, enddt=dt, exchange=Xch.EXCHANGE_BYBITSIM)
    Xch.setcurrenttime!(xc, dt)

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

    balancesdf = Xch.balances(xc; ignoresmallvolume=false)
    requestedbases = [String(c) for c in balancesdf[!, :coin] if c != EnvConfig.pairquote]

    screeningdf = Xch.screeningUSDTmarket(xc; dt=dt)
    @test nrow(screeningdf) > 0
    @test !("AAPLX" in Set(String.(screeningdf[!, :basecoin])))

    valuationdf = Xch.valuationUSDTmarket(xc, requestedbases; dt=dt)
    compatvaluation = Xch.getUSDTmarket(xc; dt=dt, requestedbases=requestedbases)

    @test nrow(valuationdf) > 0

    valuationcoins = Set(String.(valuationdf[!, :basecoin]))
    @test valuationcoins == Set(requestedbases)
    @test !("AAPLX" in valuationcoins)
    @test Set(String.(compatvaluation[!, :basecoin])) == valuationcoins

    portfolio = Xch.portfolio!(xc, balancesdf, nothing; ignoresmallvolume=false)
    @test "BTC" in Set(String.(portfolio[!, :coin]))
    @test all(ismissing.(portfolio[!, :usdtprice]) .== false)
end

end