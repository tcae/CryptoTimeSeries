module XchOrderRequestStatusTest
using Test
using Dates
using DataFrames

using EnvConfig, Xch

@testset "Xch process_order_request and order_status" begin
    EnvConfig.init(test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(5)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    # Make acceptance deterministic: enforce simulation wallet quote buying power.
    bc = Xch._routedbc(xc, Xch.trade_exchange_spot)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote],
        free=Float32[1_000f0],
        locked=Float32[0f0],
        borrowed=Float32[0f0],
        accruedinterest=Float32[0f0],
    )

    mdf = Xch.getUSDTmarket(xc)
    @test size(mdf, 1) > 0
    btcrow = mdf[mdf.basecoin .== "BTC", :]
    @test size(btcrow, 1) == 1
    price = Float32(btcrow[1, :lastprice])
    minqty = Float32(Xch.minimumbasequantity(xc, "BTC", price))

    # Accepted long-open request should create an order id and synchronize status columns.
    accepted = DataFrame(
        opentime=[startdt],
        pair=["BTCUSDT"],
        tradelabel=["longopen"],
        longopenlimit=[price * 0.98f0],
        longamount=[max(minqty * 1.5f0, 0.001f0)],
        longleverage=[UInt8(0)],
    )
    for contributor in Xch.tradesdf_contributors()
        contributor(accepted)
    end
    result = Xch.process_order_request(xc, accepted, 1)
    @test result.action == :long_open
    if result.accepted
        @test !ismissing(accepted[1, :longid])
        @test accepted[1, :longstatus] == "Submitted"

        Xch.order_status(xc, accepted, 1)
        @test accepted[1, :longstatus] != "none"
        @test !ismissing(accepted[1, :equity])
        @test !ismissing(accepted[1, :freequote])
    else
        @test result.reason == "insufficient_free_quote"
        @test accepted[1, :longstatus] == "Rejected"
        @test !ismissing(accepted[1, :longmsgid])
    end

    # Too-small quantity should be rejected and assigned a Trading catalog id.
    rejected = DataFrame(
        opentime=[startdt],
        pair=["BTCUSDT"],
        tradelabel=["longopen"],
        longopenlimit=[price],
        longamount=[max(minqty * 0.1f0, 1.0f-8)],
        longleverage=[UInt8(0)],
    )
    for contributor in Xch.tradesdf_contributors()
        contributor(rejected)
    end
    reject_result = Xch.process_order_request(xc, rejected, 1)
    @test !reject_result.accepted
    @test reject_result.reason == "below_minimum_qty"
    @test rejected[1, :longstatus] == "Rejected"
    @test !ismissing(rejected[1, :longmsgid])
end

end
