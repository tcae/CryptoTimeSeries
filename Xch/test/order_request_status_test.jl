module XchOrderRequestStatusTest
using Test
using Dates
using DataFrames

using EnvConfig, Xch, Targets

function _trade_lo_amount(df::DataFrame)::DataFrame
    if :lo_amount ∉ propertynames(df)
        df[!, :lo_amount] = fill(0f0, nrow(df))
    end
    return df
end

function _trade_lc_amount(df::DataFrame)::DataFrame
    if :lc_amount ∉ propertynames(df)
        df[!, :lc_amount] = fill(0f0, nrow(df))
    end
    return df
end

function _trade_so_amount(df::DataFrame)::DataFrame
    if :so_amount ∉ propertynames(df)
        df[!, :so_amount] = fill(0f0, nrow(df))
    end
    return df
end

function _trade_sc_amount(df::DataFrame)::DataFrame
    if :sc_amount ∉ propertynames(df)
        df[!, :sc_amount] = fill(0f0, nrow(df))
    end
    return df
end

function _apply_trade_amount_contributors!(df::DataFrame)::DataFrame
    for contributor in (_trade_lo_amount, _trade_lc_amount, _trade_so_amount, _trade_sc_amount)
        contributor(df)
    end
    return df
end

@testset "Xch process_order_request and order_status" begin
    EnvConfig.init(test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(5)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    # Make acceptance deterministic: enforce simulation wallet quote buying power.
    bc = xc.bc
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
    price = btcrow[1, :lastprice]
    minqty = Xch.minimumbasequantity(xc, "BTC", price)

    # Accepted long-open request should create an order id and synchronize status columns.
    accepted = DataFrame(
        opentime=[startdt],
        pair=["BTCUSDT"],
        label=[Targets.longopen],
        lo_limit=[price * 0.98f0],
        lo_amount=[max(minqty * 1.5f0, 0.001f0)],
        longleverage=[UInt8(0)],
    )
    for contributor in Xch.tradesdf_contributors()
        contributor(accepted)
    end
    _apply_trade_amount_contributors!(accepted)
    result = Xch.process_order_request(xc, accepted, 1)
    @test result.action == :long_open
    @test result.accepted || result.reason in ("insufficient_free_quote", "amount_not_positive", "below_minimum_qty")
    @test ismissing(accepted[1, :lastopentrade])
    if result.accepted
        @test String(accepted[1, :lo_id]) != "none"
        @test String(accepted[1, :lo_id]) != ""
        @test accepted[1, :lo_status] == "Submitted"

        Xch.order_status(xc, accepted, 1)
        @test accepted[1, :lo_status] != "none"
        @test !ismissing(accepted[1, :equity])
        @test !ismissing(accepted[1, :freequote])
    else
        @test result.reason == "insufficient_free_quote"
        @test accepted[1, :lo_status] == "Rejected"
        @test String(accepted[1, :lo_id]) == "none"
        @test !ismissing(accepted[1, :lo_msg])
    end

    # Too-small quantity should be rejected and assigned a Trading catalog id.
    rejected = DataFrame(
        opentime=[startdt],
        pair=["BTCUSDT"],
        label=[Targets.longopen],
        lo_limit=[price],
        lo_amount=[max(minqty * 0.1f0, 1.0f-8)],
        longleverage=[UInt8(0)],
    )
    for contributor in Xch.tradesdf_contributors()
        contributor(rejected)
    end
    _apply_trade_amount_contributors!(rejected)
    reject_result = Xch.process_order_request(xc, rejected, 1)
    @test !reject_result.accepted
    @test reject_result.reason == "below_minimum_qty"
    @test ismissing(rejected[1, :lastopentrade])
    @test rejected[1, :lo_status] == "Rejected"
    @test !ismissing(rejected[1, :lo_msg])
end

@testset "Xch close amount rounds to holding when dust gap is tiny" begin
    EnvConfig.init(test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(5)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)

    bc = xc.bc
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[1_000f0, 1f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )

    mdf = Xch.getUSDTmarket(xc)
    btcrow = mdf[mdf.basecoin .== "BTC", :]
    price = btcrow[1, :lastprice]
    minqty = Xch.minimumbasequantity(xc, "BTC", price)

    close_req = DataFrame(
        opentime=[startdt],
        pair=["BTCUSDT"],
        label=[Targets.longclose],
        lc_limit=[price * 0.98],
        lc_amount=[1 - (minqty * 0.25)],
        longleverage=[UInt8(0)],
    )
    for contributor in Xch.tradesdf_contributors()
        contributor(close_req)
    end
    _apply_trade_amount_contributors!(close_req)

    close_result = Xch.process_order_request(xc, close_req, 1)
    @test close_result.accepted
    @test close_req[1, :lc_status] == "Submitted"
    @test close_req[1, :lc_filled] == 0f0
end

@testset "Xch direct order builders validate quantity" begin
    EnvConfig.init(test)

    startdt = DateTime("2022-01-01T01:00:00")
    enddt = startdt + Dates.Day(5)
    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.addbase!(xc, "BTC", startdt, enddt)

    mdf = Xch.getUSDTmarket(xc)
    btcrow = mdf[mdf.basecoin .== "BTC", :]
    @test size(btcrow, 1) == 1
    price = btcrow[1, :lastprice]
    minqty = Xch.minimumbasequantity(xc, "BTC", price)

    @test isnothing(Xch.createopenorder(xc, "BTC"; limitprice=price, basequantity=max(minqty * 0.1, 1e-8), maker=false, configside=:long))
    @test_throws ArgumentError Xch.createopenorder(xc, "BTC"; limitprice=price, basequantity=-1, maker=false, configside=:long)
end

@testset "Xch close-before-open sequencing helper" begin
    calls = String[]

    seq = Xch.closebeforeopenflip!(
        () -> begin
            push!(calls, "close")
            "close-oid"
        end,
        () -> begin
            push!(calls, "open")
            "open-oid"
        end,
    )
    @test calls == ["close", "open"]
    @test seq.closeorderid == "close-oid"
    @test seq.openorderid == "open-oid"

    empty!(calls)
    aborted = Xch.closebeforeopenflip!(
        () -> begin
            push!(calls, "close")
            nothing
        end,
        () -> begin
            push!(calls, "open")
            "should-not-run"
        end,
    )
    @test calls == ["close"]
    @test isnothing(aborted.closeorderid)
    @test isnothing(aborted.openorderid)
end

end
