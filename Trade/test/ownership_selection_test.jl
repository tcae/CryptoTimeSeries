using Test
using Dates
using DataFrames
using EnvConfig, CryptoXch, Trade, TradeAudit, Ohlcv, Targets

"""Write one synthetic spot fill audit event for ownership reconstruction tests."""
function _write_spot_fill!(xc::CryptoXch.XchCache, event_root::AbstractString;
    exchange_name::AbstractString,
    baseasset::AbstractString,
    side::AbstractString,
    fill_base_qty::Real,
    leverage=missing)
    event = TradeAudit.AuditEventRow(
        event_type=TradeAudit.ORDER_FILLED,
        event_time_utc=Dates.now(Dates.UTC),
        created_at_utc=Dates.now(Dates.UTC),
        source_module="TradeTest",
        environment=string(Symbol(EnvConfig.configmode)),
        run_mode=CryptoXch.auditrunmode(xc),
        run_id=CryptoXch.auditrunid(xc),
        exchange=String(exchange_name),
        account_alias=String(exchange_name),
        routing_role=TradeAudit.routing_trade_exchange_spot,
        market_type=TradeAudit.market_spot,
        asset_class=TradeAudit.crypto,
        instrument_type=TradeAudit.spot_pair,
        venue_instrument_type="spot",
        symbol=String(baseasset) * EnvConfig.cryptoquote,
        baseasset=String(baseasset),
        quoteasset=EnvConfig.cryptoquote,
        settlement_asset=EnvConfig.cryptoquote,
        side=String(side),
        fill_base_qty=Float64(fill_base_qty),
        leverage=leverage,
    )
    TradeAudit.writeeventwithhash(event; root=event_root)
    return nothing
end

@testset "ownership-aware selection and sizing" begin
    oldauditroot = get(ENV, "CTS_AUDIT_ROOT", nothing)
    tmpdir = mktempdir()
    try
        ENV["CTS_AUDIT_ROOT"] = tmpdir
        EnvConfig.init(test)

        startdt = DateTime("2026-05-25T12:00:00")
        xc = CryptoXch.XchCache(startdt=startdt, enddt=startdt + Minute(1), exchange=CryptoXch.EXCHANGE_BYBIT)
        tc = Trade.TradeCache(xc=xc, trademode=Trade.closeonly)

        _write_spot_fill!(xc, tmpdir; exchange_name=CryptoXch.EXCHANGE_BYBIT, baseasset="BTC", side="Buy", fill_base_qty=1.2)
        _write_spot_fill!(xc, tmpdir; exchange_name=CryptoXch.EXCHANGE_BYBIT, baseasset="BTC", side="Sell", fill_base_qty=0.4)
        _write_spot_fill!(xc, tmpdir; exchange_name=CryptoXch.EXCHANGE_BYBIT, baseasset="ETH", side="Sell", fill_base_qty=0.7, leverage=2)
        _write_spot_fill!(xc, tmpdir; exchange_name=CryptoXch.EXCHANGE_KRAKENSPOT, baseasset="BTC", side="Buy", fill_base_qty=5.0)

        owned = Trade._robotownedqtymap(tc, ["BTC", "ETH"])
        @test isapprox(owned["BTC"].longqty, 0.8f0; atol=1f-5)
        @test isapprox(owned["BTC"].shortqty, 0f0; atol=1f-5)
        @test isapprox(owned["ETH"].longqty, 0f0; atol=1f-5)
        @test isapprox(owned["ETH"].shortqty, 0.7f0; atol=1f-5)

        tc.cfg = DataFrame(
            basecoin=["BTC", "ETH", "XRP"],
            minquotevol=Bool[true, true, true],
            continuousminvol=Bool[true, true, true],
            classifieraccepted=Bool[true, true, false],
            whitelisted=Bool[true, false, true],
            inportfolio=Bool[false, false, false],
            buyenabled=Bool[false, false, false],
            sellenabled=Bool[false, false, false],
        )
        Trade._annotate_robotownership!(tc)
        Trade._sync_tradeflags!(tc)
        @test tc.cfg[tc.cfg.basecoin .== "BTC", :buyenabled][1]
        @test tc.cfg[tc.cfg.basecoin .== "BTC", :sellenabled][1]
        @test !tc.cfg[tc.cfg.basecoin .== "ETH", :buyenabled][1]
        @test tc.cfg[tc.cfg.basecoin .== "ETH", :sellenabled][1]
        @test !tc.cfg[tc.cfg.basecoin .== "XRP", :sellenabled][1]

        ohlcv = Ohlcv.OhlcvData(
            DataFrame(
                opentime=[startdt, startdt + Minute(1)],
                open=Float32[100, 100],
                high=Float32[101, 101],
                low=Float32[99, 99],
                close=Float32[100, 100],
                basevolume=Float32[10, 10],
                pivot=Float32[100, 100],
            ),
            "BTC",
            uppercase(EnvConfig.cryptoquote),
            "1m",
            2,
            nothing,
        )
        CryptoXch.addbase!(xc, ohlcv)
        xc.currentdt = startdt + Minute(1)

        basecfgdf = DataFrame(
            basecoin=["BTC"],
            buyenabled=[false],
            sellenabled=[true],
            robotownedlongqty=Float32[0.8],
            robotownedshortqty=Float32[0.0],
        )
        assets = DataFrame(
            coin=[EnvConfig.cryptoquote, "BTC"],
            locked=Float32[0, 0],
            free=Float32[1000, 2.0],
            borrowed=Float32[0, 0],
            accruedinterest=Float32[0, 0],
            usdtprice=Float32[1, 100],
            usdtvalue=Float32[1000, 200],
        )
        advice = Trade.StrategyAdvice(
            classifier=tc.cl,
            configid=0,
            tradelabel=Targets.longclose,
            relativeamount=1f0,
            base="BTC",
            price=nothing,
            datetime=xc.currentdt,
            hourlygain=0f0,
            probability=1f0,
            investmentid=nothing,
            source=:classifier,
            allowreversal=true,
        )
        Trade.trade!(tc, basecfgdf[1, :], advice, assets)
        @test nrow(tc.dbgdf) > 0
        @test isapprox(Float32(tc.dbgdf[end, :baseqty]), 0.8f0; atol=1f-5)
    finally
        if isnothing(oldauditroot)
            delete!(ENV, "CTS_AUDIT_ROOT")
        else
            ENV["CTS_AUDIT_ROOT"] = oldauditroot
        end
        rm(tmpdir; force=true, recursive=true)
    end
end