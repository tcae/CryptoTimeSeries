module MultilegOrderTest
using Test
using Dates, DataFrames

using EnvConfig, Ohlcv, CryptoXch, TradeAudit

# Unit tests for createocoorder — verifies that the three bracket legs are emitted with:
# - a shared leg_group_id written into the notes of every leg
# - correct leg_label values ("entry", "take_profit", "stop_loss")
# - take-profit and stop-loss legs linked to the entry via correlation_id (parent_order_id chaining)
# - optional signal metadata forwarded to all legs

"Build a minimal single-row OhlcvData for `base` at a given close price."
function _syntheticohlcv(base::String, close::Float32)
    dt = DateTime("2025-01-01T00:00:00")
    df = DataFrame(
        opentime=DateTime[dt],
        open=Float32[close],
        high=Float32[close * 1.001f0],
        low=Float32[close * 0.999f0],
        close=Float32[close],
        basevolume=Float32[100.0f0],
        pivot=Float32[close],
    )
    ohlcv = Ohlcv.defaultohlcv(base)
    ohlcv = Ohlcv.setdataframe!(ohlcv, df)
    Ohlcv.setix!(ohlcv, 1)
    return ohlcv
end

"Return a minimal symbol-info NamedTuple for seeding the CryptoXch symbol info cache."
function _synthsyminfo(base::String, close::Float32)
    symbol = uppercase(base * EnvConfig.cryptoquote)
    return (
        symbol         = symbol,
        status         = "Trading",
        basecoin       = uppercase(base),
        quotecoin      = uppercase(EnvConfig.cryptoquote),
        ticksize       = Float32(0.01),
        baseprecision  = Float32(0.00001),
        quoteprecision = Float32(0.01),
        minbaseqty     = Float32(0.00001),
        minquoteqty    = Float32(1.0),
        innovation     = 0,
    )
end

@testset "createocoorder audit linkage" begin
    oldauditroot = get(ENV, "CTS_AUDIT_ROOT", nothing)
    tmpdir = mktempdir()
    try
        ENV["CTS_AUDIT_ROOT"] = tmpdir
        EnvConfig.init(test)

        xc = CryptoXch.XchCache()

        # Populate synthetic OHLCV data and symbol-info cache so simulation works
        # without a live exchange connection.
        btc_close = 60_000.0f0
        eth_close = 3_000.0f0
        xc.bases["BTC"] = _syntheticohlcv("BTC", btc_close)
        xc.bases["ETH"] = _syntheticohlcv("ETH", eth_close)
        CryptoXch.setsymbolinfocache!(xc, "BTCUSDT", _synthsyminfo("BTC", btc_close))
        CryptoXch.setsymbolinfocache!(xc, "ETHUSDT", _synthsyminfo("ETH", eth_close))

        # ---- long bracket: buy entry, sell take-profit/stop-loss ----
        # entry at current close; TP above, SL below (all within MAXLIMITDELTA=10%).
        # For simulation:
        #   buy fills when low <= limitprice; low=close*0.999=59940
        #   sell fills when high >= limitprice; high=close*1.001=60060
        # entry_price = close: buy limitprice clamped to min(close,close)=60000; low(59940)<=60000 → fills OK
        # take_profit (sell) at close*1.05=63000: high(60060)<63000 → no fill during getorder
        # stop_loss  (sell) at close*0.95=57000: high(60060)<57000? no, 60060>57000 → FILLS → uses cache
        result = CryptoXch.createocoorder(xc, "BTC";
            entry_side=:buy,
            entry_price=btc_close,
            take_profit_price=btc_close * 1.05f0,
            stop_loss_price=btc_close * 0.95f0,
            basequantity=0.01,
            signal_label="longbuy",
            signal_score=0.85,
            strategy_engine="test_engine",
            strategy_config_ref="trenddetector:unit",
        )

        @test !isnothing(result.leg_group_id)
        @test !isempty(result.leg_group_id)
        # All three order ids should be non-nothing (simulation always produces an id)
        @test !isnothing(result.entry_order_id)
        @test !isnothing(result.take_profit_order_id)
        @test !isnothing(result.stop_loss_order_id)
        # All three ids must be distinct
        @test result.entry_order_id != result.take_profit_order_id
        @test result.entry_order_id != result.stop_loss_order_id
        @test result.take_profit_order_id != result.stop_loss_order_id

        # Read all written audit events
        submitted = TradeAudit.AuditEventRow(
            event_type=TradeAudit.ORDER_SUBMITTED,
            environment=string(Symbol(EnvConfig.configmode)),
            run_mode=CryptoXch.auditrunmode(xc),
            exchange=CryptoXch.exchange(xc),
            account_alias=something(CryptoXch.authname(xc), ""),
            asset_class=TradeAudit.crypto,
            instrument_type=TradeAudit.spot_pair,
        )
        auditpath = TradeAudit.auditfile(submitted)
        @test isfile(auditpath)
        events = readlines(auditpath)

        grp      = result.leg_group_id
        entry_id = string(result.entry_order_id)
        tp_id    = string(result.take_profit_order_id)
        sl_id    = string(result.stop_loss_order_id)

        # --- leg_group_id present in all three ORDER_SUBMITTED events ---
        @test any(e -> occursin("leg_group_id=$(grp)", e) && occursin("leg_label=entry", e), events)
        @test any(e -> occursin("leg_group_id=$(grp)", e) && occursin("leg_label=take_profit", e), events)
        @test any(e -> occursin("leg_group_id=$(grp)", e) && occursin("leg_label=stop_loss", e), events)

        # --- signal metadata forwarded to all three legs ---
        for lbl in ("entry", "take_profit", "stop_loss")
            @test any(e ->
                occursin("leg_label=$(lbl)", e) &&
                occursin("\"strategy_engine\":\"test_engine\"", e) &&
                occursin("\"signal_label\":\"longbuy\"", e),
                events)
        end

        # --- take-profit and stop-loss carry correlation_id pointing to entry order ---
        @test any(e ->
            occursin("\"exchange_order_id\":\"$(tp_id)\"", e) &&
            occursin("\"correlation_id\":\"$(entry_id)\"", e),
            events)
        @test any(e ->
            occursin("\"exchange_order_id\":\"$(sl_id)\"", e) &&
            occursin("\"correlation_id\":\"$(entry_id)\"", e),
            events)

        # --- entry leg has correlation_id = its own id (root of the chain, no parent) ---
        entry_events = filter(e -> occursin("\"exchange_order_id\":\"$(entry_id)\"", e), events)
        @test !isempty(entry_events)
        @test any(e -> occursin("\"correlation_id\":\"$(entry_id)\"", e), entry_events)

        # ---- short bracket: sell entry, buy take-profit/stop-loss ----
        result2 = CryptoXch.createocoorder(xc, "ETH";
            entry_side=:sell,
            entry_price=eth_close * 1.05f0,
            take_profit_price=eth_close * 0.95f0,
            stop_loss_price=eth_close * 1.05f0,
            basequantity=0.1,
        )
        @test !isnothing(result2.entry_order_id)
        @test !isnothing(result2.take_profit_order_id)
        @test !isnothing(result2.stop_loss_order_id)
        @test result2.leg_group_id != result.leg_group_id  # each OCO gets its own group id

        # Confirm context was fully cleared after the call
        ctx = get(xc.mc, :audit_event_context, Dict())
        @test !haskey(ctx, :leg_group_id)
        @test !haskey(ctx, :signal_label)

        # Verify the local symbol info cache is populated
        @test haskey(xc.mc, :syminfo_cache)
        @test haskey(xc.mc[:syminfo_cache], "BTCUSDT")
        @test xc.mc[:syminfo_cache]["BTCUSDT"].minbaseqty == 0.00001f0

    finally
        if isnothing(oldauditroot)
            delete!(ENV, "CTS_AUDIT_ROOT")
        else
            ENV["CTS_AUDIT_ROOT"] = oldauditroot
        end
    end
end

end  # module MultilegOrderTest
