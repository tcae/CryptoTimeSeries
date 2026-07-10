module XchSyncLatestTradesRowsTest
using Test
using Dates
using DataFrames
using CategoricalArrays: CategoricalVector

using EnvConfig, Ohlcv, Xch

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

@testset "Xch sync_latest_trades_rows! uses current cache snapshots" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(2)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), [_trade_lo_amount, _trade_lc_amount, _trade_so_amount, _trade_sc_amount]))
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.addbase!(xc, "ETH", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = xc.bc
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC", "ETH"],
        free=Float32[5_000f0, 1.5f0, 0.75f0],
        locked=Float32[0f0, 0f0, 0f0],
        borrowed=Float32[0f0, 0.25f0, 0f0],
        accruedinterest=Float32[0f0, 0f0, 0f0],
    )

    empty!(bc.orders)
    push!(bc.orders, (
        orderid="oid-lo-filled",
        symbol="BTCUSDT",
        side="Buy",
        baseqty=1.0f0,
        ordertype="Limit",
        isLeverage=false,
        timeinforce="GTC",
        limitprice=100.0f0,
        avgprice=100.5f0,
        executedqty=1.0f0,
        status="Filled",
        created=currentdt,
        updated=currentdt,
        rejectreason="NO ERROR",
        lastcheck=currentdt,
        marginleverage=Int32(0),
        reduceonly=false,
    ))
    push!(bc.orders, (
        orderid="oid-lc-open",
        symbol="BTCUSDT",
        side="Sell",
        baseqty=0.5f0,
        ordertype="Limit",
        isLeverage=false,
        timeinforce="GTC",
        limitprice=101.0f0,
        avgprice=101.5f0,
        executedqty=0.25f0,
        status="PartiallyFilled",
        created=currentdt,
        updated=currentdt,
        rejectreason="NO ERROR",
        lastcheck=currentdt,
        marginleverage=Int32(0),
        reduceonly=true,
    ))
    push!(bc.orders, (
        orderid="oid-sc-rejected",
        symbol="BTCUSDT",
        side="Buy",
        baseqty=0.3f0,
        ordertype="Limit",
        isLeverage=false,
        timeinforce="GTC",
        limitprice=99.0f0,
        avgprice=0f0,
        executedqty=0f0,
        status="Rejected",
        created=currentdt,
        updated=currentdt,
        rejectreason="manual rejection",
        lastcheck=currentdt,
        marginleverage=Int32(0),
        reduceonly=true,
    ))

    btcdf = Xch.trades(xc, "BTC", EnvConfig.pairquote)
    push!(btcdf, (
        opentime=currentdt - Dates.Minute(1),
        pair="BTCUSDT",
        lp_amount=1.0f0,
        sp_amount=0.25f0,
        lastopentrade=currentdt - Dates.Minute(1),
    ); cols=:subset)
    push!(btcdf, (
        opentime=currentdt,
        pair="BTCUSDT",
        lo_id="oid-lo-filled",
        lo_amount=1.0f0,
        lc_id="oid-lc-open",
        lc_amount=0.5f0,
        sc_id="oid-sc-rejected",
        sc_amount=0.3f0,
        lastopentrade=missing,
    ); cols=:subset)

    ethdf = Xch.trades(xc, "ETH", EnvConfig.pairquote)
    push!(ethdf, (
        opentime=currentdt - Dates.Minute(1),
        pair="ETHUSDT",
        lp_amount=0.75f0,
        sp_amount=0f0,
        lastopentrade=currentdt - Dates.Minute(1),
    ); cols=:subset)
    push!(ethdf, (
        opentime=currentdt,
        pair="ETHUSDT",
        lastopentrade=missing,
    ); cols=:subset)

    oodf = Xch.getopenorders(xc)
    @test :avgprice in Symbol.(names(oodf))

    orderinfo = Xch.getorder(xc, "oid-lo-filled"; auditevent=false)
    @test hasproperty(orderinfo, :avgprice)

    rowsbybase = Xch.sync_latest_trades_rows!(xc)
    @test Set(keys(rowsbybase)) == Set(["BTC", "ETH"])

    btcrowix = rowsbybase["BTC"].rowix
    ethrowix = rowsbybase["ETH"].rowix
    btcrow = rowsbybase["BTC"].tradesdf
    ethrow = rowsbybase["ETH"].tradesdf

    btcohlcv = Xch.getohlcv(xc, "BTC")
    btcodf = Ohlcv.dataframe(btcohlcv)
    btcoix = Ohlcv.ix(btcohlcv)
    @test btcrow[btcrowix, :opentime] == btcodf[btcoix, :opentime]
    @test btcrow[btcrowix, :low] == Float32(btcodf[btcoix, :low])
    @test btcrow[btcrowix, :high] == Float32(btcodf[btcoix, :high])
    @test btcrow[btcrowix, :close] == Float32(btcodf[btcoix, :close])
    @test btcrow[btcrowix, :lp_amount] == 1.5f0
    @test btcrow[btcrowix, :sp_amount] == 0.25f0
    @test btcrow[btcrowix, :lastopentrade] == btcrow[btcrowix, :opentime]
    @test btcrow[btcrowix, :lo_status] == "Filled"
    @test btcrow[btcrowix, :lc_status] == "PartiallyFilled"
    @test btcrow[btcrowix, :sc_status] == "Rejected"
    @test btcrow[btcrowix, :lo_filled] == 1.0f0
    @test btcrow[btcrowix, :lc_filled] == 0.25f0
    @test btcrow[btcrowix, :sc_filled] == 0f0
    @test btcrow[btcrowix, :lo_pavg] == 100.5f0
    @test btcrow[btcrowix, :lc_pavg] == 101.5f0
    @test !ismissing(btcrow[btcrowix, :sc_msg])

    acct = Xch.account_status(xc; force_refresh=true, ttl_seconds=0)
    @test btcrow[btcrowix, :maintmargin] == Float32(acct.maintenance_margin_quote)
    @test btcrow[btcrowix, :equity] == Float32(acct.equity_quote)
    @test btcrow[btcrowix, :balance] == Float32(acct.free_quote)
    @test btcrow[btcrowix, :freemargin] == Float32(acct.free_margin_quote)
    @test btcrow[btcrowix, :freequote] == Float32(acct.free_quote)

    @test ethrow[ethrowix, :lp_amount] == 0.75f0
    @test ethrow[ethrowix, :sp_amount] == 0f0
    @test ethrow[ethrowix, :lastopentrade] == currentdt - Dates.Minute(1)
end

@testset "Xch sync_latest_trades_rows! appends row when OHLCV advanced" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(3)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), [_trade_lo_amount, _trade_lc_amount, _trade_so_amount, _trade_sc_amount]))
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = xc.bc
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[2_000f0, 0.5f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orders)

    btcdf = Xch.trades(xc, "BTC", EnvConfig.pairquote)
    push!(btcdf, (
        opentime=currentdt - Dates.Minute(1),
        pair="BTCUSDT",
        lastopentrade=currentdt - Dates.Minute(1),
    ); cols=:subset)

    prevrows = nrow(btcdf)
    rowsbybase = Xch.sync_latest_trades_rows!(xc)
    @test haskey(rowsbybase, "BTC")

    btcrowix = rowsbybase["BTC"].rowix
    btcrow = rowsbybase["BTC"].tradesdf
    @test nrow(btcrow) == prevrows + 1
    @test btcrowix == nrow(btcrow)

    btcohlcv = Xch.getohlcv(xc, "BTC")
    btcodf = Ohlcv.dataframe(btcohlcv)
    btcoix = Ohlcv.ix(btcohlcv)
    @test btcrow[btcrowix, :opentime] == btcodf[btcoix, :opentime]
end

@testset "Xch sync_latest_trades_rows! creates missing pair entry from requested pairs" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(2)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), [_trade_lo_amount, _trade_lc_amount, _trade_so_amount, _trade_sc_amount]))
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = xc.bc
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[1_000f0, 0.25f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orders)

    @test !Xch.hastrades(xc, "BTCUSDT")
    rowsbybase = Xch.sync_latest_trades_rows!(xc, ["BTCUSDT"])

    @test Xch.hastrades(xc, "BTCUSDT")
    @test haskey(rowsbybase, "BTC")

    btcrow = rowsbybase["BTC"].tradesdf
    btcrowix = rowsbybase["BTC"].rowix
    @test nrow(btcrow) == 1
    @test btcrowix == 1

    btcohlcv = Xch.getohlcv(xc, "BTC")
    btcodf = Ohlcv.dataframe(btcohlcv)
    btcoix = Ohlcv.ix(btcohlcv)
    @test btcrow[1, :pair] == "BTCUSDT"
    @test btcrow[1, :opentime] == btcodf[btcoix, :opentime]
end

@testset "Xch sync_latest_trades_rows! accepts categorical pair vector" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(2)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt)
    Xch.ensuretradesschema(xc, vcat(Xch.tradesdf_contributors(), [_trade_lo_amount, _trade_lc_amount, _trade_so_amount, _trade_sc_amount]))
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = xc.bc
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[1_000f0, 0.25f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orders)

    pairs = CategoricalVector(["BTCUSDT"])
    rowsbybase = Xch.sync_latest_trades_rows!(xc, pairs)

    @test haskey(rowsbybase, "BTC")
    @test Xch.hastrades(xc, "BTCUSDT")
    @test rowsbybase["BTC"].tradesdf[1, :pair] == "BTCUSDT"
end

end