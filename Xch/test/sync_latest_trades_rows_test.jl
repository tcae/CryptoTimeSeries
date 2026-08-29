module XchSyncLatestTradesRowsTest
using Test
using Dates
using DataFrames
using CategoricalArrays: CategoricalVector

using Bybit, EnvConfig, Ohlcv, Xch, Targets
using TSM

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

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    TSM.trades(xc.tsm, "BTC", EnvConfig.pairquote)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.addbase!(xc, "ETH", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC", "BTC", "ETH"],
        side=String["quote", "long", "short", "long"],
        free=Float32[5_000f0, 1.5f0, 0.25f0, 0.75f0],
        locked=Float32[0f0, 0f0, 0f0, 0f0],
    )

    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)
    Bybit._simappendorder!(bc, (
        orderid="oid-lo-filled",
        symbol="BTCUSDT",
        side="Buy",
        positionside="long",
        lane="lo",
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
    Bybit._simappendorder!(bc, (
        orderid="oid-lc-open",
        symbol="BTCUSDT",
        side="Sell",
        positionside="long",
        lane="lc",
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
    Bybit._simappendorder!(bc, (
        orderid="oid-sc-rejected",
        symbol="BTCUSDT",
        side="Buy",
        positionside="short",
        lane="sc",
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

    btcrow_prev = TSM.ensuretradesrow!(xc.tsm, "BTC", EnvConfig.pairquote, currentdt - Dates.Minute(1))
    btcdf = btcrow_prev.tradesdf
    btcdf[btcrow_prev.rowix, :label] = ignore
    btcdf[btcrow_prev.rowix, :lp_amount] = 1.0f0
    btcdf[btcrow_prev.rowix, :sp_amount] = 0.25f0
    btcdf[btcrow_prev.rowix, :lastopentrade] = currentdt - Dates.Minute(1)

    btcrow_now = TSM.ensuretradesrow!(xc.tsm, "BTC", EnvConfig.pairquote, currentdt)
    btcdf = btcrow_now.tradesdf
    btcdf[btcrow_now.rowix, :label] = ignore
    btcdf[btcrow_now.rowix, :lo_id] = "oid-lo-filled"
    btcdf[btcrow_now.rowix, :lo_amount] = 1.0f0
    btcdf[btcrow_now.rowix, :lc_id] = "oid-lc-open"
    btcdf[btcrow_now.rowix, :lc_amount] = 0.5f0
    btcdf[btcrow_now.rowix, :sc_id] = "oid-sc-rejected"
    btcdf[btcrow_now.rowix, :sc_amount] = 0.3f0
    btcdf[btcrow_now.rowix, :lastopentrade] = missing

    ethrow_prev = TSM.ensuretradesrow!(xc.tsm, "ETH", EnvConfig.pairquote, currentdt - Dates.Minute(1))
    ethdf = ethrow_prev.tradesdf
    ethdf[ethrow_prev.rowix, :label] = ignore
    ethdf[ethrow_prev.rowix, :lp_amount] = 0.75f0
    ethdf[ethrow_prev.rowix, :sp_amount] = 0f0
    ethdf[ethrow_prev.rowix, :lastopentrade] = currentdt - Dates.Minute(1)

    ethrow_now = TSM.ensuretradesrow!(xc.tsm, "ETH", EnvConfig.pairquote, currentdt)
    ethdf = ethrow_now.tradesdf
    ethdf[ethrow_now.rowix, :label] = ignore
    ethdf[ethrow_now.rowix, :lastopentrade] = missing

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
    @test btcrow[btcrowix, :low] == (btcodf[btcoix, :low])
    @test btcrow[btcrowix, :high] == (btcodf[btcoix, :high])
    @test btcrow[btcrowix, :close] == (btcodf[btcoix, :close])
    @test btcrow[btcrowix, :lp_amount] == 1.5f0
    @test btcrow[btcrowix, :sp_amount] == 0.25f0
    @test btcrow[btcrowix, :lastopentrade] == btcrow[btcrowix, :opentime]
    @test lowercase(String(btcrow[btcrowix, :lo_status])) == "closed"
    @test lowercase(String(btcrow[btcrowix, :lc_status])) == "submitted"
    @test lowercase(String(btcrow[btcrowix, :sc_status])) == "rejected"
    @test btcrow[btcrowix, :lol_filled] == 1.0f0
    @test btcrow[btcrowix, :lcl_filled] == 0.25f0
    @test btcrow[btcrowix, :scl_filled] == 0f0
    @test btcrow[btcrowix, :lol_pavg] == 100.5f0
    @test btcrow[btcrowix, :lcl_pavg] == 101.5f0
    @test !ismissing(btcrow[btcrowix, :sc_msg])

    acct = Xch.account_status(xc; force_refresh=true, ttl_seconds=0)
    @test isapprox(btcrow[btcrowix, :equity], acct.equity_quote; rtol=1f-6, atol=1f-6)
    @test isapprox(btcrow[btcrowix, :freemargin], acct.free_margin_quote; rtol=1f-6, atol=1f-6)
    @test isapprox(btcrow[btcrowix, :freequote], acct.free_quote; rtol=1f-6, atol=1f-6)

    @test ethrow[ethrowix, :lp_amount] == 0.75f0
    @test ethrow[ethrowix, :sp_amount] == 0f0
    @test ethrow[ethrowix, :lastopentrade] == currentdt - Dates.Minute(1)
end

@testset "Xch sync_latest_trades_rows! appends row when OHLCV advanced" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(3)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[2_000f0, 0.5f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    btcrow_prev = TSM.ensuretradesrow!(xc.tsm, "BTC", EnvConfig.pairquote, currentdt - Dates.Minute(1))
    btcdf = btcrow_prev.tradesdf
    btcdf[btcrow_prev.rowix, :lastopentrade] = currentdt - Dates.Minute(1)

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

@testset "Xch sync_latest_trades_rows! carries lol_pavg forward while position stays open" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(3)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[2_000f0, 0.5f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    btcrow_prev = TSM.ensuretradesrow!(xc.tsm, "BTC", EnvConfig.pairquote, currentdt - Dates.Minute(1))
    btcdf = btcrow_prev.tradesdf
    btcdf[btcrow_prev.rowix, :lastopentrade] = currentdt - Dates.Minute(1)
    TSM.settrades_lp_amount!(btcdf, btcrow_prev.rowix, 0.5f0)
    TSM.settrades_last_pavg!(btcdf, btcrow_prev.rowix, Targets.longopen, 100.0f0)

    rowsbybase = Xch.sync_latest_trades_rows!(xc)
    btcrowix = rowsbybase["BTC"].rowix
    btcrow = rowsbybase["BTC"].tradesdf
    @test btcrow[btcrowix, :lp_amount] == 0.5f0
    @test btcrow[btcrowix, :lol_pavg] == 100.0f0
end

@testset "Xch account_status does not double-count a fully-reserved long position" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(1)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    # The whole BTC position sits in :locked (e.g. fully reserved for a pending
    # reduce-only stop-loss close order) - positionsnapshot's long_qty (free+locked)
    # must not be added on top of the already-reserved :locked share again.
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        side=String["quote", "long"],
        free=Float32[500.5f0, 0f0],
        locked=Float32[0f0, 0.5f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    price = Ohlcv.dataframe(Xch.getohlcv(xc, "BTC"))[Ohlcv.ix(Xch.getohlcv(xc, "BTC")), :close]
    acct = Xch.account_status(xc; force_refresh=true, ttl_seconds=0)
    @test isapprox(acct.equity_quote, 500.5 + 0.5 * price; rtol=1f-3)
end

@testset "Xch sync_latest_trades_rows! creates missing pair entry from requested pairs" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-01-01T00:00:00")
    enddt = startdt + Dates.Day(1)
    currentdt = startdt + Dates.Minute(2)

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    TSM.trades(xc.tsm, "BTC", EnvConfig.pairquote)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[1_000f0, 0.25f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    @test TSM.haspairstate(xc.tsm, "BTCUSDT")
    rowsbybase = Xch.sync_latest_trades_rows!(xc, ["BTCUSDT"])

    @test TSM.haspairstate(xc.tsm, "BTCUSDT")
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

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    TSM.trades(xc.tsm, "BTC", EnvConfig.pairquote)
    Xch.addbase!(xc, "BTC", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "BTC"],
        free=Float32[1_000f0, 0.25f0],
        locked=Float32[0f0, 0f0],
        borrowed=Float32[0f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    pairs = CategoricalVector(["BTCUSDT"])
    rowsbybase = Xch.sync_latest_trades_rows!(xc, pairs)

    @test haskey(rowsbybase, "BTC")
    @test TSM.haspairstate(xc.tsm, "BTCUSDT")
    @test rowsbybase["BTC"].tradesdf[1, :pair] == "BTCUSDT"
end

@testset "Xch sync_latest_trades_rows! values short exposure without liability discount" begin
    EnvConfig.init(EnvConfig.test)

    startdt = DateTime("2025-07-01T08:30:00")
    enddt = startdt + Dates.Hour(1)
    currentdt = DateTime("2025-07-01T08:32:00")

    xc = Xch.XchCache(startdt=startdt, enddt=enddt, exchange=Xch.EXCHANGE_BYBITSIM)
    TSM.trades(xc.tsm, "DOUBLESINE", EnvConfig.pairquote)
    Xch.addbase!(xc, "DOUBLESINE", startdt, enddt)
    Xch.setcurrenttime!(xc, currentdt)

    bc = Xch.rawcache(xc.bc)
    bc.assets = DataFrame(
        coin=String[EnvConfig.pairquote, "DOUBLESINE"],
        free=Float32[1000f0, 0f0],
        locked=Float32[500.5f0, 0f0],
        borrowed=Float32[0f0, 249.40323f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    empty!(bc.orderbook)
    Bybit._simrebuildorderindexes!(bc)

    rowsbybase = Xch.sync_latest_trades_rows!(xc, ["DOUBLESINEUSDT"])
    @test haskey(rowsbybase, "DOUBLESINE")

    tradesdf = rowsbybase["DOUBLESINE"].tradesdf
    rowix = rowsbybase["DOUBLESINE"].rowix
    acct = Xch.account_status(xc; force_refresh=true, ttl_seconds=0)

    @test acct.equity_quote > 1_500f0
    @test tradesdf[rowix, :equity] == acct.equity_quote
    @test tradesdf[rowix, :freemargin] == acct.free_margin_quote
    @test tradesdf[rowix, :freequote] == acct.free_quote
    @test tradesdf[rowix, :freequote] == 1000f0
    @test tradesdf[rowix, :equity] > tradesdf[rowix, :freequote]
end

end