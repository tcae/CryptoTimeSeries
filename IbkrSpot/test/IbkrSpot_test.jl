using IbkrSpot, Test, Dates
using XchAdapter
using JSON3

# Offline tests: no TWS connection required.

@testset "IbkrSpot execution config" begin
    cfg = IbkrSpot.executionconfig()
    @test String(cfg["exchange"]) == "IbkrSpot"

    conn = IbkrSpot._connectionconfig()
    @test conn.host == "127.0.0.1"
    @test conn.port in (7496, 7497, 4001, 4002)
    @test conn.clientid > 0
    @test conn.currency == "USD"
    @test conn.exchange_route == "SMART"
    withenv("IBKR_PORT" => "7496") do
        @test IbkrSpot._connectionconfig().port == 7496
    end

    long = IbkrSpot._executionorderspec(:long)
    @test long.side == :long
    @test long.instrument == "stock"
    @test long.max_quote > 0

    short = IbkrSpot._executionorderspec(:short)
    @test short.side == :short
    # IBKR nets positions per contract; a short is a margin sale of the same stock contract.
    @test short.instrument == "stock_margin"
    @test short.leverage >= 1
    @test short.max_quote > 0

    @test_throws ErrorException IbkrSpot._executionorderspec(:sideways)
end

@testset "IbkrSpot adapter core" begin
    cache = IbkrSpot.IbkrSpotCache()
    @test cache isa XchAdapter.XchAdapterCache
    @test XchAdapter.exchangeid(cache) == "IbkrSpot"
    @test IbkrSpot.symboltoken(cache, "aapl") == "AAPLUSD"
    @test IbkrSpot.executionorderspec(cache, :long).instrument == "stock"
    @test cache.minimum_daily_usd_volume == 20_000_000.0
    configured_symbols = [instrument.symbol for instrument in cache.market_scan_symbols]
    @test all(symbol -> symbol in configured_symbols, ["AAPLUSD", "MSFTUSD"])
    @test_throws ErrorException IbkrSpot.get24h(cache; symbols=["NOTCONFIGUREDUSD"])
    contract = IbkrSpot._stock_contract(cache, "AAPLUSD")
    @test contract.symbol == "AAPL"
    @test contract.currency == "USD"
    @test contract.exchange == "SMART"
    @test contract.primaryExchange == "NASDAQ"
    @test isempty(IbkrSpot.emptyorders(cache))
    @test names(IbkrSpot.emptyorders(cache)) == [
        "orderid", "orderLinkId", "symbol", "side", "baseqty", "ordertype",
        "isLeverage", "timeinforce", "limitprice", "avgprice", "executedqty",
        "status", "created", "updated", "rejectreason", "reduceonly", "lastcheck",
    ]
    @test XchAdapter.normalize_order_status(cache, "Submitted") == "submitted"
    @test XchAdapter.normalize_order_status(cache, "Filled") == "closed"
    @test XchAdapter.normalize_order_status(cache, "Cancelled") == "cancelled"
    order = IbkrSpot._neworder(cache, 12, "buy", 2.0, 100.25, :long; maker=true)
    @test order.orderId == 12
    @test order.action == "BUY"
    @test order.orderType == "LMT"
    @test order.totalQuantity == 2.0
    @test order.lmtPrice == 100.25
    @test order.postOnly
    @test order.orderRef == "IbkrSpot|long|12|open"
    @test IbkrSpot._tickkey("BID_PRICE") == "BID"
    @test IbkrSpot._tickkey("8") == "VOLUME"
    @test IbkrSpot._bar_datetime("1790343000") == DateTime(2026, 9, 25, 13, 30)
    @test IbkrSpot._exchange_market_date(DateTime(2026, 9, 28, 0, 30), "US/Eastern") == Date(2026, 9, 27)
    bars = [
        (time="20260925", volume=100.0, wap=10.0),
        (time="20260928", volume=200.0, wap=11.0),
    ]
    previous_session = IbkrSpot._latest_completed_daily_bar(bars, Date(2026, 9, 28))
    @test previous_session.date == Date(2026, 9, 25)
    @test previous_session.bar.volume * previous_session.bar.wap == 1000.0
    @test isnothing(IbkrSpot._latest_completed_daily_bar(bars[2:2], Date(2026, 9, 28)))
end

@testset "IbkrSpot watchlist import" begin
    mktempdir() do directory
        write(joinpath(directory, "Watchlist USD.csv"), "DES,AAPL,STK,SMART/AMEX\nDES,NEO,STK,SMART/TSE\nDES,TSLA,STK,SMART/NASDAQ\nDES,SPY,CASH,IDEALPRO\n")
        configpath = joinpath(directory, "market_scan_config.json")
        fixture = Dict{String, Any}(
            "version" => 1,
            "universe" => Dict{String, Any}(
                "mode" => "whitelist",
                "symbols" => [
                    Dict{String, String}(
                        "base_symbol" => "AAPL",
                        "quote_currency" => "USD",
                        "exchange" => "SMART",
                        "primary_exchange" => "NASDAQ",
                    ),
                    Dict{String, String}(
                        "base_symbol" => "NEO",
                        "quote_currency" => "CAD",
                        "exchange" => "SMART",
                        "primary_exchange" => "TSE",
                    ),
                ],
            ),
            "minimum_daily_usd_volume" => 20_000_000.0,
        )
        open(configpath, "w") do io
            JSON3.pretty(io, fixture)
        end

        imported = IbkrSpot.addwatchlists!(["Watchlist USD"]; watchlist_directory=directory, config_path=configpath)
        @test imported == (added=1, duplicates=1, quote_conflicts=1, skipped_nonstocks=1)
        config = JSON3.read(read(configpath, String))
        symbols = config["universe"]["symbols"]
        tsla = only(filter(entry -> entry["base_symbol"] == "TSLA", symbols))
        @test tsla["quote_currency"] == "USD"
        @test tsla["exchange"] == "SMART"
        @test tsla["primary_exchange"] == ""
        @test config["minimum_daily_usd_volume"] == 20_000_000.0

        repeated = IbkrSpot.addwatchlists!(["Watchlist USD"]; watchlist_directory=directory, config_path=configpath)
        @test repeated == (added=0, duplicates=2, quote_conflicts=1, skipped_nonstocks=1)
        @test count(entry -> entry["base_symbol"] == "TSLA", config["universe"]["symbols"]) == 1
    end
end

@testset "IbkrSpot request correlation" begin
    @testset "ids are unique and increasing" begin
        ids = [IbkrSpot._nextreqid() for _ in 1:100]
        @test length(unique(ids)) == 100
        @test issorted(ids)
    end

    @testset "collect then complete returns buffered values" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid, "a")
        IbkrSpot._push!(reqid, "b")
        IbkrSpot._complete!(reqid)
        @test IbkrSpot._await(reqid, "unit test") == ["a", "b"]
    end

    @testset "completion with no values returns empty" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._complete!(reqid)
        @test isempty(IbkrSpot._await(reqid, "unit test"))
    end

    @testset "unsolicited ids are ignored" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid + 1, "stray")
        IbkrSpot._push!(reqid, "kept")
        IbkrSpot._complete!(reqid)
        @test IbkrSpot._await(reqid, "unit test") == ["kept"]
    end

    @testset "TWS request error throws instead of returning partial data" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._push!(reqid, "partial")
        IbkrSpot._fail!(reqid, "code=200 No security definition found")
        @test_throws ErrorException IbkrSpot._await(reqid, "contract details")
    end

    @testset "missing response times out instead of hanging" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        @test_throws ErrorException IbkrSpot._await(reqid, "unit test"; timeout=Dates.Millisecond(150))
    end

    @testset "await releases the buffer" begin
        reqid = IbkrSpot._nextreqid()
        IbkrSpot._opencollector!(reqid)
        IbkrSpot._complete!(reqid)
        IbkrSpot._await(reqid, "unit test")
        # a second await on a released id must not silently succeed with stale data
        @test_throws ErrorException IbkrSpot._await(reqid, "unit test"; timeout=Dates.Millisecond(100))
    end
end

@testset "IbkrSpot order id allocation" begin
    # Without a connection TWS has not supplied nextValidId, so allocation must fail fast
    # rather than invent an id that the venue would reject.
    if IbkrSpot._nextvalidid[] == 0
        @test_throws AssertionError IbkrSpot._takeorderid()
    else
        first = IbkrSpot._takeorderid()
        @test IbkrSpot._takeorderid() == first + 1
    end
end
