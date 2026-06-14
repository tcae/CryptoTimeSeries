
module XchProductionTest
using Dates, DataFrames
using Test

using Ohlcv, EnvConfig, Xch, TestOhlcv

const RUN_PRODUCTION_TESTS = lowercase(get(ENV, "CTS_RUN_PRODUCTION_TESTS", "false")) in ("1", "true", "yes")
const RUN_KRAKEN_ORDER_TESTS = lowercase(get(ENV, "CTS_RUN_KRAKEN_ORDER_TESTS", "false")) in ("1", "true", "yes")

function _run_kraken_order_lifecycle!(exchange::String)
    xc = Xch.XchCache(exchange=exchange)
    mdf = Xch.getUSDTmarket(xc)
    @test nrow(mdf) > 0

    row = mdf[1, :]
    base = String(row.basecoin)
    quotecoin = String(EnvConfig.pairquote)
    mid = Float32(max(row.lastprice, row.bidprice, row.askprice))
    # Post-only far-from-touch price to avoid accidental fills in tests.
    limit = mid * 0.75f0
    minqty = Float32(Xch.minimumbasequantity(xc, base, limit))
    amount = max(minqty * 1.25f0, minqty + 1.0f-6)

    req = DataFrame(
        pair=[string(base, quotecoin)],
        tradelabel=["longopen"],
        longopenlimit=[limit],
        longamount=[amount],
        longleverage=[UInt8(0)],
    )

    first_result = Xch.process_order_request(xc, req, 1)
    @test first_result.accepted
    @test first_result.action == :long_open
    @test !ismissing(req[1, :longid])
    oid = String(req[1, :longid])

    Xch.order_status(xc, req, 1)
    @test req[1, :longstatus] != "none"

    # Re-submit longopen for same pair/side to exercise amend path through process_order_request.
    req[1, :longamount] = amount
    second_result = Xch.process_order_request(xc, req, 1)
    @test second_result.accepted
    @test second_result.action == :long_open
    @test !ismissing(req[1, :longid])

    oid2 = Xch.cancelorder(xc, base, oid)
    @test oid2 == oid
    Xch.order_status(xc, req, 1)
    @test lowercase(String(req[1, :longstatus])) in ["cancelled", "canceled", "pendingcancel", "pending_cancel"]
end


# EnvConfig.init(test)  # test Xch Testnet production
if RUN_PRODUCTION_TESTS
    EnvConfig.init(production)  # test Xch real production
end
println("XchTest runtests")

if RUN_PRODUCTION_TESTS
@testset "Xch production tests" begin
    Xch.verbosity =0

    xc = Xch.XchCache()
    # df = Xch.klines2jdf(xc, missing)
    # @test nrow(df) == 0
    mdf = Xch.getUSDTmarket(xc)
    @test size(mdf, 1) > 100
    # println(mdf)
    @test all([col in ["basecoin", "quotevolume24h", "pricechangepercent", "lastprice", "askprice", "bidprice"] for col in names(mdf)])
    @test nrow(mdf) > 10

    EnvConfig.init(production; newdatafolder=true) #! stay with newdatafolder because deleting the data is part of it
    # EnvConfig.init(EnvConfig.production)
    ohlcv = Ohlcv.defaultohlcv("btc")

    testcoins = TestOhlcv.testbasecoin()
    for tc in testcoins
        @test Xch.validbase(xc, tc)
        ohlcv = Xch.cryptodownload(xc, tc, "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
        # println(ohlcv)
        @test size(Ohlcv.dataframe(ohlcv), 1) == 20
        @test all([name in names(Ohlcv.dataframe(ohlcv)) for name in names(Ohlcv.defaultohlcvdataframe())])
    end

    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:45:03"), DateTime("2022-01-02T22:49:35"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5
    @test names(Ohlcv.dataframe(ohlcv)) == ["opentime", "open", "high", "low", "close", "basevolume", "pivot"]

    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:45:01"), DateTime("2022-01-02T22:49:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 5

    Ohlcv.write(ohlcv)
    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:48:01"), DateTime("2022-01-02T22:51:55"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 7

    Ohlcv.write(ohlcv)
    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 9

    Ohlcv.write(ohlcv)
    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:53:03"), DateTime("2022-01-02T22:55:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 3  # not using canned data if no overlap

    Ohlcv.write(ohlcv)
    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:40:03"), DateTime("2022-01-02T22:41:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 2  # not using canned data if no overlap

    Ohlcv.write(ohlcv)
    ohlcv = Xch.cryptodownload(xc, "btc", "1m", DateTime("2022-01-02T22:38:03"), DateTime("2022-01-02T22:57:45"))
    # println(Ohlcv.dataframe(ohlcv))
    @test size(Ohlcv.dataframe(ohlcv), 1) == 20

    ohlcv1 = Ohlcv.copy(ohlcv)
    Ohlcv.timerangecut!(ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:46:45"))
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 4

    Xch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:47:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 5

    Xch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:43:03"), DateTime("2022-01-02T22:49:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 7

    Xch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:42:00"), DateTime("2022-01-02T22:49:45"))
    # does not add anything for DateTime("2022-01-02T22:42:03")
    # println(Ohlcv.dataframe(ohlcv1))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 8

    Xch.cryptoupdate!(xc, ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    Ohlcv.timerangecut!(ohlcv1, DateTime("2022-01-02T22:50:03"), DateTime("2022-01-02T22:55:45"))
    @test size(Ohlcv.dataframe(ohlcv1), 1) == 6

    Ohlcv.delete(ohlcv)
    sleep(1)
    rm(EnvConfig.datafolderpath(); force=true, recursive=true)

    @test Xch.onlyconfiguredsymbols("BTCUSDT")
    @test !Xch.onlyconfiguredsymbols("BTCBNB")
    @test !Xch.onlyconfiguredsymbols("EURUSDT")


    # EnvConfig.init(test)
    EnvConfig.init(production)
    mdf = Xch.getUSDTmarket(xc)
    btcprice = mdf[mdf.basecoin .== "BTC", :lastprice][1,1]
    # println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

    bdf = Xch.balances(xc)
    @test size(bdf, 2) >= 3
    # println(bdf)

    pdf = Xch.portfolio!(xc, bdf, mdf)
    @test size(pdf, 2) >= 5
    # println(pdf)

    oodf = Xch.getopenorders(xc, nothing)
    @test isa(oodf, AbstractDataFrame)
    # println("getopenorders(xc, nothing): $oodf")

    oo2 = Xch.getorder(xc, "invalid_or_unknown_id")
    @test isnothing(oo2)

    oid = Xch.createbuyorder(xc, "btc", limitprice=btcprice*1.2, basequantity=26.01/btcprice, maker=false) # limitprice out of allowed range
    @test isnothing(oid)
    # println("createbuyorder: $(string(oid)) - error expected")
    oid = Xch.createbuyorder(xc, "btc", limitprice=btcprice * 1.001, basequantity=26.01/btcprice, maker=false) # PostOnly will cause reject if price < limitprice due to taker order
    @test !isnothing(oid)
    oo2 = Xch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oo2.orderid == oid
    @test oo2.status == "Filled"  # due to GTC as long as taker fee == maker fee

    oid = Xch.createbuyorder(xc, "btc", limitprice=btcprice * 0.9, basequantity=6.01/btcprice, maker=false)
    # println("createbuyorder: $(string(oid))")
    oo2 = Xch.getorder(xc, oid)
    # println("getorder: $oo2")
    @test oid == oo2.orderid
    # println("getorder: $(Xch.getorder(xc, oid))")

    oidc = Xch.changeorder(xc, oid; basequantity=4.02/btcprice)
    @test oidc == oid
    # println("getorder: $(Xch.getorder(xc, oid))")

    oidc = Xch.changeorder(xc, oid; limitprice=btcprice * 0.8)
    @test oidc == oid
    # println("getorder: $(Xch.getorder(xc, oid))")


    oodf = Xch.getopenorders(xc, nothing)
    @test isa(oodf, AbstractDataFrame)
    @test (size(oodf, 1) > 0)
    @test (size(oodf, 2) >= 13)
    # println("getopenorders(nothing): $oodf")
    oodf = Xch.getopenorders(xc, "xrp")
    # println("getopenorders(\"xrp\"): $oodf")
    oid2 = Xch.cancelorder(xc, "btc", oid)
    # println("cancelorder: $(string(oid2))")
    @test oid == oid2
    oo2 = Xch.getorder(xc, oid)
    # println("getorder: $oo2")
    oodf = Xch.getopenorders(xc)
    # println("getopenorders(nothing): $oodf")

    # println("test IP with CLI: wget -qO- http://ipecho.net/plain | xargs echo")

end

if RUN_KRAKEN_ORDER_TESTS
    @testset "KrakenSpot live order lifecycle via Xch" begin
        _run_kraken_order_lifecycle!(Xch.EXCHANGE_KRAKENSPOT)
    end

    @testset "KrakenFutures live order lifecycle via Xch" begin
        _run_kraken_order_lifecycle!(Xch.EXCHANGE_KRAKENFUTURES)
    end
else
    @info "Skipping live Kraken order lifecycle tests. Set CTS_RUN_KRAKEN_ORDER_TESTS=true to enable." 
end
else
    @info "Skipping production integration tests. Set CTS_RUN_PRODUCTION_TESTS=true to enable live exchange checks."
end


end  # module