using DataFrames, Dates, KrakenFutures, Test

@testset "KrakenFutures offline interface tests" begin
    emptycache = KrakenFutures.KrakenFuturesCache(autoloadexchangeinfo=false, publickey="", secretkey="")
    @test emptycache.syminfodf isa DataFrame
    @test size(emptycache.syminfodf, 1) == 0

    query = KrakenFutures._dict2paramsget(Dict("b" => 2, "a" => "x y"))
    @test query == "a=x%20y&b=2"

    @test KrakenFutures._normalizeasset("XXBT") == "BTC"
    @test KrakenFutures._ws2symbol("PI_XBTUSDT") == "BTCUSDT"
    @test KrakenFutures._symbol2ws("BTCUSDT") == "PI_XBTUSDT"

    syminfo = KrakenFutures._emptyexchangeinfo()
    push!(syminfo, (
        symbol="BTCUSDT",
        status="online",
        basecoin="BTC",
        quotecoin="USDT",
        ticksize=0.1f0,
        baseprecision=1f0,
        quoteprecision=0.1f0,
        minbaseqty=1f0,
        minquoteqty=0f0,
        krakenpairname="PI_XBTUSDT",
        wsname="PI_XBTUSDT",
    ))

    cache = KrakenFutures.KrakenFuturesCache(syminfo, KrakenFutures.KRAKEN_FUTURES_APIREST, "", "")

    info = KrakenFutures.symbolinfo(cache, "BTCUSDT")
    @test !isnothing(info)
    @test KrakenFutures.validsymbol(cache, info)
    @test KrakenFutures.validsymbol(cache, "BTCUSDT")
    @test !KrakenFutures.validsymbol(cache, "ETHUSD")

    ticker = Dict(
        "ask" => "101.5",
        "bid" => "101.0",
        "last" => "101.2",
        "open24h" => "100.0",
        "volume24h" => "12.0",
        "turnover24h" => "1214.4",
    )
    tickrow = KrakenFutures._tickerrow(cache, "PI_XBTUSDT", ticker)
    @test tickrow.symbol == "BTCUSDT"
    @test tickrow.lastprice == 101.2f0
    @test tickrow.quotevolume24h ≈ 1214.4f0 atol = 1f-3

    klines = Any[
        Dict("time" => 1700000000, "open" => "100", "high" => "110", "low" => "90", "close" => "105", "volume" => "1.2"),
        Any[1700000060, "105", "120", "100", "118", "2.3"],
    ]
    klinedf = KrakenFutures._convertklines(klines)
    @test names(klinedf) == ["opentime", "open", "high", "low", "close", "basevolume"]
    @test size(klinedf, 1) == 2
    @test issorted(klinedf.opentime)
    @test klinedf[2, :close] == 118f0

    orders = KrakenFutures.emptyorders()
    @test "orderid" in names(orders)
    @test "lastcheck" in names(orders)

    filtered = KrakenFutures.filterOnRegex("BTC", [
        Dict("symbol" => "BTCUSDT"),
        Dict("symbol" => "ETHUSDT"),
        Dict("other" => "ignored"),
    ])
    @test length(filtered) == 1
    @test filtered[1]["symbol"] == "BTCUSDT"
end
