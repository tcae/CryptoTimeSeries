using DataFrames, Dates, KrakenSpot, Test

@testset "KrakenSpot offline interface tests" begin
    emptycache = KrakenSpot.KrakenSpotCache(autoloadexchangeinfo=false, publickey="", secretkey="")
    @test emptycache.syminfodf isa DataFrame
    @test size(emptycache.syminfodf, 1) == 0

    query = KrakenSpot._dict2paramsget(Dict("b" => 2, "a" => "x y"))
    @test query == "a=x%20y&b=2"

    @test KrakenSpot._normalizeasset("XXBT") == "BTC"
    @test KrakenSpot._ws2symbol("BTC/USDT") == "BTCUSDT"
    @test KrakenSpot._symbol2ws("BTCUSDT") == "BTC/USDT"

    syminfo = KrakenSpot._emptyexchangeinfo()
    push!(syminfo, (
        symbol="BTCUSDT",
        status="online",
        basecoin="BTC",
        quotecoin="USDT",
        ticksize=0.1f0,
        baseprecision=1f-6,
        quoteprecision=0.1f0,
        minbaseqty=0.0001f0,
        minquoteqty=0f0,
        krakenpairname="XBTUSDT",
        wsname="BTC/USDT",
    ))

    cache = KrakenSpot.KrakenSpotCache(syminfo, KrakenSpot.KRAKEN_APIREST, "", "")

    info = KrakenSpot.symbolinfo(cache, "BTCUSDT")
    @test !isnothing(info)
    @test KrakenSpot.validsymbol(cache, info)
    @test KrakenSpot.validsymbol(cache, "BTCUSDT")
    @test !KrakenSpot.validsymbol(cache, "ETHUSD")

    ticker = Dict(
        "a" => Any["101.5", "1", "1"],
        "b" => Any["101.0", "1", "1"],
        "c" => Any["101.2", "0.2"],
        "o" => "100.0",
        "v" => Any["12.0", "13.0"],
    )
    tickrow = KrakenSpot._tickerrow(cache, "XBTUSDT", ticker)
    @test tickrow.symbol == "BTCUSDT"
    @test tickrow.lastprice == 101.2f0
    @test tickrow.quotevolume24h ≈ 1214.4f0 atol = 1f-3

    klines = Any[
        Any[1700000000, "100", "110", "90", "105", "104", "1.2", "12"],
        Any[1700000060, "105", "120", "100", "118", "117", "2.3", "20"],
    ]
    klinedf = KrakenSpot._convertklines(klines)
    @test names(klinedf) == ["opentime", "open", "high", "low", "close", "basevolume"]
    @test size(klinedf, 1) == 2
    @test issorted(klinedf.opentime)
    @test klinedf[2, :close] == 118f0

    orders = KrakenSpot.emptyorders()
    @test "orderid" in names(orders)
    @test "lastcheck" in names(orders)

    filtered = KrakenSpot.filterOnRegex("BTC", [
        Dict("symbol" => "BTCUSDT"),
        Dict("symbol" => "ETHUSDT"),
        Dict("other" => "ignored"),
    ])
    @test length(filtered) == 1
    @test filtered[1]["symbol"] == "BTCUSDT"
end
