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

    n1 = parse(Int, KrakenSpot._nextnonce())
    n2 = parse(Int, KrakenSpot._nextnonce())
    @test n2 > n1

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
        minquoteqty=5f0,
        krakenpairname="XBTUSDT",
        wsname="BTC/USDT",
    ); cols=:subset)

    cache = KrakenSpot.KrakenSpotCache(syminfo, KrakenSpot.KRAKEN_APIREST, "", "")

    positions = Dict(
        "tx1" => Dict("type" => "sell", "pair" => "XBTUSDT", "vol" => "0.25"),
        "tx2" => Dict("type" => "buy", "pair" => "XBTUSDT", "vol" => "99"),
        "tx3" => Dict("type" => "sell", "pair" => "UNKNOWNPAIR", "vol" => "1.0"),
    )
    borrowed = KrakenSpot._borrowedfromopenpositionsresult(cache, positions)
    @test haskey(borrowed, "BTC")
    @test borrowed["BTC"] ≈ 0.25f0
    @test !haskey(borrowed, "UNKNOWN")

    balancedf = DataFrame(
        coin=AbstractString["BTC", "USDT"],
        locked=Float32[0f0, 0f0],
        free=Float32[1f0, 100f0],
        borrowed=Float32[0.1f0, 0f0],
        accruedinterest=Float32[0f0, 0f0],
    )
    KrakenSpot._mergeborrowedbalances!(balancedf, Dict("BTC" => 0.25f0, "ETH" => 0.5f0))
    btcix = findfirst(==("BTC"), balancedf[!, :coin])
    ethix = findfirst(==("ETH"), balancedf[!, :coin])
    @test !isnothing(btcix)
    @test !isnothing(ethix)
    @test balancedf[btcix, :borrowed] ≈ 0.35f0
    @test balancedf[ethix, :borrowed] ≈ 0.5f0
    @test balancedf[ethix, :free] == 0f0

    info = KrakenSpot.symbolinfo(cache, "BTCUSDT")
    @test !isnothing(info)
    @test KrakenSpot._istradablestatus("online")
    @test !KrakenSpot._istradablestatus("cancel_only")

    norm = KrakenSpot._normalizelimitorderparams(info, 0.00001f0, 101.234f0)
    @test norm.limitprice == 101.2f0
    @test norm.basequantity >= info.minbaseqty
    @test norm.basequantity * norm.limitprice >= info.minquoteqty

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
    @test isapprox(tickrow.lastprice, 101.2f0; atol=1f-4)
    @test tickrow.quotevolume24h ≈ 1315.6f0 atol = 1f-3

    @test isapprox(KrakenSpot._makerlimitprice(info, tickrow, "Buy"), 101.4f0; atol=1f-4)
    @test isapprox(KrakenSpot._makerlimitprice(info, tickrow, "Sell"), 101.1f0; atol=1f-4)

    displayqty = KrakenSpot._icebergdisplayqty(info, 1.5f0, 100f0, 20.0)
    @test displayqty >= 0.2f0
    @test KrakenSpot._usenativeiceberg("limit", 1.5f0, 100f0, 20.0)
    @test !KrakenSpot._usenativeiceberg("market", 1.5f0, 100f0, 20.0)

    validated_params = KrakenSpot._addorderparams("XBTUSDT", "Buy", "limit", 0.5f0, "ROBO-TEST";
        effectiveprice=100.0f0,
        maker=true,
        effective_marginleverage=3,
        reduceonly=true,
        validate=true,
    )
    @test validated_params["pair"] == "XBTUSDT"
    @test validated_params["type"] == "buy"
    @test validated_params["ordertype"] == "limit"
    @test validated_params["oflags"] == "post"
    @test validated_params["leverage"] == "3"
    @test validated_params["reduce_only"] == true
    @test validated_params["validate"] == true

    normal_params = KrakenSpot._addorderparams("XBTUSDT", "Sell", "market", 0.5f0, "ROBO-TEST-2";
        validate=false,
    )
    @test !haskey(normal_params, "validate")

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

    call_order = String[]
    seq = KrakenSpot._closebeforeopenflip(
        () -> begin
            push!(call_order, "close")
            return (orderid="close-1",)
        end,
        () -> begin
            push!(call_order, "open")
            return (orderid="open-1",)
        end,
    )
    @test call_order == ["close", "open"]
    @test !isnothing(seq.closeorderid)
    @test !isnothing(seq.openorderid)

    call_order_abort = String[]
    aborted = KrakenSpot._closebeforeopenflip(
        () -> begin
            push!(call_order_abort, "close")
            return nothing
        end,
        () -> begin
            push!(call_order_abort, "open")
            return (orderid="open-should-not-run",)
        end,
    )
    @test call_order_abort == ["close"]
    @test isnothing(aborted.closeorderid)
    @test isnothing(aborted.openorderid)
end
