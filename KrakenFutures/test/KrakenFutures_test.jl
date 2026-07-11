using DataFrames, Dates, EnvConfig, KrakenFutures, Test

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
        maxleveragebuy=0,
        maxleveragesell=0,
        ticksize=0.1f0,
        baseprecision=1f0,
        quoteprecision=0.1f0,
        minbaseqty=1f0,
        minquoteqty=500f0,
        krakenpairname="PI_XBTUSDT",
        wsname="PI_XBTUSDT",
    ))

    cache = KrakenFutures.KrakenFuturesCache(syminfo, KrakenFutures.KRAKEN_FUTURES_APIREST, "", "")

    info = KrakenFutures.symbolinfo(cache, "BTCUSDT")
    @test !isnothing(info)
    @test KrakenFutures._istradablestatus("online")
    @test !KrakenFutures._istradablestatus("cancel_only")

    norm = KrakenFutures._normalizelimitorderparams(info, 0.1f0, 101.234f0)
    @test norm.limitprice == 101.2f0
    @test norm.basequantity >= info.minbaseqty
    @test norm.basequantity * norm.limitprice >= info.minquoteqty

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
    @test isapprox(tickrow.lastprice, 101.2f0; atol=1f-4)
    @test tickrow.quotevolume24h ≈ 1214.4f0 atol = 1f-3

    @test isapprox(KrakenFutures._makerlimitprice(info, tickrow, "Buy"), 101.4f0; atol=1f-4)
    @test isapprox(KrakenFutures._makerlimitprice(info, tickrow, "Sell"), 101.1f0; atol=1f-4)

    chunk, split = KrakenFutures._icebergchunkamount(10.0, 100.0, 1.0, 250.0)
    @test split
    @test chunk ≈ 2.5 atol = 1e-6

    lock(KrakenFutures._iceberg_sequence_lock) do
        empty!(KrakenFutures._iceberg_sequences)
    end
    rootid = "root-seq"
    spec = KrakenFutures._executionorderspec(:long, "Buy")
    KrakenFutures._seticebergstate!(rootid, Dict{Symbol, Any}(
        :current_order_id => "child-1",
        :remaining_baseqty => 5.0,
        :symbol => "BTCUSDT",
        :orderside => "Buy",
        :configside => :long,
        :reduceonly => false,
        :maker => true,
        :price => 100.0,
        :refprice => 100.0,
        :minqty => 1.0,
        :max_quote => 200.0,
        :execution_spec => spec,
        :root_order_link_id => "link-root",
    ))

    submit_calls = Vector{NamedTuple{(:basequantity, :orderLinkId), Tuple{Float64, String}}}()
    submit_counter = Ref(1)
    submit_stub = function (_bc, _symbol, _orderside, basequantity, _price, _maker; orderLinkId=nothing, kwargs...)
        _ = kwargs
        push!(submit_calls, (basequantity=(basequantity), orderLinkId=String(orderLinkId)))
        submit_counter[] += 1
        return (orderid="child-$(submit_counter[])", orderLinkId=String(orderLinkId))
    end

    active_df = DataFrame(orderid=String["child-1"])
    @test !KrakenFutures._advanceicebergsequences!(cache, active_df; submitfn=submit_stub)
    @test length(submit_calls) == 0

    empty_df = DataFrame(orderid=String[])
    @test KrakenFutures._advanceicebergsequences!(cache, empty_df; submitfn=submit_stub)
    @test length(submit_calls) == 1
    @test submit_calls[1].basequantity ≈ 2.0 atol = 1e-9
    @test submit_calls[1].orderLinkId == "link-root"

    _root, state1 = KrakenFutures._icebergstate(rootid)
    @test !isnothing(state1)
    @test state1[:current_order_id] == "child-2"
    @test state1[:remaining_baseqty] ≈ 3.0 atol = 1e-9

    @test KrakenFutures._advanceicebergsequences!(cache, empty_df; submitfn=submit_stub)
    _root, state2 = KrakenFutures._icebergstate(rootid)
    @test !isnothing(state2)
    @test state2[:current_order_id] == "child-3"
    @test state2[:remaining_baseqty] ≈ 1.0 atol = 1e-9

    @test KrakenFutures._advanceicebergsequences!(cache, empty_df; submitfn=submit_stub)
    _root, state3 = KrakenFutures._icebergstate(rootid)
    @test isnothing(state3)
    @test length(submit_calls) == 3
    @test submit_calls[2].basequantity ≈ 2.0 atol = 1e-9
    @test submit_calls[3].basequantity ≈ 1.0 atol = 1e-9

    klines = Any[
        Dict("time" => 1700000000, "open" => "100", "high" => "110", "low" => "90", "close" => "105", "volume" => "1.2"),
        Any[1700000060, "105", "120", "100", "118", "2.3"],
    ]
    klinedf = KrakenFutures._convertklines(klines)
    @test names(klinedf) == ["opentime", "open", "high", "low", "close", "basevolume"]
    @test size(klinedf, 1) == 2
    @test issorted(klinedf.opentime)
    @test klinedf[2, :close] == 118f0

    orders = KrakenFutures.emptyorders(emptycache)
    @test "orderid" in names(orders)
    @test "lastcheck" in names(orders)

    filtered = KrakenFutures.filterOnRegex("BTC", [
        Dict("symbol" => "BTCUSDT"),
        Dict("symbol" => "ETHUSDT"),
        Dict("other" => "ignored"),
    ])
    @test length(filtered) == 1
    @test filtered[1]["symbol"] == "BTCUSDT"

    # Official Kraken Futures websocket signing vector:
    # https://docs.kraken.com/api/docs/guides/futures-websockets/
    challenge = "c100b894-1729-464d-ace1-52dbce11db42"
    api_secret = "7zxMEF5p/Z8l2p2U7Ghv6x14Af+Fx+92tPgUdVQ748FOIrEoT9bgT+bTRfXc5pz8na+hL/QdrCVG7bh9KpT0eMTm"
    expected_signed = "4JEpF3ix66GA2B+ooK128Ift4XQVtc137N9yeg4Kqsn9PI0Kpzbysl9M1IeCEdjg0zl00wkVqcsnG4bmnlMb3A=="
    @test KrakenFutures._wssignedchallenge(api_secret, challenge) == expected_signed
end

@testset "KrakenFutures testing environment selection" begin
    oldmode = EnvConfig.configmode
    try
        EnvConfig.init(EnvConfig.test)
        has_testing_auth = false
        auth = nothing
        try
            auth = EnvConfig.Authentication(nothing; exchange="KrakenFutures", purpose="testing")
            has_testing_auth = true
        catch
            has_testing_auth = false
        end

        if !has_testing_auth
            @test_skip "No KrakenFutures testing auth entry available in auth.json"
        else
            @test auth.purpose == "testing"

            bc = KrakenFutures.KrakenFuturesCache(autoloadexchangeinfo=false, publickey="", secretkey="")
            if isnothing(auth.derivatives) || String(auth.derivatives) == ""
                @test bc.apirest == KrakenFutures.KRAKEN_FUTURES_APIREST
            else
                @test occursin("demo-futures.kraken.com", String(auth.derivatives))
                @test bc.apirest == String(auth.derivatives)
            end
        end
    finally
        EnvConfig.init(oldmode)
    end
end
