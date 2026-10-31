using Test, Dates, DataFrames
import EnvConfig, Ohlcv, Trade, Xch, Arrow
import XchAdapter
include("../scripts/supplement.jl")

"""Construct one-minute OHLCV fixtures for supplementation tests."""
function supplementfixture(start::DateTime, rows::Int)::DataFrame
    return DataFrame(opentime=collect(start:Minute(1):start+Minute(rows-1)),
        open=fill(10f0, rows), high=fill(11f0, rows), low=fill(9f0, rows),
        close=fill(10f0, rows), basevolume=fill(1000f0, rows))
end

"""Public-data-only fake adapter; unimplemented private methods fail if called."""
struct SupplementAdapter <: XchAdapter.XchAdapterCache
    candles::Dict{String, DataFrame}
    tickers::DataFrame
end

"""Return fake USDT instrument metadata for the offline selection test."""
function XchAdapter.symbolinfo(adapter::SupplementAdapter, symbol)
    haskey(adapter.candles, symbol) || return nothing
    return (symbol=symbol, status="Trading", basecoin=chop(symbol; tail=4),
        quotecoin="USDT", ticksize=0.01f0, baseprecision=0.001f0,
        quoteprecision=0.01f0, minbaseqty=0.001f0, minquoteqty=1f0)
end

"""Treat fake test symbols as tradable."""
XchAdapter.validsymbol(::SupplementAdapter, symbol) = true

"""Return the concatenated USDT symbol for a fake instrument."""
XchAdapter.symboltoken(::SupplementAdapter, base::AbstractString, quotecoin::AbstractString) = base * quotecoin

"""Return a copy of the fake public ticker table."""
XchAdapter.get24h(adapter::SupplementAdapter) = copy(adapter.tickers)

"""Return the requested fake candle range without network access."""
function XchAdapter.getklines(adapter::SupplementAdapter, symbol;
        startDateTime=nothing, endDateTime=nothing, interval="1m")
    df = adapter.candles[symbol]
    return df[(df.opentime .>= startDateTime) .& (df.opentime .<= endDateTime), :]
end

@testset "Public selection and persistent supplementation" begin
    EnvConfig.init(EnvConfig.test)
    previousfolder = EnvConfig.activecoinspath
    previousquote = EnvConfig.pairquote
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        try
            EnvConfig.setcoinspath!(folder)
            EnvConfig.setpairquote!("USDT")
            start = DateTime(2025, 1, 1)
            rows = Ohlcv.ld.checkperiod + Ohlcv.ld.accumulate + 10
            candles = Dict(coin*"USDT" => supplementfixture(start, rows)
                for coin in ("LIQ", "THIN", "LOW", "OLD"))
            candles["THINUSDT"].basevolume .= 0f0
            tickers = DataFrame(symbol=["LIQUSDT", "THINUSDT", "LOWUSDT", "OLDUSDT"],
                quotevolume24h=Float32[2e6, 2e6, 1e5, 2e6], lastprice=fill(10f0, 4),
                askprice=fill(10f0, 4), bidprice=fill(10f0, 4), pricechangepercent=zeros(Float32, 4))
            adapter = SupplementAdapter(candles, tickers)
            enddt = last(candles["LIQUSDT"].opentime)
            xc = Xch.XchCache(; bc=adapter, startdt=enddt)
            selected = Trade.tradeselection!(xc; datetime=enddt, excludedbases=["OLD"])
            @test selected.basecoin == ["LIQ", "THIN"]
            @test selected.continuousminvol == [true, false]
            @test Set(Xch.bases(xc)) == Set(["LIQ"])
            Supplement.supplementcoin!(xc, "LIQ", enddt; newcoin=true)
            path = EnvConfig.coinfile("LIQ", "USDT", "ohlcv"; extension=".arrow")
            @test DataFrame(Arrow.Table(path)) == candles["LIQUSDT"]
            @test isempty(Xch.bases(xc))
            original = DataFrame(Arrow.Table(path))
            candles["LIQUSDT"] = supplementfixture(start, rows+2)
            candles["LIQUSDT"].close .= 12f0
            Supplement.supplementcoin!(xc, "LIQ", enddt+Minute(2))
            stored = DataFrame(Arrow.Table(path))
            @test stored[1:rows-1, :] == original[1:end-1, :]
            @test nrow(stored) == rows+2
            @test stored.close[end-2:end] == fill(12f0, 3)
            Supplement.supplementcoin!(xc, "LIQ", enddt+Minute(2))
            @test DataFrame(Arrow.Table(path)) == stored
            @test !isfile(joinpath(dirname(path), "ohlcv-supplement.arrow"))
        finally
            EnvConfig.setcoinspath!(previousfolder)
            EnvConfig.setpairquote!(previousquote)
        end
    end
end

@testset "OHLCV supplementation" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        for pair in ("BTC-USDT", "ETH-USDT", "BTC-USDC", "EMPTY-USDT")
            mkpath(joinpath(folder, pair))
            pair == "EMPTY-USDT" || touch(joinpath(folder, pair, "ohlcv.arrow"))
        end
        @test Supplement.cachedbases(folder) == ["BTC", "ETH"]
        @test_throws AssertionError Supplement.cachedbases(joinpath(folder, "missing"))
    end
    start = DateTime(2025, 1, 1)
    old = supplementfixture(start, 5)
    fetched = supplementfixture(start+Minute(3), 5)
    fetched.close .= 12f0
    merged = Supplement.extendedohlcv(old, fetched, "BTC")
    @test nrow(merged) == 8
    @test merged[1:4, :] == old[1:4, :]
    @test merged[5:end, :] == fetched[2:end, :]
    @test old.close == fill(10f0, 5)
    @test Supplement.extendedohlcv(DataFrame(), fetched, "NEW") == fetched
    @test_throws AssertionError Supplement.extendedohlcv(old, supplementfixture(start+Minute(5), 3), "GAP")
    @test_throws AssertionError Supplement.extendedohlcv(old, supplementfixture(start, 3), "STALE")
    @test_throws AssertionError Supplement.extendedohlcv(old, select(fetched, Not(:close)), "SCHEMA")
    @test Supplement.checkohlcv(old, "BTC") === nothing
    @test_throws AssertionError Supplement.checkohlcv(old[1:0, :], "EMPTY")
    @test_throws AssertionError Supplement.checkohlcv(vcat(old, old), "DUPLICATE")
    invalid = copy(old)
    invalid.close[1] = NaN32
    @test_throws AssertionError Supplement.checkohlcv(invalid, "NAN")
    @test Supplement.checkohlcv(old[[1, 3, 5], :], "HISTORICAL-GAPS") === nothing
    @test Supplement.main(["--help"]) === nothing
    @test_throws AssertionError Supplement.main(["unknown"])
end
