"""
tradedebug.jl

Read-only live account diagnostics for trade incident analysis.
The script does NOT place, amend, or cancel any orders.

Usage:
  julia --project=scripts scripts/tradedebug.jl
  julia --project=scripts scripts/tradedebug.jl xch=KrakenSpot quote=USD
  julia --project=scripts scripts/tradedebug.jl xch=KrakenSpot quote=USD maxassetfraction=0.10 top=25

Arguments:
  xch=KrakenSpot|KrakenFutures|Bybit   Exchange route (default: KrakenSpot)
  quote=USD|USDT|...                   Quote coin context (default: USD)
  maxassetfraction=0.10                Allocation threshold for risk report (default: 0.10)
  top=25                               Max rows printed in summaries (default: 25)
  refresh=true|false                   Invalidate exchange open-order cache before snapshot (default: true)
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using Dates
using DataFrames
using CSV
using CryptoXch
using EnvConfig

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing, AbstractString}=nothing)
    prefix = String(key) * "="
    for arg in args
        startswith(arg, prefix) || continue
        return strip(arg[(length(prefix)+1):end])
    end
    return default
end

function _argbool(args::Vector{String}, key::AbstractString, default::Bool)::Bool
    raw = lowercase(strip(String(_argvalue(args, key, default ? "true" : "false"))))
    return raw in ("1", "true", "yes", "on")
end

function _argfloat(args::Vector{String}, key::AbstractString, default::Float64)::Float64
    raw = _argvalue(args, key, string(default))
    try
        v = parse(Float64, String(raw))
        @assert isfinite(v) "argument $(key) must be finite, got $(raw)"
        return v
    catch err
        error("invalid $(key)=$(raw): $(sprint(showerror, err))")
    end
end

function _argint(args::Vector{String}, key::AbstractString, default::Int)::Int
    raw = _argvalue(args, key, string(default))
    try
        v = parse(Int, String(raw))
        @assert v > 0 "argument $(key) must be > 0, got $(raw)"
        return v
    catch err
        error("invalid $(key)=$(raw): $(sprint(showerror, err))")
    end
end

function _resolve_exchange(raw::Union{Nothing, AbstractString})::String
    isnothing(raw) && return CryptoXch.EXCHANGE_KRAKENSPOT
    key = lowercase(strip(String(raw)))
    aliases = Dict(
        "krakenspot" => CryptoXch.EXCHANGE_KRAKENSPOT,
        "krakenfutures" => CryptoXch.EXCHANGE_KRAKENFUTURES,
        "bybit" => CryptoXch.EXCHANGE_BYBIT,
    )
    haskey(aliases, key) || error("unsupported xch=$(raw). Expected krakenspot|krakenfutures|bybit")
    return aliases[key]
end

function _invalidate_openorders_cache!(exchange::String)
    if exchange == CryptoXch.EXCHANGE_KRAKENSPOT
        try
            CryptoXch.KrakenSpot._invalidate_openorders_cache!()
        catch
        end
    elseif exchange == CryptoXch.EXCHANGE_KRAKENFUTURES
        try
            CryptoXch.KrakenFutures._invalidate_openorders_cache!()
        catch
        end
    end
    return nothing
end

function _orderisleverage(orow)::Bool
    if hasproperty(orow, :isLeverage)
        return Bool(getproperty(orow, :isLeverage))
    end
    if hasproperty(orow, :marginleverage)
        return Float32(getproperty(orow, :marginleverage)) > 0f0
    end
    return false
end

function _remaining_open_qty(orow)::Float32
    total = hasproperty(orow, :baseqty) ? Float32(getproperty(orow, :baseqty)) : 0f0
    executed = hasproperty(orow, :executedqty) ? Float32(getproperty(orow, :executedqty)) : 0f0
    return max(0f0, total - executed)
end

function _covered_qty(oo::AbstractDataFrame, symbol::AbstractString, side::AbstractString; require_leverage::Union{Nothing, Bool}=nothing)::Float32
    wanted_side = uppercase(String(side))
    total = 0f0
    for orow in eachrow(oo)
        CryptoXch.openstatus(String(orow.status)) || continue
        String(orow.symbol) == String(symbol) || continue
        uppercase(String(orow.side)) == wanted_side || continue
        if !isnothing(require_leverage)
            (_orderisleverage(orow) == require_leverage) || continue
        end
        total += _remaining_open_qty(orow)
    end
    return total
end

function _safe_min_qty(xc::CryptoXch.XchCache, base::AbstractString, usdtprice::Real)::Float32
    if !(usdtprice > 0)
        return 0f0
    end
    try
        q = CryptoXch.minimumbasequantity(xc, base, Float32(usdtprice))
        return isnothing(q) ? 0f0 : Float32(q)
    catch
        return 0f0
    end
end

function _write_snapshot_tables!(folder::AbstractString; balances::AbstractDataFrame, assets::AbstractDataFrame, openorders::AbstractDataFrame, coverage::AbstractDataFrame, grouped::AbstractDataFrame)
    mkpath(folder)
    CSV.write(joinpath(folder, "balances.csv"), balances)
    CSV.write(joinpath(folder, "portfolio.csv"), assets)
    CSV.write(joinpath(folder, "open_orders.csv"), openorders)
    CSV.write(joinpath(folder, "coverage.csv"), coverage)
    CSV.write(joinpath(folder, "open_orders_grouped.csv"), grouped)
    return nothing
end

function main(args::Vector{String})
    quote_coin = uppercase(String(_argvalue(args, "quote", "USD")))
    exchange = _resolve_exchange(_argvalue(args, "xch", nothing))
    maxassetfraction = Float32(_argfloat(args, "maxassetfraction", 0.10))
    topn = _argint(args, "top", 25)
    refresh = _argbool(args, "refresh", true)

    EnvConfig.init(EnvConfig.production)
    EnvConfig.cryptoquote = quote_coin
    CryptoXch.verbosity = 1

    println("$(EnvConfig.now()): tradedebug start (READ-ONLY)")
    println("$(EnvConfig.now()): exchange=$(exchange) quote=$(quote_coin) maxassetfraction=$(maxassetfraction) refresh=$(refresh)")

    xc = CryptoXch.XchCache(exchange=exchange)

    if refresh
        _invalidate_openorders_cache!(exchange)
    end

    balances = try
        CryptoXch.balances(xc; ignoresmallvolume=false)
    catch err
        println(stderr, "$(EnvConfig.now()): failed to fetch balances (credentials/private access issue): $(sprint(showerror, err))")
        rethrow(err)
    end

    assets = CryptoXch.portfolio!(xc, balances; ignoresmallvolume=false)
    oo = CryptoXch.getopenorders(xc)

    if !(:isLeverage in names(oo))
        oo.isLeverage = [_orderisleverage(r) for r in eachrow(oo)]
    end

    q = uppercase(String(EnvConfig.cryptoquote))
    total_usdt = size(assets, 1) == 0 ? 0f0 : Float32(sum(Float32.(assets[!, :usdtvalue])))

    rows = NamedTuple[]
    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == q && continue

        freebase = Float32(row.free)
        lockedbase = Float32(row.locked)
        borrowedbase = Float32(row.borrowed)
        usdtprice = Float32(row.usdtprice)
        usdtvalue = Float32(row.usdtvalue)
        sellablelong = max(0f0, freebase - borrowedbase)
        symbol = CryptoXch.symboltoken(xc, base, EnvConfig.cryptoquote; role=CryptoXch.trade_exchange_spot)

        minqty = _safe_min_qty(xc, base, usdtprice)
        sell_cov = _covered_qty(oo, symbol, "Sell"; require_leverage=false)
        buy_cov = _covered_qty(oo, symbol, "Buy"; require_leverage=true)
        sell_gap = max(0f0, sellablelong - sell_cov)
        buy_gap = max(0f0, borrowedbase - buy_cov)
        allocation = total_usdt > 0f0 ? usdtvalue / total_usdt : 0f0

        push!(rows, (
            base=base,
            free=freebase,
            locked=lockedbase,
            borrowed=borrowedbase,
            sellablelong=sellablelong,
            usdtprice=usdtprice,
            usdtvalue=usdtvalue,
            allocation=allocation,
            minqty=minqty,
            sell_cov=sell_cov,
            sell_gap=sell_gap,
            need_long_close=(sell_gap >= minqty) && (sellablelong > 0f0),
            buy_cov=buy_cov,
            buy_gap=buy_gap,
            need_short_close=(buy_gap >= minqty) && (borrowedbase > 0f0),
            sell_overcommit=max(0f0, sell_cov - sellablelong),
            over_alloc=allocation > maxassetfraction,
        ))
    end

    coverage = DataFrame(rows)

    grouped = if size(oo, 1) == 0
        DataFrame(symbol=String[], side=String[], isLeverage=Bool[], status=String[], orders=Int[], remainingqty=Float32[])
    else
        tmp = deepcopy(oo)
        tmp.remainingqty = [_remaining_open_qty(r) for r in eachrow(tmp)]
        combine(groupby(tmp, [:symbol, :side, :isLeverage, :status]),
                nrow => :orders,
                :remainingqty => sum => :remainingqty)
    end

    ts = Dates.format(Dates.now(Dates.UTC), "yymmdd-HHMMSS")
    outfolder = joinpath(homedir(), "crypto", "debug", "tradedebug-$(ts)-$(lowercase(exchange))")
    _write_snapshot_tables!(outfolder; balances=balances, assets=assets, openorders=oo, coverage=coverage, grouped=grouped)

    println("\n=== SNAPSHOT ===")
    println("timestamp_utc=$(Dates.now(Dates.UTC))")
    println("balance_rows=$(size(balances, 1)) portfolio_rows=$(size(assets, 1)) open_orders=$(size(oo, 1))")
    println("total_quote_value=$(round(total_usdt, digits=4)) $(quote_coin)")
    println("snapshot_folder=$(outfolder)")

    if size(coverage, 1) == 0
        println("\nNo non-quote assets found.")
        return nothing
    end

    sorted_cov = sort(coverage, :usdtvalue, rev=true)
    println("\n=== TOP ASSETS BY VALUE ===")
    show(first(sorted_cov, min(topn, size(sorted_cov, 1))), allrows=true, allcols=true)
    println()

    over_alloc = sorted_cov[sorted_cov.over_alloc .== true, [:base, :usdtvalue, :allocation, :sellablelong, :sell_gap, :sell_cov]]
    println("\n=== OVER-ALLOCATION (> maxassetfraction) ===")
    if size(over_alloc, 1) == 0
        println("none")
    else
        show(over_alloc, allrows=true, allcols=true)
        println()
    end

    missing_long = sorted_cov[sorted_cov.need_long_close .== true, [:base, :sellablelong, :sell_cov, :sell_gap, :minqty, :allocation]]
    println("\n=== LONG CLOSE GAPS (SELL) ===")
    if size(missing_long, 1) == 0
        println("none")
    else
        show(missing_long, allrows=true, allcols=true)
        println()
    end

    missing_short = sorted_cov[sorted_cov.need_short_close .== true, [:base, :borrowed, :buy_cov, :buy_gap, :minqty, :allocation]]
    println("\n=== SHORT CLOSE GAPS (BUY) ===")
    if size(missing_short, 1) == 0
        println("none")
    else
        show(missing_short, allrows=true, allcols=true)
        println()
    end

    overcommit = sorted_cov[sorted_cov.sell_overcommit .> 0f0, [:base, :sellablelong, :sell_cov, :sell_overcommit, :allocation]]
    println("\n=== SELL OVERCOMMIT (open sell > sellable long) ===")
    if size(overcommit, 1) == 0
        println("none")
    else
        show(overcommit, allrows=true, allcols=true)
        println()
    end

    println("\n=== OPEN ORDERS GROUPED ===")
    show(sort(grouped, [:symbol, :side]), allrows=true, allcols=true)
    println()

    println("\n$(EnvConfig.now()): tradedebug completed (READ-ONLY)")
    return nothing
end

main(ARGS)
