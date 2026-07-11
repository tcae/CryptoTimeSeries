"""
assess_close_orders.jl

Live close-order coverage report for one exchange/quote context.

Usage:
  julia --project=scripts scripts/assess_close_orders.jl
  julia --project=scripts scripts/assess_close_orders.jl quote=USD xch=KrakenSpot
  julia --project=scripts scripts/assess_close_orders.jl quote=USDT xch=KrakenSpot

Arguments:
  quote=USD|USDT|...         Quote coin used to build symbols and valuation context.
                             Default: USD
  xch=KrakenSpot|KrakenFutures|Bybit
                             Default: KrakenSpot
"""

import Pkg
Pkg.activate(joinpath(@__DIR__), io=devnull)

using DataFrames
using Xch, EnvConfig

function _argvalue(args::Vector{String}, key::AbstractString, default::Union{Nothing, AbstractString}=nothing)
    prefix = String(key) * "="
    for arg in args
        startswith(arg, prefix) || continue
        return strip(arg[(length(prefix)+1):end])
    end
    return default
end

function _resolve_exchange(raw::Union{Nothing, AbstractString})::String
    isnothing(raw) && return Xch.EXCHANGE_KRAKENSPOT
    key = lowercase(strip(String(raw)))
    aliases = Dict(
        "krakenspot" => Xch.EXCHANGE_KRAKENSPOT,
        "krakenfutures" => Xch.EXCHANGE_KRAKENFUTURES,
        "bybit" => Xch.EXCHANGE_BYBIT,
    )
    haskey(aliases, key) || error("unsupported xch=$(raw). Expected one of krakenspot|krakenfutures|bybit")
    return aliases[key]
end

function _covered_qty(oo::AbstractDataFrame, symbol::String, side::String; require_leverage::Union{Nothing, Bool}=nothing)::Float32
    total = 0f0
    wanted_side = uppercase(String(side))
    for r in eachrow(oo)
        Xch.openstatus(String(r.status)) || continue
        String(r.symbol) == symbol || continue
        uppercase(String(r.side)) == wanted_side || continue
        if !isnothing(require_leverage)
            hasproperty(r, :isLeverage) || continue
            Bool(getproperty(r, :isLeverage)) == require_leverage || continue
        end
        executed = hasproperty(r, :executedqty) ? (r.executedqty) : 0f0
        total += max(0f0, (r.baseqty) - executed)
    end
    return total
end

function main(args::Vector{String})
    quote_coin = uppercase(get(args, 1, nothing) === nothing ? (_argvalue(args, "quote", "USD")) : (_argvalue(args, "quote", "USD")))
    exchange = _resolve_exchange(_argvalue(args, "xch", nothing))

    EnvConfig.init(EnvConfig.production)
    EnvConfig.setpairquote!(quote_coin)

    xc = Xch.XchCache(exchange=exchange)

    if exchange == Xch.EXCHANGE_KRAKENSPOT
        Xch.KrakenSpot._invalidate_openorders_cache!()
    elseif exchange == Xch.EXCHANGE_KRAKENFUTURES
        Xch.KrakenFutures._invalidate_openorders_cache!()
    end

    balances = Xch.balances(xc; ignoresmallvolume=false)
    assets = Xch.portfolio!(xc, balances; ignoresmallvolume=false)
    oo = Xch.getopenorders(xc)

    if !(:isLeverage in names(oo))
        oo.isLeverage = falses(size(oo, 1))
    end

    q = uppercase(String(EnvConfig.pairquote))
    rows = NamedTuple[]

    for row in eachrow(assets)
        base = uppercase(String(row.coin))
        base == q && continue

        freebase = (row.free)
        borrowed = (row.borrowed)
        symbol = Xch.symboltoken(xc, base, EnvConfig.pairquote)

        minq = try
            (Xch.minimumbasequantity(xc, base, (row.usdtprice)))
        catch
            0f0
        end

        sell_cov = _covered_qty(oo, symbol, "Sell"; require_leverage=false)
        buy_cov = _covered_qty(oo, symbol, "Buy"; require_leverage=true)
        sell_gap = max(0f0, freebase - sell_cov)
        buy_gap = max(0f0, borrowed - buy_cov)

        push!(rows, (
            base=base,
            free=freebase,
            locked=(row.locked),
            borrowed=borrowed,
            usdtvalue=(row.usdtvalue),
            usdtprice=(row.usdtprice),
            minqty=minq,
            sell_cov=sell_cov,
            sell_gap=sell_gap,
            need_long_close=(freebase > 0f0) && (sell_gap >= minq),
            buy_cov=buy_cov,
            buy_gap=buy_gap,
            need_short_close=(borrowed > 0f0) && (buy_gap >= minq),
        ))
    end

    rep = DataFrame(rows)

    println("exchange=$(exchange) quote=$(EnvConfig.pairquote)")
    println("asset_rows=$(size(rep,1)) open_orders_rows=$(size(oo,1))")
    println("open_sell_nonleverage=$(sum((uppercase.(String.(oo.side)) .== "SELL") .& .!Bool.(oo.isLeverage) .& Xch.openstatus.(String.(oo.status))))")
    println("open_buy_leverage=$(sum((uppercase.(String.(oo.side)) .== "BUY") .& Bool.(oo.isLeverage) .& Xch.openstatus.(String.(oo.status))))")

    println("\n--- MISSING LONG CLOSE ---")
    missing_long = rep[rep.need_long_close .== true, [:base, :free, :locked, :borrowed, :usdtvalue, :minqty, :sell_cov, :sell_gap]]
    show(missing_long, allrows=true, allcols=true)
    println()

    println("\n--- MISSING SHORT CLOSE ---")
    missing_short = rep[rep.need_short_close .== true, [:base, :free, :locked, :borrowed, :usdtvalue, :minqty, :buy_cov, :buy_gap]]
    show(missing_short, allrows=true, allcols=true)
    println()

    println("\n--- ASSETS ---")
    show(sort(rep, :usdtvalue, rev=true), allrows=true, allcols=true)
    println()
end

main(ARGS)
