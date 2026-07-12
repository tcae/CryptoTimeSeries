using Dates, DataFrames
using EnvConfig, Xch, KrakenFutures, KrakenSpot

"""
Read and print balances for one exchange/auth tuple pair.
"""
function print_balances(xc)
    println("\n=== $(Xch.exchange(xc)) ===")
    try
        # This script is a visibility probe; do not hide balances by min-qty filtering.
        bdf = Xch.balances(xc, ignoresmallvolume=false)
        println("rows=$(size(bdf, 1)) cols=$(size(bdf, 2))")
        println(bdf)
        return bdf
    catch err
        println("balance read failed: $(sprint(showerror, err))")
        return DataFrame()
    end
end

"""
Entry point for printing Kraken spot and futures balances.
"""
function mymain()
    EnvConfig.init(EnvConfig.production)

    xckf = Xch.XchCache(KrakenFutures.KrakenFuturesCache())
    futures = print_balances(xckf)

    xcks = Xch.XchCache(KrakenSpot.KrakenSpotCache())
    spot = print_balances(xcks)

    println("\nDone at $(Dates.now(Dates.UTC))")
    return (spot=spot, futures=futures)
end

mymain()
