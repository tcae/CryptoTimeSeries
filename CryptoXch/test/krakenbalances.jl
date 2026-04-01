using Dates, DataFrames
using EnvConfig, CryptoXch

"""
Read and print balances for one exchange/auth tuple pair.
"""
function print_balances(exchange_name::String, auth_tuple::String)
    println("\n=== $(exchange_name) ($(auth_tuple)) ===")
    try
        xc = CryptoXch.XchCache(exchange=exchange_name, authname=auth_tuple)
        bdf = CryptoXch.balances(xc)
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
function main()
    EnvConfig.init(EnvConfig.production)

    spot = print_balances(CryptoXch.EXCHANGE_KRAKENSPOT, "krakenspot-tcae1")
    futures = print_balances(CryptoXch.EXCHANGE_KRAKENFUTURES, "krakenfutures-tcae2")

    println("\nDone at $(Dates.now(Dates.UTC))")
    return (spot=spot, futures=futures)
end

main()
