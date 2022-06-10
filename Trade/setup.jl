# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

import Pkg: activate, add, status, resolve
activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

add([ "Dates", "DataFrames", "JSON", "Profile" ])

# resolve()
# develop(path="/home/tor/TorProjects/CryptoTimeSeries")


# required packages for CryptoExchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
