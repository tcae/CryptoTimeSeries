# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)
# Pkg.upgrade_manifest()

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

Pkg.add([ "Dates", "Logging", "DataFrames", "DataAPI", "JDF", "CSV" ])

# develop(path="/home/tor/TorProjects/CryptoTimeSeries")

# Pkg.resolve()
# Pkg.update()
Pkg.gc()
