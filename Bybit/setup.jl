# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
Pkg.add([ "Logging", "HTTP", "SHA", "JSON3", "Dates", "Printf", "DataFrames", "Formatting" ])

# develop(path="/home/tor/TorProjects/CryptoTimeSeries")

Pkg.resolve()
# Pkg.update()
Pkg.gc()
Pkg.instantiate()
