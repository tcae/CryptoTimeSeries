# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
Pkg.add([ "Dates", "Logging", "DataFrames", "RollingFunctions", "Combinatorics", "Indicators", "Statistics", "JDF", "PlotlyJS", "LinearRegression" ])
# build()
# resolve()
# develop(path="/home/tor/TorProjects/CryptoTimeSeries")

Pkg.instantiate()
Pkg.resolve()
# Pkg.update()
Pkg.gc()
