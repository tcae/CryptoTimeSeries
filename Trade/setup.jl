# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

import Pkg: activate, add, status, resolve
using Pkg
activate(@__DIR__)
cd(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

mypackages = ["EnvConfig", "Ohlcv", "CryptoXch", "Features", "Targets"]
rootpath = ".."
for mypackage in mypackages
    folderpath = joinpath(rootpath, mypackage)
    println("preparing $folderpath")
    # Pkg.activate(folderpath)
    Pkg.develop(path=folderpath)
    # Pkg.gc()
end

Pkg.add([ "Dates", "DataFrames", "JSON", "Profile", "Logging", "CSV", "LoggingExtras", "JDF", "Statistics", "cuDNN"])
Pkg.resolve()
Pkg.update()
Pkg.gc()
Pkg.precompile()

activate("..")
cd("..")
