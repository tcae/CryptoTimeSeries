# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
Pkg.add([
    "Test", "JuliaInterpreter",
    "DataFrames",
    "Dates",
    "Logging"
    ])

mypackages = ["EnvConfig", "Ohlcv", "Features"]
rootpath = ".."
for mypackage in mypackages
    folderpath = joinpath(rootpath, mypackage)
    println("preparing $folderpath")
    # Pkg.activate(folderpath)
    Pkg.develop(path=folderpath)
    # Pkg.gc()
end

# Pkg.resolve()
# Pkg.update()
Pkg.gc()
Pkg.precompile()
cd("..")
Pkg.activate(".")
