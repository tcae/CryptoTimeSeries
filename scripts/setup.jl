# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
# rootpath = ".."
rootpath = joinpath(@__DIR__, "..")
if Sys.islinux()
    # rootpath = joinpath(@__DIR__, "..")
    println("Linux, rootpath: $rootpath, homepath: $(homedir())")
elseif Sys.isapple()
    # rootpath = joinpath(@__DIR__, "..")
    println("Apple, rootpath: $rootpath, homepath: $(homedir())")
elseif Sys.iswindows()
    # rootpath = joinpath(@__DIR__, "..")
    println("Windows, rootpath: $rootpath, homepath: $(homedir())")
else
    # rootpath = joinpath(@__DIR__, "..")
    println("unknown OS, rootpath: $rootpath, homepath: $(homedir())")
end
Pkg.activate(rootpath)
cd(rootpath)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
mypackages = ["EnvConfig", "Ohlcv", "Features", "Targets", "TestOhlcv", "Bybit", "CryptoXch", "Assets", "Classify", "TradingStrategy", "Trade"]
# Pkg.upgrade_manifest()
# for mypackage in mypackages
#     try
#         Pkg.free(mypackage)
#     catch err
#         println(err)
#     end
# end
# for mypackage in mypackages
#     try
#         Pkg.rm(mypackage)
#     catch err
#         println(err)
#     end
# end
rootpath = "."
for mypackage in mypackages
    folderpath = joinpath(rootpath, mypackage)
    println("preparing $folderpath")
    # Pkg.activate(folderpath)
    Pkg.develop(path=folderpath)
    # Pkg.gc()
end
# Pkg.add(url="https://github.com/tlienart/OpenSpecFun_jll.jl")  # fix for MKL issue in Scikit-learn - see MLJ manual
Pkg.add([
    "Test", # "JuliaInterpreter",
    "JSON",  # EnvConfig
    "SHA", "Printf", "HTTP",  # Binance
    "Dates",  # EnvConfig, Binance, CryptoExchange, Ohlcv, Assets
    "DataFrames",  # CryptoExchange, Ohlcv, Assets, Features
    "DataAPI",   # CryptoExchange
    "Indicators",  # CryptoExchange, Ohlcv, Assets, Features
    "JDF",  # CryptoExchange, Ohlcv, Assets
    "CSV",  # CryptoExchange, Ohlcv
    # "CategoricalArrays", "JLSO",  # Ohlcv, Classify
    "Logging", "LoggingExtras",  # CryptoExchange, Ohlcv, Assets, Features
    "PrettyPrinting",
    "RollingFunctions", "Statistics",  # Features
    # "MLJ", "PartialLeastSquaresRegressor", "ScikitLearn", "MLJFlux", "cuDNN",
     "StatisticalMeasures", "Distributions", "CategoricalDistributions", "CategoricalArrays",
    # "MLJGLMInterface", "GLM",
    # "MLJLinearModels", "MLJDecisionTreeInterface", "MLJScikitLearnInterface",
    "Combinatorics", # MLJtest -
    "Dash", "PlotlyJS", "Colors", "ProfileView",  # CryptoCockpit
    # "IJulia", "Plots",  # Regressionsim,  Notebooks
    # "Flux", "ProgressMeter", "MLUtils", # ML
    "SortingAlgorithms",  # Targets
    "Profile",  # Trade
    "BetterFileWatching"
    ])

# Pkg.resolve()
# Pkg.update()
Pkg.gc()
