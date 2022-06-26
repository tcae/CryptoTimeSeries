# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

using Pkg
Pkg.activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")
# Pkg.upgrade_manifest()
Pkg.add(url="https://github.com/tlienart/OpenSpecFun_jll.jl")  # fix for MKL issue in Scikit-learn - see MLJ manual
Pkg.add([
    "Test",
    "JSON",  # EnvConfig
    "SHA", "Printf", "HTTP",  # Binance
    "Dates",  # EnvConfig, Binance, CryptoExchange, Ohlcv, Assets
    "DataFrames",  # CryptoExchange, Ohlcv, Assets, Features
    "DataAPI",   # CryptoExchange
    "JDF",  # CryptoExchange, Ohlcv, Assets
    "CSV",  # CryptoExchange, Ohlcv
    "CategoricalArrays",  # Ohlcv
    "Logging",  # CryptoExchange, Ohlcv, Assets, Features
    "RollingFunctions", "Statistics",  # Features
    "MLJ", "PartialLeastSquaresRegressor", "ScikitLearn", "MLJFlux",
    "MLJGLMInterface", "GLM",
    "MLJLinearModels", "MLJDecisionTreeInterface", "MLJScikitLearnInterface",
    "Combinatorics", # MLJtest -
    "IJulia", "Plots", "WebIO", "Dash", "PlotlyJS", "Colors",  # Regressionsim, CryptoCockpit, Notebooks
    "Profile"  # Trade
    ])
# develop(path="/home/tor/TorProjects/CryptoTimeSeries")
Pkg.develop(path=joinpath(@__DIR__, "../Assets"))
Pkg.develop(path=joinpath(@__DIR__, "../Classify"))
Pkg.develop(path=joinpath(@__DIR__, "../CryptoXch"))
Pkg.develop(path=joinpath(@__DIR__, "../EnvConfig"))
Pkg.develop(path=joinpath(@__DIR__, "../Features"))
Pkg.develop(path=joinpath(@__DIR__, "../MyBinance"))
Pkg.develop(path=joinpath(@__DIR__, "../Ohlcv"))
Pkg.develop(path=joinpath(@__DIR__, "../Targets"))
Pkg.develop(path=joinpath(@__DIR__, "../TestOhlcv"))
Pkg.develop(path=joinpath(@__DIR__, "../Trade"))
# Pkg.resolve()
# Pkg.update()
Pkg.gc()
