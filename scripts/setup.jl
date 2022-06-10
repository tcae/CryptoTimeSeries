# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

import Pkg: activate, add, status
activate(pwd())

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

add(url="https://github.com/tlienart/OpenSpecFun_jll.jl")  # fix for MKL issue in Scikit-learn - see MLJ manual
add([
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


# required packages for CryptoExchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
