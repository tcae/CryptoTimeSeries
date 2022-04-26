# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

import Pkg: activate, add, status, resolve
activate(@__DIR__)

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

add(url="https://github.com/tlienart/OpenSpecFun_jll.jl")  # fix for MKL issue in Scikit-learn - see MLJ manual
add([
    "Test",
    "JSON",
    "DataFrames",
    "CSV",
    "CategoricalArrays",
    "Logging",
    "Statistics",
    "MLJ", "PartialLeastSquaresRegressor", "ScikitLearn", "MLJFlux",
    "MLJGLMInterface", "GLM",
    "MLJLinearModels", "MLJDecisionTreeInterface", "MLJScikitLearnInterface",
    "Combinatorics", # MLJtest -
    "IJulia", "Plots", "WebIO", "Dash", "PlotlyJS", "Colors"  # Regressionsim, CryptoCockpit, Notebooks
    ])

# resolve()
# develop(path="/home/tor/TorProjects/CryptoTimeSeries")


# required packages for CryptoExchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
