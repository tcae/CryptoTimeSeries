# push!(LOAD_PATH, "/home/tor/TorProjects/CryptoTimeSeries/src")
# push!(DEPOT_PATH, "/home/tor/TorProjects", "/home/tor/TorProjects/CryptoTimeSeries", "/home/tor/TorProjects/CryptoTimeSeries/src")

import Pkg: activate, add, status, develop
activate(pwd())

println("load path: $LOAD_PATH   depot path: $DEPOT_PATH")

add(["Pkg", "Test"])
develop(path="/home/tor/TorProjects/CryptoTimeSeries")

# required for env_config
# import JSON
add(["JSON", "JSON3"])
# develop(path="src/env_config.jl")

# required packages for Binance
# import HTTP, SHA, JSON, Dates, Printf
add(["SHA", "JSON", "Dates", "Printf", "HTTP"])
# develop(["MyBinance"])

# required packages for CryptoExchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# using Dates, DataFrames, DataAPI, JDF, CSV, Logging
# using ..MyBinance, ..Config
add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV", "Logging"])
# develop(["CryptoXch"])

# requies packages for ohlcv
# using Dates, DataFrames, CategoricalArrays, JDF, CSV, TimeZones, Logging
add(["Dates", "DataFrames", "TimeZones", "JDF", "CSV", "CategoricalArrays", "Logging"])
# develop(["Ohlcv"])

# requies packages for assets
# using Dates, DataFrames, Logging, JDF
add(["Dates", "DataFrames", "Logging", "JDF"])
# develop(["Assets"])


# requies packages for targets
# develop(["Targets"])

# requies packages for classify
# develop(["Classify"])

# requies packages for features
# import RollingFunctions: rollmedian, rolling
# import DataFrames: DataFrame, Statistics, Logging
add(["RollingFunctions", "DataFrames", "Statistics", "Logging"])
# develop(["Features"])

# required packages for trade
# using Dates, DataFrames
# using ..Config, ..Ohlcv, ..Classify, ..Exchange
add(["Dates", "DataFrames"])
# develop(["Trade"])

# required packages for gradientgaindistribution
# using DataFrames, Logging, Statistics
add(["DataFrames", "Logging", "Statistics"])

# for applications (not yet used)
# using LoggingExtras

# required packages for quick visualizations, e.g. in regressionsim
add(["IJulia", "Plots", "WebIO", "Dash", "PlotlyJS"])
