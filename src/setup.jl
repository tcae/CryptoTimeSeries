import Pkg: activate, add, status
activate(pwd())
add(["Pkg", "Test"])

# required for env_config
# import JSON
add(["JSON"])

# required packages for Binance
# import HTTP, SHA, JSON, Dates, Printf
add(["SHA", "JSON", "Dates", "Printf", "HTTP"])

# required packages for CryptoExchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# using Dates, DataFrames, DataAPI, JDF, CSV, Logging
# using ..MyBinance, ..Config
add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV", "Logging"])

# requies packages for ohlcv
# using Dates, DataFrames, CategoricalArrays, JDF, CSV, TimeZones, Logging
add(["Dates", "DataFrames", "TimeZones", "JDF", "CSV", "CategoricalArrays", "Logging"])

# requies packages for assets
# using Dates, DataFrames, Logging, JDF
add(["Dates", "DataFrames", "Logging", "JDF"])


# requies packages for targets
# noting yet

# requies packages for features
# import RollingFunctions: rollmedian, rolling
# import DataFrames: DataFrame
add(["RollingFunctions", "DataFrames"])

# required packages for trade
# using Dates, DataFrames
# using ..Config, ..Ohlcv, ..Classify, ..Exchange
add(["Dates", "DataFrames"])

# required packages for gradientgaindistribution
# using DataFrames, Logging, Statistics
add(["DataFrames", "Logging", "Statistics"])

# for applications (not yet used)
# using LoggingExtras