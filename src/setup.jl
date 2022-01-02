import Pkg: activate, add, status
activate(pwd())
add(["Pkg", "Test"])

# required for env_config
# import JSON
add(["JSON"])

# required packages for Binance
# import HTTP, SHA, JSON, Dates, Printf
add(["SHA", "JSON", "Dates", "Printf", "HTTP"])

# required packages for Exchange
# add(PackageSpec(url="https://github.com/DennisRutjes/Binance.jl",rev="master"))
# using Dates, DataFrames, DataAPI
# using JDF, CSV
# using ..MyBinance, ..Config
add(["Dates", "DataFrames", "DataAPI", "JDF", "CSV"])

# requies packages for ohlcv
# using Dates, DataFrames, CategoricalArrays, JDF, CSV, TimeZones
add(["Dates", "DataFrames", "TimeZones", "JDF", "CSV", "CategoricalArrays"])

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

