using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/env_config.jl")  # in contrast to srcdir() this is used by vscode
# include(srcdir("env_config.jl"))

module ConfigTest
using ..Config
using Test

# greet2()
# x = Authentication(production)
# println("a = $(x.secret)")
# println("a.key = $(x.key)")

# println(Config.datafile("btc_OHLCV", "csv"))
Config.init(production)

@testset "Config tests" begin

@test Config.datetimeformat == "%Y-%m-%d_%Hh%Mm"
@test Config.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"
@test Config.Authentication().key == "5gchI8bnzXYAimGmv4Wn6yQ2Yp5o6cwDBsyhrRawVDPcqTD43Rd6sOe13Xbbbrpv"

end

# println(Config.setsplitfilename())

end  # module
