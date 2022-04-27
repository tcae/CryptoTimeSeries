module ConfigTest
using EnvConfig
using Test

# greet2()
# x = Authentication(production)
# println("a = $(x.secret)")
# println("a.key = $(x.key)")

# println(EnvConfig.datafile("btc_OHLCV", "csv"))
EnvConfig.init(production)

@testset "Config tests" begin

@test EnvConfig.datetimeformat == "yyyy-mm-dd HH:MM"
# @test EnvConfig.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"
@test EnvConfig.Authentication().key == "5gchI8bnzXYAimGmv4Wn6yQ2Yp5o6cwDBsyhrRawVDPcqTD43Rd6sOe13Xbbbrpv"

end

# println(EnvConfig.setsplitfilename())

end  # module
