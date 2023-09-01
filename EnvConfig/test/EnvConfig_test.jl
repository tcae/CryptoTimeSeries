module EnvConfigTest
using EnvConfig
using Test

# greet2()
# x = Authentication(production)
# println("a = $(x.secret)")
# println("a.key = $(x.key)")

# println(EnvConfig.datafile("btc_OHLCV", "csv"))
EnvConfig.init(production)

@testset "Config tests" begin

@test EnvConfig.datetimeformat == "yymmdd HH:MM"
# @test EnvConfig.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"
@test EnvConfig.Authentication().key == "vvUXlBGy67KRzHlLYJ"

end

# println(EnvConfig.setsplitfilename())

end  # module
