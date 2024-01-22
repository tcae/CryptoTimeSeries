module EnvConfigTest
using EnvConfig
using Test

# greet2()
# x = Authentication(production)
# println("a = $(x.secret)")
# println("a.key = $(x.key)")

# println(EnvConfig.datafile("btc_OHLCV", "csv"))
EnvConfig.init(production)


function testcheckbackup()
end
@testset "Config tests" begin

# println(pwd())
filename = "EnvConfigTest.txt"
write(filename, "world1")
EnvConfig.checkbackup(filename)
@test isfile(filename * "_1")
write(filename, "world2")
EnvConfig.checkbackup(filename)
@test isfile(filename * "_2")
rm(filename * "_1")
rm(filename * "_2")

dirname = "EnvConfigTest"
mkdir(dirname)
EnvConfig.checkbackup(dirname)
@test isdir(dirname * "_1")
mkdir(dirname)
EnvConfig.checkbackup(dirname)
@test isdir(dirname * "_2")
rm(dirname * "_1")
rm(dirname * "_2")

@test EnvConfig.datetimeformat == "yymmdd HH:MM"
# @test EnvConfig.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"
@test EnvConfig.Authentication().key == "vvUXlBGy67KRzHlLYJ"

EnvConfig.init(test)
@test EnvConfig.Authentication().key == "u3xKh7YRaqgP2PDnS8"


end

# println(EnvConfig.setsplitfilename())

end  # module
