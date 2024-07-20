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

@test length(EnvConfig.Authentication().key) > 0

EnvConfig.init(test)
@test length(EnvConfig.Authentication().key) > 0

struct ConfigTest <: AbstractConfiguration end
ct = ConfigTest()
dd = Dict(
    "param1" => Int16[3, 5],
    "param2" => Float32[1.2, 5]
)
df1 = EnvConfig.emptyconfigdf(ct)
@test size(df1) == (0,1)
@test eltype(df1[!, "cfgid"]) == Int16
# println("df1=$df1")
df2 = EnvConfig.emptyconfigdf(ct, dd)
@test size(df2) == (0,3)
@test eltype(df2[!, "param1"]) == Int16
@test eltype(df2[!, "param2"]) == Float32
# println("df2=$df2")
end

# println(EnvConfig.setsplitfilename())

end  # module
