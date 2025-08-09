module EnvConfigTest
using EnvConfig
using Test

# greet2()
# x = Authentication(production)
# println("a = $(x.secret)")
# println("a.key = $(x.key)")

# println(EnvConfig.datafile("btc_OHLCV", "csv"))
EnvConfig.init(production)


function testsavebackup()
    println(pwd())
    filename = "EnvConfigTest.txt"
    write(filename, "world1")
    bf = EnvConfig.savebackup(filename)
    @test all(isfile(f) for f in bf)
    write(filename, "world2")
    bf = EnvConfig.savebackup(filename)
    @test all(isfile(f) for f in bf)
    @test length(bf) == 2
    write(filename, "world3")
    bf = EnvConfig.savebackup(filename, maxbackups=2)
    @test all(isfile(f) for f in bf)
    @test length(bf) == 2
    rm.(bf)

    dirname = "EnvConfigTest"
    mkdir(dirname)
    bf = EnvConfig.savebackup(dirname)
    @test all(isdir(f) for f in bf)
    mkdir(dirname)
    bf = EnvConfig.savebackup(dirname)
    @test all(isdir(f) for f in bf)
    rm.(bf)
end

function testauthentication()
    EnvConfig.init(production)
    @test length(EnvConfig.Authentication().key) > 0

    EnvConfig.init(test)
    @test length(EnvConfig.Authentication().key) > 0
end

struct ConfigTest <: AbstractConfiguration end

function testconfig()
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

@testset "Config tests" begin

EnvConfig.verbosity = 3
testsavebackup()
testauthentication()
testconfig()

@test EnvConfig.datetimeformat == "yymmdd HH:MM"
# @test EnvConfig.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"


end

# println(EnvConfig.setsplitfilename())

end  # module
