module EnvConfigTest
using EnvConfig
using Test, DataFrames, CategoricalArrays

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
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)
    write(joinpath(dirname, filename), "world1")
    bf = EnvConfig.savebackup(dirname)
    @test all(isdir(f) for f in bf)
    mkdir(dirname)
    write(joinpath(dirname, filename), "world2")
    bf = EnvConfig.savebackup(dirname, maxbackups=1)
    @test all(isdir(f) for f in bf)
    rm.(bf, recursive=true)
end

function testauthentication()
    EnvConfig.init(production)
    @test length(EnvConfig.Authentication().key) > 0

    EnvConfig.init(test)
    @test length(EnvConfig.Authentication().key) > 0
end

struct ConfigTest <: AbstractConfiguration end

@enum ExampleArrowState begin
    alpha_state
    beta_state
end

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

function testtableio()
    df = DataFrame(ix=Int32[1, 2, 3], rangeid=Int64[1, 256, 21571], offset=Int64[-2, 0, 3], value=Float32[1.5f0, 2.5f0, 3.5f0], label=["a", "b", "c"])
    oldformat = EnvConfig.dfformat()
    mktempdir() do tmpdir
        try
            EnvConfig.setdfformat!(:jdf)
            jdfpath = EnvConfig.savedf(df, "sample_default"; folderpath=tmpdir)
            arrowpath = EnvConfig.savedf(df, "sample_arrow"; folderpath=tmpdir, format=:arrow)

            @test isdir(jdfpath)
            @test isfile(arrowpath)
            @test EnvConfig.tableexists("sample_default"; folderpath=tmpdir, format=:jdf)
            @test EnvConfig.tableexists("sample_arrow"; folderpath=tmpdir, format=:arrow)
            @test EnvConfig.tableexists("sample_default"; folderpath=tmpdir, format=:auto)
            @test endswith(EnvConfig.tablepath("sample_arrow"; folderpath=tmpdir, format=:arrow), ".arrow")
            @test endswith(EnvConfig.tablepath("sample_default"; folderpath=tmpdir, format=:jdf), ".jdf")

            jdfdf = EnvConfig.readdf("sample_default"; folderpath=tmpdir)
            arrowdf = EnvConfig.readdf("sample_arrow"; folderpath=tmpdir, format=:arrow)
            arrowmutable = EnvConfig.readdf("sample_arrow"; folderpath=tmpdir, format=:arrow, copycols=true)
            fallbackdf = EnvConfig.readdf("sample_default"; folderpath=tmpdir, format=:arrow)

            @test size(jdfdf) == size(df)
            @test size(arrowdf) == size(df)
            @test size(fallbackdf) == size(df)
            @test jdfdf[!, :ix] == df[!, :ix]
            @test arrowdf[!, :ix] == df[!, :ix]
            @test arrowdf[!, :rangeid] == df[!, :rangeid]
            @test arrowdf[!, :offset] == df[!, :offset]
            @test eltype(arrowdf[!, :ix]) == UInt8
            @test eltype(arrowdf[!, :rangeid]) == UInt16
            @test eltype(arrowdf[!, :offset]) == Int8
            @test arrowdf[!, :value] == df[!, :value]
            @test fallbackdf[!, :label] == df[!, :label]
            @test_throws ReadOnlyMemoryError arrowdf[1, :value] = 9.0f0
            arrowmutable[1, :value] = 9.0f0
            @test arrowmutable[1, :value] == 9.0f0

            EnvConfig.setdfformat!(:arrow)
            switchedpath = EnvConfig.savedf(df, "sample_switched"; folderpath=tmpdir)
            switcheddf = EnvConfig.readdf("sample_switched"; folderpath=tmpdir)
            @test isfile(switchedpath)
            @test size(switcheddf) == size(df)
            @test switcheddf[!, :ix] == df[!, :ix]

            enumdf = DataFrame(state=ExampleArrowState[alpha_state, beta_state], maybe=Union{Missing, ExampleArrowState}[alpha_state, missing])
            enumpath = EnvConfig.savedf(enumdf, "enum_sample"; folderpath=tmpdir, format=:arrow)
            enumloaded = EnvConfig.readdf("enum_sample"; folderpath=tmpdir, format=:arrow)
            @test isfile(enumpath)
            @test enumloaded[!, :state] == Int8[Int(alpha_state), Int(beta_state)]
            @test enumloaded[1, :maybe] == Int8(Int(alpha_state))
            @test ismissing(enumloaded[2, :maybe])
            @test eltype(enumloaded[!, :state]) == Int8
        finally
            EnvConfig.setdfformat!(oldformat)
        end
    end
end

@testset "Config tests" begin

EnvConfig.verbosity = 3
testsavebackup()
testauthentication()
testconfig()
testtableio()

@test EnvConfig.datetimeformat == "yymmdd HH:MM"
# @test EnvConfig.datafile("btc_OHLCV", "_df.csv") == "/home/tor/crypto/Features/btc_OHLCV_df.csv"


end

# println(EnvConfig.setsplitfilename())

end  # module
