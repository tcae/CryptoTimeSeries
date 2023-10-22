using DataFrames, Features
using Test

rdf = DataFrame()
colname = ""
df = DataFrame((colA = [1.1f0, 3.5f0, 8.0f0]))
@testset "Features tests" begin

    rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
    @test rdf[!, "colA01"] == [1.1f0, 1.1f0, 3.5f0]
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",2, 1, size(df,1); fill=nothing)
    @test rdf[!, "colA02"] == [1.1f0, 1.1f0, 1.1f0]
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",3, 1, size(df,1); fill=nothing)
    @test rdf[!, "colA03"] == [1.1f0, 1.1f0, 1.1f0]
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",3, 1, size(df,1); fill=0)
    @test rdf[!, "colA03"] == [0.0f0, 0.0f0, 0.0f0]
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",2, 2, size(df,1); fill=0)
    @test rdf[!, "colA02"] == [0.0f0, 1.1f0]
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
    rdf, colname = Features.lookbackrow!(rdf, df, "colA",2, 1, size(df,1); fill=nothing)
    @test size(rdf) == (3, 2)
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",1, 1, size(df,1); fill=nothing)
    @test_throws AssertionError Features.lookbackrow!(rdf, df, "colA",2, 2, size(df,1); fill=0)
    rdf, colname = Features.lookbackrow!(nothing, df, "colA",0, 1, size(df,1); fill=nothing)
    @test rdf[!, "colA00"] == [1.1f0, 3.5f0, 8.0f0]
end # testset
