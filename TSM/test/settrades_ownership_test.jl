module TsmSettradesOwnershipTest
using Test
using Dates
using DataFrames
using CategoricalArrays

using EnvConfig, TSM, Targets

# A Trades frame stored in TsmCache must own its columns. Storing a view-backed frame
# (e.g. `DataFrame(groupview; copycols=false)`) would make every write in the replay row
# loop mutate the source table instead of the Trades state.

@testset "settrades! rejects view-backed columns" begin
    source = DataFrame(
        opentime=[DateTime(2026, 1, 1) + Minute(i) for i in 1:6],
        score=Float32[0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0, 0.6f0],
        close=Float32[10f0, 11f0, 12f0, 13f0, 14f0, 15f0],
        grp=[1, 1, 1, 2, 2, 2],
    )
    groupview = groupby(source, :grp)[1]

    tsm = TSM.TsmCache()
    TSM.ensuretradesschema!(tsm, TSM.tradesdf_all_contributors())

    aliased = DataFrame(groupview; copycols=false)
    select!(aliased, Not(:grp))
    @test_throws AssertionError TSM.settrades!(tsm, "BTC", "USDC", aliased)

    owned = DataFrame(groupview)
    select!(owned, Not(:grp))
    TSM.settrades!(tsm, "BTC", "USDC", owned)

    stored = TSM.trades(tsm, "BTC", "USDC")
    @test stored[!, :score] isa Vector{Float32}
    @test TSM.TradesColumns(stored) isa TSM.TradesColumns

    # writes must not reach the source frame
    TSM.settrades_score!(stored, 1, 0.99f0)
    @test stored[1, :score] == 0.99f0
    @test source[1, :score] == 0.1f0
end

# EnvConfig._arrow_safe_table compacts integer columns to their narrowest type on write, so
# any frame reloaded from cache carries storage types that differ from the Trades schema.
@testset "settrades! normalizes Arrow round-trip column types" begin
    oldformat = EnvConfig.dfformat()
    tmpdir = mktempdir()
    try
        EnvConfig.setdfformat!(:arrow)
        n = 6
        results = DataFrame(
            opentime=[DateTime(2026, 3, 1) + Minute(i) for i in 1:n],
            rangeid=fill(10_000, n),              # Int64 in memory, UInt16 once compacted
            score=Float32[0.5f0 + 0.01f0 * i for i in 1:n],
            close=Float32[100f0 + i for i in 1:n],
            label=fill(Targets.longopen, n),
        )
        EnvConfig.savedf(results, "results-roundtrip"; folderpath=tmpdir)
        reloaded = DataFrame(EnvConfig.readdf("results-roundtrip"; folderpath=tmpdir, copycols=true))

        # the storage layer really does change the integer width
        @test !(reloaded[!, :rangeid] isa Vector{Int32})

        tsm = TSM.TsmCache()
        TSM.ensuretradesschema!(tsm, TSM.tradesdf_all_contributors())
        TSM.settrades!(tsm, "BTC", "USDC", reloaded)
        stored = TSM.trades(tsm, "BTC", "USDC")

        @test stored[!, :rangeid] isa Vector{Int32}
        @test all(stored[!, :rangeid] .== Int32(10_000))
        @test stored[!, :score] isa Vector{Float32}
        @test eltype(stored[!, :label]) <: Targets.TradeLabel
        @test TSM.TradesColumns(stored) isa TSM.TradesColumns
    finally
        EnvConfig.setdfformat!(oldformat)
        rm(tmpdir; force=true, recursive=true)
    end
end

# Classify builds :set with `allowmissing!`, and Arrow dictionary encoding yields UInt32
# refs, so adopted categoricals differ from the schema in both eltype and reference width.
@testset "settrades! normalizes foreign categorical pools" begin
    n = 4
    df = DataFrame(opentime=[DateTime(2026, 4, 1) + Minute(i) for i in 1:n])
    df[!, :set] = CategoricalVector(fill("train", n), levels=["train", "test"])
    allowmissing!(df, :set)
    df[!, :lo_id] = CategoricalArray{String, 1, UInt8}(fill("none", n))

    @test !(df[!, :set] isa TSM.TradesCat8Column)
    @test !(df[!, :lo_id] isa TSM.TradesCat32Column)

    tsm = TSM.TsmCache()
    TSM.ensuretradesschema!(tsm, TSM.tradesdf_all_contributors())
    TSM.settrades!(tsm, "BTC", "USDC", df)
    stored = TSM.trades(tsm, "BTC", "USDC")

    @test stored[!, :set] isa TSM.TradesCat8Column
    @test stored[!, :lo_id] isa TSM.TradesCat32Column
    @test all(String.(stored[!, :set]) .== "train")
    @test TSM.TradesColumns(stored) isa TSM.TradesColumns
end

@testset "settrades! materializes unset categorical cells to the schema default" begin
    n = 3
    df = DataFrame(opentime=[DateTime(2026, 4, 1) + Minute(i) for i in 1:n])
    df[!, :set] = CategoricalVector(fill("train", n))
    allowmissing!(df, :set)
    df[2, :set] = missing

    tsm = TSM.TsmCache()
    TSM.ensuretradesschema!(tsm, TSM.tradesdf_all_contributors())
    TSM.settrades!(tsm, "BTC", "USDC", df)
    stored = TSM.trades(tsm, "BTC", "USDC")

    @test stored[!, :set] isa TSM.TradesCat8Column
    @test String.(stored[!, :set]) == ["train", TSM.TSM_CATEGORICAL_DEFAULT, "train"]
    @test TSM.TradesColumns(stored) isa TSM.TradesColumns
end

@testset "every categorical Trades column defaults to TSM_CATEGORICAL_DEFAULT" begin
    # `_canonicalcategorical` materializes unset cells to a single shared default, which is
    # only correct while every categorical column agrees on it.
    for col in (TSM.TSM_CATEGORICAL_COLUMNS..., TSM.TSM_ID_COLUMNS...)
        @test String(TSM._defaultcolumn(col, 1)[1]) == TSM.TSM_CATEGORICAL_DEFAULT
    end
end

end # module
