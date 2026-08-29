module TsmTradesEpochAllocationTest
using Test
using Dates
using DataFrames

using EnvConfig, TSM, Targets

# A scheduled tradeselection defines an epoch whose rows can be allocated once. Growing the
# frame per tick instead either copies the whole frame (_inserttradesrow!) or invalidates
# held TradesColumns, so these tests pin that the epoch is allocated up front and that a
# never-processed minute stays identifiable.

const QUOTE = "USDT"

function epochcache()
    tsm = TSM.TsmCache()
    TSM.ensuretradesschema!(tsm, TSM.tradesdf_all_contributors())
    return tsm
end

@testset "preparetradesepoch! allocates the whole minute grid" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    enddt = startdt + Minute(9)
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, enddt; startdt=startdt)

    @test nrow(tdf) == 10
    @test tdf[!, :opentime] == collect(startdt:Minute(1):enddt)
    @test all(String(p) == TSM.tradingpairkey("BTC", QUOTE) for p in tdf[!, :pair])
end

@testset "allocated but unprocessed minutes are identifiable" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(4); startdt=startdt)

    @test all(==(0f0), tdf[!, :score])
    @test all(String(s) == TSM.TSM_NO_STATE for s in tdf[!, :tsmstate])
end

@testset "epoch rows remove the per-tick grow paths" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    enddt = startdt + Minute(19)
    TSM.preparetradesepoch!(tsm, "BTC", QUOTE, enddt; startdt=startdt)
    before = nrow(TSM.trades(tsm, "BTC", QUOTE))

    for opentime in startdt:Minute(1):enddt
        entry = TSM.ensuretradesrow!(tsm, "BTC", QUOTE, opentime)
        @test entry.tradesdf[entry.rowix, :opentime] == opentime
    end
    # every tick found a pre-allocated row, so the frame never grew
    @test nrow(TSM.trades(tsm, "BTC", QUOTE)) == before
end

@testset "column handles stay valid across an allocated epoch" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    enddt = startdt + Minute(29)
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, enddt; startdt=startdt)
    cols = TSM.TradesColumns(tdf)

    for (i, opentime) in enumerate(startdt:Minute(1):enddt)
        entry = TSM.ensuretradesrow!(tsm, "BTC", QUOTE, opentime)
        cols.close[entry.rowix] = Float32(i)
    end
    # writes through the pre-epoch handle must be visible in the stored frame
    @test TSM.trades(tsm, "BTC", QUOTE)[!, :close] == Float32.(1:30)
end

@testset "preparetradesepoch! preserves already stored rows" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(4); startdt=startdt)
    # the row cursor only advances, so walk up to the row that is written
    local entry
    for opentime in startdt:Minute(1):(startdt + Minute(2))
        entry = TSM.ensuretradesrow!(tsm, "BTC", QUOTE, opentime)
    end
    TSM.settrades_close!(entry.tradesdf, entry.rowix, 123f0)

    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(9))
    @test nrow(tdf) == 10
    @test tdf[3, :close] == 123f0
    @test tdf[!, :opentime] == collect(startdt:Minute(1):(startdt + Minute(9)))
end

@testset "preparetradesepoch! fills minutes missing inside the stored range" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    # a seeded replay source carries only liquid minutes, so the stored range has holes
    seeded = DataFrame(opentime=[startdt, startdt + Minute(1), startdt + Minute(5)], close=Float32[10f0, 11f0, 15f0])
    TSM.settrades!(tsm, "BTC", QUOTE, seeded)

    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(7))
    @test tdf[!, :opentime] == collect(startdt:Minute(1):(startdt + Minute(7)))
    @test tdf[1, :close] == 10f0
    @test tdf[2, :close] == 11f0
    @test tdf[6, :close] == 15f0
    # the filled gap minutes stay identifiable as never processed
    @test tdf[3, :close] == 0f0
    @test all(String(tdf[i, :tsmstate]) == TSM.TSM_NO_STATE for i in (3, 4, 5, 7, 8))
end

@testset "live epoch (enddt===nothing) extends by epochminutes" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, nothing; startdt=startdt, epochminutes=5)
    @test tdf[!, :opentime] == collect(startdt:Minute(1):(startdt + Minute(5)))

    # the next scheduled tradeselection extends the open ended epoch again
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, nothing; epochminutes=3)
    @test tdf[!, :opentime] == collect(startdt:Minute(1):(startdt + Minute(8)))
end

@testset "live epoch without epochminutes fails fast" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    @test_throws AssertionError TSM.preparetradesepoch!(tsm, "BTC", QUOTE, nothing; startdt=startdt)
end

@testset "preparetradesepoch! is a no-op when the epoch is already covered" begin
    tsm = epochcache()
    startdt = DateTime(2026, 5, 1, 0, 0)
    TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(9); startdt=startdt)
    tdf = TSM.preparetradesepoch!(tsm, "BTC", QUOTE, startdt + Minute(4))
    @test nrow(tdf) == 10
end

end # module
