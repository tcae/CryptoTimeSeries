using Test, Dates, DataFrames
using Classify, EnvConfig, Ohlcv, TestOhlcv

@testset "Explicit sets store rebuild" begin
    EnvConfig.init(EnvConfig.test)
    oldpath = EnvConfig.coinspath()
    mktempdir(EnvConfig.defaultdebugfilespath) do folder
        EnvConfig.setcoinspath!(folder)
        try
            startdt = DateTime("2025-02-17T13:30:00")
            enddt = startdt + Day(2) - Minute(1)
            ohlcv = TestOhlcv.testohlcv("SINE", startdt, enddt)
            partition = (samplesets=["train", "test", "train", "eval"],
                partitionsize=60, gapsize=10, minpartitionsize=30, maxpartitionsize=120)
            id = "PTEST_DTEST"
            original = Classify.updatesets!(ohlcv, id, partition, startdt, enddt; confighash="old")
            @test nrow(original) > 0
            path = joinpath(Classify.setsfolder("SINE", ohlcv.quotecoin), id * ".arrow")
            bytes = read(path)
            @test isequal(Classify.updatesets!(ohlcv, id, partition, startdt, enddt;
                confighash="old"), original)
            @test read(path) == bytes
            @test_throws AssertionError Classify.updatesets!(ohlcv, id, partition,
                startdt, enddt; confighash="new")
            @test read(path) == bytes

            newenddt = enddt - Hour(2)
            rebuilt = Classify.updatesets!(ohlcv, id, partition, startdt + Day(1),
                newenddt; confighash="new", rebuild=true)
            @test nrow(rebuilt) > 0
            @test all(dt -> startdt + Day(1) <= dt <= newenddt, rebuilt.opentime)
            @test nrow(rebuilt) < nrow(original)
            @test_throws AssertionError Classify.readsets("SINE", ohlcv.quotecoin, id; confighash="old")
            @test isequal(Classify.readsets("SINE", ohlcv.quotecoin, id; confighash="new"), rebuilt)

            metadata = EnvConfig.tablemetadata(id; folderpath=dirname(path))
            altered = DataFrame(rebuilt; copycols=true)
            altered.set .= "stale"
            EnvConfig.savedf(altered, id; folderpath=dirname(path), format=:arrow, metadata=metadata)
            @test isequal(Classify.updatesets!(ohlcv, id, partition, startdt + Day(1),
                newenddt; confighash="new"), altered)
            @test isequal(Classify.updatesets!(ohlcv, id, partition, startdt + Day(1),
                newenddt; confighash="new", rebuild=true), rebuilt)

            bytes = read(path)
            invalid = merge(partition, (samplesets=String[],))
            @test_throws AssertionError Classify.updatesets!(ohlcv, id, invalid,
                startdt, enddt; confighash="invalid", rebuild=true)
            @test read(path) == bytes

            empty = Classify.updatesets!(ohlcv, id, partition, enddt + Day(1),
                enddt + Day(2); confighash="empty", rebuild=true)
            @test nrow(empty) == 0
            @test names(empty) == ["opentime", "set", "rangeid"]
            @test EnvConfig.tablemetadata(id; folderpath=dirname(path))["coverage"] == "empty"
        finally
            EnvConfig.setcoinspath!(oldpath)
        end
    end
end
