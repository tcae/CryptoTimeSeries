using Test, Dates, DataFrames, Targets
include("../scripts/investigate_trend05_horizon.jl")

@testset "Stored target horizon audit" begin
    start = DateTime("2025-01-01")
    times = [start + Minute(ix) for ix in 0:12]
    codes = fill(Targets.tradelabelcode(Targets.allclose), 13)
    codes[1] = Targets.tradelabelcode(Targets.longopen)
    codes[2:11] .= Targets.tradelabelcode(Targets.longhold)
    targets = DataFrame(opentime=times, label=codes)
    sets = DataFrame(opentime=times[[1, 2, 8, 9]], set=["train", "train", "eval", "eval"], rangeid=[1, 1, 2, 2])
    summary, examples = Trend05HorizonInvestigation.auditsegments(targets, sets, 5)
    @test summary.entries == 1
    @test summary.overgap == 1
    @test summary.maximum_minutes == 10
    @test summary.gapcrossings == 1
    @test summary.differentsetcrossings == 1
    @test length(examples) == 1
    @test examples[1].reached == times[11]
    sets.set .= "train"
    summary, _ = Trend05HorizonInvestigation.auditsegments(targets, sets, 5)
    @test summary.gapcrossings == 1
    @test summary.differentsetcrossings == 0
    summary, _ = Trend05HorizonInvestigation.auditsegments(targets, sets[1:2, :], 5)
    @test summary.gapcrossings == 0
    # An entry in a gap can still give a source-partition hold a future dependency.
    sets = DataFrame(opentime=times[[2, 3, 8, 9]], set=["train", "train", "eval", "eval"], rangeid=[1, 1, 2, 2])
    summary, _ = Trend05HorizonInvestigation.auditsegments(targets, sets, 5)
    @test summary.differentsetcrossings == 1
    summary, _ = Trend05HorizonInvestigation.auditsegments(targets, sets[1:0, :], 5)
    @test summary.gapcrossings == 0
    @test_throws AssertionError Trend05HorizonInvestigation.auditsegments(targets, sets, 0)
    duplicated = vcat(targets, targets[end:end, :])
    @test_throws AssertionError Trend05HorizonInvestigation.auditsegments(duplicated, sets, 5)
end
