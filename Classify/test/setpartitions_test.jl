using Classify
using CategoricalArrays, Test
Classify.verbosity = 1
verbosity = 1

@testset "set partitions test" begin
samplesets = ["train", "test", "train", "eval", "train"]
samplesets = CategoricalArray(samplesets)
# Classify._test_setpartitions(129601, Dict("train"=>2/3, "test"=>1/6, "eval"=>1/6), 24*60, 1/13)
# psets = Classify.setpartitions(11:31, samplesets, partitionsize=2, gapsize=1, minpartitionsize = 2, maxpartitionsize = 2)
# # Dict{String, Vector{Any}}("test" => [14:15, 28:29], "train" => [11:12, 17:18, 23:26, 31:31], "eval" => [20:21])
# @test psets["test"] == [14:15, 28:29]
# @test psets["train"] == [11:12, 17:18, 23:26, 31:31]
# @test psets["eval"] == [20:21]

function rps(r, samplesets = samplesets, gapsize = 1, partitionsize = 3, minpartitionsize = 1, maxpartitionsize = 6)
    # rp = Classify._realpartitionsize(rowrange, samplesets, gapsize, partitionsize, minpartitionsize, maxpartitionsize)
    rp = Classify._realpartitionsize(r, samplesets, gapsize, partitionsize, minpartitionsize, maxpartitionsize)
    psets = Classify.setpartitions(r, samplesets, gapsize=gapsize, partitionsize=partitionsize, minpartitionsize=minpartitionsize, maxpartitionsize=maxpartitionsize)
    psets = [(string(setname), range) for (setname, range) in psets]
    return psets, rp
end

psets, rp = rps(11:31, samplesets, 1, 2, 2, 2)
# println(rp)
# println(psets)
@test rp == 2
@test psets == [("train", 11:12), ("test", 14:15), ("train", 17:18), ("eval", 20:21), ("train", 23:26), ("test", 28:29), ("train", 31:31)]


psets, rp = rps(1:38)
@test rp == 3
@test psets == [("train", 1:3), ("test", 5:7), ("train", 9:11), ("eval", 13:15), ("train", 17:22), ("test", 24:26), ("train", 28:30), ("eval", 32:34), ("train", 36:38)]
# @test psets["test"] == [5:7, 24:26]
# @test psets["train"] == [1:3, 9:11, 17:22, 28:30, 36:38]
# @test psets["eval"] == [13:15, 32:34]

psets, rp = rps(1:45)
@test rp == 4
@test psets == [("train", 1:4), ("test", 6:9), ("train", 11:14), ("eval", 16:19), ("train", 21:28), ("test", 30:33), ("train", 35:38), ("eval", 40:43), ("train", 45:45)]
# @test psets["test"] == [6:9, 30:33]
# @test psets["train"] == [1:4, 11:14, 21:28, 35:38, 45:45]
# @test psets["eval"] == [16:19, 40:43]

psets, rp = rps(1:37)
@test rp == 6
@test psets == [("train", 1:6), ("test", 8:13), ("train", 15:20), ("eval", 22:27), ("train", 29:37)]
# println(psets)
# @test psets["test"] == [8:13]
# @test psets["train"] == [1:6, 15:20, 29:37]
# @test psets["eval"] == [22:27]

psets, rp = rps(1:18)
@test rp == 3
@test psets == [("train", 1:3), ("test", 5:7), ("train", 9:11), ("eval", 13:15), ("train", 17:18)]
# println(psets)
# @test psets["test"] == [5:7]
# @test psets["train"] == [1:3, 9:11, 17:18]
# @test psets["eval"] == [13:15]

Classify.verbosity = 1

res = Classify.setpartitions(1:49, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 1, 3/50)
@test res["base"] == [1:5, 19:23, 37:41]
@test res["test"] == [7:8, 25:26, 43:44]
@test res["combi"] == [10:14, 28:32, 46:49]
@test res["eval"] == [16:17, 34:35]

end;
