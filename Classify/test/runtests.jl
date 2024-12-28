
module ClassifyTest
using CategoricalArrays
using CategoricalDistributions, StatisticalMeasures
import CategoricalDistributions:classes
using Dates, DataFrames
using Test
using Classify

@testset "Classify tests" begin

# y = categorical(["X", "O", "X", "X", "O", "X", "X", "O", "O", "X"], ordered=true)
y = categorical(["O", "X", "O", "O", "X", "O", "O", "X", "X", "O"], ordered=true)
# probabilistic predictions:
# X_probs = [0.3, 0.2, 0.4, 0.9, 0.1, 0.4, 0.5, 0.2, 0.8, 0.7]
X_probs = [0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9]
# X_probs = [0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1]
ŷ = UnivariateFinite(["X", "O"], X_probs, augment=true, pool=y)
score = auc(ŷ, y)
println("auc=$score")
df = DataFrame(String(classes(ŷ)[1]) => pdf(ŷ, classes(ŷ))[:, 1], String(classes(ŷ)[2]) => pdf(ŷ, classes(ŷ))[:, 2], "truth" => y)
# println("classes=$(classes(ŷ)) pdf=$(pdf(ŷ, classes(ŷ)))")
println(df)

res = Classify.setpartitions(1:49, Dict("base"=>1/3, "combi"=>1/3, "test"=>1/6, "eval"=>1/6), 1, 3/50)
@test res["base"] == [1:5, 19:23, 37:41]
@test res["test"] == [7:8, 25:26, 43:44]
@test res["combi"] == [10:14, 28:32, 46:49]
@test res["eval"] == [16:17, 34:35]

@test Classify.score2bin(0.95, 10) == 10
@test Classify.score2bin(1.15, 10) == 10
@test Classify.score2bin(0.55, 10) == 6
@test Classify.score2bin(0.05, 10) == 1
@test Classify.score2bin(0.0, 10) == 1
@test Classify.score2bin(-0.05, 10) == 1
end

end  # module