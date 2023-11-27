
module ClassifyTest
using Dates, DataFrames, MLJ
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
@test res["base"] == [1:6, 20:24, 38:42]
@test res["test"] == [8:9, 26:27, 44:45]
@test res["combi"] == [11:15, 29:33, 47:49]
@test res["eval"] == [17:18, 35:36]
end

end  # module