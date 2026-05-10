using Test

@testset "Trade tests" begin
    include("storage_format_test.jl")
    include("algorithm03_adapter_test.jl")
end
