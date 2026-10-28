@testset "Meaningful loss improvement and patience" begin
    @test !Classify.nnconverged(Float32[])
    @test !Classify.nnconverged(fill(1.0, 50))
    @test Classify.nnconverged(fill(1.0, 51))
    @test Classify.nnconverged([isodd(epoch) ? 1.0 : 1.01 for epoch in 1:51])
    @test !Classify.nnconverged([1.0 - 0.001 * epoch for epoch in 1:100])
    @test Classify.nnconverged(zeros(51))
    @test !Classify.nnconverged([1.0, 0.99]; patience=1)
    @test Classify.nnconverged([1.0, 1.01]; patience=1)

    # Binary-exact values exercise equality and cumulative smaller records.
    @test !Classify.nnconverged([1.0, 0.875]; patience=1, minrelativeimprovement=0.125)
    @test Classify.nnconverged([1.0, 0.9375]; patience=1, minrelativeimprovement=0.125)
    @test !Classify.nnconverged([1.0, 0.9375, 0.875]; patience=2, minrelativeimprovement=0.125)
    @test Classify.nnconverged([1.0, 0.9375, 0.875, 0.9, 0.89]; patience=2, minrelativeimprovement=0.125)
    @test Classify.nnconverged([1.0, 1.0]; patience=1, minrelativeimprovement=0)
    @test !Classify.nnconverged([1.0, 0.9999]; patience=1, minrelativeimprovement=0)
    @test Classify.nnconverged(vcat([1.0], fill(0.9999, 50)))
    @test !Classify.nnconverged(vcat(fill(1.0, 50), [0.999]))
    @test_throws AssertionError Classify.nnconverged([NaN])
    @test_throws AssertionError Classify.nnconverged([Inf])
    @test_throws AssertionError Classify.nnconverged([-1.0])
    @test_throws AssertionError Classify.nnconverged([1.0]; patience=0)
    @test_throws AssertionError Classify.nnconverged([1.0]; minrelativeimprovement=-0.1)
    @test_throws AssertionError Classify.nnconverged([1.0]; minrelativeimprovement=1)
    @test_throws AssertionError Classify.nnconverged([1.0]; minrelativeimprovement=NaN)
end

@testset "Adaptation total epoch limit and resumed patience" begin
    EnvConfig.init(EnvConfig.test)
    folder = mktempdir(mkpath(EnvConfig.defaultdebugfilespath))
    try
        x = zeros(Float32, 1, 4)
        labels = ["A", "A", "B", "B"]
        nn = _reinforce_test_nn("patience")
        src = Classify.MatrixSampleSource(nn, x, labels)
        Classify.adaptnn!(nn, src; maxepochs=1000, folderpath=folder)
        @test length(nn.losses) == 51
        @test Classify.nnconverged(nn)
        restored = Classify.loadnn(nn.fileprefix * "-lastepoch"; folderpath=folder)
        Classify.adaptnn!(restored, src; maxepochs=200, folderpath=folder)
        @test restored.losses == nn.losses

        nn = _reinforce_test_nn("totalcap")
        nn.losses = [2.0 - 0.001 * epoch for epoch in 1:199]
        Classify.adaptnn!(nn, src; maxepochs=1000, folderpath=folder)
        @test length(nn.losses) == 200
        @test !Classify.nnconverged(nn)
        previous = copy(nn.losses)
        Classify.adaptnn!(nn, src; maxepochs=200, folderpath=folder)
        @test nn.losses == previous

        nn = _reinforce_test_nn("smallcap")
        Classify.adaptnn!(nn, src; maxepochs=2, folderpath=folder)
        Classify.adaptnn!(nn, src; maxepochs=3, folderpath=folder)
        @test length(nn.losses) == 3
        @test_throws AssertionError Classify.adaptnn!(nn, src; maxepochs=0, folderpath=folder)
    finally
        rm(folder; recursive=true)
    end
end
