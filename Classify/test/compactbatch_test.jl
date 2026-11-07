using Test, DataFrames
import Classify, EnvConfig

@testset "Coin-batched compact files" begin
    EnvConfig.init(EnvConfig.test)
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        features = DataFrame(first=Float32[1,2,3],second=Float32[4,5,6])
        cols = ["first","second"]
        index = DataFrame(coinix=[1,1,2],row=[3,1,2],labelix=[1,2,1])
        Classify.writecompactbatch!(folder,features,cols,index[1:2,:];append=false,nlabels=2)
        Classify.writecompactbatch!(folder,features,cols,index[3:3,:];append=true,nlabels=2)
        actual = Classify.readcompactsource(folder,2,3;nlabels=2,classbalancing=true)
        expected = Classify.CompactMmapSource([features,features],cols,index;
            folderpath=joinpath(folder,"oracle"),nlabels=2,classbalancing=true)
        @test actual.X==expected.X
        @test actual.labelix==expected.labelix
        @test actual.classweight==expected.classweight
        @test Classify.batch(actual,[1,3])==Classify.batch(expected,[1,3])
        @test_throws AssertionError Classify.readcompactsource(folder,2,4;nlabels=2,classbalancing=false)
        @test_throws AssertionError Classify.readcompactsource(folder,0,3;nlabels=2,classbalancing=false)
        @test_throws AssertionError Classify.writecompactbatch!(folder,features,cols,
            DataFrame(row=[1],labelix=[3]);append=false,nlabels=2)
        @test_throws AssertionError Classify.writecompactbatch!(folder,DataFrame(first=[1.0]),
            ["first"],DataFrame(row=[1],labelix=[1]);append=false,nlabels=2)
        large = DataFrame(coinix=ones(Int,8201),row=fill(2,8201),labelix=ones(Int,8201))
        Classify.writecompactbatch!(folder,features,cols,large;append=false,nlabels=2)
        mapped = Classify.readcompactsource(folder,2,8201;nlabels=2,classbalancing=false)
        @test all(==(2f0),mapped.X[1,:])
        @test all(==(5f0),mapped.X[2,:])
        @test all(==(1),mapped.labelix)
        Classify.writecompactbatch!(folder,features,cols,index[1:0,:];append=false,nlabels=2)
        @test filesize(joinpath(folder,Classify.TRAINFEATURES_FILE))==0
        @test filesize(joinpath(folder,Classify.TRAINLABELS_FILE))==0
    end
end
