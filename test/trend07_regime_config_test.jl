using Test, Dates, DataFrames
import EnvConfig, TradingStrategy, Targets, Features
include("../scripts/RegimeTraining.jl")

@testset "Trend07 common RegimeTraining preparation and indices" begin
    EnvConfig.init(EnvConfig.test)
    rc = RegimeTraining.buildrunconfig(Dict("ts"=>"071"))
    @test rc.mode == "test"
    @test rc.clcfg.targets == "T28"
    @test TradingStrategy.regimepurge(TradingStrategy.targetconfig("T28")) == 240
    synthetic = RegimeTraining.inputs(Dict("mode"=>"synthetic","ts"=>"071"))
    input = merge(synthetic,(mode="test",rc=rc,
        folds=TradingStrategy.FoldConfig(anchor=synthetic.startdt,foldminutes=24000,
            blockminutes=720,gapminutes=240,minminutes=120),
        featurefactory=()->TradingStrategy.featureconfig("F22"),
        targetfactory=()->TradingStrategy.targetconfig("T28")))
    mktempdir(mkpath(EnvConfig.defaultdebugfilespath)) do folder
        prepared = RegimeTraining.prepare!(input,folder)
        @test nrow(prepared.calendar) > 0
        @test !isdir(joinpath(folder,"isolated"))
        for (coin,features,targets) in zip(prepared.stores.coins,
                prepared.stores.featurestores,prepared.stores.targetstores)
            @test isfile(joinpath(folder,"stores",coin,"targets.arrow"))
            @test features.opentime[1] >= input.startdt
            @test any(==(Targets.tradelabelcode(Targets.longopen)),targets.label)
            @test any(==(Targets.tradelabelcode(Targets.shortopen)),targets.label)
            @test all(==(Targets.tradelabelcode(Targets.ignore)),targets.label[end-239:end])
        end
        labels = TradingStrategy.classifierlabels(rc.clcfg)
        indices = RegimeTraining.persistedtrainingindices!(prepared,prepared.sets,labels,
            mkpath(joinpath(folder,"indices")),"trend07-test";folds=Int[],regimes=String[])
        @test nrow(indices.trainindex) > 0
        @test nrow(indices.evalindex) > 0
        @test nrow(indices.testindex) > 0
    end
end
