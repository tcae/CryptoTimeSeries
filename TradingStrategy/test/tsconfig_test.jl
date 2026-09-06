@testset "trading strategy configs" begin
    for (name, cfg) in TradingStrategy.TS_CONFIGS
        @test cfg.configname == name
        @test keys(cfg) == (:configname, :tdconfigname, :tradingstrategy)
        @test cfg.tradingstrategy isa TradingStrategy.StrategyConfig
        # tdconfigname must resolve to a trend detector config supplying classifier, predictions and featconfig
        tdcfg = TradingStrategy.tstrendconfig(cfg)
        @test String(tdcfg.configname) == String(cfg.tdconfigname)
        @test TradingStrategy.tsconfig(name).configname == name
    end

    cfg1 = TradingStrategy.tsconfig("001")
    cfg2 = TradingStrategy.tsconfig("002")
    @test TradingStrategy.tsconfigref(cfg1) == "001"
    @test TradingStrategy.tsconfig("ts001").configname == "001"
    @test occursin("001", TradingStrategy.tsconfigsource(cfg1))
    # same classifier source, different trading strategies
    @test cfg1.tdconfigname == cfg2.tdconfigname
    @test cfg1.tradingstrategy.openthreshold != cfg2.tradingstrategy.openthreshold

    @test_throws AssertionError TradingStrategy.tsconfig("doesnotexist")
end
