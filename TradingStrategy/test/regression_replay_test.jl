using Dates, Test, DataFrames, Classify, Features, Targets, TSM, TradingStrategy

"""Create independent canonical rows for checking shared live/replay gate behavior."""
function regressionreplayrows(labels::Vector{Targets.TradeLabel})
    frame = DataFrame(opentime=collect(DateTime(2025):Minute(1):DateTime(2025)+Minute(length(labels)-1)),
        pair=fill("SINEUSDT",length(labels)),close=fill(2f0,length(labels)),
        high=fill(2.01f0,length(labels)),low=fill(1.99f0,length(labels)),
        score=fill(0.9f0,length(labels)),label=copy(labels))
    for contributor in TSM.tradesdf_all_contributors()
        contributor(frame)
    end
    return frame
end

"""Measure warmed replay guidance with its row mutations observable to the caller."""
function regressionreplayallocations(cfg,context,cols,n)
    TradingStrategy.gain_limit_reversal_below_regression!(cfg,context,cols,1)
    return @allocated for _ in 1:n
        cols.label[1] = Targets.shortopen
        TradingStrategy.gain_limit_reversal_below_regression!(cfg,context,cols,1)
    end
end

@testset "Prealigned regression gates preserve live semantics" begin
    labels = Targets.TradeLabel[Targets.longopen,Targets.longstrongopen,
        Targets.shortopen,Targets.shortstrongopen,Targets.longhold,Targets.shorthold,Targets.allclose]
    reference = regressionreplayrows(labels)
    times = reference.opentime
    prices = Float32[2.1,2.1,1.9,1.9,2,2,2]
    gradients = Float32[0,0,0,0,0,0,0]
    columns = TradingStrategy.RegressionColumns(times,prices,gradients)
    @test TradingStrategy._regressionvalue(columns,times[1]) == (price=2.1f0,gradient=0f0)
    @test_throws AssertionError TradingStrategy._regressionvalue(columns,times[end]+Minute(1))
    @test_throws AssertionError TradingStrategy._alignregression(columns,[times[end]+Minute(1)])
    @test_throws AssertionError TradingStrategy._alignregression(
        TradingStrategy.RegressionColumns(reverse(times),prices,gradients),times)
    @test_throws AssertionError TradingStrategy._alignregression(
        TradingStrategy.RegressionColumns(times,prices[1:end-1],gradients),times)
    @test_throws AssertionError TradingStrategy._regressioncolumns(Features.Features006(),240)
    @test TradingStrategy._alignregression(columns,times).price == prices

    nn = Classify.model011(1,labels,"regression-replay-test")
    classifier = Classify.TrendClassifier001(nn;featconfig=TradingStrategy.trendf6config09)
    featcfg = Features.Features006()
    featcfg.fdfno = DataFrame(opentime=times)
    for window in (60,240)
        featcfg.fdfno[!,"ry+$window"] = copy(prices)
        featcfg.fdfno[!,"rg+$window"] = copy(gradients)
    end
    classifier.bc["SINE"] = (featcfg=featcfg,)
    for (trigger,trend) in ((240,240),(240,60),(240,0),(0,240),(0,0))
        cfg = TradingStrategy.GainLimitReversalBelowRegressionConfig(
            triggerregrwindow=trigger,trendregrwindow=trend,trendgainthreshold=0.01f0)
        live = regressionreplayrows(labels)
        replay = regressionreplayrows(labels)
        cols = TSM.TradesColumns(replay)
        context = TradingStrategy._replayregressions(cfg,classifier,cols,length(labels),nothing)
        @test context isa TradingStrategy.ReplayRegressionContext
        trigger == trend && trigger > 0 && @test context.trigger === context.trend
        for ix in eachindex(labels)
            TradingStrategy.gain_limit_reversal_below_regression!(cfg,classifier,TSM.TradesColumns(live),ix)
            TradingStrategy.gain_limit_reversal_below_regression!(cfg,context,cols,ix)
        end
        @test isequal(live,replay)
        @test regressionreplayallocations(cfg,context,cols,1000) <= 128*1000
    end
    @test_throws AssertionError TradingStrategy._replayregressions(
        TradingStrategy.GainLimitReversalBelowRegressionConfig(),nothing,
        TSM.TradesColumns(reference),length(labels),Dict{Int,DataFrame}())

    for price in (-1f0,0f0,NaN32,Inf32,-Inf32), label in labels[1:4]
        cfg = TradingStrategy.GainLimitReversalBelowRegressionConfig()
        rows = regressionreplayrows([label])
        cols = TSM.TradesColumns(rows)
        invalid = Dict(240=>DataFrame(opentime=rows.opentime,regry=[price],grad=[0f0]))
        context = TradingStrategy._replayregressions(cfg,nothing,cols,1,invalid)
        @test TradingStrategy._regressiondistance(price,2f0) === nothing
        @test TradingStrategy._regressionslope(price,0f0) === nothing
        TradingStrategy.gain_limit_reversal_below_regression!(cfg,context,cols,1)
        @test cols.label[1] == (label in labels[1:2] ? Targets.longhold : Targets.shorthold)
    end
    @test TradingStrategy._regressionslope(2f0,NaN32) === nothing
    @test TradingStrategy._regressionslope(2f0,Inf32) === nothing
    @test TradingStrategy._regressiondistance(1f0,2f0) == 1f0
    @test TradingStrategy._regressionslope(2f0,0f0) == 0f0
end
