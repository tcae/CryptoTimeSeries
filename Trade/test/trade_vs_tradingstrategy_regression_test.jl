using Test
using Dates
using DataFrames
using EnvConfig, Trade, TradingStrategy, Classify, CryptoXch, Ohlcv, Targets

function _normalize_tradepairs(tradedf::AbstractDataFrame)
    return [
        (
            trend=row.trend,
            startix=row.startix,
            endix=row.endix,
            startdt=row.startdt,
            enddt=row.enddt,
            gain=round(Float32(row.gain), digits=7),
            gainfee=round(Float32(row.gainfee), digits=7),
        )
        for row in eachrow(tradedf)
    ]
end

@testset "Trade vs TradingStrategy regression harness" begin
    EnvConfig.init(EnvConfig.test)

    base = "BTC"
    startdt = DateTime("2025-05-17T00:00:00")
    opentimes = [startdt + Minute(ix - 1) for ix in 1:5]
    closes = Float32[100, 100, 100, 100, 100]
    rawlabels = TradeLabel[allclose, longbuy, longclose, allclose, allclose]
    scores = Float32[0.1, 0.9, 0.9, 0.1, 0.1]

    predictionsdf = DataFrame(
        opentime=opentimes,
        close=closes,
    )

    bulk_gs = TradingStrategy.GainSegment(
        maxwindow=length(opentimes),
        openthreshold=0.6f0,
        closethreshold=0.5f0,
        algorithm=TradingStrategy.gain_reversal!,
        makerfee=0f0,
        takerfee=0f0,
        limitreduction=0f0,
    )
    bulk_gs.buygain = 0f0
    bulk_gs.sellgain = 0f0
    bulk_pairs = TradingStrategy.getgains(
        bulk_gs,
        predictionsdf,
        scores,
        rawlabels,
        false;
        lastix=length(opentimes),
        openthreshold=0.6f0,
        closethreshold=0.5f0,
    )

    ohlcv = Ohlcv.OhlcvData(
        DataFrame(
            opentime=opentimes,
            open=closes,
            high=closes,
            low=closes,
            close=closes,
            basevolume=fill(1f0, length(opentimes)),
            pivot=closes,
        ),
        uppercase(base),
        uppercase(EnvConfig.cryptoquote),
        "1m",
        1,
        nothing,
    )
    xc = CryptoXch.XchCache(startdt=startdt, enddt=last(opentimes))
    CryptoXch.addbase!(xc, ohlcv)
    tc = Trade.TradeCache(xc=xc, cl=Classify.Classifier011(), trademode=Trade.notrade)

    gs = TradingStrategy.GainSegment(
        maxwindow=length(opentimes),
        openthreshold=0.6f0,
        closethreshold=0.5f0,
        algorithm=TradingStrategy.gain_reversal!,
        makerfee=0f0,
        takerfee=0f0,
        limitreduction=0f0,
    )
    gs.buygain = 0f0
    gs.sellgain = 0f0
    Trade.apply_tradingstrategy!(tc, gs; strategy_engine=:getgainsalgo, source="test")

    history = Trade._strategyhistory!(tc, base)
    minute_gs = Trade._strategystate!(tc, base)
    for (ix, dt) in enumerate(opentimes)
        CryptoXch.setcurrenttime!(xc, dt)
        Trade._upsert_getgainsalgo_sample!(history, ohlcv, rawlabels[ix], scores[ix])
    end

    minute_pairs = TradingStrategy.getgains(
        minute_gs,
        history.predictionsdf,
        history.scores,
        history.labels,
        true;
        lastix=length(history.scores),
        openthreshold=minute_gs.openthreshold,
        closethreshold=minute_gs.closethreshold,
    )

    @test nrow(bulk_pairs) == 1
    @test nrow(minute_pairs) == 1
    @test _normalize_tradepairs(minute_pairs) == _normalize_tradepairs(bulk_pairs)
end