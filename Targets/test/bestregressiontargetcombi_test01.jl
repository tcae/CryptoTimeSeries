using Targets, Features, TestOhlcv, Ohlcv
using DataFrames, Dates
using Logging, LoggingExtras

all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
logger = EarlyFilteredLogger(all_logger) do args
    r = Logging.Debug <= args.level < Logging.AboveMaxLevel && args._module === Targets
    # r = Logging.Info <= args.level < Logging.AboveMaxLevel && args._module === Targets
    return r
end

with_logger(logger) do
    enddt = DateTime("2022-01-02T22:54:00")
    startdt = enddt - Dates.Day(20)
    ohlcv = TestOhlcv.testohlcv("DOUBLESINEUSDT", startdt, enddt)
    df = Ohlcv.dataframe(ohlcv)
    ol = size(df,1)
    f2 = Features.Features002(ohlcv)
    # println(f2)

    # layout = Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
    # yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)

    # p= [candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close),
    #     bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")]

    # display(plot(p, layout))

    pricepeaks = Targets.peaksbeforeregressiontargets(f2)
end