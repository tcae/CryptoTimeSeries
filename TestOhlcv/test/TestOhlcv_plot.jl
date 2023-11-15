module TestOhlcvTest

using Dates, DataFrames, Plots, TimeSeries
using Plots.PlotMeasures

using EnvConfig, Ohlcv, Features, TestOhlcv

function plotit(ohlcv::OhlcvData)
    df = Ohlcv.dataframe(ohlcv)[!, [:opentime, :open, :low, :high, :close]]
    df = DataFrames.rename(df, :opentime => :timestamp, :open => :Open, :high => :High, :low => :Low, :close => :Close)
    ta = TimeArray(df, timestamp = :timestamp)

    # # Create the candlestick plot
    p = plot(ta, seriestype = :candlestick, xticks = 10, xrotation = 60,
        title = "OHLC Candles",
        xlabel = "Timestamp",
        ylabel = "Price", bottom_margin=25mm)
end

ohlc1 = TestOhlcv.testohlcv(
    "sine", Dates.DateTime("2022-01-01T00:00", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2022-01-01T09:31", dateformat"yyyy-mm-ddTHH:MM"), "1m")
ohlc2 = TestOhlcv.testohlcv(
    "doublesine", Dates.DateTime("2022-01-01T00:15", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2022-01-02T09:31", dateformat"yyyy-mm-ddTHH:MM"), "1m")
# plotit(ohlc1)
p = plot(plotit(ohlc1), plotit(ohlc2), layout=(2,1), size=(2000, 1600))
display(p)

end # of TestOhlcvTest
