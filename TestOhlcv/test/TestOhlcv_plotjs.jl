# module TestOhlcvTest

using Dates, DataFrames, PlotlyJS

using EnvConfig, Ohlcv, TestOhlcv

function plotit(ohlcv::OhlcvData)
    df = Ohlcv.dataframe(ohlcv)[!, :]
    [candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close), bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")]
end

ohlc1 = TestOhlcv.testohlcv(
    "sine", Dates.DateTime("2022-01-01T00:00", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2022-01-01T09:31", dateformat"yyyy-mm-ddTHH:MM"), "1m")
ohlc2 = TestOhlcv.testohlcv(
    "doublesine", Dates.DateTime("2022-01-01T00:15", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2022-01-02T09:31", dateformat"yyyy-mm-ddTHH:MM"), "1m")
layout = Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)
display(plot(plotit(ohlc1), layout))
# display([plot(plotit(ohlc1), layout) plot(plotit(ohlc2), layout)])
# display(p)

# end # of TestOhlcvTest
