# module TestOhlcvTest

using Dates, DataFrames, PlotlyJS

using EnvConfig, Ohlcv, TestOhlcv

function plotit(ohlcv::OhlcvData)
    df = Ohlcv.dataframe(ohlcv)[!, :]
    @assert size(df, 1) > 0 "No OHLCV rows to plot for $(ohlcv.base). Check requested date range against generated test data timestamps."
    [candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close), bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")]
end

ohlc1 = TestOhlcv.testohlcv(
    "SINE", Dates.DateTime("2025-08-01T00:01:00", dateformat"yyyy-mm-ddTHH:MM:SS"),
    Dates.DateTime("2025-08-01T09:31:00", dateformat"yyyy-mm-ddTHH:MM:SS"), "1m")
ohlc2 = TestOhlcv.testohlcv(
    "DOUBLESINE", Dates.DateTime("2025-08-01T01:15:00", dateformat"yyyy-mm-ddTHH:MM:SS"),
    Dates.DateTime("2025-08-02T09:31:00", dateformat"yyyy-mm-ddTHH:MM:SS"), "1m")
layout = Layout(title=ohlc1.base, yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)
display(plot(plotit(ohlc1), layout))
layout = Layout(title=ohlc2.base, yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)
display(plot(plotit(ohlc2), layout))
# display([plot(plotit(ohlc1), layout) plot(plotit(ohlc2), layout)])
# display(p)

# end # of TestOhlcvTest
