using Targets, Features, TestOhlcv, Ohlcv
using DataFrames, Dates

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(20)
ohlcv = TestOhlcv.testohlcv("doublesine", startdt, enddt)
df = Ohlcv.dataframe(ohlcv)
ol = size(df,1)
f2 = Features.Features002(ohlcv)
# println(f2)

# layout = Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
# yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)

# p= [candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close),
#     bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")]

# display(plot(p, layout))

combi, df = Targets.bestregressiontargetcombi(f2)
# println("s=$s length of first = $(s[1][2])")
println(df)

