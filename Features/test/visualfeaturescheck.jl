# using Plots
using PlotlyJS, Dates, DataFrames
using Ohlcv, TestOhlcv, Features

enddt = DateTime("2022-01-02T22:54:00")
startdt = enddt - Dates.Day(2)
ohlcv = TestOhlcv.testohlcv("sine", startdt, enddt)
df = Ohlcv.dataframe(ohlcv)
ol = size(df,1)
f2 = Features.Features002(ohlcv)
println(f2)
f12x = Features.features12x1m01(f2)
println(first(f12x,5))
println(describe(f12x))
println(describe(df))
println(all(describe(f12x).eltype .== Float32))

# plotlyjs()
# gr()

x = 1:10
y = rand(10)
layout = Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true)

# p = scatter(;x=x, y=y)
p= [candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close),
    bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")]

# ylabel!("Y-axis")
# title!("My Plot")
display(plot(p, layout))
