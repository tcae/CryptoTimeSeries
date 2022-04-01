using Pkg
cd("/home/tor/TorProjects/CryptoTimeSeries")
Pkg.activate(".")
# cd("/home/tor/TorProjects/CryptoTimeSeries/notebooks")

# include(pwd() * "/" * "../src/features.jl")

using PlotlyJS, WebIO, Dates, DataFrames
using Features, TestOhlcv

function test1()
    normpercent(ydata, ynormref) = (ydata ./ ynormref .- 1) .* 100

    iny = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
    normref = iny[end]
    regrwindow = size(iny, 1)

    innormy = normpercent(iny, iny[end])
    regry, grad = Features.rollingregression(iny, regrwindow)
    println("regression y: $regry, size(y,1): $(size(regry,1)), y[end]: $(regry[end])")
    println("grad: $grad")
    intercept = regry[end] - size(regry,1) * grad[end]
    println("intercept: $intercept")

    regrliney = [regry[end] - grad[end] * (regrwindow - 1), regry[end]]
    regrlinenormy = normpercent(regrliney, normref)
    # lastregry = normpercent([regry[end]], normref)[1]
    # lastgrad = normpercent([grad[end]], normref)[1]
    # firstregry = lastregry - lastgrad * (regrwindow - 1)


    traces = [
        scatter(;x=1:size(iny, 1), y=iny, mode="markers", name="input y")
        # scatter(;x=1:size(regry, 1), y=regry, mode="lines+markers", name="regression y")
        scatter(;x=1:size(innormy, 1), y=innormy, mode="markers", name="norm input y")
        # scatter(;x=[1,size(regry, 1)], y=[firstregry, lastregry], mode="lines+markers", name="regression y")
        scatter(;x=[1,size(regry, 1)], y=regrliney, mode="lines+markers", name="regression y")
        scatter(;x=[1,size(regry, 1)], y=regrlinenormy, mode="lines+markers", name="norm regression y")

    ]

    std, mean, normy = Features.rollingregressionstdxt(iny, regry, grad, regrwindow)
    traceadd = [scatter(;x=1:x1, y=normy[x1, 1:x1], mode="lines+markers", name="notrend y") for x1 in 1:regrwindow]
    append!(traces, traceadd)
    traceadd = [
        scatter(;x=1:regrwindow, y=std[1:regrwindow], mode="lines+markers", name="std notrend y")
        scatter(;x=1:regrwindow, y=mean[1:regrwindow], mode="lines+markers", name="mean notrend y")
    ]
    append!(traces, traceadd)
    plot(traces)
end

function test2()
    x = 0:0.01:2*pi
    y = [sin(ix) for ix in x]
    # println(y)
    traces = [
        scatter(y=y, x=x, mode="lines", name="sin")
    ]
    plot(traces)
end

function test3()
    x = 0:400
    y = [sin(ix/400*2*pi) for ix in x]
    # println(y)
    traces = [
        scatter(y=y, x=x, mode="lines", name="sin")
    ]
    plot(traces)
end

function test3b()
    x, y = TestOhlcv.sinesamples(400, 2, [(150, 0, 0.5)])
    println(y)
    traces = [
        scatter(y=y, x=x, mode="lines", name="sin")
    ]
    plot(traces)
end

function test4()

    x = DateTime(2022, 01, 17, 00, 00):Minute(1):DateTime(2022, 01, 23, 00, 00)
    y = TestOhlcv.triplesine(15, 2*60, 24*60, 1.5, 0, 10, size(x,1))
    # println(y)
    traces = [
        scatter(y=y, x=x, mode="lines", name="sin")
    ]
    plot(traces, Layout(xaxis_rangeslider_visible=true))
end

function test5()

    x = DateTime(2022, 01, 22, 12, 00):Minute(1):DateTime(2022, 01, 23, 00, 00)
    (o, h, l, c, v) = TestOhlcv.triplesineohlcv(15, 2*60, 24*60, 1.5, 0, 10, size(x,1))
    df = DataFrame(date=x, open=o, high=h, low=l, close=c, volume=v)
    # println(df)
    traces = [
        candlestick(df, x=:date, open=:open, high=:high, low=:low, close=:close),
        bar(df, x=:date, y=:volume, name="basevolume", yaxis="y2")
        # candlestick(
        #     x=subdf[!, :opentime],
        #     open=normpercent(subdf[!, :open], normref),
        #     high=normpercent(subdf[!, :high], normref),
        #     low=normpercent(subdf[!, :low], normref),
        #     close=normpercent(subdf[!, :close], normref),
        #     name="$base OHLC")
    ]
    plot(traces,
        Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
            yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true))
end

function test6()

    x = DateTime(2022, 01, 12, 12, 00):Minute(1):DateTime(2022, 01, 23, 00, 00)
    df = TestOhlcv.sinedata(2*60, size(x,1))
    df.opentime = x
    # println(df)
    traces = [
        candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close),
        bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")
        # candlestick(
        #     x=subdf[!, :opentime],
        #     open=normpercent(subdf[!, :open], normref),
        #     high=normpercent(subdf[!, :high], normref),
        #     low=normpercent(subdf[!, :low], normref),
        #     close=normpercent(subdf[!, :close], normref),
        #     name="$base OHLC")
    ]
    plot(traces,
        Layout(yaxis2=attr(title="vol", side="right"), yaxis2_domain=[0.0, 0.25],
            yaxis_domain=[0.3, 1.0], xaxis_rangeslider_visible=true))
end

function test7()

    x = DateTime(2022, 01, 12, 12, 00):Minute(1):DateTime(2022, 01, 13, 00, 00)
    df = TestOhlcv.sinedata(2*60, size(x,1))
    z = nothing
    df.opentime = x
    println(df)
    lines = [wk for wk in names(df) if !(wk in ["opentime", "basevolume"])]
    for wk in lines
        z = z === nothing ? df[:, wk] : hcat(z, df[:, wk])
    end
    z = transpose(z)

    println("$lines x: $(size(x)) y: $(size(lines)) z: $(size(z))")
    traces = [
        heatmap(x=df[!, :opentime], y=lines, z=z)
    ]
    plot(traces,
        Layout(yaxis=attr(title="vol", side="right"), xaxis_rangeslider_visible=true))
end

function test8()

    x = DateTime(2022, 01, 12, 12, 00):Minute(1):DateTime(2022, 01, 13, 00, 00)
    df = TestOhlcv.sinedata(2*60, size(x,1))
    df.opentime = x
    # println(df)

    lines = [wk for wk in names(df) if !(wk in ["opentime", "basevolume"])]
    # z = nothing
    # for wk in lines
    #     z = z === nothing ? df[:, wk] : hcat(z, df[:, wk])
    # end
    # z = transpose(z)
    # z = [df[r, wk] for r in 1:size(df, 1) for wk in lines]
    z = zeros(Float32, (size(lines,1), size(df,1)))

    for (ix, wk) in enumerate(lines)
        for r in 1:size(df,1)
            z[ix, r] = df[r, wk]
        end
    end

    println("$lines x: $(size(x)) y: $(size(lines)) z: $(size(z))")

    traces = [
        heatmap(x=df[!, :opentime], y=lines, z=z, yaxis="y4"),
        scatter(x=df[[begin, end], :opentime], y=df[[begin, end], :open], mode="lines", showlegend=false),
        candlestick(df, x=:opentime, open=:open, high=:high, low=:low, close=:close),
        bar(df, x=:opentime, y=:basevolume, name="basevolume", yaxis="y2")
        # candlestick(
        #     x=subdf[!, :opentime],
        #     open=normpercent(subdf[!, :open], normref),
        #     high=normpercent(subdf[!, :high], normref),
        #     low=normpercent(subdf[!, :low], normref),
        #     close=normpercent(subdf[!, :close], normref),
        #     name="$base OHLC")
    ]
    plot(traces,
        Layout(
            xaxis_rangeslider_visible=false,
            yaxis=attr(title_text="% of last pivot", domain=[0.25, 0.75]),
            yaxis2=attr(title="vol", side="right", domain=[0.0, 0.2]),
            yaxis4=attr(visible =true, side="left", domain=[0.8, 1.0]),
            yaxis3=attr(overlaying="y", visible =false, side="right", color="black", range=[0, 1], autorange=false)))
end

test3b()
