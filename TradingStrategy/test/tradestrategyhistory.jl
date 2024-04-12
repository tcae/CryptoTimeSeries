module TradingStrategyHistory

using Test, Dates, Logging, LoggingExtras, DataFrames, JDF
using EnvConfig, TradingStrategy, Classify, Features, Ohlcv, CryptoXch


println("$(EnvConfig.now()): started")

CryptoXch.verbosity = 3
Ohlcv.verbosity = 1
Features.verbosity = 1
# EnvConfig.init(training)
EnvConfig.init(production)
xc = CryptoXch.XchCache(true)


# tc = TradingStrategy.train!(TradingStrategy.TradeConfig(xc), []; datetime=dt, assetonly=true)
function ohlcv2df()
    df = DataFrame()
    stopafter = 3000
    for ohlcv in Ohlcv.OhlcvFiles()
        if size(ohlcv.df, 1) > 0
            if ohlcv.interval != "1m"
                @warn "unexpected ohlcv interval=$(ohlcv.interval) - skipping"
                continue
            end
            push!(df, (base=ohlcv.base, startdt=ohlcv.df[begin, :opentime], enddt=ohlcv.df[end, :opentime], ohlcv=ohlcv))
            stopafter -= 1
            if (stopafter <= 0)
                break
            end
        else
            @warn "empty ohlcv for $(ohlcv.base)"
        end
    end
    return df
end

function USDTmarketdf(ohlcvdf)
    mindt = minimum(ohlcvdf[!, :startdt])
    enddt = maxdt = floor(maximum(ohlcvdf[!, :enddt]), Day(1))
    startdt = enddt - Day(1)
    while mindt <= enddt <= maxdt
        mdf = DataFrame()
        for row in eachrow(ohlcvdf)
            startix = Ohlcv.rowix(row.ohlcv, startdt)
            endix = Ohlcv.rowix(row.ohlcv, enddt)
            period = row.ohlcv.df[endix, :opentime] - row.ohlcv.df[startix, :opentime]
            if Hour(20) <= period <= Day(1)
                vdf = @view row.ohlcv.df[startix:endix, :]
                quotevolume24h = sum(Ohlcv.pivot!(vdf) .* vdf[!, :basevolume])
                pricechangepercent = (vdf[end, :close] - vdf[begin, :open]) / vdf[begin, :open] * 100
                lastprice = vdf[end, :close]
                askprice = lastprice * (1 + 0.0001)
                bidprice = lastprice * (1 - 0.0001)
                push!(mdf, (basecoin=row.ohlcv.base, quotevolume24h=quotevolume24h, pricechangepercent=pricechangepercent, lastprice=lastprice, askprice=askprice, bidprice=bidprice))
            # else
            #     println("not 1 day: period=$period, start=$(row.ohlcv.df[startix, :opentime]), end=$(row.ohlcv.df[endix, :opentime]), ohlcv=$(row.ohlcv)")
            end
        end
        println("writing USDTmarket file of size=$(size(mdf)) at enddt=$enddt")
        # println(describe(mdf, :all))
        JDF.savejdf(CryptoXch._usdtmarketfilename(CryptoXch.USDTMARKETFILE, enddt), mdf)
        enddt = startdt
        startdt -= Day(1)
    end
end

xc.currentdt = xc.enddt = DateTime("2024-03-30T10:03:00")  # set time simulation on
# ohlcvdf = ohlcv2df()
# println(describe(ohlcvdf, :all))
# usdtmarketdf = USDTmarketdf(ohlcvdf)

xc.currentdt = xc.enddt = nothing  # set time simulation off
CryptoXch.getUSDTmarket(xc)

xc.currentdt = xc.enddt = DateTime("2024-03-30T10:03:00")  # set time simulation on
df = CryptoXch.getUSDTmarket(xc)
println("check loading of saved USDT amrket data: $(describe(df, :all))")
println("done")


end  # module