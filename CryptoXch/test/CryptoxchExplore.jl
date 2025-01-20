
module CryptoXchTest
using Dates, DataFrames

using Ohlcv, EnvConfig, CryptoXch, Bybit

function balances_test()
    result = CryptoXch.balances(xc)
    display(result)
    display(EnvConfig.bases)
    display(EnvConfig.trainingbases)
    display(EnvConfig.datapath)
end

# EnvConfig.init(test)
EnvConfig.init(production)
xc = CryptoXch.XchCache()
# balances_test()

userdataChannel = Channel(10)
startdt = DateTime("2020-08-11T22:45:00")
enddt = DateTime("2020-09-11T22:49:00")
# res = Bybit.getklines("BTCUSDT"; startDateTime=startdt, endDateTime=enddt, interval="1m")
# display(res)
# display(last(res[:body], 3))
# display(first(res[:body], 3))
# display(res[:body][1:3, :])
# display(res[:body][end-3:end, :])

# Binance.wsKlineStreams(cb, ["BTCUSDT", "XRPUSDT"])


function initialbtcdownload(xc)
    startdt = DateTime("2022-01-02T22:45:03")
    enddt = DateTime("2022-01-02T22:49:35")
    ohlcv = CryptoXch.cryptodownload(xc, "btc", "1m", startdt, enddt)
    return ohlcv
end



function testorder(xc, price, basevol)
    oo = nothing
    try
        oo = CryptoXch.createbuyorder("btc", limitprice=price, basequantity=basevol, maker=false)
    catch err
        @error err
    end
    !isnothing(oo) && @info oo
end


# testorder(xc, 19001.0, 20.0/19001.0)
# testorder(xc, 19001.0001, 5.0/19001.0)
# testorder(xc, 19001.00000000002, 20.00008/19001.0)
allbb = Bybit.get24h(xc.bc)
println(describe(allbb, :all))
# println(allbb)

allusdt = CryptoXch.getUSDTmarket(xc)
println(describe(allusdt, :all))
# println(allusdt)

bdf = CryptoXch.balances(xc)
println("balances():")
println(bdf)
bdf = CryptoXch.portfolio!(xc, bdf, allusdt)
println("portfolio():")
println(bdf)

bdf = CryptoXch.portfolio!(xc)
println("portfolio():")
println(bdf)

end  # module
