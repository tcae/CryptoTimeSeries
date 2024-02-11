using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(test)

println(Bybit.servertime()) # > DateTime("2023-08-18T20:07:54.209")

acc = Bybit.account()
println(acc["marginMode"] == "REGULAR_MARGIN")
println(isa(acc, AbstractDict))
println(length(acc) > 1)
println("account()=$acc")

syminfo = Bybit.exchangeinfo()
println(isa(syminfo, AbstractDataFrame))
println(size(syminfo, 1) > 100)
println("syminfo = Bybit.exchangeinfo()")

syminfo = Bybit.symbolinfo("BTCUSDT")
println("syminfo= $syminfo = Bybit.symbolinfo(BTCUSDT)")

dayresult = Bybit.get24h()
println(isa(dayresult, AbstractDataFrame))
println(size(dayresult, 1) > 100)
println("dayresult = Bybit.get24h()")
println(dayresult[dayresult.symbol .== "BTCUSDT", :])

btcprice = dayresult[dayresult.symbol .== "BTCUSDT", :lastprice][1,1]
println("btcprice=$btcprice  (+1%=$(btcprice * 1.01))")

dayresult = Bybit.get24h("BTCUSDT")
println(isa(dayresult, AbstractDataFrame))
println(size(dayresult, 2) >= 6)
println(size(dayresult, 1) == 1)
println("dayresult = Bybit.get24h(BTCUSDT)")

klines = Bybit.getklines("BTCUSDT")
println(isa(klines, AbstractDataFrame))
println("klines = Bybit.getklines(BTCUSDT) size(klines)=$(size(klines))")
# println(klines)


oo = Bybit.openorders()
println(oo)

oid = Bybit.createorder("BTCUSDT", "Buy", 0.00001, btcprice * 0.9)
println("create order id = $(string(oid))")

oo = Bybit.order(oid)
println(isa(oo, NamedTuple))
println(length(oo) == 12)
# println(describe(oo, :min, :eltype))
println(oo)

oidc = Bybit.amendorder("BTCUSDT", oid; quantity=0.00011)
println("amendorder quantity: oidc=$oidc content: $(Bybit.order(oidc))")

oidc = Bybit.amendorder("BTCUSDT", oid; limitprice=btcprice * 0.8)
println("amendorder limit price: oidc=$oidc content: $(Bybit.order(oidc))")

oo = Bybit.openorders()
println(isa(oo, AbstractDataFrame))
println(oo)
# println(describe(oo))

coid = Bybit.cancelorder("BTCUSDT", oid)
println("cancelled order id = $(string(oid)) returned=$(string(coid))")

oo = Bybit.order(oid)
println(oo)
println((oo.orderid == oid) && (oo.status == "Cancelled"))

wb = Bybit.balances()
println(isa(wb, AbstractDataFrame))
println(size(wb, 2) == 4)
println(wb)


;
