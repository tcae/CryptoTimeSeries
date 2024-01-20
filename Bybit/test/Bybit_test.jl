using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)

println(Bybit.servertime()) # > DateTime("2023-08-18T20:07:54.209")

acc = Bybit.account()
println(acc["marginMode"] == "REGULAR_MARGIN")
println(isa(acc, AbstractDict))
println(length(acc) > 1)
println("account()=$acc")

syminfo = Bybit.init()
println(isa(syminfo, AbstractDataFrame))
println(size(syminfo, 1) > 100)
println("syminfo = Bybit.init()")

syminfo = Bybit.symbolinfo("BTCUSDT")
println("syminfo= $syminfo = Bybit.symbolinfo(BTCUSDT)")

dayresult = Bybit.get24h()
println(isa(dayresult, AbstractDataFrame))
println(size(dayresult, 1) > 100)
println("dayresult = Bybit.get24h()")

dayresult = Bybit.get24h("BTCUSDT")
println(isa(dayresult, AbstractDataFrame))
println(size(dayresult, 2) >= 6)
println(size(dayresult, 1) == 1)
println("dayresult = Bybit.get24h(BTCUSDT)")

klines = Bybit.getklines("BTCUSDT")
println(isa(klines, AbstractDataFrame))
println("klines = Bybit.getklines(BTCUSDT)")
# println(klines)


oo = Bybit.openorders()
println(oo)

wb = Bybit.balances()
println(describe(wb, :min))

oid = Bybit.createorder("BTCUSDT", "Buy", 0.00001, 39899)
println("create order id = $(string(oid))")

wb = Bybit.balances()
println(wb)

oo = Bybit.order(oid)
println(isa(oo, AbstractDataFrame))
println(size(oo, 1) == 1)
# println(describe(oo, :min, :eltype))

oo = Bybit.openorders()
println(isa(oo, AbstractDataFrame))
println(oo)
# println(describe(oo))

coid = Bybit.cancelorder("BTCUSDT", oid)
println("cancelled order id = $(string(oid))")

oo = Bybit.order(oid)
println(size(oo, 1) == 0)
println((size(oo, 1) == 1) && (oo[1, "orderId"] == oid) && (oo[1, "orderStatus"] == "Cancelled"))
println(oo)

wb = Bybit.balances()
println(isa(wb, AbstractDataFrame))
println(size(wb, 2) >= 18)
println(wb)


;
