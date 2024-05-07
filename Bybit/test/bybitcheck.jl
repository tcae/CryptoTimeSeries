using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(test)
bc = Bybit.BybitCache()

now = Bybit.get24h(bc, "BTCUSDT")
println("$now")
oocreate = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, now.lastprice)
println("create order = $(isnothing(oocreate) ? string(oocreate) : oocreate)")

oo = Bybit.order(bc, oocreate.orderid)
println(oo)

# println("balances: $(Bybit.balances(bc))")
