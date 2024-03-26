using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(test)
bc = Bybit.BybitCache()

now = Bybit.get24h(bc, "BTCUSDT")
println("$now")
oid = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, now.lastprice)
println("create order id = $(string(oid))")

oo = Bybit.order(bc, oid)
println(oo)

# println("balances: $(Bybit.balances(bc))")
