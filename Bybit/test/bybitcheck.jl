using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(test)
bc = Bybit.BybitCache()

println("$(Bybit.get24h(bc, "BTCUSDT"))")
oid = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, btcprice * 0.9)
println("create order id = $(string(oid))")

oo = Bybit.order(bc, oid)
println(oo)

# println("balances: $(Bybit.balances(bc))")
