using Bybit, EnvConfig

EnvConfig.init(production)
# Bybit.getMarket() - not found
# Bybit.getMarket("BTCUSDT") - not found

# p24dictarray = Bybit.get24HR()  # without symbol -> symbold can be derived from AllPrices sequence
# for (index, p24dict) in enumerate(p24dictarray)
#     # println(index)
#     if p24dict["symbol"] == "BTCUSDT"
#         println(p24dict)
#     end
# end
# println("array entries: $(length(p24dictarray))")
# Bybit.get24HR("BTCUSDT")
# Bybit.ping()
# Bybit.serverTime()
# Bybit.getDepth("BTCUSDT"; limit=5) # 500(5), 1000(10)
Bybit.getExchangeInfo()
# Bybit.getKlines("BTCUSDT")
# Bybit.account(EnvConfig.authorization.key, EnvConfig.authorization.secret)
# Bybit.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
# order = Bybit.createOrder("BTCUSDT", )
# res = Bybit.openOrders(nothing, EnvConfig.authorization.key, EnvConfig.authorization.secret)
# res = Bybit.openOrders("CHZUSDT", EnvConfig.authorization.key, EnvConfig.authorization.secret)
# println(res)
# for o in res
#     for dictentry in o
#         println(dictentry)
#     end
#     println("=")
# end
