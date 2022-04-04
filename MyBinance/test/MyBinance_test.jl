using MyBinance, EnvConfig

EnvConfig.init(production)
# MyBinance.getMarket() - not found
# MyBinance.getMarket("BTCUSDT") - not found

# MyBinance.getAllPrices()
# MyBinance.get24HR()  # without symbol -> symbold can be derived from AllPrices sequence
# MyBinance.get24HR("BTCUSDT")
# MyBinance.getAllBookTickers()
# MyBinance.ping()
# MyBinance.serverTime()
# MyBinance.getDepth("BTCUSDT"; limit=5) # 500(5), 1000(10)
# MyBinance.getExchangeInfo()
# MyBinance.getKlines("BTCUSDT")
# MyBinance.account(EnvConfig.authorization.key, EnvConfig.authorization.secret)
# MyBinance.balances(EnvConfig.authorization.key, EnvConfig.authorization.secret)
res = MyBinance.openOrders(nothing, EnvConfig.authorization.key, EnvConfig.authorization.secret)
# res = MyBinance.openOrders("CHZUSDT", EnvConfig.authorization.key, EnvConfig.authorization.secret)
# println(res)
for o in res
    for dictentry in o
        println(dictentry)
    end
    println("=")
end
