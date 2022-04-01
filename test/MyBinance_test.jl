using MyBinance

# MyBinance.getMarket() - not found
# MyBinance.getMarket("BTCUSDT") - not found

MyBinance.getAllPrices()
# MyBinance.get24HR()  # without symbol -> symbold can be derived from AllPrices sequence
# MyBinance.get24HR("BTCUSDT")
# MyBinance.getAllBookTickers()
# MyBinance.ping()
# MyBinance.serverTime()
# MyBinance.getDepth("BTCUSDT"; limit=5) # 500(5), 1000(10)
# MyBinance.getExchangeInfo()
# MyBinance.getKlines("BTCUSDT")
