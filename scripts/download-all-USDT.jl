using CryptoXch, Dates

println("download all USDT crypto but those on CryptoXch.baseignore ist")
CryptoXch.downloadallUSDT(Dates.now(Dates.UTC), Dates.Year(4))

