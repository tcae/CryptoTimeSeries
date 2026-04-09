using EnvConfig, Trade, Classify, CryptoXch, Bybit
using DataFrames

function loadtransactions(xc::CryptoXch.XchCache)
    tdf = Bybit.alltransactions(xc.bc)
    odf = Bybit.allorders(xc.bc)
    println("orders of size=$(size(odf)) describe: $(describe(odf, :all))")
    println("transactions of size=$(size(tdf)) describe: $(describe(tdf, :all))")
    EnvConfig.savedf(tdf, "BybitTransactions")
    EnvConfig.savedf(odf, "BybitOrders")
    return odf, tdf
end

function analyzeloaded()
    tdf = EnvConfig.readdf("BybitTransactions")
    odf = EnvConfig.readdf("BybitOrders")
    println("orders: $odf")
    println("transactions: $tdf")
    return odf, tdf
end

EnvConfig.init(production)
logpath = EnvConfig.logfolder()
EnvConfig.setlogpath("admin")
xc = CryptoXch.XchCache()
# odf, tdf = loadtransactions(xc)
odf, tdf = analyzeloaded()
println("finished")

