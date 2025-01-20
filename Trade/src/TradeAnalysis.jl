using EnvConfig, Trade, Classify, CryptoXch, Bybit
using DataFrames, JDF

function loadtransactions(xc::CryptoXch.XchCache)
    tdf = Bybit.alltransactions(xc.bc)
    odf = Bybit.allorders(xc.bc)
    println("orders of size=$(size(odf)) describe: $(describe(odf, :all))")
    println("transactions of size=$(size(tdf)) describe: $(describe(tdf, :all))")
    JDF.savejdf(EnvConfig.logpath("BybitTransactions.jdf"), tdf)
    JDF.savejdf(EnvConfig.logpath("BybitOrders.jdf"), odf)
    return odf, tdf
end

function analyzeloaded()
    tdf = DataFrame(JDF.loadjdf(EnvConfig.logpath("BybitTransactions.jdf")))
    odf = DataFrame(JDF.loadjdf(EnvConfig.logpath("BybitOrders.jdf")))
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

