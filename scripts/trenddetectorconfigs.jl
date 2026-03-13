testcoins() = ["SINE", "DOUBLESINE"]
traincoins() = ["1INCH", "AAVE", "ACH", "ADA", "AI16Z", "ALGO", "ANKR", "APEX", "APE", "APT", "ARB", "AR", "ATOM", "AVAX", "AXS", "BCH", "BNB", "BONK", "BRETT", "BTC", "C98", "CAKE", "CARV", "CELO", "CHILLGUY", "CHZ", "COMP", "CRV", "CTC", "DEEP", "DEGEN", "DGB", "DOGE", "DOGS", "DOT", "DRIFT", "DYDX", "EGLD", "ELX", "ENA", "ENJ", "ENS", "EOS", "ETC", "ETH", "FET", "FIL", "FIRE", "FLOCK", "FLOKI", "FLOW", "FTM", "FTT", "FXS", "GALA", "GAL", "GLMR", "GMT", "GOAT", "GPS", "GRASS", "GRT", "HBAR", "HFT", "HNT", "HOOK", "HOT", "H", "ICP", "ID", "IMX", "INIT", "INJ", "IP", "JASMY", "JUP", "KAS", "KAVA", "KLAY", "KSM", "LDO", "LINK", "LRC", "LTC", "LUNA", "LUNC", "MAGIC", "MANA", "MASK", "MATIC", "MAVIA", "MBOX", "MERL", "MINA", "MKR", "MNT", "MOVE", "MYRO", "NAKA", "NEAR", "NOT", "NXPC", "OMG", "ONDO", "ONE", "OP", "PENGU", "PEOPLE", "PEPE", "PLANET", "PLUME", "POL", "POPCAT", "PUFFER", "PUMP", "PYTH", "QNT", "QTUM", "RDNT", "RENDER", "RNDR", "ROSE", "RUNE", "RVN", "SAND", "SC", "SEI", "SERAPH", "SHIB", "SNX", "SOL", "SPX", "STETH", "STG", "STRK", "STX", "SUI", "SUNDOG", "SUSHI", "TAI", "THETA", "TIA", "TON", "TRUMP", "TRX", "TWT", "UNI", "UXLINK", "VIRTUAL", "WAVES", "WAXP", "WIF", "WLD", "XLM", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEN", "ZIL", "ZRX"]
# traincoins() = ["ETH", "BTC", "ADA", "SOL", "XRP"]

partitionconfig01() =(samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
partitionconfig02() =(samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=8*60, minpartitionsize=12*60, maxpartitionsize=2*24*60)

resultsfilename(coin=nothing) = isnothing(coin) ? "results.jdf" : "results_$coin.jdf" # includes hlcp, sets, ranges, targets
featuresfilename(coin=nothing) = isnothing(coin) ? "features.jdf" : "features_$coin.jdf"
predictionsfilename() = "maxpredictions.jdf"
confusionfilename() = "confusion.jdf"
xconfusionfilename() = "xconfusion.jdf"
distancesfilename() = "distances.jdf"
gainsfilename() = "gains.jdf"

tradingstrategy01() = TradingStrategy.GainSegment(maxwindow=4*60, algorithm=TradingStrategy.algorithm01!, openthreshold=0.6, closethreshold=0.5, makerfee=0.0015, takerfee=0.002)
tradingstrategy02() = TradingStrategy.GainSegment(maxwindow=4*60, algorithm=TradingStrategy.algorithm02!, openthreshold=0.6, closethreshold=0.5, makerfee=0.0015, takerfee=0.002)

trend01targetconfig(minwindow, maxwindow, buy, hold) = Targets.Trend01(minwindow, maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))
targetconfig01() = trend01targetconfig(10, 4*60, 0.01, 0.01)
targetconfig02() = trend01targetconfig(10, 4*60, 0.05, 0.03)
targetconfig03() = trend01targetconfig(10, 4*60, 0.02, 0.01)
targetconfig04() = trend01targetconfig(10, 4*60, 0.007, 0.005)
targetconfig05() = trend01targetconfig(0, 4*60, 0.01, 0.01)
targetconfig06() = trend01targetconfig(2, 4*60, 0.01, 0.01)
targetconfig07() = trend01targetconfig(10, 4*60, 0.01, 0.005)
targetconfig08() = trend01targetconfig(10, 2*60, 0.01, 0.01)
targetconfig09() = trend01targetconfig(10, 1*60, 0.01, 0.01)

trend02targetconfig(maxwindow, buy, hold) = Targets.Trend02(maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))
targetconfig08() = trend02targetconfig(4*60, 0.01, 0.005)

settypes() = ["train", "test", "eval"]

" start 3* 5min with short segments and also add a 5min std element for high volatile situations then become larger "
function f6config01()
    featcfg = Features.Features006()
    Features.addstd!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=5, offset=5)
    Features.addgrad!(featcfg, window=5, offset=10)
    Features.addgrad!(featcfg, window=15, offset=15)
    Features.addgrad!(featcfg, window=15, offset=30)
    Features.addgrad!(featcfg, window=15, offset=45)
    Features.addgrad!(featcfg, window=60, offset=60)
    Features.addgrad!(featcfg, window=60*4, offset=120)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"like f6config01 but without the 5min std element and only with 1* 5min regr"
function f6config03()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=5)
    Features.addgrad!(featcfg, window=60, offset=20)
    Features.addgrad!(featcfg, window=60*4, offset=80)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"like f6config01 but without the 5min std and regr elements - just starting with 15min"
function f6config04()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=15)
    Features.addgrad!(featcfg, window=60*4, offset=75)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"like f6config03 but the larger regr grad starting without offset"
function f6config05()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"like f6config05 but with an added 12h regr grad"
function f6config06()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=60*12, offset=0)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"like f6config06 but with an added 3*24h regr grad"
function f6config07()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=60*12, offset=0)
    Features.addgrad!(featcfg, window=3*60*24, offset=0)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60*5, offset=60)
    Features.addmindist!(featcfg, window=60*5, offset=60)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"""
mk1 = mix adapted with multiple adaptations per coin with **good results**: ppv(longbuy) = 72%
"""
mk1config() = (configname="001", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk2 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk2config() = (configname="002", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk3 = one iteration adaptation with one merged set (allclose=>0.9, longbuy=>0.1), but target config targetconfig02() with **poor results**: ppv(longbuy) = 26%
"""
mk3config() = (configname="003", featconfig = f6config01(), targetconfig = targetconfig02(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk4 = targetconfig03() with one merged set (allclose=>0.726, longbuy=>0.274)
"""
mk4config() = (configname="004", featconfig = f6config01(), targetconfig = targetconfig03(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk5 = targetconfig04() with one merged set (allclose=>?, longbuy=>?)
"""
mk5config() = (configname="005", featconfig = f6config01(), targetconfig = targetconfig04(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk6 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk6config() = (configname="006", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
same as mk6 butwith copied mix classifier from mk2
mk7 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk7config() = (configname="007", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk8 = mix adapted with all coin features/targets in one set, features are clipped, normalized, shifted, and in addition batch norm layer after initial layer with relu activation in model001
equal mean, q25, q75, min, max does not look like healthy feature values - longbuy ppv classification performance is with close to 70% also worse
"""
# mk8config() = (configname="008", featconfig = f6config02(), targetconfig = targetconfig01(), classifiermodel=Classify.model001)

""" **my favorite**  
mk9 = mix adapted with all coin features/targets in one set, features are not clipped, batch norm layer before and between layers with relu activation in model002
"""
mk9config() = (configname="009", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""
mk10 = mix adapted with all coin features/targets in one set, features are clipped, initial batch norm layer in model002
"""
# mk10config() = (configname="010", featconfig = f6config02(), targetconfig = targetconfig01(), classifiermodel=Classify.model002)

"""
mk11 = mix adapted with all coin features/targets in one set, no clipping, initial batch norm layer but no further internal batch norm layers
"""
mk11config() = (configname="011", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model003, tradingstrategy=tradingstrategy02())

"""
mk12 = like mk9 but with an additional layer
"""
mk12config() = (configname="012", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model004, tradingstrategy=tradingstrategy02())

"""
mk13 = like mk9 but with 4/3 broader layers
"""
mk13config() = (configname="013", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model005, tradingstrategy=tradingstrategy02())

"""
mk14 = like mk11 but removed layer 3
"""
mk14config() = (configname="014", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model006, tradingstrategy=tradingstrategy02())

"""
mk15 = like mk11 but reduced number of nodes of layers by reducing factor of layer 1 from 3 to 2
"""
mk15config() = (configname="015", featconfig = f6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model007, tradingstrategy=tradingstrategy02())

"""
mk16 = no tolerance against target disturbances (minwindow=0), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk16config() = (configname="016", featconfig = f6config01(), targetconfig = targetconfig05(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""
mk17 = short tolerance against target disturbances (minwindow=2), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk17config() = (configname="017", featconfig = f6config01(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk18 = mk9 but with simplified features
"""
mk18config() = (configname="018", featconfig = f6config03(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk19 = mk9 but with simplified features
"""
mk19config() = (configname="019", featconfig = f6config04(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk20 = mk17 but feature grads without offset
"""
mk20config() = (configname="020", featconfig = f6config05(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk21 = mk17 but feature grads without offset and 12h regr grad added
"""
mk21config() = (configname="021", featconfig = f6config06(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk22 = mk21 but with 3*24h regr grad added
"""
mk22config() = (configname="022", featconfig = f6config07(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())
#* from mk23 onwards partitionconfig02() shall be used with a fixed partition gap of 8h to get the same targets gains for the same targetconfig but different requiredminutes(features)

"""  
mk023 = equal to mk9 with the only difference that hold thresholds are lowered to 0.5% instead of being equal to the buy threshold of 1%
"""
mk023config() = (configname="023", featconfig = f6config01(), targetconfig = targetconfig07(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk024 = equal to mk23 but using Trend02 targets
"""
mk024config() = (configname="024", featconfig = f6config01(), targetconfig = targetconfig08(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

""" **my favorite**  
mk9 = mk9 with short term target, i.e. maxwindow 2h
"""
mk025config() = (configname="025", featconfig = f6config01(), targetconfig = targetconfig08(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

""" **my favorite**  
mk9 = mk9 with short term target, i.e. maxwindow 1h
"""
mk026config() = (configname="026", featconfig = f6config01(), targetconfig = targetconfig09(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

