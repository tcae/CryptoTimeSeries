testcoins() = ["SINE", "DOUBLESINE"]
traincoins() = ["1INCH", "AAVE", "ACH", "ADA", "AI16Z", "ALGO", "ANKR", "APEX", "APE", "APT", "ARB", "AR", "ATOM", "AVAX", "AXS", "BCH", "BNB", "BONK", "BRETT", "BTC", "C98", "CAKE", "CARV", "CELO", "CHILLGUY", "CHZ", "COMP", "CRV", "CTC", "DEEP", "DEGEN", "DGB", "DOGE", "DOGS", "DOT", "DRIFT", "DYDX", "EGLD", "ELX", "ENA", "ENJ", "ENS", "EOS", "ETC", "ETH", "FET", "FIL", "FIRE", "FLOCK", "FLOKI", "FLOW", "FTM", "FTT", "FXS", "GALA", "GAL", "GLMR", "GMT", "GOAT", "GPS", "GRASS", "GRT", "HBAR", "HFT", "HNT", "HOOK", "HOT", "H", "ICP", "ID", "IMX", "INIT", "INJ", "IP", "JASMY", "JUP", "KAS", "KAVA", "KLAY", "KSM", "LDO", "LINK", "LRC", "LTC", "LUNA", "LUNC", "MAGIC", "MANA", "MASK", "MATIC", "MAVIA", "MBOX", "MERL", "MINA", "MKR", "MNT", "MOVE", "MYRO", "NAKA", "NEAR", "NOT", "NXPC", "OMG", "ONDO", "ONE", "OP", "PENGU", "PEOPLE", "PEPE", "PLANET", "PLUME", "POL", "POPCAT", "PUFFER", "PUMP", "PYTH", "QNT", "QTUM", "RDNT", "RENDER", "RNDR", "ROSE", "RUNE", "RVN", "SAND", "SC", "SEI", "SERAPH", "SHIB", "SNX", "SOL", "SPX", "STETH", "STG", "STRK", "STX", "SUI", "SUNDOG", "SUSHI", "TAI", "THETA", "TIA", "TON", "TRUMP", "TRX", "TWT", "UNI", "UXLINK", "VIRTUAL", "WAVES", "WAXP", "WIF", "WLD", "XLM", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEN", "ZIL", "ZRX"]
# traincoins() = ["ETH", "BTC", "ADA", "SOL", "XRP"]

settypes() = ["train", "test", "eval"]

partitionconfig01() =(samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=Features.requiredminutes(featconfig), minpartitionsize=12*60, maxpartitionsize=2*24*60)
partitionconfig02() =(samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=8*60, minpartitionsize=12*60, maxpartitionsize=2*24*60)

resultsfilename(coin=nothing) = isnothing(coin) ? joinpath("results", "all") : joinpath("results", String(coin)) # includes hlcp, sets, ranges, targets
featuresfilename(coin=nothing) = isnothing(coin) ? joinpath("features", "all") : joinpath("features", String(coin))
predictionsfilename() = joinpath("predictions", "maxpredictions")
confusionfilename() = joinpath("predictions", "confusion")
xconfusionfilename() = joinpath("predictions", "xconfusion")
distancesfilename() = joinpath("trades", "distances")
gainsfilename() = joinpath("trades", "gains_all")
targetissuesfilename() = joinpath("results", "targetissues")

default_openthresholds() = Float32[0.8f0, 0.7f0, 0.6f0, 0.5f0, 0.4f0, 0.3f0]
default_closethresholds() = Float32[0.1f0]

tradingstrategy01() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.6, closethreshold=0.5, makerfee=0.0015)
tradingstrategy02() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
tradingstrategy03() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015)
tradingstrategy04() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.4, makerfee=0.0015, buygain=0f0, limitreduction=0.05f0)
tradingstrategy05() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015, minpricedelta=0.002f0, max_classify_staleness_minutes=5)
tradingstrategy06() = TradingStrategy.StrategyConfig(maxwindow=4*60, algorithm=TradingStrategy.gain_limit_reversal!, openthreshold=0.6, makerfee=0.0015, minpricedelta=0.002f0, max_classify_staleness_minutes=5)
# Trend01/Trend02 were replaced by Trend04.
trend04targetconfig(minwindow, maxwindow, buy, hold; holdbehaviormode=beyond_maxwindow) = Targets.Trend04(minwindow, maxwindow, Targets.thresholds((longopen=buy, longhold=hold, shorthold=-hold, shortopen=-buy)), holdbehaviormode=holdbehaviormode)

targetconfig01() = trend04targetconfig(10, 4*60, 0.01, 0.01, holdbehaviormode=beyond_maxwindow)
targetconfig02() = trend04targetconfig(10, 4*60, 0.05, 0.03)
targetconfig03() = trend04targetconfig(10, 4*60, 0.02, 0.01)
targetconfig04() = trend04targetconfig(10, 4*60, 0.007, 0.005)
targetconfig05() = trend04targetconfig(0, 4*60, 0.01, 0.01)
targetconfig06() = trend04targetconfig(2, 4*60, 0.01, 0.01)
targetconfig07() = trend04targetconfig(10, 4*60, 0.01, 0.005, holdbehaviormode=beyond_maxwindow)
targetconfig08() = trend04targetconfig(10, 2*60, 0.01, 0.005) # Trend02 replaced by Trend04
targetconfig09() = trend04targetconfig(10, 2*60, 0.01, 0.01)
targetconfig10() = trend04targetconfig(10, 1*60, 0.01, 0.01)
targetconfig11() = trend04targetconfig(10, 2*60, 0.01, 0.005) # Trend04
targetconfig12() = trend04targetconfig(10, 24*60, 0.05, 0.01, holdbehaviormode=beyond_maxwindow) # Trend04 with long term target
targetconfig13() = trend04targetconfig(10, 2*60, 0.01, 0.01, holdbehaviormode=beyond_maxwindow) 
targetconfig14() = trend04targetconfig(10, 4*60, 0.01, 0.01, holdbehaviormode=no_hold)
targetconfig15() = trend04targetconfig(10, 60, 0.01, 0.01, holdbehaviormode=no_hold) 
targetconfig16() = trend04targetconfig(10, 2*60, 0.01, 0.01, holdbehaviormode=no_hold) 
targetconfig17() = trend04targetconfig(60, 24*60, 0.05, 0.05, holdbehaviormode=no_hold) # Trend04 with long term target
targetconfig18() = Targets.TrendRegression(5, 0.01, -0.01) # for SINE and DOUBLESINE regression tests only
targetconfig19() = Targets.TrendRegression(24*60, 0.01, -0.01) 
targetconfig20() = Targets.TrendRegression(4*60, 0.005, -0.005) 
targetconfig21() = Targets.TrendRegression(3*24*60, 0.05, -0.05) 
targetconfig22() = Targets.TrendRegression(4*60, 0.01, -0.01) 
targetconfig23() = Targets.TrendRegression(4*60, 0.02, -0.02) 

boundstargetsconfig01(window) = Targets.Bounds01(window)



#region FeaturesConfig

" start 3* 5min with short segments and also add a 5min std element for high volatile situations then become larger "
function trendf6config01()
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

"like trendf6config01 but without the 5min std element and only with 1* 5min regr"
function trendf6config03()
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

"like trendf6config01 but without the 5min std and regr elements - just starting with 15min"
function trendf6config04()
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

"like trendf6config03 but the larger regr grad starting without offset"
function trendf6config05()
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

"like trendf6config05 but with an added 12h regr grad"
function trendf6config06()
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

"like trendf6config06 but with an added 3*24h regr grad"
function trendf6config07()
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

"long trend features without offset"
function trendf6config08()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=12*60, offset=0)
    Features.addgrad!(featcfg, window=24*60, offset=0)
    Features.addgrad!(featcfg, window=3*24*60, offset=0)
    Features.addrelvol!(featcfg, short=15, long=3*24*60, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config09()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config10()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=60*12, offset=0)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config11()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addmaxdist!(featcfg, window=60, offset=0)
    Features.addmindist!(featcfg, window=60, offset=0)
    Features.addrelvol!(featcfg, short=5, long=60*6, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset - short regressions for SINE and DOUBLESINE only"
function trendf6config12()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=5, offset=0)
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=12*60, offset=0)
    Features.addrelvol!(featcfg, short=15, long=24*60, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config13()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=12*60, offset=0)
    Features.addgrad!(featcfg, window=24*60, offset=0)
    Features.addgrad!(featcfg, window=3*24*60, offset=0)
    Features.addrelvol!(featcfg, short=15, long=3*24*60, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config14()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=15, offset=0)
    Features.addgrad!(featcfg, window=60, offset=0)
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=12*60, offset=0)
    Features.addgrad!(featcfg, window=24*60, offset=0)
    Features.addrelvol!(featcfg, short=5, long=24*60, offset=0)
    return featcfg
end

"short feature vector relying on grad without offset"
function trendf6config15()
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=60*4, offset=0)
    Features.addgrad!(featcfg, window=12*60, offset=0)
    Features.addgrad!(featcfg, window=24*60, offset=0)
    Features.addgrad!(featcfg, window=3*24*60, offset=0)
    Features.addgrad!(featcfg, window=10*24*60, offset=0)
    Features.addrelvol!(featcfg, short=60, long=10*24*60, offset=0)
    return featcfg
end

"provide direction with standard deviation and their upper and lower relative bounds distance for given window"
function boundsf6config01(window)
    featcfg = Features.Features006()
    Features.addgrad!(featcfg, window=window, offset=0)
    Features.addstd!(featcfg, window=window, offset=0)
    Features.addmaxdist!(featcfg, window=window, offset=0)
    Features.addmindist!(featcfg, window=window, offset=0)
    return featcfg
end

#endregion FeaturesConfig

#region TrendConfig

"""Return a one-step TrendDetector config payload."""
function trendmkconfig(configname::AbstractString, featconfig, targetconfig, classifiermodel, tradingstrategy; classifiertype::Type{<:Classify.AbstractClassifier}=Classify.TrendClassifier001, classbalancing::Bool=true)
    return (
        configname=String(configname),
        featconfig=featconfig,
        targetconfig=targetconfig,
        classifiermodel=classifiermodel,
        classifiertype=classifiertype,
        tradingstrategy=tradingstrategy,
        classbalancing=classbalancing,
    )
end

"""
mk1 = mix adapted with multiple adaptations per coin with **good results**: ppv(longbuy) = 72%
"""
mk001config() = trendmkconfig("001", trendf6config01(), targetconfig01(), Classify.model001, tradingstrategy02())

"""
mk2 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk002config() = trendmkconfig("002", trendf6config01(), targetconfig01(), Classify.model001, tradingstrategy02())

"""
mk3 = one iteration adaptation with one merged set (allclose=>0.9, longbuy=>0.1), but target config targetconfig02() with **poor results**: ppv(longbuy) = 26%
"""
mk003config() = trendmkconfig("003", trendf6config01(), targetconfig02(), Classify.model001, tradingstrategy02())

"""
mk4 = targetconfig03() with one merged set (allclose=>0.726, longbuy=>0.274)
"""
mk004config() = trendmkconfig("004", trendf6config01(), targetconfig03(), Classify.model001, tradingstrategy02())

"""
mk5 = targetconfig04() with one merged set (allclose=>?, longbuy=>?)
"""
mk005config() = trendmkconfig("005", trendf6config01(), targetconfig04(), Classify.model001, tradingstrategy02())

"""
mk6 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk006config() = trendmkconfig("006", trendf6config01(), targetconfig01(), Classify.model001, tradingstrategy02())

"""
same as mk6 butwith copied mix classifier from mk2
mk7 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk007config() = trendmkconfig("007", trendf6config01(), targetconfig01(), Classify.model001, tradingstrategy02())

"""
mk8 = mix adapted with all coin features/targets in one set, features are clipped, normalized, shifted, and in addition batch norm layer after initial layer with relu activation in model001
equal mean, q25, q75, min, max does not look like healthy feature values - longbuy ppv classification performance is with close to 70% also worse
"""
# mk008config() = trendmkconfig("008", trendf6config02(), targetconfig01(), Classify.model001, tradingstrategy02())

""" **my favorite**  
mk9 = mix adapted with all coin features/targets in one set, features are not clipped, batch norm layer before and between layers with relu activation in model002
"""
mk009config() = trendmkconfig("009", trendf6config01(), targetconfig01(), Classify.model002, tradingstrategy02(); classbalancing=true)

"""
mk10 = mix adapted with all coin features/targets in one set, features are clipped, initial batch norm layer in model002
"""
# mk010config() = (configname="010", featconfig = trendf6config02(), targetconfig = targetconfig01(), classifiermodel=Classify.model002)

"""
mk11 = mix adapted with all coin features/targets in one set, no clipping, initial batch norm layer but no further internal batch norm layers
"""
mk011config() = trendmkconfig("011", trendf6config01(), targetconfig01(), Classify.model003, tradingstrategy02())

"""
mk12 = like mk9 but with an additional layer
"""
mk012config() = trendmkconfig("012", trendf6config01(), targetconfig01(), Classify.model004, tradingstrategy02())

"""
mk13 = like mk9 but with 4/3 broader layers
"""
mk013config() = trendmkconfig("013", trendf6config01(), targetconfig01(), Classify.model005, tradingstrategy02())

"""
mk14 = like mk11 but removed layer 3
"""
mk014config() = trendmkconfig("014", trendf6config01(), targetconfig01(), Classify.model006, tradingstrategy02())

"""
mk15 = like mk11 but reduced number of nodes of layers by reducing factor of layer 1 from 3 to 2
"""
mk015config() = trendmkconfig("015", trendf6config01(), targetconfig01(), Classify.model007, tradingstrategy02())

"""
mk16 = no tolerance against target disturbances (minwindow=0), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk016config() = trendmkconfig("016", trendf6config01(), targetconfig05(), Classify.model002, tradingstrategy02())

"""
mk17 = short tolerance against target disturbances (minwindow=2), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk017config() = trendmkconfig("017", trendf6config01(), targetconfig06(), Classify.model002, tradingstrategy02())

"""  
mk18 = mk9 but with simplified features
"""
mk018config() = trendmkconfig("018", trendf6config03(), targetconfig01(), Classify.model002, tradingstrategy02())

"""  
mk19 = mk9 but with simplified features
"""
mk019config() = trendmkconfig("019", trendf6config04(), targetconfig01(), Classify.model002, tradingstrategy02())

"""  
mk20 = mk17 but feature grads without offset
"""
mk020config() = trendmkconfig("020", trendf6config05(), targetconfig06(), Classify.model002, tradingstrategy02())

"""  
mk21 = mk17 but feature grads without offset and 12h regr grad added
"""
mk021config() = trendmkconfig("021", trendf6config06(), targetconfig06(), Classify.model002, tradingstrategy02())

"""  
mk22 = mk21 but with 3*24h regr grad added
"""
mk022config() = trendmkconfig("022", trendf6config07(), targetconfig06(), Classify.model002, tradingstrategy02())
#* from mk23 onwards partitionconfig02() shall be used with a fixed partition gap of 8h to get the same targets gains for the same targetconfig but different requiredminutes(features)

"""  
mk023 = equal to mk9 with the only difference that hold thresholds are lowered to 0.5% instead of being equal to the buy threshold of 1%
"""
mk023config() = trendmkconfig("023", trendf6config01(), targetconfig07(), Classify.model002, tradingstrategy02())

"""  
mk024 = equal to mk23 but Trend01/Trend02 were replaced by Trend04 targets
"""
mk024config() = trendmkconfig("024", trendf6config01(), targetconfig08(), Classify.model002, tradingstrategy02())

"""
mk9 = mk9 with short term target, i.e. maxwindow 2h
"""
mk025config() = trendmkconfig("025", trendf6config01(), targetconfig09(), Classify.model002, tradingstrategy02())
mk025bconfig() = trendmkconfig("025b", trendf6config01(), targetconfig09(), Classify.model002, tradingstrategy02(); classbalancing=false)
mk025Cconfig() = trendmkconfig("025C", trendf6config01(), targetconfig08(), Classify.model002, tradingstrategy02(); classbalancing=false)
mk025Dconfig() = trendmkconfig("025D", trendf6config01(), targetconfig11(), Classify.model002, tradingstrategy02(); classbalancing=false)
# mk25D and mk25E are the same config but in teh meanwhile the missing hold was fixed
mk025Econfig() = trendmkconfig("025E", trendf6config01(), targetconfig11(), Classify.model002, tradingstrategy02(); classbalancing=false)

"""
mk026 = mk9 with short term target, i.e. maxwindow 1h
"""
mk026config() = trendmkconfig("026", trendf6config01(), targetconfig10(), Classify.model002, tradingstrategy02())

"""
mk27 = long term trend with long term window and 5% ambition
"""
mk027config() = trendmkconfig("027", trendf6config08(), targetconfig12(), Classify.model002, tradingstrategy02())
mk039config() = trendmkconfig("039", trendf6config12(), targetconfig17(), Classify.model002, tradingstrategy02(); classbalancing=true)

"""   
mk028 = mk009 without class balancing
"""
mk028config() = trendmkconfig("028", trendf6config01(), targetconfig01(), Classify.model002, tradingstrategy02(); classbalancing=false)

"""   
mk029 = mk009 without class balancing but implementation for hold equally strict than for buy (no config change)
"""
mk029config() = trendmkconfig("029", trendf6config01(), targetconfig01(), Classify.model002, tradingstrategy02(); classbalancing=false)

"""   
relies on grad without offset and 2h target, i.e. maxwindow 2h, hold threshold equal to buy threshold, no class balancing
"""
mk030config() = trendmkconfig("030", trendf6config09(), targetconfig09(), Classify.model002, tradingstrategy02(); classbalancing=false)
mk031config() = trendmkconfig("031", trendf6config09(), targetconfig09(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk032config() = trendmkconfig("032", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk046config() = trendmkconfig("046", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy03(); classbalancing=true)
mk047config() = trendmkconfig("047", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy04(); classbalancing=true)
mk048config() = trendmkconfig("048", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy05(); classbalancing=true)
mk049config() = trendmkconfig("049", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy06(); classbalancing=true)
mk033config() = trendmkconfig("033", trendf6config09(), targetconfig01(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk034config() = trendmkconfig("034", trendf6config09(), targetconfig07(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk035config() = trendmkconfig("035", trendf6config10(), targetconfig14(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk037config() = trendmkconfig("037", trendf6config09(), targetconfig13(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk038config() = trendmkconfig("038", trendf6config09(), targetconfig16(), Classify.model002, tradingstrategy02(); classbalancing=true)

mk036config() = trendmkconfig("036", trendf6config11(), targetconfig15(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk040config() = trendmkconfig("040", trendf6config12(), targetconfig18(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk041config() = trendmkconfig("041", trendf6config13(), targetconfig19(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk042config() = trendmkconfig("042", trendf6config14(), targetconfig20(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk043config() = trendmkconfig("043", trendf6config15(), targetconfig21(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk044config() = trendmkconfig("044", trendf6config14(), targetconfig22(), Classify.model002, tradingstrategy02(); classbalancing=true)
mk045config() = trendmkconfig("045", trendf6config14(), targetconfig23(), Classify.model002, tradingstrategy02(); classbalancing=true)

#endregion TrendConfig

#region BoundsConfig
"Bounds estimator for short term limits"
boundsmk001config() = (configname="001", featconfig = boundsf6config01(5), targetconfig = boundstargetsconfig01(5), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

"Bounds estimator for mid term limits"
boundsmk002config() = (configname="002", featconfig = boundsf6config01(4*60), targetconfig = boundstargetsconfig01(4*60), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

#endregion BoundsConfig

function _config_from_dict(configs::Dict{String, NamedTuple}, ref::AbstractString; label::AbstractString="config", prefixes=("",))
    raw = strip(String(ref))
    lowerraw = lowercase(raw)
    for prefix in prefixes
        lowerprefix = lowercase(prefix)
        if startswith(lowerraw, lowerprefix)
            raw = strip(raw[(length(prefix) + 1):end])
            lowerraw = lowercase(raw)
            break
        end
    end

    matches = [key for key in keys(configs) if lowercase(key) == lowerraw]
    @assert length(matches) == 1 "unknown $(label) config ref=$(ref); available confignames=$(join(sort!(collect(keys(configs)); by=lowercase), ", "))"
    return configs[only(matches)]
end

const TREND_DETECTOR_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    mk001config(), mk002config(), mk003config(), mk004config(), mk005config(), mk006config(), mk007config(),
    mk009config(), mk011config(), mk012config(), mk013config(), mk014config(), mk015config(), mk016config(),
    mk017config(), mk018config(), mk019config(), mk020config(), mk021config(), mk022config(), mk023config(),
    mk024config(), mk025config(), mk025bconfig(), mk025Cconfig(), mk025Dconfig(), mk025Econfig(), mk026config(),
    mk027config(), mk028config(), mk029config(), mk030config(), mk031config(), mk032config(), mk033config(), 
    mk034config(), mk035config(), mk036config(), mk037config(), mk038config(), mk039config(), mk040config(), 
    mk041config(), mk042config(), mk043config(), mk044config(), mk045config(), mk046config(), mk047config(), mk048config(), mk049config(),
])

const BOUNDS_ESTIMATOR_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    boundsmk001config(), boundsmk002config(),
])

trenddetectorconfig(ref::AbstractString) = _config_from_dict(TREND_DETECTOR_CONFIGS, ref; label="trend", prefixes=("trenddetector", "trend", "mk"))
boundsestimatorconfig(ref::AbstractString) = _config_from_dict(BOUNDS_ESTIMATOR_CONFIGS, ref; label="bounds", prefixes=("boundsestimator", "boundsmk", "bounds", "mk"))

"Return a TradingStrategy.StrategyConfig with an instantiated classifier for a TrendDetector config payload."
function strategyconfig(cfg::NamedTuple; mnemonic::AbstractString="mix", mode=EnvConfig.configmode)::TradingStrategy.StrategyConfig
    classifier = loadtrendclassifier(cfg; mnemonic=mnemonic, mode=mode)
    return _strategy_with_classifier(cfg.tradingstrategy, classifier)
end

"Return a TradingStrategy.StrategyConfig with an instantiated classifier for a TrendDetector config reference."
function strategyconfig(ref::AbstractString; mnemonic::AbstractString="mix", mode=EnvConfig.configmode)::TradingStrategy.StrategyConfig
    return strategyconfig(trenddetectorconfig(ref); mnemonic=mnemonic, mode=mode)
end

"Return the canonical config reference string for a TrendDetector config payload."
trendconfigref(cfg::NamedTuple)::String = String(cfg.configname)

"Return the canonical TrendDetector model folder name for a given config payload."
trendconfigfolder(cfg::NamedTuple, phase::AbstractString)::String = "Trend-$(trendconfigref(cfg))-$(String(phase))"

"Return the source tag used when wiring a TrendDetector config into Trade."
trendconfigsource(cfg::NamedTuple; prefix::AbstractString="trenddetector")::String = "$(String(prefix)):$(trendconfigref(cfg))"

"Return the feature-config factory for a TrendDetector config payload."
trendconfigfeaturefactory(cfg::NamedTuple)::Function = () -> cfg.featconfig

"Load a runtime classifier for a TrendDetector config payload."
function loadtrendclassifier(cfg::NamedTuple; mnemonic::AbstractString="mix", mode=EnvConfig.configmode)::Classify.TrendClassifier001
    nnstub = cfg.classifiermodel(Features.featurecount(cfg.featconfig), Targets.uniquelabels(cfg.targetconfig), mnemonic)
    required_minutes = max(Features.requiredminutes(cfg.featconfig), 2)
    spec = (
        config_ref=trendconfigref(cfg),
        nn_fileprefix=nnstub.fileprefix,
        featconfig=trendconfigfeaturefactory(cfg),
        required_minutes=required_minutes,
    )
    return Classify.load(Classify.TrendClassifier001, spec; mode=mode)
end

