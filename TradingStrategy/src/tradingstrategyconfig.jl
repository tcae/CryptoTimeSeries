testcoins() = ["SINE", "DOUBLESINE"]
traincoins() = ["1INCH", "AAVE", "ACH", "ADA", "AI16Z", "ALGO", "ANKR", "APEX", "APE", "APT", "ARB", "AR", "ATOM", "AVAX", "AXS", "BCH", "BNB", "BONK", "BRETT", "BTC", "C98", "CAKE", "CARV", "CELO", "CHILLGUY", "CHZ", "COMP", "CRV", "CTC", "DEEP", "DEGEN", "DGB", "DOGE", "DOGS", "DOT", "DRIFT", "DYDX", "EGLD", "ELX", "ENA", "ENJ", "ENS", "EOS", "ETC", "ETH", "FET", "FIL", "FIRE", "FLOCK", "FLOKI", "FLOW", "FTM", "FTT", "FXS", "GALA", "GAL", "GLMR", "GMT", "GOAT", "GPS", "GRASS", "GRT", "HBAR", "HFT", "HNT", "HOOK", "HOT", "H", "ICP", "ID", "IMX", "INIT", "INJ", "IP", "JASMY", "JUP", "KAS", "KAVA", "KLAY", "KSM", "LDO", "LINK", "LRC", "LTC", "LUNA", "LUNC", "MAGIC", "MANA", "MASK", "MATIC", "MAVIA", "MBOX", "MERL", "MINA", "MKR", "MNT", "MOVE", "MYRO", "NAKA", "NEAR", "NOT", "NXPC", "OMG", "ONDO", "ONE", "OP", "PENGU", "PEOPLE", "PEPE", "PLANET", "PLUME", "POL", "POPCAT", "PUFFER", "PUMP", "PYTH", "QNT", "QTUM", "RDNT", "RENDER", "RNDR", "ROSE", "RUNE", "RVN", "SAND", "SC", "SEI", "SERAPH", "SHIB", "SNX", "SOL", "SPX", "STETH", "STG", "STRK", "STX", "SUI", "SUNDOG", "SUSHI", "TAI", "THETA", "TIA", "TON", "TRUMP", "TRX", "TWT", "UNI", "UXLINK", "VIRTUAL", "WAVES", "WAXP", "WIF", "WLD", "XLM", "XRP", "XTER", "XTZ", "XWG", "X", "YFI", "ZEN", "ZIL", "ZRX"]
# traincoins() = ["ETH", "BTC", "ADA", "SOL", "XRP"]

settypes() = ["train", "test", "eval"]

partitionconfig02() = (samplesets = ["train", "test", "train", "train", "eval", "train"], partitionsize=24*60, gapsize=8*60, minpartitionsize=12*60, maxpartitionsize=2*24*60)

#region TradingStrategies
tradingstrategy01() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.6, makerfee=0.0025, stoploss=0.05f0), 
    maxbudgetquote=500f0,
    makerfee=0.0025)
tradingstrategy02() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.6, makerfee=0.0025), 
    makerfee=0.0025)
# original: tradingstrategy03() = TradingStrategy.StrategyConfig(algorithm=TradingStrategy.gain_limit_reversal!, algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.6), makerfee=0.0015)
tradingstrategy03() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.3, makerfee=0.0025, incrementgain=0.001f0, stoploss=0.05f0), 
    makerfee=0.0025, 
    takerfee=0.004, 
    enforcemakerlimits=true)
tradingstrategy04() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.4, buygain=0f0, makerfee=0.0025), 
    makerfee=0.0025)
tradingstrategy05() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.6, minpricedelta=0.002f0, makerfee=0.0025), 
    makerfee=0.0025)
tradingstrategy06() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.6, minpricedelta=0.002f0, makerfee=0.0025), 
    makerfee=0.0025)
tradingstrategy07() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal!, 
    algorithmconfig=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.4, makerfee=0.0025, incrementgain=0f0, stoploss=0.05f0), 
    makerfee=0.0025, 
    takerfee=0.004, 
    enforcemakerlimits=true)
tradingstrategy08() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal_below_regression!, 
    algorithmconfig=TradingStrategy.GainLimitReversalBelowRegressionConfig(
        gainlimit=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.4, buygain=0.001f0, incrementgain=0f0, makerfee=0.0025), 
        triggerdist=0.005f0, 
        triggerregrwindow=0, # zero disables the trigger regression window, previously 12*60, 
        trendregrwindow=0, # zero disables the trend regression window, previously 24*60, 
        trendgainthreshold=0.01f0 / 4f0), 
    makerfee=0.0025, 
    takerfee=0.004, 
    enforcemakerlimits=true)
tradingstrategy09() = TradingStrategy.StrategyConfig(
    algorithm=TradingStrategy.gain_limit_reversal_below_regression!, 
    algorithmconfig=TradingStrategy.GainLimitReversalBelowRegressionConfig(
        gainlimit=TradingStrategy.GainLimitReversalConfig(maxwindow=4*60, openthreshold=0.4, buygain=0.001f0, incrementgain=0f0, makerfee=0.0025), 
        triggerdist=0.005f0, 
        triggerregrwindow=4*60, # zero disables the trigger regression window, previously 12*60, 
        trendregrwindow=4*60, # zero disables the trend regression window, previously 24*60, 
        trendgainthreshold=0.01f0 / 4f0), 
    makerfee=0.0025, 
    takerfee=0.004, 
    enforcemakerlimits=true)
#endregion TradingStrategies

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

function trendf6config16()
    featcfg = Features.Features006()
    for window in (5, 15, 60, 4*60)
        Features.addnormgrad!(featcfg, window=window)
    end
    Features.addmaxdist!(featcfg, window=60)
    Features.addmindist!(featcfg, window=60)
    Features.addrelvol!(featcfg, short=5, long=6*60)
    return featcfg
end

function trendf6config17()
    featcfg = Features.Features006()
    for window in (15, 60, 4*60, 12*60)
        Features.addnormgrad!(featcfg, window=window)
    end
    Features.addrelvol!(featcfg, short=5, long=6*60)
    return featcfg
end

function trendf6config18()
    featcfg = Features.Features006()
    for window in (60, 4*60, 24*60, 3*24*60)
        Features.addnormgrad!(featcfg, window=window)
    end
    Features.addrelvol!(featcfg, short=5, long=6*60)
    return featcfg
end

function trendf6configlst(lead, support, xsupport, trend)
    featcfg = Features.Features006()
    for window in (lead, support, xsupport, trend)
        Features.addnormgrad!(featcfg, window=window)
    end
    Features.addnormgrad!(featcfg, window=support, offset=support-1)
    Features.addregry!(featcfg, window=trend)
    Features.addrelvol!(featcfg, short=lead, long=trend)
    return featcfg
end

trendf6config19() = trendf6configlst(5, 15, 60, 4*60)
trendf6config20() = trendf6configlst(15, 60, 4*60, 12*60)
trendf6config21() = trendf6configlst(60, 4*60, 24*60, 3*24*60)

#endregion FeaturesConfig

#region Registries

# Every reusable artifact is identified by registry ids of the components it depends on:
# - features `Fxx` -> per-coin feature store `coins/<pair>/features/Fxx.arrow`
# - targets `Txx` -> per-coin target store `coins/<pair>/targets/Txx.arrow`
# - partition `Pxx` + date range `Dxx` -> per-coin sets store `coins/<pair>/sets/Pxx_Dxx.arrow`
# - classifier `Fxx-Txx-Pxx-Mxxx-Bx[-variant]-CSxx-Dxx` -> classifier folder with NN, training index and predictions
# - trading strategy context `TS_CONFIGS` = classifier base name + strategy `Sxx`
# Content hashes of features, targets and sets configs are stored with the artifacts and asserted on load,
# i.e. a changed registry entry requires a new id.

const FEATURE_CONFIGS = Dict{String, Function}(
    "F01" => trendf6config01, "F03" => trendf6config03, "F04" => trendf6config04, "F05" => trendf6config05,
    "F06" => trendf6config06, "F07" => trendf6config07, "F08" => trendf6config08, "F09" => trendf6config09,
    "F10" => trendf6config10, "F11" => trendf6config11, "F12" => trendf6config12, "F13" => trendf6config13,
    "F14" => trendf6config14, "F15" => trendf6config15,
    "F16" => trendf6config16, "F17" => trendf6config17, "F18" => trendf6config18,
    "F19" => trendf6config19, "F20" => trendf6config20, "F21" => trendf6config21
)

const TARGET_CONFIGS = Dict{String, Function}(
    "T01" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T02" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.05, longhold=0.03, shorthold=-0.03, shortopen=-0.05)), holdbehaviormode=Targets.beyond_maxwindow),
    "T03" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.02, longhold=0.01, shorthold=-0.01, shortopen=-0.02)), holdbehaviormode=Targets.beyond_maxwindow),
    "T04" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.007, longhold=0.005, shorthold=-0.005, shortopen=-0.007)), holdbehaviormode=Targets.beyond_maxwindow),
    "T05" => () -> Targets.Trend04(0, 4*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T06" => () -> Targets.Trend04(2, 4*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T07" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.01, longhold=0.005, shorthold=-0.005, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T08" => () -> Targets.Trend04(10, 2*60, Targets.thresholds((longopen=0.01, longhold=0.005, shorthold=-0.005, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T09" => () -> Targets.Trend04(10, 2*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T10" => () -> Targets.Trend04(10, 1*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.beyond_maxwindow),
    "T12" => () -> Targets.Trend04(10, 24*60, Targets.thresholds((longopen=0.05, longhold=0.01, shorthold=-0.01, shortopen=-0.05)), holdbehaviormode=Targets.beyond_maxwindow),
    "T14" => () -> Targets.Trend04(10, 4*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.no_hold),
    "T15" => () -> Targets.Trend04(10, 60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.no_hold),
    "T16" => () -> Targets.Trend04(10, 2*60, Targets.thresholds((longopen=0.01, longhold=0.01, shorthold=-0.01, shortopen=-0.01)), holdbehaviormode=Targets.no_hold),
    "T17" => () -> Targets.Trend04(60, 24*60, Targets.thresholds((longopen=0.05, longhold=0.05, shorthold=-0.05, shortopen=-0.05)), holdbehaviormode=Targets.no_hold),
    "T18" => () -> Targets.TrendRegression(5, 0.01, -0.01),
    "T19" => () -> Targets.TrendRegression(24*60, 0.01, -0.01),
    "T20" => () -> Targets.TrendRegression(4*60, 0.005, -0.005),
    "T21" => () -> Targets.TrendRegression(3*24*60, 0.05, -0.05),
    "T22" => () -> Targets.TrendRegression(4*60, 0.01, -0.01),
    "T23" => () -> Targets.TrendRegression(4*60, 0.02, -0.02),
    "T24" => () -> Targets.Trend05(leadregr=5, supportregr=15, trendregr=4*60, triggerdist=0.01f0, targetgain=0.01f0),
    "T25" => () -> Targets.Trend05(leadregr=15, supportregr=60, trendregr=12*60, triggerdist=0.02f0, targetgain=0.02f0),
    "T26" => () -> Targets.Trend05(leadregr=60, supportregr=4*60, trendregr=3*24*60, triggerdist=0.05f0, targetgain=0.05f0),
)

const PARTITION_CONFIGS = Dict{String, Function}("P02" => partitionconfig02)

"Coin sets with the epoch budget of their training, which scales with the data volume of the coin set."
const COINSET_CONFIGS = Dict{String, NamedTuple}(
    "CS00" => (coins=testcoins(), maxepochs=10),
    "CS01" => (coins=traincoins(), maxepochs=200),
)

const DATERANGE_CONFIGS = Dict{String, NamedTuple}(
    "D00" => (startdt=DateTime("2025-01-17T20:56:00"), enddt=DateTime("2025-08-10T15:00:00")),
    "D01" => (startdt=DateTime("2017-11-17T20:56:00"), enddt=DateTime("2025-08-10T15:00:00")),
)

const MODEL_CONFIGS = Dict{String, Function}(
    "M001" => Classify.model001, "M002" => Classify.model002, "M003" => Classify.model003, "M004" => Classify.model004,
    "M005" => Classify.model005, "M006" => Classify.model006, "M007" => Classify.model007, "M008" => Classify.model008,
    "M009" => Classify.model009, "M010" => Classify.model010, "M011" => Classify.model011,
)

const STRATEGY_CONFIGS = Dict{String, Function}(
    "S01" => tradingstrategy01, "S02" => tradingstrategy02, "S03" => tradingstrategy03, "S04" => tradingstrategy04,
    "S05" => tradingstrategy05, "S06" => tradingstrategy06, "S07" => tradingstrategy07, "S08" => tradingstrategy08,
)

"Return a classifier config that is independent of the coin set and date range it is trained on."
clconfig(features::AbstractString, targets::AbstractString, model::AbstractString; classbalancing::Bool=true, partition::AbstractString="P02", variant::AbstractString="") =
    (features=String(features), targets=String(targets), partition=String(partition), model=String(model), classbalancing=classbalancing, variant=String(variant))

"Return the classifier base name `Fxx-Txx-Pxx-Mxxx-Bx[-variant]` of a classifier config."
classifierbasename(clcfg::NamedTuple)::String = join([clcfg.features, clcfg.targets, clcfg.partition, clcfg.model, clcfg.classbalancing ? "B1" : "B0"], "-") * (isempty(clcfg.variant) ? "" : "-" * clcfg.variant)

"Return the unique classifier artifact name of a classifier config trained on `coinset` within `daterange`."
classifiername(clcfg::NamedTuple; coinset::AbstractString, daterange::AbstractString)::String = classifierbasename(clcfg) * "-" * String(coinset) * "-" * String(daterange)

const CLASSIFIER_CONFIGS = Dict{String, NamedTuple}(classifierbasename(c) => c for c in [
    clconfig("F01", "T01", "M002"),                         # former mk009
    clconfig("F01", "T05", "M002"),                         # former mk016
    clconfig("F01", "T06", "M002"),                         # former mk017
    clconfig("F03", "T01", "M002"),                         # former mk018
    clconfig("F04", "T01", "M002"),                         # former mk019
    clconfig("F05", "T06", "M002"),                         # former mk020
    clconfig("F06", "T06", "M002"),                         # former mk021
    clconfig("F07", "T06", "M002"),                         # former mk022
    clconfig("F01", "T07", "M002"),                         # former mk023
    clconfig("F01", "T08", "M002"),                         # former mk024
    clconfig("F01", "T09", "M002"),                         # former mk025
    clconfig("F01", "T09", "M002"; classbalancing=false),   # former mk025b
    clconfig("F01", "T08", "M002"; classbalancing=false),   # former mk025C, mk025D, mk025E
    clconfig("F01", "T10", "M002"),                         # former mk026
    clconfig("F08", "T12", "M002"),                         # former mk027
    clconfig("F01", "T01", "M002"; classbalancing=false),   # former mk028, mk029
    clconfig("F09", "T09", "M002"; classbalancing=false),   # former mk030
    clconfig("F09", "T09", "M002"),                         # former mk031, mk032, mk037, mk046 - mk049
    clconfig("F09", "T09", "M011"),                         
    clconfig("F09", "T01", "M002"),                         # former mk033
    clconfig("F09", "T07", "M002"),                         # former mk034
    clconfig("F10", "T14", "M002"),                         # former mk035
    clconfig("F11", "T15", "M002"),                         # former mk036
    clconfig("F09", "T16", "M002"),                         # former mk038, mk038b
    clconfig("F09", "T16", "M011"),                         
    clconfig("F12", "T17", "M002"),                         # former mk039
    clconfig("F12", "T18", "M002"),                         # former mk040, short regressions for SINE and DOUBLESINE
    clconfig("F13", "T19", "M002"),                         # former mk041
    clconfig("F14", "T20", "M002"),                         # former mk042
    clconfig("F15", "T21", "M002"),                         # former mk043
    clconfig("F14", "T22", "M002"),                         # former mk044
    clconfig("F14", "T23", "M002"),                         # former mk045
    clconfig("F09", "T24", "M002"),                         # former mk050
    clconfig("F10", "T25", "M002"),
    clconfig("F09", "T24", "M008"),                         # former mk050
    clconfig("F10", "T25", "M008"),
    clconfig("F16", "T16", "M008"),
    clconfig("F16", "T09", "M008"),
    clconfig("F16", "T24", "M008"),
    clconfig("F17", "T25", "M008"),
    clconfig("F18", "T26", "M008"),
    clconfig("F19", "T24", "M008"),
    clconfig("F19", "T24", "M009"),
    clconfig("F19", "T24", "M010"),
    clconfig("F19", "T24", "M011"),
    clconfig("F20", "T25", "M008"),
    clconfig("F20", "T25", "M011"),
    clconfig("F21", "T26", "M008"),
    clconfig("F21", "T26", "M011"),
])

"Return a trading strategy context of a classifier (base name) and a strategy id."
tsentry(configname::AbstractString, classifier::AbstractString, strategy::AbstractString) = (configname=String(configname), classifier=String(classifier), strategy=String(strategy))

# Keys of former TrendDetector configs are kept for CLI continuity; 101-103 are the former ts001-ts003.
const TS_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    tsentry("009", "F01-T01-P02-M002-B1", "S02"),
    tsentry("016", "F01-T05-P02-M002-B1", "S02"),
    tsentry("017", "F01-T06-P02-M002-B1", "S02"),
    tsentry("018", "F03-T01-P02-M002-B1", "S02"),
    tsentry("019", "F04-T01-P02-M002-B1", "S02"),
    tsentry("020", "F05-T06-P02-M002-B1", "S02"),
    tsentry("021", "F06-T06-P02-M002-B1", "S02"),
    tsentry("022", "F07-T06-P02-M002-B1", "S02"),
    tsentry("023", "F01-T07-P02-M002-B1", "S02"),
    tsentry("024", "F01-T08-P02-M002-B1", "S02"),
    tsentry("025", "F01-T09-P02-M002-B1", "S02"),
    tsentry("025b", "F01-T09-P02-M002-B0", "S02"),
    tsentry("025C", "F01-T08-P02-M002-B0", "S02"),
    tsentry("025D", "F01-T08-P02-M002-B0", "S02"),
    tsentry("025E", "F01-T08-P02-M002-B0", "S02"),
    tsentry("026", "F01-T10-P02-M002-B1", "S02"),
    tsentry("027", "F08-T12-P02-M002-B1", "S02"),
    tsentry("028", "F01-T01-P02-M002-B0", "S02"),
    tsentry("029", "F01-T01-P02-M002-B0", "S02"),
    tsentry("030", "F09-T09-P02-M002-B0", "S02"),
    tsentry("031", "F09-T09-P02-M002-B1", "S02"),
    tsentry("032", "F09-T09-P02-M002-B1", "S02"),
    tsentry("033", "F09-T01-P02-M002-B1", "S02"),
    tsentry("034", "F09-T07-P02-M002-B1", "S02"),
    tsentry("035", "F10-T14-P02-M002-B1", "S02"),
    tsentry("036", "F11-T15-P02-M002-B1", "S02"),
    tsentry("037", "F09-T09-P02-M002-B1", "S02"),
    tsentry("038", "F09-T16-P02-M002-B1", "S03"),
    tsentry("038b", "F09-T16-P02-M002-B1", "S03"),
    tsentry("039", "F12-T17-P02-M002-B1", "S02"),
    tsentry("040", "F12-T18-P02-M002-B1", "S02"),
    tsentry("041", "F13-T19-P02-M002-B1", "S02"),
    tsentry("042", "F14-T20-P02-M002-B1", "S02"),
    tsentry("043", "F15-T21-P02-M002-B1", "S02"),
    tsentry("044", "F14-T22-P02-M002-B1", "S02"),
    tsentry("045", "F14-T23-P02-M002-B1", "S02"),
    tsentry("046", "F09-T09-P02-M002-B1", "S03"),  # favorite
    tsentry("047", "F09-T09-P02-M002-B1", "S04"),
    tsentry("048", "F09-T09-P02-M002-B1", "S05"),
    tsentry("049", "F09-T09-P02-M002-B1", "S06"),
    tsentry("050", "F09-T24-P02-M002-B1", "S08"),
    tsentry("051", "F10-T25-P02-M002-B1", "S08"),
    tsentry("052", "F09-T24-P02-M008-B1", "S08"),
    tsentry("053", "F10-T25-P02-M008-B1", "S08"),
    tsentry("054", "F16-T16-P02-M008-B1", "S03"),  # = 038 but with F16=F09 with normalized gradients and model008 learning rate
    tsentry("055", "F16-T09-P02-M008-B1", "S03"),  # = 046 but with F16=F09 with normalized gradients and model008 learning rate
    tsentry("056", "F16-T24-P02-M008-B1", "S08"),  # = 052 but with F16=F09 with normalized gradients
    tsentry("057", "F17-T25-P02-M008-B1", "S08"),  
    tsentry("058", "F18-T26-P02-M008-B1", "S08"),  

    tsentry("059", "F19-T24-P02-M008-B1", "S08"),
    tsentry("060", "F20-T25-P02-M008-B1", "S08"),
    tsentry("061", "F21-T26-P02-M008-B1", "S08"),

    tsentry("062", "F19-T24-P02-M011-B1", "S08"),
    tsentry("063", "F20-T25-P02-M011-B1", "S08"),
    tsentry("064", "F21-T26-P02-M011-B1", "S08"),

    tsentry("101", "F09-T16-P02-M002-B1", "S03"),
    tsentry("102", "F09-T16-P02-M002-B1", "S07"),
    tsentry("103", "F09-T16-P02-M002-B1", "S08"),
])

#endregion Registries

#region BoundsConfig

boundstargetsconfig01(window) = Targets.Bounds01(window)

"Bounds estimator for short term limits"
boundsmk001config() = (configname="001", featconfig = boundsf6config01(5), targetconfig = boundstargetsconfig01(5), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

"Bounds estimator for mid term limits"
boundsmk002config() = (configname="002", featconfig = boundsf6config01(4*60), targetconfig = boundstargetsconfig01(4*60), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

const BOUNDS_ESTIMATOR_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    boundsmk001config(), boundsmk002config(),
])

#endregion BoundsConfig

#region Accessors

"Return the registry entry `ref` of `configs` (case insensitive) after stripping the first matching prefix of `prefixes`."
function _config_from_dict(configs::AbstractDict{String}, ref::AbstractString; label::AbstractString="config", prefixes=("",))
    raw = strip(String(ref))
    lowerraw = lowercase(raw)
    for prefix in prefixes
        if !isempty(prefix) && startswith(lowerraw, lowercase(prefix)) && (length(raw) > length(prefix))
            raw = strip(raw[(length(prefix) + 1):end])
            lowerraw = lowercase(raw)
            break
        end
    end
    matches = [key for key in keys(configs) if lowercase(key) == lowerraw]
    @assert length(matches) == 1 "unknown $(label) config ref=$(ref); available confignames=$(join(sort!(collect(keys(configs)); by=lowercase), ", "))"
    return configs[only(matches)]
end

"Return a fresh (unbased) Features006 config of the features id `featuresid`."
featureconfig(featuresid::AbstractString) = _config_from_dict(FEATURE_CONFIGS, featuresid; label="features")()
"Return the Features006 factory of the features id `featuresid`."
featurefactory(featuresid::AbstractString)::Function = _config_from_dict(FEATURE_CONFIGS, featuresid; label="features")
"Return a fresh (unbased) target config of the targets id `targetsid`."
targetconfig(targetsid::AbstractString) = _config_from_dict(TARGET_CONFIGS, targetsid; label="targets")()
"Return the partition config NamedTuple of the partition id `partitionid`."
partitionconfig(partitionid::AbstractString)::NamedTuple = _config_from_dict(PARTITION_CONFIGS, partitionid; label="partition")()
"Return the coin set `(coins, maxepochs)` of the coin set id `coinsetid`."
coinsetconfig(coinsetid::AbstractString)::NamedTuple = _config_from_dict(COINSET_CONFIGS, coinsetid; label="coinset")
"Return the date range `(startdt, enddt)` of the date range id `daterangeid`."
daterangeconfig(daterangeid::AbstractString)::NamedTuple = _config_from_dict(DATERANGE_CONFIGS, daterangeid; label="daterange")
"Return the NN model factory `(featurecount, labels, mnemonic) -> NN` of the model id `modelid`."
modelconfig(modelid::AbstractString)::Function = _config_from_dict(MODEL_CONFIGS, modelid; label="model")
"Return the StrategyConfig template (without classifier) of the strategy id `strategyid`."
strategytemplate(strategyid::AbstractString)::StrategyConfig = _config_from_dict(STRATEGY_CONFIGS, strategyid; label="strategy")()
"Return the classifier config of the classifier base name `basename`."
classifierconfig(basename::AbstractString)::NamedTuple = _config_from_dict(CLASSIFIER_CONFIGS, basename; label="classifier")
"Return the trading strategy context `(configname, classifier, strategy)` for `ref`."
tsconfig(ref::AbstractString)::NamedTuple = _config_from_dict(TS_CONFIGS, ref; label="trading strategy", prefixes=("tradingstrategy", "ts", "mk"))
boundsestimatorconfig(ref::AbstractString) = _config_from_dict(BOUNDS_ESTIMATOR_CONFIGS, ref; label="bounds", prefixes=("boundsestimator", "boundsmk", "bounds", "mk"))

"Coin set id used by default: test mode trains and evaluates on the artificial test coins."
defaultcoinset()::String = EnvConfig.configmode == EnvConfig.test ? "CS00" : "CS01"
"Date range id used by default: test mode uses a short recent date range."
defaultdaterange()::String = EnvConfig.configmode == EnvConfig.test ? "D00" : "D01"

"Return the hex SHA1 content hash of a canonical config description."
confighash(description::AbstractString)::String = bytes2hex(SHA.sha1(String(description)))
"Return the content hash of the features id `featuresid`."
featureshash(featuresid::AbstractString)::String = confighash(Features.configdescription(featureconfig(featuresid)))
"Return the content hash of the targets id `targetsid`."
targetshash(targetsid::AbstractString)::String = confighash(Targets.configdescription(targetconfig(targetsid)))
"Return the per-coin sets store id of a partition id and a date range id."
setsid(partitionid::AbstractString, daterangeid::AbstractString)::String = "$(partitionid)_$(daterangeid)"
"Return the canonical description of a partition and date range; explicit formatting keeps it independent of `show` methods of loaded packages."
function setsdescription(partitionid::AbstractString, daterangeid::AbstractString)::String
    p = partitionconfig(partitionid)
    dr = daterangeconfig(daterangeid)
    return "samplesets=$(join(p.samplesets, ","));partitionsize=$(p.partitionsize);gapsize=$(p.gapsize);minpartitionsize=$(p.minpartitionsize);maxpartitionsize=$(p.maxpartitionsize);startdt=$(Dates.format(dr.startdt, dateformat"yyyy-mm-ddTHH:MM:SS"));enddt=$(Dates.format(dr.enddt, dateformat"yyyy-mm-ddTHH:MM:SS"))"
end
setshash(partitionid::AbstractString, daterangeid::AbstractString)::String = confighash(setsdescription(partitionid, daterangeid))
"Return the content hash of the coin list of the coin set id `coinsetid`."
coinsethash(coinsetid::AbstractString)::String = confighash(join(coinsetconfig(coinsetid).coins, ","))

"Return the classifier labels in class index order of the classifier config `clcfg`."
classifierlabels(clcfg::NamedTuple)::Vector{Targets.TradeLabel} = Targets.uniquelabels(targetconfig(clcfg.targets))

"Return the classifier input feature columns of the classifier config `clcfg`."
classifierfeaturecols(clcfg::NamedTuple)::Vector{String} = Features.requestedcolumns(featureconfig(clcfg.features))

"Return the expected classifier meta data (component ids and content hashes) that identifies the trained classifier."
function classifiermeta(clcfg::NamedTuple; coinset::AbstractString, daterange::AbstractString)::Dict{String, String}
    return Dict(
        "name" => classifiername(clcfg; coinset=coinset, daterange=daterange),
        "features" => clcfg.features, "featureshash" => featureshash(clcfg.features),
        "targets" => clcfg.targets, "targetshash" => targetshash(clcfg.targets),
        "sets" => setsid(clcfg.partition, daterange), "setshash" => setshash(clcfg.partition, daterange),
        "model" => clcfg.model, "classbalancing" => string(clcfg.classbalancing),
        "coinset" => String(coinset), "coinsethash" => coinsethash(coinset),
        "labels" => join(string.(classifierlabels(clcfg)), ","),
        "featurecols" => join(classifierfeaturecols(clcfg), ","),
    )
end

"Assert that the stored meta data of the classifier `classifiername` matches `expected`; report the artifact path, active mode and mode-selection guidance when meta data is missing."
function checkclassifiermeta(classifiername::AbstractString, expected::AbstractDict{String, String})
    stored = Classify.readmeta(classifiername)
    @assert !isnothing(stored) "classifier $(classifiername) has no meta data at $(joinpath(EnvConfig.classifierfolder(classifiername), "meta.arrow")) (mode=$(EnvConfig.configmode)). Test mode defaults to CS00/D00 and reads debug classifiers; train mode defaults to CS01/D01 and reads normal classifiers. If TrendDetector adapted the classifier in train mode, run tradesim with the train flag and the same coinset/daterange as the adaptation. In tradesim, train selects trained artifacts; it does not retrain or enable live trading. Replay coins/startdt/enddt do not select the classifier's training identity."
    mismatches = [key for key in keys(expected) if get(stored, key, "") != expected[key]]
    @assert isempty(mismatches) "classifier $(classifiername) meta data mismatch for $(mismatches): stored=$(Dict(k => get(stored, k, "") for k in mismatches)) expected=$(Dict(k => expected[k] for k in mismatches))"
    return stored
end

"Return the classifier artifact name of a trading strategy context for `coinset` and `daterange`."
tsclassifiername(tscfg::NamedTuple; coinset::AbstractString=defaultcoinset(), daterange::AbstractString=defaultdaterange())::String =
    classifiername(classifierconfig(tscfg.classifier); coinset=coinset, daterange=daterange)

"Load the adapted classifier of a classifier config after asserting its meta data."
function loadtrendclassifier(clcfg::NamedTuple; coinset::AbstractString=defaultcoinset(), daterange::AbstractString=defaultdaterange())::Classify.TrendClassifier001
    name = classifiername(clcfg; coinset=coinset, daterange=daterange)
    checkclassifiermeta(name, classifiermeta(clcfg; coinset=coinset, daterange=daterange))
    return Classify.loadclassifier(name; featconfig=featurefactory(clcfg.features))
end

"Return a `StrategyConfig` for a trading strategy context: the strategy template of `tscfg.strategy` with the adapted classifier of `tscfg.classifier`."
function strategyconfig(tscfg::NamedTuple; coinset::AbstractString=defaultcoinset(), daterange::AbstractString=defaultdaterange())::StrategyConfig
    classifier = loadtrendclassifier(classifierconfig(tscfg.classifier); coinset=coinset, daterange=daterange)
    return _strategy_with_classifier(strategytemplate(tscfg.strategy), classifier)
end

"Return a `StrategyConfig` for a trading strategy context reference."
strategyconfig(ref::AbstractString; coinset::AbstractString=defaultcoinset(), daterange::AbstractString=defaultdaterange())::StrategyConfig =
    strategyconfig(tsconfig(ref); coinset=coinset, daterange=daterange)

"Return the source tag used when wiring a trading strategy context into Trade."
tsconfigsource(tscfg::NamedTuple; prefix::AbstractString="tradingstrategy")::String = "$(String(prefix)):$(tscfg.configname)/$(tscfg.classifier)/$(tscfg.strategy)"

#endregion Accessors
