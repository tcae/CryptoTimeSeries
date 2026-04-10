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

tradingstrategy01() = TradingStrategy.GainSegment(maxwindow=4*60, algorithm=TradingStrategy.algorithm01!, openthreshold=0.6, closethreshold=0.5, makerfee=0.0015, takerfee=0.002)
tradingstrategy02() = TradingStrategy.GainSegment(maxwindow=4*60, algorithm=TradingStrategy.algorithm02!, openthreshold=0.6, closethreshold=0.5, makerfee=0.0015, takerfee=0.002)

# Trend01/Trend02 were replaced by Trend04.
trend04targetconfig(minwindow, maxwindow, buy, hold) = Targets.Trend04(minwindow, maxwindow, Targets.thresholds((longbuy=buy, longhold=hold, shorthold=-hold, shortbuy=-buy)))

targetconfig01() = trend04targetconfig(10, 4*60, 0.01, 0.01)
targetconfig02() = trend04targetconfig(10, 4*60, 0.05, 0.03)
targetconfig03() = trend04targetconfig(10, 4*60, 0.02, 0.01)
targetconfig04() = trend04targetconfig(10, 4*60, 0.007, 0.005)
targetconfig05() = trend04targetconfig(0, 4*60, 0.01, 0.01)
targetconfig06() = trend04targetconfig(2, 4*60, 0.01, 0.01)
targetconfig07() = trend04targetconfig(10, 4*60, 0.01, 0.005)
targetconfig08() = trend04targetconfig(10, 2*60, 0.01, 0.005) # Trend02 replaced by Trend04
targetconfig09() = trend04targetconfig(10, 2*60, 0.01, 0.01)
targetconfig10() = trend04targetconfig(10, 1*60, 0.01, 0.01)
targetconfig11() = trend04targetconfig(10, 2*60, 0.01, 0.005) # Trend04
targetconfig12() = trend04targetconfig(10, 24*60, 0.05, 0.03) # Trend04 with long term target

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
    Features.addmindist!(featcfg, window=12*60, offset=0)
    Features.addmaxdist!(featcfg, window=24*60, offset=0)
    Features.addmindist!(featcfg, window=3*24*60, offset=0)
    Features.addrelvol!(featcfg, short=15, long=3*24*60, offset=0)
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

"""
mk1 = mix adapted with multiple adaptations per coin with **good results**: ppv(longbuy) = 72%
"""
mk001config() = (configname="001", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk2 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk002config() = (configname="002", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk3 = one iteration adaptation with one merged set (allclose=>0.9, longbuy=>0.1), but target config targetconfig02() with **poor results**: ppv(longbuy) = 26%
"""
mk003config() = (configname="003", featconfig = trendf6config01(), targetconfig = targetconfig02(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk4 = targetconfig03() with one merged set (allclose=>0.726, longbuy=>0.274)
"""
mk004config() = (configname="004", featconfig = trendf6config01(), targetconfig = targetconfig03(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk5 = targetconfig04() with one merged set (allclose=>?, longbuy=>?)
"""
mk005config() = (configname="005", featconfig = trendf6config01(), targetconfig = targetconfig04(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk6 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk006config() = (configname="006", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
same as mk6 butwith copied mix classifier from mk2
mk7 = mix adapted in just one iteration with all coin features/targets in one set, which is a fairer class representation in the adaptation, (allclose=>0.494, longbuy=>0.506) with **good results**: ppv(longbuy) = 73%
"""
mk007config() = (configname="007", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model001, tradingstrategy=tradingstrategy02())

"""
mk8 = mix adapted with all coin features/targets in one set, features are clipped, normalized, shifted, and in addition batch norm layer after initial layer with relu activation in model001
equal mean, q25, q75, min, max does not look like healthy feature values - longbuy ppv classification performance is with close to 70% also worse
"""
# mk008config() = (configname="008", featconfig = trendf6config02(), targetconfig = targetconfig01(), classifiermodel=Classify.model001)

""" **my favorite**  
mk9 = mix adapted with all coin features/targets in one set, features are not clipped, batch norm layer before and between layers with relu activation in model002
"""
mk009config() = (configname="009", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=true)

"""
mk10 = mix adapted with all coin features/targets in one set, features are clipped, initial batch norm layer in model002
"""
# mk010config() = (configname="010", featconfig = trendf6config02(), targetconfig = targetconfig01(), classifiermodel=Classify.model002)

"""
mk11 = mix adapted with all coin features/targets in one set, no clipping, initial batch norm layer but no further internal batch norm layers
"""
mk011config() = (configname="011", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model003, tradingstrategy=tradingstrategy02())

"""
mk12 = like mk9 but with an additional layer
"""
mk012config() = (configname="012", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model004, tradingstrategy=tradingstrategy02())

"""
mk13 = like mk9 but with 4/3 broader layers
"""
mk013config() = (configname="013", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model005, tradingstrategy=tradingstrategy02())

"""
mk14 = like mk11 but removed layer 3
"""
mk014config() = (configname="014", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model006, tradingstrategy=tradingstrategy02())

"""
mk15 = like mk11 but reduced number of nodes of layers by reducing factor of layer 1 from 3 to 2
"""
mk015config() = (configname="015", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model007, tradingstrategy=tradingstrategy02())

"""
mk16 = no tolerance against target disturbances (minwindow=0), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk016config() = (configname="016", featconfig = trendf6config01(), targetconfig = targetconfig05(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""
mk17 = short tolerance against target disturbances (minwindow=2), the rest is the same as mk9: mix adapted with all coin features/targets in one set, features are not clipped, batch norm before and between layers with relu activation in model002
"""
mk017config() = (configname="017", featconfig = trendf6config01(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk18 = mk9 but with simplified features
"""
mk018config() = (configname="018", featconfig = trendf6config03(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk19 = mk9 but with simplified features
"""
mk019config() = (configname="019", featconfig = trendf6config04(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk20 = mk17 but feature grads without offset
"""
mk020config() = (configname="020", featconfig = trendf6config05(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk21 = mk17 but feature grads without offset and 12h regr grad added
"""
mk021config() = (configname="021", featconfig = trendf6config06(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk22 = mk21 but with 3*24h regr grad added
"""
mk022config() = (configname="022", featconfig = trendf6config07(), targetconfig = targetconfig06(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())
#* from mk23 onwards partitionconfig02() shall be used with a fixed partition gap of 8h to get the same targets gains for the same targetconfig but different requiredminutes(features)

"""  
mk023 = equal to mk9 with the only difference that hold thresholds are lowered to 0.5% instead of being equal to the buy threshold of 1%
"""
mk023config() = (configname="023", featconfig = trendf6config01(), targetconfig = targetconfig07(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""  
mk024 = equal to mk23 but Trend01/Trend02 were replaced by Trend04 targets
"""
mk024config() = (configname="024", featconfig = trendf6config01(), targetconfig = targetconfig08(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""
mk9 = mk9 with short term target, i.e. maxwindow 2h
"""
mk025config() = (configname="025", featconfig = trendf6config01(), targetconfig = targetconfig09(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())
mk025bconfig() = (configname="025b", featconfig = trendf6config01(), targetconfig = targetconfig09(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)
mk025Cconfig() = (configname="025C", featconfig = trendf6config01(), targetconfig = targetconfig08(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)
mk025Dconfig() = (configname="025D", featconfig = trendf6config01(), targetconfig = targetconfig11(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)
# mk25D and mk25E are the same config but in teh meanwhile the missing hold was fixed
mk025Econfig() = (configname="025E", featconfig = trendf6config01(), targetconfig = targetconfig11(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)

"""
mk026 = mk9 with short term target, i.e. maxwindow 1h
"""
mk026config() = (configname="026", featconfig = trendf6config01(), targetconfig = targetconfig10(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""
mk27 = long term trend with long term window and 5% ambition
"""
mk027config() = (configname="027", featconfig = trendf6config08(), targetconfig = targetconfig12(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02())

"""   
mk028 = mk009 without oversampling
"""
mk028config() = (configname="028", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)

"""   
mk029 = mk009 without oversampling but implementation for hold equally strict than for buy (no config change)
"""
mk029config() = (configname="029", featconfig = trendf6config01(), targetconfig = targetconfig01(), classifiermodel=Classify.model002, tradingstrategy=tradingstrategy02(), oversampling=false)

#endregion TrendConfig

#region BoundsConfig
"Bounds estimator for short term limits"
boundsmk001config() = (configname="001", featconfig = boundsf6config01(5), targetconfig = boundstargetsconfig01(5), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

"Bounds estimator for mid term limits"
boundsmk002config() = (configname="002", featconfig = boundsf6config01(4*60), targetconfig = boundstargetsconfig01(4*60), regressormodel=Classify.boundsregressor001, tradingstrategy=tradingstrategy02())

#endregion BoundsConfig

#region AdviceConfig

"Trade advice LSTM baseline config for phase smoothing over TrendDetector outputs."
tradeadvicemk001config() = (
    configname="001",
    trendconfigref="029",
    seqlen=15,
    hidden_dim=32,
    maxepoch=200,
    batchsize=64,
    openthresholds=Float32[0.8f0, 0.7f0, 0.6f0],
    closethresholds=Float32[0.6f0, 0.5f0],
)

"Alternative eval-sweep with a slightly longer sequence and tighter thresholds for comparison on the eval split."
tradeadvicemk002config() = (
    configname="002",
    trendconfigref="025E",
    seqlen=5,
    hidden_dim=48,
    maxepoch=200,
    batchsize=64,
    openthresholds=Float32[0.85f0, 0.75f0, 0.65f0],
    closethresholds=Float32[0.65f0, 0.6f0, 0.55f0],
)

#endregion AdviceConfig

"""
    _config_from_dict(configs, ref; label, prefixes)

Resolve a config reference by its `configname` from one of the shared preset dicts.
Returns a `deepcopy` so each caller gets an isolated mutable configuration payload.
"""
function _config_from_dict(configs::AbstractDict{String, <:NamedTuple}, ref::AbstractString; label::AbstractString, prefixes::Tuple)
    raw = strip(replace(String(ref), r"config$"i => ""))
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
    return deepcopy(configs[only(matches)])
end

const TREND_DETECTOR_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    mk001config(), mk002config(), mk003config(), mk004config(), mk005config(), mk006config(), mk007config(),
    mk009config(), mk011config(), mk012config(), mk013config(), mk014config(), mk015config(), mk016config(),
    mk017config(), mk018config(), mk019config(), mk020config(), mk021config(), mk022config(), mk023config(),
    mk024config(), mk025config(), mk025bconfig(), mk025Cconfig(), mk025Dconfig(), mk025Econfig(), mk026config(),
    mk027config(), mk028config(), mk029config(),
])

const BOUNDS_ESTIMATOR_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    boundsmk001config(), boundsmk002config(),
])

const TREND_LSTM_CONFIGS = Dict{String, NamedTuple}(cfg.configname => cfg for cfg in [
    tradeadvicemk001config(), tradeadvicemk002config(),
])

trenddetectorconfig(ref::AbstractString) = _config_from_dict(TREND_DETECTOR_CONFIGS, ref; label="trend", prefixes=("trenddetector", "trend", "mk"))
boundsestimatorconfig(ref::AbstractString) = _config_from_dict(BOUNDS_ESTIMATOR_CONFIGS, ref; label="bounds", prefixes=("boundsestimator", "boundsmk", "bounds", "mk"))
trendlstmconfig(ref::AbstractString) = _config_from_dict(TREND_LSTM_CONFIGS, ref; label="TrendLstm", prefixes=("trendlstm", "tradeadvicelstm", "tradeadvice", "tradeadvicemk", "mk"))

