module TradingStrategy

using DataFrames, Logging, JDF
using Dates, DataFrames
using EnvConfig, Ohlcv, CryptoXch, Classify, Features

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 2

mutable struct TradeConfig
    cfg::AbstractDataFrame
    xc::Union{Nothing, CryptoXch.XchCache}
    function TradeConfig(xc::CryptoXch.XchCache)
        return new(DataFrame(), xc)
    end
end

MINIMUMDAYUSDTVOLUME = 2*1000000
TRADECONFIG_CONFIGFILE = "TradeConfig"

function continuousminimumvolume(ohlcv::Ohlcv.OhlcvData, datetime::Union{DateTime, Nothing}, checkperiod=Day(1), accumulateminutes=5, minimumaccumulatequotevolume=1000f0)::Bool
    datetime = isnothing(datetime) ? ohlcv.df[end, :opentime] : datetime
    endix = Ohlcv.rowix(ohlcv, datetime)
    startdt = datetime - checkperiod
    startix = Ohlcv.rowix(ohlcv, startdt)
    vol = minimumaccumulatequotevolume
    count = countok = 0
    for ix in startix:endix
        if ((ix - startix) % accumulateminutes) == 0
            if ix > startix
                count += 1
                if vol >= minimumaccumulatequotevolume
                    countok += 1
                end
            end
            vol = ohlcv.df[end, :basevolume] * ohlcv.df[end, :pivot]
        else
            vol += ohlcv.df[end, :basevolume] * ohlcv.df[end, :pivot]
        end
    end
    if count == countok
        return true
    else
        (verbosity >= 3) && println("$(ohlcv.base) has in $(round((1 - (countok / count)) * 100))% insuficient continuous $(accumulateminutes) minimum volume of $minimumaccumulatequotevolume $(EnvConfig.cryptoquote) over a period of $checkperiod ending $datetime")
        return false
    end
end

"""
Loads all USDT coins, checks last24h volume and other continuous minimum volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to enable backtesting.
"""
function train!(tc::TradeConfig, assetbases::Vector; datetime=Dates.now(Dates.UTC), minimumdayquotevolume=MINIMUMDAYUSDTVOLUME, assetonly=false)
    datetime = floor(datetime, Minute(1))

    # make memory available
    tc.cfg = DataFrame() # return stored config, if one exists from same day
    CryptoXch.removeallbases(tc.xc)

    usdtdf = CryptoXch.getUSDTmarket(tc.xc)  # superset of coins with 24h volume price change and last price
    if assetonly
        usdtdf = filter(row -> row.basecoin in assetbases, usdtdf)
    end
    (verbosity >= 3) && println("USDT market: $(describe(usdtdf, :all)) of size=$(size(usdtdf, 1)) at $datetime")
    tradablebases = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :basecoin]
    tradablebases = [base for base in tradablebases if CryptoXch.validbase(tc.xc, base)]
    allbases = union(tradablebases, assetbases)
    allbases = setdiff(allbases, CryptoXch.baseignore)

    # download latest OHLCV and classifier features
    count = length(allbases)
    cld = Dict()
    skippedbases = []
    for (ix, base) in enumerate(allbases)
        (verbosity >= 2) && print("\r$(EnvConfig.now()) updating $base ($ix of $count)                                                  ")
        ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", datetime - Year(10), datetime)
        Ohlcv.write(ohlcv)
        cl = Classify.Classifier001(ohlcv)
        if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
            Classify.write(cl)
            Classify.timerangecut!(cl, datetime - Day(10), datetime)
            cld[base] = cl
        elseif base in assetbases
            @warn "skipping asset $base because classifier features cannot be calculated"
            push!(skippedbases, base)
        end
    end

    # qualify buy+sell coins as tradablebases
    (verbosity >= 2) && print("\r$(EnvConfig.now()) finished updating $count bases                                                  ")
    startdt = datetime - Day(10)
    (verbosity >= 2) && print("\r$(EnvConfig.now()) start classifier set training                                             ")
    cfg = Classify.trainset!(collect(values(cld)), startdt, datetime, true)
    if assetonly
        tradablebases = size(cfg, 1) > 0 ? intersect(cfg[!, :basecoin], tradablebases) : []
    else
        trainsetminperfdf = Classify.trainsetminperf(cfg)
        (verbosity >= 4) && println("trainsetminperfdf=$trainsetminperfdf")
        performerbases = size(trainsetminperfdf, 1) > 0 ? intersect(trainsetminperfdf[!, :basecoin], tradablebases) : []
        tradablebases = [base for base in performerbases if continuousminimumvolume(cld[base].ohlcv, datetime)]
        (verbosity >= 4) && println("tradablebases=$tradablebases = setdiff(performerbases=$performerbases, insufficientcontinuousvolume=$(setdiff(performerbases, tradablebases))")
    end

    sellonlybases = setdiff(assetbases, tradablebases, skippedbases)
    (verbosity >= 4) && println("sellonlybases=$sellonlybases = setdiff(assetbases=$assetbases, tradablebases=$tradablebases, skippedbases=$skippedbases)")

    # create config DataFrame
    allbases = union(tradablebases, sellonlybases)
    (verbosity >= 4) && println("allbases=$allbases")
    usdtdf = filter(row -> row.basecoin in allbases, usdtdf)
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent)
    if size(tc.cfg, 1) == 0
        (verbosity >= 1) && @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :buysell] = [base in tradablebases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :sellonly] = [base in sellonlybases for base in tc.cfg[!, :basecoin]]
    cldf1 = DataFrame()
    cldf2 = DataFrame()
    for cl in values(cld)
        push!(cldf1, cl.cfg[cl.bestix, :])
        push!(cldf2, (basecoin=cl.cfg[cl.bestix, :basecoin], classifier=cl))
    end
    tc.cfg = leftjoin(tc.cfg, cldf1, on = :basecoin)
    tc.cfg = leftjoin(tc.cfg, cldf2, on = :basecoin)
    (verbosity >= 3) && println("$(CryptoXch.ttstr(tc.xc)) result of TrainingStrategy.train! $(tc.cfg)")
    write(tc, datetime)
    (verbosity >= 2) && println("\r$(EnvConfig.now())/$(CryptoXch.ttstr(tc.xc)) trained and saved trade config data including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    return tc
end

function emptytradeconfig()
    #TODO does not work with double column basecoin but leftjoin without data also not possible
    cfg = hcat(DataFrame(basecoin=String[], quotevolume24h_M=Float32[], pricechangepercent=Float32[]), Classify.emptyconfigdf(), DataFrame(classifier=[]))
end

function addtradeconfig(tc::TradeConfig, cl::Classify.AbstractClassifier)
    tcfg = DataFrame([(basecoin=cl.ohlcv.base, quotevolume24h_M=1f0, pricechangepercent=10f0, buysell=true, sellonly=false)])
    clcfg1 = DataFrame(cl.cfg[cl.bestix, :])  # Classify.addconfig!(cl, cl.ohlc.base, regrwindow, gainthreshold, model)
    tcfg = leftjoin(tcfg, clcfg1, on = :basecoin)
    clcfg2 = DataFrame([(basecoin=cl.ohlcv.base, classifier=cl)])
    tcfg = leftjoin(tcfg, clcfg2, on = :basecoin)
    push!(tc.cfg, tcfg[1, :])
end

"""
train! loads all data up to enddt. This function will reduce the memory starting from startdt plus OHLCV data required for feature calculation.
"""
function timerangecut!(tc::TradeConfig, startdt, enddt)
    for ohlcv in CryptoXch.ohlcv(tc.xc)
        tcix = findfirst(x -> x == ohlcv.base, tc.cfg[!, :basecoin])
        if isnothing(tcix)
            CryptoXch.removebase!(tc.xc, ohlcv.base)
        else
            cl = tc.cfg[tcix, :classifier]
            Classify.timerangecut!(cl, startdt, enddt)
        end
    end
end


function _cfgfilename(timestamp::Union{Nothing, DateTime}, ext="jdf")
    if isnothing(timestamp)
        cfgfilename = TRADECONFIG_CONFIGFILE
    else
        cfgfilename = join([TRADECONFIG_CONFIGFILE, Dates.format(timestamp, "yy-mm-dd")], "_")
    end
    return EnvConfig.datafile(cfgfilename, "TradeConfig", ".jdf")
end

"if timestamp=nothing then no extension otherwise timestamp extension"
function write(tc::TradeConfig, timestamp::Union{Nothing, DateTime}=nothing)
    if (size(tc.cfg, 1) == 0)
        @warn "trade config is empty - not stored"
        return
    end
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(timestamp)
    # EnvConfig.checkbackup(cfgfilename)
    (verbosity >=3) && println("cfgfilename=$cfgfilename  tc.cfg=$(tc.cfg)")
    JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifier)])
    if isnothing(timestamp)
        cfgfilename = _cfgfilename(Dates.now(UTC))
        JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifier)])
    end
    EnvConfig.setlogpath(sf)
end

"""
Will return the already stored trade strategy config, if filename from the same date exists. Also loads the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function read!(tc::TradeConfig, datetime)
    df = DataFrame()
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(datetime, "jdf")
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
        if isnothing(df)
            (verbosity >=2) && println("Loading $cfgfilename failed")
        else
            (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config from $cfgfilename")
            clvec = []
            rows = size(df, 1)
            for ix in eachindex(df[!, :basecoin])
                (verbosity >= 2) && print("\r$(EnvConfig.now()) loading $(df[ix, :basecoin]) from trade config ($ix of $rows)                                                  ")
                ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", floor(datetime-Minute(Classify.requiredminutes()), Dates.Minute), floor(datetime, Dates.Minute))
                cl = Classify.Classifier001(ohlcv)
                if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
                    push!(clvec, cl)
                else
                    @warn "skipping asset $(df[ix, :basecoin]) because classifier features cannot be calculated" cl=cl ohlcv=ohlcv f4=cl.f4
                end
            end
            (verbosity >= 2) && println("\r$(EnvConfig.now())/$(CryptoXch.ttstr(tc.xc)) loaded trade config data including $rows base classifier (ohlcv, features) data      ")
            df[:, :classifier] = clvec
        end
    end
    EnvConfig.setlogpath(sf)
    tc.cfg = df
    return size(df, 1) > 0 ? tc : nothing
end

end # module
