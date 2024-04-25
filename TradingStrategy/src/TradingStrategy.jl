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
    if size(ohlcv.df, 1) == 0
        (verbosity >= 4) && println("$(ohlcv.base) has an empty dataframe")
        return false
    end
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
            vol = ohlcv.df[ix, :basevolume] * ohlcv.df[ix, :pivot]
        else
            vol += ohlcv.df[ix, :basevolume] * ohlcv.df[ix, :pivot]
        end
    end
    if count == countok
        return true
    else
        (verbosity >= 3) && println("$(ohlcv.base) has ($(count - countok) of $count) in $(round(((count - countok) / count) * 100.0))% insuficient continuous $(accumulateminutes) minimum volume of $minimumaccumulatequotevolume $(EnvConfig.cryptoquote) over a period of $checkperiod ending $datetime")
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
    (verbosity >= 3) && println("USDT market of size=$(size(usdtdf, 1)) at $datetime")
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
        ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", datetime - Minute(Classify.requiredminutes()-1), datetime)
        cl = Classify.Classifier001(ohlcv)
        if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
            cld[base] = cl
        elseif base in assetbases
            @warn "skipping asset $base because classifier features cannot be calculated"
            push!(skippedbases, base)
        end
        if !continuousminimumvolume(ohlcv, datetime)
            push!(skippedbases, base)
        end
    end

    # qualify buy+sell coins as tradablebases
    (verbosity >= 2) && print("\r$(EnvConfig.now()) finished updating $count bases                                                  ")
    startdt = datetime - Day(3)  # Day(10)
    (verbosity >= 2) && print("\r$(EnvConfig.now()) start classifier set training                                             ")
    cfg = Classify.trainset!(collect(values(cld)), startdt, datetime, true)
    (verbosity >= 4) && println("cld=$(collect(values(cld))) trainset! cfg=$cfg")
    if assetonly
        (verbosity >= 4) && println("tradablebases=$(size(cfg, 1) > 0 ? intersect(cfg[!, :basecoin], tradablebases) : []) = intersect($(cfg[!, :basecoin]), tradablebases=$tradablebases)")
        tradablebases = size(cfg, 1) > 0 ? intersect(cfg[!, :basecoin], tradablebases) : []
    else
        (verbosity >= 4) && println("tradablebases=$(setdiff(tradablebases, skippedbases)) = setdiff(tradablebases=$tradablebases, skippedbases=$skippedbases)")
        tradablebases = setdiff(tradablebases, skippedbases)
        trainsetminperfdf = Classify.trainsetminperf(cfg, 1.5)  # 0.5% gain per day
        (verbosity >= 4) && println("trainsetminperfdf=$trainsetminperfdf")
        tradablebases = size(trainsetminperfdf, 1) > 0 ? intersect(trainsetminperfdf[!, :basecoin], tradablebases) : []
        (verbosity >= 4) && println("tradablebases=$tradablebases")
    end

    sellonlybases = setdiff(assetbases, tradablebases)
    (verbosity >= 4) && println("sellonlybases=$sellonlybases = setdiff(assetbases=$assetbases, tradablebases=$tradablebases)")

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
    cldf = DataFrame()
    for cl in values(cld)
        push!(cldf, (Classify.configuration(cl)..., simgain=cl.cfg[cl.bestix, :simgain], minsimgain=cl.cfg[cl.bestix, :minsimgain], medianccbuycnt=cl.cfg[cl.bestix, :medianccbuycnt], update=cl.cfg[cl.bestix, :enddt], classifier=cl))
    end
    tc.cfg = leftjoin(tc.cfg, cldf, on = :basecoin)
    (verbosity >= 3) && println("$(CryptoXch.ttstr(tc.xc)) result of TrainingStrategy.train! $(tc.cfg)")
    if !assetonly
        write(tc, datetime)
    end
    (verbosity >= 2) && println("\r$(EnvConfig.now())/$(CryptoXch.ttstr(tc.xc)) trained and saved trade config data including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    return tc
end

function addtradeconfig(tc::TradeConfig, cl::Classify.AbstractClassifier)
    tcfg = (basecoin=cl.ohlcv.base, quotevolume24h_M=1f0, pricechangepercent=10f0, buysell=true, sellonly=false, Classify.configuration(cl)..., classifier=cl)
    push!(tc.cfg, tcfg)
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
    (verbosity >=3) && println("saved trading config in cfgfilename=$cfgfilename")
    JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifier)])
    if isnothing(timestamp)
        cfgfilename = _cfgfilename(Dates.now(UTC))
        JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifier)])
    end
    EnvConfig.setlogpath(sf)
end

"""
Will return the already stored trade strategy config, if filename from the same date exists but does not load the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function readconfig!(tc::TradeConfig, datetime)
    df = DataFrame()
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(datetime, "jdf")
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
    end
    EnvConfig.setlogpath(sf)
    if !isnothing(df) && (size(df, 1) > 0 )
        (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config from $cfgfilename")
        tc.cfg = df
        return tc
    else
        (verbosity >=2) && !isnothing(df) && println("Loading $cfgfilename failed")
        return nothing
    end
end

"""
Will return the already stored trade strategy config, if filename from the same date exists. Also loads the ohlcv and classifier features.
If no trade strategy config can be loaded then `nothing` is returned.
"""
function read!(tc::TradeConfig, datetime)
    tc = readconfig!(tc, datetime)
    df = nothing
    if !isnothing(tc) && !isnothing(tc.cfg) && (size(tc.cfg, 1) > 0)
        clvec = []
        df = tc.cfg
        rows = size(df, 1)
        for ix in eachindex(df[!, :basecoin])
            (verbosity >= 2) && println("\r$(EnvConfig.now()) loading $(df[ix, :basecoin]) from trade config ($ix of $rows)                                                  ")
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
    return !isnothing(df) && (size(df, 1) > 0) ? tc : nothing
end

"Returns the current TradeConfig dataframe with usdtprice and usdtvalue added as well as the portfolio dataframe as a tuple"
function assetsconfig!(tc::TradeConfig)
    assets = CryptoXch.portfolio!(tc.xc)
    sort!(assets, [:coin])
    startdt = Dates.now(UTC)
    tc = TradingStrategy.readconfig!(tc, startdt)

    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:locked, :free])]
    sort!(tc.cfg, [:basecoin])
    sort!(tc.cfg, rev=true, [:buysell])
    return tc.cfg, assets
end

end # module
