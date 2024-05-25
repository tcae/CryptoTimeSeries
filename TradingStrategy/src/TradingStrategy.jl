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
TRADECONFIGFILE = "TradeConfig"

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
        (verbosity >= 3) && println("\r$(ohlcv.base) has ($(count - countok) of $count) in $(round(((count - countok) / count) * 100.0))% insuficient continuous $(accumulateminutes) minimum volume of $minimumaccumulatequotevolume $(EnvConfig.cryptoquote) over a period of $checkperiod ending $datetime")
        return false
    end
end

"""
Loads all USDT coins, checks last24h volume and other continuous minimum volume criteria, removes risk coins.
If isnothing(datetime) or datetime > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
`assetonly` is an input parameter to enable backtesting.
"""
function tradeselection!(tc::TradeConfig, assetbases::Vector; datetime=Dates.now(Dates.UTC), minimumdayquotevolume=MINIMUMDAYUSDTVOLUME, assetonly=false, updatecache=false)
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
        if updatecache
            (verbosity >= 2) && print("\r$(EnvConfig.now()) updating $base ($ix of $count) including cache update                           ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", datetime - Year(10), datetime)
            Ohlcv.write(ohlcv)
            cl = Classify.BaseClassifier001(ohlcv)
            if !isnothing(cl)
                Classify.write(cl)
                Classify.timerangecut!(cl, datetime - Minute(Classify.requiredminutes()-1), datetime)
            end
        else
            (verbosity >= 2) && print("\r$(EnvConfig.now()) updating $base ($ix of $count)                                                  ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", datetime - Minute(Classify.requiredminutes()-1), datetime)
            cl = Classify.BaseClassifier001(ohlcv)
        end
        if !isnothing(cl) # else Ohlcv history may be too short to calculate sufficient features
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
    (verbosity >= 2) && print("\r$(EnvConfig.now()) finished updating $count bases                                            ")
    startdt = datetime - Day(3)  # Day(10)
    (verbosity >= 2) && print("\r$(EnvConfig.now()) start classifier set training                                             ")
    cfg = Classify.trainset!(collect(values(cld)), startdt, datetime, true) #TODO to be removed from TradingStrategy -> internal to Classify
    (verbosity >= 4) && println("cld=$(collect(values(cld))) trainset! cfg=$cfg")
    if assetonly
        (verbosity >= 4) && println("tradablebases=$(size(cfg, 1) > 0 ? intersect(cfg[!, :basecoin], tradablebases) : []) = intersect($(cfg[!, :basecoin]), tradablebases=$tradablebases)")
        tradablebases = size(cfg, 1) > 0 ? intersect(cfg[!, :basecoin], tradablebases) : []
    else
        (verbosity >= 4) && println("tradablebases=$(setdiff(tradablebases, skippedbases)) = setdiff(tradablebases=$tradablebases, skippedbases=$skippedbases)")
        tradablebases = setdiff(tradablebases, skippedbases)
        trainsetminperfdf = Classify.trainsetminperf(cfg, 1.5)  # 0.5% gain per day #TODO to be removed from TradingStrategy -> select only on liquidity criteria
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
        #TODO don't merge all data into one dataframe - each module should manage their own dataframe synced via timestamp and orderid
    end
    tc.cfg = leftjoin(tc.cfg, cldf, on = :basecoin)
    (verbosity >= 3) && println("$(CryptoXch.ttstr(tc.xc)) result of TrainingStrategy.tradeselection! $(tc.cfg)")
    if !assetonly
        write(tc, datetime)
    end
    (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) trained and saved trade config data including $(size(tc.cfg, 1)) base classifier (ohlcv, features) data      ")
    return tc
end

function addtradeconfig(tc::TradeConfig, cl::Classify.AbstractClassifier)
    tcfg = (basecoin=cl.ohlcv.base, quotevolume24h_M=1f0, pricechangepercent=10f0, buysell=true, sellonly=false, Classify.configuration(cl)..., classifier=cl)
    #TODO don't merge all data into one dataframe - each module should manage their own dataframe synced via timestamp and orderid
    push!(tc.cfg, tcfg)
end

"""
tradeselection! loads all data up to enddt. This function will reduce the memory starting from startdt plus OHLCV data required for feature calculation.
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
        cfgfilename = TRADECONFIGFILE
    else
        cfgfilename = join([TRADECONFIGFILE, Dates.format(timestamp, "yy-mm-dd")], "_")
    end
    return EnvConfig.datafile(cfgfilename, TRADECONFIGFILE, ".jdf")
end

"Saves the trade configuration. If timestamp!=nothing then save 2x with and without timestamp in filename otherwise only without timestamp"
function write(tc::TradeConfig, timestamp::Union{Nothing, DateTime}=nothing)
    if (size(tc.cfg, 1) == 0)
        @warn "trade config is empty - not stored"
        return
    end
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(nothing)
    # EnvConfig.checkbackup(cfgfilename)
    if isdir(cfgfilename)
        rm(cfgfilename; force=true, recursive=true)
    end
    (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
    JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifier)])
    if !isnothing(timestamp)
        cfgfilename = _cfgfilename(timestamp)
        (verbosity >=3) && println("saving trade config in cfgfilename=$cfgfilename")
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
function read!(tc::TradeConfig, datetime=nothing)
    tc = readconfig!(tc, datetime)
    df = nothing
    if !isnothing(tc) && !isnothing(tc.cfg) && (size(tc.cfg, 1) > 0)
        clvec = []
        df = tc.cfg
        rows = size(df, 1)
        for ix in eachindex(df[!, :basecoin])
            (verbosity >= 2) && print("\r$(EnvConfig.now()) loading $(df[ix, :basecoin]) from trade config ($ix of $rows)                                                  ")
            ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", floor(datetime-Minute(Classify.requiredminutes()), Dates.Minute), floor(datetime, Dates.Minute))
            cl = Classify.BaseClassifier001(ohlcv)
            if !isnothing(cl) # else Ohlcv history may be too short to calculate sufficient features
                push!(clvec, cl)
            else
                @warn "skipping asset $(df[ix, :basecoin]) because classifier features cannot be calculated" cl=cl ohlcv=ohlcv f4=cl.f4
            end
        end
        (verbosity >= 2) && println("\r$(CryptoXch.ttstr(tc.xc)) loaded trade config data including $rows base classifier (ohlcv, features) data      ")
        df[:, :classifier] = clvec
    end
    return !isnothing(df) && (size(df, 1) > 0) ? tc : nothing
end

"Adds usdtprice and usdtvalue added as well as the portfolio dataframe to trade config and returns trade config and portfolio as tuple"
function addassetsconfig!(tc::TradeConfig)
    assets = CryptoXch.portfolio!(tc.xc)
    sort!(assets, [:coin])  # for readability only

    tc.cfg = leftjoin(tc.cfg, assets, on = :basecoin => :coin)
    tc.cfg = tc.cfg[!, Not([:locked, :free])]
    sort!(tc.cfg, [:basecoin])  # for readability only
    sort!(tc.cfg, rev=true, [:buysell])  # for readability only
    return tc.cfg, assets
end

"Returns the current TradeConfig dataframe with usdtprice and usdtvalue added as well as the portfolio dataframe as a tuple"
function assetsconfig!(tc::TradeConfig, datetime=nothing)
    tc = TradingStrategy.readconfig!(tc, datetime)
    return addassetsconfig!(tc)
end

end # module
