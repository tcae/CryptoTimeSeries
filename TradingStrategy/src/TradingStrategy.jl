module TradingStrategy

using DataFrames, Logging, JDF
using Dates, DataFrames
using EnvConfig, Ohlcv, CryptoXch, Classify

"""
verbosity =
- 0: suppress all output if not an error
- 1: log warnings
- 2: load and save messages are reported
- 3: print debug info
"""
verbosity = 3

mutable struct TradeConfig
    cfg::AbstractDataFrame
    xc::Union{Nothing, CryptoXch.XchCache}
    function TradeConfig(xc::CryptoXch.XchCache)
        return new(DataFrame(), xc)
    end
end

MINIMUMDAYUSDTVOLUME = 10*1000000
TRADECONFIG_CONFIGFILE = "TradeConfig"

emptytradeconfig() = DataFrame(basecoin=String[], buysell=Bool[], sellonly=Bool[], update=DateTime[])

function continuousminimumvolume(ohlcv::Ohlcv.OhlcvData, enddt::Union{DateTime, Nothing}, checkperiod=Day(1), accumulateperiod=Minute(5), minimumaccumulatequotevolume=1000f0)::Bool
    enddt = isnothing(enddt) ? ohlcv.df[end, :opentime] : enddt
    endix = Ohlcv.rowix(ohlcv, enddt)
    startdt = enddt - checkperiod
    startix = Ohlcv.rowix(ohlcv, startdt)
    df = @view ohlcv.df[startix:endix, :]
    adf = Ohlcv.accumulate(df, accumulateperiod)
    accquotevol = adf[!, :basevolume] .* adf[!, :pivot]
    accquotevoltest = accquotevol .>= minimumaccumulatequotevolume
    if all(accquotevoltest)
        return true
    else
        (verbosity >= 3) && println("$(ohlcv.base) has in $(round((1 - (count(accquotevoltest)/length(accquotevoltest)))*100))% insuficient continuous $(accumulateperiod) minimum volume of $minimumaccumulatequotevolume $(EnvConfig.cryptoquote) over a period of $checkperiod ending $enddt")
        return false
    end
end

"""
Loads all USDT coins, checks last24h volume, checks minimum volume of every aggregated 5minutes, removes risk coins.
If isnothing(enddt) or enddt > last update then uploads latest OHLCV and calculates F4 of remaining coins that are then stored.
The resulting DataFrame table of tradable coins is stored.
assetbases is an input parameter to enable backtesting.
"""
function train!(tc::TradeConfig, assetbases::Vector; enddt=Dates.now(Dates.UTC), minimumdayquotevolume=MINIMUMDAYUSDTVOLUME)
    read!(tc, enddt)
    if size(tc.cfg, 1) > 0
        (verbosity >= 2) && println("read trade config from file")
        (verbosity >= 3) && println(tc.cfg)
        return
    end
    usdtdf = CryptoXch.getUSDTmarket(tc.xc)
    (verbosity >= 3) && println("USDT market: $usdtdf at $enddt")
    # assetbases = CryptoXch.assetbases(tc.xc)
    tradablebases = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :basecoin]
    tradablebases = [base for base in tradablebases if CryptoXch.validbase(tc.xc, base)]
    allbases = union(tradablebases, assetbases)
    allbases = setdiff(allbases, CryptoXch.baseignore)
    count = length(allbases)
    cld = Dict()
    for (ix, base) in enumerate(allbases)
        (verbosity >= 2) && print("\r$(EnvConfig.now()) start updating $base ($ix of $count)      ")
        startdt = enddt - Year(10)
        ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        Ohlcv.write(ohlcv)
        cl = Classify.Classifier001(ohlcv)
        if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
            cld[base] = cl
            Classify.write(cld[base])
        elseif base in assetbases
            @warn "skipping asset $base because classifier features cannot be calculated"
        end
    end
    startdt = enddt - Day(10)
    (verbosity >= 2) && print("\r$(EnvConfig.now()) start classifier set training")
    cfg = Classify.trainset!(collect(values(cld)), startdt, enddt, true)
    tradablebases = intersect(Classify.trainsetminperf(cfg)[!, :basecoin], tradablebases)
    # tradablebases = [base for base in tradablebases if continuousminimumvolume(cld[base].ohlcv, enddt)]
    sellonlybases = setdiff(assetbases, tradablebases)
    allbases = union(tradablebases, sellonlybases)
    usdtdf = filter(row -> row.basecoin in allbases, usdtdf)
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent)
    tc.cfg[:, :buysell] = [base in tradablebases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :sellonly] = [base in sellonlybases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :regrwindow] .= missing
    tc.cfg[:, :gainthreshold] .= missing
    tc.cfg[:, :update] .= enddt
    tc.cfg[:, :classifier] .= missing
    for tcix in eachindex(tc.cfg[!, :basecoin])
        if tc.cfg[tcix, :basecoin] in keys(cld)
            cl = cld[tc.cfg[tcix, :basecoin]]
            tc.cfg[tcix, :regrwindow] = cl.regrwindow
            tc.cfg[tcix, :gainthreshold] = cl.gainthreshold
            tc.cfg[tcix, :classifier] = cl
        else
            @warn "unexpected missing classifier for basecoin $(tc.cfg[tcix, :basecoin])"
        end
    end
    # (verbosity >= 3) && println(describe(tc.cfg, :all))
    write(tc, enddt)
    return tc
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
    cfgfilename = EnvConfig.logpath(TRADECONFIG_CONFIGFILE)
    if isnothing(timestamp)
        cfgfilename = join([cfgfilename, ext], ".")
    else
        cfgfilename = join([cfgfilename, Dates.format(timestamp, "yy-mm-dd"), ext], "_", ".")
    end
    return cfgfilename
end

"if timestamp=nothing then no extension otherwise timestamp extension"
function write(tc::TradeConfig, timestamp::Union{Nothing, DateTime}=nothing)
    if (size(tc.cfg, 1) == 0) || (tc.cfg[tc.cfg[!, :active] .== true, :] == 0)
        @warn "trade config is empty or without active bases - not stored"
        return
    end
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(timestamp)
    # EnvConfig.checkbackup(cfgfilename)
    (verbosity >=3) && println("cfgfilename=$cfgfilename  tc.cfg=$(tc.cfg)")
    JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifer)])
    if isnothing(timestamp)
        cfgfilename = _cfgfilename(Dates.now(UTC))
        JDF.savejdf(cfgfilename, tc.cfg[!, Not(:classifer)])
    end
    EnvConfig.setlogpath(sf)
end

function read!(tc::TradeConfig, startdt, enddt)
    df = DataFrame()
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(timestamp, "jdf")
    if isdir(cfgfilename)
        df = DataFrame(JDF.loadjdf(cfgfilename))
        println("config df: $df")
        if isnothing(df)
            (verbosity >=2) && println("Loading $cfgfilename failed")
        end
    end
    EnvConfig.setlogpath(sf)
    df[:, :classifier] .= missing
    for ix in df[!, :basecoin]
        ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        cl = Classify.Classifier001(ohlcv)
        if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
            df[ix, :classifier] = cl
        else
            @warn "skipping asset $(df[ix, :basecoin]) because classifier features cannot be calculated"
        end
    end
    tc.cfg = df
    return tc
end

end # module
