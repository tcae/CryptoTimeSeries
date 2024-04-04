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
function train!(tc::TradeConfig, assetbases::Vector; enddt=floor(Dates.now(Dates.UTC), Minute(1)), minimumdayquotevolume=MINIMUMDAYUSDTVOLUME, assetonly=false)
    read!(tc, enddt, enddt)
    if size(tc.cfg, 1) > 0
        (verbosity >= 3) && println(tc.cfg)
        return tc
    end
    usdtdf = CryptoXch.getUSDTmarket(tc.xc)
    if assetonly
        usdtdf = filter(row -> row.basecoin in assetbases, usdtdf)
    end
    (verbosity >= 3) && println("USDT market: $(describe(usdtdf, :all)) of size=$(size(usdtdf, 1)) at $enddt")
    # assetbases = CryptoXch.assetbases(tc.xc)
    tradablebases = usdtdf[usdtdf.quotevolume24h .>= minimumdayquotevolume, :basecoin]
    tradablebases = [base for base in tradablebases if CryptoXch.validbase(tc.xc, base)]
    allbases = union(tradablebases, assetbases)
    allbases = setdiff(allbases, CryptoXch.baseignore)
    count = length(allbases)
    cld = Dict()
    skippedbases = []
    for (ix, base) in enumerate(allbases)
        (verbosity >= 2) && print("\r$(EnvConfig.now()) updating $base ($ix of $count)                                                  ")
        startdt = enddt - Year(10)
        ohlcv = CryptoXch.cryptodownload(tc.xc, base, "1m", floor(startdt, Dates.Minute), floor(enddt, Dates.Minute))
        Ohlcv.write(ohlcv)
        cl = Classify.Classifier001(ohlcv)
        if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
            cld[base] = cl
            Classify.write(cld[base])
        elseif base in assetbases
            @warn "skipping asset $base because classifier features cannot be calculated"
            push!(skippedbases, base)
        end
    end
    (verbosity >= 2) && print("\r$(EnvConfig.now()) finished updating $count bases                                                  ")
    startdt = enddt - Day(10)
    (verbosity >= 2) && print("\r$(EnvConfig.now()) start classifier set training                                             ")
    cfg = Classify.trainset!(collect(values(cld)), startdt, enddt, true)
    trainsetminperfdf = Classify.trainsetminperf(cfg)
    (verbosity >= 4) && println("trainsetminperfdf=$trainsetminperfdf")
    tradablebases = size(trainsetminperfdf, 1) > 0 ? intersect(trainsetminperfdf[!, :basecoin], tradablebases) : []
    # tradablebases = [base for base in tradablebases if continuousminimumvolume(cld[base].ohlcv, enddt)]
    (verbosity >= 4) && println("tradablebases=$tradablebases")
    sellonlybases = setdiff(assetbases, tradablebases)
    (verbosity >= 4) && println("sellonlybases=$sellonlybases")
    allbases = union(tradablebases, sellonlybases)
    (verbosity >= 4) && println("allbases=$allbases, skippedbases=$skippedbases")
    allbases = setdiff(allbases, skippedbases)
    (verbosity >= 4) && println("allbases=$allbases")
    usdtdf = filter(row -> row.basecoin in allbases, usdtdf)
    tc.cfg = select(usdtdf, :basecoin, :quotevolume24h => (x -> x ./ 1000000) => :quotevolume24h_M, :pricechangepercent)
    if size(tc.cfg, 1) == 0
        @warn "no basecoins selected - empty result tc.cfg=$(tc.cfg)"
        return tc
    end
    tc.cfg[:, :buysell] = [base in tradablebases for base in tc.cfg[!, :basecoin]]
    tc.cfg[:, :sellonly] = [base in sellonlybases for base in tc.cfg[!, :basecoin]]
    # cldf = DataFrame(regrwindow=Int16[], gainthreshold=Float32[], model=String[], medianccbuycnt=Float32[], simgain=Float32[], update=DateTime[], classifier=Classify.Classifier001[])
    # for tcix in eachindex(tc.cfg[!, :basecoin])
    #     if tc.cfg[tcix, :basecoin] in keys(cld)
    #         cl = cld[tc.cfg[tcix, :basecoin]]
    #         push!(cldf, (cl.cfg[cl.bestix, :regrwindow], cl.cfg[cl.bestix, :gainthreshold], cl.cfg[cl.bestix, :model], cl.cfg[cl.bestix, :medianccbuycnt], cl.cfg[cl.bestix, :simgain], enddt, cl))
    #     else
    #         @error "unexpected missing classifier for basecoin $(tc.cfg[tcix, :basecoin])"
    #     end
    # end
    cldf1 = DataFrame()
    cldf2 = DataFrame()
    for cl in values(cld)
        push!(cldf1, cl.cfg[cl.bestix, :])
        push!(cldf2, (basecoin=cl.cfg[cl.bestix, :basecoin], classifier=cl))
    end
    (verbosity >= 3) && println("tc.cfg=$(tc.cfg)")
    (verbosity >= 3) && println("cldf1=$cldf1")
    (verbosity >= 3) && println("cldf2=$cldf2")
    tc.cfg = leftjoin(tc.cfg, cldf1, on = :basecoin)
    tc.cfg = leftjoin(tc.cfg, cldf2, on = :basecoin)
    (verbosity >= 3) && println(describe(tc.cfg, :all))
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
    if isnothing(timestamp)
        cfgfilename = TRADECONFIG_CONFIGFILE
    else
        cfgfilename = join([TRADECONFIG_CONFIGFILE, Dates.format(timestamp, "yy-mm-dd")], "_")
    end
    return EnvConfig.datafile(cfgfilename, "TradeConfig", ".jdf")
    # cfgfilename = EnvConfig.logpath(TRADECONFIG_CONFIGFILE)
    # if isnothing(timestamp)
    #     cfgfilename = join([cfgfilename, ext], ".")
    # else
    #     cfgfilename = join([cfgfilename, Dates.format(timestamp, "yy-mm-dd"), ext], "_", ".")
    # end
    # return cfgfilename
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

function read!(tc::TradeConfig, startdt, enddt)
    df = DataFrame()
    sf = EnvConfig.logsubfolder()
    EnvConfig.setlogpath(nothing)
    cfgfilename = _cfgfilename(startdt, "jdf")
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
                ohlcv = CryptoXch.cryptodownload(tc.xc, df[ix, :basecoin], "1m", floor(startdt-Minute(Classify.requiredminutes()), Dates.Minute), floor(enddt, Dates.Minute))
                cl = Classify.Classifier001(ohlcv)
                if !isnothing(cl.f4) # else Ohlcv history may be too short to calculate sufficient features
                    push!(clvec, cl)
                else
                    @warn "skipping asset $(df[ix, :basecoin]) because classifier features cannot be calculated" cl=cl ohlcv=ohlcv f4=cl.f4
                end
            end
            (verbosity >= 2) && println("\r$(EnvConfig.now()) loaded trade config data including $rows base classifier (ohlcv, features) data      ")
            df[:, :classifier] = clvec
        end
    end
    EnvConfig.setlogpath(sf)
    tc.cfg = df
    return tc
end

end # module
