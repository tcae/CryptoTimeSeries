"""
This script measures the performace on a trinaing set of cryptocurrencies when buying at the begin of an up slope and selling at the end of an up slope at various regression time window lenghts.

- Use 24h as main investment guideline and add performance if volatility around a positive regression gradient exceeds threshold to add buy and sell within that volatility.
  Assumption: Thresholds of such approach shall be significantly higher than 1%.

  What are the gain distributioins per regression window? Result: longer regression windows have a broader distribution and are less focused on the small <1% gains, i.e. also more extreme gains.

  Thesis: a regression qualifies if its low/high gain exceeds y% of threshold (e.g. 1%) and the next longer regression is positive.
  Distribution of regression gain > 1% in relation to last gain per regression and in relation to longer positive regression y/n?
  The same for losses: per regression window distribution of regression loss < -0.5% in relation to last loss

  include 2min regression

  2 windows (e.g. 5min, 15min) deviating in the same direction with increasing gradient is a signal to follow. Gradient thesholds may to be considered.

 aspect1 thesis: maximum slope gradients are correlated with trade control. The higher the gradient the shorter teh regression window.

 aspect2:  label strategy: if the real upslope gain is larger than x% then label according to that slope with real maximum backtrack.
  assumption: the higher the gain threshold the less frequent gains but the more reliable gain trade signals.

  to be investigated: is a larger window trend supervisor beneficial? assumption = yes
  consider factor 6 between windows: 5min, 30min, 3h, 18h, 36h (1,5d)

  approach: use the last n=3 predecessor slopes to determine by majority vote how the mnext slope likely works out in terms of gain
do that for different windows and switch after each sell to the best performing window

Break out handling approach: if smaller regression windows are within x* standard deviation (rollstd) or within a fixed threshold (e.g. 1%) with higher likelihood that the change is significant then follow the shorter outlier.
    As soon as the shortest outlier is again within the x * standard deviation of a longer then switch back to that one.

Consider out of spread deviations down from a reached price level as trade criteria.

    """
using Test, DataFrames, NamedArrays, Dates
using EnvConfig, Ohlcv, TestOhlcv, Features, Targets

fee = 0.001  # 0.1%


"""
Returns an array of price gains between a buy at slope begin and sell at slope end.
"""
function gradientgains(prices, regressions)
    # @info "gradientgains" size(prices, 1), size(regressions, 1)
    @assert size(prices, 1) == size(regressions, 1)
    gains = zeros(Float32, size(prices, 1))
    lastix = ix = 1
    nextix = Features.nextextremeindex(regressions, lastix)
    while nextix > 0
        if (lastix < nextix) && (regressions[lastix] > 0)
            gains[ix] = Ohlcv.relativegain(prices, lastix, nextix) - 2 * fee
            ix += 1
        end
        lastix = nextix
        nextix = Features.nextextremeindex(regressions, lastix)
    end
    return gains[1:ix-1]
end

"""
Returns an array of price gains between a buy at slope begin and sell at slope end.
"""
function gradientextremesindex(prices, regressions)
    # @info "gradientgains" size(prices, 1), size(regressions, 1)
    gains = zeros(Int32, size(prices, 1))
    lastix = ix = 1
    nextix = Features.nextextremeindex(regressions, lastix)
    while nextix > 0
        if regressions[lastix] > 0
            gains[ix] = Ohlcv.relativegain(prices, lastix, nextix) - 2 * fee
            ix += 1
        end
        lastix = nextix
        nextix = Features.nextextremeindex(regressions, lastix)
    end
    return gains[1:ix]
end

"""
This function uses a fixed regression window with a single base to buy at slope begin and sell at slope end.
"""
function singlebasegradientgain(prices, regressions)
    @assert size(prices, 1) == size(regressions, 1)
    gains = zeros(Float32, size(prices, 1))
    gains[1] = 1.0  # start: 1 USDT
    lastix = 1
    for ix in 2:size(prices, 1)
        if regressions[ix] > 0.0
            if regressions[ix - 1] <= 0.0  # start of upslope
                lastix = ix
                gains[ix] = gains[ix - 1] * (1 - fee)  # buy
            else
                gains[ix] = gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix))
            end
        else  # regressions[ix] <= 0.0
            if regressions[ix - 1] > 0.0  # start of downslope
                gains[ix] = gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix) - fee)  # sell
            else
                gains[ix] = gains[ix - 1]
            end
        end
    end
    return gains
end

"""
This function uses a fixed regression window with a single base to buy at slope begin and sell at slope end.
Only buy on upslope if lastgain > lastupgainthreshold.
If upgtdown == true then only buy if also lastgain > lastloss.
"""
function singlebasegradientgainhistory(prices, regressions; lastupgainthreshold, upgtdown::Bool)
    @assert size(prices, 1) == size(regressions, 1)
    gains = zeros(Float32, size(prices, 1))
    gains[1] = 1.0  # start: 1 USDT
    lastix = 0
    gldf = Features.lastgainloss(prices, regressions)
    for ix in 2:size(prices, 1)
        if regressions[ix] > 0.0
            if regressions[ix - 1] <= 0.0  # start of upslope
                if (gldf[ix, :lastgain] > lastupgainthreshold) && ((!upgtdown) || (gldf[ix, :lastgain] > abs(gldf[ix, :lastloss])))
                    lastix = ix
                    gains[ix] = gains[ix - 1] * (1 - fee)  # buy
                else
                    lastix = 0  # indicating no upslope to consider as gain
                    gains[ix] = gains[ix - 1]
                end
            else
                gains[ix] = lastix > 0 ? gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix)) : gains[ix - 1]
            end
        else  # regressions[ix] <= 0.0
            if (regressions[ix - 1] > 0.0 ) && (lastix > 0) # end of gain considered upslope
                gains[ix] = gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix) - fee)  # sell
            else
                gains[ix] = gains[ix - 1]
            end
        end
    end
    return gains
end

"""
This function uses a fixed regression window with a single base to buy at slope begin and sell at slope end.
Only buy on upslope if lastgain > lastupgainthreshold.
If upgtdown == true then only buy if also lastgain > lastloss.
"""
function lastregressionamplitudes(regressions, nbramplitudes=2)
    modlimit = nbramplitudes+2
    xtreme = ones(Int32, (modlimit-1))
    xix = 1

    function xixoffset(offset)
        return (xix + offset + modlimit) % modlimit
    end

    function xtremeoffset(offset)
        return xtreme[xixoffset(offset)]
    end

    up = zeros(Float32, size(regressions[:prices], 1))
    down = zeros(Float32, size(regressions[:prices], 1))
    lastix = 0
    for ix in 2:size(regressions,1)
        down[ix] = down[ix-1]
        up[ix] = up[ix-1]
        if (regressions[ix-1] <= 0) && (regressions[ix] > 0)  # start up slope
            xix = xixoffset(1)
            xtreme[xix] = ix
            down[ix] = (regressions[:prices][ix] - regressions[:prices][xtremeoffset(-1)]) / regressions[:prices][xtremeoffset(-1)]
        end
        if (regressions[ix-1] >= 0) && (regressions[ix] < 0)  # start down slope
            xix = xixoffset(1)
            xtreme[xix] = ix
            up[ix] = (regressions[:prices][ix] - regressions[:prices][xtremeoffset(-1)]) / regressions[:prices][xtremeoffset(-1)]
        end
    end
    return up, down
end

# """
# This function uses a fixed regression window with a single base to buy at slope begin and sell at slope end.
# Only buy on upslope if lastgain > lastupgainthreshold.
# If upgtdown == true then only buy if also lastgain > lastloss.
# """
# function multibasegradientgainhistory(prices, regressionminutes; lastupgainthreshold)
#     gains = zeros(Float32, size(prices, 1))
#     gains[1] = 1.0  # start: 1 USDT
#     lastix = 0
#     regr = Dict()
#     for regrminutes in regressionminutes
#         regr[regrminutes] = Dict()
#         regr[regrminutes][:prices], regr[regrminutes][:gradient] = Features.normrollingregression(prices, regrminutes)
#     end
#     control = regressionminutes[end]
#     for ix in 2:size(prices, 1)
#         if regressions[ix] > 0.0
#             if regressions[ix - 1] <= 0.0  # start of upslope
#                 if (gldf[ix, :lastgain] > lastupgainthreshold) && ((!upgtdown) || (gldf[ix, :lastgain] > abs(gldf[ix, :lastloss])))
#                     lastix = ix
#                     gains[ix] = gains[ix - 1] * (1 - fee)  # buy
#                 else
#                     lastix = 0  # indicating no upslope to consider as gain
#                     gains[ix] = gains[ix - 1]
#                 end
#             else
#                 gains[ix] = lastix > 0 ? gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix)) : gains[ix - 1]
#             end
#         else  # regressions[ix] <= 0.0
#             if (regressions[ix - 1] > 0.0 ) && (lastix > 0) # end of gain considered upslope
#                 gains[ix] = gains[lastix] * (1 + Ohlcv.relativegain(prices, lastix, ix) - fee)  # sell
#             else
#                 gains[ix] = gains[ix - 1]
#             end
#         end
#     end
#     return gains
# end

function maxgradient(regressions::DataFrame, bases, rowix)
    maxgrad = 0.0
    maxgradix = 1
    for (bix, base) in enumerate(bases)
        if (bix == 1) || (maxgrad < regressions[rowix, base])
            maxgrad = regressions[rowix, base]
            maxgradix = bix
        end
    end
    return (maxgradix, maxgrad)
end

"""
This function will always change to the currency with steepest gradient to optimize gain.
prices, regressions are all dataframes with equal bases as columns. All regressions are based on the same regressioin minutes.

Idea: only switch when 1%, 5% 10% better than the current base (introduces a hysteresis and avoid high frequent changes)
"""
function steepestbasegain(prices::DataFrame, regressions::DataFrame, bases)
    gains = zeros(Float32, size(prices[!, bases[1]], 1))
    bestix = zeros(Int8, size(gains, 1))
    bestix[1], maxgrad = maxgradient(regressions, bases, 1)
    gains[1] = 1.0  # start: 1 USDT
    lastix = 1
    for rix in 2:size(gains, 1)
        bestix[rix], maxgrad = maxgradient(regressions, bases, rix)
        if regressions[rix, bases[bestix[rix]]] > 0.0
            if regressions[rix - 1, bases[bestix[rix - 1]]] <= 0.0  # start of upslope, no need to sell because last gradient was negative
                lastix = rix
                gains[rix] = gains[rix - 1] * (1 - fee)  # buy
            elseif bestix[rix - 1] == bestix[rix]
                gains[rix] = gains[lastix] * (1 + Ohlcv.relativegain(prices[!, bases[bestix[rix]]], lastix, rix))
            else  # (bestix[rix-1] != bestix[rix]) && (regressions[rix-1, bestix[rix-1]] > 0.0)
                gains[rix] = gains[lastix] * (1 + Ohlcv.relativegain(prices[!, bases[bestix[rix - 1]]], lastix, rix) - fee)  # sell to change currency
                gains[rix] = gains[rix] * (1 - fee)  # buy
                lastix = rix
            end
        else  # regressions[rix] <= 0.0
            if regressions[rix - 1, bases[bestix[rix - 1]]] > 0.0  # start of downslope
                gains[rix] = gains[lastix] * (1 + Ohlcv.relativegain(prices[!, bases[bestix[rix - 1]]], lastix, rix) - fee)  # sell
            else
                gains[rix] = gains[rix - 1]
            end
        end
    end
    return gains
end

"""
Measurement of all training data to always use the steepest gradient shows that this does not lead to an optimized result.
"""
#=
base ╲ regrmin │           5           15           30           60          240          720         1440         4320        12960
───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
btc            │    2.68f-43  2.21999f-28  4.57771f-15   3.89028f-8    0.0267389     0.280552      2.79424      2.79975      1.22937
xrp            │    2.62f-43  5.72351f-26  1.35658f-13   8.99405f-8   0.00797346     0.167137      1.25013     0.752855     0.924814
eos            │     2.9f-43  5.58948f-26  5.20058f-14    7.6781f-8     0.015268     0.126651      1.69915      1.01844      1.20973
bnb            │    2.72f-43  2.71842f-25   1.8376f-13    9.6841f-8    0.0144519     0.770767      6.70104      3.78419      1.66535
eth            │    3.11f-43  1.28499f-24  3.40796f-13   4.77021f-7    0.0155053     0.201327      2.02797      2.52807      1.59899
neo            │    3.08f-43  3.41745f-28  2.94916f-14   7.26838f-8    0.0197463     0.275406      1.30217      1.29236      1.00782
ltc            │    2.96f-43  7.93366f-27  2.43653f-13   2.18716f-7   0.00656795     0.212881     0.725985     0.519932      1.57691
trx            │     2.8f-43  6.64184f-24   2.3627f-12   5.58659f-7    0.0362146     0.106338      5.55083      1.36395      1.13959
steepest       │    2.35f-43     1.36f-43     1.14f-43  3.31587f-34  2.32478f-10  0.000533894      1.25244     0.562211     0.776625 =#
function steepesttrainingbasesgain()
    regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    pdf = DataFrame()
    for (bix, base) in enumerate(EnvConfig.trainingbases)
        # @info "" bix base
        # base = "xrp"
        @info "reading ohlcv $base"
        ohlcv = Ohlcv.read(base)
        pdf[!, base] = ohlcv.df.pivot
    end
    bases = copy(EnvConfig.trainingbases)
    bases = push!(bases, "steepest")
    perfs = NamedArray(zeros(Float32, (size(bases, 1), size(regressionminutesset, 1))), (bases, regressionminutesset), ("base", "regrmin"))
    for (rix, regrminutes) in enumerate(regressionminutesset)
        rdf = DataFrame()
        gdf = DataFrame()
        for (bix, base) in enumerate(bases)
            if base != "steepest"
                rdf[!, base] = Features.normrollingregression(pdf[!, base], regrminutes)
                gdf[!, base] = singlebasegradientgain(pdf[!, base], rdf[!, base])
            else
                gdf[!, base] = steepestbasegain(pdf, rdf, EnvConfig.trainingbases)
            end
            perfs[bix, rix] = gdf[end, base]
            @info "simple performance at regression $regrminutes for $base = $(perfs[bix, rix])"
        end
    end
    display(perfs)
end

"""
Measurement of all training data to always buy at slope begin and sell at slope end with a fixed regression window shows that this shows best results for all currencies with 24h=1440min regression window.
"""
# base ╲ regrmin │           5           15           30           60          240          720         1440         4320        12960
# ───────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# btc            │    2.68f-43  2.21999f-28  4.57771f-15   3.89028f-8    0.0267389     0.280552      2.79424      2.79975      1.22937
# xrp            │    2.62f-43  5.72351f-26  1.35658f-13   8.99405f-8   0.00797346     0.167137      1.25013     0.752855     0.924814
# eos            │     2.9f-43  5.58948f-26  5.20058f-14    7.6781f-8     0.015268     0.126651      1.69915      1.01844      1.20973
# bnb            │    2.72f-43  2.71842f-25   1.8376f-13    9.6841f-8    0.0144519     0.770767      6.70104      3.78419      1.66535
# eth            │    3.11f-43  1.28499f-24  3.40796f-13   4.77021f-7    0.0155053     0.201327      2.02797      2.52807      1.59899
# neo            │    3.08f-43  3.41745f-28  2.94916f-14   7.26838f-8    0.0197463     0.275406      1.30217      1.29236      1.00782
# ltc            │    2.96f-43  7.93366f-27  2.43653f-13   2.18716f-7   0.00656795     0.212881     0.725985     0.519932      1.57691
# trx            │     2.8f-43  6.64184f-24   2.3627f-12   5.58659f-7    0.0362146     0.106338      5.55083      1.36395      1.13959
# TODO: use train, eval,test split to cross check consistency of results in subsets.
function singletrainingbasesgradientgain()
    regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    perfs = NamedArray(zeros(Float32, (size(EnvConfig.trainingbases, 1), size(regressionminutesset, 1))), (EnvConfig.trainingbases, regressionminutesset), ("base", "regrmin"))
    for (bix, base) in enumerate(EnvConfig.trainingbases)
        # @info "" bix base
        # base = "xrp"
        @info "reading ohlcv $base"
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            # @info "" rix regr
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            gains = singlebasegradientgain(ohlcv.df.pivot, regr)
            # @info "simple performance at regression $regrminutes for $base = $(round(gains[end], digits=3))"
            @info "simple performance at regression $regrminutes for $base = $(gains[end])"
            # @debug "check" bix rix size(gains, 1)
            perfs[bix, rix] = gains[end]
        end
    end
    display(perfs)
end

function singletrainingbasesgradientgain()
    regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    perfs = NamedArray(zeros(Float32, (size(EnvConfig.trainingbases, 1), size(regressionminutesset, 1))), (EnvConfig.trainingbases, regressionminutesset), ("base", "regrmin"))
    for (bix, base) in enumerate(EnvConfig.trainingbases)
        # @info "" bix base
        # base = "xrp"
        @info "reading ohlcv $base"
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            # @info "" rix regr
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            gains = singlebasegradientgain(ohlcv.df.pivot, regr)
            # @info "simple performance at regression $regrminutes for $base = $(round(gains[end], digits=3))"
            @info "simple performance at regression $regrminutes for $base = $(gains[end])"
            # @debug "check" bix rix size(gains, 1)
            perfs[bix, rix] = gains[end]
        end
    end
    display(perfs)
end

"""
Measurement of all training data to buy at slope begin and sell at slope end with a fixed regression window if lastgain > 1% and lastgain > lastloss.
It shows that this shows best results for all currencies with 24h=1440min regression window and that threshold and lastgain>lastloss only in a few cases improved the result above 1.
It is also worth to note that lastgain seems to be negative sometimes, which is the case when the actual price jumps down much faster than the regression.

Results see: ../data/singletrainingbasesgradientgainhistory.data
"""
function singletrainingbasesgradientgainhistory()
    # regressionminutesset = [5, 15]  # , 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    thresholds = [-1000, 0.0, 0.005, 0.01]
    gainlosstest = [false, true]
    perfs = NamedArray(zeros(Float32, (size(EnvConfig.trainingbases, 1), size(regressionminutesset, 1), size(thresholds, 1), size(gainlosstest, 1))), (EnvConfig.trainingbases, regressionminutesset, thresholds, gainlosstest), ("base", "regrmin", "threshold", "gain>loss"))
    for (bix, base) in enumerate(EnvConfig.trainingbases)
        # @info "" bix base
        # base = "xrp"
        @info "reading ohlcv $base"
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            # @info "" rix regr
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            for (switchix, switch) in enumerate(gainlosstest)
                for (thresholdix, threshold) in enumerate(thresholds)
                    gains = singlebasegradientgainhistory(ohlcv.df.pivot, regr, lastupgainthreshold=threshold, upgtdown=switch)
                    # @info "simple performance at regression $regrminutes for $base = $(round(gains[end], digits=3))"
                    @info "simple performance at regression $regrminutes for $base = $(gains[end])"
                    # @debug "check" bix rix size(gains, 1)
                    perfs[bix, rix, thresholdix, switchix] = gains[end]
                end
            end
        end
    end
    for (bix, base) in enumerate(EnvConfig.trainingbases)
        for (switchix, switch) in enumerate(gainlosstest)
        # for (thresholdix, threshold) in enumerate(thresholds)
            # println("threshold=$threshold  switch=$switch")
            println("base=$base  switch=$switch")
            display(perfs[bix, :, :, switchix])
        end
        println(" ")
    end
end

"""
gainborders is a vector to map a price gain value to an array index and search function (Base.Sort.searchsortedfirst/searchsortedlast) to search index of a given gain or gradient.
"""
function preparegainborders(gainrange=0.1, gainstep=0.01)
    # gainrange = 0.1  # 10% = 0.1 considered OK to cover -5% .. +5% gain range
    # gainrange = 0.02  # 2% = 0.02 considered OK to decided about gradients for target label
    gainstep = 0.01
    gainborders = [g for g in (-gainrange / 2):gainstep:(gainrange / 2)]
    push!(gainborders, gainborders[end] * 10000)  # add very big gain that is surely not topped as last
    # println("gainborders=$gainborders  length(gainborders)=$(length(gainborders))")
    return gainborders
end

"""
Returns the next extreme or the last array index.
"""
function nextslope!(slopesix, regressions, startix)
    reglen = size(regressions, 1)
    if startix >= reglen
        return reglen
    end
    nextix = Features.nextextremeindex(regressions, startix)

    if nextix == 0
        nextix = reglen
    end
    if (startix < nextix) && (regressions[startix] > 0)
        laststartix, lastendix = size(slopesix, 1) > 0 ? slopesix[end] : (0,0)
        if lastendix < startix
            push!(slopesix, (startix, nextix))
        else
            @warn "unexpected slope overlap" laststartix lastendix startix nextix
            # pop!(slopesix)
            # push!(slopesix, (laststartix, nextix))
        end
    end
    return nextix
end

"""
Returns the sum of gains of slopes identified via an array `slopesix` of non overlapping (startix, endix) tuples. Only tuples / tuple parts are considered than are within fromix - toix.
"""
function gainfromix!(slopesix, prices, fromix, toix)
    @assert toix <= size(prices, 1)
    @assert fromix < toix
    gain = 0.0

    if size(slopesix, 1) > 0
        startix, endix = slopesix[1]
        while endix < fromix
            popfirst!(slopesix)  # remove slopes that are no longer needed
            startix, endix = size(slopesix, 1) > 0 ? slopesix[1] : (toix, toix)
        end
    end

    lastendix = 0
    for (startix, endix) in slopesix
        startix = startix < fromix ? fromix : startix
        endix = endix > toix ? toix : endix
        if startix < endix
            gain += Ohlcv.relativegain(prices, startix, endix) - 2 * fee
            if lastendix > startix
                @warn "lastendix=$lastendix > startix=$startix, endix=$endix"
                display(slopesix)
            end
            lastendix = endix
        # else  # can happen due to fromix / toix correction
        #     @error "startix >= endix" startix endix
        end
    end
    return gain
end

"""
Returns the price index from which gains of different regression windows are compared
"""
function spreadindex(slopesix, spreadrix, bestnextix)
    spreadsix = 1
    six = size(slopesix[spreadrix], 1)
    while six > 0
        startix, endix = slopesix[spreadrix][six]
        if endix > bestnextix
            six -= 1
        else
            spreadsix = startix  # spread index at start of first full slope of longest regression window
            break
        end
    end
    return spreadsix
end

function checkgains(prices, regressions, slopesix)
    ggains = gradientgains(prices, regressions)
    g = zeros(Float32, (size(slopesix, 1)))
    for (ix, (startix, endix)) in enumerate(slopesix)
        g[ix] = Ohlcv.relativegain(prices, startix, endix) - 2 * fee
    end
    g2 = gainfromix!(slopesix, prices, 1, size(prices, 1))
    println("len(ggains)=$(size(ggains, 1)), ggain=$(sum(ggains)), len(slopesix)=$(size(slopesix, 1)), len(g)=$(size(g, 1)), g=$(sum(g)), g2=$g2")
    # display(ggains[1:3])
    # display(ggains[end-3:end])
    # display(g[1:3])
    # display(g[end-3:end])
    # display(slopesix[end-1:end])
end

"""
Returns gain sum of the slopes of the best regression windows
"""
function bestgradientgain(bases, regressionminutesset, gainthresholds)
    println("gainthreshold=$gainthresholds bases: $bases")
    display(regressionminutesset)
    basegrad = [[[] for rix in regressionminutesset] for bix in bases]
    _, spreadrix = findmax(regressionminutesset)
    gainsum = NamedArray(zeros(Float32, (size(bases, 1), size(gainthresholds, 1))), (bases, gainthresholds), ("base", "gain threshold"))
    for (gix, gainthreshold) in enumerate(gainthresholds)
        for (bix, base) in enumerate(bases)
            ohlcv = Ohlcv.read(base)
            prices = ohlcv.df.pivot
            reglen = size(prices, 1)
            bestslopesix = []  # array of tuples (startix, endix)
                gains = zeros(Float32, size(regressionminutesset, 1))
            # lastix = ones(Int32, size(regressionminutesset, 1))
            nextix = ones(Int32, size(regressionminutesset, 1))
            slopesix = [[] for ix in regressionminutesset]
            for (rix, regrminutes) in enumerate(regressionminutesset)
                regry, grads = Features.rollingregression(prices, regrminutes)
                basegrad[bix][rix] = grads
                nextix[rix] = nextslope!(slopesix[rix], basegrad[bix][rix], nextix[rix])
            end
            bestrix = 0  # signals not valid
            bestlastix = bestspread = 1
            bestnextix = minimum(nextix)
            bestgain = 0.0
            step = 0
            while bestnextix <= reglen
                step += 1
                if (step % 10000) == 1
                    println("$(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")) $base ($gainthreshold) bestnextix=$bestnextix")
                end
                bestspread = spreadindex(slopesix, spreadrix, bestnextix)
                for (rix, regrminutes) in enumerate(regressionminutesset)
                    gains[rix] = gainfromix!(slopesix[rix], prices, bestspread, bestnextix)
                end
                bestgain, rix = findmax(gains)
                if bestrix == 0  # not investigated
                    if bestgain >= gainthreshold
                        if basegrad[bix][rix][bestnextix] > 0  # only consider when up slope
                            bestrix = rix  # only first best and not N best within 10% -> to be improved
                            bestlastix = bestnextix
                        end
                    end
                else  # invested
                    if (bestrix != rix) && (bestgain >= gainthreshold) && (bestgain >= (gainthreshold + gains[bestrix]))
                        bestrix = rix  # on the fly change of regression window without sell and buy
                    end
                    if (bestnextix == nextix[bestrix]) && (basegrad[bix][rix][bestnextix] <= 0) && (bestlastix < bestnextix) # end of up slope
                        push!(bestslopesix, (bestlastix, bestnextix))
                        # println("$bestlastix, $bestnextix")
                        bestrix = 0
                    end
                end
                if bestnextix == reglen
                    break  # exit while loop
                end
                for (rix, regrminutes) in enumerate(regressionminutesset)
                    if bestnextix >= nextix[rix]
                        nextix[rix] = nextslope!(slopesix[rix], basegrad[bix][rix], nextix[rix])
                    end
                end
                bestnextix = minimum(nextix)
                # println("bestnextix=$bestnextix")
            end
            # for (rix, regrminutes) in enumerate(regressionminutesset)
            #     println("check $base $regrminutes")
            #     checkgains(prices, basegrad[bix][rix], slopesix[rix])
            # end
            bestgain = gainfromix!(bestslopesix, prices, 1, reglen)
            # println("best gain slopes=$(size(bestslopesix, 1)), best gain sum = $bestgain")
            gainsum[bix, gix] = bestgain
        end
    end
    display(gainsum)
    # return bestgain
end


"""
Collects all gains and losses per base per regression window in a histogram.

Results see: ../data/gainperregressionwindow
"""
function gradientgainhisto(bases, regressionminutesset)
    gainborders = preparegainborders(0.16)
    gainsum = NamedArray(zeros(Float32, (size(bases, 1), size(regressionminutesset, 1))), (bases, regressionminutesset), ("base", "window"))
    histo = NamedArray(zeros(Int32, (size(bases, 1), size(regressionminutesset, 1), size(gainborders, 1))), (bases, regressionminutesset, gainborders), ("base", "window", "gain"))
    for (bix, base) in enumerate(bases)
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            regry, grads = Features.rollingregression(ohlcv.df.pivot, regrminutes)
            gains = gradientgains(ohlcv.df.pivot, grads)
            for gain in gains
                gainsum[bix, rix] += gain
                gix = searchsortedlast(gainborders, gain) + 1
                histo[bix, rix, gix] += 1
            end
        end
    end
    println("gainborders=$gainborders")
    for (rix, regrminutes) in enumerate(regressionminutesset)
        println("window=$regrminutes")
        display(histo[:,rix,:])
    end
    for (bix, base) in enumerate(bases)
        println("base=$base")
        display(histo[bix,:,:])
    end
    display(gainsum)
end

"""
Collects all gainsand losses per base per regression window in a histogram.

Results see: ../data/gainperregressionwindow
"""
function gainperregressionwindow(bases = EnvConfig.trainingbases)
    # regressionminutesset = [5, 15]  # , 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60]
    gainborders = preparegainborders()
    histo = NamedArray(zeros(Float64, (size(bases, 1), size(regressionminutesset, 1), size(gainborders, 1))), (bases, regressionminutesset, gainborders), ("base", "window", "gain"))
    for (bix, base) in enumerate(bases)
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            buyix = Features.nextextremeindex(regr, 1)
            sellix = buyix > 0 ? Features.nextextremeindex(regr, buyix) : 0
            while sellix !=0
                gain = Ohlcv.relativegain(ohlcv.df.pivot, buyix, sellix)
                gix = searchsortedlast(gainborders, gain) + 1
                histo[bix, rix, gix] += 1
                buyix = sellix
                sellix = Features.nextextremeindex(regr, buyix)
                #! WRONG  2* Features.nextextremeindex(regr, buyix) required
            end
        end
    end
    println("gainborders=$gainborders")
    for (rix, regrminutes) in enumerate(regressionminutesset)
        for (bix, base) in enumerate(bases)
            histo[bix, rix, :] = (histo[bix, rix, :] ./ sum(histo[bix, rix, :])) * 100
        end
        map!(x -> round(x, digits=3), histo, histo)
        println("window=$regrminutes")
        display(histo[:,rix,:])
    end
end


"""
Collects all gains and losses per base per regression window per last gain in a histogram.
Result: very low if any correlation with predecessor gain. That may be related that the predecessor gain is measured with actual prices at regression extremes. Especially with larger regression windows the actual price can be already significnat deviate from the slope direction.

Results see: ../data/gainperregressionwindowlastgain
"""
function gainperregressionwindowlastgain(bases = EnvConfig.trainingbases, regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60])
    gainborders = preparegainborders(0.02, 0.01)
    histo = NamedArray(zeros(Float64, (size(regressionminutesset, 1), size(gainborders, 1), size(gainborders, 1))), (regressionminutesset, gainborders, gainborders), ("window", "gain", "lastgain"))
    for (bix, base) in enumerate(bases)
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            lastgix1 = lastgix2 = searchsortedlast(gainborders, 0.0) + 1
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            buyix = Features.nextextremeindex(regr, 1)
            sellix = buyix > 0 ? Features.nextextremeindex(regr, buyix) : 0
            while sellix !=0
                gain = Ohlcv.relativegain(ohlcv.df.pivot, buyix, sellix)
                gix = searchsortedlast(gainborders, gain) + 1
                histo[rix, gix, lastgix2] += 1
                lastgix2 = lastgix1
                lastgix1 = gix
                buyix = sellix
                sellix = Features.nextextremeindex(regr, buyix)
            end
        end
    end
    println("gainborders=$gainborders")
    for (rix, regrminutes) in enumerate(regressionminutesset)
        println("window=$regrminutes with $(sum(histo[rix, :, :])) slopes")
        histo[rix, :, :] = (histo[rix, :, :] ./ sum(histo[rix, :, :])) * 100
        map!(x -> round(x, digits=3), histo, histo)
        display(histo[rix,:, :])
    end
end


"""
Collects all gains and losses per base per regression window per last gain in a histogram.
Result: very low if any correlation with predecessor gain. That may be related that the predecessor gain is measured with actual prices at regression extremes. Especially with larger regression windows the actual price can be already significnat deviate from the slope direction.

Results see: ../data/gainperregressionwindowlastgain
"""
function gainperregressionwindowlastgain2(bases = EnvConfig.trainingbases, regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60, 3 * 24 * 60, 9 * 24 * 60])
    #! TODO why are there negative predecessor gains for upslopes and vice versa?
    gainborders = preparegainborders(0.02, 0.01)
    println("gainborders=$gainborders")
    histo = NamedArray(zeros(Float64, (size(regressionminutesset, 1), size(gainborders, 1), size(gainborders, 1))), (regressionminutesset, gainborders, gainborders), ("window", "gain", "lastgain"))
    for (bix, base) in enumerate(bases)
        ohlcv = Ohlcv.read(base)
        for (rix, regrminutes) in enumerate(regressionminutesset)
            regr = Features.normrollingregression(ohlcv.df.pivot, regrminutes)
            buyix = Features.nextextremeindex(regr, 1)
            sellix = buyix > 0 ? Features.nextextremeindex(regr, buyix) : 0
            lastbuyix = lastsellix = hix = lix = 1
            low = high = ohlcv.df.pivot[1]
            while sellix != 0
                if regr[lastbuyix] > 0  # upslope
                    high, hix = findmax(ohlcv.df.pivot[lastbuyix:lastsellix])
                else
                    low, lix = findmin(ohlcv.df.pivot[lastbuyix:lastsellix])
                end
                lastgain = hix > lix ? (high - low) / low : (low - high) / high
                lastgix = searchsortedlast(gainborders, lastgain) + 1

                if regr[buyix] > 0  # upslope
                    high, hix = findmax(ohlcv.df.pivot[buyix:sellix])
                else
                    low, lix = findmin(ohlcv.df.pivot[buyix:sellix])
                end
                gain = hix > lix ? (high - low) / low : (low - high) / high
                gix = searchsortedlast(gainborders, gain) + 1
                if gix > size(gainborders, 1)
                    @warn "gix out of bounds" hix lix high low gain
                end

                lastbuyix = buyix
                lastsellix = sellix
                # gain = Ohlcv.relativegain(ohlcv.df.pivot, buyix, sellix)
                # gix = searchsortedlast(gainborders, gain) + 1
                histo[rix, gix, lastgix] += 1
                buyix = sellix
                sellix = Features.nextextremeindex(regr, buyix)
            end
        end
    end
    for (rix, regrminutes) in enumerate(regressionminutesset)
        println("window=$regrminutes with $(sum(histo[rix, :, :])) slopes")
        histo[rix, :, :] = (histo[rix, :, :] ./ sum(histo[rix, :, :])) * 100
        map!(x -> round(x, digits=3), histo, histo)
        display(histo[rix,:, :])
    end
end


function prettyprint(prices, regressions, gains)
    df = DataFrame(prices=prices, regressions=regressions, gains=gains)
    display(df)
end

function singlebasegradientgain_test()
    enddt = DateTime("2022-04-02T01:00:00")
    startdt = enddt - Dates.Minute(40)
    ohlcv = TestOhlcv.sinedata(20, 40, 0, 2)
    regr5 = Features.normrollingregression(ohlcv.df.pivot, 5)
    gains = singlebasegradientgain(ohlcv.df.pivot, regr5)
    # prettyprint(ohlcv.df.pivot, regr5, gains)
    return gains[end]
end

function steepestbasegain_test()
    shortsine = TestOhlcv.sinedata(20, 2, -1)
    longsine = TestOhlcv.sinedata(40, 1, 2)
    pdf = DataFrame(shortsine=shortsine.df.pivot, longsine=longsine.df.pivot)
    # show(pdf, allrows=true)
    shortsineregr = Features.normrollingregression(shortsine.df.pivot, 2)
    longsineregr = Features.normrollingregression(longsine.df.pivot, 2)
    rdf = DataFrame(shortsine=shortsineregr, longsine=longsineregr)
    # show(rdf, allrows=true)
    gdf = DataFrame()
    gdf[:, :shortsine] = singlebasegradientgain(shortsine.df.pivot, shortsineregr)
    gdf[:, :longsine] = singlebasegradientgain(longsine.df.pivot, longsineregr)
    gdf[:, :steepest] = steepestbasegain(pdf, rdf, ["shortsine", "longsine"])
    # show(gdf, allrows=true)
    # println("steepest gain=$(gdf[end, :steepest])")
    return gdf[end, :steepest]
end


# Config.init(Config.test)
EnvConfig.init(EnvConfig.production)
println("\nconfig mode = $(EnvConfig.configmode)")
# gainperregressionwindow()
# gainperregressionwindowlastgain2()
# gainperregressionwindowlastgain(["btc", "xrp"], [5, 15])

# @info simplegains_test()
# singletrainingbasesgradientgain()
# singletrainingbasesgradientgainhistory()
# steepesttrainingbasesgain()

# regressionminutesset = [18 * 60, 24 * 60, 30 * 60, 36 * 60]
regressionminutesset = [5, 15, 30, 60, 4 * 60, 12 * 60, 24 * 60]
# regressionminutesset = [60, 24 * 60]
# regressionminutesset = [24 * 60]
bases = EnvConfig.trainingbases
# bases = ["bnb"]
# gradientgainhisto(bases, regressionminutesset)
println("")
bestgradientgain(bases, regressionminutesset, [-1000.0, 0.0, 0.01, 0.02])
# @testset "simple gain performance" begin

#     @test isapprox(steepestbasegain_test(), 1.0195855)
#     @test isapprox(singlebasegradientgain_test(), 1.006738275018905)

# end  # of testset


