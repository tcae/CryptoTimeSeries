#region FeatureStore

# Per-coin stores under `coins/<BASE-QUOTE>/`:
# - `regression/regr<window>.arrow`: opentime, regry, grad over the full OHLCV history
# - `regression/regrstd<window>.arrow`: opentime, std over the full OHLCV history
# - `features/<featuresid>.arrow`: opentime + requested Features006 columns without warm-up rows

regressionfolder(base::AbstractString, quotecoin::AbstractString) = EnvConfig.coinfolderpath(base, quotecoin, "regression")
featuresfolder(base::AbstractString, quotecoin::AbstractString) = EnvConfig.coinfolderpath(base, quotecoin, "features")
_regressionstem(window::Integer; std::Bool=false) = std ? "regrstd$(window)" : "regr$(window)"

"Return the canonical config-only description of `f6` that identifies its feature columns independent of any OHLCV base."
configdescription(f6::Features006)::String = "Features006(" * join(requestedcolumns(f6), ",") * ")"

"Return the regression windows of `f6` and the subset that requires a std column."
function regressionwindows(f6::Features006)
    regrfeatures = [f for f in f6all(f6) if f.f in ("ry", "rg", "rs")]
    return sort!(unique([f.w for f in regrfeatures])), Set(f.w for f in regrfeatures if f.f == "rs")
end

"""
    readregression(base, quotecoin, window; std=false)

Return the per-coin regression store of `window` as (memory mapped) DataFrame with columns
`opentime, regry, grad` (or `opentime, std` with `std=true`), or `nothing` if not stored.
"""
function readregression(base::AbstractString, quotecoin::AbstractString, window::Integer; std::Bool=false)
    return EnvConfig.readdf(_regressionstem(window; std=std); folderpath=regressionfolder(base, quotecoin), format=:arrow)
end

"Return the number of leading `stored` rows that equal the leading OHLCV opentimes, i.e. the reusable prefix."
function _reusableprefix(stored, ot::AbstractVector{DateTime})::Int
    (isnothing(stored) || (size(stored, 1) == 0) || (size(stored, 1) > length(ot))) && return 0
    sot = stored[!, :opentime]
    return (sot[begin] == ot[begin]) && (sot[end] == ot[length(sot)]) ? length(sot) : 0
end

"""
    updateregression!(ohlcv, window; std=false)

Bring the per-coin regression store of `window` up to date with the complete OHLCV history
behind `ohlcv` (the parent dataframe if `ohlcv` is a view). Only rows that are not yet stored
are calculated. With `std=true` also the regression std store is updated, which is expensive
for large windows and therefore only calculated on request.
"""
function updateregression!(ohlcv::Ohlcv.OhlcvData, window::Integer; std::Bool=false)
    odf = parent(Ohlcv.dataframe(ohlcv))
    @assert size(odf, 1) > 1 "regression store of $(ohlcv.base) requires more than 1 OHLCV row; got $(size(odf, 1))"
    ot = odf[!, :opentime]
    folder = regressionfolder(ohlcv.base, ohlcv.quotecoin)
    stored = readregression(ohlcv.base, ohlcv.quotecoin, window)
    prefixlen = _reusableprefix(stored, ot)
    if prefixlen < length(ot)
        ry, rg = rollingregression(odf[!, :pivot], window, prefixlen + 1)
        if prefixlen > 0
            ry = vcat(stored[!, :regry], ry)
            rg = vcat(stored[!, :grad], rg)
        end
        @assert length(ry) == length(rg) == length(ot) "regression window=$window of $(ohlcv.base): length(ry)=$(length(ry)) length(rg)=$(length(rg)) != length(ot)=$(length(ot))"
        EnvConfig.savedf(DataFrame(opentime=copy(ot), regry=ry, grad=rg), _regressionstem(window); folderpath=folder, format=:arrow)
        (verbosity >= 3) && println("$(EnvConfig.now()) updated regression store window=$window of $(ohlcv.base) from row $(prefixlen + 1) until $(length(ot))")
        stored = readregression(ohlcv.base, ohlcv.quotecoin, window)
    end
    if std
        storedstd = readregression(ohlcv.base, ohlcv.quotecoin, window; std=true)
        stdprefixlen = _reusableprefix(storedstd, ot)
        if stdprefixlen < length(ot)
            rs = rollingregressionstdmv([odf[!, :open], odf[!, :high], odf[!, :low], odf[!, :close]], stored[!, :regry], stored[!, :grad], window, stdprefixlen + 1)
            rs = stdprefixlen > 0 ? vcat(storedstd[!, :std], rs) : rs
            @assert length(rs) == length(ot) "regression std window=$window of $(ohlcv.base): length(rs)=$(length(rs)) != length(ot)=$(length(ot))"
            EnvConfig.savedf(DataFrame(opentime=copy(ot), std=rs), _regressionstem(window; std=true); folderpath=folder, format=:arrow)
            (verbosity >= 3) && println("$(EnvConfig.now()) updated regression std store window=$window of $(ohlcv.base) from row $(stdprefixlen + 1) until $(length(ot))")
        end
    end
    return nothing
end

"""
    readfeatures(base, quotecoin, featuresid; confighash)

Return the per-coin feature store `featuresid` as (memory mapped) DataFrame or `nothing` if not stored.
Asserts that the stored config hash equals `confighash`.
"""
function readfeatures(base::AbstractString, quotecoin::AbstractString, featuresid::AbstractString; confighash::AbstractString)
    folder = featuresfolder(base, quotecoin)
    meta = EnvConfig.tablemetadata(featuresid; folderpath=folder)
    isnothing(meta) && return nothing
    @assert get(meta, "confighash", "") == confighash "feature store $(featuresid) of $(base) has confighash=$(get(meta, "confighash", "")) but expected $(confighash) - the feature config was changed without a new id"
    return EnvConfig.readdf(featuresid; folderpath=folder, format=:arrow)
end

"""
    updatefeatures!(ohlcv, featuresid, f6; confighash)

Return the per-coin feature store `featuresid` after bringing it up to date with the complete
OHLCV history behind `ohlcv`. `f6` is an unbased Features006 config instance that is used for
the calculation; regressions are taken from (and added to) the regression store.
The store starts after the warm-up of `requiredminutes(f6)` minutes and has the columns
`opentime` followed by `requestedcolumns(f6)`. A store that matches the OHLCV history is only
extended by the rows of new OHLCV data, which Features006 calculates with the required lookback.
"""
function updatefeatures!(ohlcv::Ohlcv.OhlcvData, featuresid::AbstractString, f6::Features006; confighash::AbstractString)
    @assert isnothing(f6.ohlcv) "updatefeatures! expects an unbased Features006 config but it is based on $(f6.ohlcv.base)"
    odf = Ohlcv.dataframe(ohlcv)
    @assert !(odf isa SubDataFrame) "updatefeatures! requires the complete OHLCV history of $(ohlcv.base), not a view"
    ot = odf[!, :opentime]
    stored = readfeatures(ohlcv.base, ohlcv.quotecoin, featuresid; confighash=confighash)
    if !isnothing(stored) && (size(stored, 1) > 0) && (stored[end, :opentime] == ot[end])
        return stored
    end
    windows, stdwindows = regressionwindows(f6)
    for window in windows
        updateregression!(ohlcv, window; std=(window in stdwindows))
    end
    cols = requestedcolumns(f6)
    # the store starts at OHLCV row requiredminutes, which is the first row with complete feature windows
    firstohlcvix = requiredminutes(f6)
    @assert firstohlcvix <= lastindex(ot) "insufficient OHLCV history of $(ohlcv.base) for $(featuresid): $(size(odf, 1)) rows with requiredminutes=$(requiredminutes(f6))"
    storedlastix = isnothing(stored) ? 0 : firstohlcvix + size(stored, 1) - 1
    extendable = !isnothing(stored) && (size(stored, 1) > 0) && (stored[begin, :opentime] == ot[firstohlcvix]) && (storedlastix < lastindex(ot)) && (stored[end, :opentime] == ot[storedlastix])
    if extendable
        tailohlcv = Ohlcv.ohlcvview(ohlcv, (storedlastix + 1):lastindex(ot))
        setbase!(f6, tailohlcv, usecache=true)
        taildf = f6.fdf[!, vcat(["opentime"], cols)]
        @assert (size(taildf, 1) == lastindex(ot) - storedlastix) && (taildf[begin, :opentime] == ot[storedlastix + 1]) "feature tail of $(ohlcv.base) for $(featuresid) covers $(size(taildf, 1)) rows from $(taildf[begin, :opentime]) but expected $(lastindex(ot) - storedlastix) rows from $(ot[storedlastix + 1])"
        storedf = vcat(DataFrame(stored; copycols=true), DataFrame(taildf; copycols=true))
    else
        setbase!(f6, ohlcv, usecache=true)
        # fdf row k corresponds to OHLCV row k + maxoffset
        storedf = DataFrame(f6.fdf[(firstohlcvix - f6.maxoffset):end, vcat(["opentime"], cols)]; copycols=true)
    end
    removebase!(f6)
    meta = Dict("confighash" => String(confighash), "requiredminutes" => string(requiredminutes(f6)), "columns" => join(cols, ","))
    EnvConfig.savedf(storedf, featuresid; folderpath=featuresfolder(ohlcv.base, ohlcv.quotecoin), format=:arrow, metadata=meta)
    (verbosity >= 3) && println("$(EnvConfig.now()) $(extendable ? "extended" : "calculated") feature store $(featuresid) of $(ohlcv.base) to $(size(storedf, 1)) rows")
    return readfeatures(ohlcv.base, ohlcv.quotecoin, featuresid; confighash=confighash)
end

#endregion FeatureStore
