using EnvConfig, Ohlcv, Features, CryptoXch
using Dates, Ohlcv.DataFrames

function deleteoutdateohlcv()
    EnvConfig.init(production)
    Ohlcv.verbosity = 1

    xc = CryptoXch.XchCache(true)
    outdated = invalid = valid = 0
    latest = oldest = nothing
    deadline = DateTime("2024-01-01T01:00:00")
    for ohlcv in Ohlcv.OhlcvFiles()
        if CryptoXch.validbase(xc, ohlcv.base)
            valid += 1
            if length(ohlcv.df[!, :opentime]) == 0
                println("empty ohlcv: $(Ohlcv.file(ohlcv))")
            else
                oldest = isnothing(oldest) ? ohlcv.df[end, :opentime] : min(oldest, ohlcv.df[end, :opentime])
                if ohlcv.df[end, :opentime] < deadline
                    println("outdated ohlcv: $ohlcv")
                    Ohlcv.delete(ohlcv)
                    outdated += 1
                end
            end
        else
            invalid += 1
            if length(ohlcv.df[!, :opentime]) > 0
                latest = isnothing(latest) ? ohlcv.df[end, :opentime] : max(latest, ohlcv.df[end, :opentime])
            end
            Ohlcv.delete(ohlcv)
            # println("invalid symbol: $ohlcv")
        end
    end
    println("total of $(invalid+valid) ohlcv files of which $valid are valid symbols and $invalid are invalid symbols")
    !isnothing(latest) && println("latest update of an invalid symbol is $latest")
    !isnothing(oldest) && println("oldest update of a valid symbol is $oldest")
    !isnothing(outdated) && println("number of outdated updates of valid symbols is $outdated")
end

deleteoutdateohlcv()
