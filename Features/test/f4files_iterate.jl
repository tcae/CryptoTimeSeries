using Features, Ohlcv, CryptoXch, EnvConfig, Dates
Features.verbosity = 1
Ohlcv.verbosity = 1
verbosity = 1

EnvConfig.init(production)
xc=CryptoXch.XchCache()

"""
Implementation of regression features was changed to enable starting with the first ohlcv sample and acepting history impact but enabling less complex logic due to differences in feature vector length.
This function helps adding the initial missing features to the Features004 feature cache.
"""
function addearlysamples() 
    # datetime = Dates.now(UTC)
    for f4 in Features.Features004Files()
        # ohlcv = Ohlcv.read(f4.basecoin)
        datetime = first(values(f4.rw))[end, :opentime] + Minute(1)
        ohlcv = CryptoXch.cryptodownload(xc, f4.basecoin, "1m", datetime - Year(20), datetime)
        if size(Ohlcv.dataframe(ohlcv), 1) < size(first(values(f4.rw)), 1)
            (verbosity >= 1) && println("deleting f4: ohlcv=$ohlcv, f4:$f4")
            Features.delete(f4)
        else
            Ohlcv.write(ohlcv) # write ohlcv even if data length is too short to calculate features
            (verbosity >= 3) && println("ohlcv=$ohlcv, f4:$f4")
            (verbosity >= 1) && println("before coin=$(ohlcv.base) size(ohlcv, 1)=$(size(Ohlcv.dataframe(ohlcv), 1))==size(f4, 1)=$(size(first(values(f4.rw)), 1))== $(size(Ohlcv.dataframe(ohlcv), 1)==size(first(values(f4.rw)), 1)))")
            Features.supplement!(f4, ohlcv) # purpose: add potentially missing start
            (verbosity >= 3) && println("after supplement f4:$f4")
            (verbosity >= 1) && println("after coin=$(ohlcv.base) size(ohlcv, 1)=$(size(Ohlcv.dataframe(ohlcv), 1))==size(f4, 1)=$(size(first(values(f4.rw)), 1))== $(size(Ohlcv.dataframe(ohlcv), 1)==size(first(values(f4.rw)), 1)))")
            Features.write(f4)
        end
        # break
    end
end

function showcachef4df() 
    Features.verbosity = 4
    for f4 in Features.Features004Files()
        (verbosity >= 1) && println(f4)
        break
    end
end


# addearlysamples()
showcachef4df() 
