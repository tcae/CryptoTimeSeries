
using DrWatson
@quickactivate "CryptoTimeSeries"

include(srcdir("targets.jl"))

module TargetsTest

using Dates, DataFrames
using Test

using ..Targets

function testohlcvinit(base::String)
    ohlcv1 = CacheData.readcsv("test")
    println("ohlcv1: $ohlcv1")
    CacheData.write(ohlcv1)
    ohlcv2 = CacheData.read("test")
    println("ohlcv2: $ohlcv2")
    return ohlcv1.df == ohlcv2.df
    return isapprox(ohlcv1.df, ohlcv2.df)
end

function rollingregression_test()
    y = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
    """
    65  slope = 0.310714  y_regr = 4.717857
    """
    r = CacheData.rollingregression(y, size(y, 1))
    # r2 = CacheData.rolling_regression2(y, size(y, 1))
    # r3 = CacheData.rolling_regression2(y, 3)
    # println("r=$r   r2=$r2  r3(3)=$r3")
    return r[7] == 0.310714285714285
end


@testset "Targets tests" begin

@test true

end

end  # module