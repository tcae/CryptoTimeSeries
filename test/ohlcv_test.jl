using DrWatson
@quickactivate "CryptoTimeSeries"

include("../src/ohlcv.jl")
# include(srcdir("ohlcv.jl"))


module OhlcvTest

using DataFrames
using Test
using ..Config
using ..Ohlcv

function testohlcvinit(base::String)
    ohlcv1 = Ohlcv.readcsv("test")
    # println("ohlcv1: $ohlcv1")
    Ohlcv.write(ohlcv1)
    ohlcv2 = Ohlcv.read("test")
    # println("ohlcv2: $ohlcv2")
    return ohlcv1.df == ohlcv2.df
    return isapprox(ohlcv1.df, ohlcv2.df)
end

function rollingregression_test()
    y = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
    """
    65  slope = 0.310714  y_regr = 4.717857
    """
    r = Ohlcv.rollingregression(y, size(y, 1))
    # r2 = Ohlcv.rolling_regression2(y, size(y, 1))
    # r3 = Ohlcv.rolling_regression2(y, 3)
    # println("r=$r   r2=$r2  r3(3)=$r3")
    return r[7] == 0.310714285714285
end

function setsplit_test()
    splitdf = Ohlcv.setsplit()
    nrow(splitdf) == 3 || return false
    return true
end

function setassign_test()
    ohlcv = Ohlcv.read("test")
    splitdf = Ohlcv.setassign!(ohlcv)
    nrow(ohlcv.df) == 9 || return false
    names(ohlcv.df) == ["timestamp", "open", "high", "low", "close", "volume", "pivot", "set"] || return false
    return true
end

function columnarray_test()
    expected = [ 0.205815  0.204343     0.204343      0.204522      0.214546;
    725.0       0.0       3137.0       14150.0       33415.0;
    0.207287  0.204343     0.204343      0.204703      0.213031]
    # display(expected)
    ohlcv2 = Ohlcv.read("test")
    # display(ohlcv2)
    cols = [:pivot, :volume, :open]
    colarray = Ohlcv.columnarray(ohlcv2, "training", cols)
    return isapprox(expected, colarray)
end

function pivot_test()
    expected = [ 0.205815  0.204343     0.204343      0.204522      0.214546;
    725.0       0.0       3137.0       14150.0       33415.0;
    0.207287  0.204343     0.204343      0.204703      0.213031]
    # display(expected)
    ohlcv2 = Ohlcv.read("test")
    # rename!(ohlcv2.df,:pivot => :pivot2)
    ohlcv2 = Ohlcv.addpivot!(ohlcv2)
    display(ohlcv2)
    # return isapprox(ohlcv2.df.pivot, ohlcv2.df.pivot2)
end

# println("ohlcv_test")
Config.init(test)
# pivot_test()

@testset "Ohlcv tests" begin

@test testohlcvinit("test")
@test Ohlcv.mnemonic(Ohlcv.OhlcvData(DataFrame(), "test")) == "test_OHLCV"
@test setsplit_test()
@test setassign_test()
@test columnarray_test()

# @test Ohlcv.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)[7] == 0.310714285714285
end

end  # OhlcvTest
