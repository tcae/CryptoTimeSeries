module OhlcvTest

using DataFrames, Dates
using Test
using EnvConfig
using Ohlcv

function testohlcvinit(base::String)
    ohlcv1 = Ohlcv.defaultohlcv("test")
    ohlcv1 = Ohlcv.readcsv!(ohlcv1)
    # println("ohlcv1: $ohlcv1")
    Ohlcv.write(ohlcv1)
    return ohlcv1
end

function readwrite(ohlcv1)
    ohlcv2 = Ohlcv.defaultohlcv(Ohlcv.basesymbol(ohlcv1))
    ohlcv2 = Ohlcv.read!(ohlcv2)
    # println("ohlcv2: $ohlcv2")
    return ohlcv2
    # return ohlcv1.df == ohlcv2.df
    # return isapprox(ohlcv1.df, ohlcv2.df)
end

# function rollingregression_test()
#     y = [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]
#     """
#     65  slope = 0.310714  y_regr = 4.717857
#     """
#     r = Ohlcv.rollingregression(y, size(y, 1))
#     # r2 = Ohlcv.rolling_regression2(y, size(y, 1))
#     # r3 = Ohlcv.rolling_regression2(y, 3)
#     # println("r=$r   r2=$r2  r3(3)=$r3")
#     return r[7] == 0.310714285714285
# end

function setsplit_test()
    splitdf = Ohlcv.setsplit()
    # println("splitdf: $splitdf")
    nrow(splitdf) == 3 || return false
    return true
end

function setassign_test()
    ohlcv = Ohlcv.defaultohlcv("test")
    ohlcv = Ohlcv.read!(ohlcv)
    Ohlcv.setassign!(ohlcv)
    # println("ohlcv after split set: $ohlcv")
    nrow(ohlcv.df) == 9 || return false
    names(ohlcv.df) == ["opentime", "open", "high", "low", "close", "basevolume", "set"] || return false
    return true
end

function columnarray_test()
    expected = [ 0.205815  0.204343     0.204343      0.204522      0.214546;
    725.0       0.0       3137.0       14150.0       33415.0;
    0.207287  0.204343     0.204343      0.204703      0.213031]
    # display(expected)
    ohlcv = Ohlcv.defaultohlcv("test")
    ohlcv = Ohlcv.read!(ohlcv)
    Ohlcv.addpivot!(ohlcv)
    # display(ohlcv2)
    cols = [:pivot, :basevolume, :open]
    colarray = Ohlcv.columnarray(ohlcv, "training", cols)
    return isapprox(expected, colarray)
end

function pivot_test()
    expected = [ 0.205815  0.204343     0.204343      0.204522      0.214546;
    725.0       0.0       3137.0       14150.0       33415.0;
    0.207287  0.204343     0.204343      0.204703      0.213031]
    # display(expected)
    ohlcv2 = Ohlcv.defaultohlcv("test")
    ohlcv2 = Ohlcv.read!(ohlcv2)
    # rename!(ohlcv2.df,:pivot => :pivot2)
    ohlcv2 = Ohlcv.addpivot!(ohlcv2)
    display(ohlcv2)
    # return isapprox(ohlcv2.df.pivot, ohlcv2.df.pivot2)
end

function ohlcvab(offset)
    dfa = DataFrame(
        opentime=[DateTime("2022-01-02T22:55:00")+Dates.Minute(i) for i in 1:3],
        open=[1.3+i for i in 1:3],
        high=[1.3+i for i in 1:3],
        low=[1.3+i for i in 1:3],
        close=[1.3+i for i in 1:3],
        basevolume=[1.3+i for i in 1:3]
        )
    dfb = DataFrame(
        opentime=[DateTime("2022-01-02T22:55:00")+Dates.Minute(i+offset) for i in 1:5],
        open=[1.5+i for i in 1:5],
        high=[1.5+i for i in 1:5],
        low=[1.5+i for i in 1:5],
        close=[1.5+i for i in 1:5],
        basevolume=[1.5+i for i in 1:5]
        )
    ohlcva = Ohlcv.defaultohlcv("test")
    Ohlcv.setdataframe!(ohlcva, dfa)
    ohlcvb = Ohlcv.defaultohlcv("test")
    Ohlcv.setdataframe!(ohlcvb, dfb)

    return ohlcva,ohlcvb
end

# println("ohlcv_test")
EnvConfig.init(test)
# pivot_test()

@testset "Ohlcv tests" begin

ohlcv1 = testohlcvinit("test")

@test names(Ohlcv.dataframe(ohlcv1)) == ["opentime", "open", "high", "low", "close", "basevolume"]
@test nrow(Ohlcv.dataframe(ohlcv1)) == 9
@test Ohlcv.mnemonic(ohlcv1) == "test_usdt_binance_1m_OHLCV"

ohlcv2 = readwrite(ohlcv1)
@test names(Ohlcv.dataframe(ohlcv1)) == ["opentime", "open", "high", "low", "close", "basevolume"]
@test nrow(Ohlcv.dataframe(ohlcv1)) == 9
@test Ohlcv.dataframe(ohlcv1)[1, :open] == Ohlcv.dataframe(ohlcv2)[1, :open]
@test Ohlcv.dataframe(ohlcv1)[1, :opentime] == Ohlcv.dataframe(ohlcv2)[1, :opentime]
@test Ohlcv.dataframe(ohlcv1)[9, :basevolume] == Ohlcv.dataframe(ohlcv2)[9, :basevolume]
@test setsplit_test()
@test setassign_test()
@test columnarray_test()

ohlcva, ohlcvb = ohlcvab(-3)  # add ohlcvb at start ohlcba
# println(ohlcva.df)
# println(ohlcvb.df)
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 6  # last line ohlcva stays

ohlcva, ohlcvb = ohlcvab(-4)  # add ohlcvb at start ohlcba with 1 line overlap
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 7  # last line ohlcva stays

ohlcva, ohlcvb = ohlcvab(-5)  # add ohlcvb at start ohlcba without overlap
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 8  # last line ohlcva stays

ohlcva, ohlcvb = ohlcvab(2)  # add ohlcvb at end ohlcba with 1 line overlap
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 7

ohlcva, ohlcvb = ohlcvab(3)  # add ohlcvb at end ohlcba without overlap
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 8

ohlcva, ohlcvb = ohlcvab(-1)  # ohclvb fully covers ohlcv
ohlcva = Ohlcv.merge!(ohlcva, ohlcvb)
# println(ohlcva.df)
@test size(Ohlcv.dataframe(ohlcva), 1) == 5

# @test Ohlcv.rollingregression([2.9, 3.1, 3.6, 3.8, 4, 4.1, 5], 7)[7] == 0.310714285714285
end


end  # OhlcvTest
