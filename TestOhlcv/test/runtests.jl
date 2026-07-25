module TestOhlcvTest

using Dates, DataFrames
using Test

using EnvConfig, Ohlcv, TestOhlcv

ohlc = TestOhlcv.testohlcv(
    "SINE", Dates.DateTime("2025-01-02T01:11", dateformat"yyyy-mm-ddTHH:MM"),
    Dates.DateTime("2025-01-02T01:42", dateformat"yyyy-mm-ddTHH:MM"), "1m")
df = Ohlcv.dataframe(ohlc)


@testset begin
@test size(df) == (31, 7)

end # testset

@testset "fixed-anchor timestamp reproducibility" begin
    targetdt = Dates.DateTime("2025-08-01T06:00:00", dateformat"yyyy-mm-ddTHH:MM:SS")

    short_sine = TestOhlcv.testohlcv(
        "SINE",
        targetdt - Dates.Minute(5),
        targetdt + Dates.Minute(5),
        "1m",
    )
    long_sine = TestOhlcv.testohlcv(
        "SINE",
        targetdt - Dates.Day(1),
        targetdt + Dates.Day(1),
        "1m",
    )

    sshort = Ohlcv.dataframe(short_sine)
    slong = Ohlcv.dataframe(long_sine)
    six_short = Ohlcv.rowix(sshort[!, :opentime], targetdt)
    six_long = Ohlcv.rowix(slong[!, :opentime], targetdt)
    for col in (:open, :high, :low, :close, :basevolume, :pivot)
        @test sshort[six_short, col] == slong[six_long, col]
    end

    short_double = TestOhlcv.testohlcv(
        "DOUBLESINE",
        targetdt - Dates.Minute(5),
        targetdt + Dates.Minute(5),
        "1m",
    )
    long_double = TestOhlcv.testohlcv(
        "DOUBLESINE",
        targetdt - Dates.Day(1),
        targetdt + Dates.Day(1),
        "1m",
    )

    dshort = Ohlcv.dataframe(short_double)
    dlong = Ohlcv.dataframe(long_double)
    dix_short = Ohlcv.rowix(dshort[!, :opentime], targetdt)
    dix_long = Ohlcv.rowix(dlong[!, :opentime], targetdt)
    for col in (:open, :high, :low, :close, :basevolume, :pivot)
        @test dshort[dix_short, col] == dlong[dix_long, col]
    end
end

end # of TestOhlcvTest
