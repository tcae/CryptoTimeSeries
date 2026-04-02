import Pkg
Pkg.activate(joinpath(@__DIR__, ".."), io=devnull)

using Dates
using Targets
using Ohlcv
using TestOhlcv
using EnvConfig

function _run_sine_scenario(name::String, th::Targets.LabelThresholds)
    ohlcv = TestOhlcv.testohlcv("SINE", DateTime("2025-07-31T09:32:00"), DateTime("2025-08-01T09:31:00"))
    trd = Targets.Trend04(10, 4 * 60, th)

    Targets.reset_trend04_diagnostics!()
    Targets.enable_trend04_diagnostics!(true)
    Targets.setbase!(trd, ohlcv)
    Targets.enable_trend04_diagnostics!(false)

    labels = trd.df[!, :label]
    checks = Targets.crosscheck(trd)
    diag = Targets.trend04_diagnostics()

    println("Scenario: $name")
    println("  crosscheck_valid=$(isempty(checks)) issues=$(length(checks))")
    println("  labels: LB=$(count(==(longbuy), labels)) LH=$(count(==(longhold), labels)) SB=$(count(==(shortbuy), labels)) SH=$(count(==(shorthold), labels)) AC=$(count(==(allclose), labels))")
    _print_hold_diag(diag)
    println()
end

function _run_btc_slice(name::String, th::Targets.LabelThresholds, startix::Int, endix::Int)
    EnvConfig.init(training)
    ohlcv = Ohlcv.read("BTC")
    Ohlcv.timerangecut!(ohlcv, startix, endix)
    trd = Targets.Trend04(10, 4 * 60, th)

    Targets.reset_trend04_diagnostics!()
    Targets.enable_trend04_diagnostics!(true)
    Targets.setbase!(trd, ohlcv)
    Targets.enable_trend04_diagnostics!(false)

    labels = trd.df[!, :label]
    checks = Targets.crosscheck(trd)
    diag = Targets.trend04_diagnostics()

    println("Scenario: $name")
    println("  slice=$startix:$endix crosscheck_valid=$(isempty(checks)) issues=$(length(checks))")
    println("  labels: LB=$(count(==(longbuy), labels)) LH=$(count(==(longhold), labels)) SB=$(count(==(shortbuy), labels)) SH=$(count(==(shorthold), labels)) AC=$(count(==(allclose), labels))")
    _print_hold_diag(diag)
    println()
end

function _print_hold_diag(diag::Dict{String, Int})
    cand_long = get(diag, "cand.longhold.reversal", 0) + get(diag, "cand.longhold.from_longhold", 0) + get(diag, "cand.longhold.from_longbuy", 0)
    acc_long = get(diag, "acc.longhold.reversal", 0) + get(diag, "acc.longhold.from_longhold", 0) + get(diag, "acc.longhold.from_longbuy", 0) + get(diag, "acc.longhold.from_micro_shortpeak", 0)
    cand_short = get(diag, "cand.shorthold.reversal", 0) + get(diag, "cand.shorthold.from_shorthold", 0) + get(diag, "cand.shorthold.from_shortbuy", 0)
    acc_short = get(diag, "acc.shorthold.reversal", 0) + get(diag, "acc.shorthold.from_shorthold", 0) + get(diag, "acc.shorthold.from_shortbuy", 0) + get(diag, "acc.shorthold.from_micro_longpeak", 0)

    println("  hold-flow:")
    println("    long candidates=$cand_long accepted=$acc_long")
    println("    short candidates=$cand_short accepted=$acc_short")
    println("    transient micro peaks: short-from-long=$(get(diag, "transient.micro.shortpeak.from_long", 0)) long-from-short=$(get(diag, "transient.micro.longpeak.from_short", 0))")

    rej = [(k, v) for (k, v) in diag if startswith(k, "rej.longhold") || startswith(k, "rej.shorthold")]
    sort!(rej, by=x -> x[2], rev=true)
    topn = min(8, length(rej))
    println("    top rejection reasons:")
    for i in 1:topn
        println("      $(rej[i][1]) => $(rej[i][2])")
    end
end

function main()
    EnvConfig.init(test)
    Targets.verbosity = 1

    println("Trend04 hold diagnostics")
    println("========================")

    _run_sine_scenario("SINE strict buy10 hold1", Targets.LabelThresholds(0.1f0, 0.01f0, -0.01f0, -0.1f0))
    _run_sine_scenario("SINE relaxed buy3 hold0.5", Targets.LabelThresholds(0.03f0, 0.005f0, -0.005f0, -0.03f0))

    _run_btc_slice("BTC baseline thresholds", Targets.LabelThresholds(0.01f0, 0.005f0, -0.005f0, -0.01f0), 1200, 1290)
end

main()
