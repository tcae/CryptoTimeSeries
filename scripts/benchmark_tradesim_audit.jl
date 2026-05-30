"""
benchmark_tradesim_audit.jl

Run controlled A/B benchmarks for tradesim audit overhead by executing
`scripts/tradesim.jl` with identical inputs under three audit modes:
1) full audit on
2) global audit off
3) simulation audit off (global on)

Usage:
    julia --project=scripts scripts/benchmark_tradesim_audit.jl

Optional environment overrides:
    TRADESIM_BENCH_STARTDT   (default: 2025-01-01T00:00:00)
    TRADESIM_BENCH_ENDDT     (default: 2025-01-01T12:00:00)
    TRADESIM_BENCH_WHITELIST (default: BTC,ETH,SOL,XRP,MNT,PEPE,DOGE,LINK,ADA,WIF)
    TRADESIM_BENCH_SHOW_CHILD_OUTPUT=true|false (default: false)

Notes:
- This script measures end-to-end wall time for each run.
- Child process output is inherited so tradesim diagnostics remain visible.
"""

using Dates
using Printf

struct BenchMode
    name::String
    audit_enabled::Bool
    sim_audit_enabled::Bool
end

"""Return a normalized benchmark environment map for child tradesim processes."""
function bench_env()::Dict{String, String}
    startdt = get(ENV, "TRADESIM_BENCH_STARTDT", get(ENV, "TRADESIM_STARTDT", "2025-01-01T00:00:00"))
    enddt = get(ENV, "TRADESIM_BENCH_ENDDT", get(ENV, "TRADESIM_ENDDT", "2025-01-01T12:00:00"))
    whitelist = get(
        ENV,
        "TRADESIM_BENCH_WHITELIST",
        get(ENV, "TRADESIM_WHITELIST", "BTC,ETH,SOL,XRP,MNT,PEPE,DOGE,LINK,ADA,WIF"),
    )

    @assert DateTime(startdt) <= DateTime(enddt) "benchmark start/end invalid: start=$(startdt), end=$(enddt)"

    return Dict(
        "TRADESIM_STARTDT" => startdt,
        "TRADESIM_ENDDT" => enddt,
        "TRADESIM_WHITELIST" => whitelist,
    )
end

"""Execute one benchmark mode and return elapsed wall time in seconds."""
function run_mode(mode::BenchMode, env::Dict{String, String})::Float64
    child_env = Dict{String, String}(env)
    child_env["CTS_TRADELOG_ENABLED"] = mode.audit_enabled ? "true" : "false"
    child_env["CTS_TRADELOG_SIMULATION_ENABLED"] = mode.sim_audit_enabled ? "true" : "false"
    # Backward-compatible flags for older modules still reading audit names.
    child_env["CTS_AUDIT_ENABLED"] = child_env["CTS_TRADELOG_ENABLED"]
    child_env["CTS_AUDIT_SIMULATION_ENABLED"] = child_env["CTS_TRADELOG_SIMULATION_ENABLED"]

    cmd = `$(Base.julia_cmd()) --project=scripts scripts/tradesim.jl`
    cmd_with_env = addenv(cmd, collect(pairs(child_env))...)

    println()
    println("="^80)
    audit_enabled = child_env["CTS_TRADELOG_ENABLED"]
    sim_audit_enabled = child_env["CTS_TRADELOG_SIMULATION_ENABLED"]
    run_startdt = child_env["TRADESIM_STARTDT"]
    run_enddt = child_env["TRADESIM_ENDDT"]
    run_whitelist = child_env["TRADESIM_WHITELIST"]
    println("Running mode=$(mode.name) CTS_TRADELOG_ENABLED=$(audit_enabled) CTS_TRADELOG_SIMULATION_ENABLED=$(sim_audit_enabled)")
    println("window=$(run_startdt) -> $(run_enddt)")
    println("whitelist=$(run_whitelist)")
    println("="^80)

    show_child_output = lowercase(strip(get(ENV, "TRADESIM_BENCH_SHOW_CHILD_OUTPUT", "false"))) == "true"
    elapsed = @elapsed begin
        if show_child_output
            run(cmd_with_env)
        else
            run(pipeline(cmd_with_env; stdout=devnull, stderr=devnull))
        end
    end
    @printf("mode=%s elapsed_seconds=%.3f\n", mode.name, elapsed)
    return elapsed
end

"""Print benchmark summary table and relative speedups."""
function print_summary(results::Dict{String, Float64})
    base = results["audit_on"]
    global_off = results["audit_global_off"]
    sim_off = results["audit_simulation_off"]

    println()
    println("="^80)
    println("A/B Audit Benchmark Summary")
    println("="^80)
    @printf("%-24s %12s %12s\n", "mode", "seconds", "speedup")
    @printf("%-24s %12.3f %12.3fx\n", "audit_on", base, 1.0)
    @printf("%-24s %12.3f %12.3fx\n", "audit_global_off", global_off, base / max(global_off, eps()))
    @printf("%-24s %12.3f %12.3fx\n", "audit_simulation_off", sim_off, base / max(sim_off, eps()))

    delta_global = 100.0 * (base - global_off) / max(base, eps())
    delta_sim = 100.0 * (base - sim_off) / max(base, eps())
    @printf("global_off improvement: %+.2f%%\n", delta_global)
    @printf("sim_off improvement   : %+.2f%%\n", delta_sim)
end

function main()
    env = bench_env()
    modes = BenchMode[
        BenchMode("audit_on", true, true),
        BenchMode("audit_global_off", false, false),
        BenchMode("audit_simulation_off", true, false),
    ]

    results = Dict{String, Float64}()
    started = now()
    println("Started benchmark at $(started)")

    for mode in modes
        results[mode.name] = run_mode(mode, env)
    end

    print_summary(results)
    finished = now()
    println("Finished benchmark at $(finished)")
    println("Total wall time: $(round(Dates.value(finished - started) / 1000; digits=3)) seconds")
end

main()
