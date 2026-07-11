
using Pkg

"Parse boolean env vars in a strict but user-friendly way."
function _env_bool(name::AbstractString, default::Bool)
	haskey(ENV, name) || return default
	raw = lowercase(strip(String(ENV[name])))
	raw in ("1", "true", "yes", "on") && return true
	raw in ("0", "false", "no", "off") && return false
	throw(ArgumentError("invalid boolean env $(name)=$(repr(ENV[name])); expected one of 1/0,true/false,yes/no,on/off"))
end

function _env_string(name::AbstractString, default::AbstractString)
	haskey(ENV, name) || return String(default)
	raw = strip(String(ENV[name]))
	isempty(raw) && return String(default)
	return raw
end

function _cleanup_legacy_cov!(root::AbstractString)::Nothing
	for (dir, _, files) in walkdir(root)
		for file in files
			endswith(file, ".cov") || continue
			rm(joinpath(dir, file); force=true)
		end
	end
	return nothing
end

const WORKSPACE_PACKAGES = [
	"EnvConfig",
	"Ohlcv",
	"Features",
	"Targets",
	"Classify",
	"TestOhlcv",
	"Bybit",
	"KrakenSpot",
	"KrakenFutures",
	"Xch",
	"TradingStrategy",
	"Trade",
]

const RUN_COVERAGE = _env_bool("CTS_TEST_COVERAGE", false ) # true)
const COVERAGE_DIR = joinpath(@__DIR__, "coverage", "latest")
const COVERAGE_TRACEFILE = _env_string("CTS_TEST_COVERAGE_TRACEFILE", joinpath(COVERAGE_DIR, "lcov.info"))
const COVERAGE_ARG = RUN_COVERAGE ? COVERAGE_TRACEFILE : false

const ROOT_DEPENDENCIES = Set(String.(collect(keys(Pkg.project().dependencies))))
const TESTABLE_PACKAGES = [pkg for pkg in WORKSPACE_PACKAGES if pkg in ROOT_DEPENDENCIES]
const SKIPPED_PACKAGES = [pkg for pkg in WORKSPACE_PACKAGES if !(pkg in ROOT_DEPENDENCIES)]
const PKGTEST_PACKAGES = copy(TESTABLE_PACKAGES)

if !isempty(SKIPPED_PACKAGES)
	@warn "Skipping workspace packages that are not dependencies of root Project.toml" skipped=SKIPPED_PACKAGES
end

if RUN_COVERAGE
	mkpath(dirname(COVERAGE_TRACEFILE))
	rm(COVERAGE_TRACEFILE; force=true)
	_cleanup_legacy_cov!(normpath(joinpath(@__DIR__, "..")))
	@info "Coverage output configured" tracefile=COVERAGE_TRACEFILE
end

# Root test runner for complete workspace coverage across package suites.
Pkg.test(PKGTEST_PACKAGES; coverage=COVERAGE_ARG)

include(joinpath(@__DIR__, "bounds_estimator_test.jl"))
include(joinpath(@__DIR__, "trend_detector_cache_test.jl"))
include(joinpath(@__DIR__, "script_help_test.jl"))
