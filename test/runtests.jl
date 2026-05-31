
using Pkg

"Parse boolean env vars in a strict but user-friendly way."
function _env_bool(name::AbstractString, default::Bool)
	haskey(ENV, name) || return default
	raw = lowercase(strip(String(ENV[name])))
	raw in ("1", "true", "yes", "on") && return true
	raw in ("0", "false", "no", "off") && return false
	throw(ArgumentError("invalid boolean env $(name)=$(repr(ENV[name])); expected one of 1/0,true/false,yes/no,on/off"))
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
	"CryptoXch",
	"Assets",
	"TradeLog",
	"TradingStrategy",
	"Trade",
	"TradeAudit",
]

const RUN_COVERAGE = _env_bool("CTS_TEST_COVERAGE", true)

const ROOT_DEPENDENCIES = Set(String.(collect(keys(Pkg.project().dependencies))))
const TESTABLE_PACKAGES = [pkg for pkg in WORKSPACE_PACKAGES if pkg in ROOT_DEPENDENCIES]
const SKIPPED_PACKAGES = [pkg for pkg in WORKSPACE_PACKAGES if !(pkg in ROOT_DEPENDENCIES)]

if !isempty(SKIPPED_PACKAGES)
	@warn "Skipping workspace packages that are not dependencies of root Project.toml" skipped=SKIPPED_PACKAGES
end

# Root test runner for complete workspace coverage across package suites.
Pkg.test(TESTABLE_PACKAGES; coverage=RUN_COVERAGE)

include(joinpath(@__DIR__, "bounds_estimator_test.jl"))
include(joinpath(@__DIR__, "trend_detector_cache_test.jl"))
include(joinpath(@__DIR__, "script_help_test.jl"))
include(joinpath(@__DIR__, "tradeadvice_lstm_test.jl"))
