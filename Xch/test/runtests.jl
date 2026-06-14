
# How to run skipped suites on demand

# Production integration tests:
# CTS_RUN_PRODUCTION_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'

run_production_tests = lowercase(get(ENV, "CTS_RUN_PRODUCTION_TESTS", "false")) in ("1", "true", "yes")
if run_production_tests
	include("productionruntests.jl")
else
	@info "Skipping production integration tests. Set CTS_RUN_PRODUCTION_TESTS=true to enable test/productionruntests.jl"
end
include("routingtests.jl")
include("openstatus_test.jl")
include("order_request_status_test.jl")
include("trades_schema_contract_test.jl")
include("messagecatalogtests.jl")
include("log_integration_test.jl")
include("multileg_order_test.jl")
include("usdtmarket_intent_test.jl")

