using Logging, LoggingExtras

# Define the Foo module
module Foo
    info() = @debug "Information from Foo"
    warn() = @warn "Warning from Foo"
end
using .Foo

all_logger = ConsoleLogger(stderr, Logging.BelowMinLevel)
# Create the logger
logger = EarlyFilteredLogger(all_logger) do args
    r = Logging.Debug <= args.level <= Logging.Warn && args._module === Foo
    return r
end

# Test it
with_logger(logger) do
    @debug "Debug Info from Main"
    @warn "Warning from Main"
    Foo.info()
    Foo.warn()
end
