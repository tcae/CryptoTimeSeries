#!/usr/bin/env julia

"""
Preflight environment check for local OneDrive access.

Usage:
  julia scripts/check_env.jl
  julia scripts/check_env.jl --verbose
  julia scripts/check_env.jl --debug

Flags:
  -v, --verbose   Print progress messages
  --debug         Verbose + diagnostic details
"""

import Pkg
Pkg.develop("EnvConfig")
using EnvConfig
# ----------------------------
# Argument parsing
# ----------------------------
const VERBOSE = any(x -> x in ("-v", "--verbose", "--debug"), ARGS)
const DEBUG   = any(x -> x == "--debug", ARGS)

EnvConfig.checkfolders(VERBOSE, DEBUG)
exit(0)
