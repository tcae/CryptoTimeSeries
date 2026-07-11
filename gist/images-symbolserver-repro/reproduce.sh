#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
STORE_DIR="$ROOT_DIR/symbolstore"

# Optional override for non-default VS Code extension path.
if [[ -n "${JULIA_SYMBOLSERVER_SERVER_JL:-}" ]]; then
  SERVER_JL="$JULIA_SYMBOLSERVER_SERVER_JL"
else
  SERVER_JL="$HOME/.vscode/extensions/julialang.language-julia-1.219.2/scripts/packages/SymbolServer/src/server.jl"
fi

if [[ ! -f "$SERVER_JL" ]]; then
  echo "SymbolServer server.jl not found at: $SERVER_JL"
  echo "Set JULIA_SYMBOLSERVER_SERVER_JL to the full path of server.jl."
  exit 1
fi

julia --project="$ROOT_DIR" -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# This command reproduces the crash on affected setups.
julia --project="$ROOT_DIR" "$SERVER_JL" "$STORE_DIR"
