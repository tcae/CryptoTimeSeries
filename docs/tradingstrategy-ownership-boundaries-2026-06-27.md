# TradingStrategy ownership boundaries

This note summarizes the refactor that moved reusable strategy configuration into the `TradingStrategy` package.

## What moved

- The preset config factory layer now lives in [TradingStrategy/src/tradingstrategyconfig.jl](../TradingStrategy/src/tradingstrategyconfig.jl).
- `TradingStrategy` includes that file directly and exposes the reusable config helpers to consumers.
- `tradereal`, `tradesim`, `TrendDetector`, `BoundsEstimator`, `simulationcheck`, and `cryptocockpit` now consume the package-owned config API instead of including a top-level script.

## Ownership boundaries

- `TradingStrategy` owns reusable strategy configuration, preset selection, gain-materialization logic, and the `TsCache` runtime.
- `Classify` owns classifier resolution, abstract classifier behavior, and concrete classifier adaptation/loading.
- `Xch` owns Trades DataFrame lifecycle, schema materialization, and execution/account feedback columns.
- `Trade` owns trade-selection policy, risk and sizing decisions, and order submission wiring.
- `TrendDetector` owns experiment orchestration and diagnostics, but consumes configs and runtimes from the package layers above it.

## Resulting shape

- Config presets are reusable across simulation and live trading.
- Legacy top-level script inclusion is removed.
- The config path is now package-owned instead of script-owned, which makes the boundary between reusable strategy code and entrypoint orchestration explicit.