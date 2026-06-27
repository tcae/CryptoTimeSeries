# Dash dependency assessment 2026-06-24

## Scope

This note assesses the Dash.jl dependency as currently resolved in this workspace and its practical impact on CryptoTimeSeries.

## Current state

- Root workspace dependency declaration includes `Dash` and `DashTable`.
- The root workspace manifest currently resolves:
  - `Dash` = `1.5.0`
  - `DashTable` = `5.0.0`
  - `DashBase` = `1.0.0`
- The separate `scripts` environment is behind the root workspace and currently resolves:
  - `Dash` = `1.1.2`
  - `DashTable` = `5.0.0`
  - `DashBase` = `0.1.0`

## Upstream status

- The latest upstream Dash.jl release is `1.5.0`.
- Dash.jl `1.5.0` is the current tip-of-release upstream and includes Julia 1.10-related compatibility work.
- Upstream Dash.jl declares `HTTP = "1"` compatibility and `julia = "1.6"` compatibility.

## Workspace usage surface

- Dash usage appears limited to script and cockpit-style UI entrypoints under `scripts/`.
- No Dash usage was found in the package source trees that implement the core trading, exchange, feature, target, or strategy logic.
- Representative Dash entrypoints include:
  - `scripts/cryptocockpit.jl`
  - `scripts/DashHeatmapTest.jl`
  - `scripts/dashtest.jl`
  - `scripts/range_dash.jl`
  - `scripts/dashexperiments.jl`
  - `scripts/mljgist.jl`

## Assessment

### Root workspace

Assessment: low immediate dependency-update pressure.

Reasoning:

- The root workspace is already on the latest published Dash.jl release (`1.5.0`).
- There is no evidence from the current dependency state that a Dash.jl package upgrade is pending or required.
- Any remaining risk is therefore not primarily a "stale Dash version" problem in the root environment.

### Scripts environment

Assessment: moderate maintenance drift.

Reasoning:

- The separate `scripts` environment still resolves Dash.jl `1.1.2`, which is older than the root workspace's `1.5.0`.
- This creates version skew between the environment used for interactive dashboard scripts and the root environment.
- Even if there is no confirmed security issue specific to Dash.jl here, this drift increases the chance of behavior differences, callback incompatibilities, and duplicated troubleshooting effort.

### Operational exposure

Assessment: limited and localized.

Reasoning:

- Dash is used for local visualization and cockpit-style tooling rather than core trading or exchange transport paths.
- A Dash issue would be expected to impact monitoring and interactive UI workflows, not the core strategy, classifier, exchange, or order-management packages.
- The more security-relevant transitive surface behind Dash is `HTTP.jl`, but Dash.jl itself currently depends on the `HTTP 1.x` line. Any broader HTTP hardening discussion should therefore be treated separately from Dash version staleness.

## Important compatibility note

- The root workspace `Project.toml` still declares `julia = "1.5.3"`.
- Upstream Dash.jl requires `julia = "1.6"`.
- The active manifest in this workspace is on Julia `1.12.6`, so current local execution is consistent with Dash.jl.
- However, the declared root Julia compatibility is stale relative to the actual resolved dependency set and should not be treated as authoritative.

## Related workspace note: move to HTTP.jl 2.4.0

Assessment: high migration risk for the overall workspace; not a routine patch-level dependency update.

Concise clarification:

- The risk is not that the HTTP protocol itself changes.
- The risk is that `HTTP.jl 2.4.0` is a major client-library upgrade, so Julia-side request APIs, keyword behavior, retry semantics, redirect handling, websocket behavior, parsing strictness, and dependency compatibility may change even if the on-the-wire HTTP mechanisms remain broadly the same.

Reasoning:

- The workspace's exchange adapters are currently aligned to `HTTP.jl 1.11.0`, not the `2.x` line.
- A move to `HTTP.jl 2.4.0` would be a major-version migration and should be treated separately from the Dash.jl review.
- The most exposed areas are `Bybit`, `KrakenSpot`, and `KrakenFutures`, which use `HTTP.request(...)` directly and also rely on `HTTP.WebSockets` for public and private exchange streams.
- That means the practical impact of an `HTTP.jl 2.4.0` move would fall primarily on exchange transport behavior, retry behavior, websocket behavior, and authenticated request handling rather than on Dash-driven UI code.
- Dash.jl itself does not remove that risk because upstream Dash.jl still declares `HTTP = "1"` compatibility; updating the workspace HTTP stack to `2.4.0` would therefore need explicit compatibility review for Dash and likely broader resolver changes.
- The active local runtime is new enough for `HTTP.jl 2.4.0`, but the declared root Julia compatibility remains stale, so package metadata should be reconciled before treating such an upgrade as supported workspace policy.

Overall workspace conclusion for `HTTP.jl 2.4.0`:

- Do not treat a move to `HTTP.jl 2.4.0` as an immediate dependency hygiene task.
- Treat it as a planned migration with focused validation of exchange adapters and websocket paths.
- For this workspace, Dash is a secondary concern in that migration; the core risk is in network-facing trading and market-data packages.

## Recommendation

1. Keep the root workspace on Dash.jl `1.5.0`; no Dash.jl version bump is indicated from the current state.
2. Align the separate `scripts` environment to the same Dash.jl release line as the root workspace to remove avoidable version skew.
3. If dependency hygiene is the goal, prioritize reconciling the declared Julia compatibility and reviewing the transitive `HTTP.jl` exposure used by Dash and other network-facing packages.
4. Treat Dash issues as UI-surface issues unless there is evidence that a script is deployed as a long-running exposed service.

## Conclusion

Dash is not currently the main dependency concern in this workspace. The root environment is already on the latest upstream Dash.jl release, and the practical risk is mostly confined to script-based UI tooling. The main actionable issue is environment skew: the `scripts` environment lags behind the root environment and should be aligned if these dashboards are still actively used.