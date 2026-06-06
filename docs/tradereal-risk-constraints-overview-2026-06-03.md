# Tradereal Risk Constraints Overview (2026-06-03)

This note summarizes the currently implemented runtime constraints for live trading, with emphasis on KrakenFutures behavior.

## Goals

1. Bound total risk by one global quote-currency budget.
2. Bound per-symbol risk by a fractional cap of that global budget.
3. Enter reduce-only mode when margin health is weak.

## Constraint Inputs

- maxbudgetquote
  - Source: Trade cache runtime config.
  - Meaning: optional hard cap for overall budget in quote currency.
- budgetsafetymargin
  - Source: Trade cache runtime config.
  - Meaning: reserve fraction on available opening capacity.
- maxassetfraction
  - Source: Trade cache runtime config.
  - Meaning: per-symbol fraction cap relative to effective budget.
- marginhealth_reduceonly_threshold
  - Source: Trade cache runtime config, env override supported.
  - Meaning: if margin health <= threshold, opening trades are blocked.
- marginhealth_reduceonly_enabled
  - Source: Trade cache runtime config, env override supported.
  - Meaning: enable/disable margin-health gate.

## Effective Budget

effectivebudgetquote is computed as:

min(maxbudgetquote, available_opening_quote * (1 - budgetsafetymargin))

When maxbudgetquote is not set or invalid, budget defaults to the capacity-with-safety term.

## Exposure Model Used For Opening-Order Admission

Opening order checks now include all of the following:

1. Existing positions (gross non-quote exposure from portfolio rows).
2. Pending opening orders from current open-order snapshot.
   - Opening exposure order classes:
     - non-leverage buy
     - leverage sell
3. Per-cycle committed opening quote from newly accepted opening orders in the same cycle.

This prevents multiple opening advices in one cycle from bypassing limits before portfolio/open-order snapshots refresh.

## Hard Admission Constraints For Opening Orders

A candidate opening order is rejected if either condition is true:

1. Aggregate cap violation
   - positions_total + pending_open_total + cycle_committed_total + candidate_open_quote > effectivebudgetquote
2. Per-symbol cap violation
   - symbol_positions + symbol_pending_open + symbol_cycle_committed + candidate_open_quote > maxassetfraction * effectivebudgetquote

Close/reduce flows remain allowed so the robot can deleverage even when over budget.

## Margin Health Gate

margin_health = equity_quote / maintenance_margin_quote

Policy:

1. If marginhealth_reduceonly_enabled is false:
   - no reduce-only gating from margin health.
2. If maintenance_margin_quote > 0:
   - opening labels are blocked when margin_health <= marginhealth_reduceonly_threshold.
3. If maintenance_margin_quote <= 0 (unavailable/zero):
   - currently treated as healthy (no reduce-only activation from this condition).

Note:
- This behavior was intentionally changed on 2026-06-03 per user request.

## Reduce-Only Semantics In Trade Flow

- Opening labels:
  - longbuy, longstrongbuy, shortbuy, shortstrongbuy
  - blocked while reduce-only mode is active.
- Close labels:
  - longclose, longstrongclose, shortclose, shortstrongclose
  - still allowed.
- Close order submissions use reduceonly=true for both long and short close paths.

## Logging Signals

When constraints block actions, warning logs include explicit reason classes such as:

- skip opening trade due to reduce-only risk mode
- skip <base> longbuy due to aggregate opening exposure budget limit
- skip <base> longbuy due to per-symbol opening exposure limit
- skip <base> shortbuy due to aggregate opening exposure budget limit
- skip <base> shortbuy due to per-symbol opening exposure limit

## Open Questions / Pending Decisions

1. Should reduce-only mode also cancel existing opening orders immediately when activated?
2. Should we add hysteresis to reduce-only exit (for example, require health > threshold + buffer)?
3. Should maintenance_margin_quote <= 0 continue to be treated as healthy, or be configurable by dedicated policy mode?
4. Should additional hard-stop thresholds be added (for example emergency mode below a critical health value)?

## Summary

Current implementation now enforces:

1. Global budget cap for opening exposure including positions, pending openings, and in-cycle commits.
2. Per-symbol fractional cap including positions, pending openings, and in-cycle commits.
3. Margin-health reduce-only gate with maintenance-unavailable currently treated as healthy.
