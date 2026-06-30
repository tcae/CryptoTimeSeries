# TradingStrategy Lifecycle Refactor Plan

Date: 2026-06-30
Status: In progress, execution plan finalized, workspace tests validated

## Goal

Unify the runtime lifecycle used by Trade and TrendDetector so that TradingStrategy has explicit phases:

1. Initialize runtime once.
2. Prepare or reselect bases only when the base universe changes.
3. Process samples without hidden universe preparation.
4. Reset explicitly on teardown.

## Current State

Completed foundation:

- Trade and TradingStrategy runtime boundary cleanup is partially complete.
- Selection-time prepare versus tick-time process split is implemented for Trade.
- TradingStrategy replay preparation and replay processing are separated by explicit public helpers.
- TrendDetector replay gain evaluation now calls explicit replay lifecycle helpers.

Still open:

- Focused lifecycle invariant tests are not yet complete across all edge cases.
- Deterministic replay checks need final hardening and coverage integration.
- Workspace-level validation was run successfully (user-confirmed); coverage summarization and final documentation closure remain pending.

## Lifecycle Contract (Target)

### Trade runtime path

1. Trade performs `tradeselection!`.
2. Trade invokes `TradingStrategy.preparebases!` once for the selected base set.
3. Trade processes minute ticks via `gettradesrow!` or `gettradesrows!` without any hidden base preparation.
4. Trade calls explicit runtime reset on teardown.

### TrendDetector replay path

1. TrendDetector builds or loads replay runtime (`TsCache`).
2. TrendDetector invokes explicit replay preparation for the selected base universe and range.
3. TrendDetector invokes explicit replay processing for gain generation.
4. TrendDetector resets runtime explicitly between independent replay sessions.

## Non-Negotiable Invariants

- `preparebases!` must be selection-time only.
- Tick-time strategy processing must not mutate base universe membership.
- Runtime state must be deterministic for the same inputs and range.
- Invalid lifecycle usage must fail fast with actionable errors.
- `TradingStrategy.TsCache` remains the single strategy runtime object.
- `Xch` remains owner of mutable Trades DataFrames.

## Work Breakdown (Remaining)

### Work package A: TradingStrategy lifecycle assertions

Objective: codify and enforce lifecycle usage rules.

Tasks:

- Add explicit assertions around unprepared runtime usage in tick-time processing.
- Add explicit assertions around replay processing without replay preparation.
- Ensure assertion messages include base set and relevant runtime counters.
- Verify wrappers (`gettradesrow!`, `gettradesrows!`) stay thin and side-effect bounded.

Done criteria:

- All lifecycle misuse paths produce deterministic assertion failures.
- No hidden prepare path is reachable from tick-time processing.

### Work package B: Trade lifecycle integration tests

Objective: prove selection-time preparation and tick-time execution split.

Tasks:

- Add tests for initial prepare after first `tradeselection!`.
- Add tests for scheduled reselection refresh invoking `preparebases!` exactly once per universe change.
- Add tests that minute-by-minute advice collection does not call prepare.
- Add tests for teardown reset semantics.

Done criteria:

- Tests fail if prepare is called from tick-time path.
- Tests pass for unchanged and changed base universes.

### Work package C: TrendDetector replay determinism tests

Objective: prove replay lifecycle stability and deterministic outputs.

Tasks:

- Add deterministic replay tests for config `046` with `SINE` and `DOUBLESINE`.
- Add tests for malformed replay state (missing replay columns, insufficient `closeprices`) with fail-fast assertions.
- Add repeated-run checks to ensure byte-stable or semantically identical gain outputs.

Done criteria:

- Two consecutive runs on identical input produce identical gain summary metrics.
- Replay misuse paths fail with explicit assertion text.

### Work package D: Workspace-level validation

Objective: close the refactor with integrated quality checks.

Tasks:

- Run workspace tests with coverage.
- Run coverage summarization.
- Inspect for any lifecycle-related regressions in Trade, TradingStrategy, TrendDetector paths.
- Update this document with final execution notes and residual risk list.

Done criteria:

- Coverage/test pipeline is green.
- No lifecycle regression found in package and workspace entrypoints.

## Validation Matrix

### Package-level

- Trade tests cover selection-time prepare, tick-time processing, teardown reset.
- TradingStrategy tests cover lifecycle assertions and replay preparation contract.
- TrendDetector tests cover explicit replay lifecycle and deterministic outcomes.

### Workspace-level

- Run task: Julia: Run tests with coverage.
- Run task: Julia: Summarize coverage.
- Optional combined task: Julia: Run tests with coverage and summarize.

## Risks and Mitigations

- Risk: hidden runtime mutation reappears through convenience wrappers.
	- Mitigation: explicit assertion tests and wrapper side-effect checks.
- Risk: replay output drift due to implicit state carry-over.
	- Mitigation: explicit reset between sessions and repeated deterministic replay tests.
- Risk: broad test matrix hides lifecycle regressions.
	- Mitigation: focused lifecycle testsets run before workspace matrix.

## Exit Criteria

This plan is complete when all items below are true:

- Work packages A through D are marked done.
- Lifecycle invariant tests pass in TradingStrategy and Trade.
- TrendDetector replay determinism tests pass for configured synthetic patterns.
- Workspace coverage/test tasks complete successfully.
- This document is updated from In progress to Complete with final evidence notes.

## Execution Order

1. Implement work package A.
2. Implement work package B.
3. Implement work package C.
4. Execute work package D.
5. Update this document with final evidence and status change.

## Completion Notes (to fill when done)

- Date completed:
- Commits/PR:
- Test commands run: workspace test entrypoint executed successfully (user-confirmed on 2026-06-30).
- Coverage summary:
- Residual risks:
