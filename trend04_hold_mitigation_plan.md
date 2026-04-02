# Trend04 Hold Suppression Investigation Plan

## Baseline Status
- Baseline branch is crosscheck-valid for SINE and BTC.
- No parser/lint errors in Targets/src/Targets.jl.

## Evidence Collected (Diagnostics)
### SINE strict buy10 hold1
- labels: LB=470, LH=0, SB=961, SH=0, AC=9
- hold candidates: long=445, short=960
- hold accepted: long=0, short=0
- dominant rejections:
  - rej.shorthold.reversal.threshold=768
  - rej.longhold.reversal.threshold=275
  - rej.shorthold.continues_shortbuy=192
  - rej.longhold.continues_longbuy=170

### SINE relaxed buy3 hold0.5
- labels: LB=470, LH=0, SB=961, SH=0, AC=9
- hold candidates: long=446, short=972
- hold accepted: long=0, short=0
- dominant rejections:
  - rej.shorthold.reversal.threshold=624
  - rej.shorthold.continues_shortbuy=348
  - rej.longhold.continues_longbuy=314
  - rej.longhold.reversal.threshold=132

### BTC full (2017-08 to 2026-03)
- labels: LB=1,635,600, LH=14, SB=1,459,652, SH=176, AC=1,415,562
- hold candidates: long=1,825,057, short=1,562,261
- hold accepted: long=3, short=82
- dominant rejections:
  - rej.longhold.reversal.threshold=1,684,725
  - rej.shorthold.reversal.threshold=1,443,523
  - rej.longhold.continues_longbuy=140,290
  - rej.shorthold.continues_shortbuy=118,574

## Root-Cause Hypotheses (Ranked by Evidence)
1. Reversal-path hold threshold is too strict for observed micro-structure.
   - Strong signal: reversal-threshold rejections dominate in all scenarios.
2. Buy-continuation precedence suppresses hold transitions.
   - Strong signal: continues_longbuy / continues_shortbuy are large second-order rejection classes.
3. Hold transition gate uses comparison points that are too close to current bar.
   - Medium signal: from_longbuy/from_shortbuy threshold rejections exist but are much smaller than reversal and continuation precedence.

## Mitigation Options To Evaluate
### Option A: Reversal Hold Hysteresis
- Change: on reversal-path hold checks, use a small hysteresis factor around hold threshold.
- Example: accept longhold when rd >= longhold - eps, shorthold when rd <= shorthold + eps.
- Risk: can reintroduce stale hold lock if eps is too large.

### Option B: Continuation Priority Softening (Preferred First)
- Change: when continuation still qualifies as buy, allow hold if buy edge over threshold is very small and trend is not making meaningful new extreme.
- Example: if buy threshold is met but edge < delta and local retrace is present, emit hold instead of buy.
- Benefit: directly addresses second-largest rejection class without broad threshold changes.

### Option C: Reversal Anchor Adjustment
- Change: evaluate reversal hold threshold against the last directional extreme anchor instead of immediate transition anchor.
- Benefit: may convert many reversal threshold rejects to valid holds.
- Risk: can bias toward stale hold if not bounded.

### Option D: Combined Conservative Patch
- Apply B first, then tiny A only if hold counts remain near zero.
- Keep C as fallback due to stale-anchor risk.

## Evaluation Matrix
For each option:
1. Crosscheck must remain valid on:
   - scripts/trend04crosscheck.jl
   - scripts/trend04crosscheck.jl BTC
   - scripts/trend04crosscheck.jl BTC 1200 1290
2. Hold acceptance metrics must improve materially:
   - accepted/candidate ratio in diagnostics must increase for at least one synthetic scenario.
3. No major drift in label distribution:
   - avoid collapse to mostly hold or mostly allclose.

## Proposed Execution Order
1. Implement Option B minimally in Trend04 _filltrendanchor!.
2. Run diagnostics + crosschecks.
3. If hold remains near zero, add very small Option A hysteresis (single scalar).
4. Re-run diagnostics + crosschecks.
5. If needed, trial Option C in a guarded branch with bounded window.

## Commands
- Baseline diagnostics:
  - julia --project=. scripts/trend04_hold_diagnostics.jl
- Crosschecks:
  - julia --project=. scripts/trend04crosscheck.jl
  - julia --project=. scripts/trend04crosscheck.jl BTC
  - julia --project=. scripts/trend04crosscheck.jl BTC 1200 1290
