## Final consolidated design note

Below is the consolidated **Phase 1 + Phase 2 storage and migration design**.

> **Status update (2026-04-11)**
> - Active shared storage is now **Arrow-first** under `~/crypto/coins/<BASE>-<QUOTE>/`.
> - `ohlcv` and `F4` use the consolidated files `ohlcv.arrow` and `f4.arrow`.
> - Any remaining `JDF` wording below should be read as **legacy/backfill context only**, not as the current hot-path format.

---

# 🎯 Overall storage strategy

Use a **hybrid model**:

## A. Shared reusable data → `coins/`
For data reused across experiments and independent of config:

- ohlcv
- `F4` features

Example:

```text
coins/
  BTC-USDT/
    ohlcv.arrow
    f4.arrow
  ETH-USDT/
    ohlcv.arrow
    f4.arrow
```

### Rationale
- reused across many experiments
- expensive to recalculate
- should not be duplicated per config

---

## B. Config-specific experiment data → `logs/<config-folder>/`
For all derived artifacts that belong to one experiment run/config:

- `F6` features
- targets
- `results`
- `predictions`
- `trades`
- `summary`

Example:

```text
logs/Trend-009-training/
  features/
    BTCUSDT.arrow
    ETHUSDT.arrow
  targets/
    BTCUSDT.arrow
  results/
    BTCUSDT.arrow
  predictions/
    BTCUSDT.arrow
  trades/
    BTCUSDT.arrow
  summary.arrow
  confusion.arrow
  xconfusion.arrow
```

### Rationale
- matches workflow: `config -> experiment -> assess -> refine`
- easy to remove one config folder completely
- avoids clutter at top level

---

# Phase 1 — Arrow pilot for `TradeAdviceLstm`

## Scope
Only `TradeAdviceLstm` and its immediate upstream inputs.

## Goal
Assess the **memory footprint gain** from Arrow before broader migration.

## Phase 1 actions
1. add Arrow dependency and helpers
2. create JDF → Arrow copies for:
   - trend features
   - `results`
   - `maxpredictions`
3. read these through Arrow-backed loaders
4. keep config-specific outputs under `logs/<config>/...`
5. keep summary as a single overview file

## Expected result
- lower peak memory than current ~13 GB
- low-risk pilot
- direct evidence for deciding Phase 2

---

# Phase 2 — workspace migration

## 2A. Shared base data migration
Move reusable data to `coins/`:

- ohlcv per coin
- `F4` per coin as a single shared `f4.arrow` table

Example:

```text
coins/BTC-USDT/ohlcv.arrow
coins/BTC-USDT/f4.arrow
```

### Important rule
- `F4` is stored once per coin in a single shared `f4.arrow` table
- `F6` is **not** split per column; it stays config-specific

---

## 2B. Features redesign
- keep current expensive F4 generation semantics
- store F4 outputs as a reusable per-coin Arrow table
- adapt `F6` to select the required F4 columns from that shared table instead of recalculating/taking over `F004`

This avoids repeated expensive feature computation.

---

## 2C. Classify adaptation
### Sequence models
- naturally consume ordered per-coin sequences

### Non-sequence classifiers/regressors
Need one of:
- a global cross-coin sample index for shuffle
- or per-coin shuffle plus mixed iteration

Either is valid; the first is more statistically clean, the second simpler.

---

## 2D. TradingStrategy persistence
Store trade-pair results:

- **per coin**
- but under the **config-specific** folder

Example:

```text
logs/Trend-009-training/trades/BTCUSDT.arrow
```

Summary remains:

```text
logs/Trend-009-training/summary.arrow
```

---

## 2E. Historical migration
Create a migration utility to convert existing JDF data to Arrow:

- do not delete old JDF immediately
- keep fallback support during transition
- validate row counts and key alignment

---

# Workspace impact summary

## High impact
- EnvConfig
- Features
- Targets
- Classify
- TradingStrategy
- TrendDetector.jl
- TradeAdviceLstm.jl
- BoundsEstimator.jl

## Medium impact
- dashboard / cockpit / analysis scripts
- backtest result readers

## Low impact
- model internals that are storage-agnostic

---

# On creating a workspace/code index

## Recommendation
**Yes — before Phase 2.**

### Why
Phase 2 is broad and touches many file readers/writers and assumptions about:
- JDF paths
- monolithic tables
- result discovery
- config folder structure

### Practical advice
- **not necessary for Phase 1**
- **recommended before starting Phase 2 impact analysis**

---

# Final architecture principle

## Shared immutable/reusable
- `coins/`
- ohlcv
- `F4` columns

## Experiment/config-specific and disposable
- `logs/<config>/`
- `F6`
- targets
- `results`
- `predictions`
- `trades`
- `summary`

---

# Recommended next step

Proceed with **Phase 2**:

1. refresh the workspace/code index and the JDF usage inventory
2. move shared reusable `ohlcv` and `F4` storage toward `coins/`
3. adapt `Features`, `Targets`, and `Classify` to the new storage model
4. keep the remaining `TradeAdviceLstm` run-control cleanup as non-blocking follow-up work

---

# Implementation checklist with target files and milestones

## Phase 1 — Arrow pilot for `TradeAdviceLstm`

### Milestone P1.1 — storage abstraction and Arrow dependency
**Goal:** introduce Arrow read/write support without breaking existing JDF workflows.

**Target files:**
- `EnvConfig/Project.toml`
- `EnvConfig/src/EnvConfig.jl`
- optionally root `Project.toml` / `Manifest.toml`

**Checklist:**
- add `Arrow` as dependency in `EnvConfig`
- keep `savedf(...)` and `readdf(...)` as the primary API, but add a `format` parameter with a default that can later be switched globally
- add storage helpers such as:
  - `tablepath(...)`
  - `tableexists(...)`
  - internal format-aware load/save helpers as needed
- keep `JDF` as fallback during transition
- define conventions for config subfolders:
  - `features/`
  - `targets/`
  - `results/`
  - `predictions/`
  - `trades/`

**Acceptance criteria:**
- existing JDF-based workflows still run
- Arrow write/read roundtrip works for a sample dataframe

---

### Milestone P1.2 — JDF → Arrow conversion utilities
**Goal:** allow one-time or repeated conversion of large config-specific artifacts.

**Target files:**
- `scripts/Jdf2Arrow.jl` or similar new migration utility
- `EnvConfig/src/EnvConfig.jl`
- optional notes in `Jdf2Arrow.md`

**Checklist:**
- create conversion command for selected folders
- convert at least the main artifact families:
  - `features` → `features.arrow`
  - `results` → `results.arrow`
  - `maxpredictions` → `maxpredictions.arrow`
- write Arrow copies into config-scoped subfolders rather than cluttering the root folder
- verify row counts and key columns after conversion
- do not delete original JDF files

**Acceptance criteria:**
- conversion runs without data loss
- converted Arrow tables can be reloaded and validated against JDF row counts

---

### Milestone P1.3 — switch `TradeAdviceLstm` input path to Arrow-backed loading
**Goal:** reduce peak memory for the LSTM pipeline.

**Target files:**
- `scripts/TradeAdviceLstm.jl`
- `Classify/src/Classify.jl`
- `Classify/src/Classifier016.jl`
- related tests in `Classify/test/lstm_phase2_training_test.jl`

**Checklist:**
- prefer Arrow-backed reads for `features`, `results`, and `maxpredictions`
- avoid full wide `DataFrame` materialization when only selected columns are needed
- keep batched hidden feature extraction and batched LSTM window generation
- store config-specific intermediate outputs in subfolders
- preserve the single config-level `summary`

**Acceptance criteria:**
- `Classify/test/lstm_phase2_training_test.jl` passes
- the first fresh training run should use
  ```bash
  julia --project=. scripts/TradeAdviceLstm.jl trend=009 configname=trend009lstm folder=TradeAdviceLstm-trend009-training maxepoch=1 batchsize=16 train retrain
  ```
  and should run with lower peak memory than the current JDF path
- subsequent evaluation/regeneration runs may omit `retrain` and reuse the saved checkpoint

---

### Milestone P1.4 — memory benchmark and decision gate
**Goal:** decide whether Arrow alone gives enough benefit to justify broader migration.

**Target files:**
- `Jdf2Arrow.md`
- optional benchmark notes under the relevant log folder

**Checklist:**
- record peak memory before and after Arrow pilot
- compare runtime and memory footprint
- note whether remaining pressure comes from storage loading or from model/training tensors

**Acceptance criteria:**
- documented benchmark result
- explicit decision: continue to full Phase 2 or refine Phase 1 further

**Status note (2026-04-08):**
- memory footprint improved substantially in the Arrow-backed `TradeAdviceLstm` path
- no evidence that memory is the remaining blocker
- decision: move forward with **Phase 2** and take the LSTM adaptation out of the critical path

---

## Phase 2 — workspace migration

### Milestone P2.1 — workspace impact analysis and indexing
**Goal:** identify all JDF assumptions before widespread changes.

**Target files / areas to inspect:**
- `EnvConfig/`
- `Features/`
- `Targets/`
- `Classify/`
- `TradingStrategy/`
- `scripts/TrendDetector.jl`
- `scripts/BoundsEstimator.jl`
- `scripts/TradeAdviceLstm.jl`
- dashboard/cockpit scripts

**Checklist:**
- create or refresh workspace/code index
- inventory all `JDF.loadjdf`, `savejdf`, `savedf`, `readdf` usage
- classify each artifact as either:
  - shared reusable (`coins/`)
  - config-specific (`logs/<config>/`)

**Acceptance criteria:**
- migration inventory exists
- high-impact files are identified before edits begin

---

### Milestone P2.2 — move shared reusable data to `coins/`
**Goal:** separate shared data from experiment-specific data.

> **Status update (2026-04-11):** implemented. Shared `OHLCV` and `F4` now live under `coins/` as `ohlcv.arrow` and `f4.arrow`. Legacy JDF caches are retained only for fallback/backfill.

**Shared reusable targets:**
- `ohlcv`
- `F4` features only

**Target files:**
- `Ohlcv/src/Ohlcv.jl`
- `Features/src/Features*.jl`
- `EnvConfig/src/EnvConfig.jl`

**Checklist:**
- create `coins/` as sibling to `logs/`
- store `ohlcv` per coin in Arrow
- store `F4` per coin as one shared Arrow table, e.g.:
  - `coins/BTC-USDT/ohlcv.arrow`
  - `coins/BTC-USDT/f4.arrow`
- keep these independent of experiment/config folders

**Acceptance criteria:**
- shared data can be reused across experiments without duplication
- `ohlcv` and shared `F4` can be loaded coin-by-coin

---

### Milestone P2.3 — adapt `Features` to reuse F4 column storage
**Goal:** avoid recomputation of expensive F4 features while keeping F6 config-specific.

> **Status update (2026-04-11):** implemented. Shared `OHLCV` / `F4` readers now load the local Arrow caches under `coins/` first; JDF is legacy fallback only.

**Target files:**
- `Features/src/Features004.jl`
- `Features/src/Features.jl`
- any F6-related feature selection code

**Checklist:**
- persist F4 outputs as one reusable per-coin `f4.arrow` table
- adapt F6 to read the required F4 columns from that shared table instead of recalculating/taking over a monolithic F004 object
- keep F6 outputs config-specific under `logs/<config>/features/`
- do **not** split F6 per column

**Acceptance criteria:**
- F6 consumes persisted F4 data correctly
- repeated experiments avoid recomputing expensive shared features

---

### Milestone P2.4 — adapt `Targets` and `Classify`
**Goal:** migrate training inputs to the new storage model.

**Target files:**
- `Targets/src/Targets.jl`
- `Classify/src/Classify.jl`
- classifier-specific files in `Classify/src/`
- test files under `Targets/test/` and `Classify/test/`

**Checklist:**
- keep `targets` config-specific under `logs/<config>/targets/`
- update `Classify` readers to load per-coin config-scoped features/targets/results
- for non-sequence models, decide and implement one shuffle strategy:
  - cross-coin sample index, or
  - per-coin shuffle plus mixed iteration
- for sequence models, preserve per-coin order and range locality

**Acceptance criteria:**
- classification/regression training still works
- tests pass with new storage-backed loaders

---

### Milestone P2.5 — adapt `TradingStrategy` and result persistence
**Goal:** align per-coin trade results with the config-specific storage model.

> **Status update (2026-04-08):** implemented. `TrendDetector` and `TradeAdviceLstm` now persist trade outputs under `logs/<config>/trades/`, with aggregate `*_all` files for compatibility and per-coin copies beneath the corresponding `trades/<artifact>/` folder.

**Target files:**
- `TradingStrategy/src/TradingStrategy.jl`
- `TradingStrategy/test/`
- result/analysis scripts under `scripts/`

**Checklist:**
- persist trade-pair results per coin under:
  - `logs/<config>/trades/<coin>.arrow`
- keep config-level summary as a single overview file
- update result readers and backtest reporting code accordingly

**Acceptance criteria:**
- trade simulations still work
- summaries remain easy to inspect and compare

---

### Milestone P2.6 — read-only Arrow loading policy
**Goal:** keep Arrow's runtime and memory advantages by preferring read-only loads and copying only at mutation boundaries.

> **Status update (2026-04-08):** started. `EnvConfig.readdf` / `readtable` now expose an explicit `copycols` switch so mutable working copies are opt-in rather than the default.

**Target files:**
- `EnvConfig/src/EnvConfig.jl`
- mutation-heavy callers in `Trade`, `Classify`, and result scripts
- `EnvConfig/test/` and package tests that exercise storage-backed mutation

**Checklist:**
- restore read-only Arrow materialization as the default behavior
- add an explicit `copycols=true` option for callers that need to mutate loaded dataframes
- patch known mutation sites (`TradeConfig`, classifier simulation/prediction repair, similar working tables)
- keep pure read paths cheap and lazy where possible
- verify memory/runtime-sensitive training and regression paths still work with Arrow

**Acceptance criteria:**
- Arrow-backed reads remain memory efficient by default
- mutable workflows succeed only where they explicitly opt into copying
- storage regression tests cover both read-only and mutable cases

---

### Milestone P2.7 — historical migration and deprecation plan
**Goal:** preserve all existing JDF data while moving to Arrow.

> **Status update (2026-04-09):** implemented and smoke-verified. `scripts/Jdf2Arrow.jl` now supports historical backfill scans across log folders, report-only discovery of legacy JDF artifacts, and targeted batch conversion for selected config folders. Verified on `Trend-029-test` and `Bounds-001-test`: the initial scan found 6 legacy artifact rows needing backfill, and the follow-up report showed 0 pending after conversion.

**Target files:**
- new migration scripts under `scripts/`
- `EnvConfig/src/EnvConfig.jl`
- documentation in `Jdf2Arrow.md`

**Checklist:**
- convert historical JDF datasets to Arrow
- preserve original JDF files during a transition period
- make Arrow preferred, JDF fallback
- define the eventual deprecation/removal plan for JDF-only paths
- support batch discovery and backfill of legacy config folders under `~/crypto/logs`

**Execution notes:**
- report pending historical backfills:
  ```bash
  julia --project=. scripts/Jdf2Arrow.jl scan=true reportonly=true missingonly=true
  ```
- backfill selected folders:
  ```bash
  julia --project=. scripts/Jdf2Arrow.jl scan=true folders=Trend-029-test,Bounds-001-test
  ```
- backfill one folder explicitly:
  ```bash
  julia --project=. scripts/Jdf2Arrow.jl folder=Trend-029-test artifacts=features,results,maxpredictions
  ```

**Acceptance criteria:**
- no historical experiment data is lost
- new code can read both legacy and migrated runs during transition
- historical config folders can be scanned and backfilled without deleting the original JDF caches

***Cross check
scans the real CloudStorage JDF sources in
crypto/Features/OHLCV and crypto/Features/Features004
checks that each JDF cache has an Arrow counterpart under ~/crypto/coins
compares recursive byte sizes
flags each item as:
ok
missing-arrow
size-mismatch
empty-source

Usage
Full check: julia --project=. scripts/Jdf2Arrow.jl crosscheck=true
Focused check: julia --project=. scripts/Jdf2Arrow.jl crosscheck=true artifacts=ohlcv,f4 bases=BTC,ETH savesummary=false

---

## Suggested execution order

1. `EnvConfig` Arrow abstraction
2. JDF → Arrow conversion utility
3. `TradeAdviceLstm` Arrow pilot and benchmark
4. workspace index / impact analysis
5. `coins/` storage for `ohlcv` and `F4`
6. `Features` redesign for F4/F6 separation
7. `Targets` and `Classify` migration
8. `TradingStrategy` and reporting migration
9. read-only Arrow audit and mutation-boundary copies
10. historical data conversion and cleanup policy

---

## Definition of done

The migration is considered complete when:
- shared reusable data (`ohlcv`, `F4`) lives under `coins/`
- config-specific outputs live under `logs/<config>/...`
- `TradeAdviceLstm` and other main scripts prefer Arrow-backed loading
- JDF legacy data remains readable during the transition
- benchmarked memory usage is clearly improved
- summary files remain simple single-file config overviews