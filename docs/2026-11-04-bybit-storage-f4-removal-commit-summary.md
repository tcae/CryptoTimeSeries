# Bybit storage defaults and Features004 removal: commit summary

- Base commit: `75d21a2`
- Date: 2026-11-04
- Scope: storage-default migration, legacy feature removal, related tests and documentation.
- The pre-existing workspace configuration edit is outside this summary.

## Suggested commit message

```text
Default per-coin storage to Bybit and remove Features004

- Select Bybit for default storage and no-argument storage-root resets.
- Route legacy OHLCV conversion and cross-check explicitly to Bybit.
- Remove Features004 implementation, cache utilities and maintenance scripts.
- Preserve cockpit regression bands using standalone regression stores.
- Give Features005 an independent Arrow regression cache.
- Add storage, conversion and regression-band tests and update documentation.
- Bump EnvConfig to 0.4.0 and Features to 3.0.0.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

## Changes

### Exchange-specific storage

- [EnvConfig](../EnvConfig/src/EnvConfig.jl) defaults the per-coin root to
  `~/crypto/Bybit` instead of `~/crypto/coins`.
- `setcoinspath!()` without arguments resets to Bybit. Explicit exchange and
  test roots remain supported; exchange adapters still select their own roots.
- Path documentation now describes
  `~/crypto/<exchange>/<BASE>-<QUOTE>/` for OHLCV and derived stores.
- [EnvConfig's version](../EnvConfig/Project.toml) increases from `0.3.0` to `0.4.0`.

### Legacy OHLCV conversion

- [Jdf2Arrow](../scripts/Jdf2Arrow.jl) explicitly selects Bybit for shared
  conversion and cross-check, regardless of the previously active root.
- Shared conversion supports only OHLCV; unsupported artifact selections fail
  explicitly.
- Conversion reads the selected legacy JDF source directly, preserves its
  base/quote identity, and leaves the source untouched. It no longer uses
  `Ohlcv.read!`, which could select an existing Arrow cache or migrate/delete
  the legacy source.
- Existing destination OHLCV files are protected from overwrite. Conversion
  failures are logged and recorded as `status=error` in the summary.
- Shared cross-check remains a destination-presence and byte-size check, not
  value-by-value verification.

### Features004 removal

- Removed the Features004 module, its include, shared/split cache helpers,
  iterator, storage tests and legacy test usages.
- Removed the F4 regression-maintenance and split-cache migration scripts.
- Removed F4 conversion/cross-check support and obsolete F4 documentation.
- [Features' version](../Features/Project.toml) increases from `2.3.0` to `3.0.0`
  to reflect removal of the public implementation.

### Preserved feature and cockpit behavior

- [Features005](../Features/src/Features.jl) retains an independent per-pair
  `f5.arrow` cache with its own column names. It does not read or migrate
  retired F4 caches.
- Its reader reuses only a timestamp-aligned prefix of the current feature
  interval and checks required columns and timestamp consistency.
- Its regression-window accessor no longer references an F4 instance.
- [Regression bands](../Features/src/featurestore.jl) reuse standalone
  regression and standard-deviation stores when both timestamp vectors match
  the supplied minute OHLCV interval; otherwise they calculate in memory.
  The plotting helper never writes regression stores.
- [Cockpit](../scripts/cryptocockpit.jl) keeps its regression-band selectors
  and plots, using lazy per-window caching rather than Features004.
- Features006 remains supported and uses standalone regression stores.

### Documentation

- [Storage and conversion documentation](Jdf2Arrow.md) replaces the obsolete
  F4 migration design with the current exchange-specific layout and commands.
- [README](../README.md) uses Bybit paths in OHLCV transfer examples.
- OHLCV, feature-store and trade-selection descriptions no longer describe
  retired storage paths or F4 calculation.

## Validation performed

Successful runs:

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test("EnvConfig")'
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test("Features"); Pkg.test("Ohlcv")'
julia --project=. --startup-file=no -e 'include("test/jdf2arrow_test.jl"); Meta.parseall(read("scripts/cryptocockpit.jl", String)); println("Cockpit syntax parsed")'
```

- EnvConfig: 76 assertions.
- Features: 307 assertions, including 13 regression-band assertions and
  16 independent Features005 cache assertions.
- Ohlcv: 30 assertions.
- Legacy OHLCV conversion: 14 assertions covering Bybit routing, quote
  preservation, source retention, overwrite protection and unsupported input.
- Cockpit syntax parsed successfully; no browser or full cockpit runtime
  validation was performed.
- Diff whitespace checks passed with existing CRLF line endings respected.

No full workspace suite or end-to-end market refresh was run.

## Operational notes

- Existing data files were not moved or deleted by this change.
- Old Features004 caches are no longer read or migrated.
- Remaining files in `~/crypto/coins` require a separate retention/deletion
  decision; changing the default does not migrate their contents.
- The F09 config-hash mismatch was assessed separately. This change does not
  rebuild stale F09 stores or relabel their metadata.
