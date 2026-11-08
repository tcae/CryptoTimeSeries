# Trend07: bounded directional first-passage targets

`Trend07(bandratio=0.01, openwindow=120, maxwindow=240)` uses the four matching
Features006 extreme columns. It does not wait for confirmation or
reject an entire trend because its extreme-to-extreme leg exceeds `maxwindow`.

For each minute sample, fix its minimum and maximum anchors from the causal
feature ages. A long candidate requires positive minimum-relative regression
slope; a short candidate requires negative maximum-relative slope. Search future
pivots for the first return >= `bandratio` or <= `-bandratio`, respectively.

- Long is invalidated by a future pivot **at or below** the fixed minimum before
  success. Short is invalidated by a pivot **at or above** the fixed maximum.
- Opens require success within `openwindow` elapsed minutes and the corresponding
  sign of the existing five-sample regression ending at the current sample.
  All five samples must be available in the same contiguous minute sequence.
- Holds require success within `maxwindow` elapsed minutes, without the recent
  regression condition. A hold can therefore occur within `openwindow`.
- Open takes priority over hold. Otherwise the direction whose valid threshold
  hit occurs first wins. No qualifying candidate means `allclose`.

Both time limits include their exact endpoint: `openwindow=120` includes the
sample 120 minutes after the current sample. Require
`2 <= openwindow <= maxwindow <= 14400`. Relative gains are measured from the
current pivot to the first valid hit, not to an eventual extreme.

Missing-minute gaps reset feature anchors and recent regressions, and terminate
forward searches. No dependency crosses a gap. In-memory labels use all available
future observations. Reusable stores conservatively mark the final `maxwindow`
rows of the last contiguous sequence as `ignore`, even when an early hit exists;
completed sequences before gaps are final. Extensions recalculate that trailing
region with enough backward context for exact extreme replay and recent regression.

## Registered experiment

T28 uses `bandratio=0.01`, `openwindow=120`, and `maxwindow=240`.
`ts=071` combines **F22-T28-P02-M011-B1** with **S09**.
F22 supplies the bounded causal extreme inputs.

Trend07 declares a finite 240-minute forward horizon for T28 and uses the common
shared-store, partition-gap, and RegimeTraining purge paths, including test mode.
Features have at most `maxwindow - 1` extreme lookback intervals and four recent
regression lookback intervals; other F22 features retain their own windows.
Targets can be calculated independently of calendar blocks. Start a new ts=071
experiment; do not reuse checkpoints trained with different target semantics.
