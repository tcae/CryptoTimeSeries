# Features006 extreme development inputs

Register four inputs before setting the OHLCV base:

```julia
f6 = Features.Features006()
Features.addextremes!(f6; bandratio=0.01, maxwindow=14400)
Features.setbase!(f6, ohlcv)
```

The inputs, in order, are minimum-relative regression slope (`minnrg`), minimum
age (`minage`), maximum-relative regression slope (`maxnrg`), and maximum age
(`maxage`). Slopes use ordinary least squares on pivots between the selected
extreme and the current sample, divided by that segment's mean pivot, in relative
change per minute. Ages are elapsed minutes, not sample counts. Single-sample
segments have zero slope and age. There are no offsets or slope-age products.

Each row is equivalent to replaying its trailing `maxwindow` samples, with both
anchors seeded at the first pivot. An opposite price move must strictly exceed
`bandratio` relative to the active extreme's price. A new active extreme
reestablishes only its own anchor, preserving the established opposite extreme.
The initial direction is selected by the first qualifying move between the running
minimum and maximum. Equal prices retain the earliest anchor.
The initial qualifying move selects the active side while preserving both running
anchors. Every later direction switch starts the new active extreme at the
crossing sample and preserves the preceding leg's endpoint as its opposite anchor.

For example, after 110 falls to 100 with a 1% band, a rebound to 100.5 does not
qualify an upward reversal. A subsequent fall to 99.8 updates only the minimum,
preserving the maximum at 110. A rise above 100.798 starts the upward leg; later
maximums preserve its established minimum at 99.8. The rules apply symmetrically.
Previously emitted rows never change. Semantic cache hashes reject stores generated
with the older opposite-anchor reset behavior.

`maxwindow` counts samples, so ages are at most `maxwindow - 1` minutes.
Registration accepts 2 through 14,400 samples, matching Features006's existing
window limit. Missing-minute gaps split the data into independent consecutive
sequences: anchors and regressions reset, with zero slopes and ages at each start.
Duplicate, reversed, and non-minute-aligned timestamps fail explicitly. No prices
are synthesized. Existing fixed-window features are unchanged.
Older history cannot influence an output. Range summaries skip band-contained
and monotone stretches during replay; transition-dense data can still require
work proportional to the window size per row. Regression range queries merge
centered statistics instead of subtracting large prefix sums.
Overlapping chunks keep summary memory proportional to `maxwindow`, rather than
the complete coin history.

Column names include the configuration (for example `minnrg+0.01+14400`) to
distinguish multiple registrations and persistent feature identities. Neither
parameter is a separate input column. Existing features retain their names.

`Features.confirmedextremes(pivots; bandratio)` provides occurrence indices,
directions, and confirmation indices for target generation. It uses the same
state transitions, does not emit the initial seed, and imposes no confirmation
deadline. This future-outcome metadata must not be used as causal classifier
inputs; it never modifies the four feature columns.
For finite target datasets, `sequenceended=true` additionally accepts the final
established opposite extreme at the sequence boundary without a confirming reversal.

## Regression accuracy

`rollingregression` preserves its arguments, endpoint/slope ordering, output
alignment, and floating input/output element types. Its shared numerical core
uses centered Float64 statistics and equidistant sample coordinates. Sliding
statistics are periodically rebuilt, giving linear runtime while limiting
roundoff accumulation. This also avoids Float32 cancellation in shallow trends.
Per-minute slope units remain unchanged.

Previously stored regression values are not rewritten automatically. Rebuild
existing regression stores and dependent feature artifacts when improved
numerical accuracy is required; old stored values retain their original accuracy.
