# F09-T24-P02-M008-B1-CS01-D01 investigation

Investigation date: 2026-10-21.

## Objective and scope

The classifier serves a PPV-first entry strategy. The important predictions are
`longopen` and `shortopen`: the strategy seeks turning points in a trend that
meanders around its regression line. Hold and close predictions are not reliable
enough to drive actions and would generate excessive trades and fees.

Relative volume is intended as a rough indication that a run is starting. Its
precise magnitude at very large ratios is not important: distinguishing a 100x
peak from a 10,000x peak is not an objective.

The current BatchNorm architecture reflects prior experiments with no
normalization, input-only normalization, and normalization at successive layers.
It should be retained for the first feature-stabilization experiment. The
diagnostics below do not establish that removing BatchNorm improves trading
performance.

The initial investigation was read-only apart from this document. Following the
agreements below, feature stabilization and new configurations were implemented.
No production training process or saved classifier was modified by this work.

## Evidence and timing

- Classifier artifacts: `~/crypto/classifiers/F09-T24-P02-M008-B1-CS01-D01`.
- Initial investigation used the best checkpoint saved October 20 at 20:51 and
  last-epoch checkpoint saved October 21 at 01:20.
- The original best checkpoint also exists as `nn.bson initial training best`.
- Saved confusion and extended-confusion tables were dated October 21 at 01:36.
  They describe the earlier evaluation, not subsequent reinforced checkpoints.
- The full feature scan below was performed later on October 21, after the
  retraining had started. The matrix had 100,273,418 observations and seven
  features. File size and modification time were unchanged during the scan.
- Feature extrema, nonfinite counts, and threshold exceedance counts are exact
  over that matrix. Quantiles are estimates from 262,144 randomly sampled
  observations with a fixed seed of 713.
- In-memory normalization experiments used 131,072 randomly sampled training
  observations with seed 713. They are diagnostic training-sample results, not
  held-out performance measurements.

Later checkpoint updates must not be confused with the historical loss and
evaluation evidence recorded here.

## 1. Feature value volatility: first topic to address

### Full-matrix ranges

| Feature | Minimum | Maximum | Rows with absolute value >100 | Rows with absolute value >1,000 |
|---|---:|---:|---:|---:|
| `rg+5+0` | -2,170.97 | 1,602.335 | 17,506 | 39 |
| `rg+15+0` | -1,296.5752 | 952.0381 | 3,792 | 6 |
| `rg+60+0` | -416.55612 | 225.02773 | 401 | 0 |
| `rg+240+0` | -96.683395 | 70.78096 | 0 | 0 |
| `maxd+60+0` | -2.4254344 | 0 | 0 | 0 |
| `mind+60+0` | 0 | 0.9999643 | 0 | 0 |
| `rv+5+360+0` | 0 | approximately 1.288e13 | 43,495 | 11,424 |

All seven features contain zero nonfinite values. Only relative volume has
absolute values above one million or one billion:

| Relative-volume threshold | Rows exceeding threshold | Share of training rows |
|---|---:|---:|
| 100 | 43,495 | 0.04338% |
| 1,000 | 11,424 | 0.01139% |
| 1,000,000 | 9,229 | 0.00920% |
| 1,000,000,000 | 3,157 | 0.00315% |

Sampled relative-volume median, 99th percentile, and 99.9th percentile were
approximately 0.965, 12.75, and 58.37.

The maximum relative-volume row maps through the saved training index to PENGU,
feature row 178,833. The largest absolute gradients for all four windows map to
YFI. These origin mappings assume the retained index matches the compact matrix;
the row counts agree, but source rows were not independently recomputed.

### Interpretation

Relative volume is the only feature exhibiting the observed million-to-trillion
scale explosion. The implementation divides a short-window volume median by a
long-window median. A zero or sufficiently small long-window median is replaced
with machine epsilon, allowing an ordinary positive numerator to produce a huge
finite ratio. Finite-value validation alone therefore does not catch this issue.

The gradient features also span wide ranges, but they currently use raw
price-unit slopes. Cross-coin price differences contribute to their scale: a
large YFI slope is not directly comparable to the same numeric slope in a
low-priced coin. Their extrema alone do not establish corrupt market data or
prove that these features cause the catastrophic loss spikes.

The price-distance features have no comparable numerical explosion. Their
unusual extremes may merit a later source-data check; no data-quality conclusion
was established from those extrema alone.

Relevant implementation:

- [F09 configuration](../TradingStrategy/src/tradingstrategyconfig.jl).
- [Volume calculation and Features006 assembly](../Features/src/Features.jl):
  `relativevolume`, `_relvol!`, and `supplement!`. The historical epsilon-based
  `correctunworkablevolume` helper was removed with the agreed implementation.

### Agreed implementation

Keep the current M008 model and BatchNorm layout. Relative volume is capped at
**60**, superseding the initial experimental suggestion of 100. This pragmatic
modeling threshold retains a rough peak indicator without distinguishing
exceptionally large ratios.

The shared `relativevolume` calculation retains the ordinary long-window median
when positive. Only if it is zero does it use the median of strictly positive
volumes from that same trailing long window. An entirely inactive window yields
zero. For example, 355 zeros followed by five volumes of 10 produce a ratio of
one for short=5, long=360, rather than an artificial peak. This does not guarantee
that every onset sequence starts at one: the short median can remain zero or
differ from the positive-volume baseline. No future observations are used.
Tiny positive medians remain real baselines, rather than being replaced by
machine epsilon. Leading partial windows use the available history.

Features006 applies the cap in `_relvol!` to the assembled vector, including
cached, prepended, and appended values. Its behavior is shared by offline
training and runtime feature generation.

A separate `nrg` feature is defined as raw `rg / mean_pivot`, in relative change
per minute, where `mean_pivot` is the mean pivot of the regression window.
`addnormgrad!` requires the raw regression columns; `_normgrad!` creates the
normalized no-offset column in `fdfno`. `supplement!` selects offset views
without normalizing or mutating raw columns. Negative slopes remain intentional.

A least-squares line passes through the window mean, so for full windows
`mean_pivot = regry - rg * (window - 1) / 2` (`regressionmeanpivot`) without an
additional pass over the prices. The first `window - 1` rows of a calculation
range use the mean of their available pivots, because their regression may come
from a longer cached history. With positive prices the mean is strictly
positive, and `|nrg| <= 6 / (window + 1)` holds by construction: about 1.0,
0.375, 0.098, and 0.025 for windows 5, 15, 60, and 240. No cap is needed.
Nonfinite slopes, nonpositive/nonfinite mean pivots, and nonfinite results fail
explicitly with diagnostic assertions; with positive prices they indicate
corrupted data.

The first definition `rg / regry` was rejected after `trend ts=056 train` failed
for CTC at 2022-09-21T11:30. A single trade printed 0.00008 after a constant
0.3036, the price stayed there for 53 minutes with a volume of 4, and then
recovered. The window-5 regression extrapolated `regry=-0.0303` from positive
prices. The cached regression stores contained non-positive `regry` for CTC,
LUNA, LUNC, MAVIA, OP, SERAPH, and WIF (3 to 1,505 rows per window), mostly from
real crashes. The last pivot was also considered as a denominator. It is always
positive, but it is unbounded: at the CTC glitch it gives about -1,100 per minute
instead of -0.60 with the mean pivot. In normal markets both denominators differ
by only about `nrg * (window - 1) / 2`.

The CTC glitch itself is accepted as noise for now (53 of about 1.37M rows).
A general data-quality check for extreme price jumps on near-zero volume is
deferred.

Raw regression stores, `rg`, and `regry` are unchanged. Consequently the
TradingStrategy hourly slope gate, regression deviations, and target geometry
continue using raw price-unit slopes. Trade has no direct regression consumer;
its changed inputs are the new classifiers' signals.

New feature configurations:

- F16: F09 with its four `rg` inputs replaced by `nrg`; distances and volume retained.
- F17: F10 with its four `rg` inputs replaced by `nrg`; volume retained.

New strategy configurations preserve the original targets, strategy, partition,
and balancing, using M008 in every case:

| New TS | Equivalent original TS | Classifier | Strategy |
|---|---|---|---|
| 054 | 038, with M008 | F16-T16-P02-M008-B1 | S03 |
| 055 | 046, with M008 | F16-T09-P02-M008-B1 | S03 |
| 056 | 052 | F16-T24-P02-M008-B1 | S08 |
| 057 | 053 | F17-T25-P02-M008-B1 | S08 |

Use these new classifier identities with fresh training and optimizer state.
The volume semantics change applies to all Features006 volume configurations,
including older IDs. Their canonical descriptions now include the positive-
median rule and cap, so old feature stores and classifier metadata fail hash
validation rather than silently mixing old and new semantics. Existing artifacts
are not automatically deleted or migrated. Rebuild affected feature stores
before training; old volume-dependent classifiers cannot be treated as compatible.
Raw regression and target stores need not be regenerated for this feature change.

Validate:

1. Zero-baseline, zero-volume, ordinary-ratio, exact-cap, and above-cap cases.
2. Feature values at or below 60 throughout the new training matrix.
3. Identical feature semantics in offline and live paths.
4. Epoch loss stability and training/inference normalization agreement.
5. Held-out entry PPV and fee-aware trade-event results, with event coverage
   recorded to prevent a trivial improvement from emitting almost no signals.

The agreed new configurations combine volume stabilization and `nrg`, but retain
the model and BatchNorm layout. A separate volume-only ablation remains useful
to distinguish their effects. These code-level validations do not establish
held-out PPV or loss stability; those require new training and evaluation.

## 2. Historical loss instability and BatchNorm diagnosis

The saved initial-training history contained:

| Measure | Value |
|---|---:|
| Recorded epochs | 87 |
| Best epoch | 72 |
| Best weighted training loss | 1.37735 |
| Median epoch loss | 53.48 |
| Maximum epoch loss | 164,739.83 |
| Epochs with loss above 1,000 | 30 |
| Last epoch loss | 97,284.38 |

M008 already uses Adam at 0.0001, ten times below M002. The remaining spikes are
not ordinary small fluctuations.

For the historical last-epoch checkpoint, on identical sampled inputs:

| Diagnostic mode | Weighted loss |
|---|---:|
| Saved BatchNorm statistics, normal inference | 8,952.31 |
| Training-mode BatchNorm in batches of 64 | 1.10 |
| Population BatchNorm recalibrated on the sample, weights unchanged | 1.27 |
| Relative volume capped at 100, saved statistics and weights unchanged | 1.61 |

Seventeen sampled observations had relative volume above one million and
contributed approximately 99.98% of the weighted inference loss.

This strongly implicates extreme relative volume interacting with BatchNorm
running statistics. Recalibration on the diagnostic sample is not a proposed
production procedure or proof of validation improvement. The cap experiment
demonstrates loss sensitivity; it does not establish entry PPV after retraining.

Relevant implementation: [M008 and checkpoint selection](../Classify/src/Classify.jl)
and [adaptation/full-set loss](../Classify/src/SampleSource.jl).

## 3. Target imbalance

| Target | Training count | Training share | Test share | Inverse-frequency weight |
|---|---:|---:|---:|---:|
| longopen | 564,479 | 0.563% | 0.567% | 35.53 |
| longhold | 7,595,862 | 7.575% | 7.632% | 2.64 |
| shortopen | 578,083 | 0.577% | 0.577% | 34.69 |
| shorthold | 7,879,797 | 7.858% | 7.852% | 2.55 |
| allclose | 83,655,197 | 83.427% | 83.372% | 0.240 |

Evaluation-set proportions are similarly distributed. Both opening classes are
rare, but have substantial absolute support. B1 already uses inverse-frequency
class weights intended to equalize each class's total weight.

With natural-frequency batches of 64, approximately 70% contain no longopen,
69% no shortopen, and 48% neither opening class. Stratified batches are a possible
later experiment. Equal-class sampling must not be combined blindly with the
existing inverse-frequency weights because that would compensate twice.

## 4. Evaluation semantics and PPV

The existing extended confusion matrix groups rows by their winning predicted
class. Its `fn` counts correct winning predictions rejected by the score
threshold, not all true class members assigned to other winning classes.
Consequently its `tpr` and `fpr` are not standard one-versus-rest recall/FPR.

Its PPV remains meaningful for accepted winning predictions. Preserving PPV as
the primary trading metric is intentional and appropriate to the stated strategy.
Correcting recall accounting is an observability improvement, not a proposal to
maximize recall at the expense of PPV.

The threshold rows are cumulative: `5/[0.4-0.5]` means winning score >=0.4,
not just scores inside that interval.

Corrected historical test metrics, using the full confusion matrix denominators:

| Minimum winning score | Longopen PPV | Longopen recall | Shortopen PPV | Shortopen recall |
|---|---:|---:|---:|---:|
| No score rejection | 5.38% | 7.13% | 1.11% | 68.15% |
| 0.4 | 5.28% | 4.63% | 2.60% | 36.82% |
| 0.6 | 4.47% | 1.50% | 6.93% | 5.38% |
| 0.9 | 3.62% | 0.50% | 6.67% | 0.38% |

Longopen confidence does not usefully rank exact-label correctness here.
Shortopen PPV improves with thresholding but remains low. Training PPV is also
poor, approximately 5.58% longopen and 1.12% shortopen.

Exact-minute target PPV and profitable turning-point entry PPV are different
quantities. A prediction may fail exact-minute target matching yet still be a
useful nearby entry. Future event-level evaluation should explicitly define
entry timing tolerance, repeated-signal suppression, and fees rather than
assuming every exact-label false positive is an unprofitable trade.

Relevant implementation: [extended confusion matrix](../Classify/src/Classify.jl)
and [aggregation](../Classify/src/ClassifierStore.jl).

## 5. Reinforcement limits

`reinforce=3` refreshes hard-example flags every three epochs; it does not mean
three epochs of reinforcement. The current implementation gives misclassified
examples 2x relative emphasis, normalized within their class to preserve the
class's intended total weight.

At the historical training confusion matrix, about 92.4% of longopen targets were
misclassified. After normalization, a hard longopen example receives about
1.04x its original weight, while an easy example receives about 0.52x. Thus this
does not substantially increase the weight of most missed longopen examples.

Reinforcement does not fix feature explosions, missing causal inputs, or target
ambiguity. Outcomes of the later reinforced run require a new evaluation.

## 6. Topics deferred until feature stabilization

- Preserve the tested BatchNorm architecture initially; reconsider it only with
  controlled evidence after bounding feature values.
- Evaluate the new endpoint-normalized slopes for cross-coin comparability.
- Expose distance to the 240-minute regression endpoint directly: T24 uses this
  quantity in its entry condition, but F09 does not supply it directly.
- Assess target alignment with the desired turning-point entries. Trend05 emits
  an open at a successful episode's start and hold labels on the subsequent
  path; the distinction includes episode history and future success.
- Consider entry-specific heads or event-based targets only as later experiments,
  not as prerequisites for the volume fix.
- Select checkpoints using held-out open-signal PPV with meaningful coverage and
  fee-aware trading outcomes, rather than training loss alone.
- Replace the current five-monotonically-nondecreasing-loss stopping criterion
  with an appropriate validation/patience criterion. The historical final five
  losses rose from about 2.34 to 97,284; labeling that convergence is misleading.

Address these topics independently so that improvements can be attributed to
specific changes.

## Implementation validation

### Follow-up: aligning Trend05 position with the existing ry feature (2026-10-25)

Trend05 now calculates its signed trend distance as
`(pivot - trend.regry) / pivot`, exactly matching the existing Features006 `ry`
feature for `window=trendregr, offset=0`. The observed pivot is strictly positive;
the distance itself is negative below the regression endpoint and positive above
it. `triggerdist` is unchanged and the comparison still includes equality.
This slightly relaxes the long distance boundary and tightens the short boundary
relative to the former regression-endpoint denominator.

The existing rejection of nonpositive trend regression endpoints is retained
explicitly. Support/lead gradient conditions, hold rules, gain targets, and the
existing feature definitions remain unchanged. No new position feature or
classifier configuration is introduced by this change.

Targets is increased from 0.3.2 to 0.3.3. The Trend05 canonical description now
includes `distance=(pivot-regry)/pivot`, changing its target hash. Existing
Trend05 target stores and dependent classifier metadata are incompatible and
must be explicitly rebuilt/retrained; predictions and evaluations must then be
regenerated. They are not silently reused or automatically deleted. OHLCV,
regression, feature, and partition stores do not need rebuilding for this change.

Focused package-level Trend04 crosschecks, Trend05 tests, and target-store tests
passed 244 assertions, including exact/near distance thresholds, price-scale
invariance, matching the actual `ry` feature, retained endpoint rejection,
invalid pivot assertions, and incremental/full target equivalence.

Features was increased to 2.1.0 and TradingStrategy to 0.16.0. Consumer manifests
were resolved using Pkg. Regression tests were run sequentially at package level:

- Features complete enabled suite: 250 assertions passed, including positive-
  volume baselines, prefix causality, cap boundaries, live append equivalence,
  normalized-gradient scaling/offsets, cache/store equivalence, a crash case
  with negative `regry` and the mean-pivot bound, and explicit failures for
  invalid mean pivots or slopes.
- In-memory `nrg` over the full histories of CTC, LUNA, LUNC, MAVIA, OP, SERAPH,
  and WIF (windows 5 to 720): all values are finite. With full windows, the
  largest value reached 77% of the bound; larger values occur only in row 2,
  which is dropped from the feature store with the other warm-up rows.
- TradingStrategy complete enabled suite: 899 assertions passed, including all
  four new configuration mappings, raw regression gate invariance with `nrg`,
  and artifact pipeline integration.
- Targets focused Trend04 crosschecks and Trend05 suite: 188 assertions passed.

No new production classifier training or held-out PPV measurement was performed.
