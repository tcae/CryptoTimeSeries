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

This investigation was read-only apart from this document. No training process,
source implementation, or saved classifier was modified.

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
  `correctunworkablevolume`, `relativevolume`, and `supplement!`.

### Recommended first experiment

Keep the current model, BatchNorm layout, target definition, partitioning, and
class-balancing policy fixed. Introduce a new feature configuration that
saturates relative volume at 100, preserving all values below the cap.

This retains the intended rough volume-peak signal and changes only 0.04338% of
training rows. A cap of 100 is a pragmatic experimental choice, not an optimized
or economically universal threshold. Removing relative volume entirely is a
useful later ablation if capping does not produce useful improvement.

The feature must be bounded before it reaches BatchNorm. Apply the same feature
definition to training, stored predictions, simulation, and live inference.
Use a new feature/classifier identity rather than silently changing F09 and
reusing its caches. Train fresh weights and optimizer state so the result is not
confounded by adaptation to the original unbounded feature.

A bounded calculation should explicitly define zero-baseline behavior: positive
short-window volume with zero long-window median should saturate, while zero
volume in both windows should represent no activity. It should not depend on
machine epsilon as an economically meaningful volume baseline.

Validate:

1. Zero-baseline, zero-volume, ordinary-ratio, exact-cap, and above-cap cases.
2. Feature values at or below 100 throughout the new training matrix.
3. Identical feature semantics in offline and live paths.
4. Epoch loss stability and training/inference normalization agreement.
5. Held-out entry PPV and fee-aware trade-event results, with event coverage
   recorded to prevent a trivial improvement from emitting almost no signals.

Do not change BatchNorm, gradient scaling, targets, reinforcement, and weighting
simultaneously with this first experiment. Relative-gradient normalization is a
separate candidate for a subsequent controlled experiment.

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
- Consider relative or volatility-normalized slopes for cross-coin comparability.
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
