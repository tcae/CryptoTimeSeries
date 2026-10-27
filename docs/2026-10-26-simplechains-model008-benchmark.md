# SimpleChains alternative to model008

Measured October 26, 2026 on an Apple M1, Julia 1.12.7, SimpleChains 0.5.0,
four Julia threads and one BLAS thread.

**October 27 follow-up:** native BatchNorm is now implemented as M011. It
retains all four BatchNorm layers and recovers comparable quality while training
31.32x faster than 059 in the architecture-matched comparison below. The
original dense-only results are retained as a separate experiment.

## Implementation

- **M008** remains unchanged: configuration **059**, `F19-T24-P02-M008-B1`.
- **M009** is the SimpleChains dense-only alternative.
- **M010** is the matched dense-only Flux control.

Factories, model wrapper and training dispatch are in
[SimpleChainsNN.jl](../Classify/src/SimpleChainsNN.jl). All three models use the
existing `NN`, sample-source, prediction, reinforcement and BSON checkpoint
interfaces. M009 uses SimpleChains for forward/backward dense evaluation and
the same persistent Flux Adam optimizer as the other models, including the
learning rate 0.0001 and beta values (0.9, 0.999).

SimpleChains has no built-in BatchNorm. As agreed, M009 and M010 remove all four
BatchNorm layers rather than attempting to approximate them. For F19/T24 the
dense architecture is **7 -> 21 -> 14 -> 10 -> 5**, with ReLU in the three
hidden layers and logits at the output. There are 681 dense parameters; M008
also has 104 trainable BatchNorm parameters.

M009 and M010 start from identical seeded dense weights and biases. A custom
SimpleChains terminal loss preserves the existing weighted mean cross-entropy
contract, including per-batch normalization by the sum of class/sample weights.
The same optimizer implementation deliberately isolates forward/backward backend
performance rather than also changing optimizers.

Registry entries `F19-T24-P02-M009-B1` and `F19-T24-P02-M010-B1` select the new
factories without changing any existing strategy or saved classifier identity.
For example:

```julia
using TradingStrategy
config = TradingStrategy.classifierconfig("F19-T24-P02-M009-B1")
classifier = TradingStrategy.trainclassifier!(config; coinset="CS00", daterange="D00")
```

This uses the normal artifact pipeline. For an isolated benchmark without
modifying shared classifier artifacts, use the script below instead.

## Reproduction and timing boundaries

Run from the repository root:

```sh
julia --project=. scripts/benchmark_model008.jl
```

The script uses `EnvConfig.test`, CS00's **SINE and DOUBLESINE**, the exact D00
date range **2025-01-17T20:56 through 2025-08-10T15:00**, and 059's F19 features,
T24 targets, P02 partitions and B1 class balancing. It generates the artificial
OHLCV and builds the real feature/target/partition stores and training index.
There is no network market-data download.

- Training: **323,840 combined samples**, batch size **64**, **10 epochs**,
  **3 trials**, seed **713**.
- Label-order training counts: **20,557; 88,492; 20,492; 88,235; 106,064**.
- Each coin has **41,580 test** and **39,501 eval** samples.
- Every trial restarts model and optimizer state and reuses the same seeded
  shuffled sample order. Trials measure repeat runtime, not different seeds.
- Training timings include shuffling, batch materialization, weighted loss,
  differentiation and Adam updates. They exclude data preparation, compilation,
  per-epoch full-set evaluation and checkpoint I/O. Training uses an in-memory
  matrix; these are **not end-to-end artifact-pipeline or mmap-I/O timings**.
- Compilation is warmed on disposable model state. Inference is measured in
  inference mode, including owned logits and softmax probability output.
  Each inference measurement is the median of seven groups of repeated calls.
- Training results below are medians of three ten-epoch totals. Inference
  results pool both patterns and three trials, with per-pattern raw rows retained.

The completed persistent run is
`~/crypto/debug/model008-simplechains/run-4BuDzo/`. It contains
`metadata.json`, per-epoch `training.csv` (including allocated bytes and GC time),
`inference.csv`, `quality.csv` and isolated per-coin stores. New runs create new
persistent folders under the same debug parent. No production classifier or
shared coin store is overwritten.

## Training results

| Model | Median, 10 epochs | Trial range | Speedup versus M008 |
|---|---:|---:|---:|
| M008: Flux + BatchNorm (059) | 46.422 s | 46.322-46.713 s | 1.00x |
| M009: SimpleChains dense-only | 0.766 s | 0.760-0.766 s | 60.61x |
| M010: Flux dense-only | 6.734 s | 6.699-6.812 s | 6.89x |

**The architecture-matched backend training speedup is 8.79x**, comparing M009
against M010. The 60.61x comparison additionally removes BatchNorm and must not
be attributed entirely to SimpleChains.

## Inference results

Times are per call, not per observation.

| Observations per call | M008 Flux + BN | M009 SimpleChains | M010 Flux dense | M009 speedup vs M008 | M009 speedup vs M010 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.527 us | 0.368 us | 0.934 us | 4.15x | 2.54x |
| 64 | 10.735 us | 4.061 us | 7.866 us | 2.64x | 1.94x |
| 8,192 | 1.267 ms | 0.470 ms | 0.999 ms | 2.69x | 2.12x |

Single-observation median by pattern:

| Pattern | M008 | M009 | M010 |
|---|---:|---:|---:|
| SINE | 1.528 us | 0.379 us | 0.947 us |
| DOUBLESINE | 1.526 us | 0.361 us | 0.932 us |

## Predictive-quality check

Held-out **test** results after the same ten epochs:

| Model | Pattern | Accuracy | Cross entropy | Long-open precision / recall | Short-open precision / recall |
|---|---|---:|---:|---:|---:|
| M008 | SINE | 72.50% | 0.4174 | 34.62% / 100.00% | 36.00% / 100.00% |
| M009 | SINE | 45.00% | 1.0168 | 28.57% / 66.66% | 16.00% / 44.44% |
| M010 | SINE | 44.16% | 1.0061 | 28.57% / 66.66% | 16.66% / 44.44% |
| M008 | DOUBLESINE | 73.48% | 0.5580 | 23.16% / 56.39% | 24.16% / 33.38% |
| M009 | DOUBLESINE | 54.12% | 1.1180 | 17.49% / 53.35% | 14.31% / 13.82% |
| M010 | DOUBLESINE | 54.36% | 1.1181 | 17.53% / 53.35% | 11.25% / 9.10% |

Eval results show the same pattern and are retained in `quality.csv`, together
with entry prediction counts and supports. Each backend's quality metrics were
identical across the three repeated trials. The two dense backends are close
but not identical after 50,600 updates; their results must not be described as
bitwise equivalent. Unit tests compare initial logits, per-parameter gradients,
weighted/unweighted losses and short update sequences with explicit tolerances.

## Conclusion

This is a genuinely small network, and SimpleChains provides a substantial
training speedup and a smaller but useful inference speedup on this CPU.
However, the measured dense-only alternative is **not a quality-equivalent
replacement for 059**. Removing BatchNorm materially worsens both synthetic
patterns under the unchanged ten-epoch budget. Keep 059 unchanged; M009 is an
experimental alternative, not a justified production-model replacement.

The benchmark does not establish time to equal quality, multi-seed robustness,
real-market quality or end-to-end training time including persistence.

## Validation

Run package regressions separately, never as a workspace-wide test:

```sh
cd Classify
julia --project=. test/runtests.jl
cd ../TradingStrategy
julia --project=. test/runtests.jl
cd ..
julia --project=. test/benchmark_model008_test.jl
```

The SimpleChains tests cover probability shape/normalization, owned outputs,
weighted/unweighted gradient and optimizer parity, extreme logits, validation,
reinforcement and checkpoint/optimizer continuation. Registry tests preserve
059 and exercise both new model IDs.

## October 27: native SimpleChains BatchNorm, M011 versus 059

### Implementation and usage

[SimpleChainsBatchNorm.jl](../Classify/src/SimpleChainsBatchNorm.jl) implements
BatchNorm directly in SimpleChains' forward/backward scratch buffers. It does
not invoke Flux BatchNorm or Flux automatic differentiation during training.
Its affine parameters share the flat parameter vector with the dense weights.
Running mean/variance and training mode are nontrainable layer state, retained
in the existing BSON checkpoints.

M011 in [SimpleChainsNN.jl](../Classify/src/SimpleChainsNN.jl) preserves M008's
exact architecture:

```text
BatchNorm(7)
Dense(7 -> 21, ReLU)
BatchNorm(21)
Dense(21 -> 14, ReLU)
BatchNorm(14)
Dense(14 -> 10, ReLU)
BatchNorm(10)
Dense(10 -> 5)
```

Both have **785 trainable parameters**, identical seeded initialization,
weighted mean logit cross entropy, and the same persistent Adam implementation
and settings. Native BatchNorm uses population variance for normalization,
epsilon **1e-5**, momentum **0.1**, and Bessel-corrected variance for the moving
statistics, matching Flux. Inference uses frozen moving statistics.

The implementation targets Float32 channel-by-observation matrices and the
affine, tracked-statistics BatchNorm used by M008. It is not a general
replacement for every Flux normalization option. Training requires at least
two observations in **every batch**, including a trailing partial batch.
Singleton training fails explicitly instead of producing an invalid unbiased
moving variance. Single-observation inference is supported. It uses
SimpleChains 0.5 internals; upgrades require rerunning the parity tests.

Configuration **059 is unchanged**. To select the native implementation:

```julia
using TradingStrategy
config = TradingStrategy.classifierconfig("F19-T24-P02-M011-B1")
classifier = TradingStrategy.trainclassifier!(config; coinset="CS00", daterange="D00")
```

Reproduce the isolated benchmark from the repository root:

```sh
julia --project=. scripts/benchmark_model008.jl --batchnorm
```

Raw results are retained in
`~/crypto/debug/model008-simplechains/run-PtJFAn/`: `metadata.json`,
`training.csv`, `inference.csv`, `quality.csv`, and isolated synthetic coin
stores. The benchmark now also records per-epoch full-set weighted training
loss, evaluated **outside** the timed region.

### Matched conditions

Apple M1, Julia **1.12.7**, Flux **0.16.10**, SimpleChains **0.5.0**; four Julia
threads, one BLAS thread. As in the original experiment: F19/T24/P02/B1,
CS00/D00, **323,840 combined training observations**, **10 epochs**, **64 samples
per batch**, **3 repeated trials**, seed **713**, and **41,580 test / 39,501 eval
observations per pattern**. Model/optimizer state and shuffled sample order
restart identically for each trial. Training sample count is divisible by 64.

Timing boundaries are unchanged: shuffled in-memory batch extraction, native
or Flux forward/backward, weighted loss and Adam are included. Compilation,
data preparation, full-set loss/evaluation and checkpoint I/O are excluded.
These are not end-to-end artifact-pipeline times.

### Runtime and allocation results

| Backend | Median training, 10 epochs | Trial range | Speedup |
|---|---:|---:|---:|
| Flux M008 + BatchNorm, 059 | 47.711 s | 47.272-47.720 s | 1.00x |
| SimpleChains M011 + native BatchNorm | **1.523 s** | 1.447-1.584 s | **31.32x** |

Training allocated **19.196 GB** for Flux versus **0.202 GB** for M011 per
ten-epoch trial, about **95.2x less**. These are cumulative allocations, not
peak resident memory.

Inference times include owned output and softmax, and pool both patterns and
three trials:

| Observations per call | Flux M008 | Native M011 | Speedup |
|---:|---:|---:|---:|
| 1 | 1.554 us | **0.656 us** | **2.37x** |
| 64 | 10.672 us | **5.644 us** | **1.89x** |
| 8,192 | 1.258 ms | **1.013 ms** | **1.24x** |

Per-pattern inference medians:

| Pattern | Batch | Flux M008 | Native M011 |
|---|---:|---:|---:|
| SINE | 1 | 1.556 us | 0.654 us |
| DOUBLESINE | 1 | 1.552 us | 0.658 us |
| SINE | 64 | 10.585 us | 5.736 us |
| DOUBLESINE | 64 | 10.758 us | 5.555 us |
| SINE | 8,192 | 1.237 ms | 1.023 ms |
| DOUBLESINE | 8,192 | 1.279 ms | 0.993 ms |

### Predictive quality

Held-out test results after ten epochs:

| Model | Pattern | Accuracy | Cross entropy | Long-open precision / recall | Short-open precision / recall |
|---|---|---:|---:|---:|---:|
| Flux M008 | SINE | 72.50% | 0.41742 | 34.62% / 100.00% | 36.00% / 100.00% |
| Native M011 | SINE | 72.50% | 0.41474 | 34.62% / 100.00% | 36.00% / 100.00% |
| Flux M008 | DOUBLESINE | 73.48% | 0.55804 | 23.16% / 56.39% | 24.16% / 33.38% |
| Native M011 | DOUBLESINE | 73.53% | 0.55878 | 23.19% / 57.18% | 24.62% / 33.43% |

Eval results:

| Model | Pattern | Accuracy | Cross entropy | Long-open precision / recall | Short-open precision / recall |
|---|---|---:|---:|---:|---:|
| Flux M008 | SINE | 72.51% | 0.41730 | 34.61% / 100.00% | 36.00% / 100.00% |
| Native M011 | SINE | 72.51% | 0.41463 | 34.61% / 100.00% | 36.00% / 100.00% |
| Flux M008 | DOUBLESINE | 73.47% | 0.55753 | 23.09% / 56.51% | 24.22% / 33.20% |
| Native M011 | DOUBLESINE | 73.52% | 0.55827 | 23.12% / 57.29% | 24.76% / 33.40% |

The weighted full-training-set loss after epoch 10 was **0.52156** for Flux
and **0.52110** for native SimpleChains. Loss trajectories remain close:

| Epoch | Flux M008 | Native M011 |
|---:|---:|---:|
| 1 | 0.68437 | 0.68437 |
| 2 | 0.59946 | 0.59946 |
| 5 | 0.55217 | 0.55193 |
| 10 | 0.52156 | 0.52110 |

Each backend reproduced its own quality metrics exactly in all three trials.
Across backends, floating-point reductions and dense kernels are not bitwise
identical, so the 50,600-update trajectories need not remain identical.
The results establish comparable quality for this seed, data and epoch budget,
not superiority, multi-seed robustness or real-market equivalence.

### Validation and conclusion

Native-layer tests compare training/inference forward outputs, affine and
input gradients, moving statistics (including the variance correction), constant
channels, batch sizes 2 and 17, singleton inference, parameter initialization,
dimension/configuration errors, weighted/unweighted whole-model gradients,
short optimizer sequences and checkpoint/statistics continuation against Flux.
Full artifact-pipeline tests train, reload, predict and evaluate M011 using the
normal compact memory-mapped sample source.

The package regressions passed separately: **298 Classify assertions**,
**989 TradingStrategy assertions**, plus **14 benchmark helper assertions**.

**Conclusion:** keeping BatchNorm removes the large quality deficit observed
with the dense-only alternatives. Native M011 offers a substantial measured
training speedup over 059 while retaining comparable synthetic-pattern
quality. Inference also improves, though much less dramatically than training.
No existing strategy identity or production model was switched automatically.
