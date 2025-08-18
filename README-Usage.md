Examples: build_blob_dataframe clustering
-------------------------------------

Minimal examples showing how to use `build_blob_dataframe` with the new clustering options:

```python
from parquet_segmenter.testing.blob_df import build_blob_dataframe, ChunkSize

# Spread top_outliers across batches (default)
df = build_blob_dataframe(total_df_size=10 * ChunkSize.ONE_MB, top_outliers=3, seed=42)

# Cluster into a single batch (contiguous)
df_cluster = build_blob_dataframe(
    total_df_size=10 * ChunkSize.ONE_MB,
    top_outliers=3,
    spread_top_outliers=False,
    cluster_batches=1,
    contiguous_within_batch=True,
    seed=42,
)

# Two clusters across batches
df_two = build_blob_dataframe(
    total_df_size=20 * ChunkSize.ONE_MB,
    top_outliers=4,
    spread_top_outliers=False,
    cluster_batches=2,
    seed=42,
)
```

These examples are deterministic when `seed` is provided.
README — Usage: index generators

This document describes the public index-generation helpers in
`parquet_segmenter.index_generators` and how to use them from code and tests.

Files / symbols
- `parquet_segmenter.index_generators.stdlib_choice_factory`
- `parquet_segmenter.index_generators.numpy_choice_factory`
- `parquet_segmenter.index_generators.generate_random_indices`
- `parquet_segmenter.index_generators.histogram_from_generator`

Goals
- Produce deterministic or random streams of integer indices in the range
  `[0, num_bins)`.
- Support both sampling with replacement (infinite stream) and without
  replacement (single permutation or weighted sample without replacement).
- Accept optional weight vectors (`probs`) which are validated and normalized
  before use.

Key behaviors (summary)
- `probs` (if provided) is validated and normalized to sum to 1.0. Validation
  checks performed:
  - length must equal `num_bins`
  - weights must be non-negative
  - weights must not all be zero
  If validation fails a `ValueError` is raised.

- `replace=True` produces an (effectively) infinite stream. When `probs` is
  provided samples are drawn according to the probability vector; when
  `probs` is `None` sampling is uniform.

- `replace=False` produces a single sample of `num_bins` unique indices:
  - If `probs` is provided, weighted sampling without replacement is used
    (NumPy's `choice(..., replace=False, p=...)`).
  - If `probs` is `None`, an unweighted permutation of `0..num_bins-1` is
    returned.

- `seed` (when present) is used to create the RNG so results are deterministic
  for the same inputs.

Examples

1) stdlib factory — sampling with replacement (uniform)

```python
from parquet_segmenter.index_generators import stdlib_choice_factory

factory = stdlib_choice_factory(num_bins=3, replace=True, seed=42)
it = factory()
# consume 5 values
vals = [next(it) for _ in range(5)]
print(vals)  # e.g. [0, 2, 1, 0, 2]
```

2) stdlib factory — sampling with replacement using weights

```python
# weights will be normalized automatically
factory = stdlib_choice_factory(num_bins=3, probs=[0.8, 0.1, 0.1], replace=True, seed=1)
it = factory()
# most values will be 0
print([next(it) for _ in range(10)])
```

3) stdlib factory — without replacement (unweighted permutation)

```python
factory = stdlib_choice_factory(num_bins=5, replace=False, seed=7)
print(list(factory()))  # a permutation of 0..4
```

4) numpy factory — weighted without replacement

```python
from parquet_segmenter.index_generators import numpy_choice_factory

# numpy is required; the factory raises RuntimeError if numpy is not present
factory = numpy_choice_factory(num_bins=4, probs=[0.7, 0.1, 0.1, 0.1], replace=False, seed=2)
print(list(factory()))  # length == 4, weighted permutation
```

5) Interfacing with `generate_random_indices`

The `generate_random_indices` helper supports two modes:
- Old / legacy: call with an integer `count` and optional `seed` to get a
  deterministic permutation: `generate_random_indices(count, seed=...)`.
- New: pass either a factory (callable returning an iterator) or an iterator
  itself plus `k` to collect `k` samples.

```python
from parquet_segmenter.index_generators import generate_random_indices, stdlib_choice_factory

# legacy permutation mode
perm = generate_random_indices(5, seed=123)  # returns 5-element permutation

# factory mode (collect k samples)
factory = stdlib_choice_factory(3, replace=True, seed=9)
samples = generate_random_indices(factory, k=10)  # collects 10 samples from the factory
```

6) Building a histogram of samples

```python
from parquet_segmenter.index_generators import histogram_from_generator, stdlib_choice_factory

factory = stdlib_choice_factory(3, replace=True, seed=42)
# count how many samples fall into each of the 3 bins after 1000 draws
hist = histogram_from_generator(factory, n=1000, num_bins=3)
print(hist)  # list of 3 integers summing to 1000
```

Errors and edge cases
- If `num_bins <= 0` a `ValueError` is raised.
- If `probs` length doesn't match `num_bins`, contains negative values, or
  sums to zero a `ValueError` is raised.
- `numpy_choice_factory` raises `RuntimeError` if NumPy is not available.
- `generate_random_indices` in factory mode raises `TypeError` if `k` is
  omitted, and `ValueError` if the iterator is exhausted before producing
  `k` values.

Testing
- The repository contains unit tests under `__tests__/unit`. Run them with:

```bash
.venv/bin/python -m pytest __tests__/unit -q
```

Notes
- `probs` are normalized automatically. If you need to preserve raw weights,
  normalize them yourself before passing them to the factories.
- The RNG `seed` is used to create deterministic behavior for tests; however
  different backends (stdlib vs NumPy) will not necessarily produce identical
  sequences for the same seed.

If you'd like, I can also add a short example script under `examples/` that
shows these factories in action and prints histograms for a few sample
configurations.
