# CLI Usage for parquet_segmenter

This repository includes a tiny CLI to help generate Parquet test files using
precalculated binary blobs. The CLI is primarily for local development and
quick experiments.

Location: `src/parquet_segmenter/cli.py`

Examples
--------

1) List binaries in the default store (creates the store directory if needed):

```bash
python -m parquet_segmenter.cli list
```

2) Create a precalculated binary and list it:

```bash
# ensure store has a 1 MB blob
python -c "from parquet_segmenter.functional_testing.binary_store import PrecalculatedBinaryStore; PrecalculatedBinaryStore().ensure_size('1 MB')"
python -m parquet_segmenter.cli list
```

3) Generate a parquet with edge-case large blobs:

```bash
python -m parquet_segmenter.cli edge --out /tmp/edge.parquet --rows-mean 500 --rows-std 50 --num-edge-cases 2 --edge-sizes '900 kB,1 MB'
```

4) Generate a parquet by target total size using store indices:

```bash
# pick indices from `list` output and then
python -m parquet_segmenter.cli by-size --out /tmp/target.parquet --target 10485760 --indices 0,1,2
```

5) Quick random parquet writer (optionally attach blobs from store or dir):

```bash
python -m parquet_segmenter.cli simple --out /tmp/simple.parquet --rows 200 --store-dir ~/.cache/precalc_binaries --target 1048576 --blob-strategy index
```

Notes
-----
- Sizes in the store use powers of 1024 (1 kB = 1024 bytes).
- The CLI is intentionally small and synchronous; for heavy workloads use
  the underlying functions directly from Python.
