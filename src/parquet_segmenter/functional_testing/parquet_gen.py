"""Parquet test file generators.

Provides simple helpers to create pandas DataFrames with different random
distributions and write them to Parquet using pyarrow.
"""
from __future__ import annotations

import glob
import os
import tempfile
import random
from typing import List, Optional, Literal

import numpy as np
import pandas as pd  # type: ignore[import]

from ..log import logger
from .binary_store import PrecalculatedBinaryStore


def random_dataframe(
    n: int = 100,
    int_range: tuple[int, int] = (0, 100),
    float_mean: float = 0.0,
    float_std: float = 1.0,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create a DataFrame with several columns of random data.

    Columns:
      - id: integer range 0..n-1
      - ints: uniform random ints in int_range
      - normals: normal distribution (float_mean, float_std)
      - cats: categorical values drawn from `categories` if provided
    """
    logger.debug(
        "Generating random dataframe n=%d int_range=%s mean=%s std=%s categories=%s",
        n,
        int_range,
        float_mean,
        float_std,
        bool(categories),
    )
    ids = np.arange(n)
    ints = np.random.randint(int_range[0], int_range[1], size=n)
    normals = np.random.normal(loc=float_mean, scale=float_std, size=n)
    data = {"id": ids, "ints": ints, "normals": normals}
    if categories:
        cats = np.random.choice(categories, size=n)
        data["cats"] = pd.Categorical(cats, categories=categories)
    return pd.DataFrame(data)


ParquetEngine = Literal["auto", "pyarrow", "fastparquet"]
ParquetCompression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]


def write_parquet(
    df: pd.DataFrame,
    path: str,
    engine: ParquetEngine = "pyarrow",
    compression: Optional[ParquetCompression] = None,
) -> None:
    """Write DataFrame to a Parquet file using the chosen engine.

    The repository already depends on `pyarrow` so the default engine is fine.
    """
    logger.info("Writing dataframe to parquet %s rows=%d", path, len(df))
    # Ensure parent dir exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Pass compression if provided (None lets pandas/pyarrow choose default)
    df.to_parquet(path, engine=engine, index=False, compression=compression)


def _write_temp_parquet_and_get_size(df: pd.DataFrame) -> int:
    """Write dataframe to a temp parquet and return its file size in bytes.

    This is used to obtain a baseline size estimate before attaching blobs.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name
    try:
        write_parquet(df, tmp_path, compression=None)
        return os.path.getsize(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _candidates_from_store(store: PrecalculatedBinaryStore) -> tuple[list[str], list[int]]:
    """Return (paths, sizes) from a store's public index listing."""
    indexed = store.list_index()
    paths = [p for (_, _, p) in indexed]
    sizes = [s for (_, s, _) in indexed]
    return paths, sizes


def _candidates_from_dir(binaries_dir: str) -> tuple[list[str], list[int]]:
    """Return (paths, sizes) by scanning a directory for files."""
    candidates = sorted(glob.glob(os.path.join(binaries_dir, "*")))
    sizes = [os.path.getsize(p) for p in candidates]
    return candidates, sizes


def _choose_nearest(
    candidates: list[str], sizes: list[int], per_row_target: float, n: int
) -> list[str]:
    """Choose `n` candidate paths by nearest size to per_row_target."""
    chosen: list[str] = []
    for _ in range(n):
        # keep the lambda short by capturing the outer variable
        idx = min(range(len(candidates)), key=lambda j: abs(sizes[j] - per_row_target))
        chosen.append(candidates[idx])
    return chosen


def _choose_index_cycle(candidates: list[str], n: int) -> list[str]:
    """Choose `n` candidate paths by cycling indices (index strategy)."""
    return [candidates[i % len(candidates)] for i in range(n)]


def _load_blob_column(chosen: list[str]) -> list[bytes]:
    """Load chosen binary files into memory with a small cache and return list of bytes."""
    cache: dict[str, bytes] = {}
    blob_column: list[bytes] = []
    for p in chosen:
        if p not in cache:
            with open(p, "rb") as f:
                cache[p] = f.read()
        blob_column.append(cache[p])
    return blob_column


def _ensure_local_store(
    binary_store: Optional[PrecalculatedBinaryStore],
) -> tuple[PrecalculatedBinaryStore, bool]:
    """Return a store instance and a bool indicating if we created it locally."""
    if binary_store is not None:
        return binary_store, False
    tmpd = tempfile.mkdtemp(prefix="bstore_")
    return PrecalculatedBinaryStore(tmpd), True


def _ensure_sizes_in_store(store: PrecalculatedBinaryStore, sizes: list[str]) -> None:
    """Ensure each human-readable size string exists in the store."""
    for s in sizes:
        store.ensure_size(s)


def _assign_edge_blobs_from_store(
    store: PrecalculatedBinaryStore,
    n: int,
    edge_sizes: list[str],
    num_edge_cases: int,
) -> tuple[list[bytes], dict]:
    """Assign explicit edge-case blobs from `edge_sizes` into a blob column.

    Returns (blob_column, edge_map).
    """
    index_list = store.list_index()
    num_edge_cases = min(num_edge_cases, n)
    # pick unique row indices for the edge cases
    picked_rows = sorted(random.sample(range(n), num_edge_cases))

    blob_column: List[bytes] = [b"" for _ in range(n)]
    edge_map: dict[int, int] = {}

    def _closest_path(size_bytes: int) -> tuple[str, int]:
        """Return (path, actual_size_bytes) for exact or closest size in index_list."""
        # build a small map for fast exact lookup
        for (_, s, p) in index_list:
            if s == size_bytes:
                return p, s
        # use a short lambda to keep line length down
        found = min(index_list, key=lambda t: abs(t[0] - size_bytes))
        return found[2], found[0]

    def _read_into_blob(idx: int, path: str, out_column: list[bytes]) -> None:
        with open(path, "rb") as f:
            out_column[idx] = f.read()

    for ridx, size_str in zip(picked_rows, edge_sizes[:num_edge_cases]):
        sb = PrecalculatedBinaryStore.parse_size(size_str)
        p, actual_size = _closest_path(sb)
        _read_into_blob(ridx, p, blob_column)
        edge_map[ridx] = actual_size

    return blob_column, edge_map


def _build_seq_for_target(
    index_list: list[int],
    binary_store: Optional[PrecalculatedBinaryStore],
    target_size_bytes: int,
) -> tuple[list[str], list[int]]:
    """Build the sequence of file paths and indices to reach the target size.

    Returns (seq_paths, seq_indices).
    """
    local_store, _ = _ensure_local_store(binary_store)
    candidates, sizes = _resolve_candidates_from_indices(index_list, local_store)

    total_per_cycle = sum(sizes)
    if total_per_cycle <= 0:
        raise ValueError("sum of candidate sizes must be > 0")

    full_cycles = target_size_bytes // total_per_cycle
    remainder = target_size_bytes - full_cycles * total_per_cycle

    seq: list[str] = []
    seq_indices: list[int] = []
    for _ in range(int(full_cycles)):
        seq.extend(candidates)
        seq_indices.extend(index_list)

    if remainder > 0:
        frac = remainder / total_per_cycle
        k = len(index_list)
        prefix_fraction = frac * k
        prefix_count = max(1, int(prefix_fraction + 0.9999))
        seq.extend(candidates[:prefix_count])
        seq_indices.extend(index_list[:prefix_count])

    return seq, seq_indices


def _resolve_candidates_from_indices(
    index_list: list[int],
    local_store: PrecalculatedBinaryStore,
) -> tuple[list[str], list[int]]:
    """Resolve index_list into candidate paths and sizes using given store."""
    candidates: list[str] = []
    sizes: list[int] = []
    for idx in index_list:
        p = local_store.get(idx)
        candidates.append(p)
        sizes.append(os.path.getsize(p))
    return candidates, sizes


def _sample_n_rows(
    n_rows_mean: int, n_rows_std: int, categories: Optional[list[str]]
) -> tuple[int, pd.DataFrame]:
    """Sample the number of rows and return (n, dataframe).

    Encapsulates the random normal sampling and DataFrame creation.
    """
    n = max(1, int(round(np.random.normal(loc=n_rows_mean, scale=n_rows_std))))
    df = random_dataframe(n=n, categories=categories)
    return n, df


def _fill_small_blobs_in_column(store: PrecalculatedBinaryStore, blob_column: list[bytes]) -> None:
    """Fill empty entries in blob_column with small 1 kB blobs from store."""
    for i, v in enumerate(blob_column):
        if v == b"":
            small_path = store.ensure_size("1 kB")
            with open(small_path, "rb") as f:
                blob_column[i] = f.read()


def _write_parquet_and_meta(
    path: str,
    df: pd.DataFrame,
    compression: Optional[ParquetCompression],
) -> int:
    """Write DataFrame to parquet and return the file size in bytes."""
    write_parquet(df, path, compression=compression)
    return os.path.getsize(path)


def _get_baseline_size(df: pd.DataFrame) -> int:
    """Write df to a temp parquet and return its size; helper to reduce locals."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name
    try:
        write_parquet(df, tmp_path, compression=None)
        return os.path.getsize(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _attach_blob_column(
    df: pd.DataFrame,
    source: tuple[Optional[PrecalculatedBinaryStore], Optional[str], Literal["nearest", "index"]],
    per_row_target: float,
    n: int,
) -> None:
    """Attach a 'blob' column to `df` choosing files from store or dir."""
    use_store, binaries_dir, blob_strategy = source
    if use_store:
        candidates, sizes = _candidates_from_store(use_store)
    elif binaries_dir is not None:
        candidates, sizes = _candidates_from_dir(binaries_dir)
    else:
        candidates, sizes = [], []

    if not candidates:
        logger.warning("No candidates available; skipping blobs")
        return

    if blob_strategy == "nearest":
        chosen = _choose_nearest(candidates, sizes, per_row_target, n)
    else:
        chosen = _choose_index_cycle(candidates, n)

    df["blob"] = _load_blob_column(chosen)


# pylint: disable=R0913
def generate_and_write_parquet(
    path: str,
    n: int = 100,
    categories: Optional[list[str]] = None,
    compression: Optional[ParquetCompression] = None,
    *,
    binaries_dir: Optional[str] = None,
    binary_store: Optional[PrecalculatedBinaryStore] = None,
    blob_strategy: Literal["nearest", "index"] = "nearest",
    target_file_size_bytes: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience to create a random dataframe and write to path.

    Returns the DataFrame for further assertions in tests.
    """
    df = random_dataframe(n=n, categories=categories)

    # Determine source of precalculated binaries: a store instance takes
    # precedence, otherwise a legacy `binaries_dir` may be used.
    use_store = binary_store if binary_store is not None else None

    # If user provided binaries (either via store or dir) and a target
    # file size, attach a BLOB column to each row by picking binary files
    # whose sizes combine with the base dataframe to approximate the target.
    if (use_store or binaries_dir) and target_file_size_bytes is not None:
        logger.info(
            "Attaching binaries from %s to reach target size %d bytes",
            binaries_dir,
            target_file_size_bytes,
        )

        # Write a temporary parquet of the dataframe without blobs to get a
        # baseline size estimate using a small helper to reduce locals.
        base_size = _get_baseline_size(df)

        # Now that we have a baseline, compute remaining bytes to reach the
        # target and select candidate binaries.
        remaining = target_file_size_bytes - base_size
        if remaining <= 0:
            logger.warning(
                "Target size %d <= baseline parquet size %d; not adding blobs",
                target_file_size_bytes,
                base_size,
            )
        else:
            per_row_target = remaining / max(1, n)
            source = (use_store, binaries_dir, blob_strategy)
            _attach_blob_column(df, source, per_row_target, n)

    write_parquet(df, path, compression=compression)
    logger.debug("Wrote parquet to %s", path)
    return df


# pylint: disable=R0913
def generate_parquet_with_edge_cases(
    path: str,
    *,
    n_rows_mean: int = 1000,
    n_rows_std: int = 200,
    num_edge_cases: int = 3,
    edge_sizes: Optional[list[str]] = None,
    binary_store: Optional[PrecalculatedBinaryStore] = None,
    compression: Optional[ParquetCompression] = None,
    categories: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate a parquet file with random number of rows and explicit edge cases.

    Returns (df, edge_map) where edge_map maps row_index -> blob_size_bytes for
    the rows that were designated as edge cases (close to or over 1 MiB).

    Behavior:
      - Number of rows sampled from a normal distribution (mean/std) and
        clipped to at least 1.
      - `num_edge_cases` unique rows are chosen randomly and assigned blobs
        of sizes given by `edge_sizes` (strings parsed by the store). If
        `edge_sizes` is None, a sensible default set around 0.9-2.0 MiB is
        used.
      - A `binary_store` may be provided; otherwise a local temporary store
        is created.
    """
    # sample number of rows and create DataFrame
    n, df = _sample_n_rows(n_rows_mean, n_rows_std, categories)

    # defaults for edge sizes if not provided
    if edge_sizes is None:
        edge_sizes = ["900 kB", "1024 kB", "1300 kB"]

    # prepare or create a local store
    local_store, _ = _ensure_local_store(binary_store)
    # ensure requested sizes exist
    _ensure_sizes_in_store(local_store, edge_sizes)
    # assign explicit edge case blobs from the store
    blob_column, edge_map = _assign_edge_blobs_from_store(
        local_store, n, edge_sizes, num_edge_cases
    )

    # fill remaining rows with small blobs
    _fill_small_blobs_in_column(local_store, blob_column)

    df["blob"] = blob_column

    # write and return mapping
    file_size = _write_parquet_and_meta(path, df, compression)
    logger.info(
        "Wrote edge-case parquet %s rows=%d size=%d bytes",
        path,
        n,
        file_size,
    )

    return df, {"edge_map": edge_map, "file_size": file_size}


# pylint: disable=R0913
def generate_parquet_by_size(
    path: str,
    target_size_bytes: int,
    index_list: list[int],
    *,
    binary_store: Optional[PrecalculatedBinaryStore] = None,
    compression: Optional[ParquetCompression] = None,
    categories: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate a parquet file by concatenating blobs chosen by store indices.

    - `index_list` is a list of global indices into `binary_store` (ascending
      sizes). The sequence defined by `index_list` is repeated enough times to
      reach or exceed `target_size_bytes`. If the remainder is fractional,
      a prefix of the index_list (ceil fraction*K) is appended.
    - Returns (df, meta) where meta includes produced size and the sequence
      of indices used.
    """
    if target_size_bytes <= 0:
        raise ValueError("target_size_bytes must be > 0")
    if not index_list:
        raise ValueError("index_list must be non-empty")

    seq, seq_indices = _build_seq_for_target(index_list, binary_store, target_size_bytes)
    n_rows = len(seq)
    df = random_dataframe(n=n_rows, categories=categories)
    df["blob"] = _load_blob_column(seq)

    produced = _write_parquet_and_meta(path, df, compression)
    logger.info(
        "Wrote parquet_by_size %s target=%d produced=%d rows=%d",
        path,
        target_size_bytes,
        produced,
        n_rows,
    )

    return df, {"target": target_size_bytes, "produced": produced, "seq_indices": seq_indices}
