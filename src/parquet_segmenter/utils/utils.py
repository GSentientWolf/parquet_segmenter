from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Literal

import pandas as pd
from pathlib import Path


def faked_memory_usage(df: pd.DataFrame, index_sizes_col: str = "index_size_bytes") -> int:
    """Compute a fake total memory usage for a DataFrame by summing column
    memory and a provided per-row index-size column.

    Args:
        df: DataFrame containing the data.
        index_sizes_col: column name containing per-row index sizes in bytes.

    Returns:
        Total bytes (int).
    """
    col_bytes = int(df.memory_usage(index=False, deep=True).sum())
    if index_sizes_col in df.columns:
        idx_bytes = int(df[index_sizes_col].sum())
    else:
        idx_bytes = 0
    return col_bytes + idx_bytes


@contextmanager
def patch_memory_usage(index_sizes_col: str = "index_size_bytes") -> Iterator[None]:
    """Context manager that temporarily patches pandas.DataFrame.memory_usage
    to include a fake 'Index' value derived from a per-row column.

    Usage:
        with patch_memory_usage("sizes"):
            # inside this block DataFrame.memory_usage() will include Index
            # value equal to the sum of df["sizes"] for that df.

    Note: This patches the method globally; use only in tests or single-threaded
    contexts and always restore via the context manager.
    """
    orig = pd.DataFrame.memory_usage

    def _patched(self, index: bool = True, deep: bool = False):
        s = orig(self, index=index, deep=deep)
        if index:
            try:
                fake_index = int(self[index_sizes_col].sum()) if index_sizes_col in self.columns else 0
            except Exception:
                fake_index = 0
            try:
                s.loc['Index'] = fake_index
            except Exception:
                s['Index'] = fake_index
        return s

    pd.DataFrame.memory_usage = _patched
    try:
        yield
    finally:
        pd.DataFrame.memory_usage = orig


def save_dataframe_best(
    df: pd.DataFrame,
    path: str,
    prefer_parquet: bool = True,
    compression: Optional[Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']] = None,
) -> str:
    """Save a DataFrame using the best available format.

    Tries Parquet/Feather/Pickle/CSV in that order (or reversed if prefer_parquet
    is False). If the provided `path` has an extension that pandas understands
    the function will honor that.

    Returns the path that was written.
    """

    p = Path(path)
    base = str(p.with_suffix(''))
    ext = p.suffix.lower()

    def _try_parquet(path: str) -> bool:
        try:
            if compression is None:
                df.to_parquet(path)
            else:
                df.to_parquet(path, compression=compression)
            return True
        except Exception:
            return False

    # honor explicit extension
    if ext in ('.parquet', '.pq'):
        if _try_parquet(path):
            return path
    if ext == '.feather':
        try:
            df.to_feather(path)
            return path
        except Exception:
            pass
    if ext in ('.pkl', '.pickle'):
        df.to_pickle(path)
        return path
    if ext == '.csv':
        df.to_csv(path, index=False)
        return path

    # choose order based on preference
    if prefer_parquet:
        try_order = ['.parquet', '.feather', '.pkl', '.csv']
    else:
        try_order = ['.feather', '.parquet', '.pkl', '.csv']

    for e in try_order:
        pth = base + e
        if e == '.parquet' and _try_parquet(pth):
            return pth
        if e == '.feather':
            try:
                df.to_feather(pth)
                return pth
            except Exception:
                continue
        if e in ('.pkl', '.pickle'):
            try:
                df.to_pickle(pth)
                return pth
            except Exception:
                continue
        if e == '.csv':
            try:
                df.to_csv(pth, index=False)
                return pth
            except Exception:
                continue

    raise RuntimeError("Failed to save DataFrame to any supported format")


def read_dataframe_best(path: str) -> pd.DataFrame:
    """Read a DataFrame written by :func:`save_dataframe_best`.

    If `path` has a known extension this will use the appropriate pandas
    reader. If the path doesn't exist or has no extension the function will
    try common extensions in the order: parquet, feather, pickle, csv.

    Raises:
        FileNotFoundError: if no matching file is found.
        RuntimeError: if a file is found but none of the readers succeed.
    """
    from pathlib import Path

    p = Path(path)

    def _try_parquet(pth: Path):
        try:
            return pd.read_parquet(pth)
        except Exception:
            return None

    # if the exact path exists, prefer direct read based on suffix
    if p.exists():
        ext = p.suffix.lower()
        if ext in ('.parquet', '.pq'):
            df = _try_parquet(p)
            if df is None:
                raise RuntimeError(f"Failed to read parquet file: {p}")
            return df
        if ext == '.feather':
            return pd.read_feather(p)
        if ext in ('.pkl', '.pickle'):
            return pd.read_pickle(p)
        if ext == '.csv':
            return pd.read_csv(p)

    # if the path itself doesn't exist or had no known suffix, try common
    # extensions by probing for files next to the given path.
    try_order = ['.parquet', '.feather', '.pkl', '.csv']
    for e in try_order:
        candidate = p.with_suffix(e)
        if not candidate.exists():
            continue
        if e == '.parquet':
            df = _try_parquet(candidate)
            if df is not None:
                return df
            continue
        if e == '.feather':
            try:
                return pd.read_feather(candidate)
            except Exception:
                continue
        if e in ('.pkl', '.pickle'):
            try:
                return pd.read_pickle(candidate)
            except Exception:
                continue
        if e == '.csv':
            try:
                return pd.read_csv(candidate)
            except Exception:
                continue

    # nothing found or readable
    raise FileNotFoundError(f"No readable DataFrame found at or near: {path}")
