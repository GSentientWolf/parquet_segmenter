"""Utility helpers for memory-usage estimation and DataFrame persistence.

This module contains small helpers used across the project for tests and
examples. Some functions are intentionally pragmatic (trying multiple file
formats) and therefore trigger pylint complexity warnings; those are locally
disabled with in-line pragmas where a deeper refactor would be disruptive.
"""

# Some functions in this module intentionally perform multi-format probing
# and return from multiple branches. We add a module-level disable for the
# specific pylint checks to keep the implementation readable; a full refactor
# can be scheduled separately.
# pylint: disable=too-many-branches,too-many-return-statements

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

import pickle
import pandas as pd


# --- Format-specific helpers -------------------------------------------------
def _write_parquet(
    df: pd.DataFrame,
    path: str,
    compression: Optional[Literal["snappy", "gzip", "brotli", "lz4", "zstd"]]
    = None,
) -> bool:
    try:
        # Use the typing-friendly Literal for compression so static checkers
        # accept passing the value through. pandas may raise ImportError if
        # optional parquet engines are not installed.
        if compression is None:
            df.to_parquet(path)
        else:
            df.to_parquet(path, compression=compression)
        return True
    except (ImportError, OSError, ValueError):
        return False


def _write_feather(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_feather(path)
        return True
    except (ImportError, OSError, ValueError):
        return False


def _write_pickle(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_pickle(path)
        return True
    except (OSError, pickle.PicklingError, TypeError):
        return False


def _write_csv(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except (OSError, ValueError):
        return False


def _read_parquet(pth: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(pth)
    except (ImportError, OSError, ValueError):
        return None


def _read_feather(pth: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_feather(pth)
    except (ImportError, OSError, ValueError):
        return None


def _read_pickle(pth: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_pickle(pth)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        return None


def _read_csv(pth: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(pth)
    except (OSError, ValueError):
        return None



def faked_memory_usage(
    df: pd.DataFrame, index_sizes_col: str = "index_size_bytes"
) -> int:
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

    # pylint: disable=too-many-return-statements,too-many-branches
    def _patched(self, index: bool = True, deep: bool = False):
        s = orig(self, index=index, deep=deep)
        if index:
            try:
                fake_index = (
                    int(self[index_sizes_col].sum())
                    if index_sizes_col in self.columns
                    else 0
                )
            except (KeyError, TypeError, ValueError):
                fake_index = 0
            try:
                s.loc["Index"] = fake_index
            except (AttributeError, KeyError):
                s["Index"] = fake_index
        return s

    pd.DataFrame.memory_usage = _patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        pd.DataFrame.memory_usage = orig  # type: ignore[method-assign]


def save_dataframe_best(
    df: pd.DataFrame,
    path: str,
    prefer_parquet: bool = True,
    compression: Optional[Literal["snappy", "gzip", "brotli", "lz4", "zstd"]] = None,
) -> str:
    """Save a DataFrame using the best available format.

    Tries Parquet/Feather/Pickle/CSV in that order (or reversed if prefer_parquet
    is False). If the provided `path` has an extension that pandas understands
    the function will honor that.

    Returns the path that was written.
    """

    p = Path(path)
    base = str(p.with_suffix(""))
    ext = p.suffix.lower()

    # Delegate actual format-specific writes to the module-level helpers
    # which centralize import/IO error handling.

    # honor explicit extension
    if ext in (".parquet", ".pq"):
        if _write_parquet(df, path, compression=compression):
            return path
    if ext == ".feather":
        if _write_feather(df, path):
            return path
    if ext in (".pkl", ".pickle"):
        if _write_pickle(df, path):
            return path
    if ext == ".csv":
        if _write_csv(df, path):
            return path

    # choose order based on preference
    if prefer_parquet:
        try_order = [".parquet", ".feather", ".pkl", ".csv"]
    else:
        try_order = [".feather", ".parquet", ".pkl", ".csv"]

    for e in try_order:
        pth = base + e
        if e == ".parquet" and _write_parquet(df, pth, compression=compression):
            return pth
        if e == ".feather" and _write_feather(df, pth):
            return pth
        if e in (".pkl", ".pickle") and _write_pickle(df, pth):
            return pth
        if e == ".csv" and _write_csv(df, pth):
            return pth

    raise RuntimeError("Failed to save DataFrame to any supported format")


def read_dataframe_best(path: str) -> pd.DataFrame:  # pylint: disable=too-many-branches,too-many-return-statements
    """Read a DataFrame written by :func:`save_dataframe_best`.

    If `path` has a known extension this will use the appropriate pandas
    reader. If the path doesn't exist or has no extension the function will
    try common extensions in the order: parquet, feather, pickle, csv.

    Raises:
        FileNotFoundError: if no matching file is found.
        RuntimeError: if a file is found but none of the readers succeed.
    """
    p = Path(path)

    # Use the module-level readers which centralize optional-dependency
    # handling and error mapping to None on failure.

    # if the exact path exists, prefer direct read based on suffix
    if p.exists():
        ext = p.suffix.lower()
        if ext in (".parquet", ".pq"):
            df = _read_parquet(p)
            if df is None:
                raise RuntimeError(f"Failed to read parquet file: {p}")
            return df
        if ext == ".feather":
            df = _read_feather(p)
            if df is None:
                raise RuntimeError(f"Failed to read feather file: {p}")
            return df
        if ext in (".pkl", ".pickle"):
            df = _read_pickle(p)
            if df is None:
                raise RuntimeError(f"Failed to read pickle file: {p}")
            return df
        if ext == ".csv":
            df = _read_csv(p)
            if df is None:
                raise RuntimeError(f"Failed to read csv file: {p}")
            return df

    # if the path itself doesn't exist or had no known suffix, try common
    # extensions by probing for files next to the given path.
    try_order = [".parquet", ".feather", ".pkl", ".csv"]
    for e in try_order:
        candidate = p.with_suffix(e)
        if not candidate.exists():
            continue
        if e == ".parquet":
            df = _read_parquet(candidate)
            if df is not None:
                return df
            continue
        if e == ".feather":
            df = _read_feather(candidate)
            if df is not None:
                return df
            continue
        if e in (".pkl", ".pickle"):
            df = _read_pickle(candidate)
            if df is not None:
                return df
            continue
        if e == ".csv":
            df = _read_csv(candidate)
            if df is not None:
                return df
            continue

    # nothing found or readable
    raise FileNotFoundError(f"No readable DataFrame found at or near: {path}")
