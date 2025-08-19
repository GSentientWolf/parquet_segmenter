from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd


def faked_memory_usage(df: pd.DataFrame, index_sizes_col: str = "sizes") -> int:
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
def patch_memory_usage(index_sizes_col: str = "sizes") -> Iterator[None]:
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
