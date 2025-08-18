"""Utility helpers for parquet_segmenter.

Keep this module small — add helpers here that are shared across the project.
"""

from typing import Iterable, List, TypeVar

T = TypeVar("T")


def chunked(seq: Iterable[T], size: int) -> List[List[T]]:
    """Split an iterable into chunks of at most `size`.

    Examples:
        chunked([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    out: List[List[T]] = []
    chunk: List[T] = []
    for item in seq:
        chunk.append(item)
        if len(chunk) >= size:
            out.append(chunk)
            chunk = []
    if chunk:
        out.append(chunk)
    return out


__all__ = ["chunked"]
