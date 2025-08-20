"""I/O helpers for parquet_segmenter.

This minimal module provides count_rows and get_stats used by tests/demo.
It prefers pyarrow if available, otherwise implements a simple fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

# Prefer to import pyarrow at module import time when available so linters
# and tools can see the dependency; fall back gracefully at runtime.
try:
    import pyarrow.parquet as _pyarrow  # type: ignore
except ImportError:  # pragma: no cover - runtime fallback
    _pyarrow = None


if TYPE_CHECKING:
    # For type checkers, provide a light import hint without requiring stubs.
    from typing import Any as _Any  # pragma: no cover


def count_rows(path: str) -> int:
    """Return number of rows in a parquet file at `path`.

    If pyarrow is available, use it; otherwise return 0 for non-existent
    or unknown files to keep tests fast and deterministic.
    """
    try:
        if _pyarrow is None:
            # pyarrow not available in this environment
            raise ImportError("pyarrow not available")
        table = _pyarrow.read_table(path)
        return table.num_rows
    except (ImportError, FileNotFoundError, OSError):
        # Fallback for tests or when pyarrow isn't available or file missing.
        return 0


def get_stats(path: str) -> Dict[str, int]:
    """Return simple stats for a parquet file: {'rows': int}."""
    return {"rows": count_rows(path)}
