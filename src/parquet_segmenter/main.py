"""Small CLI-like helpers for module demo and tests."""
from __future__ import annotations

from .io import count_rows


def summarize(path: str) -> str:
    """Return a one-line summary for the provided path."""
    rows = count_rows(path)
    return f"{path}: {rows} rows"


if __name__ == "__main__":
    print(summarize("example.parquet"))
