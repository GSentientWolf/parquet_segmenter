"""Small demo for parquet_segmenter.utils utilities.

Run this from the project root with the venv python, for example:

    .venv/bin/python examples/utils/demo_utils.py

This script demonstrates:
- faked_memory_usage()
- patch_memory_usage() context manager
- save_dataframe_best() and read_dataframe_best()

The script will attempt to write a compressed parquet file when a parquet
engine is available, otherwise it will fall back to pickle.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from parquet_segmenter.utils import (
    faked_memory_usage,
    patch_memory_usage,
    save_dataframe_best,
    read_dataframe_best,
)


def main():
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "val": [10, 20, 30],
        "index_size_bytes": [100, 200, 50],
    })

    print("DataFrame:\n", df)

    real_mem = int(df.memory_usage(index=False, deep=True).sum())
    fake_mem = faked_memory_usage(df, index_sizes_col="index_size_bytes")
    print(f"Real column bytes: {real_mem}")
    print(f"Faked total bytes (columns + index sizes): {fake_mem}")

    # Show the patch in action
    print("\nMemory usage breakdown (before patch):")
    print(df.memory_usage(index=True, deep=True))

    with patch_memory_usage("index_size_bytes"):
        print("\nMemory usage breakdown (inside patch):")
        print(df.memory_usage(index=True, deep=True))

    # Save & read back using the helpers
    prefers_parquet = True
    try:
        import pyarrow  # type: ignore
        compression = "gzip"
    except ImportError:
        try:
            import fastparquet  # type: ignore
            compression = "gzip"
        except ImportError:
            compression = None

    target = out_dir / ("example.parquet" if compression else "example.pkl")
    written = save_dataframe_best(df, str(target), prefer_parquet=prefers_parquet, compression=compression)
    print(f"\nWrote dataframe to: {written}")

    df2 = read_dataframe_best(str(target))
    print("Read dataframe equals original:", df2.equals(df))


if __name__ == "__main__":
    main()
