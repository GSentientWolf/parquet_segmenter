"""Demo that writes a DataFrame to multiple formats and prints summaries.

Run with:
    .venv/bin/python examples/utils/demo_utils_more_formats.py

This script will:
- create a small DataFrame with mixed dtypes
- write it to parquet (compressed if engine available), feather, pickle, csv
- print file sizes and whether reading back preserves data (non-strict dtype check)
"""
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd

from parquet_segmenter.utils import save_dataframe_best, read_dataframe_best


def has_parquet_engine() -> bool:
    try:
        import pyarrow  # type: ignore
        return True
    except ImportError:
        try:
            import fastparquet  # type: ignore
            return True
        except ImportError:
            return False


def try_assert_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        return False


def main():
    out_dir = Path(__file__).parent / "out_more"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "i": range(5),
            "f": [0.1, 2.5, 3.14, 4.0, 5.5],
            "s": ["a", "bb", "ccc", "d", "e"],
            "dt": pd.date_range("2020-01-01", periods=5, freq="D"),
            "cat": pd.Categorical(["x", "y", "x", "y", "z"]),
        }
    )

    print("Original DataFrame:\n", df)
    print("Memory usage (deep):", int(df.memory_usage(index=True, deep=True).sum()))

    parquet_ok = has_parquet_engine()
    compression = "gzip" if parquet_ok else None

    formats = [".parquet", ".feather", ".pkl", ".csv"]

    results = []
    for ext in formats:
        target = out_dir / ("demo" + ext)
        written = None
        try:
            if ext == ".parquet":
                # use helper to prefer parquet and provide compression when available
                written = save_dataframe_best(df, str(target), prefer_parquet=True, compression=compression)
            elif ext == ".feather":
                df.to_feather(target)
                written = str(target)
            elif ext in (".pkl", ".pickle"):
                df.to_pickle(target)
                written = str(target)
            elif ext == ".csv":
                df.to_csv(target, index=False)
                written = str(target)
        except (OSError, ValueError, TypeError) as exc:
            results.append((ext, None, False, f"write-failed: {exc}"))
            continue

        if not written or not Path(written).exists():
            results.append((ext, None, False, "not-written"))
            continue

        size = Path(written).stat().st_size
        # attempt read back
        try:
            read = read_dataframe_best(written)
        except (OSError, ValueError, TypeError) as exc:
            results.append((ext, size, False, f"read-failed: {exc}"))
            continue

        ok = try_assert_equal(df, read)
        results.append((ext, size, ok, None))

    print("\nSummary:")
    for ext, size, ok, err in results:
        if err:
            status = f"ERROR: {err}"
        else:
            status = "OK" if ok else "MISMATCH"
        print(f"  {ext:8} size={size:8} bytes  status={status}")


if __name__ == "__main__":
    main()
