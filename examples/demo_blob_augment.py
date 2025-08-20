#!/usr/bin/env python3
"""Demo: use build_blob_dataframe + augment_with_faker_and_timestamps.

Run with the project venv python, e.g.:
    .venv/bin/python examples/demo_blob_augment.py

This script builds a small blob-size DataFrame, augments it with Faker
generated left-side columns and monotonic timestamps, and prints a few
representative outputs.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import List

from parquet_segmenter.functional_testing.blob_df import (
    build_blob_dataframe,
    augment_with_faker_and_timestamps,
    visualize_batch_layout,
    OutlierStrategy,
    ChunkSize,
)
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
from parquet_segmenter.utils.utils import faked_memory_usage, patch_memory_usage


def main(left_cols: int = 4, rows: int = 50, no_stdout: bool = False) -> None:
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_lines: List[str] = []

    def emit(s: str = "") -> None:
        """Collect output or print depending on no_stdout."""
        if no_stdout:
            out_lines.append(s)
        else:
            print(s)

    # Build a small deterministic blob DataFrame
    df = build_blob_dataframe(
        total_df_size=4 * ChunkSize.ONE_MB,
        top_outliers=2,
        outlier_strategy=OutlierStrategy.SPREAD,
        cluster_count=1,
        seed=42,
    )

    print("Original blob DataFrame (head):")
    print(df.head().to_string(index=False))

    # Augment with fake left-side columns and monotonic timestamps
    initial_date = "2020-01-01T00:00:00"
    augmented = augment_with_faker_and_timestamps(
        df,
        initial_date=initial_date,
        faker_seed=42,
        faker_locale="en_US",
        tz="UTC",
    )

    # Pretty-print the augmented DataFrame (tries rich, then tabulate, then pandas fallback)
    try:
        # Show only first N columns (left_cols) plus blob_size and size for console output
        cols = list(augmented.columns[:left_cols])
        for extra in ("blob_size", "size"):
            if extra in augmented.columns and extra not in cols:
                cols.append(extra)

        # Prepare the rows to display and an appended summary row
        disp_df = augmented[cols].head(rows)
        # Build a tabulate-friendly representation and a summary row
        try:
            size_col_name = next(c for c in cols if "size" in str(c).lower() or "byte" in str(c).lower() or "length" in str(c).lower())
        except StopIteration:
            size_col_name = None

        # Attempt to use rich for pretty display when not capturing to file
        if not no_stdout:
            console = Console()
            table = Table(show_header=True, header_style="bold")
            for col in cols:
                table.add_column(str(col))
            for row in disp_df.itertuples(index=False):
                table.add_row(*[("" if v is None else str(v)) for v in row])
            table.add_section()
            if size_col_name is not None:
                total_size = int(augmented[size_col_name].sum())
                summary_row = ["" for _ in cols]
                summary_row[0] = "TOTAL"
                idx = cols.index(size_col_name)
                summary_row[idx] = str(total_size / 1024**2) + " MB"
                summary_row[(idx + 1) % len(summary_row)] = f"{len(augmented)} rows"
                table.add_row(*summary_row)
            console.print(table)
        else:
            # Build plain-text table via tabulate for file output
            rows_list = [list(map(lambda v: "" if v is None else str(v), r)) for r in disp_df.values.tolist()]
            if size_col_name is not None:
                total_size = int(augmented[size_col_name].sum())
                summary = ["" for _ in cols]
                summary[0] = "TOTAL"
                idx = cols.index(size_col_name)
                summary[idx] = str(total_size / 1024**2) + " MB"
                summary[(idx + 1) % len(summary)] = f"{len(augmented)} rows"
                rows_list.append(summary)
            tab_text = tabulate(rows_list, headers=cols, tablefmt="psql", showindex=False)
            out_lines.append(tab_text)
    except (ImportError, AttributeError, TypeError, ValueError):
        try:
            tab_text = tabulate(augmented[cols].head(rows).values.tolist(), headers=cols, tablefmt="psql", showindex=False)
            if no_stdout:
                out_lines.append(tab_text)
            else:
                print(tab_text)
        except (ImportError, TypeError, ValueError):
            text = augmented[cols].head(min(8, rows)).to_string(index=False)
            if no_stdout:
                out_lines.append(text)
            else:
                print(text)

    # Basic statistics
    print("\nColumns:", list(augmented.columns))
    print("Total size (sum of 'size'):", int(augmented['size'].sum()), "bytes")

    # Compute batch_rows the same way used by other examples for visualization
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 10, 14]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    print("\nBatch layout visualization:")
    print(visualize_batch_layout(augmented, batch_rows))

    # Use the existing 'size' column as the per-row size footprint and demo memory usage helpers
    if 'size' in augmented.columns:
        fake_total = faked_memory_usage(augmented, index_sizes_col='size')
        emit(f"\nFake total bytes (computed from 'size'): {fake_total} bytes")

        # Show original memory usage for comparison (best-effort)
        try:
            orig_mem = augmented.memory_usage(index=True, deep=True)
            emit("Original memory usage series:\n" + str(orig_mem))
        except (ValueError, TypeError, OSError):
            pass

        # Demonstrate temporary patching of DataFrame.memory_usage to use 'size'
        with patch_memory_usage('size'):
            mem_series = augmented.memory_usage(index=True, deep=True)
            emit("Patched memory usage series:\n" + str(mem_series))

            # Show fake memory usage for comparison
            try:
                fake_mem = augmented.memory_usage(index=True, deep=True)
                emit("Fake memory usage series:\n" + str(fake_mem))
            except (ValueError, TypeError, OSError):
                pass
            try:
                emit("\nPatched memory usage for sections:")
                n = min(5, len(augmented))
                sections = {
                    "all": augmented.iloc[:],
                    "first": augmented.head(n),
                    "middle": augmented.iloc[len(augmented) // 2 - n + 1 : len(augmented) // 2 + n],
                    "last": augmented.tail(n),
                }
                for name, sec in sections.items():
                    mem = sec.memory_usage(index=True, deep=True)
                    emit(f"\nSection '{name}' ({len(sec)} rows):")
                    emit(str(mem))
                    emit(f"Total bytes: {int(mem.sum())}")

                total_sections = sum(int(sec.memory_usage(index=True, deep=True).sum()) for sec in sections.values())
                total_whole = int(augmented.memory_usage(index=True, deep=True).sum())
                emit(f"\nSum of section totals: {total_sections} bytes")
                emit(f"Patched whole-DataFrame total: {total_whole} bytes")
            except (ValueError, TypeError, OSError):
                pass

    else:
        print("Column 'size' not found in DataFrame.")

    # Save a small CSV sample
    sample_path = out_dir / "augmented_sample.csv"
    augmented.head(rows).to_csv(sample_path, index=False)
    emit(f"\nWrote sample CSV to: {sample_path}")

    # If capturing output, write to file
    if no_stdout:
        out_path = out_dir / "augmented_output.txt"
        out_path.write_text("\n".join(out_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo augmenting a blob DataFrame and visualizing the result")
    parser.add_argument("--left-cols", type=int, default=4, help="number of left columns to display in the console table (default: 4)")
    parser.add_argument("--rows", type=int, default=50, help="number of rows to include in table output and sample CSV (default: 50)")
    parser.add_argument("--no-stdout", action="store_true", help="capture output to file instead of printing to stdout")
    args = parser.parse_args()
    main(left_cols=args.left_cols, rows=args.rows, no_stdout=args.no_stdout)
