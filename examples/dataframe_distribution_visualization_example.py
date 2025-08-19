#!/usr/bin/env python3
"""Test the visualization function."""

from src.parquet_segmenter.functional_testing.blob_df import build_blob_dataframe, visualize_batch_layout, ChunkSize
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
from parquet_segmenter.utils.utils import faked_memory_usage, patch_memory_usage

def main():
    print("Testing visualization function...")
    
    # Create a small test DataFrame
    df = build_blob_dataframe(
        total_df_size=3 * ChunkSize.ONE_MB,
        top_outliers=2,
        spread_top_outliers=False,
        cluster_batches=1,
        seed=42
    )

    # Pretty-print the DataFrame as a table (tries rich, then tabulate, then pandas fallback)
    try:
        console = Console()
        table = Table(show_header=True, header_style="bold")
        for col in df.columns:
            table.add_column(str(col))
        for row in df.itertuples(index=False):
            table.add_row(*[("" if v is None else str(v)) for v in row])
        # add a summary row with the total of the size-like column (if any)
        size_col = None
        for column in df.columns:
            lower_column = str(column).lower()
            if "size" in lower_column or "byte" in lower_column or "length" in lower_column:
                size_col = column
                break
        table.add_section()
        if size_col is not None:
            total_size = df[size_col].sum()
            summary_row = ["" for _ in df.columns]
            summary_row[0] = "TOTAL"
            summary_row[list(df.columns).index(size_col)] = str(total_size / 1024**2) + " MB"
            table.add_row(*summary_row)
        console.print(table)
    except Exception:
        try:
            # tabulate produces nice ASCII tables if available
            print(tabulate(df.values.tolist(), headers=list(df.columns), tablefmt="psql", showindex=False))
        except Exception:
            # final fallback
            print(df.to_string(index=False))

    # Calculate batch_rows the same way as the function
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 10, 14]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Batch rows: {batch_rows}")
    print(f"Outliers: {df['is_outlier'].sum()}")
    print(f"Intermediates: {df['is_intermediate'].sum()}")
    print("\nVisualization:")
    print(visualize_batch_layout(df, batch_rows))

    # Use the existing 'sizes' column as the per-row size footprint
    if 'size' in df.columns:
        fake_total = faked_memory_usage(df, index_sizes_col='size')
        print(f"Fake total bytes (computed from 'size'): {fake_total} bytes")

        # Show original memory usage for comparison (best-effort)
        try:
            orig_mem = df.memory_usage(index=True, deep=True)
            print("Original memory usage series:\n", orig_mem)
        except Exception:
            pass

        # Demonstrate temporary patching of DataFrame.memory_usage to use 'size'
        with patch_memory_usage('size'):
            mem_series = df.memory_usage(index=True, deep=True)
            print("Patched memory usage series:\n", mem_series)

            # Show fake memory usage for comparison
            try:
                fake_mem = df.memory_usage(index=True, deep=True)
                print("Fake memory usage series:\n", fake_mem)
            except Exception:
                pass
            try:
                # Example: compute patched memory_usage for several sections of the DataFrame
                print("\nPatched memory usage for sections:")
                n = min(5, len(df))
                sections = {
                    "all": df.iloc[:],
                    "first": df.head(n),
                    "middle": df.iloc[len(df) // 2 - n + 1 : len(df) // 2 + n],
                    "last": df.tail(n),
                }
                for name, sec in sections.items():
                    mem = sec.memory_usage(index=True, deep=True)
                    print(f"\nSection '{name}' ({len(sec)} rows):")
                    print(mem)
                    print("Total bytes:", int(mem.sum()))

                # Compare sum of section totals to the patched whole-DataFrame total
                total_sections = sum(int(sec.memory_usage(index=True, deep=True).sum()) for sec in sections.values())
                total_whole = int(df.memory_usage(index=True, deep=True).sum())
                print(f"\nSum of section totals: {total_sections} bytes")
                print(f"Patched whole-DataFrame total: {total_whole} bytes")
            except Exception:
                pass

    else:
        print("Column 'size' not found in DataFrame.")

if __name__ == "__main__":
    main()
