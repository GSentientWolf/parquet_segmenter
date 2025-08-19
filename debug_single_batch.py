#!/usr/bin/env python3
"""Debug the SINGLE_BATCH strategy issue."""

import sys
sys.path.insert(0, 'src')

from parquet_segmenter.functional_testing.blob_df import (
    build_blob_dataframe, 
    OutlierStrategy, 
    ChunkSize
)

def debug_single_batch():
    top = 3
    df = build_blob_dataframe(
        total_df_size=15 * ChunkSize.ONE_MB, 
        top_outliers=top, 
        outlier_strategy=OutlierStrategy.SINGLE_BATCH, 
        seed=42
    )
    
    print(f"DataFrame length: {len(df)}")
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    print(f"Outlier count: {len(out_idx)}")
    print(f"Outlier indices: {out_idx[:10]}")  # First 10
    
    # Use function's actual default small_bins
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))
    
    print(f"Default small_bins: {small_bins}")
    print(f"avg_small: {avg_small}")
    print(f"batch_rows: {batch_rows}")
    
    batches = [idx // batch_rows for idx in out_idx[:top]]
    print(f"Batches for first {top} outliers: {batches}")
    print(f"Unique batches: {set(batches)}")
    
    # Check all outliers
    all_batches = [idx // batch_rows for idx in out_idx]
    print(f"All outlier batches: {all_batches}")
    print(f"All unique batches: {set(all_batches)}")

if __name__ == "__main__":
    debug_single_batch()
