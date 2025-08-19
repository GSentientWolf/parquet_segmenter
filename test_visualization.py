#!/usr/bin/env python3
"""Test the visualization function."""

from src.parquet_segmenter.testing.blob_df import build_blob_dataframe, visualize_batch_layout, ChunkSize

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

if __name__ == "__main__":
    main()
