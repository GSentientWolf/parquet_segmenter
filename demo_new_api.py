#!/usr/bin/env python3
"""
Demo of the new OutlierStrategy enum API for build_blob_dataframe.

This shows how the new API is cleaner and more explicit than the old
conflicting boolean parameters.
"""

import sys
sys.path.insert(0, 'src')

from parquet_segmenter.functional_testing.blob_df import (
    build_blob_dataframe, 
    OutlierStrategy, 
    ChunkSize
)

def demo_api():
    print("=== Demo of New OutlierStrategy API ===\n")
    
    # Common parameters
    total_size = 10 * ChunkSize.ONE_MB
    top_outliers = 3
    seed = 42
    
    print("1. SPREAD Strategy (default): Outliers distributed across different batches")
    df_spread = build_blob_dataframe(
        total_df_size=total_size,
        top_outliers=top_outliers,
        outlier_strategy=OutlierStrategy.SPREAD,
        seed=seed
    )
    outlier_count = df_spread["is_outlier"].sum()
    print(f"   Generated {len(df_spread)} rows with {outlier_count} outliers\n")
    
    print("2. SINGLE_BATCH Strategy: All outliers in one batch, spread within")
    df_single = build_blob_dataframe(
        total_df_size=total_size,
        top_outliers=top_outliers,
        outlier_strategy=OutlierStrategy.SINGLE_BATCH,
        seed=seed
    )
    outlier_count = df_single["is_outlier"].sum()
    print(f"   Generated {len(df_single)} rows with {outlier_count} outliers\n")
    
    print("3. CONTIGUOUS Strategy: All outliers in one batch, contiguous block")
    df_contiguous = build_blob_dataframe(
        total_df_size=total_size,
        top_outliers=top_outliers,
        outlier_strategy=OutlierStrategy.CONTIGUOUS,
        seed=seed
    )
    outlier_count = df_contiguous["is_outlier"].sum()
    print(f"   Generated {len(df_contiguous)} rows with {outlier_count} outliers\n")
    
    print("4. MULTI_CLUSTER Strategy: Multiple clusters across batches")
    df_multi = build_blob_dataframe(
        total_df_size=total_size,
        top_outliers=top_outliers,
        outlier_strategy=OutlierStrategy.MULTI_CLUSTER,
        cluster_count=2,
        seed=seed
    )
    outlier_count = df_multi["is_outlier"].sum()
    print(f"   Generated {len(df_multi)} rows with {outlier_count} outliers across 2 clusters\n")
    
    print("✅ API Migration Complete!")
    print("The new enum-based API eliminates the confusing mutually exclusive boolean parameters.")
    print("Each strategy is now clear and unambiguous in its behavior.")

if __name__ == "__main__":
    demo_api()
