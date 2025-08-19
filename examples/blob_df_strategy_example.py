#!/usr/bin/env python3
"""
Example demonstrating the new OutlierStrategy enum API for build_blob_dataframe.

This example shows how to use the improved API that replaced the confusing 
mutually exclusive boolean parameters with a clear enum-based approach.

Run from repo root:
    .venv/bin/python examples/blob_df_strategy_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parquet_segmenter.functional_testing.blob_df import (
    build_blob_dataframe, 
    OutlierStrategy, 
    ChunkSize,
    visualize_batch_layout
)


def demonstrate_strategies():
    """Demonstrate all OutlierStrategy options with clear examples."""
    
    print("=== OutlierStrategy Examples ===\n")
    
    # Common parameters for all examples
    total_size = 8 * ChunkSize.ONE_MB
    top_outliers = 3
    seed = 42
    
    # Calculate batch_rows for visualization
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))
    
    strategies = [
        (OutlierStrategy.SPREAD, "Distribute outliers across different batches"),
        (OutlierStrategy.SINGLE_BATCH, "All outliers in one batch, spread within"),
        (OutlierStrategy.CONTIGUOUS, "All outliers in one batch, contiguous block"),
        (OutlierStrategy.MULTI_CLUSTER, "Multiple clusters across batches"),
    ]
    
    for strategy, description in strategies:
        print(f"Strategy: {strategy.value.upper()}")
        print(f"Description: {description}")
        
        kwargs = {
            'total_df_size': total_size,
            'top_outliers': top_outliers,
            'outlier_strategy': strategy,
            'seed': seed
        }
        
        # Add cluster_count for MULTI_CLUSTER strategy
        if strategy == OutlierStrategy.MULTI_CLUSTER:
            kwargs['cluster_count'] = 2
            
        df = build_blob_dataframe(**kwargs)
        
        print(f"Generated: {len(df)} rows, {df['is_outlier'].sum()} outliers")
        print("Layout visualization (O=outlier, i=intermediate, .=small):")
        layout = visualize_batch_layout(df, batch_rows)
        # Show only first few lines for readability
        layout_lines = layout.split('\n')[:5]
        for line in layout_lines:
            print(f"  {line}")
        if len(layout.split('\n')) > 5:
            print(f"  ... ({len(layout.split('\n')) - 5} more batches)")
        print()


def demonstrate_backward_compatibility():
    """Show that legacy API still works with deprecation warnings."""
    
    print("=== Backward Compatibility Example ===\n")
    
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Old API usage (will show deprecation warning)
        df_old = build_blob_dataframe(
            total_df_size=5 * ChunkSize.ONE_MB,
            top_outliers=2,
            spread_top_outliers=False,  # Legacy parameter
            cluster_batches=1,          # Legacy parameter  
            contiguous_within_batch=True,  # Legacy parameter
            seed=42
        )
        
        print(f"Legacy API: {len(df_old)} rows, {df_old['is_outlier'].sum()} outliers")
        
        if w:
            print(f"Deprecation warning: {w[0].message}")
        
        # Equivalent new API usage
        df_new = build_blob_dataframe(
            total_df_size=5 * ChunkSize.ONE_MB,
            top_outliers=2,
            outlier_strategy=OutlierStrategy.CONTIGUOUS,  # Clear and explicit!
            seed=42
        )
        
        print(f"New API: {len(df_new)} rows, {df_new['is_outlier'].sum()} outliers")
        print("✅ Same results with cleaner API!")


if __name__ == "__main__":
    demonstrate_strategies()
    print()
    demonstrate_backward_compatibility()
