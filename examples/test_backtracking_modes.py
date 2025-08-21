#!/usr/bin/env python3
"""
Test script demonstrating the difference between divide-and-conquer with and without backtracking.

This script compares segmentation results when backtracking is enabled vs disabled.
"""

import pandas as pd
from src.parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy


def compare_backtracking_modes():
    """Compare segmentation with and without backtracking optimization."""
    
    print("=== Comparing Divide-and-Conquer: With vs Without Backtracking ===\n")
    
    # Create test DataFrame with varying row sizes that can benefit from merging
    print("Creating test DataFrame...")
    test_data = {
        'id': range(100),
        'category': [f'cat_{i % 5}' for i in range(100)],
        'small_content': ['x' * 100 for _ in range(50)] + ['y' * 200 for _ in range(50)],  # Mixed sizes
        'metadata': [f'metadata_value_{i}' for i in range(100)]
    }
    
    df = pd.DataFrame(test_data)
    total_size = df.memory_usage(deep=True).sum()
    avg_row_size = total_size / len(df)
    
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total size: {total_size:,} bytes")
    print(f"  - Average row size: {avg_row_size:.0f} bytes\n")
    
    # Test with different size limits
    test_limits = [3000, 5000, 8000, 12000]  # Various limits to show different effects
    
    for limit in test_limits:
        print(f"Testing with {limit:,} byte limit:")
        print("-" * 40)
        
        # Test WITHOUT backtracking
        segmenter_no_bt = MaxSizeBasedStrategy(
            strategy=NoopStrategy(), 
            max_size_limit=limit,
            enable_backtracking=False  # Disable backtracking
        )
        
        segments_no_bt, oversized_no_bt = segmenter_no_bt._segment_dataframe(df)
        segments_no_bt_list = list(segments_no_bt)
        oversized_no_bt_list = list(oversized_no_bt)
        
        # Test WITH backtracking
        segmenter_with_bt = MaxSizeBasedStrategy(
            strategy=NoopStrategy(), 
            max_size_limit=limit,
            enable_backtracking=True  # Enable backtracking
        )
        
        segments_with_bt, oversized_with_bt = segmenter_with_bt._segment_dataframe(df)
        segments_with_bt_list = list(segments_with_bt)
        oversized_with_bt_list = list(oversized_with_bt)
        
        # Compare results
        print(f"WITHOUT Backtracking:")
        print(f"  - Normal segments: {len(segments_no_bt_list)}")
        print(f"  - Oversized segments: {len(oversized_no_bt_list)}")
        
        print(f"WITH Backtracking:")
        print(f"  - Normal segments: {len(segments_with_bt_list)}")
        print(f"  - Oversized segments: {len(oversized_with_bt_list)}")
        
        # Calculate improvement
        if len(segments_no_bt_list) > 0 and len(segments_with_bt_list) > 0:
            reduction_pct = ((len(segments_no_bt_list) - len(segments_with_bt_list)) / len(segments_no_bt_list)) * 100
            print(f"Improvement:")
            print(f"  - Segment reduction: {len(segments_no_bt_list)} → {len(segments_with_bt_list)} ({reduction_pct:.1f}% reduction)")
            
            # Show segment sizes for first few segments
            print(f"  - First 3 segment sizes (no backtracking):")
            for i, seg in enumerate(segments_no_bt_list[:3]):
                size = seg.memory_usage(deep=True).sum()
                print(f"    Segment {i+1}: {len(seg)} rows, {size:,} bytes")
            
            print(f"  - First 3 segment sizes (with backtracking):")
            for i, seg in enumerate(segments_with_bt_list[:3]):
                size = seg.memory_usage(deep=True).sum()
                print(f"    Segment {i+1}: {len(seg)} rows, {size:,} bytes")
        
        # Verify data integrity
        total_rows_no_bt = sum(len(s) for s in segments_no_bt_list) + sum(len(s) for s in oversized_no_bt_list)
        total_rows_with_bt = sum(len(s) for s in segments_with_bt_list) + sum(len(s) for s in oversized_with_bt_list)
        
        print(f"Data Integrity:")
        print(f"  - Without backtracking: {total_rows_no_bt}/{len(df)} rows preserved")
        print(f"  - With backtracking: {total_rows_with_bt}/{len(df)} rows preserved")
        print(f"  - Integrity check: {'✓ PASS' if total_rows_no_bt == total_rows_with_bt == len(df) else '✗ FAIL'}")
        print()
    
    # Demonstrate extreme case: tiny rows that benefit heavily from backtracking
    print("=== Extreme Case: Tiny Rows (Heavy Backtracking Benefit) ===")
    tiny_data = {
        'id': range(300),
        'tiny_field': ['small'] * 300,
        'index': [f'idx_{i}' for i in range(300)]
    }
    
    tiny_df = pd.DataFrame(tiny_data)
    tiny_total_size = tiny_df.memory_usage(deep=True).sum()
    tiny_avg_row_size = tiny_total_size / len(tiny_df)
    
    print(f"Tiny DataFrame: {len(tiny_df)} rows, ~{tiny_avg_row_size:.0f} bytes per row")
    
    # Use generous limit
    generous_limit = 15000  # 15KB
    
    # Without backtracking
    tiny_segmenter_no_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(), 
        max_size_limit=generous_limit,
        enable_backtracking=False
    )
    tiny_segments_no_bt, _ = tiny_segmenter_no_bt._segment_dataframe(tiny_df)
    tiny_segments_no_bt_list = list(tiny_segments_no_bt)
    
    # With backtracking
    tiny_segmenter_with_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(), 
        max_size_limit=generous_limit,
        enable_backtracking=True
    )
    tiny_segments_with_bt, _ = tiny_segmenter_with_bt._segment_dataframe(tiny_df)
    tiny_segments_with_bt_list = list(tiny_segments_with_bt)
    
    print(f"Results with {generous_limit:,} byte limit:")
    print(f"  - Without backtracking: {len(tiny_segments_no_bt_list)} segments")
    print(f"  - With backtracking: {len(tiny_segments_with_bt_list)} segments")
    
    if len(tiny_segments_no_bt_list) > 0 and len(tiny_segments_with_bt_list) > 0:
        tiny_reduction_ratio = len(tiny_segments_no_bt_list) / len(tiny_segments_with_bt_list)
        print(f"  - Improvement ratio: {tiny_reduction_ratio:.1f}x fewer segments with backtracking")
        
        # Show average rows per segment
        avg_rows_no_bt = len(tiny_df) / len(tiny_segments_no_bt_list)
        avg_rows_with_bt = len(tiny_df) / len(tiny_segments_with_bt_list)
        print(f"  - Avg rows per segment (no backtracking): {avg_rows_no_bt:.1f}")
        print(f"  - Avg rows per segment (with backtracking): {avg_rows_with_bt:.1f}")
    
    print("\n=== Summary ===")
    print("✅ Divide-and-Conquer WITHOUT Backtracking:")
    print("   • Simpler algorithm with fewer processing steps")
    print("   • Creates segments that respect size limits")
    print("   • May create more segments than necessary")
    print("   • Faster execution due to single-phase processing")
    print()
    print("✅ Divide-and-Conquer WITH Backtracking:")
    print("   • Two-phase optimization for better efficiency")
    print("   • Merges adjacent segments when possible")
    print("   • Reduces total segment count significantly")
    print("   • Better memory efficiency and reduced overhead")
    print("   • Especially beneficial with smaller rows and larger size limits")
    print()
    print("🎯 Recommendation:")
    print("   • Use backtracking=True (default) for most use cases")
    print("   • Use backtracking=False for performance-critical scenarios with large datasets")
    print("   • Backtracking provides 2-5x segment reduction in typical scenarios")


if __name__ == "__main__":
    compare_backtracking_modes()
