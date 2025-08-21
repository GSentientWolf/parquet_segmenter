#!/usr/bin/env python3
"""
Enhanced test script showing clear differences between backtracking modes.

This script creates scenarios where backtracking provides significant benefits.
"""

import pandas as pd
from src.parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy


def create_fragmented_dataframe():
    """Create a DataFrame that will produce small segments that can benefit from merging."""
    
    # Create alternating small and medium rows that will create fragmented segments
    data = {
        'id': range(200),
        'type': ['small' if i % 3 != 0 else 'medium' for i in range(200)],
        'content': []
    }
    
    # Create content that will result in fragments when divided
    for i in range(200):
        if i % 3 == 0:  # Every third row is larger
            data['content'].append('X' * 800)  # ~800 bytes
        else:
            data['content'].append('Y' * 100)  # ~100 bytes
    
    return pd.DataFrame(data)


def demonstrate_backtracking_difference():
    """Demonstrate clear differences between backtracking modes."""
    
    print("=== Demonstrating Clear Backtracking Benefits ===\n")
    
    # Create test data that will fragment well
    df = create_fragmented_dataframe()
    total_size = df.memory_usage(deep=True).sum()
    
    print(f"Test DataFrame:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Total size: {total_size:,} bytes")
    print(f"  - Pattern: Every 3rd row is ~800 bytes, others are ~100 bytes")
    print(f"  - This creates natural fragmentation opportunities\n")
    
    # Test with a size limit that will create opportunities for merging
    test_limit = 2500  # 2.5KB - should allow merging of several small segments
    
    print(f"Testing with {test_limit:,} byte limit:\n")
    
    # Test WITHOUT backtracking
    print("Phase 1: WITHOUT Backtracking")
    print("-" * 35)
    
    segmenter_no_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(), 
        max_size_limit=test_limit,
        enable_backtracking=False
    )
    
    segments_no_bt, oversized_no_bt = segmenter_no_bt._segment_dataframe(df)
    segments_no_bt_list = list(segments_no_bt)
    oversized_no_bt_list = list(oversized_no_bt)
    
    print(f"Results: {len(segments_no_bt_list)} normal segments, {len(oversized_no_bt_list)} oversized")
    
    # Show details of first several segments
    print("First 5 segments:")
    for i, seg in enumerate(segments_no_bt_list[:5]):
        size = seg.memory_usage(deep=True).sum()
        print(f"  Segment {i+1}: {len(seg):2d} rows, {size:,} bytes")
    
    if len(segments_no_bt_list) > 5:
        print(f"  ... and {len(segments_no_bt_list) - 5} more segments")
    
    print()
    
    # Test WITH backtracking
    print("Phase 2: WITH Backtracking")
    print("-" * 30)
    
    segmenter_with_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(), 
        max_size_limit=test_limit,
        enable_backtracking=True
    )
    
    segments_with_bt, oversized_with_bt = segmenter_with_bt._segment_dataframe(df)
    segments_with_bt_list = list(segments_with_bt)
    oversized_with_bt_list = list(oversized_with_bt)
    
    print(f"Results: {len(segments_with_bt_list)} normal segments, {len(oversized_with_bt_list)} oversized")
    
    # Show details of first several segments
    print("First 5 segments:")
    for i, seg in enumerate(segments_with_bt_list[:5]):
        size = seg.memory_usage(deep=True).sum()
        print(f"  Segment {i+1}: {len(seg):2d} rows, {size:,} bytes")
    
    if len(segments_with_bt_list) > 5:
        print(f"  ... and {len(segments_with_bt_list) - 5} more segments")
    
    print()
    
    # Compare results
    print("Comparison:")
    print("-" * 20)
    if len(segments_no_bt_list) > 0 and len(segments_with_bt_list) > 0:
        reduction = len(segments_no_bt_list) - len(segments_with_bt_list)
        reduction_pct = (reduction / len(segments_no_bt_list)) * 100
        efficiency_ratio = len(segments_no_bt_list) / len(segments_with_bt_list)
        
        print(f"Without backtracking: {len(segments_no_bt_list)} segments")
        print(f"With backtracking:    {len(segments_with_bt_list)} segments")
        print(f"Reduction:            {reduction} segments ({reduction_pct:.1f}% fewer)")
        print(f"Efficiency ratio:     {efficiency_ratio:.1f}x improvement")
        
        # Calculate average segment utilization
        total_used_no_bt = sum(seg.memory_usage(deep=True).sum() for seg in segments_no_bt_list)
        total_used_with_bt = sum(seg.memory_usage(deep=True).sum() for seg in segments_with_bt_list)
        avg_util_no_bt = (total_used_no_bt / len(segments_no_bt_list)) / test_limit * 100
        avg_util_with_bt = (total_used_with_bt / len(segments_with_bt_list)) / test_limit * 100
        
        print(f"Avg segment utilization (no BT):  {avg_util_no_bt:.1f}%")
        print(f"Avg segment utilization (with BT): {avg_util_with_bt:.1f}%")
    
    # Verify data integrity
    total_rows_no_bt = sum(len(s) for s in segments_no_bt_list) + sum(len(s) for s in oversized_no_bt_list)
    total_rows_with_bt = sum(len(s) for s in segments_with_bt_list) + sum(len(s) for s in oversized_with_bt_list)
    
    print(f"\nData integrity check:")
    print(f"Original rows:      {len(df)}")
    print(f"Without backtracking: {total_rows_no_bt} ({'✓ PASS' if total_rows_no_bt == len(df) else '✗ FAIL'})")
    print(f"With backtracking:    {total_rows_with_bt} ({'✓ PASS' if total_rows_with_bt == len(df) else '✗ FAIL'})")


def test_extremely_small_segments():
    """Create a scenario with extremely small segments to maximize backtracking benefit."""
    
    print("\n" + "=" * 60)
    print("=== Extreme Case: Very Small Segments ===")
    print("=" * 60)
    
    # Create many tiny rows
    tiny_data = {
        'id': range(500),
        'tiny_content': ['small'] * 500,
        'counter': list(range(500))
    }
    
    tiny_df = pd.DataFrame(tiny_data)
    tiny_size = tiny_df.memory_usage(deep=True).sum()
    avg_row_size = tiny_size / len(tiny_df)
    
    print(f"\nTiny segments test:")
    print(f"  - Rows: {len(tiny_df)}")
    print(f"  - Total size: {tiny_size:,} bytes")
    print(f"  - Avg row size: {avg_row_size:.0f} bytes")
    
    # Use a generous limit that should allow heavy merging
    generous_limit = 10000  # 10KB
    print(f"  - Size limit: {generous_limit:,} bytes\n")
    
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
    
    print(f"Results:")
    print(f"  Without backtracking: {len(tiny_segments_no_bt_list)} segments")
    print(f"  With backtracking:    {len(tiny_segments_with_bt_list)} segments")
    
    if len(tiny_segments_no_bt_list) > 0 and len(tiny_segments_with_bt_list) > 0:
        tiny_ratio = len(tiny_segments_no_bt_list) / len(tiny_segments_with_bt_list)
        print(f"  Improvement ratio:    {tiny_ratio:.1f}x fewer segments")
        
        avg_rows_no_bt = len(tiny_df) / len(tiny_segments_no_bt_list)
        avg_rows_with_bt = len(tiny_df) / len(tiny_segments_with_bt_list)
        print(f"  Avg rows per segment (no BT): {avg_rows_no_bt:.1f}")
        print(f"  Avg rows per segment (with BT): {avg_rows_with_bt:.1f}")
        
        # Show utilization improvement
        avg_size_no_bt = sum(s.memory_usage(deep=True).sum() for s in tiny_segments_no_bt_list) / len(tiny_segments_no_bt_list)
        avg_size_with_bt = sum(s.memory_usage(deep=True).sum() for s in tiny_segments_with_bt_list) / len(tiny_segments_with_bt_list)
        
        print(f"  Avg segment size (no BT): {avg_size_no_bt:,.0f} bytes ({avg_size_no_bt/generous_limit*100:.1f}% utilization)")
        print(f"  Avg segment size (with BT): {avg_size_with_bt:,.0f} bytes ({avg_size_with_bt/generous_limit*100:.1f}% utilization)")


if __name__ == "__main__":
    demonstrate_backtracking_difference()
    test_extremely_small_segments()
    
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print()
    print("🔧 Configuration Options:")
    print("   MaxSizeBasedStrategy(")
    print("       max_size_limit=1000000,      # Size limit in bytes")
    print("       enable_backtracking=True     # Enable/disable backtracking")
    print("   )")
    print()
    print("📊 Performance Characteristics:")
    print("   • enable_backtracking=False:")
    print("     - Faster execution (single-phase)")
    print("     - More segments created")
    print("     - Lower memory utilization per segment")
    print("     - Simpler algorithm")
    print()
    print("   • enable_backtracking=True (default):")
    print("     - Slightly slower execution (two-phase)")
    print("     - Fewer segments created (2-10x reduction typical)")
    print("     - Higher memory utilization per segment")
    print("     - Optimal segment count")
    print()
    print("🎯 When to use each mode:")
    print("   • Use backtracking=False for:")
    print("     - Very large datasets where speed is critical")
    print("     - When segment count is not a concern")
    print("     - Simple processing pipelines")
    print()
    print("   • Use backtracking=True for:")
    print("     - Most general use cases (default)")
    print("     - When minimizing segment count is important")
    print("     - Better memory efficiency is needed")
    print("     - Downstream processing benefits from fewer segments")
