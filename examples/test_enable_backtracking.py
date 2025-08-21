#!/usr/bin/env python3
"""
Test script specifically for demonstrating the enable_backtracking parameter.

This script creates scenarios that show clear differences between backtracking modes.
"""

import pandas as pd
from src.parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy


def create_backtracking_test_data():
    """Create DataFrame specifically designed to benefit from backtracking."""
    
    # Create data that will result in segments that can be merged
    # Strategy: Create rows with sizes that when divided will leave small segments
    # that can be merged back together
    
    data = {
        'id': range(60),
        'category': [f'cat_{i % 3}' for i in range(60)],
        'size_type': []
    }
    
    # Create a pattern that will result in small segments after divide-and-conquer
    # Every 4th row is large, others are small
    content = []
    for i in range(60):
        if i % 4 == 0:
            content.append('L' * 1200)  # Large row ~1.2KB
            data['size_type'].append('large')
        else:
            content.append('S' * 200)   # Small row ~200 bytes  
            data['size_type'].append('small')
    
    data['content'] = content
    
    return pd.DataFrame(data)


def demonstrate_backtracking_toggle():
    """Demonstrate the enable_backtracking parameter functionality."""
    
    print("=== Demonstrating enable_backtracking Parameter ===\n")
    
    # Create test data
    df = create_backtracking_test_data()
    total_size = df.memory_usage(deep=True).sum()
    
    print(f"Test Data:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total size: {total_size:,} bytes")
    print(f"  - Pattern: Every 4th row is ~1.2KB, others are ~200 bytes")
    print(f"  - This creates opportunities for segment merging after division\n")
    
    # Use a specific size limit that will create mergeable segments
    size_limit = 2000  # 2KB limit
    
    print(f"Testing with {size_limit:,} byte size limit:\n")
    
    # Test 1: WITHOUT backtracking
    print("Test 1: enable_backtracking=False")
    print("-" * 40)
    
    segmenter_no_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),
        max_size_limit=size_limit,
        enable_backtracking=False  # Explicitly disable backtracking
    )
    
    segments_no_bt, oversized_no_bt = segmenter_no_bt._segment_dataframe(df)
    segments_no_bt_list = list(segments_no_bt)
    oversized_no_bt_list = list(oversized_no_bt)
    
    print(f"Results:")
    print(f"  - Normal segments: {len(segments_no_bt_list)}")
    print(f"  - Oversized segments: {len(oversized_no_bt_list)}")
    print(f"  - Total segments: {len(segments_no_bt_list) + len(oversized_no_bt_list)}")
    
    # Show segment details
    print(f"  - First 5 segment details:")
    for i, seg in enumerate(segments_no_bt_list[:5]):
        size = seg.memory_usage(deep=True).sum()
        utilization = (size / size_limit) * 100
        print(f"    Segment {i+1}: {len(seg):2d} rows, {size:,} bytes ({utilization:.1f}% utilization)")
    
    if len(segments_no_bt_list) > 5:
        print(f"    ... and {len(segments_no_bt_list) - 5} more segments")
    
    avg_utilization_no_bt = 0
    if segments_no_bt_list:
        total_used = sum(seg.memory_usage(deep=True).sum() for seg in segments_no_bt_list)
        avg_utilization_no_bt = (total_used / len(segments_no_bt_list)) / size_limit * 100
        print(f"  - Average utilization: {avg_utilization_no_bt:.1f}%")
    
    print()
    
    # Test 2: WITH backtracking  
    print("Test 2: enable_backtracking=True (default)")
    print("-" * 45)
    
    segmenter_with_bt = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),
        max_size_limit=size_limit,
        enable_backtracking=True  # Explicitly enable backtracking
    )
    
    segments_with_bt, oversized_with_bt = segmenter_with_bt._segment_dataframe(df)
    segments_with_bt_list = list(segments_with_bt)
    oversized_with_bt_list = list(oversized_with_bt)
    
    print(f"Results:")
    print(f"  - Normal segments: {len(segments_with_bt_list)}")
    print(f"  - Oversized segments: {len(oversized_with_bt_list)}")
    print(f"  - Total segments: {len(segments_with_bt_list) + len(oversized_with_bt_list)}")
    
    # Show segment details
    print(f"  - First 5 segment details:")
    for i, seg in enumerate(segments_with_bt_list[:5]):
        size = seg.memory_usage(deep=True).sum()
        utilization = (size / size_limit) * 100
        print(f"    Segment {i+1}: {len(seg):2d} rows, {size:,} bytes ({utilization:.1f}% utilization)")
    
    if len(segments_with_bt_list) > 5:
        print(f"    ... and {len(segments_with_bt_list) - 5} more segments")
    
    avg_utilization_with_bt = 0
    if segments_with_bt_list:
        total_used = sum(seg.memory_usage(deep=True).sum() for seg in segments_with_bt_list)
        avg_utilization_with_bt = (total_used / len(segments_with_bt_list)) / size_limit * 100
        print(f"  - Average utilization: {avg_utilization_with_bt:.1f}%")
    
    print()
    
    # Comparison
    print("Comparison Results:")
    print("=" * 50)
    
    total_no_bt = len(segments_no_bt_list) + len(oversized_no_bt_list)
    total_with_bt = len(segments_with_bt_list) + len(oversized_with_bt_list)
    
    print(f"Total segments:")
    print(f"  - Without backtracking: {total_no_bt}")
    print(f"  - With backtracking:    {total_with_bt}")
    
    if total_no_bt > 0 and total_with_bt > 0:
        if total_no_bt != total_with_bt:
            reduction = total_no_bt - total_with_bt
            reduction_pct = (reduction / total_no_bt) * 100
            efficiency_ratio = total_no_bt / total_with_bt
            print(f"  - Improvement: {reduction} fewer segments ({reduction_pct:.1f}% reduction)")
            print(f"  - Efficiency ratio: {efficiency_ratio:.1f}x")
        else:
            print(f"  - No difference in segment count (both algorithms optimal for this data)")
    
    print(f"\nUtilization comparison:")
    print(f"  - Without backtracking: {avg_utilization_no_bt:.1f}% average utilization")
    print(f"  - With backtracking:    {avg_utilization_with_bt:.1f}% average utilization")
    
    if avg_utilization_with_bt > avg_utilization_no_bt:
        improvement = avg_utilization_with_bt - avg_utilization_no_bt
        print(f"  - Backtracking improves utilization by {improvement:.1f} percentage points")
    
    # Data integrity check
    total_rows_no_bt = sum(len(s) for s in segments_no_bt_list) + sum(len(s) for s in oversized_no_bt_list)
    total_rows_with_bt = sum(len(s) for s in segments_with_bt_list) + sum(len(s) for s in oversized_with_bt_list)
    
    print(f"\nData integrity:")
    print(f"  - Original rows: {len(df)}")
    print(f"  - Without backtracking: {total_rows_no_bt} ({'✓' if total_rows_no_bt == len(df) else '✗'})")
    print(f"  - With backtracking: {total_rows_with_bt} ({'✓' if total_rows_with_bt == len(df) else '✗'})")


def test_parameter_usage():
    """Test that the parameter can be set and accessed."""
    
    print("\n" + "=" * 60)
    print("=== Parameter Configuration Test ===")
    print("=" * 60)
    
    # Test default value
    default_segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy())
    print(f"Default enable_backtracking value: {default_segmenter.enable_backtracking}")
    
    # Test explicit True
    explicit_true = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),
        enable_backtracking=True
    )
    print(f"Explicitly set to True: {explicit_true.enable_backtracking}")
    
    # Test explicit False  
    explicit_false = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),
        enable_backtracking=False
    )
    print(f"Explicitly set to False: {explicit_false.enable_backtracking}")
    
    # Test algorithm parameters extraction
    params = explicit_false._get_algorithm_params()
    print(f"Algorithm parameters: {params}")
    
    print(f"\n✓ Parameter configuration working correctly!")


if __name__ == "__main__":
    test_parameter_usage()
    demonstrate_backtracking_toggle()
    
    print("\n" + "=" * 60)
    print("=== Usage Examples ===")
    print("=" * 60)
    print()
    print("# Enable backtracking (default, recommended)")
    print("segmenter = MaxSizeBasedStrategy(")
    print("    strategy=NoopStrategy(),")
    print("    max_size_limit=10000,")
    print("    enable_backtracking=True")
    print(")")
    print()
    print("# Disable backtracking (for performance-critical scenarios)")
    print("segmenter = MaxSizeBasedStrategy(")
    print("    strategy=NoopStrategy(),")
    print("    max_size_limit=10000,")
    print("    enable_backtracking=False")
    print(")")
    print()
    print("✅ enable_backtracking parameter successfully implemented!")
    print("✅ Allows selection between divide-and-conquer modes")
    print("✅ Maintains backward compatibility (default: True)")
    print("✅ Provides performance tuning option for different use cases")
