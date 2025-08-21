#!/usr/bin/env python3
"""
Test script to demonstrate PyArrow Table segmentation with divide-and-conquer and backtracking.
"""

import sys
import os

# Add src to path to import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyarrow as pa
    import pandas as pd
    from parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy
    
    def test_pyarrow_segmentation():
        print("=== PyArrow Table Segmentation Test ===")
        
        # Create a test PyArrow Table
        data = {
            'id': list(range(1000)),
            'name': [f'name_{i}' for i in range(1000)],
            'value': [i * 2.5 for i in range(1000)],
            'description': [f'A longer description for item {i} with more text to increase memory usage' for i in range(1000)]
        }
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        
        print(f"Original PyArrow Table:")
        print(f"  - Rows: {len(table)}")
        print(f"  - Size: {table.nbytes:,} bytes")
        print(f"  - Columns: {list(table.column_names)}")
        
        # Test with divide-and-conquer + backtracking (default)
        print("\n--- Test 1: Divide-and-conquer with backtracking (enabled) ---")
        segmenter_with_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=50_000,  # 50KB segments
            enable_backtracking=True
        )
        
        segments_with_backtrack = segmenter_with_backtrack.segment(table)
        print(f"Segments created (with backtracking): {len(list(segments_with_backtrack))}")
        
        total_size_with_backtrack = 0
        for i, segment in enumerate(segments_with_backtrack):
            print(f"  Segment {i+1}: {len(segment)} rows, {segment.nbytes:,} bytes")
            total_size_with_backtrack += segment.nbytes
        
        print(f"Total size: {total_size_with_backtrack:,} bytes")
        
        # Test without backtracking
        print("\n--- Test 2: Divide-and-conquer without backtracking ---")
        segmenter_no_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=50_000,  # 50KB segments
            enable_backtracking=False
        )
        
        segments_no_backtrack = segmenter_no_backtrack.segment(table)
        print(f"Segments created (no backtracking): {len(list(segments_no_backtrack))}")
        
        total_size_no_backtrack = 0
        for i, segment in enumerate(segments_no_backtrack):
            print(f"  Segment {i+1}: {len(segment)} rows, {segment.nbytes:,} bytes")
            total_size_no_backtrack += segment.nbytes
        
        print(f"Total size: {total_size_no_backtrack:,} bytes")
        
        # Comparison
        print("\n--- Comparison ---")
        segments_with_list = list(segmenter_with_backtrack.segment(table))
        segments_without_list = list(segmenter_no_backtrack.segment(table))
        
        print(f"Segments with backtracking: {len(segments_with_list)}")
        print(f"Segments without backtracking: {len(segments_without_list)}")
        
        if len(segments_with_list) <= len(segments_without_list):
            print("✅ Backtracking optimization successful: reduced or equal segment count")
            reduction = len(segments_without_list) - len(segments_with_list)
            if reduction > 0:
                print(f"   Reduction: {reduction} fewer segments")
        else:
            print("⚠️  Backtracking resulted in more segments (unexpected)")
        
        # Test with very small limit to trigger oversized segments
        print("\n--- Test 3: Testing oversized segment handling ---")
        segmenter_small = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=1000,  # Very small 1KB limit
            enable_backtracking=True
        )
        
        def oversized_handler(oversized_segments):
            print(f"Oversized handler called with {len(oversized_segments)} segments")
            for i, segment in enumerate(oversized_segments):
                print(f"  Oversized segment {i+1}: {len(segment)} rows, {segment.nbytes:,} bytes")
        
        segments_small = segmenter_small.segment(table, oversized_handler=oversized_handler)
        print(f"Normal segments with small limit: {len(list(segments_small))}")
        
        print("\n=== PyArrow segmentation test completed! ===")

    if __name__ == "__main__":
        test_pyarrow_segmentation()
        
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install pyarrow and pandas to run this test")
    sys.exit(1)
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
