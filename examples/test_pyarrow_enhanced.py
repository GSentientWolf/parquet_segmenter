#!/usr/bin/env python3
"""
Enhanced test script for PyArrow Table segmentation to demonstrate backtracking benefits.
"""

import sys
import os

# Add src to path to import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyarrow as pa
    import pandas as pd
    from parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy
    
    def create_variable_size_table():
        """Create a PyArrow Table with variable row sizes to better test backtracking."""
        data = {
            'id': [],
            'small_field': [],
            'large_field': []
        }
        
        # Create rows with varying sizes
        for i in range(500):
            data['id'].append(i)
            if i % 3 == 0:
                # Small rows
                data['small_field'].append(f'small_{i}')
                data['large_field'].append(f'tiny')
            elif i % 3 == 1:
                # Medium rows  
                data['small_field'].append(f'medium_field_{i}')
                data['large_field'].append(f'medium_description_for_item_{i}_with_some_text')
            else:
                # Large rows
                data['small_field'].append(f'large_field_with_more_content_{i}')
                data['large_field'].append(f'very_long_description_for_item_{i}_with_lots_of_text_content_to_make_the_row_larger_and_variable_in_size')
        
        df = pd.DataFrame(data)
        return pa.Table.from_pandas(df)
    
    def test_pyarrow_backtracking():
        print("=== Enhanced PyArrow Backtracking Test ===")
        
        table = create_variable_size_table()
        
        print(f"Variable-size PyArrow Table:")
        print(f"  - Rows: {len(table)}")
        print(f"  - Size: {table.nbytes:,} bytes")
        print(f"  - Avg bytes per row: {table.nbytes / len(table):.1f}")
        
        # Test with a limit that should benefit from backtracking
        size_limit = 15_000  # 15KB
        
        print(f"\nUsing size limit: {size_limit:,} bytes")
        
        # Test with backtracking
        print("\n--- With Backtracking Enabled ---")
        segmenter_with_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=True
        )
        
        segments_with = list(segmenter_with_backtrack.segment(table))
        print(f"Segments created: {len(segments_with)}")
        
        total_size_with = 0
        for i, segment in enumerate(segments_with):
            size = segment.nbytes
            total_size_with += size
            print(f"  Segment {i+1}: {len(segment):3d} rows, {size:5,d} bytes ({size/size_limit*100:5.1f}% of limit)")
        
        # Test without backtracking
        print("\n--- Without Backtracking ---")
        segmenter_without_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=False
        )
        
        segments_without = list(segmenter_without_backtrack.segment(table))
        print(f"Segments created: {len(segments_without)}")
        
        total_size_without = 0
        for i, segment in enumerate(segments_without):
            size = segment.nbytes
            total_size_without += size
            print(f"  Segment {i+1}: {len(segment):3d} rows, {size:5,d} bytes ({size/size_limit*100:5.1f}% of limit)")
        
        # Analysis
        print("\n--- Analysis ---")
        print(f"Segments with backtracking:    {len(segments_with)}")
        print(f"Segments without backtracking: {len(segments_without)}")
        
        if len(segments_with) < len(segments_without):
            reduction = len(segments_without) - len(segments_with)
            improvement = (reduction / len(segments_without)) * 100
            print(f"✅ Backtracking reduced segments by {reduction} ({improvement:.1f}% improvement)")
        elif len(segments_with) == len(segments_without):
            print("🔄 Same number of segments (no improvement needed)")
        else:
            print("⚠️  Backtracking increased segments (unexpected)")
        
        # Check utilization
        with_utilization = sum(seg.nbytes for seg in segments_with) / (len(segments_with) * size_limit) * 100
        without_utilization = sum(seg.nbytes for seg in segments_without) / (len(segments_without) * size_limit) * 100
        
        print(f"Average utilization with backtracking:    {with_utilization:.1f}%")
        print(f"Average utilization without backtracking: {without_utilization:.1f}%")
        
        if with_utilization > without_utilization:
            print(f"✅ Backtracking improved utilization by {with_utilization - without_utilization:.1f}%")
        
        return len(segments_with), len(segments_without)

    def test_small_table_edge_case():
        """Test with a very small table that fits in one segment."""
        print("\n=== Small Table Edge Case ===")
        
        # Create a tiny table
        data = {'id': [1, 2, 3], 'name': ['a', 'b', 'c']}
        small_table = pa.Table.from_pandas(pd.DataFrame(data))
        
        print(f"Small table size: {small_table.nbytes} bytes")
        
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=10_000,  # Much larger than table
            enable_backtracking=True
        )
        
        segments = list(segmenter.segment(small_table))
        print(f"Segments created: {len(segments)}")
        assert len(segments) == 1, "Small table should create exactly 1 segment"
        print("✅ Small table handled correctly")

    if __name__ == "__main__":
        test_pyarrow_backtracking()
        test_small_table_edge_case()
        print("\n=== All PyArrow tests completed successfully! ===")
        
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
