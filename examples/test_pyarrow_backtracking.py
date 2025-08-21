#!/usr/bin/env python3
"""
Specific test to demonstrate PyArrow backtracking benefits with uneven segment sizes.
"""

import sys
import os

# Add src to path to import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyarrow as pa
    import pandas as pd
    from parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy
    
    def create_uneven_table():
        """Create a table where divide-and-conquer will create uneven segments."""
        data = {
            'id': [],
            'content': []
        }
        
        # Create data with a specific pattern that will result in uneven segments
        # First quarter: very small rows
        for i in range(100):
            data['id'].append(i)
            data['content'].append('small')
        
        # Second quarter: large rows  
        for i in range(100, 200):
            data['id'].append(i)
            data['content'].append('large_content_that_takes_up_more_space_per_row_' * 5)
        
        # Third quarter: small rows again
        for i in range(200, 300):
            data['id'].append(i)
            data['content'].append('small')
        
        # Fourth quarter: large rows again
        for i in range(300, 400):
            data['id'].append(i)
            data['content'].append('large_content_that_takes_up_more_space_per_row_' * 5)
        
        df = pd.DataFrame(data)
        return pa.Table.from_pandas(df)
    
    def test_backtracking_benefits():
        print("=== PyArrow Backtracking Benefits Test ===")
        
        table = create_uneven_table()
        print(f"Uneven PyArrow Table:")
        print(f"  - Rows: {len(table)}")
        print(f"  - Size: {table.nbytes:,} bytes")
        
        # Use a limit that will create uneven initial segments
        size_limit = 25_000  # 25KB
        
        print(f"\nUsing size limit: {size_limit:,} bytes")
        
        # Test without backtracking first
        print("\n--- Without Backtracking ---")
        segmenter_no_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=False
        )
        
        segments_without = list(segmenter_no_backtrack.segment(table))
        print(f"Segments created: {len(segments_without)}")
        
        for i, segment in enumerate(segments_without):
            size = segment.nbytes
            print(f"  Segment {i+1}: {len(segment):3d} rows, {size:6,d} bytes ({size/size_limit*100:5.1f}% of limit)")
        
        # Test with backtracking
        print("\n--- With Backtracking Enabled ---")
        segmenter_with_backtrack = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=True
        )
        
        segments_with = list(segmenter_with_backtrack.segment(table))
        print(f"Segments created: {len(segments_with)}")
        
        for i, segment in enumerate(segments_with):
            size = segment.nbytes
            print(f"  Segment {i+1}: {len(segment):3d} rows, {size:6,d} bytes ({size/size_limit*100:5.1f}% of limit)")
        
        # Analysis
        print("\n--- Detailed Analysis ---")
        print(f"Segments without backtracking: {len(segments_without)}")
        print(f"Segments with backtracking:    {len(segments_with)}")
        
        if len(segments_with) < len(segments_without):
            reduction = len(segments_without) - len(segments_with)
            improvement = (reduction / len(segments_without)) * 100
            print(f"✅ Backtracking reduced segments by {reduction} ({improvement:.1f}% improvement)")
        elif len(segments_with) == len(segments_without):
            print("🔄 Same number of segments")
        else:
            print("❌ Backtracking increased segments")
        
        # Calculate utilization efficiency
        without_utilization = [seg.nbytes / size_limit * 100 for seg in segments_without]
        with_utilization = [seg.nbytes / size_limit * 100 for seg in segments_with]
        
        print(f"\nUtilization without backtracking:")
        print(f"  Average: {sum(without_utilization)/len(without_utilization):.1f}%")
        print(f"  Min: {min(without_utilization):.1f}%, Max: {max(without_utilization):.1f}%")
        
        print(f"\nUtilization with backtracking:")
        print(f"  Average: {sum(with_utilization)/len(with_utilization):.1f}%")
        print(f"  Min: {min(with_utilization):.1f}%, Max: {max(with_utilization):.1f}%")
        
        # Show improvement
        avg_improvement = (sum(with_utilization)/len(with_utilization)) - (sum(without_utilization)/len(without_utilization))
        if avg_improvement > 0:
            print(f"✅ Average utilization improved by {avg_improvement:.1f}%")
        
        return len(segments_with), len(segments_without)

    if __name__ == "__main__":
        test_backtracking_benefits()
        print("\n=== Backtracking benefits test completed! ===")
        
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
