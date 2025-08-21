#!/usr/bin/env python3
"""
Demo script showing the modified strategy pattern in action.

This demonstrates how the dataclass-based segmenter pattern works with
algorithm parameters embedded in the segmenter class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataclasses import dataclass
from typing import Any
from src.parquet_segmenter.segmenter.base_segmenter import (
    BaseSegmenter, 
    SegmentStrategy, 
    SegmentResult,
    ListSegmentResult,
    NoopStrategy,
    SizeBasedSegmenter,
    TimeBasedSegmenter,
    MaxSizeBasedStrategy
)


class SizeBasedStrategy(SegmentStrategy):
    """Strategy that segments based on row count parameters."""
    
    def segment(self, table: Any, **params: Any) -> SegmentResult:
        """Segment table based on max_rows_per_segment parameter."""
        max_rows = params.get('max_rows_per_segment', 1000)
        
        # Simulate segmentation based on size
        if hasattr(table, '__len__'):
            total_rows = len(table)
            segments = []
            for i in range(0, total_rows, max_rows):
                end_idx = min(i + max_rows, total_rows)
                segments.append(f"Segment {i//max_rows + 1}: rows {i}-{end_idx-1}")
            return ListSegmentResult(segments)
        else:
            return ListSegmentResult([f"Single segment (max_rows={max_rows})"])


def demo_strategy_pattern():
    """Demonstrate the modified strategy pattern."""
    print("=== Modified Strategy Pattern Demo ===\n")
    
    # Sample data
    sample_data = list(range(3500))  # 3500 rows of data
    
    # 1. Basic usage with NoopStrategy
    print("1. Basic segmenter with NoopStrategy:")
    basic_segmenter = BaseSegmenter(strategy=NoopStrategy())
    result = basic_segmenter.segment(sample_data)
    print(f"   Result: {list(result)}\n")
    
    # 2. Size-based segmenter with custom parameters
    print("2. Size-based segmenter with 1000 rows per segment:")
    size_segmenter = SizeBasedSegmenter(
        strategy=SizeBasedStrategy(),
        max_rows_per_segment=1000,
        min_rows_per_segment=100
    )
    result = size_segmenter.segment(sample_data)
    print(f"   Algorithm params: {size_segmenter._get_algorithm_params()}")
    for segment in result:
        print(f"   {segment}")
    print()
    
    # 3. Different configuration of the same segmenter
    print("3. Size-based segmenter with 500 rows per segment:")
    smaller_segmenter = SizeBasedSegmenter(
        strategy=SizeBasedStrategy(),
        max_rows_per_segment=500,
        min_rows_per_segment=50
    )
    result = smaller_segmenter.segment(sample_data)
    print(f"   Algorithm params: {smaller_segmenter._get_algorithm_params()}")
    for segment in result:
        print(f"   {segment}")
    print()
    
    # 4. Time-based segmenter (example configuration)
    print("4. Time-based segmenter configuration:")
    time_segmenter = TimeBasedSegmenter(
        strategy=NoopStrategy(),  # Using noop for demo
        time_column="created_at",
        segment_duration_hours=6,
        overlap_minutes=30
    )
    print(f"   Algorithm params: {time_segmenter._get_algorithm_params()}")
    result = time_segmenter.segment({"created_at": "2025-01-01", "data": "sample"})
    print(f"   Result: {list(result)}\n")
    
    # 5. MaxSize-based segmenter with DataFrame
    print("5. MaxSize-based segmenter with 50KB limit:")
    import pandas as pd
    large_df = pd.DataFrame({
        'id': range(10000),
        'data': [f'sample_data_{i}' * 10 for i in range(10000)],  # Larger strings
        'value': [i * 1.5 for i in range(10000)]
    })
    
    max_size_segmenter = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),  # Strategy field required
        max_size_limit=50000  # 50KB limit
    )
    print(f"   Algorithm params: {max_size_segmenter._get_algorithm_params()}")
    print(f"   Original DataFrame size: {large_df.memory_usage(deep=True).sum():,} bytes")
    
    result = max_size_segmenter.segment(large_df)
    segments = list(result)
    print(f"   Number of segments created: {len(segments)}")
    for i, segment in enumerate(segments[:3]):  # Show first 3 segments
        segment_size = segment.memory_usage(deep=True).sum()
        print(f"   Segment {i+1}: {len(segment)} rows, {segment_size:,} bytes")
    if len(segments) > 3:
        print(f"   ... and {len(segments) - 3} more segments")
    print()
    
    # 6. MaxSize-based segmenter with smaller limit
    print("6. MaxSize-based segmenter with 10KB limit:")
    smaller_limit_segmenter = MaxSizeBasedStrategy(
        strategy=NoopStrategy(),  # Strategy field required
        max_size_limit=10000  # 10KB limit
    )
    result = smaller_limit_segmenter.segment(large_df)
    segments = list(result)
    print(f"   Number of segments created: {len(segments)}")
    print(f"   Average segment size: {large_df.memory_usage(deep=True).sum() / len(segments):,.0f} bytes")
    print()
    
    print("=== Pattern Benefits ===")
    print("✓ Algorithm parameters are type-safe dataclass fields")
    print("✓ Strategy receives both data and parameters")
    print("✓ Easy to create different configurations of the same algorithm")
    print("✓ Clear separation between configuration and implementation")
    print("✓ Supports both simple and complex segmentation scenarios")
    print("✓ MaxSizeBasedSegmenter enforces hard size limits for memory management")
    print("✓ Works with different data types (pandas, PyArrow, generic Python objects)")


if __name__ == "__main__":
    demo_strategy_pattern()
