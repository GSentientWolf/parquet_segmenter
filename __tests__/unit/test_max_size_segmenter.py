"""Tests for MaxSizeBasedSegmenter functionality."""

import pytest
import pandas as pd
import sys
from unittest.mock import patch

from src.parquet_segmenter.segmenter.base_segmenter import (
    MaxSizeBasedStrategy,
    NoopStrategy,
    ListSegmentResult,
)


class TestMaxSizeBasedStrategy:
    """Test suite for MaxSizeBasedStrategy."""

    def test_init_with_default_limit(self):
        """Test initialization with default size limit."""
        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy())
        assert segmenter.max_size_limit == 1000000  # Default 1MB
        assert isinstance(segmenter.strategy, NoopStrategy)

    def test_init_with_custom_limit(self):
        """Test initialization with custom size limit."""
        custom_limit = 50000
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=custom_limit,
        )
        assert segmenter.max_size_limit == custom_limit

    def test_get_algorithm_params(self):
        """Test that algorithm parameters are extracted correctly."""
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=100000,
        )
        params = segmenter._get_algorithm_params()
        assert params == {"max_size_limit": 100000, "enable_backtracking": True}

    def test_segment_small_dataframe(self):
        """Test segmentation of a DataFrame that fits within the limit."""
        small_df = pd.DataFrame({"id": range(10), "value": range(10)})

        # Use a large limit so the entire DataFrame fits in one segment
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(), max_size_limit=100000  # 100KB
        )

        segments_iter, oversized_iter = segmenter._segment_dataframe(small_df)
        segments = list(segments_iter)
        oversized = list(oversized_iter)

        assert len(segments) == 1
        assert len(oversized) == 0  # No oversized segments
        assert len(segments[0]) == 10  # All rows in one segment
        assert list(segments[0]["id"]) == list(range(10))

    def test_segment_large_dataframe(self):
        """Test segmentation of a DataFrame that exceeds the limit."""
        # Create a larger DataFrame
        large_df = pd.DataFrame(
            {
                "id": range(1000),
                "data": [f"sample_data_{i}" * 20 for i in range(1000)],  # Larger strings
                "value": [i * 1.5 for i in range(1000)],
            }
        )

        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(), max_size_limit=10000  # 10KB
        )

        segments_iter, oversized_iter = segmenter._segment_dataframe(large_df)
        segments = list(segments_iter)
        oversized = list(oversized_iter)

        # Should create multiple segments
        assert len(segments) > 1
        assert len(oversized) == 0  # Should not have oversized segments for this data

        # Each segment should respect the size limit
        for segment in segments:
            segment_size = segment.memory_usage(deep=True).sum()
            assert segment_size <= segmenter.max_size_limit

    def test_segment_empty_dataframe(self):
        """Test segmentation of an empty DataFrame."""
        empty_df = pd.DataFrame()

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=1000)

        # Should raise ValueError for empty DataFrame
        with pytest.raises(ValueError, match="Cannot segment an empty DataFrame"):
            segmenter._segment_dataframe(empty_df)

    def test_segment_oversized_single_row(self):
        """Test segmentation when single rows exceed the size limit."""
        # Create a DataFrame with very large single rows
        large_row_df = pd.DataFrame({"huge_data": ["x" * 10000, "y" * 10000]})

        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(), max_size_limit=100  # Very small limit
        )

        segments_iter, oversized_iter = segmenter._segment_dataframe(large_row_df)
        segments = list(segments_iter)
        oversized = list(oversized_iter)

        # Should have no normal segments but 2 oversized segments
        assert len(segments) == 0
        assert len(oversized) == 2
        assert len(oversized[0]) == 1  # Each oversized segment has 1 row
        assert len(oversized[1]) == 1

    def test_segment_generic_data(self):
        """Test segmentation of generic Python data."""
        data = list(range(1000))

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=1000)  # Small limit

        result = segmenter.segment(data)  # Use the main segment method
        segments = list(result)

        # Should create multiple segments for large data
        assert len(segments) > 1

        # Verify all original data is preserved
        total_items = sum(len(segment) for segment in segments)
        assert total_items == len(data)

    def test_segment_single_item(self):
        """Test segmentation of a single item."""
        data = {"key": "value"}

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=100)

        result = segmenter.segment(data)  # Use the main segment method
        segments = list(result)

        assert len(segments) == 1
        assert segments[0] == data

    def test_strategy_error_handling(self):
        """Test that strategy handles errors gracefully."""
        # Test with an object that might cause sys.getsizeof to fail
        problematic_data = type("BadObject", (), {})()

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=1000)

        # Should not raise an exception
        result = segmenter.segment(problematic_data)  # Use the main segment method
        segments = list(result)

        # Should return at least one segment
        assert len(segments) >= 1

    @pytest.mark.parametrize("limit", [1000, 10000, 100000])
    def test_different_size_limits(self, limit):
        """Test segmentation with different size limits."""
        df = pd.DataFrame({"id": range(500), "data": [f"test_{i}" * 10 for i in range(500)]})

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=limit)

        segments_iter, oversized_iter = segmenter._segment_dataframe(df)
        segments = list(segments_iter)
        oversized = list(oversized_iter)
        all_segments = segments + oversized

        # Smaller limits should create more segments
        assert len(all_segments) >= 1

        # Verify total row count is preserved
        total_rows = sum(len(segment) for segment in all_segments)
        assert total_rows == len(df)

    def test_dataframe_memory_estimation(self):
        """Test that memory estimation works reasonably for DataFrames."""
        # Create DataFrame with known characteristics
        df = pd.DataFrame({
            "small_int": range(100),
            "large_string": ["x" * 100 for _ in range(100)],
        })

        original_size = df.memory_usage(deep=True).sum()

        # Use a limit that should split the DataFrame
        limit = max(1, original_size // 3)

        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy(), max_size_limit=limit)

        segments_iter, oversized_iter = segmenter._segment_dataframe(df)
        segments = list(segments_iter)
        oversized = list(oversized_iter)
        all_segments = segments + oversized

        # Should create multiple segments
        assert len(all_segments) > 1

        # Each segment should be reasonably sized
        for segment in all_segments:
            segment_size = segment.memory_usage(deep=True).sum()
            # Allow reasonable tolerance for estimation errors
            assert segment_size <= limit * 2

