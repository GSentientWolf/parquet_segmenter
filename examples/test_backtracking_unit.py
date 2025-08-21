#!/usr/bin/env python3
"""
Unit tests for the enable_backtracking parameter functionality.
"""

import unittest
import pandas as pd
from src.parquet_segmenter.segmenter.base_segmenter import MaxSizeBasedStrategy, NoopStrategy


class TestBacktrackingParameter(unittest.TestCase):
    """Test cases for the enable_backtracking parameter."""

    def test_default_backtracking_enabled(self):
        """Test that backtracking is enabled by default."""
        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy())
        self.assertTrue(segmenter.enable_backtracking)

    def test_explicit_backtracking_enabled(self):
        """Test explicitly enabling backtracking."""
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            enable_backtracking=True
        )
        self.assertTrue(segmenter.enable_backtracking)

    def test_explicit_backtracking_disabled(self):
        """Test explicitly disabling backtracking."""
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            enable_backtracking=False
        )
        self.assertFalse(segmenter.enable_backtracking)

    def test_algorithm_params_includes_backtracking(self):
        """Test that algorithm parameters include enable_backtracking."""
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=50000,
            enable_backtracking=False
        )
        params = segmenter._get_algorithm_params()
        expected_params = {
            'max_size_limit': 50000,
            'enable_backtracking': False
        }
        self.assertEqual(params, expected_params)

    def test_backtracking_modes_produce_valid_results(self):
        """Test that both backtracking modes produce valid segmentation results."""
        # Create test DataFrame
        test_data = {
            'id': range(50),
            'content': [f'data_{i}_{"x" * (100 + i * 10)}' for i in range(50)]
        }
        df = pd.DataFrame(test_data)
        
        size_limit = 3000
        
        # Test with backtracking disabled
        segmenter_no_bt = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=False
        )
        
        segments_no_bt, oversized_no_bt = segmenter_no_bt._segment_dataframe(df)
        segments_no_bt_list = list(segments_no_bt)
        oversized_no_bt_list = list(oversized_no_bt)
        
        # Test with backtracking enabled
        segmenter_with_bt = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=size_limit,
            enable_backtracking=True
        )
        
        segments_with_bt, oversized_with_bt = segmenter_with_bt._segment_dataframe(df)
        segments_with_bt_list = list(segments_with_bt)
        oversized_with_bt_list = list(oversized_with_bt)
        
        # Both should preserve all data
        total_rows_no_bt = sum(len(s) for s in segments_no_bt_list) + sum(len(s) for s in oversized_no_bt_list)
        total_rows_with_bt = sum(len(s) for s in segments_with_bt_list) + sum(len(s) for s in oversized_with_bt_list)
        
        self.assertEqual(total_rows_no_bt, len(df))
        self.assertEqual(total_rows_with_bt, len(df))
        
        # Both should create at least one segment
        self.assertGreater(len(segments_no_bt_list) + len(oversized_no_bt_list), 0)
        self.assertGreater(len(segments_with_bt_list) + len(oversized_with_bt_list), 0)
        
        # All normal segments should respect size limit
        for segment in segments_no_bt_list:
            size = segment.memory_usage(deep=True).sum()
            self.assertLessEqual(size, size_limit)
            
        for segment in segments_with_bt_list:
            size = segment.memory_usage(deep=True).sum()
            self.assertLessEqual(size, size_limit)

    def test_backtracking_parameter_in_segment_method(self):
        """Test that the enable_backtracking parameter is used in the segment method."""
        # Create simple test data
        df = pd.DataFrame({
            'id': range(20),
            'data': ['test_data'] * 20
        })
        
        segmenter = MaxSizeBasedStrategy(
            strategy=NoopStrategy(),
            max_size_limit=1000,
            enable_backtracking=True
        )
        
        # This should not raise an exception and should return results
        result = segmenter.segment(df)
        segments = list(result)
        
        # Should have at least one segment
        self.assertGreater(len(segments), 0)
        
        # Total rows should be preserved
        total_rows = sum(len(s) for s in segments)
        self.assertEqual(total_rows, len(df))

    def test_backtracking_dataclass_field(self):
        """Test that enable_backtracking is properly defined as a dataclass field."""
        segmenter = MaxSizeBasedStrategy(strategy=NoopStrategy())
        
        # Should have the field in __dataclass_fields__
        self.assertIn('enable_backtracking', segmenter.__dataclass_fields__)
        
        # Should be accessible as an attribute
        self.assertTrue(hasattr(segmenter, 'enable_backtracking'))
        
        # Should be modifiable
        segmenter.enable_backtracking = False
        self.assertFalse(segmenter.enable_backtracking)


if __name__ == '__main__':
    unittest.main()
