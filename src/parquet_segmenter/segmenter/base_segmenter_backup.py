"""Modified strategy pattern for parquet segmentation.

This module implements a dataclass-based segmentation approach where the base
segmenter class contains algorithm parameters and delegates the actual 
segmentation logic to configurable strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, runtime_checkable, Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa  # type: ignore[import-untyped]
except ImportError:
    pa = None

from ..log import logger


@runtime_checkable
class SegmentResult(Protocol):
    """Protocol for segmentation results.
    
    Results should be iterable and contain the segmented data.
    """
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over the segments."""
        ...


class SegmentStrategy(ABC):
    """Abstract base class for segmentation strategies.
    
    Strategies implement the core segmentation algorithm and receive
    both the normalized table data and the algorithm parameters from
    the segmenter dataclass.
    """
    
    @abstractmethod
    def segment(self, table: Any, **kwargs: Any) -> SegmentResult:
        """Perform segmentation on the table using the given parameters.
        
        Args:
            table: The normalized table data to segment
            **params: Algorithm-specific parameters from the segmenter
            
        Returns:
            A SegmentResult containing the segmented data
        """
        ...


@dataclass
class BaseSegmenter:
    """Base dataclass for segmentation with algorithm parameters.
    
    This dataclass holds both the segmentation strategy and algorithm-specific
    parameters. Subclasses should extend this with their own parameter fields.
    """
    
    strategy: SegmentStrategy
    
    def _normalize_table(self, table: Any) -> Any:
        """Normalize input data to a consistent format.
        
        Args:
            table: Input data (pandas DataFrame, pyarrow Table, or dict-like)
            
        Returns:
            Normalized table data
        """
        # If pyarrow present and table is a pyarrow.Table, allow it through
        if pa is not None and isinstance(table, pa.Table):
            return table

        # If pandas present and table is a DataFrame, allow it through
        if pd is not None and isinstance(table, pd.DataFrame):
            return table

        # Fallback: try to convert dict/list-of-dicts to DataFrame
        if pd is not None:
            try:
                return pd.DataFrame(table)
            except (TypeError, ValueError) as exc:
                # Log conversion failure for diagnostics but continue
                try:
                    logger.warning(
                        "Failed to coerce input to DataFrame (type=%s): %s",
                        type(table).__name__,
                        exc,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    # Prevent logging errors from breaking the flow
                    pass
                
        return table

    def _get_algorithm_params(self) -> dict[str, Any]:
        """Extract algorithm parameters from this dataclass.
        
        Returns:
            Dictionary of parameter names to values, excluding 'strategy'
        """
        params = {}
        for field_name in self.__dataclass_fields__:
            if field_name != 'strategy':
                params[field_name] = getattr(self, field_name)
        return params

    def segment(self, table: Any, **kwargs: Any) -> SegmentResult:
        """Main entry point for segmentation.
        
        Args:
            table: Input data to segment
            **kwargs: Additional keyword arguments to pass to the strategy
            
        Returns:
            SegmentResult with the segmented data
        """
        normalized_table = self._normalize_table(table)
        params = self._get_algorithm_params()
        params.update(kwargs)  # Allow kwargs to override dataclass params
        return self.strategy.segment(normalized_table, **params)


# Example implementations for testing and reference

class ListSegmentResult(list):
    """Simple list-based segment result."""
    pass


class NoopStrategy(SegmentStrategy):
    """A no-op strategy that returns the input as a single segment."""
    
    def segment(self, table: Any, **params: Any) -> SegmentResult:
        """Return the table as a single segment, ignoring parameters."""
        return ListSegmentResult([table])


@dataclass 
class SizeBasedSegmenter(BaseSegmenter):
    """Example segmenter with size-based parameters."""
    
    max_rows_per_segment: int = 1000
    min_rows_per_segment: int = 100


@dataclass
class TimeBasedSegmenter(BaseSegmenter):
    """Example segmenter with time-based parameters."""
    
    time_column: str = "timestamp"
    segment_duration_hours: int = 24
    overlap_minutes: int = 0

@dataclass
class MaxSizeBasedStrategy(BaseSegmenter):
    """Strategy that segments based on maximum size constraints."""
    max_size_limit: int = 1_000_000  # Default 1MB
    enable_backtracking: bool = True  # Enable backtracking optimization by default

    def segment(self, table: Any, oversized_handler: Optional[Callable[[List[Any]], Any]] = None) -> SegmentResult:
        """Segment table based on max_size_limit parameter.
        
        Args:
            table: The table data to segment
            oversized_handler: Optional callable to handle oversized segments.
                              If provided, will be called with each oversized segment.
                              If not provided, oversized segments are logged and discarded.
            
        Returns:
            SegmentResult with segments respecting the size limit
        """
        
        # Handle different table types
        if pd is not None and isinstance(table, pd.DataFrame):
            try:
                segments, oversized = self._segment_dataframe(table)
                
                if oversized and oversized_handler is not None:
                    logger.info("Processing %d oversized segments with provided handler", len(list(oversized)))
                    try:
                        oversized_handler(list(oversized))
                    except Exception as e:
                        logger.error("Error in oversized_handler: %s", e)
                else:
                    logger.warning("Found %d oversized segments that exceed size limit", len(list(oversized)))
            
                # Only return segments that fit within the limit
                return segments
            except ValueError as e:
                logger.error("Error segmenting DataFrame: %s", e)
                return ListSegmentResult([])
        elif pa is not None and isinstance(table, pa.Table):
            try:
                result = self._segment_pyarrow_table(table)
                return result
            except ValueError as e:
                logger.error("Error segmenting PyArrow Table: %s", e)
                return ListSegmentResult([])
        else:
            # Fallback for other data types
            return self._segment_generic(table)


    def _segment_dataframe(self, df: Any) -> Tuple[SegmentResult, SegmentResult]:
        """Segment a pandas DataFrame using iterative divide-and-conquer with optional backtracking.
        
        Uses a deque-based algorithm to partition the DataFrame into segments where
        each segment is guaranteed to be < max_size_limit bytes. The algorithm 
        recursively divides DataFrame chunks that exceed the size limit. If 
        enable_backtracking is True, applies backtracking to merge adjacent segments 
        when possible for optimal segmentation.
        
        Args:
            df: The pandas DataFrame to segment
            
        Returns:
            Tuple of (normal_segments, oversized_segments) as SegmentResult objects
        """
        if len(df) == 0:
            raise ValueError("Cannot segment an empty DataFrame")
        
        segments = []
        over_sized_segments = []
        # Initialize deque with the full DataFrame range (start_idx, end_idx)
        work_queue = deque([(0, len(df))])
        
        # Phase 1: Divide and conquer to create initial segments
        while work_queue:
            start_idx, end_idx = work_queue.popleft()
            
            # Extract the current chunk
            chunk = df.iloc[start_idx:end_idx]
            chunk_size = chunk.memory_usage(deep=True).sum()
            
            if chunk_size <= self.max_size_limit:
                # Chunk fits within limit 
                if self.enable_backtracking:
                    # Store as (start_idx, end_idx, chunk, size) for backtracking
                    segments.append((start_idx, end_idx, chunk.copy(), chunk_size))
                else:
                    # Store just the chunk for non-backtracking mode
                    segments.append(chunk.copy())
            else:
                # Chunk is too large - divide and conquer
                if end_idx - start_idx <= 1:
                    # Cannot divide further (single row exceeds limit)
                    over_sized_segments.append(chunk.copy())
                    logger.warning(
                        "Single row DataFrame segment exceeds size limit: %d bytes > %d bytes",
                        chunk_size, self.max_size_limit
                    )
                else:
                    # Split the chunk in half and add both halves back to queue
                    mid_idx = start_idx + (end_idx - start_idx) // 2
                    
                    # Add the two halves to the front of deque for depth-first processing
                    # This tends to create more balanced segments
                    work_queue.appendleft((mid_idx, end_idx))  # Second half
                    work_queue.appendleft((start_idx, mid_idx))  # First half
        
        # Phase 2: Backtracking to merge adjacent segments when possible (if enabled)
        if self.enable_backtracking and segments:
            segments = self._apply_backtracking(df, segments)
            # Extract just the DataFrames from the segment tuples
            final_segments = [seg[2] if isinstance(seg, tuple) else seg for seg in segments]
        else:
            # No backtracking - segments are already DataFrames or need extraction
            final_segments = [seg[2] if isinstance(seg, tuple) else seg for seg in segments]
        
        return ListSegmentResult(final_segments), ListSegmentResult(over_sized_segments)
    
    def _apply_backtracking(self, df: Any, segments: list) -> list:
        """Apply backtracking to merge adjacent segments when possible.
        
        Args:
            df: The original DataFrame
            segments: List of (start_idx, end_idx, chunk, size) tuples
            
        Returns:
            List of optimized segments (either tuples or DataFrames)
        """
        if len(segments) <= 1:
            return segments
        
        # Sort segments by start index to ensure proper ordering
        segments.sort(key=lambda x: x[0])
        
        optimized_segments = []
        i = 0
        
        while i < len(segments):
            current_start, current_end, current_chunk, current_size = segments[i]
            
            # Try to merge with the next segment
            merged = False
            if i + 1 < len(segments):
                next_start, next_end, next_chunk, next_size = segments[i + 1]
                
                # Check if segments are adjacent
                if current_end == next_start:
                    # Calculate combined size by creating the merged segment
                    merged_chunk = df.iloc[current_start:next_end]
                    merged_size = merged_chunk.memory_usage(deep=True).sum()
                    
                    if merged_size <= self.max_size_limit:
                        # Merge successful - create new merged segment
                        optimized_segments.append((current_start, next_end, merged_chunk.copy(), merged_size))
                        merged = True
                        i += 2  # Skip the next segment as it's been merged
                        
                        # Log the merge for debugging
                        logger.debug(
                            "Backtracking: merged segments [%d:%d] + [%d:%d] = [%d:%d] (%d bytes)",
                            current_start, current_end, next_start, next_end,
                            current_start, next_end, merged_size
                        )
            
            if not merged:
                # Cannot merge, keep the current segment as-is
                optimized_segments.append((current_start, current_end, current_chunk, current_size))
                i += 1
        
        # Check if we can do another pass of merging (recursive backtracking)
        if len(optimized_segments) < len(segments):
            # We made progress, try another round of merging
            return self._apply_backtracking(df, optimized_segments)
        else:
            # No more merging possible
            return optimized_segments
    
    def _segment_pyarrow_table(self, table: Any) -> SegmentResult:
        """Segment a PyArrow Table based on size limit."""
        segments = []
        total_bytes = table.nbytes
        total_rows = len(table)
        
        if total_bytes <= self.max_size_limit:
            # Table is already within size limit
            segments.append(table)
        else:
            # Calculate approximate rows per segment
            bytes_per_row = total_bytes / total_rows if total_rows > 0 else 1
            rows_per_segment = max(1, int(self.max_size_limit / bytes_per_row))

            start_idx = 0
            while start_idx < total_rows:
                end_idx = min(start_idx + rows_per_segment, total_rows)
                segment = table.slice(start_idx, end_idx - start_idx)
                segments.append(segment)
                start_idx = end_idx
        
        return ListSegmentResult(segments)
    
    def _segment_generic(self, data: Any) -> SegmentResult:
        """Segment generic data based on estimated size."""
        import sys
        
        try:
            data_size = sys.getsizeof(data)
            if hasattr(data, '__len__'):
                # For sequences, try to segment by elements
                total_items = len(data)
                if data_size <= self.max_size_limit:
                    return ListSegmentResult([data])
                
                # Estimate items per segment
                bytes_per_item = data_size / total_items if total_items > 0 else 1
                items_per_segment = max(1, int(self.max_size_limit / bytes_per_item))
                
                segments = []
                for i in range(0, total_items, items_per_segment):
                    end_idx = min(i + items_per_segment, total_items)
                    segment = data[i:end_idx]
                    segments.append(segment)
                
                return ListSegmentResult(segments)
            else:
                # Single item - return as is if within limit, otherwise split representation
                return ListSegmentResult([data])
                
        except Exception:  # pylint: disable=broad-exception-caught
            # Fallback: return as single segment
            return ListSegmentResult([data])
