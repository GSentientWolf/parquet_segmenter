from __future__ import annotations

import math
import random
from enum import IntEnum, StrEnum
from typing import Callable, Iterable, List, Optional, Union
from dataclasses import dataclass

import logging
import pandas as pd
from parquet_segmenter.index_generators.strategies import stdlib_choice_factory


class OutlierStrategy(StrEnum):
    """Strategy for placing large outlier blobs in the DataFrame."""
    SPREAD = "spread"           # Distribute across different batches
    SINGLE_BATCH = "single"     # All outliers in one batch, spread within batch  
    CONTIGUOUS = "contiguous"   # All outliers in one batch, contiguous block
    MULTI_CLUSTER = "multi"     # Multiple clusters across batches


@dataclass
class PlacementConfig:
    """Configuration for outlier placement strategy."""
    strategy: OutlierStrategy = OutlierStrategy.SPREAD
    cluster_count: int = 1  # Used with MULTI_CLUSTER strategy
    
    def __post_init__(self):
        if self.strategy == OutlierStrategy.MULTI_CLUSTER and self.cluster_count < 2:
            raise ValueError("cluster_count must be >= 2 for MULTI_CLUSTER strategy")


def _validate_placement_config(config: Union[OutlierStrategy, PlacementConfig]) -> PlacementConfig:
    """Validate and normalize placement configuration."""
    if isinstance(config, OutlierStrategy):
        return PlacementConfig(strategy=config)
    elif isinstance(config, PlacementConfig):
        return config
    else:
        raise ValueError(f"Invalid placement config type: {type(config)}")


# Legacy validation function - can be removed after deprecation period



class BlobSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class ChunkSize(IntEnum):
    ONE_KB = 1024
    ONE_MB = 1024 * 1024
    ONE_GB = 1024 * 1024 * 1024


def bytes_from_human(s: str) -> int:
    """Simple helper: '5 kB' -> 5120 (uses 1024 base units).

    Accepts suffixes 'k'/'kb' and 'm'/'mb' (case-insensitive).
    """
    s = s.strip().lower().replace("kb", "k").replace("mb", "m")
    if s.endswith("k"):
        return int(float(s[:-1]) * ChunkSize.ONE_KB)
    if s.endswith("m"):
        return int(float(s[:-1]) * ChunkSize.ONE_MB)
    return int(float(s))


def build_blob_dataframe(
    *,
    total_df_size: int,
    small_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16])),
    intermediate_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB, [256, 307, 409, 470, 512])),
    outlier_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB, [820, 921, 1044, 1228])),
    batch_size: int = ChunkSize.ONE_MB,
    outlier_mb_rate: int = 10,
    top_outliers: int = 1,
    outlier_strategy: OutlierStrategy = OutlierStrategy.SPREAD,
    cluster_count: int = 1,
    seed: Optional[int] = None,
    binary_factory: Optional[Callable[[int], bytes]] = None,
    # Legacy parameters - deprecated, will be removed in next version
    spread_top_outliers: Optional[bool] = None,
    cluster_batches: Optional[int] = None,
    contiguous_within_batch: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with a 'size' column (bytes) and optional 'blob' column.

    Args:
        outlier_strategy: How to place large outlier blobs. Options:
            - SPREAD: Distribute across different batches (default)
            - SINGLE_BATCH: All outliers in one batch, spread within batch
            - CONTIGUOUS: All outliers in one batch, contiguous block  
            - MULTI_CLUSTER: Multiple clusters across batches (uses cluster_count)
        cluster_count: Number of clusters for MULTI_CLUSTER strategy (default: 1)
        
    Behavior summary:
    - Use small_bins as the common small sizes (uniform-ish).
    - Place 'intermediate' sizes sparsely (some contiguous group + some random).
    - Place outliers according to outlier_strategy: ~1 per outlier_mb_rate MB plus `top_outliers`.
    - The function guarantees at least one outlier when total_bytes >= outlier_mb_rate*1_000_000
      and always includes `top_outliers` (subject to available rows).
    - `binary_factory(size) -> bytes` can be provided to construct actual blob bytes;
      otherwise blob column is omitted (only sizes and flags).
      
    Legacy parameters (deprecated): spread_top_outliers, cluster_batches, contiguous_within_batch
    """
    rnd = random.Random(seed)

    # Handle backwards compatibility and convert legacy parameters to new enum
    if any(param is not None for param in [spread_top_outliers, cluster_batches, contiguous_within_batch]):
        import warnings
        warnings.warn(
            "Parameters 'spread_top_outliers', 'cluster_batches', and 'contiguous_within_batch' "
            "are deprecated. Use 'outlier_strategy' parameter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert legacy parameters to new enum
        if spread_top_outliers is True:
            outlier_strategy = OutlierStrategy.SPREAD
        elif spread_top_outliers is False:
            if contiguous_within_batch is True:
                outlier_strategy = OutlierStrategy.CONTIGUOUS
            elif cluster_batches is not None and cluster_batches > 1:
                outlier_strategy = OutlierStrategy.MULTI_CLUSTER
                cluster_count = cluster_batches
            else:
                outlier_strategy = OutlierStrategy.SINGLE_BATCH

    # Validate outlier strategy and cluster count
    if outlier_strategy == OutlierStrategy.MULTI_CLUSTER and cluster_count < 2:
        raise ValueError("cluster_count must be >= 2 for MULTI_CLUSTER strategy")

    small_bins = list(small_bins)
    intermediate_bins = list(intermediate_bins)
    outlier_bins = list(outlier_bins)
    
    # Convert to sets for O(1) membership testing
    small_bins_set = set(small_bins)
    intermediate_bins_set = set(intermediate_bins)
    outlier_bins_set = set(outlier_bins)

    avg_small = int(sum(small_bins) / len(small_bins))

    rows_per_mb = max(1, batch_size // max(1, avg_small))
    total_mb = total_df_size / float(batch_size)
    est_rows = max(1, int(math.ceil(rows_per_mb * total_mb)))

    # Determine how many batches (each batch contains `batch_rows` rows)
    batch_rows = rows_per_mb
    num_batches = int(math.ceil(est_rows / float(batch_rows)))

    # Build a batch template (pd.Series) of length batch_size // rows_per_mb + 1 filled with small sizes.
    # We'll copy this template across batches and then selectively override
    # positions for intermediate and outlier cases.
    small_factory = stdlib_choice_factory(num_bins=len(small_bins), replace=True, seed=seed)
    it_small = small_factory()
    batch_template: List[int] = [small_bins[next(it_small) % len(small_bins)] for _ in range(batch_rows)]

    # Second pass: insert intermediate bins into the batch template
    # Choose a contiguous block in the template and a few isolated positions.
    block_len = max(1, batch_rows // 10)
    block_start = rnd.randrange(0, max(1, batch_rows - block_len))
    for j in range(block_start, block_start + block_len):
        batch_template[j] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    num_isolated = max(1, batch_rows // 1000)
    iso_indices = rnd.sample(range(batch_rows), k=min(batch_rows, num_isolated))
    for idx in iso_indices:
        batch_template[idx] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    # Prepare full sizes list by pre-allocating with known length
    total_size = num_batches * batch_rows
    sizes: List[int] = [0] * total_size
    
    # Fill with batch template pattern
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_rows
        end_idx = start_idx + batch_rows
        sizes[start_idx:end_idx] = batch_template

    # Trim to the estimated number of rows (est_rows) if we overshot
    if len(sizes) > est_rows:
        sizes = sizes[:est_rows]

    # Determine number of outliers to add (sparse)
    # e.g., roughly 1 outlier per outlier_mb_rate MB
    # Determine number of outliers to add (sparse): roughly 1 per outlier_mb_rate MB
    num_outliers = max(1, int(math.floor(total_df_size / (outlier_mb_rate * ChunkSize.ONE_MB)))) if total_df_size >= (outlier_mb_rate * ChunkSize.ONE_MB // 2) else 0
    # ensure at least the requested top_outliers
    num_outliers = max(num_outliers, top_outliers)
    # cap to available estimated rows
    num_outliers = min(num_outliers, est_rows)

    # Choose distinct batch/position pairs for outliers. We'll ensure some
    # top_outliers are placed near the start (batch 0) and the rest are
    # distributed across random batches.
    # Generate positions only for actual available indices in the trimmed sizes array
    available_positions = [
        (b, i) 
        for b in range(num_batches) 
        for i in range(batch_rows)
        if (b * batch_rows + i) < len(sizes)
    ]
    rnd.shuffle(available_positions)
    # Ensure top_outliers occupy early positions in a randomly-chosen batch
    # (seeded for determinism). Pick positions only if their global index
    # Place outliers using new strategy-based approach
    chosen_positions: List[tuple[int, int]] = []
    
    if top_outliers > 0 and num_batches > 0:
        if outlier_strategy == OutlierStrategy.SPREAD:
            # Spread top_outliers across distinct batches when possible.
            pick_batches = (
                rnd.sample(range(num_batches), k=min(num_batches, top_outliers))
                if num_batches >= top_outliers
                else [rnd.randrange(num_batches) for _ in range(top_outliers)]
            )
            # one position per batch
            for b in pick_batches:
                valid_positions = [
                    (b, i)
                    for i in range(batch_rows)
                    if (b * batch_rows + i) < len(sizes)
                ]
                if not valid_positions:
                    continue
                rnd.shuffle(valid_positions)
                chosen_positions.append(valid_positions[0])
                
        elif outlier_strategy in [OutlierStrategy.SINGLE_BATCH, OutlierStrategy.CONTIGUOUS]:
            # Single batch clustering
            if num_batches >= 1:
                target_batch = rnd.randrange(num_batches)
            else:
                target_batch = 0
                
            valid_positions = [
                i for i in range(batch_rows)
                if (target_batch * batch_rows + i) < len(sizes)
            ]
            
            if valid_positions and outlier_strategy == OutlierStrategy.CONTIGUOUS and top_outliers <= len(valid_positions):
                # Try to place contiguously within the batch
                start_candidates = [i for i in valid_positions]
                rnd.shuffle(start_candidates)
                placed = False
                for start in start_candidates:
                    end = start + top_outliers
                    if end > batch_rows:
                        continue
                    block = list(range(start, end))
                    # ensure block maps inside sizes
                    if any((target_batch * batch_rows + idx) >= len(sizes) for idx in block):
                        continue
                    for pos in block:
                        chosen_positions.append((target_batch, pos))
                    placed = True
                    break
                
                if not placed:
                    # Fallback to non-contiguous placement
                    rnd.shuffle(valid_positions)
                    for pos in valid_positions[:top_outliers]:
                        chosen_positions.append((target_batch, pos))
            else:
                # Single batch, spread within batch
                rnd.shuffle(valid_positions)
                for pos in valid_positions[:top_outliers]:
                    chosen_positions.append((target_batch, pos))
                    
        elif outlier_strategy == OutlierStrategy.MULTI_CLUSTER:
            # Multi-cluster: create `cluster_count` clusters
            k = cluster_count
            # choose k distinct batches if possible
            if num_batches >= k:
                batches = rnd.sample(range(num_batches), k=k)
            else:
                batches = [rnd.randrange(num_batches) for _ in range(k)]

            # distribute top_outliers across clusters roughly evenly
            per_cluster = [0] * len(batches)
            for i in range(top_outliers):
                per_cluster[i % len(batches)] += 1

            for cluster_idx, b in enumerate(batches):
                want = per_cluster[cluster_idx]
                if want <= 0:
                    continue
                valid_positions = [
                    i
                    for i in range(batch_rows)
                    if (b * batch_rows + i) < len(sizes)
                ]
                if not valid_positions:
                    continue

                # spread within the target batch
                rnd.shuffle(valid_positions)
                for pos in valid_positions[:want]:
                    chosen_positions.append((b, pos))

    # Fill remaining outlier slots from shuffled available positions, skipping duplicates
    # For strategies that require specific placement, respect the batch constraints
    for (b, i) in available_positions:
        if len(chosen_positions) >= num_outliers:
            break
        if (b, i) in chosen_positions:
            continue
            
        # For SINGLE_BATCH and CONTIGUOUS strategies, only add from the target batch
        if outlier_strategy in [OutlierStrategy.SINGLE_BATCH, OutlierStrategy.CONTIGUOUS]:
            # Find the target batch from already chosen positions
            if chosen_positions:
                target_batch = chosen_positions[0][0]  # First position's batch
                if b != target_batch:
                    continue  # Skip positions not in target batch
                    
        chosen_positions.append((b, i))

    # Apply outliers into sizes (calculate global index) and record their indices
    outlier_global_indices: set[int] = set()
    for idx, (b, pos) in enumerate(chosen_positions[:num_outliers]):
        global_idx = b * batch_rows + pos
        if global_idx >= len(sizes):
            continue
        sizes[global_idx] = outlier_bins[idx % len(outlier_bins)]
        outlier_global_indices.add(global_idx)

    # Add a contiguous intermediate block somewhere in the full sequence.
    # Do not overwrite already-placed outliers (guarantee top_outliers remain).
    block_len = max(1, rows_per_mb // 10)
    # Use len(sizes) instead of est_rows for accurate bounds after trimming
    block_start = rnd.randrange(0, max(1, len(sizes) - block_len)) if len(sizes) > block_len else 0
    for j in range(block_start, min(block_start + block_len, len(sizes))):
        if j in outlier_global_indices:
            # keep outliers intact
            continue
        sizes[j] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    # random few isolated intermediate indices across the full sizes list
    # Use len(sizes) instead of est_rows for accurate bounds after trimming
    num_isolated = max(1, len(sizes) // 1000)
    iso_indices = rnd.sample(range(len(sizes)), k=min(len(sizes), num_isolated))
    for idx in iso_indices:
        if idx in outlier_global_indices:
            continue
        sizes[idx] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    # Build DataFrame
    df = pd.DataFrame({"size": sizes})
    df["is_outlier"] = df["size"].isin(outlier_bins_set)
    df["is_intermediate"] = df["size"].isin(intermediate_bins_set)

    # Optionally materialize binary blobs using binary_factory
    if binary_factory is not None:
        df["blob"] = df["size"].apply(binary_factory)

    return df


def visualize_batch_layout(df: pd.DataFrame, batch_rows: int) -> str:
    """Return a compact textual visualization of batches.

    Marks: '.' small, 'i' intermediate, 'O' outlier.
    """
    symbols = []
    for i in range(0, len(df), batch_rows):
        chunk = df["is_outlier"].iloc[i : i + batch_rows]
        chunk_i = df["is_intermediate"].iloc[i : i + batch_rows]
        s = "".join(
            "O" if o else ("i" if inter else ".")
            for o, inter in zip(chunk.tolist(), chunk_i.tolist())
        )
        symbols.append(s)
    return "\n".join(symbols)