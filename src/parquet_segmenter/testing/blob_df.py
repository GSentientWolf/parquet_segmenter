from __future__ import annotations

import math
import random
from enum import IntEnum, StrEnum
from typing import Callable, Iterable, List, Optional
from dataclasses import dataclass

import logging
import pandas as pd
from parquet_segmenter.index_generators.strategies import stdlib_choice_factory


def _validate_placement_params(
    spread_top_outliers: bool, cluster_batches: Optional[int], contiguous_within_batch: bool
) -> None:
    """Validate mutually-exclusive placement parameters.

    Fail-fast: log an error and raise ValueError when parameters conflict.
    """
    if spread_top_outliers and (cluster_batches is not None or contiguous_within_batch):
        msg = (
            "conflicting parameters: 'spread_top_outliers=True' cannot be used with "
            "'cluster_batches' or 'contiguous_within_batch'."
        )
        logging.getLogger(__name__).error(msg)
        raise ValueError(msg)


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
    small_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 10, 14])),
    intermediate_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB, [256, 307, 409, 512])),
    outlier_bins: Iterable[int] = tuple(map(lambda x: x * ChunkSize.ONE_KB, [921, 1044, 1228])),
    batch_size: int = ChunkSize.ONE_MB,
    outlier_mb_rate: int = 10,
    top_outliers: int = 1,
    spread_top_outliers: bool = True,
    cluster_batches: Optional[int] = None,
    contiguous_within_batch: bool = False,
    seed: Optional[int] = None,
    binary_factory: Optional[Callable[[int], bytes]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with a 'size' column (bytes) and optional 'blob' column.

    Behavior summary:
    - Use small_bins as the common small sizes (uniform-ish).
      estimate rows_per_mb = batch_bytes / avg_small (defaults from small_bins avg).
      total rows ~= rows_per_mb * (total_bytes / batch_bytes).
    - Place 'intermediate' sizes sparsely (some contiguous group + some random).
            - Place outliers sparsely: ~1 per outlier_mb_rate MB (floor), plus `top_outliers`.
                By default the `top_outliers` are spread across distinct batches; set
                `spread_top_outliers=False` to cluster them. Use `cluster_batches` to
                request K clusters (defaults to 1 when clustering). If `contiguous_within_batch`
                is True and a cluster contains multiple outliers, the function will attempt
                to place them contiguously within the chosen batch.
    - The function guarantees at least one outlier when total_bytes >= outlier_mb_rate*1_000_000
            and always includes `top_outliers` (subject to available rows).
    - `binary_factory(size) -> bytes` can be provided to construct actual blob bytes;
      otherwise blob column is omitted (only sizes and flags).
    """
    rnd = random.Random(seed)

    # Validate placement parameters (fail-fast on conflict)
    _validate_placement_params(spread_top_outliers, cluster_batches, contiguous_within_batch)

    small_bins = list(small_bins)
    intermediate_bins = list(intermediate_bins)
    outlier_bins = list(outlier_bins)

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
    rnd = random.Random(seed)
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

    # Prepare full sizes list by repeating the batch template
    sizes: List[int] = []
    for _ in range(num_batches):
        sizes.extend(list(batch_template))

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
    available_positions = [(b, i) for b in range(num_batches) for i in range(batch_rows)]
    rnd.shuffle(available_positions)
    # Ensure top_outliers occupy early positions in a randomly-chosen batch
    # (seeded for determinism). Pick positions only if their global index
    # will be within the trimmed `sizes` list; fall back to other available
    # positions if the chosen batch is too small.
    chosen_positions: List[tuple[int, int]] = []
    if top_outliers > 0 and num_batches > 0:
        if spread_top_outliers:
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
        else:
            # Clustering: create `cluster_batches` clusters (default 1)
            k = cluster_batches if cluster_batches and cluster_batches > 0 else 1
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

                if contiguous_within_batch and want <= len(valid_positions):
                    # try to find a contiguous block of length `want`
                    start_candidates = [i for i in valid_positions]
                    rnd.shuffle(start_candidates)
                    placed = 0
                    for start in start_candidates:
                        end = start + want
                        if end > batch_rows:
                            continue
                        block = list(range(start, end))
                        # ensure block maps inside sizes
                        if any((b * batch_rows + idx) >= len(sizes) for idx in block):
                            continue
                        for pos in block:
                            chosen_positions.append((b, pos))
                        placed = want
                        break
                    if placed == want:
                        continue

                # fallback: pick `want` distinct positions in the batch
                rnd.shuffle(valid_positions)
                for pos in valid_positions[:want]:
                    chosen_positions.append((b, pos))

    # Fill remaining outlier slots from shuffled available positions, skipping duplicates
    for (b, i) in available_positions:
        if len(chosen_positions) >= num_outliers:
            break
        if (b, i) in chosen_positions:
            continue
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
    block_start = rnd.randrange(0, max(1, est_rows - block_len))
    for j in range(block_start, min(block_start + block_len, len(sizes))):
        if j in outlier_global_indices:
            # keep outliers intact
            continue
        sizes[j] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    # random few isolated intermediate indices across the full sizes list
    num_isolated = max(1, est_rows // 1000)
    iso_indices = rnd.sample(range(len(sizes)), k=min(len(sizes), num_isolated))
    for idx in iso_indices:
        if idx in outlier_global_indices:
            continue
        sizes[idx] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    # Build DataFrame
    df = pd.DataFrame({"size": sizes})
    df["is_outlier"] = df["size"].isin(outlier_bins)
    df["is_intermediate"] = df["size"].isin(intermediate_bins)

    # Optionally materialize binary blobs using binary_factory
    if binary_factory is not None:
        df["blob"] = df["size"].apply(binary_factory)

    return df


@dataclass
class PlacementConfig:
    """Small helper dataclass describing top-outlier placement preferences.

    Not required by the public API; provided for readability and future extension.
    """
    spread_top_outliers: bool = True
    cluster_batches: Optional[int] = None
    contiguous_within_batch: bool = False


def visualize_batch_layout(df: pd.DataFrame, batch_rows: int) -> str:
    """Return a compact textual visualization of batches.

    Marks: '.' small, 'i' intermediate, 'O' outlier.
    """
    symbols = []
    for v in df["size"]:
        if v in list(map(int, df["size"])):  # quick keep (no-op but explicit)
            pass
    for i in range(0, len(df), batch_rows):
        chunk = df["is_outlier"].iloc[i : i + batch_rows]
        chunk_i = df["is_intermediate"].iloc[i : i + batch_rows]
        s = "".join(
            "O" if o else ("i" if inter else ".")
            for o, inter in zip(chunk.tolist(), chunk_i.tolist())
        )
        symbols.append(s)
    return "\n".join(symbols)