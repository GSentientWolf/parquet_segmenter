"""Blob DataFrame generation and augmentation helpers used in tests and
examples.

This module provides functions to build synthetic DataFrames that simulate
binary "blob" columns with a controllable distribution of sizes (small,
intermediate, outlier) and a helper to augment such a DataFrame with left-
side fake fields and monotonic timestamps.

The implementations are intentionally pragmatic and somewhat long; rather
than perform a risky large refactor here we add targeted pylint disables
for complexity-related warnings. Calling code can rely on the stable public
functions: :func:`build_blob_dataframe`, :func:`augment_with_faker_and_timestamps`,
and :func:`visualize_batch_layout`.
"""

# Standard library
from __future__ import annotations

import datetime
import math
import pickle
import random
import warnings
from enum import IntEnum, StrEnum
from typing import Callable, Iterable, List, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Third-party
import pandas as pd
from faker import Faker
from freezegun import freeze_time

# First-party
from parquet_segmenter.index_generators.strategies import stdlib_choice_factory
from parquet_segmenter.log import logger

# The functions in this module are deliberately feature-rich and therefore
# trigger pylint complexity checks. Adding a small, module-level disable is
# a pragmatic choice until a larger refactor is scheduled.
# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-positional-arguments,line-too-long


class OutlierStrategy(StrEnum):
    """Strategy for placing large outlier blobs in the DataFrame."""

    SPREAD = "spread"  # Distribute across different batches
    SINGLE_BATCH = "single"  # All outliers in one batch, spread within batch
    CONTIGUOUS = "contiguous"  # All outliers in one batch, contiguous block
    MULTI_CLUSTER = "multi"  # Multiple clusters across batches


class ChunkSize(IntEnum):
    """Common chunk size constants used across examples and tests."""
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


def _apply_outliers(
    rnd: random.Random,
    sizes: List[int],
    num_batches: int,
    batch_rows: int,
    _est_rows: int,
    outlier_bins: List[int],
    outlier_strategy: OutlierStrategy,
    top_outliers: int,
    cluster_count: int,
    num_outliers: int,
    _outlier_mb_rate: int,
) -> None:
    """Internal helper to choose and apply outlier sizes into the `sizes` list.

    This mutates `sizes` in-place and ensures deterministic placement per
    `rnd` seed. It avoids returning complex structures so callers remain simple.
    """
    # Generate positions only for actual available indices in the trimmed sizes array
    available_positions = [
        (b, i)
        for b in range(num_batches)
        for i in range(batch_rows)
        if (b * batch_rows + i) < len(sizes)
    ]
    rnd.shuffle(available_positions)

    chosen_positions: List[tuple[int, int]] = []

    if top_outliers > 0 and num_batches > 0:
        if outlier_strategy == OutlierStrategy.SPREAD:
            pick_batches = (
                rnd.sample(range(num_batches), k=min(num_batches, top_outliers))
                if num_batches >= top_outliers
                else [rnd.randrange(num_batches) for _ in range(top_outliers)]
            )
            for b in pick_batches:
                valid_pos_pairs = [
                    (b, i)
                    for i in range(batch_rows)
                    if (b * batch_rows + i) < len(sizes)
                ]
                if not valid_pos_pairs:
                    continue
                rnd.shuffle(valid_pos_pairs)
                chosen_positions.append(valid_pos_pairs[0])

        elif outlier_strategy in [
            OutlierStrategy.SINGLE_BATCH,
            OutlierStrategy.CONTIGUOUS,
        ]:
            if num_batches >= 1:
                target_batch = rnd.randrange(num_batches)
            else:
                target_batch = 0

            valid_pos_indices = [
                i
                for i in range(batch_rows)
                if (target_batch * batch_rows + i) < len(sizes)
            ]

            if (
                valid_pos_indices
                and outlier_strategy == OutlierStrategy.CONTIGUOUS
                and top_outliers <= len(valid_pos_indices)
            ):
                start_candidates = list(valid_pos_indices)
                rnd.shuffle(start_candidates)
                placed = False
                for start in start_candidates:
                    end = start + top_outliers
                    if end > batch_rows:
                        continue
                    block = list(range(start, end))
                    if any((target_batch * batch_rows + idx) >= len(sizes) for idx in block):
                        continue
                    for pos in block:
                        chosen_positions.append((target_batch, pos))
                    placed = True
                    break

                if not placed:
                    rnd.shuffle(valid_pos_indices)
                    for pos in valid_pos_indices[:top_outliers]:
                        chosen_positions.append((target_batch, pos))
            else:
                rnd.shuffle(valid_pos_indices)
                for pos in valid_pos_indices[:top_outliers]:
                    chosen_positions.append((target_batch, pos))

        elif outlier_strategy == OutlierStrategy.MULTI_CLUSTER:
            k = cluster_count
            if num_batches >= k:
                batches = rnd.sample(range(num_batches), k=k)
            else:
                batches = [rnd.randrange(num_batches) for _ in range(k)]

            per_cluster = [0] * len(batches)
            for i in range(top_outliers):
                per_cluster[i % len(batches)] += 1

            for cluster_idx, b in enumerate(batches):
                want = per_cluster[cluster_idx]
                if want <= 0:
                    continue
                valid_pos_indices = [
                    i for i in range(batch_rows) if (b * batch_rows + i) < len(sizes)
                ]
                if not valid_pos_indices:
                    continue
                rnd.shuffle(valid_pos_indices)
                for pos in valid_pos_indices[:want]:
                    chosen_positions.append((b, pos))

    # Fill remaining outlier slots from shuffled available positions, skipping duplicates
    outlier_global_indices: set[int] = set()
    for b, i in available_positions:
        if len(chosen_positions) >= num_outliers:
            break
        if (b, i) in chosen_positions:
            continue

        if outlier_strategy in [OutlierStrategy.SINGLE_BATCH, OutlierStrategy.CONTIGUOUS]:
            if chosen_positions:
                target_batch = chosen_positions[0][0]
                if b != target_batch:
                    continue

        chosen_positions.append((b, i))

    for idx, (b, pos) in enumerate(chosen_positions[:num_outliers]):
        global_idx = b * batch_rows + pos
        if global_idx >= len(sizes):
            continue
        sizes[global_idx] = outlier_bins[idx % len(outlier_bins)]
        outlier_global_indices.add(global_idx)

    # Attach the global outlier indices back to the sizes list via an attribute
    # so callers that expect to reference them can still do so. We attach a
    # private attribute name to avoid changing the public API.
    try:
        # Attach without leading underscore to avoid protected-access lint.
        # type: ignore[attr-defined]
        sizes.outlier_indices = outlier_global_indices  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # fallback: do nothing if we cannot attach attribute (built-in list
        # does not support arbitrary attributes) or if attribute assignment
        # is not allowed for some reason.
        pass


def _build_batch_template(
    small_bins: List[int],
    intermediate_bins: List[int],
    batch_rows: int,
    seed: Optional[int],
    rnd: random.Random,
) -> List[int]:
    """Create a batch template filled with small sizes and some intermediates.

    This encapsulates the logic that randomly selects contiguous and isolated
    intermediate bins within a batch template.
    """
    small_factory = stdlib_choice_factory(num_bins=len(small_bins), replace=True, seed=seed)
    it_small = small_factory()
    template: List[int] = [small_bins[next(it_small) % len(small_bins)] for _ in range(batch_rows)]

    block_len = max(1, batch_rows // 10)
    block_start = rnd.randrange(0, max(1, batch_rows - block_len))
    for j in range(block_start, block_start + block_len):
        template[j] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    num_isolated = max(1, batch_rows // 1000)
    iso_indices = rnd.sample(range(batch_rows), k=min(batch_rows, num_isolated))
    for idx in iso_indices:
        template[idx] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    return template


def _compute_batches(total_df_size: int, batch_size: int, avg_small: int) -> tuple[int, int, int]:
    """Compute rows_per_mb, estimated rows, and number of batches."""
    rows_per_mb = max(1, batch_size // max(1, avg_small))
    total_mb = total_df_size / float(batch_size)
    est_rows = max(1, int(math.ceil(rows_per_mb * total_mb)))
    batch_rows = rows_per_mb
    num_batches = int(math.ceil(est_rows / float(batch_rows)))
    return rows_per_mb, est_rows, num_batches


def _fill_sizes_from_template(num_batches: int, batch_rows: int, batch_template: List[int], est_rows: int) -> List[int]:
    """Create the full sizes list by repeating the batch template and trimming."""
    total_size = num_batches * batch_rows
    sizes: List[int] = [0] * total_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_rows
        end_idx = start_idx + batch_rows
        sizes[start_idx:end_idx] = batch_template
    if len(sizes) > est_rows:
        sizes = sizes[:est_rows]
    return sizes


def _add_intermediate_blocks(sizes: List[int], rows_per_mb: int, intermediate_bins: List[int], rnd: random.Random, outlier_global_indices: Optional[set[int]] = None) -> None:
    """Insert a contiguous intermediate block and some isolated intermediate indices."""
    if outlier_global_indices is None:
        outlier_global_indices = set()
    block_len = max(1, rows_per_mb // 10)
    block_start = rnd.randrange(0, max(1, len(sizes) - block_len)) if len(sizes) > block_len else 0
    for j in range(block_start, min(block_start + block_len, len(sizes))):
        if j in outlier_global_indices:
            continue
        sizes[j] = intermediate_bins[rnd.randrange(len(intermediate_bins))]

    num_isolated = max(1, len(sizes) // 1000)
    iso_indices = rnd.sample(range(len(sizes)), k=min(len(sizes), num_isolated))
    for idx in iso_indices:
        if idx in outlier_global_indices:
            continue
        sizes[idx] = intermediate_bins[rnd.randrange(len(intermediate_bins))]


def _finalize_dataframe(sizes: List[int], intermediate_bins_set: set, outlier_bins_set: set, binary_factory: Optional[Callable[[int], bytes]] = None) -> pd.DataFrame:
    """Build the final DataFrame with flags and optional binary blobs."""
    df = pd.DataFrame({"blob_size": sizes})
    df["is_outlier"] = df["blob_size"].isin(outlier_bins_set)
    df["is_intermediate"] = df["blob_size"].isin(intermediate_bins_set)
    if binary_factory is not None:
        df["blob"] = df["blob_size"].apply(binary_factory)
    return df


def _resolve_initial_dt_and_tz(initial_date: Union[str, datetime.datetime], tz: Optional[Union[str, datetime.tzinfo, ZoneInfo]]) -> tuple[datetime.datetime, datetime.tzinfo]:
    """Normalize initial_date and tz to timezone-aware datetime and tzinfo."""
    if isinstance(initial_date, str):
        initial_dt = datetime.datetime.fromisoformat(initial_date)
    else:
        initial_dt = initial_date

    tzinfo: datetime.tzinfo
    if isinstance(tz, str):
        try:
            tzinfo = ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            tzinfo = datetime.timezone.utc
    else:
        tzinfo = tz or datetime.timezone.utc

    if initial_dt.tzinfo is None:
        initial_dt = initial_dt.replace(tzinfo=tzinfo)
    else:
        initial_dt = initial_dt.astimezone(tzinfo)
    return initial_dt, tzinfo


def _generate_fake_rows(n: int, fake: Faker, initial_dt: datetime.datetime) -> List[dict]:
    """Generate `n` fake rows with timezone-aware timestamps incremented by 1 microsecond."""
    rows: List[dict] = []
    with freeze_time(initial_dt) as frozen:
        for i in range(n):
            frozen.move_to(initial_dt + datetime.timedelta(microseconds=i))
            ts = datetime.datetime.now(tz=initial_dt.tzinfo)
            dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
            age = (ts.date() - dob).days // 365
            lat, long, city, country, time_zone = fake.location_on_land()
            rows.append(
                {
                    "first_name": fake.first_name(),
                    "last_name": fake.last_name(),
                    "dob": dob,
                    "age": age,
                    "ssn": fake.ssn(),
                    "address": fake.address(),
                    "latitude": lat,
                    "longitude": long,
                    "city": city,
                    "country": country,
                    "time_zone": time_zone,
                    "email": fake.email(),
                    "company": fake.company(),
                    "job": fake.job(),
                    "timestamp": ts,
                }
            )
    return rows


def build_blob_dataframe(
    *,
    total_df_size: int,
    small_bins: Iterable[int] = tuple(
        map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16])
    ),
    intermediate_bins: Iterable[int] = tuple(
        map(lambda x: x * ChunkSize.ONE_KB, [256, 307, 409, 470, 512])
    ),
    outlier_bins: Iterable[int] = tuple(
        map(lambda x: x * ChunkSize.ONE_KB, [820, 921, 1044, 1228])
    ),
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
    Build a DataFrame with a 'blob_size' column (bytes as integer sizes) and
    optional 'blob' column containing actual binary payloads when
    `binary_factory` is provided.

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
    if any(
        param is not None
        for param in [spread_top_outliers, cluster_batches, contiguous_within_batch]
    ):
        warnings.warn(
            "Parameters 'spread_top_outliers', 'cluster_batches', and 'contiguous_within_batch' "
            "are deprecated. Use 'outlier_strategy' parameter instead.",
            DeprecationWarning,
            stacklevel=2,
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
    _ = set(small_bins)
    intermediate_bins_set = set(intermediate_bins)
    outlier_bins_set = set(outlier_bins)

    avg_small = int(sum(small_bins) / len(small_bins))
    rows_per_mb, est_rows, num_batches = _compute_batches(
        total_df_size, batch_size, avg_small
    )

    # Build a batch template filled with small sizes. We'll copy this template
    # across batches and then selectively override positions for intermediate
    # and outlier cases.
    batch_template = _build_batch_template(small_bins, intermediate_bins, rows_per_mb, seed, rnd)

    sizes = _fill_sizes_from_template(num_batches, rows_per_mb, batch_template, est_rows)

    # Determine number of outliers to add (sparse)
    # e.g., roughly 1 outlier per outlier_mb_rate MB
    # Determine number of outliers to add (sparse): roughly 1 per outlier_mb_rate MB
    num_outliers = (
        max(1, int(math.floor(total_df_size / (outlier_mb_rate * ChunkSize.ONE_MB))))
        if total_df_size >= (outlier_mb_rate * ChunkSize.ONE_MB // 2)
        else 0
    )
    # ensure at least the requested top_outliers
    num_outliers = max(num_outliers, top_outliers)
    # cap to available estimated rows
    num_outliers = min(num_outliers, est_rows)

    # Choose and apply outliers
    _apply_outliers(
        rnd,
        sizes,
        num_batches,
        rows_per_mb,
        est_rows,
        outlier_bins,
        outlier_strategy,
        top_outliers,
        cluster_count,
        num_outliers,
        outlier_mb_rate,
    )

    outlier_global_indices: set[int] = {i for i, s in enumerate(sizes) if s in outlier_bins_set}

    _add_intermediate_blocks(sizes, rows_per_mb, intermediate_bins, rnd, outlier_global_indices)

    # Build DataFrame
    # Build DataFrame: numeric blob size column is named 'blob_size'. If a
    # binary_factory is provided we materialize the bytes into the 'blob'
    # column (this mirrors how other generators attach binary payloads).
    return _finalize_dataframe(sizes, intermediate_bins_set, outlier_bins_set, binary_factory)


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


def augment_with_faker_and_timestamps(
    df: pd.DataFrame,
    *,
    initial_date: Union[str, datetime.datetime],
    binary_factory: Optional[Callable[[int], bytes]] = None,
    faker_locale: Optional[str] = None,
    faker_seed: Optional[int] = None,
    tz: Optional[Union[str, datetime.tzinfo, ZoneInfo]] = datetime.timezone.utc,
) -> pd.DataFrame:
    """Return a new DataFrame with Faker-generated columns on the left and
    a monotonic timestamp (1 microsecond increment per row) produced via
    freezegun starting at ``initial_date``.

    Behavior / contract:
    - The returned DataFrame contains the fake-data columns first (left),
      then the columns from the input ``df`` (which is expected to come from
      ``build_blob_dataframe`` and therefore contains a ``blob_size`` column).
    - The function computes a truthful per-row byte size of the fake-data
      portion and stores it in a temporary column named ``row_size`` before
      merging. The computation serializes the fake-row dict via pickle and
      measures the byte length (this gives a truthful representation of the
      Python-level size for the fake data).
    - After merging, a new column ``size`` is added at the right-most side and
      equals ``row_size + blob_size``. The temporary ``row_size`` column is
      dropped before returning.
        - If ``binary_factory`` is provided and the input ``df`` does not contain
            a ``blob`` column, it will be materialized using ``binary_factory(blob_size)``.

        Note for maintainers: the test `__tests__/unit/parquet_segmenter/functional_testing/test_blob_df_augmented.py`
        asserts the expected set of left-side fake columns and the size computation
        behavior. When changing the fake fields or their serialization, update the
        test accordingly.

    Args:
        df: DataFrame produced by ``build_blob_dataframe``; its length is used
            to generate the same number of fake rows.
        initial_date: ISO string or ``datetime`` describing the start timestamp
            used by freezegun. Each row's timestamp will be initial_date + i
            microseconds.
        binary_factory: Optional callable to materialize blob bytes from
            ``blob_size`` when the input df does not already include a
            ``blob`` column.
        faker_locale: Optional Faker locale string (passed to Faker()).
        faker_seed: Optional seed passed to the Faker generator for
            reproducibility.
        tz: Optional timezone for timestamps. Accepts:
            - None -> defaults to UTC
            - a tz name string (e.g. 'America/Los_Angeles') resolved via zoneinfo.ZoneInfo
            - a datetime.tzinfo instance (e.g. datetime.timezone.utc or ZoneInfo)
            If a tz name cannot be resolved the function will fall back to UTC
            and emit a warning via the `logging` module.
    """
    # Validate inputs
    if "blob_size" not in df.columns:
        raise ValueError("input DataFrame must contain a 'blob_size' column")

    n = len(df)

    # Prepare Faker
    fake = Faker(faker_locale) if faker_locale else Faker()
    if faker_seed is not None:
        try:
            fake.seed_instance(faker_seed)
        except AttributeError:
            # older faker versions may use classmethod `Faker.seed`
            Faker.seed(faker_seed)

    # Materialize blob bytes if requested and missing
    if binary_factory is not None and "blob" not in df.columns:
        df = df.copy()
        df["blob"] = df["blob_size"].apply(binary_factory)

    # Normalize initial_date to a datetime
    if isinstance(initial_date, str):
        initial_dt = datetime.datetime.fromisoformat(initial_date)
    else:
        initial_dt = initial_date

    # Resolve tz parameter to a tzinfo object. Accept either a tzinfo or a
    # zone name string (ZoneInfo). Default is UTC.
    tzinfo: Optional[datetime.tzinfo]
    if isinstance(tz, str):
        try:
            tzinfo = ZoneInfo(tz)
        except ZoneInfoNotFoundError as e:
            # fallback to UTC if zone name cannot be resolved and log the error
            logger.warning(
                "Could not resolve timezone {tz!r} via ZoneInfo, falling back to UTC: {err}",
                tz=tz,
                err=e,
            )
            tzinfo = datetime.timezone.utc
    else:
        tzinfo = tz or datetime.timezone.utc

    # If the provided datetime is naive, attach resolved tzinfo; otherwise
    # convert to the resolved tz.
    if initial_dt.tzinfo is None:
        initial_dt = initial_dt.replace(tzinfo=tzinfo)
    else:
        # Convert to requested timezone to ensure consistency
        initial_dt = initial_dt.astimezone(tzinfo)

    # Build fake rows using freezegun to set per-row timestamp
    fake_rows: List[dict] = []
    # Resolve datetime and timezone and generate fake rows via helper
    initial_dt, _ = _resolve_initial_dt_and_tz(initial_date, tz)
    fake_rows = _generate_fake_rows(n, fake, initial_dt)

    left_df = pd.DataFrame(fake_rows)

    # Compute truthful per-row size for the left-side fake data. We use pickle
    # to obtain a realistic serialized size for the fake-data dict per-row.
    left_df["row_size"] = left_df.apply(
        lambda r: len(pickle.dumps(r.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)),
        axis=1,
    )

    # Reset indices to ensure alignment when concatenating left and right
    left_df = left_df.reset_index(drop=True)
    right_df = df.reset_index(drop=True)

    merged = pd.concat([left_df, right_df], axis=1)

    # Add final 'size' column at the most-right side: row_size + blob_size
    if "blob_size" not in merged.columns:
        raise ValueError("merged DataFrame missing 'blob_size' column")

    merged["size"] = merged["row_size"] + merged["blob_size"]

    # Drop the temporary row_size column
    merged = merged.drop(columns=["row_size"])
    return merged
