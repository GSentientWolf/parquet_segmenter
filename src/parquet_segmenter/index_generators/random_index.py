"""Simple index generators used by tests.

This module provides a tiny deterministic random index generator for tests.
"""

# Note: Consider revisiting the index generator API in future to support
# more flexible ranges, sampling modes (with/without replacement), and
# streaming generation for large counts.

from __future__ import annotations

import random
from typing import Callable, Iterator, List, Union


def generate_random_indices(
    arg: Union[int, Callable[..., Iterator[int]], Iterator[int]],
    k: int | None = None,
    seed: int | None = None,
) -> List[int]:
    """Generate or collect random indices.

    Backwards-compatible behaviour:
    - Old API: `generate_random_indices(count: int, seed: int | None = None) -> List[int]`
      returns a deterministic permutation of `0..count-1`.

    - New API (factory/iterator): `generate_random_indices(rnd_fn, k)` where
      `rnd_fn` is either a callable returning an iterator[int] or an iterator
      itself; `k` is the number of values to collect. Returns a list of length
      `k` with values produced by the iterator.

    Args:
        arg: either `count` (int) for permutation mode, or a callable/iterator
             producing ints for streaming mode.
        k: when `arg` is a callable or iterator, `k` is the number of values to
           collect. When `arg` is an int `k` is ignored and `seed` is used.
        seed: used only when `arg` is an int to seed the permutation.

    Returns:
        A list of ints.

    Raises:
        ValueError: for invalid arguments or if an iterator is exhausted
                    before `k` values are collected.
    """
    # Permutation mode: (count, seed) signature
    if isinstance(arg, int):
        count = arg
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)
        seq: List[int] = list(range(count))
        rng.shuffle(seq)
        return seq

    # Streaming/factory mode: arg is callable or iterator, k must be provided
    if k is None:
        raise TypeError(
            "k must be provided when passing a generator factory or iterator"
        )

    rnd_fn = arg
    if callable(rnd_fn):
        rnd_iter = rnd_fn()
    else:
        rnd_iter = rnd_fn

    out: List[int] = []
    for _ in range(k):
        try:
            out.append(next(rnd_iter))
        except StopIteration as exc:
            raise ValueError("provided rnd_fn produced fewer than k values") from exc
    return out
