"""Random index generation strategies.

Provides factories that return iterators yielding indices according to various
random backends (stdlib random and NumPy) and a helper to build histograms of
samples.
"""

# Some helpers in this module are intentionally compact and accept multiple
# parameters to remain convenient for tests and examples. Defer deep
# refactors and silence the specific pylint complexity checks here.
# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

import random
from typing import Callable, Iterable, Iterator, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in some envs
    np = None  # type: ignore  # pylint: disable=invalid-name


def _validate_and_normalize_probs(
    num_bins: int,
    probs: Optional[Iterable[float]],
) -> Optional[List[float]]:
    """Validate `probs` and return a normalized list summing to 1.0 or None.

    Raises ValueError on invalid inputs. This centralizes checks used by both
    stdlib and numpy factories.
    """
    if probs is None:
        return None
    probs_list = [float(p) for p in list(probs)]
    if len(probs_list) != num_bins:
        raise ValueError("probs must be the same length as num_bins")
    if any(p < 0 for p in probs_list):
        raise ValueError("probs must contain non-negative weights")
    total = sum(probs_list)
    if total == 0:
        raise ValueError("probs must not be all zeros")
    return [p / total for p in probs_list]


def stdlib_choice_factory(
    num_bins: int,
    probs: Optional[Iterable[float]] = None,
    replace: bool = True,
    seed: Optional[int] = None,
) -> Callable[[], Iterator[int]]:
    """Return a factory that produces an iterator yielding indices using
    Python's stdlib random module.

    If `replace` is True the iterator yields an infinite stream using
    random.choices. If `replace` is False the iterator yields a single
    permutation of the indices then stops.
    If `probs` is provided it will be validated and normalized to sum to 1.0
    before being passed to the underlying RNG.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    # Validate and normalize probs (if provided). The returned list will
    # sum to 1.0 which is passed to the underlying RNGs.
    probs_list = _validate_and_normalize_probs(num_bins, probs)

    def factory() -> Iterator[int]:
        rng = random.Random(seed)
        population = list(range(num_bins))
        if replace:
            # infinite generator using choices
            while True:
                if probs_list is None:
                    yield rng.randrange(num_bins)
                else:
                    # random.choices accepts weights
                    choice = rng.choices(population, weights=probs_list, k=1)[0]
                    yield choice
        else:
            # non-replacement: yield a single shuffled permutation
            seq = population[:]
            rng.shuffle(seq)
            # seq contains ints already
            yield from seq

    return factory


def numpy_choice_factory(
    num_bins: int,
    probs: Optional[Iterable[float]] = None,
    replace: bool = True,
    seed: Optional[int] = None,
) -> Callable[[], Iterator[int]]:
    """Return a factory that yields indices using numpy's Generator.

    Requires numpy to be installed. Behaviour mirrors `stdlib_choice_factory`:
    - with replace=True: yields an infinite stream
    - with replace=False: yields a single shuffled permutation then stops
    If `probs` is provided it will be validated and normalized to sum to 1.0
    before being passed to NumPy's sampling functions.
    """
    if np is None:
        raise RuntimeError("numpy is required for numpy_choice_factory")
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    # Validate and normalize probs once. The returned list will sum to 1.0.
    probs_list = _validate_and_normalize_probs(num_bins, probs)

    def factory() -> Iterator[int]:
        # assert for static type checkers: np was validated above
        assert np is not None
        rng = np.random.default_rng(seed)
        if replace:
            # infinite generator
            while True:
                if probs_list is None:
                    # integers returns ndarray scalars; ensure int
                    yield int(rng.integers(0, num_bins))
                else:
                    # use choice with p=probs_list; a can be int
                    yield int(rng.choice(num_bins, p=probs_list))
        else:
            # non-replacement: produce a single sample of size num_bins
            if probs_list is not None:
                # weighted sampling without replacement
                arr = rng.choice(num_bins, size=num_bins, replace=False, p=probs_list)
                # convert numpy scalars to ints while yielding
                yield from (int(v) for v in arr)
            else:
                perm = list(rng.permutation(num_bins))
                yield from (int(v) for v in perm)

    return factory


def histogram_from_generator(
    generator_factory_or_iter: Callable[[], Iterator[int]] | Iterator[int],
    n: int,
    num_bins: int,
) -> List[int]:
    """Consume `n` samples from the provided generator (factory or iterator)
    and return a histogram (counts per index) of length `num_bins`.

    Raises ValueError on invalid inputs.
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    if n < 0:
        raise ValueError("n must be non-negative")

    # Resolve iterator
    if callable(generator_factory_or_iter):
        it = generator_factory_or_iter()
    else:
        it = generator_factory_or_iter

    counts = [0] * num_bins
    for _ in range(n):
        try:
            v = next(it)
        except StopIteration:
            # treat early exhaustion as stopping early
            break
        if not 0 <= v < num_bins:
            raise ValueError(f"generated index {v} out of range [0, {num_bins})")
        counts[v] += 1
    return counts


def visualize_histogram(
    counts: List[int],
    *,
    width: int = 50,
    show_counts: bool = True,
    show_percentages: bool = False,
    bar_char: str = "█",
    empty_char: str = " ",
 ) -> str:  # pylint: disable=too-many-arguments,too-many-locals
    """Create an ASCII visualization of histogram counts.

    Args:
        counts: List of counts for each bin (from histogram_from_generator)
        width: Maximum width of the bars in characters
        show_counts: Whether to show the actual count values
        show_percentages: Whether to show percentage values
        bar_char: Character to use for the bars
        empty_char: Character to use for empty space

    Returns:
        Multi-line string with the ASCII histogram
    """
    if not counts:
        return "(empty histogram)"

    max_count = max(counts) if counts else 1
    total_count = sum(counts)

    if max_count == 0:
        return "(all bins empty)"

    lines = []

    # Add header if showing percentages
    if show_percentages and total_count > 0:
        lines.append(f"Histogram of {total_count} samples:")
        lines.append("")

    for i, count in enumerate(counts):
        # Calculate bar length proportional to count
        if max_count > 0:
            bar_length = int((count * width) / max_count)
        else:
            bar_length = 0

        # Create the bar string (rename local to avoid disallowed-name 'bar')
        bar_str = bar_char * bar_length + empty_char * (width - bar_length)

        # Create the label
        label_parts = [f"Bin {i:2d}"]

        if show_counts:
            label_parts.append(f"({count:4d})")

        if show_percentages and total_count > 0:
            percentage = (count / total_count) * 100
            label_parts.append(f"{percentage:5.1f}%")

        label = " ".join(label_parts)

        # Combine label and bar
        lines.append(f"{label} |{bar_str}|")

    return "\n".join(lines)


def sample_with_histogram(
    generator_factory: Callable[[], Iterator[int]],
    n: int,
    num_bins: int,
    *,
    show_histogram: bool = False,
    histogram_width: int = 40,
) -> List[int]:
    """Generate a list of n random indices and optionally display a histogram.

    This is a convenience function that combines generation with optional
    visualization for quick experimentation.

    Args:
        generator_factory: Factory function that returns an iterator
        n: Number of indices to generate
        num_bins: Number of possible bins/indices
        show_histogram: Whether to print an ASCII histogram
        histogram_width: Width of the histogram bars

    Returns:
        List of generated indices
    """
    it = generator_factory()
    indices = []

    for _ in range(n):
        try:
            indices.append(next(it))
        except StopIteration:
            break

    if show_histogram:
        counts = histogram_from_generator(iter(indices), len(indices), num_bins)
        print(
            visualize_histogram(
                counts, width=histogram_width, show_counts=True, show_percentages=True
            )
        )

    return indices
