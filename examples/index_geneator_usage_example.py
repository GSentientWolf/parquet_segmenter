"""Small example script demonstrating the choice factories and histogram helper.

Run from repo root (inside the virtualenv):

    .venv/bin/python examples/usage_example.py

This script prints a few sample draws and a histogram.
"""
from __future__ import annotations

from parquet_segmenter.index_generators import (
    stdlib_choice_factory,
    numpy_choice_factory,
    histogram_from_generator,
    generate_random_indices,
)


def run_examples() -> None:
    print("stdlib: uniform with replacement")
    f = stdlib_choice_factory(3, replace=True, seed=42)
    it = f()
    print([next(it) for _ in range(10)])

    print("stdlib: weighted with replacement")
    f = stdlib_choice_factory(3, probs=[0.8, 0.1, 0.1], replace=True, seed=1)
    it = f()
    print([next(it) for _ in range(10)])

    try:
        print("numpy: weighted without replacement")
        f = numpy_choice_factory(4, probs=[0.6, 0.2, 0.1, 0.1], replace=False, seed=2)
        print(list(f()))
    except RuntimeError:
        print("numpy not installed; skipping numpy example")

    print("histogram example")
    f = stdlib_choice_factory(3, replace=True, seed=42)
    hist = histogram_from_generator(f, n=1000, num_bins=3)
    print(hist)


if __name__ == "__main__":
    run_examples()
