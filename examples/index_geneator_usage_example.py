"""Small example script demonstrating the choice factories and histogram helper.

Run from repo root (inside the virtualenv):

    .venv/bin/python examples/index_geneator_usage_example.py

This script prints a few sample draws and visual ASCII histograms.
"""
from __future__ import annotations

from parquet_segmenter.index_generators import (
    stdlib_choice_factory,
    numpy_choice_factory,
    histogram_from_generator,
    visualize_histogram,
    sample_with_histogram,
)


def run_examples() -> None:
    print("=== Basic Examples ===")
    print("stdlib: uniform with replacement")
    f = stdlib_choice_factory(3, replace=True, seed=42)
    it = f()
    print([next(it) for _ in range(10)])

    print("\nstdlib: weighted with replacement")
    f = stdlib_choice_factory(3, probs=[0.8, 0.1, 0.1], replace=True, seed=1)
    it = f()
    print([next(it) for _ in range(10)])

    try:
        print("\nnumpy: weighted without replacement")
        f = numpy_choice_factory(4, probs=[0.6, 0.2, 0.1, 0.1], replace=False, seed=2)
        print(list(f()))
    except RuntimeError:
        print("\nnumpy not installed; skipping numpy example")

    print("\n=== Visual Histogram Examples ===")
    
    # Example 1: Uniform distribution
    print("\n1. Uniform distribution (1000 samples, 3 bins):")
    f = stdlib_choice_factory(3, replace=True, seed=42)
    hist = histogram_from_generator(f, n=1000, num_bins=3)
    print(visualize_histogram(hist, show_counts=True, show_percentages=True))
    
    # Example 2: Heavily weighted distribution  
    print("\n2. Heavily weighted distribution (1000 samples, 4 bins):")
    f = stdlib_choice_factory(4, probs=[0.7, 0.2, 0.08, 0.02], replace=True, seed=123)
    hist = histogram_from_generator(f, n=1000, num_bins=4)
    print(visualize_histogram(hist, show_counts=True, show_percentages=True, width=60))
    
    # Example 3: Using convenience function
    print("\n3. Using sample_with_histogram with built-in histogram:")
    f = stdlib_choice_factory(5, probs=[0.4, 0.3, 0.15, 0.1, 0.05], replace=True, seed=456)
    indices = sample_with_histogram(
        f, 
        n=2000, 
        num_bins=5, 
        show_histogram=True, 
        histogram_width=50
    )
    print(f"Generated {len(indices)} indices")
    
    # Example 4: Custom bar characters
    print("\n4. Custom visualization (using different characters):")
    f = stdlib_choice_factory(6, replace=True, seed=789)
    hist = histogram_from_generator(f, n=1500, num_bins=6)
    print(visualize_histogram(
        hist, 
        width=35,
        show_counts=True, 
        show_percentages=False,
        bar_char="▓",
        empty_char="░"
    ))


if __name__ == "__main__":
    run_examples()
