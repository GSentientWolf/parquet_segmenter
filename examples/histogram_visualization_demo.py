#!/usr/bin/env python3
"""
Demo of the new ASCII histogram visualization for random index generators.

This demonstrates the visual histogram capabilities added to the parquet_segmenter
index generator system.

Run from repo root:
    .venv/bin/python examples/histogram_visualization_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parquet_segmenter.index_generators import (
    stdlib_choice_factory,
    numpy_choice_factory,
    histogram_from_generator,
    visualize_histogram,
    sample_with_histogram,
)


def main():
    print("🎯 ASCII Histogram Visualization Demo")
    print("=" * 50)
    
    print("\n📊 Example 1: Uniform Distribution")
    print("-" * 30)
    factory = stdlib_choice_factory(4, replace=True, seed=42)
    counts = histogram_from_generator(factory, n=1000, num_bins=4)
    print(visualize_histogram(counts, width=40, show_counts=True, show_percentages=True))
    
    print("\n📊 Example 2: Heavily Skewed Distribution")
    print("-" * 30)
    factory = stdlib_choice_factory(5, probs=[0.6, 0.25, 0.1, 0.04, 0.01], replace=True, seed=123)
    counts = histogram_from_generator(factory, n=2000, num_bins=5)
    print(visualize_histogram(counts, width=50, show_counts=True, show_percentages=True))
    
    print("\n📊 Example 3: Custom Bar Characters")
    print("-" * 30)
    factory = stdlib_choice_factory(3, replace=True, seed=456)
    counts = histogram_from_generator(factory, n=1500, num_bins=3)
    print(visualize_histogram(
        counts, 
        width=35, 
        show_counts=True, 
        show_percentages=False,
        bar_char="▓", 
        empty_char="░"
    ))
    
    print("\n📊 Example 4: One-Liner with Built-in Histogram")
    print("-" * 30)
    factory = stdlib_choice_factory(6, probs=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02], replace=True, seed=789)
    print("Generating samples with automatic histogram display:")
    sample_with_histogram(factory, n=3000, num_bins=6, show_histogram=True, histogram_width=45)
    
    print("\n📊 Example 5: Edge Cases")
    print("-" * 30)
    print("Empty histogram:")
    print(visualize_histogram([]))
    print("\nAll zeros:")
    print(visualize_histogram([0, 0, 0, 0]))
    print("\nSingle large value:")
    print(visualize_histogram([500], show_counts=True, show_percentages=True))
    
    try:
        print("\n📊 Example 6: NumPy Backend (if available)")
        print("-" * 30)
        factory = numpy_choice_factory(4, probs=[0.4, 0.3, 0.2, 0.1], replace=True, seed=999)
        counts = histogram_from_generator(factory, n=1000, num_bins=4)
        print(visualize_histogram(counts, width=40, show_counts=True, show_percentages=True))
    except RuntimeError:
        print("\n📊 Example 6: NumPy not available, skipping")
    
    print("\n✨ Histogram visualization features:")
    print("  • ASCII bar charts with customizable characters")
    print("  • Configurable width and styling")
    print("  • Count and percentage display")
    print("  • Works with any generator factory")
    print("  • Handles edge cases gracefully")
    print("  • Integration with existing index generators")


if __name__ == "__main__":
    main()
