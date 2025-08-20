"""Index generators subpackage.

Lightweight helpers to produce index sequences used by tests or internal code.
"""

from .random_index import generate_random_indices
from .strategies import (
    histogram_from_generator,
    numpy_choice_factory,
    sample_with_histogram,
    stdlib_choice_factory,
    visualize_histogram,
)

__all__ = [
    "generate_random_indices",
    "stdlib_choice_factory",
    "numpy_choice_factory",
    "histogram_from_generator",
    "visualize_histogram",
    "sample_with_histogram",
]
