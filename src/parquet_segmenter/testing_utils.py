from typing import Any
import importlib

import pytest


def to_int_scalar(x: Any) -> int:
    """Convert a pandas/numpy scalar or Python value to int for tests.

    This helper centralizes the pattern used in tests where pandas Scalar
    types (e.g., numpy.int64, pandas scalar) need to be converted to a
    plain Python int in a type-safe, robust way.
    """
    # Prefer direct conversion; many pandas/numpy scalars support int(x).
    try:
        return int(x)
    except Exception:
        # Fallback to .item() for scalar wrappers that expose it.
        try:
            return int(x.item())  # type: ignore[attr-defined]
        except Exception:
            # Let the original conversion error propagate for visibility.
            return int(x)


def require_pkg(pkg_name: str) -> Any:
    """Import an optional package or skip the current test.

    Tests should call this to ensure an optional dependency is present and
    otherwise skip the test with a clear message.
    """
    try:
        return importlib.import_module(pkg_name)
    except ImportError:
        pytest.skip(f"{pkg_name} not installed")
