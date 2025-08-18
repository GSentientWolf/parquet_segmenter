"""Lightweight logging wrapper that prefers loguru when available.

This module exposes `logger` with a similar interface to loguru's logger.
If loguru is not installed, it falls back to the stdlib `logging` module.
"""
from __future__ import annotations

from typing import Any

try:
    # Prefer loguru if available (import error is expected on some systems)
    from loguru import logger  # type: ignore
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("parquet_segmenter")

# Expose module-level `logger` (typed Any for downstream compatibility)
logger: Any = logger  # type: ignore

__all__ = ["logger"]
