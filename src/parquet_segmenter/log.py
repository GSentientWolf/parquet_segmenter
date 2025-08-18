"""Lightweight logging wrapper that prefers loguru when available.

This module exposes `logger` with a similar interface to loguru's logger.
If loguru is not installed, it falls back to the stdlib `logging` module.
"""
from __future__ import annotations

from typing import Any
_LOGGER: Any = None

try:
    # Import under a different name to avoid redefining the public `logger`
    from loguru import logger as _loguru_logger  # type: ignore

    _LOGGER = _loguru_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    _LOGGER = logging.getLogger("parquet_segmenter")

# Export a module-level name `logger` with a permissive type so static
# analyzers and downstream modules can import it without type errors.
logger: Any = _LOGGER

__all__ = ["logger"]
