"""Lightweight logging wrapper that prefers loguru when available.

This module exposes `logger` with a similar interface to loguru's logger.
If loguru is not installed, it falls back to the stdlib `logging` module.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import os

# Module-level logger variable (will be set to loguru.logger or stdlib logger)
logger: Any

try:
    # Prefer loguru if available (import error is expected on some systems)
    from loguru import logger  # type: ignore
    # Configure loguru with UTC timestamps and colored output.
    try:
        # remove default handlers and add a single stdout sink with a compact
        # colored format. Avoid touching protected internals (logger._core)
        # which vary between loguru versions.
        logger.remove()
        # Format time without a timezone token and append an explicit
        # indicates UTC across loguru versions.
        _FORMAT_STR = (
            "<green>{time:YYYY-MM-DDTHH:mm:ss!UTC}</green> | "
            "<level>{level}</level> | {message}"
        )

        def _sink_with_tz(msg: object) -> None:
            """Console sink that formats the record's time with timezone.

            Uses the Message.record['time'] isoformat so the timezone is
            embedded in the timestamp regardless of loguru version.
            """

            try:
                rec = getattr(msg, "record", None)
                if rec is None:
                    print(msg, end="")
                    return

                ts = rec.get("time")
                try:
                    ts_str = ts.isoformat()
                except (AttributeError, ValueError, TypeError):
                    ts_str = str(ts)

                level = rec.get("level")
                level_name = getattr(level, "name", str(level))
                message_text = rec.get("message")
                try:
                    print(f"{ts_str} | {level_name} | {message_text}")
                except (OSError, TypeError):
                    print(str(msg), end="")
            except (AttributeError, TypeError):
                try:
                    print(str(msg), end="")
                except OSError:
                    pass

        # Respect an environment-configured log level (INFO by default).
        _LOG_LEVEL = os.getenv("PARQUET_SEGMENTER_LOG_LEVEL", "INFO")
        logger.add(
            sink=_sink_with_tz,
            level=_LOG_LEVEL,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )
        # File sink: enabled and path are configurable via environment
        # variables. This avoids adding a dependency like decouple and keeps
        # behavior explicit. Environment variables:
        # - PARQUET_SEGMENTER_LOG_TO_FILE: '1','true','yes' enable file sink
        # - PARQUET_SEGMENTER_LOG_PATH: path to write logs (defaults to
        #   repo-root/parquet_segmenter.log)
        _log_to_file_raw = os.getenv("PARQUET_SEGMENTER_LOG_TO_FILE", "0")
        _LOG_TO_FILE = str(_log_to_file_raw).lower() in ("1", "true", "yes", "on")
        _LOG_PATH = Path(os.getenv("PARQUET_SEGMENTER_LOG_PATH", ""))
        if not _LOG_PATH:
            _LOG_PATH = Path(__file__).resolve().parents[2] / "parquet_segmenter.log"

        def _file_sink(msg: object) -> None:
            """Append a formatted log line to the file at `_LOG_PATH`.

            This mirrors the console output: the record's datetime is rendered
            via isoformat() so the timezone is embedded in the timestamp.
            """

            try:
                rec = getattr(msg, "record", None)
                if rec is None:
                    with _LOG_PATH.open("a", encoding="utf-8") as _f:
                        _f.write(str(msg))
                    return

                ts = rec.get("time")
                try:
                    ts_str = ts.isoformat()
                except (AttributeError, ValueError, TypeError):
                    ts_str = str(ts)

                level = rec.get("level")
                level_name = getattr(level, "name", str(level))
                message_text = rec.get("message")
                with _LOG_PATH.open("a", encoding="utf-8") as _f:
                    _f.write(f"{ts_str} | {level_name} | {message_text}\n")
            except OSError:
                # If writing fails (disk full, permission), do not raise.
                pass

        if _LOG_TO_FILE:
            logger.add(
                sink=_file_sink,
                level=_LOG_LEVEL,
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
    except (AttributeError, TypeError, OSError):
        # If loguru internals differ or the sink can't be installed, keep the
        # imported logger as-is.
        pass
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("parquet_segmenter")

# Expose module-level `logger` (typed Any for downstream compatibility)
# The concrete logger object is assigned above in the import branches.

__all__ = ["logger"]
