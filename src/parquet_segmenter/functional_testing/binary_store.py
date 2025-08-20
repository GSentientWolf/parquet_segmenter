"""Utilities to manage precalculated binary files for tests.

Provides a small store that lazily creates binary files of requested sizes
and returns file paths by index. Size strings like "10 kB" or "1.3 MB"
are parsed (KB/MB interpreted as powers of 1024).

Usage:
    store = PrecalculatedBinaryStore()
    path = store.get("10 kB", index=0)

Each call to `get` will add new files to the internal directory if they
don't already exist.
"""

from __future__ import annotations

import os
import re
from typing import Optional


class PrecalculatedBinaryStore:
    """Manage precalculated binary files keyed by human-readable size.

    The store keeps files under `base_dir` (created on demand). Size
    parsing interprets kB/KB and MB as powers of 1024 (e.g. 1 kB = 1024).
    """

    _SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgG]?B)?\s*$")

    def __init__(self, base_dir: Optional[str] = None) -> None:
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), ".cache", "precalc_binaries")
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def parse_size(size_str: str) -> int:
        """Parse human-readable size strings into bytes.

        Accepts values like "10 kB", "50 kB", "1.3 MB". Units are
        interpreted as powers of 1024 (KB=1024, MB=1024**2).
        """
        m = PrecalculatedBinaryStore._SIZE_RE.match(size_str)
        if not m:
            raise ValueError(f"Invalid size string: {size_str!r}")
        value = float(m.group("value"))
        unit = (m.group("unit") or "B").upper()
        if unit == "B":
            mult = 1
        elif unit in ("KB", "KB") or unit == "KB":  # explicit for clarity
            mult = 1024
        elif unit == "MB":
            mult = 1024**2
        elif unit == "GB":
            mult = 1024**3
        else:
            # fallback
            mult = 1
        return int(round(value * mult))

    def _filename_for(self, size_bytes: int) -> str:
        return os.path.join(self.base_dir, f"bin_{size_bytes}.bin")

    def ensure_size(self, size_str: str) -> str:
        """Ensure a single binary file exists for `size_str` and return its path.

        If the file doesn't exist it will be created with random bytes.
        """
        size = self.parse_size(size_str)
        p = self._filename_for(size)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(os.urandom(size))
        return p

    def _build_index(self) -> list[tuple[int, int, str]]:
        """Scan the base_dir and build a sorted index of available binaries.

        Returns a list of tuples (size_bytes, per_size_index, path) sorted by
        size_bytes ascending then per_size_index ascending.
        """
        entries: list[tuple[int, int, str]] = []
        for name in os.listdir(self.base_dir):
            if not name.startswith("bin_") or not name.endswith(".bin"):
                continue
            # expected pattern after change: bin_{size}.bin
            core = name[4:-4]
            try:
                size_bytes = int(core)
            except ValueError:
                continue
            entries.append((size_bytes, 0, os.path.join(self.base_dir, name)))
        entries.sort(key=lambda t: (t[0], t[1]))
        return entries

    def list_index(self) -> list[tuple[int, int, str]]:
        """Return the current index list as (global_index, size_bytes, path).

        global_index is the position in the sorted list (0-based). This is
        useful to map requested sizes to the store's index.
        """
        built = self._build_index()
        return [(i, size, path) for i, (size, _, path) in enumerate(built)]

    def get(self, index: int) -> str:
        """Return the path to a precalculated binary by global index.

        The global index maps into the sorted list of binaries (ascending
        by size). If the index is out of range, raises IndexError.
        """
        if index < 0:
            raise ValueError("index must be non-negative")
        built = self._build_index()
        if index >= len(built):
            # When a previously-known binary has been removed from the store
            # we want a deterministic, user-friendly error. Treat an out of
            # range index as a missing binary file and raise FileNotFoundError
            # so callers can consistently handle missing artifacts.
            raise FileNotFoundError(
                f"binary for index {index} not found (have {len(built)})"
            )
        return built[index][2]
