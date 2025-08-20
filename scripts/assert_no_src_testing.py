"""
Fail-fast script to detect accidental `src/parquet_segmenter/testing` folder.
Return code 1 if the forbidden folder exists.
Run in CI or locally: `python scripts/assert_no_src_testing.py`.
"""
from pathlib import Path
import sys

forbidden = Path("src/parquet_segmenter/testing")
if forbidden.exists():
    print(f"ERROR: forbidden directory exists: {forbidden}")
    sys.exit(1)
print("OK: no forbidden directory found")
sys.exit(0)
