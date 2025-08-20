"""Repo scripts exported as console entry points.

Provide a small function to assert the forbidden folder doesn't exist.
"""

from pathlib import Path


def assert_no_src_testing_main() -> None:
    """Assert the forbidden testing folder does not exist.

    This simple script is used as a console entry point to help CI detect
    accidental commits of a `src/parquet_segmenter/testing` directory.
    """
    forbidden = Path("src/parquet_segmenter/testing")
    if forbidden.exists():
        print(f"ERROR: forbidden directory exists: {forbidden}")
        raise SystemExit(1)
    print("OK: no forbidden directory found")


if __name__ == "__main__":
    assert_no_src_testing_main()
