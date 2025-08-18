import os
import tempfile

import pandas as pd

from parquet_segmenter.testing import generate_and_write_parquet
def test_generate_with_nearest(tmp_path, precalc_store):
    store = precalc_store()
    store.ensure_size("5 kB")
    store.ensure_size("10 kB")
    store.ensure_size("50 kB")
    out = tmp_path / "nearest.parquet"
    df = generate_and_write_parquet(str(out), n=10, binary_store=store, blob_strategy="nearest", target_file_size_bytes=100000)
    # ensure blob column exists
    assert "blob" in df.columns
    # the file should exist on disk
    assert os.path.exists(str(out))


def test_generate_with_index(tmp_path, precalc_store):
    store = precalc_store()
    store.ensure_size("5 kB")
    store.ensure_size("10 kB")
    store.ensure_size("50 kB")
    out = tmp_path / "index.parquet"
    df = generate_and_write_parquet(str(out), n=6, binary_store=store, blob_strategy="index", target_file_size_bytes=50000)
    assert "blob" in df.columns
    assert os.path.exists(str(out))
