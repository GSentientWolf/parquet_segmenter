import tempfile
import os

import pandas as pd

from parquet_segmenter.testing import generate_and_write_parquet


def test_generate_and_write_parquet_roundtrip():
    tmp = tempfile.mkdtemp(prefix="parquet_test_")
    path = os.path.join(tmp, "test.parquet")
    df = generate_and_write_parquet(path, n=50, categories=["a", "b", "c"]) 
    # Read back
    read = pd.read_parquet(path, engine="pyarrow")
    # Basic checks
    assert len(read) == 50
    assert set(read.columns) >= {"id", "ints", "normals"}

    # cats column may become object dtype on roundtrip; check values set
    if "cats" in read.columns:
        assert set(read["cats"].astype(str).unique()) <= {"a", "b", "c"}
