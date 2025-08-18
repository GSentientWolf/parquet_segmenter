import os
import pytest

from parquet_segmenter.testing.binary_store import PrecalculatedBinaryStore


def test_parse_size_basic():
    assert PrecalculatedBinaryStore.parse_size("10 kB") == 10 * 1024
    assert PrecalculatedBinaryStore.parse_size("1.3 MB") == int(round(1.3 * 1024 ** 2))
    assert PrecalculatedBinaryStore.parse_size("1024 B") == 1024
    with pytest.raises(ValueError):
        PrecalculatedBinaryStore.parse_size("not a size")


def test_ensure_size_and_index(tmp_path, precalc_store):
    store = precalc_store()

    p10 = store.ensure_size("10 kB")
    assert os.path.exists(p10)

    p5 = store.ensure_size("5 kB")
    assert os.path.exists(p5)

    # list_index should return entries sorted by size ascending
    idx = store.list_index()
    assert len(idx) >= 2
    # global index 0 should be the 5 kB file
    assert idx[0][1] == PrecalculatedBinaryStore.parse_size("5 kB")
    assert os.path.exists(idx[0][2])

    # get should retrieve by global index
    p = store.get(0)
    assert os.path.exists(p)
