import pytest

from parquet_segmenter.utils import chunked


def test_chunked_basic():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunked_size_one():
    assert chunked([1, 2, 3], 1) == [[1], [2], [3]]


def test_chunked_invalid_size():
    with pytest.raises(ValueError):
        chunked([1, 2, 3], 0)
