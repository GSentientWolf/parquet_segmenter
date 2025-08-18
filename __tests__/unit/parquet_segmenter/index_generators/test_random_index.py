import pytest

from parquet_segmenter.index_generators import generate_random_indices


def test_generate_random_indices_basic():
    seq = generate_random_indices(5, seed=42)
    assert len(seq) == 5
    assert set(seq) == set(range(5))
    # deterministic: repeated call with same seed yields same ordering
    assert seq == generate_random_indices(5, seed=42)


def test_generate_random_indices_zero():
    assert generate_random_indices(0, seed=0) == []


def test_generate_random_indices_negative():
    with pytest.raises(ValueError):
        generate_random_indices(-1)
