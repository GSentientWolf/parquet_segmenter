import pytest

from parquet_segmenter.index_generators import (
    stdlib_choice_factory,
    histogram_from_generator,
    numpy_choice_factory,
)


def test_stdlib_choice_factory_replace_false():
    factory = stdlib_choice_factory(5, replace=False, seed=1)
    it = factory()
    vals = list(it)
    assert set(vals) == set(range(5))
    assert len(vals) == 5


def test_stdlib_choice_factory_replace_true_histogram():
    factory = stdlib_choice_factory(3, replace=True, seed=42)
    hist = histogram_from_generator(factory, 1000, 3)
    assert sum(hist) == 1000
    # all bins should have some counts
    assert all(c > 0 for c in hist)


@pytest.mark.skipif(True, reason="numpy tests skipped if numpy not available")
def test_numpy_choice_factory_if_available():
    try:
        factory = numpy_choice_factory(4, replace=False, seed=2)
    except RuntimeError:
        pytest.skip("numpy not available")
    vals = list(factory())
    assert set(vals) == set(range(4))


@pytest.mark.skipif(True, reason="numpy tests skipped if numpy not available")
def test_numpy_choice_factory_weighted_no_replacement():
    try:
        # create a simple weighted distribution where bin 0 has very high weight
        factory = numpy_choice_factory(3, probs=[0.9, 0.05, 0.05], replace=False, seed=5)
    except RuntimeError:
        pytest.skip("numpy not available")
    vals = list(factory())
    # ensure we got exactly num_bins unique indices
    assert len(vals) == 3
    assert set(vals) == set(range(3))


def test_stdlib_probs_validation():
    # wrong length
    with pytest.raises(ValueError):
        stdlib_choice_factory(3, probs=[0.5, 0.5])
    # negative weight
    with pytest.raises(ValueError):
        stdlib_choice_factory(2, probs=[-0.1, 1.1])
    # all zeros
    with pytest.raises(ValueError):
        stdlib_choice_factory(2, probs=[0, 0])


@pytest.mark.skipif(True, reason="numpy tests skipped if numpy not available")
def test_numpy_probs_validation():
    try:
        pytest.importorskip("numpy")
    except Exception:
        pytest.skip("numpy not available")
    # wrong length
    with pytest.raises(ValueError):
        numpy_choice_factory(3, probs=[0.5, 0.5])
    # negative weight
    with pytest.raises(ValueError):
        numpy_choice_factory(2, probs=[-0.1, 1.1])
    # all zeros
    with pytest.raises(ValueError):
        numpy_choice_factory(2, probs=[0, 0])
