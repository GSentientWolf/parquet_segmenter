import random

from parquet_segmenter.index_generators.random_index import generate_random_indices


def rng_factory(seed: int = 0, upper: int = 100):
    rng = random.Random(seed)

    def _gen():
        while True:
            yield rng.randrange(upper)

    return _gen


def test_generate_random_indices_with_factory():
    gen_factory = rng_factory(seed=123, upper=10)
    vals = list(generate_random_indices(gen_factory, 5))
    assert len(vals) == 5
    # deterministic across repeated calls when same factory seed used
    gen_factory2 = rng_factory(seed=123, upper=10)
    assert vals == list(generate_random_indices(gen_factory2, 5))


def test_generate_random_indices_with_iterator_directly():
    it = iter([0, 1, 2])
    assert list(generate_random_indices(it, 2)) == [0, 1]


def test_generate_random_indices_exhaustion():
    it = iter([1, 2])
    try:
        list(generate_random_indices(it, 3))
        assert False, "expected ValueError due to exhaustion"
    except ValueError:
        pass
