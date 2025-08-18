import os

from parquet_segmenter.testing.parquet_gen import generate_parquet_with_edge_cases


def test_generate_edge_cases_basic(tmp_path):
    out = tmp_path / "edge.parquet"
    df, meta = generate_parquet_with_edge_cases(str(out), n_rows_mean=50, n_rows_std=10, num_edge_cases=2)
    # file created
    assert os.path.exists(str(out))
    # edge map should exist and map to integer byte sizes
    assert "edge_map" in meta
    for k, v in meta["edge_map"].items():
        assert isinstance(k, int)
        assert isinstance(v, int)
        assert v >= 900 * 1024 or v <= 2 * 1024 * 1024  # roughly in expected ranges
