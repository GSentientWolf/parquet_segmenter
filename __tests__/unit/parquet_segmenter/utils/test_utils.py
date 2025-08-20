import pandas as pd
import pytest

from parquet_segmenter.utils import faked_memory_usage, patch_memory_usage
from parquet_segmenter.utils import save_dataframe_best
from parquet_segmenter.utils import read_dataframe_best


def test_faked_memory_usage_computes_total():
    df = pd.DataFrame({
        "a": range(3),
        "index_size_bytes": [10, 20, 30],
    })

    col_bytes = int(df.memory_usage(index=False, deep=True).sum())
    expected = col_bytes + sum(df["index_size_bytes"].tolist())

    assert faked_memory_usage(df, index_sizes_col="index_size_bytes") == expected


def test_patch_memory_usage_context_manager_restores_original():
    df = pd.DataFrame({"x": [1, 2], "index_size_bytes": [5, 7]})

    # baseline index memory (may be present or zero)
    baseline = df.memory_usage(index=True, deep=True)
    orig_index = int(baseline.loc["Index"]) if "Index" in baseline.index else 0

    with patch_memory_usage("index_size_bytes"):
        patched = df.memory_usage(index=True, deep=True)
        assert "Index" in patched.index
        assert int(patched.loc["Index"]) == sum(df["index_size_bytes"].tolist())

    # after context the original behavior should be restored
    after = df.memory_usage(index=True, deep=True)
    after_index = int(after.loc["Index"]) if "Index" in after.index else 0
    assert after_index == orig_index


def test_save_dataframe_best_writes_pickle(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = save_dataframe_best(df, str(tmp_path / "out.pkl"))
    # ensure file exists and is readable by pandas
    assert out is not None
    read = pd.read_pickle(out)
    assert read.equals(df)


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # type: ignore
        return True
    except ImportError:
        try:
            import fastparquet  # type: ignore
            return True
        except ImportError:
            return False


@pytest.mark.skipif(not _has_parquet_engine(), reason="no parquet engine installed")
def test_save_dataframe_best_parquet_with_compression(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = save_dataframe_best(df, str(tmp_path / "out.parquet"), compression="gzip")
    assert out is not None
    read = pd.read_parquet(out)
    assert read.equals(df)


def test_read_dataframe_best_roundtrip_pickle(tmp_path):
    df = pd.DataFrame({"a": [4, 5, 6]})
    out = save_dataframe_best(df, str(tmp_path / "round.pkl"))
    read = read_dataframe_best(str(tmp_path / "round.pkl"))
    assert read.equals(df)


@pytest.mark.skipif(not _has_parquet_engine(), reason="no parquet engine installed")
def test_read_dataframe_best_roundtrip_parquet_compressed(tmp_path):
    df = pd.DataFrame({"a": [7, 8, 9]})
    out = save_dataframe_best(df, str(tmp_path / "round.parquet"), compression="gzip")
    read = read_dataframe_best(str(tmp_path / "round.parquet"))
    assert read.equals(df)
