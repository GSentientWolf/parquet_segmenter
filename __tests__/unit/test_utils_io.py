import pickle
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from parquet_segmenter.utils import utils


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_write_and_read_pickle(tmp_path: Path) -> None:
    df = _sample_df()
    p = tmp_path / "test.pkl"
    assert utils._write_pickle(df, str(p))
    read = utils._read_pickle(p)
    assert read is not None
    pdt.assert_frame_equal(df.reset_index(drop=True), read.reset_index(drop=True))


def test_write_and_read_csv(tmp_path: Path) -> None:
    df = _sample_df()
    p = tmp_path / "test.csv"
    assert utils._write_csv(df, str(p))
    read = utils._read_csv(p)
    assert read is not None
    # CSV read may infer types differently; coerce for comparison
    read["a"] = read["a"].astype(int)
    pdt.assert_frame_equal(df.reset_index(drop=True), read.reset_index(drop=True))
