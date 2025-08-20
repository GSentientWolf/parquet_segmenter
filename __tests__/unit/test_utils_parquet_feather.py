import pytest
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    # Import the ParquetEngine alias for static type-checkers only. Some
    # pandas installations don't expose this name at runtime in the same
    # place, so keep the import guarded under TYPE_CHECKING.
    from pandas._typing import ParquetEngine  # type: ignore

from parquet_segmenter.utils import utils
import importlib
testing_utils = importlib.import_module("parquet_segmenter.testing_utils")


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.mark.parametrize("engine", ["pyarrow", "fastparquet"])
def test_parquet_roundtrip(tmp_path: Path, engine: str) -> None:
    # Runtime-check the engine package and write using it explicitly so we
    # exercise engine support without type-casts.
    df = _sample_df()
    p = tmp_path / "test.parquet"
    # Use the shared helper to assert the engine is available or skip.
    testing_utils.require_pkg(engine)

    # Now call pandas with the engine name directly (runtime-checked).
    # Cast the runtime string to the ParquetEngine typing alias so static
    # type-checkers accept the call (engine is validated above by
    # testing_utils.require_pkg). Use a string name for the cast target to
    # avoid importing pandas internals at runtime which may not expose the
    # alias in all installations.
    # Use a string-literal target for cast so runtime does not require the
    # ParquetEngine name to be defined; the TYPE_CHECKING import above lets
    # static analyzers still resolve the alias.
    engine_t = cast("ParquetEngine", engine)
    df.to_parquet(p, engine=engine_t)
    read = utils._read_parquet(p)
    assert read is not None
    pdt.assert_frame_equal(df.reset_index(drop=True), read.reset_index(drop=True))


def test_feather_roundtrip(tmp_path: Path) -> None:
    # feather support in pandas requires pyarrow; perform runtime check
    testing_utils.require_pkg("pyarrow")
    df = _sample_df()
    p = tmp_path / "test.feather"
    assert utils._write_feather(df, str(p))
    read = utils._read_feather(p)
    assert read is not None
    pdt.assert_frame_equal(df.reset_index(drop=True), read.reset_index(drop=True))
