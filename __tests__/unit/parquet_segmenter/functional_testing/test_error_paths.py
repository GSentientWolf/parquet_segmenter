import runpy
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import pytest
import subprocess
import shlex

from parquet_segmenter.functional_testing.binary_store import PrecalculatedBinaryStore
from parquet_segmenter.functional_testing.parquet_gen import (
    generate_parquet_by_size,
    generate_parquet_with_edge_cases,
)
import pytest


def test_by_size_index_out_of_range_cli(tmp_path, cli_subprocess):
    # empty store, request index 0 -> should raise IndexError inside the CLI
    store_dir = tmp_path / "store_empty"
    out = tmp_path / "out.parquet"
    res = cli_subprocess([
        "by-size",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out),
        "--target",
        "1024",
        "--indices",
        "0",
    ], timeout=2)
    assert res.ret != 0
    # store.get now raises FileNotFoundError for missing indices
    assert "not found" in res.err.lower() or "file not found" in res.err.lower() or "no such file" in res.err.lower()


def test_by_size_missing_args_cli(cli_subprocess):
    # Running by-size with no args should produce a non-zero exit (argparse)
    res = cli_subprocess(["by-size"], timeout=2)
    assert res.ret != 0
    assert "usage" in res.err.lower() or "usage" in res.out.lower()


def test_parse_size_invalid():
    with pytest.raises(ValueError):
        PrecalculatedBinaryStore.parse_size("not-a-size")


def test_generate_by_size_empty_index_list(tmp_path, precalc_store):
    # calling the function directly with empty index_list should raise
    out = tmp_path / "x.parquet"
    store = precalc_store()
    with pytest.raises(ValueError):
        generate_parquet_by_size(str(out), 1024, [], binary_store=store)


def test_cli_invalid_indices_format(tmp_path, cli_subprocess):
    store_dir = tmp_path / "s"
    out = tmp_path / "o.parquet"
    res = cli_subprocess([
        "by-size",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out),
        "--target",
        "1024",
        "--indices",
        "a,b",
    ], timeout=2)
    assert res.ret != 0
    assert "ValueError" in res.err or "invalid" in res.err.lower() or "invalid literal" in res.err.lower()


def test_cli_nonint_target(tmp_path, cli_subprocess):
    store_dir = tmp_path / "s2"
    out = tmp_path / "o2.parquet"
    res = cli_subprocess([
        "by-size",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out),
        "--target",
        "notint",
        "--indices",
        "0",
    ], timeout=2)
    assert res.ret != 0
    assert "ValueError" in res.err or "invalid" in res.err.lower() or "invalid literal" in res.err.lower()


def test_generate_by_size_smaller_than_candidate(tmp_path, precalc_store):
    store = precalc_store()
    store.ensure_size("2 kB")
    out = tmp_path / "small.parquet"
    df, meta = generate_parquet_by_size(str(out), 512, [0], binary_store=store)
    # with one candidate and target smaller than candidate size, seq should be length 1
    assert len(df) == 1
    assert "produced" in meta


def test_generate_parquet_with_edge_cases_counts(tmp_path, precalc_store):
    store = precalc_store()
    store.ensure_size("1 kB")
    store.ensure_size("900 kB")
    out = tmp_path / "edge3.parquet"
    df, meta = generate_parquet_with_edge_cases(str(out), n_rows_mean=20, n_rows_std=2, num_edge_cases=2, binary_store=store)
    assert isinstance(df, object)
    assert "edge_map" in meta
    assert len(meta["edge_map"]) == 2


def test_partial_store_missing_file(tmp_path, cli_subprocess, precalc_store):
    # Create a store with two binaries, then delete one file to simulate a
    # partial store. Both the direct API and the CLI should report a clear
    # error (FileNotFoundError/OSError).
    store_dir = tmp_path / "partial_store"
    store = precalc_store()
    p0 = store.ensure_size("1 kB")
    p1 = store.ensure_size("2 kB")

    # remove the second file to simulate corruption
    import os

    os.remove(p1)

    # Direct call should raise when trying to read sizes/contents
    out = tmp_path / "partial.parquet"
    with pytest.raises(Exception) as exc:
        # pick index 1 which now points to a missing file
        generate_parquet_by_size(str(out), 2048, [1], binary_store=store)
    msg = str(exc.value).lower()
    # the store now deterministically reports missing binaries as FileNotFoundError
    assert "not found" in msg or "file not found" in msg or "no such file" in msg

    # CLI should also exit non-zero and include an error hint
    res = cli_subprocess([
        "by-size",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out),
        "--target",
        "2048",
        "--indices",
        "1",
    ], timeout=2)
    assert res.ret != 0
    stderr = res.err.lower()
    assert (
        "file not found" in stderr
        or "not found" in stderr
        or "no such file" in stderr
        or "file" in stderr
    )
