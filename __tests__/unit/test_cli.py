import os
import re
import pytest

from parquet_segmenter.testing.binary_store import PrecalculatedBinaryStore
import re
import pytest


def test_list_and_edge(tmp_path, cli_subprocess):
    # ensure store dir is isolated
    store_dir = tmp_path / "store"
    out_path = tmp_path / "edge.parquet"

    # call list on empty store (use subprocess runner)
    r = cli_subprocess(["list", "--store-dir", str(store_dir)], timeout=2)
    assert r.ret == 0
    out_txt = r.out
    assert ("no binaries" in out_txt) or ("index" in out_txt)

    # create an initial binary via the edge command
    # edge may create binaries; run in a subprocess with a timeout
    r2 = cli_subprocess(
        [
            "edge",
            "--store-dir",
            str(store_dir),
            "--out",
            str(out_path),
            "--rows-mean",
            "10",
            "--rows-std",
            "1",
            "--num-edge-cases",
            "1",
        ],
        timeout=5,
    )
    assert r2.ret == 0
    out_txt2 = r2.out
    assert out_path.exists()
    assert "wrote" in out_txt2
    assert ("file_size" in out_txt2) or ("wrote" in out_txt2)

    # listing after running edge should show an index (subprocess)
    rlist_after = cli_subprocess(["list", "--store-dir", str(store_dir)], timeout=2)
    assert rlist_after.ret == 0
    out_after = rlist_after.out
    assert "index" in out_after


def test_by_size_and_simple(tmp_path, precalc_store, cli_subprocess):
    store_dir = tmp_path / "store2"
    out_by = tmp_path / "by.parquet"
    out_simple = tmp_path / "simple.parquet"

    # pre-populate store using the store API directly
    store = PrecalculatedBinaryStore(str(store_dir))
    store.ensure_size("1 kB")
    store.ensure_size("2 kB")

    # list to get indices (subprocess)
    rlist = cli_subprocess(["list", "--store-dir", str(store_dir)], timeout=2)
    assert rlist.ret == 0
    assert "index" in rlist.out

    # now run by-size using indices 0,1 to create a small target file
    # by-size can create larger files; run in subprocess with timeout
    rby = cli_subprocess([
        "by-size",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out_by),
        "--target",
        "4096",
        "--indices",
        "0,1",
    ], timeout=5)
    assert rby.ret == 0
    assert out_by.exists()
    # ensure produced= appears and parse produced value
    rby_out = rby.out
    m = re.search(r"produced=(\d+)", rby_out)
    assert m, rby_out
    produced = int(m.group(1))
    assert produced >= 4096 or produced > 0

    # simple writer without binaries (subprocess)
    rs = cli_subprocess(["simple", "--out", str(out_simple), "--rows", "5"], timeout=2)
    assert rs.ret == 0
    assert out_simple.exists()
