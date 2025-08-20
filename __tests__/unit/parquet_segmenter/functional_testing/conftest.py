import sys
import subprocess
from types import SimpleNamespace
import pytest


# in-process runner removed; prefer using subprocess-based `cli_subprocess`


@pytest.fixture
def cli_subprocess():
    """Return a callable that runs the CLI in a subprocess with a timeout.

    Usage: res = cli_subprocess(args, timeout=5)
    Returns SimpleNamespace with ret, out, err, timeout (bool).
    """

    def _run(args: list[str], timeout: int = 5):
        cmd = [sys.executable, "-m", "parquet_segmenter.cli"] + args
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            res = SimpleNamespace()
            res.ret = cp.returncode
            res.out = cp.stdout
            res.err = cp.stderr
            res.timeout = False
            return res
        except subprocess.TimeoutExpired as te:
            res = SimpleNamespace()
            res.ret = -1
            outtxt = te.stdout.decode() if isinstance(te.stdout, (bytes, bytearray)) else (te.stdout or "")
            errtxt = te.stderr.decode() if isinstance(te.stderr, (bytes, bytearray)) else (te.stderr or "")
            res.out = outtxt
            res.err = str(errtxt) + f"\nTimeoutExpired after {timeout}s"
            res.timeout = True
            return res

    return _run


@pytest.fixture
def precalc_store(tmp_path):
    """Return a PrecalculatedBinaryStore factory rooted at a tmp path.

    Call as: store = precalc_store()
    """
    from parquet_segmenter.functional_testing.binary_store import PrecalculatedBinaryStore

    def _make():
        return PrecalculatedBinaryStore(str(tmp_path / "store"))

    return _make
