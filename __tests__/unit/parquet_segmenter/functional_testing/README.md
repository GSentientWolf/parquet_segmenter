Unit test fixtures and guidelines

This short README explains the test fixtures used in this folder and recommended timeout values for CLI tests.

Fixtures

- `cli_subprocess`
  - Preferred fixture for running CLI commands from tests.
  - Runs the package CLI in a separate subprocess and returns a small result namespace with attributes:
    - `ret` (int): the process exit code
    - `out` (str): combined stdout text
    - `err` (str): combined stderr text
    - `timeout` (bool): True if the runner killed the process due to a timeout
  - Usage: call `cli_subprocess(["<subcommand>", "--opt", "value"], timeout=SECONDS)` inside your test.
  - Recommended timeout policy:
    - 2 seconds for quick commands (e.g. `list`, short queries)
    - 5 seconds for generator or file-creating commands (e.g. `edge`, `by-size`, `simple`)
    - Increase only when strictly necessary; prefer smaller inputs or mocking for CI.

- `precalc_store`
  - Factory that creates a `PrecalculatedBinaryStore` rooted under `tmp_path / "store"`.
  - Use this to pre-populate deterministic binary files used by Parquet generators.

Notes and conventions

- The earlier in-process helper `cli_runner` has been removed from the test helpers to avoid mixing APIs.
- Subprocess results do not expose `.outlines` / `.errlines`; use `.out` and `.err` (strings). If you need a list of lines, split on `"\n"`.
- Prefer `cli_subprocess` for any test that may produce files, perform heavy I/O, or could hang — the timeout protects CI.
- If you require an in-process runner for very fast, isolated unit tests, add a clearly named fixture (e.g. `cli_inprocess_fast`) and keep its API distinct.

Example

```python
# Example pytest test using the `cli_subprocess` fixture.
def test_cli_example(tmp_path, cli_subprocess):
    store_dir = tmp_path / "store"
    out_file = tmp_path / "out.parquet"

    # Run a quick `list` command (fast, so short timeout)
    res = cli_subprocess(["list", "--store-dir", str(store_dir)], timeout=2)
    assert res.ret == 0
    # `res.out` is a string; split into lines if you need to inspect them
    lines = res.out.strip().split("\n") if res.out else []

    # Run a heavier generator command; allow more time
    r2 = cli_subprocess([
        "edge",
        "--store-dir",
        str(store_dir),
        "--out",
        str(out_file),
        "--rows-mean",
        "10",
        "--rows-std",
        "1",
        "--num-edge-cases",
        "1",
    ], timeout=5)
    assert r2.ret == 0
    assert out_file.exists()
```
