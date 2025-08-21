import importlib


def test_segmenter_package_importable():
    """Ensure the segmenter package is importable."""
    pkg = importlib.import_module("parquet_segmenter.segmenter")
    assert pkg is not None


def test_base_segmenter_module_importable():
    """Ensure the base_segmenter module is importable."""
    mod = importlib.import_module("parquet_segmenter.segmenter.base_segmenter")
    assert mod is not None
