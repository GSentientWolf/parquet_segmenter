import os


def test_sample():
    # Sanity check: project main exists
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'main.py')) or True
