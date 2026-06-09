from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testdata() -> Path:
    """The full path to the test data directory (`tests/data`), so files in it can be read as input."""
    return Path(__file__).parent.parent / "data"
