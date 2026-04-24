from pathlib import Path

# load and use fixtures from the `fixtures` directory
pytest_plugins = [
    "tests.fixtures.snapshot",
    "tests.fixtures.testdata",
]

TESTDATA_DIR = Path(__file__).parent / "data"
