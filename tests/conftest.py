from pathlib import Path
from shutil import copytree

import pytest


@pytest.fixture(scope="session")
def testdata() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True, name="data_dir")
def setup_rock_physics_open_test_data(testdata, tmp_path_factory):
    start_dir = tmp_path_factory.mktemp("data")

    copytree(testdata, start_dir, dirs_exist_ok=True)

    return start_dir
