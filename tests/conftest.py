import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def testdata() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True, name="data_dir")
def setup_rock_physics_open_test_data(testdata, tmp_path_factory):
    start_dir = tmp_path_factory.mktemp("data")

    copy_files = testdata.glob("*")
    for file in copy_files:
        shutil.copy2(Path(testdata) / file.name, start_dir / file.name)

    return start_dir
