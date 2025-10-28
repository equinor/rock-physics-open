from pathlib import Path
from shutil import copytree

import pytest

ROOT_DIR = Path(__file__).parent.resolve()
TESTDATA_DIR = ROOT_DIR.joinpath("data")


@pytest.fixture(scope="session")
def testdata() -> Path:
    return TESTDATA_DIR


@pytest.fixture(scope="session", autouse=True, name="data_dir")
def setup_rock_physics_open_test_data(
    testdata: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    start_dir = tmp_path_factory.mktemp("data")

    _ = copytree(testdata, start_dir, dirs_exist_ok=True)

    return start_dir
