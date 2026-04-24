from pathlib import Path
from shutil import copytree

import pytest


@pytest.fixture(scope="session")
def testdata(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src_testdata_folder = Path(__file__).parent.parent / "data"  # tests/data
    tmp_testdata_folder = tmp_path_factory.mktemp("testdata")
    return copytree(src_testdata_folder, tmp_testdata_folder, dirs_exist_ok=True)
