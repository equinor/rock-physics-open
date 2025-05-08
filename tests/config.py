import inspect
from pathlib import Path
from typing import Callable

ROOT_DIR = Path(__file__).parent.resolve()
TESTDATA_DIR = ROOT_DIR.joinpath("data")


def get_models_path(plugin_name: Callable) -> Path:
    """
    Get model file directory from a function assuming model files are located in ./models. Check that the directory
    actually exists.

    Parameters
    ----------
    plugin_name : Callable
        Function for which to find the model directory.

    Returns
    -------
    Path
        Path of the models folder for a plugin.
    """
    plugin_file = Path(inspect.getfile(plugin_name))
    try_dir = plugin_file.parent.joinpath("models")
    if not try_dir.exists():
        raise FileNotFoundError(f"{try_dir} does not exist")
    return try_dir
