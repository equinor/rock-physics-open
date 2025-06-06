import inspect
from pathlib import Path

import numpy as np

INITIATE = False


def get_snapshot_name(
    step: int = 1,
    include_filename: bool = True,
    include_function_name: bool = True,
    include_extension: bool = True,
    include_snapshot_dir=True,
) -> str:
    """
    Parameters
    ----------
    step: number of steps in the trace to collect information from
    include_snapshot_dir: absolute directory name included in snapshot name

    Returns
    -------
    name of snapshot file
    """
    trace = inspect.stack()
    for frame in trace[step:]:
        if not any(
            keyword in frame.function
            for keyword in [
                "pydev",
                "ipython-input",
                "interactiveshell",
                "async_helpers",
            ]
        ):
            break
    else:
        frame = trace[step]

    dir_name = Path(frame.filename).parent / "snapshots"
    file_name = Path(frame.filename).stem if include_filename else ""
    function_name = frame.function if include_function_name else ""
    extension = ".npz" if include_extension else ""
    parts = [part for part in [file_name, function_name] if part]
    base_name = "_".join(parts) + extension
    return str(dir_name / base_name) if include_snapshot_dir else base_name


def store_snapshot(snapshot_name: str, *args: np.ndarray) -> bool:
    """
    Examples
    --------
    In case there are multiple arrays to store:
    store_snapshot(snapshot_name='snap_to_store.npz', *args)

    Important: If there is only one array to store:
    store_snapshot(snapshot_name='snap_to_store.npz', args)
    """
    try:
        np.savez(snapshot_name, *args)
    except IOError as e:
        raise IOError(f"Could not store snapshot {snapshot_name}: {e}")
    return True


def read_snapshot(snapshot_name: str) -> tuple:
    try:
        with np.load(snapshot_name) as stored_npz:
            return tuple(stored_npz[arr_name] for arr_name in stored_npz.files)
    except IOError as e:
        raise ValueError(f"unable to load snapshot {snapshot_name}: {e}")
