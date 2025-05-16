import inspect
from pathlib import Path

import numpy as np

INITIATE = False


def get_snapshot_name(step: int = 1, include_snapshot_dir=True) -> str:
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
            keyword in frame.filename
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
    file_name = f"{Path(frame.filename).stem}_{frame.function}.npz"
    return str(dir_name / file_name) if include_snapshot_dir else file_name


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
