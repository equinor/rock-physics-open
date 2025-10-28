import inspect
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

INITIATE = False

SYSTEM_AND_DEBUG_FCN = [
    "pydev",
    "ipython-input",
    "interactiveshell",
    "async_helpers",
    "handle_snapshots",
    "run_code",
    "run_ast_nodes",
    "run_cell_async",
    "_pseudo_sync_runner",
    "_run_cell",
    "run_cell",
    "add_exec",
    "do_add_exec",
    "add_exec",
    "ipython_exec_code",
    "console_exec",
    "do_it",
    "process_internal_commands",
    "_do_wait_suspend",
    "do_wait_suspend",
    "compare_snapshots",
    "handle_snapshots",
    "wrapper",
    "run_tests",
    "<module>",
]


def get_snapshot_name(
    step: int = 1,
    include_filename: bool = True,
    include_function_name: bool = True,
    include_extension: bool = True,
    include_snapshot_dir: bool = True,
) -> str:
    """
    Parameters
    ----------
    step: number of steps in the trace to collect information from
    include_snapshot_dir: absolute directory name included in snapshot name
    include_filename: whether to include filename in snapshot name
    include_function_name: whether to include function name in snapshot name
    include_extension: whether to include extension in snapshot name

    Returns
    -------
    name of snapshot file
    """
    trace = inspect.stack()
    for frame in trace[step:]:
        if not any(keyword in frame.function for keyword in SYSTEM_AND_DEBUG_FCN):
            break
    else:
        frame = trace[step]

    dir_name = Path(frame.filename).parents[1] / "data" / "snapshots"
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


def read_snapshot(snapshot_name: str) -> tuple[npt.NDArray[Any], ...]:
    try:
        with np.load(snapshot_name) as stored_npz:
            return tuple(stored_npz[arr_name] for arr_name in stored_npz.files)
    except IOError as e:
        raise ValueError(f"unable to load snapshot {snapshot_name}: {e}")
