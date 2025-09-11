from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def filter_output(
    idx_inp: npt.NDArray[np.bool_],
    inp_log: list[npt.NDArray[Any] | pd.DataFrame]
    | tuple[npt.NDArray[Any] | pd.DataFrame, ...]
    | npt.NDArray[Any]
    | pd.DataFrame,
) -> list[npt.NDArray[Any] | pd.DataFrame]:
    """
    Function to restore outputs from a plugin to original length and
    with values at correct positions. The logs are assumed to go through
    matching input filtering done by gen_utilities.filter_input_log earlier.

    Parameters
    ----------
    idx_inp: np.ndarray
        boolean array which is True at locations to be filled, length idx_inp is returned length of
        arrays or data frames.
    inp_log: tuple or list or np.ndarray or pd.DataFrame
        input numpy array(s) or pandas data frame(s), in list or tuple that are to be expanded to original
        length.

    Returns
    -------
    return_logs : list
        Expanded inputs.
    """

    def _expand_array(
        idx: npt.NDArray[np.bool_], inp_single_log: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        logs = np.ones(idx.shape, dtype=float) * np.nan
        try:
            logs[idx] = inp_single_log.flatten()
        except ValueError:
            # Assume that the dtype  of the input log is not fit for casting to float, set to object and retry
            logs = logs.astype(object)
            logs[idx] = inp_single_log
        return logs.reshape(idx.shape)

    def _expand_df(idx: npt.NDArray[np.bool_], inp_df: pd.DataFrame) -> pd.DataFrame:
        logs = pd.DataFrame(
            columns=inp_df.columns, index=np.arange(idx.shape[0], dtype=np.intp)
        )
        logs.loc[idx] = inp_df
        return logs

    if not isinstance(inp_log, (list, tuple, np.ndarray, pd.DataFrame)):  # pyright: ignore[reportUnnecessaryIsInstance] | Kept for backward compatibility
        raise ValueError(  # pyright: ignore[reportUnreachable] | Kept for backward compatibility
            "filter_output: unknown input data type: {}".format(type(inp_log))
        )
    if not isinstance(idx_inp, (list, np.ndarray)):  # pyright: ignore[reportUnnecessaryIsInstance] | Kept for backward compatibility
        raise ValueError(  # pyright: ignore[reportUnreachable] | Kept for backward compatibility
            "filter_output: unknown filter array data type: {}".format(type(idx_inp))
        )

    # Make iterable in case of single input
    if isinstance(inp_log, (np.ndarray, pd.DataFrame)):
        inp_log = [inp_log]

    if isinstance(idx_inp, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance] | Kept for backward compatibility
        idx_inp_ = [idx_inp]

    # Possible to simplify?
    if len(idx_inp_) != len(inp_log):
        if len(idx_inp_) == 1:
            idx_inp_ = idx_inp_ * len(inp_log)
        else:
            raise ValueError(
                "filter_output: mismatch between length of filter arrays and inputs: {} and {}".format(
                    len(idx_inp), len(inp_log)
                )
            )

    return_logs: list[npt.NDArray[Any] | pd.DataFrame] = []
    for this_idx, this_log in zip(idx_inp_, inp_log):
        if isinstance(this_log, np.ndarray):
            return_logs.append(_expand_array(this_idx, this_log))
        elif isinstance(this_log, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance] | Kept for backward compatibility
            return_logs.append(_expand_df(this_idx, this_log))

    return return_logs
