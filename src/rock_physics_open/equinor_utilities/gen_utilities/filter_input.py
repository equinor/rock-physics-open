from sys import byteorder

import numpy as np
import pandas as pd

WRONG_BYTEORDER = ">" if byteorder == "little" else "<"


def filter_input_log(
    args, working_int=None, negative=False, no_zero=False, positive=True
):
    """
    Check for valid input values in numpy arrays or pandas data frames. Default behaviour is to
    identify missing values - assumed to be NaN and Inf. Other conditions
    can be stated in the key word arguments. Unknown conditions are ignored and a warning
    is issued. Run dim_check_vector to make sure that all inputs have the same length.
    Erroneous values in a sample in one log will remove the sample from all the logs.
    All inputs must have the same array length (data frames the same number of indices).

    Parameters
    ----------
    args : list or tuple or np.ndarray or pd.DataFrame
        Inputs to be filtered, single array or dataframe or lists of arrays or data frames.
    working_int : np.ndarray
        Valid positions are shown as values > 0.
    negative : bool
        Positive values are excluded (zero values are retained).
    no_zero : bool
        Zero values are excluded.
    positive : bool
        Negative values are excluded.

    Returns
    -------
    tuple
        idx, output_args : (np.ndarray, list)
        indices of valid values [bool],
        list of input arrays at valid indices.
    """
    type_error = "filter_input_log: unknown input data type: {}".format(type(args))
    size_error = "filter_input_log: inputs of different length"

    if not isinstance(args, (list, tuple, np.ndarray, pd.DataFrame)):
        raise ValueError(type_error)

    # Make sure that 'args' is iterable
    if isinstance(args, (np.ndarray, pd.DataFrame)):
        args = [args]

    # Input tuple
    if isinstance(args, tuple):
        args = list(args)

    # Need to preserve original inputs
    input_args = args.copy()

    # Test that inputs are of the right types and the same length
    if not np.all([isinstance(log, (np.ndarray, pd.DataFrame)) for log in args]):
        raise ValueError(type_error)
    if not np.all([log.shape[0] == args[0].shape[0] for log in args]):
        raise ValueError(size_error)

    # Generate pandas series from numpy arrays
    args = [pd.Series(log) if isinstance(log, np.ndarray) else log for log in args]
    # Merge into a data frame
    logs = pd.concat(args, axis=1)

    # If any of the input logs are of type boolean, False means that they should not be included,
    # regardless of filter flags
    # https://github.com/pandas-dev/pandas/issues/32432
    # idx = ~logs.any(bool_only=True, axis=1)
    # Need to do it the cumbersome way for the time being
    bool_col = logs.dtypes.apply(lambda dtype: dtype == "bool")
    if any(bool_col):
        idx = ~logs.loc[:, logs.columns[bool_col]].any(axis=1)
        logs.drop(columns=logs.columns[bool_col], inplace=True)
    else:
        idx = pd.Series(index=logs.index, data=np.zeros_like(logs.index).astype(bool))

    # Standard checks: NaN and Inf
    idx = np.logical_or(idx, logs.isna().any(axis=1))
    idx = np.logical_or(idx, logs.isin([np.inf, -np.inf]).any(axis=1))

    # Remove columns with dtype that is not numeric
    obj_col = [dd.kind not in ["i", "u", "f", "c"] for dd in logs.dtypes]
    logs.drop(columns=logs.columns[obj_col], inplace=True)

    # Checks according to the input options input_dict
    # Only consider working interval if it is included or set to some value
    if working_int is not None and not np.all(working_int == 0):
        idx = np.logical_or(idx, working_int == 0)
    if negative:
        # noinspection PyTypeChecker
        idx = np.logical_or(idx, (logs >= 0.0).all(axis=1))
        # idx = np.logical_or(idx, logs.loc[logs > 0.0]).any(axis=1)
    if no_zero:
        idx = np.logical_or(idx, (logs == 0.0).any(axis=1))
    if positive:
        # noinspection PyTypeChecker
        idx = np.logical_or(idx, (logs < 0.0).any(axis=1))

    # Negate idx to identify samples to retain
    idx = np.logical_not(idx)
    num_valid_samples = idx.sum()
    if num_valid_samples == 0:
        raise ValueError("No acceptable input values")
    for i in range(len(input_args)):
        if isinstance(input_args[i], np.ndarray):
            input_args[i] = input_args[i][idx]
        else:  # data frame
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#byte-ordering-issues

            check_type = (
                np.array([col_type.byteorder for col_type in input_args[i].dtypes])
                == WRONG_BYTEORDER
            )
            if np.any(check_type):
                tmp_array = (
                    input_args[i].to_numpy().byteswap().newbyteorder().astype(float)
                )
                cols = input_args[i].columns
                for j in range(check_type.shape[0]):
                    if check_type[j]:
                        input_args[i][cols[j]] = tmp_array[:, j]
            input_args[i] = input_args[i].loc[idx]
    return np.array(idx), input_args
