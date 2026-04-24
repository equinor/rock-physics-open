from typing import Any, overload

import numpy as np
import numpy.typing as npt
import pandas as pd


@overload
def dim_check_vector(
    args: tuple[npt.NDArray[np.float64] | float | int, ...]
    | list[npt.NDArray[np.float64] | float | int],
    force_type: np.dtype[np.float64] | None = ...,
) -> list[npt.NDArray[np.float64]]: ...


@overload
def dim_check_vector(
    args: list[Any] | tuple[Any, ...],
    force_type: np.dtype | None = ...,
) -> list[Any]: ...


@overload
def dim_check_vector(
    args: pd.DataFrame,
    force_type: np.dtype | None = ...,
) -> pd.DataFrame: ...


@overload
def dim_check_vector(
    args: npt.NDArray[Any],
    force_type: np.dtype | None = ...,
) -> npt.NDArray[Any]: ...


def dim_check_vector(
    args: list[Any] | tuple[Any, ...] | npt.NDArray[Any] | pd.DataFrame,
    force_type: np.dtype | None = None,
) -> npt.NDArray[Any] | pd.DataFrame | list[Any]:
    """
    Check that all inputs are of the same (one-dimensional) size.

    All inputs will be checked and possibly expanded to common length. Only the first dimension
    is harmonised.

    Parameters
    ----------
    args
        Input list or tuple of scalars, numpy arrays or pandas data frames of numerical or boolean type.
    force_type
        Force all outputs to be of a specific dtype.

    Returns
    -------
    List of inputs where all are of the same length.

    Raises
    ------
    ValueError
        If `force_type` conversion fails or if inputs have incompatible lengths.
    """
    single_types = (np.ndarray, pd.DataFrame)

    # Single array or dataframe is just returned
    if isinstance(args, single_types):
        if force_type is not None:
            try:
                args = args.astype(force_type)
            except ValueError:
                raise ValueError(
                    "dim_check_vector: not possible to force dtype to {}".format(
                        force_type
                    )
                )
        return args

    # If any input is a scalar, make it into an array
    if force_type is not None:
        try:
            args = [
                np.array(item, ndmin=1, dtype=force_type)
                if np.isscalar(item)
                else item.astype(force_type)
                for item in args
            ]
        except ValueError:
            raise ValueError(
                "dim_check_vector: not possible to force dtype to {}".format(force_type)
            )
    else:
        args = [np.array(item, ndmin=1) if np.isscalar(item) else item for item in args]

    # Can now test for length - must either be a scalar or have the same length
    max_length: int = np.max([item.shape[0] for item in args])
    if not np.all([item.shape[0] == max_length or item.shape[0] == 1 for item in args]):
        raise ValueError(
            "dim_check_vector: Unequal array lengths in input to dim_check_vector"
        )

    output_arg: list[npt.NDArray[Any] | pd.DataFrame] = []
    for item in args:
        if item.shape[0] == max_length:
            output_arg.append(item)
        else:
            item_dim = item.ndim
            repeat_tuple = tuple([max_length] + [1] * (item_dim - 1))
            if isinstance(item, pd.DataFrame):
                output_arg.append(
                    pd.DataFrame(
                        np.tile(np.array(item), repeat_tuple),
                        columns=item.columns,
                        index=np.arange(max_length),
                    )
                )
            else:
                output_arg.append(np.tile(item, repeat_tuple))

    return output_arg
