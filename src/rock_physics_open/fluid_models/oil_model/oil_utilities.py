from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLikeFloat: TypeAlias = npt.NDArray[np.float64] | float


def as_float_array(value: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Ensure that an input will be cast to a numpy array with at least one dimension."""
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def inputs_are_scalar(*values: ArrayLikeFloat) -> bool:
    """Test if all inputs are scalar values."""
    return all(np.asarray(value).ndim == 0 for value in values)
