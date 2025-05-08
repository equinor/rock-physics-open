import numpy as np


def array_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of higher order array (3 dim) using linalg inv routine.

    Parameters
    ----------
    a : np.ndarray
        An nxmxm numpy array.

    Returns
    -------
    np.ndarray
        An nxmxm array where [i, :, :] contains the inverse of A[i, :, :].

    Raises
    ------
    ValueError
        If input of wrong shape.
    """
    if not (a.ndim == 3 and a.shape[1] == a.shape[2]):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    return np.array([np.linalg.inv(x) for x in a])


def array_matrix_mult(*args: np.ndarray) -> np.ndarray:
    """3-dim arrays are matrix multiplied args[j][i, :, :] @ args[j+1][i, :, :].
    Input args should all be numpy arrays of shape nxmxm.

    Returns
    -------
    np.ndarray
        An nxmxm array with n args[i] @ args[i+1] @ ....

    Raises
    ------
    ValueError
        If input is not a list or tuple.
    ValueError
        If mismatch in input shape/dimension.
    ValueError
        If mismatch in input shape/dimension.
    """
    if not isinstance(args, (list, tuple)):
        raise ValueError(f"{__name__}: inputs must be list or tuple")
    if not len(args) > 1:
        return args[0]

    arg_shape = []
    arg_type = []
    for i in range(len(args)):
        if not (
            isinstance(args[i], np.ndarray)
            and args[i].ndim == 3
            and args[i].shape[1] == args[i].shape[2]
        ):
            raise ValueError(
                f"{__name__}: mismatch in inputs variables dimension/shape"
            )
        arg_shape.append(args[i].shape)
        arg_type.append(args[i].dtype)
    arg_shape = np.array(arg_shape)
    arg_type = np.array(arg_type)
    if not np.all(arg_shape[:, 0] == arg_shape[0, 0]):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")
    out_type = "complex128" if np.any(arg_type == np.dtype("complex128")) else "float64"

    x = args[0]

    for i in range(1, len(args)):
        x = np.einsum("...ij,...jk->...ik", x, args[i], dtype=out_type)

    return x
