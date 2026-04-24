import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array3D,
    ComplexOrFloat,
)


def array_inverse(a: Array3D[ComplexOrFloat]) -> Array3D[ComplexOrFloat]:
    """Inverse of higher order array (3 dim) using linalg inv routine.

    Parameters
    ----------
    a
        An nxmxm numpy array.

    Returns
    -------
    An nxmxm array where [i, :, :] contains the inverse of A[i, :, :].

    Raises
    ------
    ValueError
        If input of wrong shape.
    """
    if not (a.ndim == 3 and a.shape[1] == a.shape[2]):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")
    res = np.empty_like(a)
    res[:] = np.linalg.inv(a)
    return res


def array_matrix_mult(
    *args: Array3D[ComplexOrFloat],
) -> Array3D[ComplexOrFloat]:
    """3-dim arrays are matrix multiplied args[j][i, :, :] @ args[j+1][i, :, :].

    Parameters
    ----------
    *args
        Numpy arrays of shape nxmxm.

    Returns
    -------
    An nxmxm array with n args[i] @ args[i+1] @ ....

    Raises
    ------
    ValueError
        If there is a mismatch in input shape/dimension.
    """
    if len(args) == 0:
        raise ValueError(f"{__name__}: no input arguments provided")
    if not len(args) > 1:
        return args[0]

    arg_shape: list[tuple[int, ...]] = []
    for i in range(len(args)):
        if not (args[i].ndim == 3 and args[i].shape[1] == args[i].shape[2]):
            raise ValueError(
                f"{__name__}: mismatch in inputs variables dimension/shape"
            )
        arg_shape.append(args[i].shape)
    if not np.all(np.array(arg_shape)[:, 0] == np.array(arg_shape)[0, 0]):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    x = args[0]

    for i in range(1, len(args)):
        x = np.einsum("...ij,...jk->...ik", x, args[i])

    return x
