import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array2D,
    Array3D,
    Array4D,
)

from .array_functions import array_inverse, array_matrix_mult
from .g_tensor import g_tensor_vec


def calc_td_vec(
    c0: Array3D[np.float64],
    i4: Array3D[np.float64],
    s_0: Array3D[np.float64],
    kd: Array4D[np.float64],
    alpha: Array2D[np.float64],
) -> Array4D[np.float64]:
    """Returns the dry t-matrix tensors (nx6x6x(numbers of empty cavities) matrix).

    Parameters
    ----------
    c0 : np.ndarray
        Stiffness tensor of the host material (nx6x6 matrix).
    i4 : np.ndarray
        An n x Identity matrix.
    s_0 : np.ndarray
        Inverse of stiffness tensor.
    kd : np.ndarray
        Dry K tensor of all the empty cavities (nx6x6x(numbers of empty cavities) matrix) see Agersborg et al.
        2009 for explanation.
    alpha : np.ndarray
        Aspect ratios of all the empty cavities (1x(numbers of empty cavities) vector).

    Returns
    -------
    np.ndarray
        Dry t-matrix tensors.
    """
    log_len = c0.shape[0]
    if alpha.ndim == 1 and alpha.shape[0] != c0.shape[0]:
        alpha = np.tile(alpha.reshape(1, alpha.shape[0]), (log_len, 1))
    alpha_len = alpha.shape[1]

    td = np.array(np.zeros((log_len, 6, 6, alpha_len)))

    for nc in range(alpha_len):
        g = g_tensor_vec(
            c0=c0,
            s_0=s_0,
            alpha=alpha[:, nc],
        )
        td[:, :, :, nc] = array_matrix_mult(
            array_inverse(g), array_matrix_mult(kd[:, :, :, nc], c0) - i4
        )

    return td
