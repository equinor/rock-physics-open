import numpy as np

from .array_functions import array_inverse, array_matrix_mult
from .g_tensor import g_tensor_vec


def calc_kd_vec(c0, i4, s_0, alpha):
    """
    kd is a (nx6x6x(number of alphas)) matrix.

    Parameters
    ----------
    c0 : np.ndarray
        Stiffness tensor of the host material (nx6x6 matrix).
    i4 : np.ndarray
        Array of 6x6 identity matrices.
    s_0:  np.ndarray
        Inverse of stiffness tensor.
    alpha : np.ndarray
        Vector of aspect ratios (1x (number of aspect ratios) vector) or nx(number of (number of alphas)).

    Returns
    -------
    np.ndarray
        kd: stiffness tensor.
    """
    log_length = c0.shape[0]

    if alpha.ndim == 1 and alpha.shape[0] != c0.shape[0]:
        alpha = np.tile(alpha.reshape(1, alpha.shape[0]), (log_length, 1))
    L = alpha.shape[1]
    kd = np.zeros((log_length, 6, 6, L))

    for nc in range(L):
        g = g_tensor_vec(c0, s_0, alpha[:, nc])
        kd[:, :, :, nc] = array_matrix_mult(
            array_inverse(i4 + array_matrix_mult(g, c0)), s_0
        )

    return kd
