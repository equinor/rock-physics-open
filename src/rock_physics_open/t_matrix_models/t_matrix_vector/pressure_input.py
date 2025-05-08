import numpy as np

from .array_functions import array_inverse, array_matrix_mult
from .g_tensor import g_tensor_vec


def pressure_input_utility(k_min, mu_min, log_length):
    """Utility that calculates some of the elastic properties needed for T-Matrix pressure estimation.

    Parameters
    ----------
    k_min : np.ndarray
        Effective mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Effective mineral shear modulus [Pa].
    log_length : int
        Number of samples in logs.

    Returns
    -------
    tuple
        c0 (background stiffness), s0 (inverse of background stiffness), gd (G tensor for inclusions with aspect
        ratio 1.0).
    """
    # Calculate elastic parameters
    c11 = k_min + 4 / 3 * mu_min
    c44 = mu_min
    c12 = c11 - 2 * c44

    i4 = np.tile(np.eye(6).reshape(1, 6, 6), (log_length, 1, 1))
    c0 = np.zeros((log_length, 6, 6))

    for i in range(3):
        c0[:, i, i] = c11
        c0[:, i + 3, i + 3] = 2 * c44
    for i in range(2):
        c0[:, 0, i + 1] = c12
        c0[:, 1, 2 * i] = c12
        c0[:, 2, i] = c12

    s0 = array_matrix_mult(i4, array_inverse(c0))
    gd = g_tensor_vec(c0, s0, 1.0)

    return c0, s0, gd
