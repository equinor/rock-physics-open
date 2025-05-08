import numpy as np

from .array_functions import array_matrix_mult
from .check_and_tile import check_and_tile


def calc_x_vec(s_0, td):
    """Returns the x-tensor (6x6x(numbers of empty cavities) matrix) for more explanation see e.g.
    Agersborg et al. 2009 or "The effects of drained and undrained loading in
    visco-elsatic waves in rock-like composites" M. Jakobsen and T.A. Johansen.
    (2005). Int. J. Solids and Structures (42). p. 1597-1611

    Parameters
    ----------
    s_0: np.ndarray
        Inverse of stiffness tensor of the host material (nx6x6 matrix).
    td :  np.ndarray
        Dry t-matrix tensors (nx6x6x(numbers of empty cavities) matrix).

    Returns
    -------
    np.ndarray
        x-tensor.
    """
    i2_i2, log_length, alpha_length = check_and_tile(s_0, td)

    x = np.zeros((log_length, 6, 6, alpha_length))
    for j in range(alpha_length):
        x[:, :, :, j] = array_matrix_mult(
            td[:, :, :, j], s_0, i2_i2, s_0, td[:, :, :, j]
        )

    return x
