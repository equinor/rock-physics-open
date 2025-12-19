import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    Array3D,
    Array4D,
)

from .array_functions import array_matrix_mult
from .check_and_tile import check_and_tile


def calc_z_vec(
    s0: Array3D[np.float64],
    td: Array4D[np.float64],
    td_bar: Array4D[np.float64],
    omega: float,
    gamma: Array2D[np.float64],
    v: Array2D[np.float64],
    tau: Array1D[np.float64],
) -> tuple[Array4D[np.complex128], Array4D[np.complex128]]:
    """Returns the z tensor (6x6x(numbers of empty cavities) matrix) for more explanation see e.g.
    Agersborg et al. 2009 or "The effects of drained and undrained loading in
    visco-elastic waves in rock-like composites" M. Jakobsen and T.A. Johansen.
    (2005). Int. J. Solids and Structures (42). p. 1597-1611.

    Parameters
    ----------
    s0  : np.ndarray
        Stiffness tensor of the host material (6x6 matrix).
    td  : np.ndarray
        Dry t-matrix tensors, (6x6x(numbers of empty cavities) matrix).
    td_bar  : np.ndarray
        Dry t-matrix tensors, (6x6x(numbers of empty cavities) matrix).
    omega : np.ndarray
        Frequency (2*pi*f).
    gamma : np.ndarray
        Gamma factor of all the inclusions (1x(numbers of empty cavities) vector).
    v  :  np.ndarray
        Concentration of all the empty cavities (1x(numbers of empty cavities) vector).
    tau  :  np.ndarray
        Relaxation time constant (1x(numbers of empty cavities) vector).

    Returns
    -------
    tuple
        z, z_bar : (np.ndarray, np.ndarray).
    """
    i2_i2, log_length, alpha_length = check_and_tile(s0, td)

    sum_z = 0.0
    z = np.zeros((log_length, 6, 6, alpha_length), dtype="complex128")
    z_bar = np.zeros((log_length, 6, 6, alpha_length), dtype="complex128")

    for j in range(alpha_length):
        sum_z = sum_z + (td[:, :, :, j] + td_bar[:, :, :, j]) * (
            v[:, j] / (1 + 1j * omega * gamma[:, j] * tau[j])
        ).reshape(log_length, 1, 1)

    for j in range(alpha_length):
        z[:, :, :, j] = array_matrix_mult(
            td[:, :, :, j],
            s0,
            i2_i2,
            s0,
            sum_z,  # pyright: ignore[reportArgumentType] | sum_z is 3D array after summation
        )
        z_bar[:, :, :, j] = array_matrix_mult(
            td_bar[:, :, :, j],
            s0,
            i2_i2,
            s0,
            sum_z,  # pyright: ignore[reportArgumentType] | sum_z is 3D array after summation
        )

    return z, z_bar
