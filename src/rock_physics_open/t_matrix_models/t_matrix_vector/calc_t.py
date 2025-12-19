import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    Array3D,
    Array4D,
)


def calc_t_vec(
    td: Array4D[np.float64],
    theta: Array3D[np.complex128],
    x: Array4D[np.float64],
    z: Array4D[np.complex128],
    omega: float,
    gamma: Array2D[np.float64],
    tau: Array1D[np.float64],
    k_fluid: Array1D[np.float64],
) -> Array4D[np.complex128]:
    """
    Returns the t-matrices (6x6x(numbers of connected pores)) of the
    connected pores.

    Parameters
    ----------
    td : np.ndarray
        Dry t-matrix tensors, (nx6x6x(numbers of empty cavities) matrix).
    theta : np.ndarray
        Theta-tensor (nx6x6 matrix).
    x : np.ndarray
        X-tensor (nx6x6x(numbers of empty cavities) matrix).
    z : np.ndarray
        Z-tensor (nx6x6x(numbers of empty cavities) matrix).
    omega : np.ndarray
        Frequency (2*pi*f).
    gamma : np.ndarray
        Gamma factor of all the inclusions (nx(numbers of empty cavities) vector).
    tau : np.ndarray
        Relaxation time constant (1x(numbers of empty cavities) vector).
    k_fluid : np.ndarray
        Bulk modulus of the fluid.

    Returns
    -------
    np.ndarray
        t-matrices.
    """
    if not (
        td.ndim == 4
        and x.ndim == 4
        and z.ndim == 4
        and gamma.ndim == 2
        and k_fluid.ndim == 1
        and theta.ndim == 1
        and np.all(np.array([td.shape[1:2], x.shape[1:2], z.shape[1:2]]) == td.shape[1])
        and np.all(
            np.array(
                [
                    x.shape[0],
                    z.shape[0],
                    gamma.shape[0],
                    theta.shape[0],
                    k_fluid.shape[0],
                ]
            )
            == td.shape[0]
        )
        and np.all(
            np.array([x.shape[3], z.shape[3], tau.shape[0], gamma.shape[1]])
            == td.shape[3]
        )
    ):
        raise ValueError(f"{__name__}: mismatch in inputs dimension/shape")

    log_len = k_fluid.shape[0]
    alpha_len = td.shape[3]

    # Reshape to enable broadcast
    k_fluid_ = k_fluid.reshape((log_len, 1, 1, 1))
    gamma_ = gamma.reshape((log_len, 1, 1, alpha_len))
    tau_ = tau.reshape((1, 1, 1, alpha_len))
    theta_ = theta.reshape((log_len, 1, 1, 1))

    return td + (theta_ * z + 1j * omega * tau_ * k_fluid_ * x) / (
        1 + 1j * omega * gamma_ * tau
    )
