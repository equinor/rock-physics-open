import numpy as np


def calc_t_vec(td, theta, x, z, omega, gamma, tau, k_fluid):
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
    k_fluid = k_fluid.reshape((log_len, 1, 1, 1))
    gamma = gamma.reshape((log_len, 1, 1, alpha_len))
    tau = tau.reshape((1, 1, 1, alpha_len))
    theta = theta.reshape((log_len, 1, 1, 1))

    return td + (theta * z + 1j * omega * tau * k_fluid * x) / (
        1 + 1j * omega * gamma * tau
    )
