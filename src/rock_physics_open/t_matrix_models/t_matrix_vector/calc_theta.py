import numpy as np


def calc_theta_vec(v, omega, gamma, tau, kd_uuvv, dr, k, kappa, kappa_f):
    """Returns the theta tensor (6x6 matrix) for more explanation see e.g.
    Agersborg et al. 2009 or "The effects of drained and undrained loading in
    visco-elsatic waves in rock-like composites" M. Jakobsen and T.A. Johansen.
    (2005). Int. J. Solids and Structures (42). p. 1597-1611.

    Parameters
    ----------
    v  : np.ndarray
        Concentration of all the empty cavities (1x(numbers of empty cavities) vector).
    omega : np.ndarray
        Frequency (2*pi*f).
    gamma : np.ndarray
        Gamma factor of all the inclusions (1x(numbers of empty cavities) vector).
    tau : np.ndarray
        Relaxation time constant (1x(numbers of empty cavities) vector).
    kd_uuvv : np.ndarray
        Tensor sum of the dry K tensor for all the cavities (1x(numbers of empty cavities) vector).
    dr : np.ndarray
        Permeability/viscosity.
    k : np.ndarray
        Wave number vector.
    kappa : np.ndarray
        Bulk modulus of the host material.
    kappa_f : np.ndarray
        Bulk modulus of the fluid.

    Returns
    -------
    np.ndarray
        Theta tensor.
    """
    sigma_a = np.sum((v / (1 + 1j * omega * gamma * tau)), axis=1)
    sigma_b = np.sum((v / (1 + 1j * omega * gamma * tau)) * kd_uuvv, axis=1)

    return kappa_f / (
        (1 - kappa_f / kappa) * sigma_a
        + kappa_f * sigma_b
        - (1j * k * k / omega) * dr * kappa_f
    )
