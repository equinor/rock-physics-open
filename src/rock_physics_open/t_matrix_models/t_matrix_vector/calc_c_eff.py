import numpy as np

from .array_functions import array_inverse, array_matrix_mult
from .calc_t import calc_t_vec
from .calc_theta import calc_theta_vec
from .calc_z import calc_z_vec


def calc_c_eff_vec(c0, c1, gd, t, t_bar, v, frac_aniso):
    """
    Equation  4 (page 222) Jakobsen et al. 2003 (The acoustic signature of fluid flow in complex
    porous media).
    Returns the effective stiffness tensor C* (nx6x6 matrix) calculated from the t-matrices t(r) (Eq. 1).

    Parameters
    ----------
    c0 : np.ndarray
        Stiffness tensor of the host of the inclusion (nx6x6 matrix).
    c1 : np.ndarray
        Sum of the concentration and t-matrices (nx6x6 matrix).
    gd : np.ndarray
        Correlation function (nx6x6 matrix).
    t : np.ndarray
        t-matrices of the different inclusions (nx6x6xr matrix) (anisotropic).
    t_bar: np.ndarray
        t-matrices of the different inclusions (nx6x6xr matrix) (isotropic).
    v : np.ndarray
        Concentration of the inclusions (nxr vector).
    frac_aniso: np.ndarray
        Fraction of anisotropic.

    Returns
    -------
    tuple
        c_eff: np.ndarray.
        c_eff: effective stiffness tensor C.
    """
    if not (
        c0.ndim == 3
        and c1.ndim == 3
        and gd.ndim == 3
        and t.ndim == 4
        and t_bar.ndim == 4
        and v.ndim == 2
        and np.all(
            np.array(
                [
                    c0.shape[1:2],
                    c1.shape[1:2],
                    gd.shape[1:2],
                    t.shape[1:2],
                    t_bar.shape[1:2],
                ]
            )
            == c0.shape[1]
        )
        and np.all(
            np.array(
                [
                    c0.shape[0],
                    c1.shape[0],
                    gd.shape[0],
                    t.shape[0],
                    t_bar.shape[0],
                    v.shape[0],
                ]
            )
            == c0.shape[0]
        )
    ):
        raise ValueError("calc_c_eff_vec: inconsistencies in input shapes")

    log_length = c0.shape[0]
    alpha_len = v.shape[1]
    v = v.reshape((log_length, 1, 1, alpha_len))
    i4 = np.eye(6)

    c1 = c1 + np.sum(frac_aniso * v * t + (1.0 - frac_aniso) * v * t_bar, axis=3)

    return c0 + array_matrix_mult(c1, array_inverse(i4 + array_matrix_mult(gd, c1)))


def calc_c_eff_visco_vec(
    vs,
    k_r,
    eta_f,
    v,
    gamma,
    tau,
    kd_uuvv,
    kappa,
    kappa_f,
    c0,
    s0,
    c1,
    td,
    td_bar,
    x,
    x_bar,
    gd,
    frequency,
    frac_ani,
):
    """
    Returns the effective stiffness tensor C* for a visco-elastic system (6x6xnumber of frequencies).

    Parameters
    ----------
    vs : np.ndarray
        The velocity used to calculate the wave number.
    k_r : np.ndarray
        Klinkenberg permability.
    eta_f : np.ndarray
        Viscosity (P).
    v : np.ndarray
        Concentration of the inclusions which are connected with respect to fluid flow.
    gamma : np.ndarray
        Gamma factor for each inclusion (1x(number of connected inclusions) vector).
    tau : float or np.ndarray
        Relaxation time constant.
    kd_uuvv : np.ndarray
        Kd_uuvv for each connected inclusion (1x(number of connected inclusions) vector.
    kappa : np.ndarray
        Bulk modulus of host material.
    kappa_f : np.ndarray
        Bulk modulus of the fluid.
    c0 : np.ndarray
        The stiffness tensor of host material (6x6 matrix).
    s0 : np.ndarray
        Inverse of C0.
    c1 : np.ndarray
        First order correction matrix(6x6 matrix). If there are isolated inclusions, C1 is sum of concentration and
        t-matrices of the isolated part of the porosity.
    td : np.ndarray
        t-matrix tensors.
    td_bar : np.ndarray
        t-matrices of the connected inclusions(6x6x(numbers of inclusions) matrix).
    x : np.ndarray
        X-tensor.
    x_bar : np.ndarray
        X-tensor of the connected inclusions (6x6x(numbers of inclusions) matrix).
    gd : np.ndarray
        Correlation function (6x6 matrix).
    frequency : float
        Frequency under consideration.
    frac_ani : np.ndarray
        Fraction of anisotropic inclusions.

    Returns
    -------
    np.ndarray
        Effective stiffness tensor.
    """
    dr = k_r / eta_f

    omega = 2 * np.pi * frequency
    k = omega / vs
    theta = calc_theta_vec(v, omega, gamma, tau, kd_uuvv, dr, k, kappa, kappa_f)
    z, z_bar = calc_z_vec(s0, td, td_bar, omega, gamma, v * frac_ani, tau)
    t = calc_t_vec(td, theta, x, z, omega, gamma, tau, kappa_f)
    t_bar = calc_t_vec(td_bar, theta, x_bar, z_bar, omega, gamma, tau, kappa_f)

    return calc_c_eff_vec(c0, c1, gd, t, t_bar, v, frac_ani)
