import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    Array3D,
    Array4D,
)

from .array_functions import array_inverse, array_matrix_mult
from .calc_t import calc_t_vec
from .calc_theta import calc_theta_vec
from .calc_z import calc_z_vec


def calc_c_eff_vec(
    c0: Array3D[np.float64],
    c1: Array3D[np.float64],
    gd: Array3D[np.float64],
    t: Array4D[np.complex128],
    t_bar: Array4D[np.complex128],
    v: Array2D[np.float64],
    frac_aniso: float,
) -> Array3D[np.float64]:
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
    v_ = v.reshape((log_length, 1, 1, alpha_len))
    i4 = np.eye(6)

    c1 = c1 + np.sum(frac_aniso * v_ * t + (1.0 - frac_aniso) * v_ * t_bar, axis=3)

    return c0 + array_matrix_mult(c1, array_inverse(i4 + array_matrix_mult(gd, c1)))


def calc_c_eff_visco_vec(
    vs: Array1D[np.float64],
    k_r: Array1D[np.float64],
    eta_f: Array1D[np.float64],
    v: Array2D[np.float64],
    gamma: Array2D[np.float64],
    tau: Array1D[np.float64],
    kd_uuvv: Array2D[np.float64],
    kappa: Array1D[np.float64],
    kappa_f: Array1D[np.float64],
    c0: Array3D[np.float64],
    s0: Array3D[np.float64],
    c1: Array3D[np.float64],
    td: Array4D[np.float64],
    td_bar: Array4D[np.float64],
    x: Array4D[np.float64],
    x_bar: Array4D[np.float64],
    gd: Array3D[np.float64],
    frequency: float,
    frac_ani: float,
) -> Array3D[np.float64]:
    """
    Returns the effective stiffness tensor C* for a visco-elastic system (6x6xnumber of frequencies).

    Parameters
    ----------
    vs : np.ndarray
        The velocity used to calculate the wave number.
    k_r : np.ndarray
        Klinkenberg permeability.
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
    theta = calc_theta_vec(
        v=v,
        omega=omega,
        gamma=gamma,
        tau=tau,
        kd_uuvv=kd_uuvv,
        dr=dr,
        k=k,
        kappa=kappa,
        kappa_f=kappa_f,
    )
    z, z_bar = calc_z_vec(
        s0=s0,
        td=td,
        td_bar=td_bar,
        omega=omega,
        gamma=gamma,
        v=v * frac_ani,
        tau=tau,
    )
    t = calc_t_vec(
        td=td,
        theta=theta,
        x=x,
        z=z,
        omega=omega,
        gamma=gamma,
        tau=tau,
        k_fluid=kappa_f,
    )
    t_bar = calc_t_vec(
        td=td_bar,
        theta=theta,
        x=x_bar,
        z=z_bar,
        omega=omega,
        gamma=gamma,
        tau=tau,
        k_fluid=kappa_f,
    )

    return calc_c_eff_vec(
        c0=c0,
        c1=c1,
        gd=gd,
        t=t,
        t_bar=t_bar,
        v=v,
        frac_aniso=frac_ani,
    )
