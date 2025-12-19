from typing import Literal

import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    Array3D,
    Array4D,
)

from .array_functions import array_inverse, array_matrix_mult
from .calc_isolated import calc_isolated_part_vec
from .g_tensor import g_tensor_vec


def calc_kd_eff_vec(
    c0: Array3D[np.float64],
    s_0: Array3D[np.float64],
    k_fl: Array1D[np.float64],
    alpha_con: Array2D[np.float64],
    alpha_iso: Array2D[np.float64],
    v_con: Array2D[np.float64],
    v_iso: Array2D[np.float64],
    gd: Array3D[np.float64],
    ctrl: Literal[0, 1, 2],
    frac_ani: float,
) -> tuple[Array4D[np.float64] | None, Array4D[np.float64] | None]:
    """Returns the effective dry K-tensor (6x6x(numbers of inclusions) matrix.
    If there is no connected or no isolated pores, the function returns a NaN for
    the case which is not considered. E.g. if only isolated pores, the kd_eff_connected = NaN.

    Note: When isolated pores, the pores are considered as filled when
    calculating the dry effective K-tensor.

    Parameters
    ----------
    c0 : np.ndarray
        Stiffness tensor of the host material (nx6x6 matrix).
    s_0 : np.ndarray
        Inverse of the stiffness tensor.
    k_fl : np.ndarray
        Bulk modulus of the fluid (n length vector).
    alpha_con : np.ndarray
        Aspect ratio of connected inclusions.
    alpha_iso : np.ndarray
        Aspect ratio of isolated inclusions.
    v_con : np.ndarray
        Concentration of connected pores.
    v_iso : np.ndarray
        Concentration of isolated pores.
    gd : np.ndarray
        Correlation function (nx6x6 matrix).
    ctrl : int
        0 :only isolated pores, 1 :both isolated and connected pores, 2 :only connected pores.
    frac_ani : float
        Fraction of anisotropic inclusions.

    Returns
    -------
    tuple
         kd_eff_isolated, kd_eff_connected: (np.ndarray, np.ndarray).

    Notes
    -----
    Equations used can be found in:
    Agersborg (2007), phd thesis:
    https://bora.uib.no/handle/1956/2422

    09.03.2012
    Remy Agersborg
    email: remy@agersborg.com

    Translated to Python and vectorised by Harald Flesche, hfle@equinor.com 2020.
    """
    log_len = c0.shape[0]
    c1dry = np.zeros((log_len, 6, 6))
    c2dry = np.zeros((log_len, 6, 6))

    kd_eff_isolated = None
    kd_eff_connected = None

    c1_isolated = None
    if ctrl != 2:
        c1_isolated = calc_isolated_part_vec(
            c0=c0,
            s_0=s_0,
            kappa_f=k_fl,
            alpha=alpha_iso,
            v=v_iso,
            case_iso=ctrl,
            frac_ani=frac_ani,
        )
        c1dry = c1dry + c1_isolated
        if alpha_iso.ndim == 1 and alpha_iso.shape[0] != c0.shape[0]:
            alpha_iso = np.tile(alpha_iso.reshape(1, alpha_iso.shape[0]), (log_len, 1))
        c2dry = c2dry + array_matrix_mult(c1_isolated, gd, c1_isolated)
    if ctrl != 0:
        c1_connected = calc_isolated_part_vec(
            c0, s_0, np.zeros_like(k_fl), alpha_con, v_con, ctrl, frac_ani
        )
        c1dry = c1dry + c1_connected
        if alpha_con.ndim == 1 and alpha_con.shape[0] != c0.shape[0]:
            alpha_con = np.tile(alpha_con.reshape(1, alpha_con.shape[0]), (log_len, 1))
        c2dry = c2dry + array_matrix_mult(c1_connected, gd, c1_connected)
        if c1_isolated is not None:
            c2dry = (
                c2dry
                + array_matrix_mult(c1_connected, gd, c1_isolated)
                + array_matrix_mult(c1_isolated, gd, c1_connected)
            )

    i4 = np.tile(np.eye(6).reshape(1, 6, 6), (log_len, 1, 1))
    c_eff_dry = c0 + array_matrix_mult(
        c1dry, array_inverse(i4 + array_matrix_mult(array_inverse(c1dry), c2dry))
    )
    temp = array_matrix_mult(
        i4,
        array_inverse(i4 + array_matrix_mult(array_inverse(c1dry), c2dry)),
        array_inverse(c_eff_dry),
    )

    # if only connected or mixed connected and isolated
    if ctrl != 0:
        kd_eff_connected = np.zeros((log_len, 6, 6, alpha_con.shape[1]))
        for j in range(alpha_con.shape[1]):
            g = g_tensor_vec(
                c0=c0,
                s_0=s_0,
                alpha=alpha_con[:, j],
            )
            kd_eff_connected[:, :, :, j] = array_matrix_mult(
                array_inverse(i4 + array_matrix_mult(g, c0)), temp
            )

    if ctrl != 2:
        kd_eff_isolated = np.zeros((log_len, 6, 6, alpha_iso.shape[1]))
        for j in range(alpha_iso.shape[1]):
            g = g_tensor_vec(
                c0=c0,
                s_0=s_0,
                alpha=alpha_iso[:, j],
            )
            kd_eff_isolated[:, :, :, j] = array_matrix_mult(
                array_inverse(i4 + array_matrix_mult(g, c0)), temp
            )

    return kd_eff_isolated, kd_eff_connected
