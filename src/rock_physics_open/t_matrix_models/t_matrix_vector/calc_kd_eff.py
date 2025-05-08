import numpy as np

from .array_functions import array_inverse, array_matrix_mult
from .calc_isolated import calc_isolated_part_vec
from .g_tensor import g_tensor_vec


def calc_kd_eff_vec(
    c0, s_0, k_fl, alpha_con, alpha_iso, v_con, v_iso, gd, ctrl, frac_ani
):
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
            c0, s_0, k_fl, alpha_iso, v_iso, ctrl, frac_ani
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
            g = g_tensor_vec(c0, s_0, alpha_con[:, j])
            kd_eff_connected[:, :, :, j] = array_matrix_mult(
                array_inverse(i4 + array_matrix_mult(g, c0)), temp
            )

    if ctrl != 2:
        kd_eff_isolated = np.zeros((log_len, 6, 6, alpha_iso.shape[1]))
        for j in range(alpha_iso.shape[1]):
            g = g_tensor_vec(c0, s_0, alpha_iso[:, j])
            kd_eff_isolated[:, :, :, j] = array_matrix_mult(
                array_inverse(i4 + array_matrix_mult(g, c0)), temp
            )

    return kd_eff_isolated, kd_eff_connected
