import numpy as np

from .calc_kd_eff import calc_kd_eff_vec


def calc_pressure_vec(
    alpha_con,
    alpha_iso,
    v_con,
    v_iso,
    c0,
    s_0,
    gd,
    d_p,
    tau,
    gamma,
    k_fl,
    ctrl,
    frac_ani,
):
    """Calculate the effect of depletion on aspect ratios.

    Parameters
    ----------
    alpha_con : np.ndarray
        Aspect ratio for connected inclusions (r length vector).
    alpha_iso : np.ndarray
        Aspect ratio for connected inclusions (s length vector).
    v_con : np.ndarray
        Volume of connected inclusions (r length vector).
    v_iso : np.ndarray
        Volume of connected inclusions (s length vector).
    c0 : np.ndarray
        Stiffness tensor of host material (nx6x6 array).
    s_0 : np.ndarray
        Inverse of stiffness tensor (nx6x6 array).
    gd : np.ndarray
        The correlation function (green's tensor nx6x6 matrix).
    d_p : float
        Change in effective pressure.
    tau : np.ndarray
        Relaxation time constant ((numbers of connected pores) vector).
    gamma : np.ndarray
        Gamma factor ((numbers of connected pores) vector).
    k_fl : np.ndarray
        Fluid bulk modulus (n length vector).
    ctrl : int
        Control parameter.
    frac_ani : float
        Fraction of anisotropic inclusions.

    Returns
    -------
    tuple
        alpha_n_connected, v_n_connected, alpha_n_isolated, v_n_isolated, tau_n_out, gamma_n_out -
        modified alphas, volumes, relaxation times and gamma factors.

    Notes
    -----
    Equations used can be found in:
    Agersborg (2007), phd thesis:
    https://bora.uib.no/handle/1956/2422

    09.03.2012
    Remy Agersborg
    email: remy@agersborg.com

    Translated to Python and vectorised by Harald Flesche, hfle@equinor.com 2020
    """

    def _new_values(k, sum_k, a, v, d_p, t, g):
        # Local helper function to avoid code duplication
        len_alpha = a.shape[1]
        len_log = k.shape[0]

        v_new = v * (
            1
            - (np.sum(k[:, 0:3, 0:3, :], axis=(1, 2)) - sum_k.reshape(len_log, 1)) * d_p
        )
        alpha_new = a * (
            1
            - (np.sum(k[:, 2, 0:3, :], axis=1) - np.sum(k[:, 0, 0:3, :], axis=1)) * d_p
        )

        idx_neg = (alpha_new < 0.0) | (v_new < 0.0)
        idx_high = (alpha_new > 1.0) | (v_new > 1.0)
        idx_inval = np.logical_or(idx_neg, idx_high)

        for i in range(len_alpha):
            # Maybe best to set undefined, but for now keep old values
            # alpha_new[idx_inval] = np.nan
            # v_new[idx_inval] = np.nan
            alpha_new[idx_inval[:, i], i] = a[idx_inval[:, i], i]
            v_new[idx_inval[:, i], i] = v[idx_inval[:, i], i]
        tau_n = np.array(t)
        gamma_n = np.array(g)

        return v_new, alpha_new, tau_n, gamma_n

    kd_eff_isolated, kd_eff_connected = calc_kd_eff_vec(
        c0, s_0, k_fl, alpha_con, alpha_iso, v_con, v_iso, gd, ctrl, frac_ani
    )
    # Find the sum in the eq. 21 Jakobsen and Johansen 2005
    sum_kd = 0.0
    if ctrl != 2 and kd_eff_isolated is not None:
        count_isolated = kd_eff_isolated.shape[3]
        for j in range(count_isolated):
            sum_kd = sum_kd + v_iso[:, j] * np.sum(
                kd_eff_isolated[:, 0:3, 0:3, j], axis=(1, 2)
            )
    if ctrl != 0 and kd_eff_connected is not None:
        count_connected = kd_eff_connected.shape[3]
        for j in range(count_connected):
            sum_kd = sum_kd + v_con[:, j] * np.sum(
                kd_eff_connected[:, 0:3, 0:3, j], axis=(1, 2)
            )
    # Find the new concentration of inclusion
    alpha_n_isolated = None
    alpha_n_connected = None
    v_n_isolated = None
    v_n_connected = None
    gamma_n_out = []
    tau_n_out = []
    if ctrl != 2:
        v_n_isolated, alpha_n_isolated, _, _ = _new_values(
            kd_eff_isolated, sum_kd, alpha_iso, v_iso, d_p, tau, gamma
        )
    if ctrl != 0:
        v_n_connected, alpha_n_connected, tau_n_out, gamma_n_out = _new_values(
            kd_eff_connected, sum_kd, alpha_con, v_con, d_p, tau, gamma
        )

    return (
        alpha_n_connected,
        v_n_connected,
        alpha_n_isolated,
        v_n_isolated,
        tau_n_out,
        gamma_n_out,
    )
