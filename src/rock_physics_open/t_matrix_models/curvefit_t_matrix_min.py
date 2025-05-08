import numpy as np

from rock_physics_open.t_matrix_models import t_matrix_porosity_c_alpha_v


def curve_fit_2_inclusion_sets(x_data, frac_ani, frac_con, alpha1, alpha2, v1):
    """Optimisation of input parameters to T-Matrix for carbonate in case where the mineral composition for each
    sample is known, so that effective mineral moduli for each sample are part of the inputs. The optimisation
    is made for the inclusion parameters in addition to fraction of connected and anisotropic porosity and the
    VTI plane angle.

    Parameters
    ----------
    x_data : np.ndarray
        Inputs to unpack.
    frac_ani : float
        Fraction of anisotropic inclusions.
    frac_con : float
        Fraction of connected inclusions.
    alpha1 : float
        Aspect ratio of first inclusion set.
    alpha2 : float
        Aspect ratio of second inclusion set.
    v1 : float
        Concentration ratio of first inclusion set.

    Returns
    -------
    np.ndarray
        Modelled velocity.
    """
    # Unpack x inputs
    # In calling function:
    # x_data = np.stack((por, k_min, mu_min, rho_min, k_fl, rho_fl, k_r, eta_f, tau, freq, def_vp_vs_ratio), axis=1)
    phi = x_data[:, 0]
    k_min = x_data[:, 1]
    mu_min = x_data[:, 2]
    rho_min = x_data[:, 3]
    k_fl = x_data[:, 4]
    rho_fl = x_data[:, 5]
    angle_sym_plane = x_data[0, 6]
    perm = x_data[:, 7]
    visco = x_data[:, 8]
    tau = x_data[:, 9]
    # Both frequency and vp/vs ratio has to be in the same format as the others, but only a scalar value is used
    freq = x_data[0, 10]
    def_vp_vs_ratio = x_data[0, 11]

    log_length = len(phi)

    # alpha and v values should follow these conditions:
    # - alphas are given in decreasing values
    # - the sum of v's should add up to unity
    if not (alpha1 > alpha2 and 0.0 < v1 <= 1.0):
        raise ValueError(
            "curvefit_t_matrix: alpha 1 must have higher value than alpha 2, and v1 must be a "
            "positive fraction"
        )
    alpha = np.array([alpha1, alpha2]).reshape(1, 2)
    v = np.array([v1, 1.0 - v1]).reshape(1, 2)
    alpha_vec = alpha * np.ones((log_length, 1))
    v_vec = v * np.ones((log_length, 1))

    try:
        vp, vsv, _, _ = t_matrix_porosity_c_alpha_v(
            k_min,
            mu_min,
            rho_min,
            k_fl,
            rho_fl,
            phi,
            perm,
            visco,
            alpha_vec,
            v_vec,
            tau,
            freq,
            angle_sym_plane,
            frac_con,
            frac_ani,
        )
    except ValueError:
        vp = np.zeros(k_min.shape)
        vsv = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vsv), axis=1).flatten("F")
