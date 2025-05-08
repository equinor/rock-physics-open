import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector
from rock_physics_open.equinor_utilities.std_functions import hashin_shtrikman_average
from rock_physics_open.t_matrix_models import t_matrix_porosity_c_alpha_v

from .opt_subst_utilities import opt_param_info


def curvefit_t_matrix_exp(
    x_data,
    frac_ani,
    frac_con,
    alpha1,
    alpha2,
    v1,
    k_c,
    mu_c,
    rho_c,
    k_sh,
    mu_sh,
    rho_sh,
):
    """Optimisation of input parameters to T-Matrix for carbonate in case where the mineral composition for each
    sample is not known, so that effective mineral moduli for each sample are part of the parameters to be optimised
    for. The optimisation is also made for the inclusion parameters in addition to fraction of connected and anisotropic
    porosity and the VTI plane angle.

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
    k_c : float
        p-wave velocity for carbonate matrix.
    mu_c : float
        vp/vs ratio for carbonate matrix.
    rho_c : float
        vp/vs ratio for carbonate matrix.
    k_sh : float
        p-wave velocity for shale.
    mu_sh : float
        vp/vs ratio for shale.
    rho_sh : float
        Density for shale.

    Returns
    -------
    np.ndarray
        Modelled velocity, vp and vs.
    """
    # Restore original value range for parameters - must match the scaling performed in calling function
    scale_val = opt_param_info()[1]
    k_c *= scale_val["k_carb"]
    mu_c *= scale_val["mu_carb"]
    rho_c *= scale_val["rho_carb"]
    k_sh *= scale_val["k_sh"]
    mu_sh *= scale_val["mu_sh"]
    rho_sh *= scale_val["rho_sh"]

    # Unpack x inputs
    # In calling function:
    # x_data = np.stack((por, vsh, k_fl, rho_fl, k_r, eta_f, tau, freq, def_vpvs), axis=1)
    phi = x_data[:, 0]
    vsh = x_data[:, 1]
    k_fl = x_data[:, 2]
    rho_fl = x_data[:, 3]
    angle_sym_plane = x_data[0, 4]
    perm = x_data[:, 5]
    visco = x_data[:, 6]
    tau = x_data[:, 7]
    # Both frequency and vp/vs ratio has to be in the same format as the others, but only a scalar value is used
    freq = x_data[0, 8]
    def_vp_vs_ratio = x_data[0, 9]

    # Mineral properties
    # Expand elastic properties to vectors of the same length as the x_data inputs
    k_c, mu_c, rho_c, k_sh, mu_sh, rho_sh, _ = dim_check_vector(
        (k_c, mu_c, rho_c, k_sh, mu_sh, rho_sh, phi)
    )
    k_min, mu_min = hashin_shtrikman_average(k_sh, mu_sh, k_c, mu_c, vsh)
    rho_min = vsh * rho_sh + (1.0 - vsh) * rho_c

    # Inclusion aspect ratios and concentrations
    log_len = phi.shape[0]
    alpha = np.zeros((log_len, 2))
    v = np.zeros((log_len, 2))
    alpha[:, 0] = alpha1 * np.ones(log_len)
    alpha[:, 1] = alpha2 * np.ones(log_len)
    v[:, 0] = v1 * np.ones(log_len)
    v[:, 1] = (1 - v1) * np.ones(log_len)

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
            alpha,
            v,
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
