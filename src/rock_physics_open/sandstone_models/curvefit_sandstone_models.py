import numpy as np

from .constant_cement_models import constant_cement_model
from .friable_models import friable_model
from .patchy_cement_model import patchy_cement_model_weight


def curvefit_patchy_cement(x_data, weight_k, weight_mu, shear_red, frac_cem):
    """Run patchy_cement_model_weight with parameter optimisation for closest possible fit to observations

    Inputs: vp, vs, rho, por, k_fl, rho_fl, p_eff, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, phi_c
    Parameters to optimise for: frac_cem, shear_red, weight_k, weight_mu
    Optimise for vp, vs
    """
    # Unpack x inputs
    # In calling function:
    k_min = x_data[:, 0]
    mu_min = x_data[:, 1]
    rho_min = x_data[:, 2]
    k_cem = x_data[:, 3]
    mu_cem = x_data[:, 4]
    rho_cem = x_data[:, 5]
    k_fl = x_data[:, 6]
    rho_fl = x_data[:, 7]
    phi = x_data[:, 8]
    p_eff = x_data[:, 9]
    def_vp_vs_ratio = x_data[0, 10]
    phi_c = x_data[0, 11]

    # Catch phi values that are above phi_c - frac_cem, reduce silently to phi_c - frac_cem
    phi = np.minimum(phi, phi_c - frac_cem)

    try:
        vp, vs = patchy_cement_model_weight(
            k_min,
            mu_min,
            rho_min,
            k_cem,
            mu_cem,
            rho_cem,
            k_fl,
            rho_fl,
            phi,
            p_eff,
            frac_cem,
            phi_c,
            "por_based",
            9.0,
            shear_red,
            weight_k,
            weight_mu,
        )[0:2]
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")


def curvefit_friable(x_data, phi_c, shear_red):
    """Run friable sand model with parameter optimisation for closest possible fit to observations

    Inputs: vp, vs, por, k_fl, rho_fl, p_eff, k_min, mu_min, rho_min
    Parameters to optimise for: phi_c, shear_red
    Optimise for vp, vs
    """
    # Unpack x inputs
    k_min = x_data[:, 0]
    mu_min = x_data[:, 1]
    rho_min = x_data[:, 2]
    k_fl = x_data[:, 3]
    rho_fl = x_data[:, 4]
    phi = x_data[:, 5]
    p_eff = x_data[:, 6]
    def_vp_vs_ratio = x_data[0, 7]

    # Catch phi values that are above phi_c, reduce silently to phi_c
    phi = np.minimum(phi, phi_c)

    try:
        vp, vs = friable_model(
            k_min,
            mu_min,
            rho_min,
            k_fl,
            rho_fl,
            phi,
            p_eff,
            phi_c,
            "por_based",
            1.0,
            shear_red,
        )[0:2]
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")


def curvefit_constant_cement(x_data, phi_c, shear_red, frac_cem):
    """Run constant_cement_model with parameter optimisation for closest possible fit to observations

    Inputs: vp, vs, por, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, k_fl, rho_fl
    Parameters to optimise for: phi_c, shear_red, frac_cem
    Optimise for vp, vs
    """
    # Unpack x inputs
    k_min = x_data[:, 0]
    mu_min = x_data[:, 1]
    rho_min = x_data[:, 2]
    k_cem = x_data[:, 3]
    mu_cem = x_data[:, 4]
    rho_cem = x_data[:, 5]
    k_fl = x_data[:, 6]
    rho_fl = x_data[:, 7]
    phi = x_data[:, 8]
    def_vp_vs_ratio = x_data[0, 9]

    # Catch phi values that are above phi_c - frac_cem, reduce silently to phi_c - frac_cem
    phi = np.minimum(phi, phi_c - frac_cem)

    try:
        vp, vs = constant_cement_model(
            k_min,
            mu_min,
            rho_min,
            k_cem,
            mu_cem,
            rho_cem,
            k_fl,
            rho_fl,
            phi,
            frac_cem,
            phi_c,
            9.0,
            shear_red,
        )[0:2]
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")
