import numpy as np
import numpy.typing as npt

from .constant_cement_models import constant_cement_model
from .friable_models import friable_model
from .patchy_cement_model import patchy_cement_model_weight


def curvefit_patchy_cement(
    x_data: npt.NDArray[np.float64],
    weight_k: float,
    weight_mu: float,
    shear_red: float,
    frac_cem: float,
) -> npt.NDArray[np.float64]:
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
        vp, vs, _, _, _ = patchy_cement_model_weight(
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_cem=k_cem,
            mu_cem=mu_cem,
            rho_cem=rho_cem,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=phi,
            p_eff=p_eff,
            frac_cem=frac_cem,
            phi_c=phi_c,
            coord_num_func="PorBased",
            n=9.0,
            shear_red=shear_red,
            weight_k=weight_k,
            weight_mu=weight_mu,
        )
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")


def curvefit_friable(
    x_data: npt.NDArray[np.float64],
    phi_c: float,
    shear_red: float,
) -> npt.NDArray[np.float64]:
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
        vp, vs, _, _, _ = friable_model(
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=phi,
            p_eff=p_eff,
            phi_c=phi_c,
            coord_num_func="PorBased",
            n=1.0,
            shear_red=shear_red,
        )
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")


def curvefit_constant_cement(
    x_data: npt.NDArray[np.float64],
    phi_c: float,
    shear_red: float,
    frac_cem: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
        vp, vs, _, _, _ = constant_cement_model(
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_cem=k_cem,
            mu_cem=mu_cem,
            rho_cem=rho_cem,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=phi,
            frac_cem=frac_cem,
            phi_c=phi_c,
            n=9.0,
            shear_red=shear_red,
        )
    except ValueError:
        vp = np.zeros(k_min.shape)
        vs = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vs), axis=1).flatten("F")
