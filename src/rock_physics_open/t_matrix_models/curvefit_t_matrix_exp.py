from typing import cast

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector
from rock_physics_open.equinor_utilities.optimisation_utilities import opt_param_info
from rock_physics_open.equinor_utilities.std_functions import hashin_shtrikman_average
from rock_physics_open.equinor_utilities.various_utilities.types import Array1D, Array2D

from .t_matrix_C import t_matrix_porosity_c_alpha_v


def curvefit_t_matrix_exp(
    x_data: Array2D[np.float64],
    frac_ani: float,
    frac_con: float,
    alpha1: float,
    alpha2: float,
    v1: float,
    k_c: float,
    mu_c: float,
    rho_c: float,
    k_sh: float,
    mu_sh: float,
    rho_sh: float,
) -> Array1D[np.float64]:
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
    _, scale_val, _ = opt_param_info()
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
    k_c_, mu_c_, rho_c_, k_sh_, mu_sh_, rho_sh_, _ = cast(
        list[npt.NDArray[np.float64]],
        dim_check_vector(
            (
                k_c,
                mu_c,
                rho_c,
                k_sh,
                mu_sh,
                rho_sh,
                phi,
            )
        ),
    )
    k_min, mu_min = hashin_shtrikman_average(
        k1=k_sh_,
        mu1=mu_sh_,
        k2=k_c_,
        mu2=mu_c_,
        f=vsh,
    )
    rho_min = vsh * rho_sh_ + (1.0 - vsh) * rho_c_

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
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=phi,
            perm=perm,
            visco=visco,
            alpha=alpha,
            v=v,
            tau=tau,
            frequency=freq,
            angle=angle_sym_plane,
            frac_inc_con=frac_con,
            frac_inc_ani=frac_ani,
        )
    except ValueError:
        vp = np.zeros(k_min.shape)
        vsv = np.zeros(k_min.shape)

    return np.stack((vp, def_vp_vs_ratio * vsv), axis=1).flatten("F")
