from typing import Callable, cast

import numpy as np
import numpy.typing as npt
from numpy import float64

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.optimisation_utilities import (
    gen_opt_routine,
    opt_param_to_ascii,
    save_opt_params,
)

from .curvefit_sandstone_models import curvefit_constant_cement


def constant_cement_model_optimisation(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    por: npt.NDArray[np.float64],
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rhob: npt.NDArray[np.float64],
    file_out_str: str = "constant_cement_optimal_params.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Patchy cement model with optimisation for a selection of parameters.

    Parameters
    ----------
    k_min : np.ndarray
        Cement bulk modulus [Pa].
    mu_min : np.ndarray
        Cement shear modulus [Pa].
    rho_min : np.ndarray
        Cement density [kg/m^3].
    k_cem : np.ndarray
        Cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement density [kg/m^3].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid density [kg/m^3].
    por : np.ndarray
        Inclusion porosity [ratio].
    vp : np.ndarray
        Compressional velocity log [m/s].
    vs : np.ndarray
        Shear velocity log [m/s].
    rhob : np.ndarray
        Bulk density log [kg/m^3].
    file_out_str : str
        Output file name (string) to store optimal parameters (pickle format).
    display_results : bool
        Display optimal parameters in a window after run.
    well_name : str
        Name of well to be displayed in info box title.

    Returns
    -------
    tuple
        vp_mod, vs_mod - modelled logs, vp_res, vs_res : np.ndarray
        residual logs.
    """

    # Skip hardcoded Vp/Vs ratio
    def_vpvs = np.mean(vp / vs)
    # Set weight to vs to give vp and vs similar influence on optimisation
    y_data = np.stack([vp, vs * def_vpvs], axis=1)
    # Optimisation function for selected parameters
    opt_fun: Callable[..., npt.NDArray[float64]] = curvefit_constant_cement
    # expand single value parameters to match logs length
    por, def_vpvs = cast(
        list[npt.NDArray[np.float64]],
        gen_utilities.dim_check_vector((por, def_vpvs)),
    )
    x_data = np.stack(
        (k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, k_fl, rho_fl, por, def_vpvs),
        axis=1,
    )

    # Params: weight_k, weight_mu, shear_red, frac_cem
    lower_bound = np.array(
        [
            0.35,  # phi_c
            0.0,  # shear_red
            0.01,  # frac_cem
        ],
        dtype=float,
    )
    upper_bound = np.array(
        [
            0.45,  # phi_c
            1.0,  # shear_red
            0.1,  # frac_cem
        ],
        dtype=float,
    )

    x0 = (upper_bound + lower_bound) / 2.0
    # Optimisation step without fluid substitution
    vel_mod, vel_res, opt_params = gen_opt_routine(
        opt_function=opt_fun,
        x_data_orig=x_data,
        y_data=y_data,
        x_init=x0,
        low_bound=lower_bound,
        high_bound=upper_bound,
    )
    frac_cem = opt_params[2]

    # Reshape outputs and remove weight from vs
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(vel_mod, 2, axis=1)]
    vp_res, vs_res = [arr.flatten() for arr in np.split(vel_res, 2, axis=1)]
    vs_mod = vs_mod / def_vpvs
    vs_res = vs_res / def_vpvs
    vpvs_mod = vp_mod / vs_mod
    # Calculate the modelled density
    rhob_mod = rho_min * (1.0 - por - frac_cem) + frac_cem * rho_cem + por * rho_fl
    ai_mod = vp_mod * rhob_mod
    rhob_res = rhob_mod - rhob
    # Save the optimal parameters
    save_opt_params(
        opt_type="const_cem",
        opt_params=opt_params,
        file_name=file_out_str,
        well_name=well_name,
    )
    if display_results:
        opt_param_to_ascii(in_file=file_out_str, well_name=well_name)

    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res
