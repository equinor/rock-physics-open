from typing import Any, cast

import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.optimisation_utilities import (
    gen_opt_routine,
    opt_param_info,
    save_opt_params,
)
from rock_physics_open.equinor_utilities.optimisation_utilities.opt_subst_utilities import (
    opt_param_to_ascii,
)
from rock_physics_open.equinor_utilities.various_utilities.types import Array1D

from .curvefit_t_matrix_exp import curvefit_t_matrix_exp
from .t_matrix_parameter_optimisation_min import DEF_VP_VS_RATIO


def t_matrix_optimisation_exp(
    k_fl: Array1D[np.float64],
    rho_fl: Array1D[np.float64],
    por: Array1D[np.float64],
    vsh: Array1D[np.float64],
    vp: Array1D[np.float64],
    vs: Array1D[np.float64],
    rhob: Array1D[np.float64],
    angle: float = 0.0,
    k_r: float = 50.0,
    eta_f: float = 1.0,
    tau: float = 1.0e-7,
    freq: float = 1.0e3,
    file_out_str: str = "opt_params_exp.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
    **opt_kwargs: Any,
) -> tuple[
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
]:
    """T-Matrix optimisation adapted to an exploration setting, where detailed well information is generally not known.
    Inclusion parameters are optimised for the whole well.

    Parameters
    ----------
    por : np.ndarray
        Inclusion porosity [ratio].
    vsh : np.ndarray
        Shale volume [ratio].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid density [kg/m^3].
    vp : np.ndarray
        Compressional velocity log [m/s].
    vs : np.ndarray
        Shear velocity log [m/s].
    rhob : np.ndarray
        Bulk density log [kg/m^3].
    angle : float
        Angle of symmetry plane [degrees]
    k_r : float
        Permeability [mD].
    eta_f : float
        Fluid viscosity [cP].
    tau : float
        Relaxation time constant [s].
    freq : float
        Signal frequency [Hz].
    file_out_str : str
        Output file name (string) to store optimal parameters (pickle format).
    display_results : bool
        Display optimal parameters in a window after run.
    well_name : str
        Name of well to be displayed in info box title.
    opt_kwargs : Any
        Additional keywords to be passed to optimisation function

    Returns
    -------
    tuple
        vp_mod, vs_mod - modelled logs, vp_res, vs_res - residual logs.
    """
    # 1. Preparation that is independent of search for minimum possible aspect ratio for second inclusion set

    # Optimisation function for selected parameters
    opt_fun = curvefit_t_matrix_exp
    # rho_min = (rhob - por * rho_fl) / (1 - por)
    # EXP adapted inputs: include fluid data and other params in x_data
    # expand single value parameters to match logs length
    por, angle_, k_r_, eta_f_, tau_, freq_, def_vpvs = cast(
        list[Array1D[np.float64]],
        gen_utilities.dim_check_vector(
            (por, angle, k_r, eta_f, tau, freq, DEF_VP_VS_RATIO)
        ),
    )
    x_data = np.stack(
        (por, vsh, k_fl, rho_fl, angle_, k_r_, eta_f_, tau_, freq_, def_vpvs), axis=1
    )
    # Set weight to vs to give vp and vs similar influence on optimisation
    y_data = np.stack([vp, vs * DEF_VP_VS_RATIO], axis=1)

    # 2. Search for minimum aspect ratio of inclusion set no. 2, given that inclusion set no. 1 will at least represent
    # 50% of inclusions. Minimum aspect ratio is linked to the porosity, and the most conservative estimate is to use
    # the maximum value. This is likely to deteriorate the optimisation results, so a search for a more appropriate
    # value that still produces valid results is made
    percentiles = [50, 75, 80, 85, 90, 95, 99, 100]
    valid_result = False
    vel_mod = None
    vel_res = None
    opt_params = None
    _, scale_val, _ = opt_param_info()
    while not valid_result and percentiles:
        try:
            # Make sure that parameters are not in conflict with T Matrix assumptions
            min_v1 = 0.5
            max_por = np.percentile(por, percentiles[0])
            min_a2 = (1.0 - min_v1) * max_por
            # Test with all parameters in the range 0.0 - 1.0
            # Params:      f_ani f_con a1 a2 v1 k_carb mu_carb rho_carb k_sh mu_sh rho_sh
            lower_bound = np.array(
                [
                    0.0,
                    0.0,
                    0.5,
                    min_a2,
                    min_v1,  # f_ani f_con a1 a2 v1
                    35.0 / scale_val["k_carb"],
                    30.0 / scale_val["mu_carb"],
                    2650.0 / scale_val["rho_carb"],  # k_carb, mu_carb, rho_carb
                    15.0 / scale_val["k_sh"],
                    7.0 / scale_val["mu_sh"],
                    2500.0 / scale_val["rho_sh"],
                ],
                dtype=float,
            )  # k_sh, mu_sh, rho_sh
            upper_bound = np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    0.30,
                    1.0,  # f_ani f_con a1 a2 v1
                    1.0,
                    1.0,
                    1.0,  # k_carb, mu_carb, rho_carb
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=float,
            )  # k_sh, mu_sh, rho_sh
            x0 = (upper_bound + lower_bound) / 2.0
            # Optimisation step without fluid substitution
            vel_mod, vel_res, opt_params = gen_opt_routine(
                opt_function=opt_fun,
                x_data_orig=x_data,
                y_data=y_data,
                x_init=x0,
                low_bound=lower_bound,
                high_bound=upper_bound,
                **opt_kwargs,
            )
            valid_result = True
        except ValueError:
            _ = percentiles.pop(0)
            valid_result = False

    if not valid_result:
        raise ValueError(
            f"{__file__}: unable to find stable value for T Matrix optimisation, second inclusion"
        )
    if vel_mod is None or vel_res is None or opt_params is None:
        raise ValueError(
            f"{__file__}: expected optimisation routine to return values, but got None"
        )

    # Reshape outputs and remove weight from vs
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(vel_mod, 2, axis=1)]
    vp_res, vs_res = [arr.flatten() for arr in np.split(vel_res, 2, axis=1)]
    vs_mod = vs_mod / DEF_VP_VS_RATIO
    vs_res = vs_res / DEF_VP_VS_RATIO
    vpvs_mod = vp_mod / vs_mod
    # Calculate the modelled density
    rhob_mod = (
        (1.0 - vsh) * opt_params[7] * scale_val["rho_carb"]
        + vsh * opt_params[10] * scale_val["rho_sh"]
    ) * (1.0 - por) + por * rho_fl
    ai_mod = vp_mod * rhob_mod
    rhob_res = rhob_mod - rhob
    # Save the optimal parameters
    save_opt_params(
        opt_type="exp",
        opt_params=opt_params,
        file_name=file_out_str,
        well_name=well_name,
    )
    if display_results:
        opt_param_to_ascii(
            in_file=file_out_str,
            well_name=well_name,
        )

    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res
