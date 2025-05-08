import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.t_matrix_models import gen_opt_routine, save_opt_params

from .curvefit_sandstone_models import curvefit_constant_cement


def constant_cement_model_optimisation(
    k_min: np.ndarray,
    mu_min: np.ndarray,
    rho_min: np.ndarray,
    k_cem: np.ndarray,
    mu_cem: np.ndarray,
    rho_cem: np.ndarray,
    k_fl: np.ndarray,
    rho_fl: np.ndarray,
    por: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    rhob: np.ndarray,
    file_out_str: str = "constant_cement_optimal_params.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
):
    """Patchy cement model with optimisation for a selection of parameters.

    Parameters
    ----------
    k_min :
        Cement bulk modulus [Pa].
    mu_min :
        Cement shear modulus [Pa].
    rho_min :
        Cement density [kg/m^3].
    k_cem :
        Cement bulk modulus [Pa].
    mu_cem :
        Cement shear modulus [Pa].
    rho_cem :
        Cement density [kg/m^3].
    k_fl :
        Fluid bulk modulus [Pa].
    rho_fl :
        Fluid density [kg/m^3].
    por :
        Inclusion porosity [ratio].
    vp :
        Compressional velocity log [m/s].
    vs :
        Shear velocity log [m/s].
    rhob :
        Bulk density log [kg/m^3].
    file_out_str :
        Output file name (string) to store optimal parameters (pickle format).
    display_results :
        Display optimal parameters in a window after run.
    well_name :
        Name of well to be displayed in info box title.

    Returns
    -------
    tuple
        vp_mod, vs_mod - modelled logs, vp_res, vs_res - residual logs.
    """

    # Skip hardcoded Vp/Vs ratio
    def_vpvs = np.mean(vp / vs)
    # Set weight to vs to give vp and vs similar influence on optimisation
    y_data = np.stack([vp, vs * def_vpvs], axis=1)
    # Optimisation function for selected parameters
    opt_fun = curvefit_constant_cement
    # expand single value parameters to match logs length
    por, def_vpvs = gen_utilities.dim_check_vector((por, def_vpvs))
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
        opt_fun, x_data, y_data, x0, lower_bound, upper_bound
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
    save_opt_params("const_cem", opt_params, file_out_str, well_name=well_name)
    if display_results:
        from rock_physics_open.t_matrix_models import opt_param_to_ascii

        opt_param_to_ascii(file_out_str, well_name=well_name)

    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res
