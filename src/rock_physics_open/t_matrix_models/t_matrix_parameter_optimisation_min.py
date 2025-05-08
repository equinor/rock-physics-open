import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities

from .curvefit_t_matrix_min import curve_fit_2_inclusion_sets
from .opt_subst_utilities import gen_opt_routine, save_opt_params

# Trade-off between calcite, dolomite and quartz, vs is weighted by this in order to make it count as much as vp
# in the optimisation
DEF_VP_VS_RATIO = 1.8


def t_matrix_optimisation_petec(
    k_min: np.ndarray,
    mu_min: np.ndarray,
    rho_min: np.ndarray,
    k_fl: np.ndarray,
    rho_fl: np.ndarray,
    por: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    rhob: np.ndarray,
    angle: float = 0.0,
    k_r: float = 50.0,
    eta_f: float = 1.0,
    tau: float = 1.0e-7,
    freq: float = 1.0e3,
    file_out_str: str = "opt_params_min.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
    **opt_kwargs,
):
    """T-Matrix optimisation adapted to a case with detailed information available, such as in a development or production
    setting. Mineral and fluid composition should be known on a sample basis. Inclusion parameters are regarded as
    unknown and they are optimised for.

    Parameters
    ----------
    k_min :
        Effective mineral bulk modulus [Pa].
    mu_min :
        Effective mineral shear modulus [Pa].
    rho_min :
        Effective mineral bulk density [kg/m^3].
    k_fl :
        Effective fluid bulk modulud [Pa].
    rho_fl :
        Effective fluid density [kg/m^3].
    por :
        Inclusion porosity [ratio].
    vp :
        Compressional velocity log [m/s].
    vs :
        Shear velocity log [m/s].
    rhob :
        Bulk density log [kg/m^3].
    angle : float
        Angle of symmetry plane [degrees]
    k_r :
        Permeability [mD].
    eta_f :
        Fluid viscosity [cP].
    tau :
        Relaxation time constant [s].
    freq :
        Signal frequency [Hz].
    file_out_str :
        Output file name (string) to store optimal parameters (pickle format).
    display_results :
        Display optimal parameters in a window after run.
    well_name:
        Name of well to be displayed in info box title.
    opt_kwargs:
        Additional keywords to be passed to optimisation function

    Returns
    -------
    tuple
        vp_mod, vs_mod, rho_mod, ai_mod, vpvs_mod - modelled logs, vp_res, vs_res, rho_res - residual logs.
    """
    # 1. Preparation that is independent of search for minimum possible aspect ratio for second inclusion set

    # Optimisation function for selected parameters, with effective mineral properties known and 2 inclusion sets
    opt_fun = curve_fit_2_inclusion_sets
    rhob_mod = rho_min * (1 - por) + rho_fl * por
    rhob_res = rhob - rhob_mod
    # PETEC adapted inputs: include fluid data and other params in x_data
    por, angle, k_r, eta_f, tau, freq, def_vp_vs_ratio = gen_utilities.dim_check_vector(
        (por, angle, k_r, eta_f, tau, freq, DEF_VP_VS_RATIO)
    )
    x_data = np.stack(
        (
            por,
            k_min,
            mu_min,
            rho_min,
            k_fl,
            rho_fl,
            angle,
            k_r,
            eta_f,
            tau,
            freq,
            def_vp_vs_ratio,
        ),
        axis=1,
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
    while not valid_result and percentiles:
        try:
            # Make sure that parameters are not in conflict with T Matrix assumptions
            min_v1 = 0.5
            max_por = np.percentile(por, percentiles[0])
            min_a2 = (1.0 - min_v1) * max_por
            # Test with all parameters in the range 0.0 - 1.0 for best optimiser performance
            # Params:               f_ani f_con a1   a2    v1
            lower_bound = np.array([0.0, 0.0, 0.5, min_a2, min_v1], dtype=float)
            upper_bound = np.array([1.0, 1.0, 1.0, 0.30, 1.0], dtype=float)
            x0 = (upper_bound + lower_bound) / 2.0
            # Optimisation step without fluid substitution
            vel_mod, vel_res, opt_params = gen_opt_routine(
                opt_fun, x_data, y_data, x0, lower_bound, upper_bound, **opt_kwargs
            )
            valid_result = True
        except ValueError:
            percentiles.pop(0)
            valid_result = False

    if not valid_result:
        raise ValueError(
            f"{__file__}: unable to find stable value for T Matrix optimisation, second inclusion"
        )

    # Reshape outputs and remove weight from vs
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(vel_mod, 2, axis=1)]
    ai_mod = vp_mod * rhob_mod
    vp_res, vs_res = [arr.flatten() for arr in np.split(vel_res, 2, axis=1)]
    vs_mod = vs_mod / DEF_VP_VS_RATIO
    vs_res = vs_res / DEF_VP_VS_RATIO
    vpvs_mod = vp_mod / vs_mod
    # Save the optimal parameters
    save_opt_params("min", opt_params, file_out_str, well_name=well_name)
    if display_results:
        from .opt_subst_utilities import opt_param_to_ascii

        opt_param_to_ascii(file_out_str, well_name=well_name)

    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res
