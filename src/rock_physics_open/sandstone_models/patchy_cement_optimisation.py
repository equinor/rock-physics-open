from typing import Callable, cast

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.optimisation_utilities import (
    gen_opt_routine,
    opt_param_to_ascii,
    save_opt_params,
)
from rock_physics_open.sandstone_models.patchy_cement_model import (
    patchy_cement_model_weight,
)

from .curvefit_sandstone_models import (
    curvefit_patchy_cement,
)


def patchy_cement_model_optimisation(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    por: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rhob: npt.NDArray[np.float64],
    phi_c: float,
    file_out_str: str = "patchy_cement_optimal_params.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
    opt_params_only: bool = False,
) -> (
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
    | npt.NDArray[np.float64]
):
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
    p_eff : np.ndarray
        Effective pressure log [Pa].
    vp : np.ndarray
        Compressional velocity log [m/s].
    vs : np.ndarray
        Shear velocity log [m/s].
    rhob : np.ndarray
        Bulk density log [kg/m^3].
    phi_c : float
        Critical porosity [fraction]
    file_out_str : str
        Output file name (string) to store optimal parameters (pickle format).
    display_results : bool
        Display optimal parameters in a window after run.
    well_name : str
        Name of well to be displayed in info box title.
    opt_params_only : bool
        return parameters from optimisation only
    Returns
    -------
    tuple
        vp_mod, vs_mod, rho_mod, ai_mod, vpvs_mod - modelled logs,
        vp_res, vs_res, rho_res - residual logs.
    """

    # Skip hardcoded Vp/Vs ratio
    def_vpvs = np.mean(vp / vs)
    # Set weight to vs to give vp and vs similar influence on optimisation
    y_data = np.stack([vp, vs * def_vpvs], axis=1)
    # Optimisation function for selected parameters
    opt_fun: Callable[..., npt.NDArray[np.float64]] = curvefit_patchy_cement
    # expand single value parameters to match logs length
    por, phi_c_, def_vpvs = cast(
        list[npt.NDArray[np.float64]],
        gen_utilities.dim_check_vector((por, phi_c, def_vpvs)),
    )
    x_data = np.stack(
        (
            k_min,
            mu_min,
            rho_min,
            k_cem,
            mu_cem,
            rho_cem,
            k_fl,
            rho_fl,
            por,
            p_eff,
            def_vpvs,
            phi_c_,
        ),
        axis=1,
    )

    # Params: weight_k, weight_mu, shear_red, frac_cem
    lower_bound = np.array(
        [
            0.0,  # weight_k
            0.0,  # weight_mu
            0.0,  # shear_red
            0.01,  # frac_cem
        ],
        dtype=float,
    )
    upper_bound = np.array(
        [
            1.0,  # weight_k
            1.0,  # weight_mu
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

    # Reshape outputs and remove weight from vs
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(vel_mod, 2, axis=1)]
    vp_res, vs_res = [arr.flatten() for arr in np.split(vel_res, 2, axis=1)]
    vs_mod = vs_mod / def_vpvs
    vs_res = vs_res / def_vpvs
    vpvs_mod = vp_mod / vs_mod
    # Calculate the modelled density
    # rho_cem??
    rhob_mod = rho_min * (1.0 - por) + por * rho_fl
    ai_mod = vp_mod * rhob_mod
    rhob_res = rhob_mod - rhob
    # Save the optimal parameters
    save_opt_params(
        opt_type="pat_cem",
        opt_params=opt_params,
        file_name=file_out_str,
        well_name=well_name,
    )
    if display_results:
        opt_param_to_ascii(in_file=file_out_str, well_name=well_name)
    if opt_params_only:
        return opt_params
    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res


def patchy_cement_model_optimisation_multiwell(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    por: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rhob: npt.NDArray[np.float64],
    phi_c: float,
    file_out_str: str = "pat_cem.pkl",
    display_results: bool = False,
    well_name: str = "Unknown well",
) -> tuple[
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
    list[npt.NDArray[np.float64]],
]:
    # First: perform a calibration with all data, return optimal parameters only. Concatenate all list of arrays to
    # make a single set of input arrays
    k_min_all = np.concatenate(k_min, axis=0)
    mu_min_all = np.concatenate(mu_min, axis=0)
    rho_min_all = np.concatenate(rho_min, axis=0)
    k_cem_all = np.concatenate(k_cem, axis=0)
    mu_cem_all = np.concatenate(mu_cem, axis=0)
    rho_cem_all = np.concatenate(rho_cem, axis=0)
    k_fl_all = np.concatenate(k_fl, axis=0)
    rho_fl_all = np.concatenate(rho_fl, axis=0)
    por_all = np.concatenate(por, axis=0)
    p_eff_all = np.concatenate(p_eff, axis=0)
    vp_all = np.concatenate(vp, axis=0)
    vs_all = np.concatenate(vs, axis=0)
    rhob_all = np.concatenate(rhob, axis=0)

    opt_param = patchy_cement_model_optimisation(
        k_min=k_min_all,
        mu_min=mu_min_all,
        rho_min=rho_min_all,
        k_cem=k_cem_all,
        mu_cem=mu_cem_all,
        rho_cem=rho_cem_all,
        k_fl=k_fl_all,
        rho_fl=rho_fl_all,
        por=por_all,
        p_eff=p_eff_all,
        vp=vp_all,
        vs=vs_all,
        rhob=rhob_all,
        phi_c=phi_c,
        file_out_str=file_out_str,
        display_results=display_results,
        well_name=well_name,
        opt_params_only=True,
    )

    # Next: run patchy cement model for all wells using the optimal parameters
    weight_k = opt_param[0]
    weight_mu = opt_param[1]
    shear_red = opt_param[2]
    frac_cem = opt_param[3]
    vp_mod: list[npt.NDArray[np.float64]] = []
    vs_mod: list[npt.NDArray[np.float64]] = []
    rhob_mod: list[npt.NDArray[np.float64]] = []
    ai_mod: list[npt.NDArray[np.float64]] = []
    vpvs_mod: list[npt.NDArray[np.float64]] = []
    vp_res: list[npt.NDArray[np.float64]] = []
    vs_res: list[npt.NDArray[np.float64]] = []
    rhob_res: list[npt.NDArray[np.float64]] = []
    for i in range(len(k_min)):
        vp_tmp, vs_tmp, rhob_tmp, _, _ = patchy_cement_model_weight(
            k_min=k_min[i],
            mu_min=mu_min[i],
            rho_min=rho_min[i],
            k_cem=k_cem[i],
            mu_cem=mu_cem[i],
            rho_cem=rho_cem[i],
            k_fl=k_fl[i],
            rho_fl=rho_fl[i],
            phi=por[i],
            p_eff=p_eff[i],
            frac_cem=frac_cem,
            phi_c=phi_c,
            coord_num_func="PorBased",
            n=9,
            shear_red=shear_red,
            weight_k=weight_k,
            weight_mu=weight_mu,
        )
        vp_mod.append(vp_tmp)
        vs_mod.append(vs_tmp)
        rhob_mod.append(rhob_tmp)
        ai_mod.append(vp_tmp * rhob_tmp)
        vpvs_mod.append(vp_tmp / vs_tmp)
        vp_res.append(vp_tmp - vp[i])
        vs_res.append(vs_tmp - vs[i])
        rhob_res.append(rhob_tmp - rhob[i])

    return vp_mod, vs_mod, rhob_mod, ai_mod, vpvs_mod, vp_res, vs_res, rhob_res
