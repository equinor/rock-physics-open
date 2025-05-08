import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities

from .curvefit_t_matrix_exp import curvefit_t_matrix_exp
from .opt_subst_utilities import (
    gen_mod_routine,
    gen_sub_routine,
    load_opt_params,
    opt_param_info,
)


def run_t_matrix_with_opt_params_exp(
    fl_k_orig,
    fl_rho_orig,
    fl_k_sub,
    fl_rho_sub,
    vp,
    vs,
    rhob,
    phi,
    vsh,
    angle,
    perm,
    visco,
    tau,
    freq,
    f_name,
    fluid_sub=True,
):
    """Based on the input file with parameters for the optimally fitted model, the correct modelling version is run.
    Fluid substitution follows, in case it is selected. If not, the vp_sub and vs_sub will contain the same values as
    the input logs.

    Parameters
    ----------
    fl_k_orig : np.ndarray
        Effective in situ fluid bulk modulus [Pa].
    fl_rho_orig : np.ndarray
        Effective in situ fluid density [kg/m^3].
    fl_k_sub : np.ndarray
        Effective substituted fluid bulk modulus [Pa].
    fl_rho_sub : np.ndarray
        Effective substituted density [kg/m^3].
    vp : np.ndarray
        Compressional velocity [m/s].
    vs : np.ndarray
        Shear velocity [m/s].
    rhob : np.ndarray
        Bulk density [kg/m^3].
    phi : np.ndarray
        Porosity [fraction].
    vsh : np.ndarray
        Shale volume [fraction].
    angle : float
        Angle of symmetry plane [degrees]
    perm : float
        Permeability [mD].
    visco : float
        Viscosity [cP].
    tau : float
        Relaxation time constant [s].
    freq : float
        Signal frequency [Hz].
    f_name : str
        File name for parameter file for optimal parameters.
    fluid_sub : bool
        Boolean parameter to perform fluid substitution.

    Returns
    -------
    tuple
        Tuple of np.ndarrays: vp and vs for pressure substituted case, vp, vs and density for fluid substituted case, vp and vs for
        optimal fitted model, vp and vs residuals (observed logs minus modelled values).
    """

    opt_type, opt_params, opt_dict = load_opt_params(f_name)
    y_data = np.stack([vp, vs], axis=1)
    y_shape = y_data.shape
    phi, angle, perm, visco, tau, freq, def_vpvs = gen_utilities.dim_check_vector(
        (phi, angle, perm, visco, tau, freq, 1.0)
    )

    rho_sub = rhob + (fl_rho_sub - fl_rho_orig) * phi
    # Set None values for inputs that will be defined in the different cases
    x_data_new = None

    opt_fcn = curvefit_t_matrix_exp
    scale_val = opt_param_info()[1]
    # Generate x_data according to method exp
    x_data = np.stack(
        (phi, vsh, fl_k_orig, fl_rho_orig, angle, perm, visco, tau, freq, def_vpvs),
        axis=1,
    )
    if fluid_sub:
        x_data_new = np.stack(
            (phi, vsh, fl_k_sub, fl_rho_sub, angle, perm, visco, tau, freq, def_vpvs),
            axis=1,
        )
    rho_mod = (
        (1.0 - vsh) * opt_dict["rho_carb"] * scale_val["rho_carb"]
        + vsh * opt_dict["rho_sh"] * scale_val["rho_sh"]
    ) * (1.0 - phi) + phi * fl_rho_orig

    if fluid_sub:
        v_sub, v_mod, v_res = gen_sub_routine(
            opt_fcn, x_data, x_data_new, y_data, opt_params
        )
        vp_sub, vs_sub = [arr.flatten() for arr in np.split(v_sub, 2, axis=1)]
        vp_mod, vs_mod = [arr.flatten() for arr in np.split(v_mod, 2, axis=1)]
        vp_res, vs_res = [arr.flatten() for arr in np.split(v_res, 2, axis=1)]
    else:
        v_mod = gen_mod_routine(opt_fcn, x_data, y_shape, opt_params)
        vp_mod, vs_mod = [arr.flatten() for arr in np.split(v_mod, 2, axis=1)]
        vp_sub = vp
        vs_sub = vs
        vp_res = vp_mod - vp
        vs_res = vs_mod - vs

    rho_res = rho_mod - rhob
    ai_sub = vp_sub * rho_sub
    vpvs_sub = vp_sub / vs_sub

    return (
        vp_sub,
        vs_sub,
        rho_sub,
        ai_sub,
        vpvs_sub,
        vp_mod,
        vs_mod,
        rho_mod,
        vp_res,
        vs_res,
        rho_res,
    )
