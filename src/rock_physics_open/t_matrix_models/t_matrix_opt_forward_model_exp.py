import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities

from .curvefit_t_matrix_exp import curvefit_t_matrix_exp
from .opt_subst_utilities import gen_mod_routine, load_opt_params, opt_param_info


def run_t_matrix_forward_model_with_opt_params_exp(
    fl_k, fl_rho, phi, vsh, angle, perm, visco, tau, freq, f_name
):
    """Based on the input file with parameters for the optimally fitted model, a forward modelling is done
    with inputs of mineral properties, fluid properties and porosity per sample. Other parameters (constants)
    can also be varied from their setting when the optimal parameters were found.

    Parameters
    ----------
    fl_k : np.ndarray.
        Effective in situ fluid bulk modulus [Pa].
    fl_rho : np.ndarray
        Effective in situ fluid density [kg/m^3].
    phi : np.ndarray
        Porosity [fraction].
    vsh : np.ndarray
        Shale volume [fraction].
    angle : float
        Angle of symmetry plane
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

    Returns
    -------
    tuple
        Tuple of np.ndarrays: vp [m/s], vs [m/s], rho [kg/m^3], ai [kg/m^3 x m/s], vp/vs [fraction] for forward model.
    """
    opt_type, opt_params, opt_dict = load_opt_params(f_name)
    scale_val = opt_param_info()[1]
    phi, angle, perm, visco, tau, freq, def_vpvs = gen_utilities.dim_check_vector(
        (phi, angle, perm, visco, tau, freq, 1.0)
    )
    rho_mod = (
        (1.0 - vsh) * opt_dict["rho_carb"] * scale_val["rho_carb"]
        + vsh * opt_dict["rho_sh"] * scale_val["rho_sh"]
    ) * (1.0 - phi) + phi * fl_rho
    y_shape = (phi.shape[0], 2)

    if opt_type != "exp":
        raise ValueError(
            f"{__file__}: incorrect type of optimal parameter input file, must come from EXP "
            f"optimisation"
        )
    # No Need for preprocessing
    opt_fcn = curvefit_t_matrix_exp
    # Generate x_data according to method min
    x_data = np.stack(
        (phi, vsh, fl_k, fl_rho, angle, perm, visco, tau, freq, def_vpvs), axis=1
    )
    v_mod = gen_mod_routine(opt_fcn, x_data, y_shape, opt_params)
    if not isinstance(v_mod, np.ndarray):
        raise ValueError(f"{__file__}: no solution to forward model")
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(v_mod, 2, axis=1)]
    vpvs_mod = vp_mod / vs_mod
    ai_mod = vp_mod * rho_mod
    return vp_mod, vs_mod, rho_mod, ai_mod, vpvs_mod
