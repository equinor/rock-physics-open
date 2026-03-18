from pathlib import Path
from typing import cast

import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.optimisation_utilities import (
    gen_mod_routine,
    load_opt_params,
    opt_param_info,
)
from rock_physics_open.equinor_utilities.various_utilities.types import Array1D

from .curvefit_t_matrix_exp import curvefit_t_matrix_exp


def run_t_matrix_forward_model_with_opt_params_exp(
    fl_k: Array1D[np.float64],
    fl_rho: Array1D[np.float64],
    phi: Array1D[np.float64],
    vsh: Array1D[np.float64],
    angle: float,
    perm: float,
    visco: float,
    tau: float,
    freq: float,
    f_name: str | Path,
) -> tuple[
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
]:
    """
    Run forward modelling with the T-Matrix model using exponentially fitted parameters.

    Based on the input file with parameters for the optimally fitted model, a forward modelling is done with inputs of mineral properties, fluid properties and porosity per sample
    Other parameters (constants) can also be varied from their setting when the optimal parameters were found.

    Parameters
    ----------
    fl_k
        Effective in situ fluid bulk modulus [Pa].
    fl_rho
        Effective in situ fluid density [kg/m^3].
    phi
        Porosity [fraction].
    vsh
        Shale volume [fraction].
    angle
        Angle of symmetry plane
    perm
        Permeability [mD].
    visco
        Viscosity [cP].
    tau
        Relaxation time constant [s].
    freq
        Signal frequency [Hz].
    f_name
        File name for parameter file for optimal parameters.

    Returns
    -------
    vp
        Modelled p-wave velocity [m/s].
    vs
        Modelled s-wave velocity [m/s].
    rho
        Modelled density [kg/m^3].
    ai
        Modelled acoustic impedance [kg/m^3 x m/s].
    vpvs
        Modelled Vp/Vs ratio [fraction].

    Raises
    ------
    TypeError
        If input file is incorrect.
    ValueError
        If forward model fails.
    """
    opt_type, opt_params, opt_dict = load_opt_params(f_name)
    _, scale_val, _ = opt_param_info()
    phi, angle_, perm_, visco_, tau_, freq_, def_vpvs = cast(
        list[Array1D[np.float64]],
        gen_utilities.dim_check_vector((phi, angle, perm, visco, tau, freq, 1.0)),
    )

    if opt_type != "exp":
        raise TypeError(
            f"{__file__}: incorrect type of optimal parameter input file, must come from EXP optimisation"
        )

    rho_mod = (
        (1.0 - vsh) * opt_dict["rho_carb"] * scale_val["rho_carb"]  # pyright: ignore[reportTypedDictNotRequiredAccess] | Should be there for exp
        + vsh * opt_dict["rho_sh"] * scale_val["rho_sh"]  # pyright: ignore[reportTypedDictNotRequiredAccess] | Should be there for exp
    ) * (1.0 - phi) + phi * fl_rho
    y_shape = (phi.shape[0], 2)

    # No Need for preprocessing
    opt_fcn = curvefit_t_matrix_exp
    # Generate x_data according to method min
    x_data = np.stack(
        (phi, vsh, fl_k, fl_rho, angle_, perm_, visco_, tau_, freq_, def_vpvs), axis=1
    )
    v_mod = gen_mod_routine(
        opt_function=opt_fcn,
        xdata_orig=x_data,
        ydata_shape=y_shape,
        opt_params=opt_params,
    )
    if not isinstance(v_mod, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance] | kept for backwards compatibility
        raise ValueError(f"{__file__}: no solution to forward model")
    vp_mod, vs_mod = [arr.flatten() for arr in np.split(v_mod, 2, axis=1)]
    vpvs_mod = vp_mod / vs_mod
    ai_mod = vp_mod * rho_mod
    return vp_mod, vs_mod, rho_mod, ai_mod, vpvs_mod
