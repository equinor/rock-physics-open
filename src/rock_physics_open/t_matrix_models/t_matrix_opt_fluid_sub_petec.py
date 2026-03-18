from pathlib import Path
from typing import cast

import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.optimisation_utilities import (
    gen_mod_routine,
    gen_sub_routine,
    load_opt_params,
)
from rock_physics_open.equinor_utilities.various_utilities.types import Array1D

from .curvefit_t_matrix_min import curve_fit_2_inclusion_sets


def run_t_matrix_with_opt_params_petec(
    min_k: Array1D[np.float64],
    min_mu: Array1D[np.float64],
    min_rho: Array1D[np.float64],
    fl_k_orig: Array1D[np.float64],
    fl_rho_orig: Array1D[np.float64],
    fl_k_sub: Array1D[np.float64],
    fl_rho_sub: Array1D[np.float64],
    vp: Array1D[np.float64],
    vs: Array1D[np.float64],
    rhob: Array1D[np.float64],
    phi: Array1D[np.float64],
    angle: float,
    perm: float,
    visco: float,
    tau: float,
    freq: float,
    f_name: str | Path,
    fluid_sub: bool = True,
) -> tuple[
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
]:
    """
    Based on the input file with parameters for the optimally fitted model, the correct modelling version is run.

    Fluid substitution follows, in case it is selected. If not, the vp_sub and vs_sub will contain the same values as the input logs.

    Parameters
    ----------
    min_k
        Effective mineral bulk modulus [Pa].
    min_mu
        Effective mineral shear modulus [Pa].
    min_rho
        Effective mineral density [kg/m^3].
    fl_k_orig
        Effective in situ fluid bulk modulus [Pa].
    fl_rho_orig
        Effective in situ fluid density [kg/m^3].
    fl_k_sub
        Effective substituted fluid bulk modulus [Pa].
    fl_rho_sub
        Effective substituted density [kg/m^3].
    vp
        Compressional velocity [m/s].
    vs
        Shear velocity [m/s].
    rhob
        Bulk density [kg/m^3].
    phi
        Porosity [fraction].
    angle
        Angle of symmetry plane [degrees]
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
    fluid_sub
        Boolean parameter to perform fluid substitution.

    Returns
    -------
    vp_sub
        Fluid substituted p-wave velocity [m/s].
    vs_sub
        Fluid substituted s-wave velocity [m/s].
    rho_sub
        Fluid substituted density [kg/m^3].
    ai_sub
        Fluid substituted acoustic impedance [kg/m^3 x m/s].
    vpvs_sub
        Fluid substituted Vp/Vs ratio [ratio].
    vp_mod
        P-wave velocity from the optimal fitted model [m/s].
    vs_mod
        S-wave velocity from the optimal fitted model [m/s].
    rho_mod
        Density from the optimal fitted model [kg/m^3].
    vp_res
        P-wave residual (observed minus modelled) [m/s].
    vs_res
        S-wave residual (observed minus modelled) [m/s].
    rho_res
        Density residual (observed minus modelled) [kg/m^3].
    """
    _, opt_params, _ = load_opt_params(f_name)
    y_data = np.stack([vp, vs], axis=1)
    y_shape = y_data.shape
    phi, angle_, perm_, visco_, tau_, freq_, def_vpvs = cast(
        list[Array1D[np.float64]],
        gen_utilities.dim_check_vector((phi, angle, perm, visco, tau, freq, 1.0)),
    )

    rho_sub = rhob + (fl_rho_sub - fl_rho_orig) * phi
    # Set None values for inputs that will be defined in the different cases
    x_data_new = None

    opt_fcn = curve_fit_2_inclusion_sets
    # Generate x_data according to method min
    x_data = np.stack(
        (
            phi,
            min_k,
            min_mu,
            min_rho,
            fl_k_orig,
            fl_rho_orig,
            angle_,
            perm_,
            visco_,
            tau_,
            freq_,
            def_vpvs,
        ),
        axis=1,
    )
    if fluid_sub:
        x_data_new = np.stack(
            (
                phi,
                min_k,
                min_mu,
                min_rho,
                fl_k_sub,
                fl_rho_sub,
                angle_,
                perm_,
                visco_,
                tau_,
                freq_,
                def_vpvs,
            ),
            axis=1,
        )

        v_sub, v_mod, v_res = gen_sub_routine(
            opt_function=opt_fcn,
            xdata_orig=x_data,
            xdata_new=x_data_new,
            ydata=y_data,
            opt_params=opt_params,
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

    rho_mod = min_rho * (1.0 - phi) + fl_rho_orig * phi
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
