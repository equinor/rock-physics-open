import inspect
import sys
from ctypes import c_double, c_int

import numpy as np
import numpy.ctypeslib as npct
from tmatrix._tmatrix import tmatrix_porosity_noscenario

from rock_physics_open.equinor_utilities import gen_utilities

# Definition of input types for the T Matrix function
# this will affect the tests on the input data, dim_check_vector is therefore set up to
# return data on the specified format
array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags="CONTIGUOUS")
array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags="CONTIGUOUS")
array_2d_double = npct.ndpointer(dtype=c_double, ndim=2, flags="CONTIGUOUS")


# noinspection PyUnusedLocal
def T_matrix_porosity_C_scenario(*args):
    """Deprecated."""
    raise DeprecationWarning(
        "{}: deprecated function, all T-Matrix runs should be parsed through run_t_matrix".format(
            inspect.stack()[0][3]
        )
    )


def t_matrix_porosity_c_alpha_v(
    k_min,
    mu_min,
    rho_min,
    k_fl,
    rho_fl,
    phi,
    perm,
    visco,
    alpha,
    v,
    tau,
    frequency,
    angle,
    frac_inc_con,
    frac_inc_ani,
):
    """This function can be called directly from top level, but the present recommendation is to go though the run_t_matrix
    in order to check inputs. It is used directly from the optimisation functions for efficiency. This gives direct
    access to the C++ compiled library for T-Matrix.

    Parameters
    ----------
    k_min : np.ndarray
        N length array, mineral bulk modulus [Pa]
    mu_min: np.ndarray
        N length array, mineral shear modulus [Pa]
    rho_min: np.ndarray
        N length array, mineral density [kg/m^3]
    k_fl:np.ndarray
        N length array, fluid bulk modulus [Pa]
    rho_fl: np.ndarray
        N length array, fluid density [kg/m^3]
    phi:np.ndaray
        N length array, porosity
    perm: np.ndarray
        N length array, permeability [mD]
    visco: np.ndarray
        N length array, viscosity [cP]
    alpha: np.ndarray or float
        aspect ratios for inclusions
    v: np. ndarray or float
        fraction of porosity with given aspect ratio
    tau: float
        relaxation time
    frequency:  float
        float single value, signal frequency [Hz]
    angle: float
        float single value, angle of symmetry plane (0 = HTI, 90 = VTI medium) [deg]
    frac_inc_con: np.ndarray or float
        float single value or array, fraction of inclusions that are connected
    frac_inc_ani: np.ndarray or float
        float single value or array, fraction of inclusions that are anisotropic

    Returns
    -------
    tuple
        Tuple of np.ndarrays. Vp: Vertical P-wave velocity [m/s], Vsv: Vertical polarity S-wave velocity [m/s],
        Vsh: Horizontal polarity S-wave velocity [m/s], Rhob [kg/m^3].
    """

    # ToDo: tau input is not used in the Calculo implementation of T-Matrix
    del tau

    # Make sure that what can be vectors are vectors of the same length.
    # frac_inc_con and frac_inc_ani can either be the same length as the log, be constants or have the same length as
    # alpha and v

    # frac_inc_con and frac_inc_ani must be of the same length
    frac_inc_con, frac_inc_ani = gen_utilities.dim_check_vector(
        (frac_inc_con, frac_inc_ani)
    )

    # test for frac_inc_con and frac_inc_ani being of the same length as the logs
    log_length = len(phi)
    if frac_inc_con.shape[0] == log_length:
        (
            k_min,
            mu_min,
            rho_min,
            k_fl,
            rho_fl,
            phi,
            perm,
            visco,
            frac_inc_con,
            frac_inc_ani,
        ) = gen_utilities.dim_check_vector(
            (
                k_min,
                mu_min,
                rho_min,
                k_fl,
                rho_fl,
                phi,
                perm,
                visco,
                frac_inc_con,
                frac_inc_ani,
            ),
            force_type=np.dtype("float64"),
        )
        frac_inc_length = log_length
    else:  # Single float value of frac_inc_con, frac_inc_ani or matching number of inclusions
        (
            k_min,
            mu_min,
            rho_min,
            k_fl,
            rho_fl,
            phi,
            perm,
            visco,
        ) = gen_utilities.dim_check_vector(
            (k_min, mu_min, rho_min, k_fl, rho_fl, phi, perm, visco),
            np.dtype("float64"),
        )
        frac_inc_length = frac_inc_ani.shape[0]

    # Create output array, gather mineral and fluid properties in 2D arrays
    out_arr = np.zeros((log_length, 4), dtype=float, order="C")
    min_prop = np.stack([k_min, mu_min, rho_min], axis=1)
    fl_prop = np.stack([k_fl, rho_fl, perm, visco], axis=1)

    # Make sure that alpha and v are of the same shape - more about length of alpha further down
    alpha_shape = alpha.shape
    alpha, v = gen_utilities.dim_check_vector(
        (alpha, v), force_type=np.dtype("float64")
    )
    alpha = alpha.reshape(alpha_shape)
    v = v.reshape(alpha_shape)

    # Number of alphas can vary from sample to sample - not used here, regular number of alphas for all
    # samples. Need to declare the number of alphas per sample in an array. Alpha can also be a constant, in which case
    # it should be expanded to an array of log_length. Both alpha and v should be a 2D array
    if len(alpha) != log_length and len(alpha) < 5:
        # Interpret alpha as a vector of aspect ratios
        alpha_vec = (
            np.ones((log_length, len(alpha)), dtype=float, order="c") * alpha.flatten()
        )
        v_vec = np.ones((log_length, len(alpha)), dtype=float, order="c") * v.flatten()
        alpha = alpha_vec
        v = v_vec

    # Expect 2-dimensional input to t_mat_lib for alpha and v
    if alpha.ndim == 1:
        alpha = alpha.reshape((len(alpha), 1))
        v = v.reshape((len(alpha), 1))

    # Have to declare the number of alphas per sample, even if it is constant
    alpha_length_array = np.ones(log_length, dtype=c_int, order="c") * alpha.shape[1]
    alpha_length = alpha.shape[0]

    try:
        tmatrix_porosity_noscenario(
            out_arr,
            log_length,
            min_prop,
            fl_prop,
            phi,
            alpha,
            v,
            alpha_length_array,
            alpha_length,
            frequency,
            angle,
            frac_inc_con,
            frac_inc_ani,
            frac_inc_length,
        )
    except ValueError:
        # Get more info in case this goes wrong
        raise TypeError(
            "tMatrix:t_matrix_porosity_c_alpha_v: {0}".format(str(sys.exc_info()))
        )

    vp = out_arr[:, 0]
    vsv = out_arr[:, 1]
    vsh = out_arr[:, 2]
    rhob = out_arr[:, 3]

    return vp, vsv, vsh, rhob
