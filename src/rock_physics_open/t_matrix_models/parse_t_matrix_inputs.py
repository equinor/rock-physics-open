import sys
from typing import Any, Literal, cast

import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities
from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    TMatrixCallable,
)

from .t_matrix_C import t_matrix_porosity_c_alpha_v


def parse_t_matrix_inputs(
    k_min: Array1D[np.float64],
    mu_min: Array1D[np.float64],
    rho_min: Array1D[np.float64],
    k_fl: Array1D[np.float64],
    rho_fl: Array1D[np.float64],
    phi: Array1D[np.float64],
    perm: Array1D[np.float64] | float,
    visco: Array1D[np.float64] | float,
    alpha: Array1D[np.float64] | Array2D[np.float64],
    v: Array1D[np.float64] | Array2D[np.float64],
    tau: Array1D[np.float64],
    frequency: float,
    angle: float,
    frac_inc_con: Array1D[np.float64] | float,
    frac_inc_ani: Array1D[np.float64] | float,
    pressure: Array1D[np.float64] | None,
    scenario: int | None,
    fcn: TMatrixCallable | str | None,
) -> tuple[
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64],
    Array2D[np.float64],
    Array2D[np.float64],
    Array1D[np.float64],
    float,
    float,
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.float64] | None,
    TMatrixCallable,
    Literal[0, 1, 2],
    Literal[0, 1, 2],
]:
    """Function to do all necessary checking of input type, dimension, reshaping etc. that clutter up the start of T-Matrix.

    NB: Setting scenario will override the settings for alpha, v and tau.

    Parameters
    ----------
    k_min
        N length numpy array, mineral bulk modulus [Pa].
    mu_min
        N length numpy array, mineral shear modulus [Pa].
    rho_min
        N length numpy array, mineral density [kg/m^3].
    k_fl
        N length numpy array, fluid bulk modulus [Pa].
    rho_fl
        N length numpy array, fluid density [kg/m^3].
    phi
        N length numpy array, total porosity [ratio].
    perm
        float or N length numpy array, permeability [mD].
    visco
        float or N length numpy array, fluid viscosity [cP].
    alpha
        M or NxM length numpy array, inclusion aspect ratio [ratio].
    v
        M or NxM length numpy array, inclusion concentration [ratio].
    tau
        M length numpy array, relaxation time [s].
    frequency
        single float, signal frequency [Hz].
    angle
        single float, angle of symmetry plane [degree].
    frac_inc_con
        single float or N length numpy array, fraction of connected inclusions [ratio].
    frac_inc_ani
        single float or N length numpy array, fraction of anisotropic inclusions [ratio].
    pressure
         > 1 value list or numpy array in ascending order, effective pressure [Pa].
    scenario
        pre-set scenarios for alpha, v and tau
    fcn
        function with which to run the T-Matrix model or string with function name within t_matrix_models. If None, the C++ implementation is used.

    Returns
    -------
    k_min
        Mineral bulk modulus, validated and reshaped [Pa].
    mu_min
        Mineral shear modulus, validated and reshaped [Pa].
    rho_min
        Mineral density, validated and reshaped [kg/m^3].
    k_fl
        Fluid bulk modulus, validated and reshaped [Pa].
    rho_fl
        Fluid density, validated and reshaped [kg/m^3].
    phi
        Porosity, validated and reshaped [fraction].
    perm
        Permeability, validated and reshaped.
    visco
        Viscosity, validated and reshaped.
    tau
        Relaxation time array, validated and reshaped.
    alpha
        Aspect ratios, validated and reshaped.
    v
        Inclusion volumes, validated and reshaped.
    frequency
        Frequency value.
    angle
        Incidence angle value.
    frac_inc_con
        Connected inclusion fraction, validated and reshaped.
    frac_inc_ani
        Anisotropic inclusion fraction, validated and reshaped.
    pressure
        Pressure, validated and reshaped (or None if unused).
    t_matrix_fcn
        Resolved T-Matrix callable.
    ctrl_connected
        Control parameter for connected inclusions (0, 1 or 2).
    ctrl_anisotropy
        Control parameter for anisotropic inclusions (0, 1 or 2).

    Raises
    ------
    ValueError
        If inputs have wrong type, dimension, or invalid values.
    """

    def _assert_type(
        arg: Any,
        exp_dtype: type | tuple[type, ...],
        err_str: str = "check inputs, wrong type encountered",
    ) -> None:
        """Assert type.

        Parameters
        ----------
        arg
            To be asserted.
        exp_dtype
            Expected data type.
        err_str
            Error string, by default 'check inputs, wrong type encountered'.

        Raises
        ------
        ValueError
            Content of err_str.
        """
        if not isinstance(arg, exp_dtype):
            raise ValueError("t-matrix inputs: " + err_str)

    # 1: Check all single float values, silently cast int to float
    # Permeability and viscosity: Check that they are floats and convert them to to SI units
    _assert_type(
        arg=perm,
        exp_dtype=(float, int),
        err_str="expect permeability given as single float in units mD",
    )
    perm = perm * 0.986923e-15

    _assert_type(
        arg=visco,
        exp_dtype=(float, int),
        err_str="expect viscosity given as single float in units cP",
    )
    visco = visco * 1.0e-2

    _assert_type(
        arg=frequency,
        exp_dtype=(float, int),
        err_str="expect frequency given as single float value in Hz",
    )
    frequency = float(frequency)

    _assert_type(
        arg=angle,
        exp_dtype=(float, int),
        err_str="expect angle given as single float value in degrees",
    )
    angle = float(angle)

    # 2: Determine the T-Matrix function, use the C++ implementation as default in case of None
    # If it given as a string, it must belong to the t_matrix_models module
    if not fcn:  # None
        t_matrix_fcn = t_matrix_porosity_c_alpha_v
    elif not callable(fcn):
        fcn_err_str = (
            "T-Matrix function should be given as the callable function or a string "
            "to the function name within t_matrix_models "
        )
        _assert_type(fcn, str, fcn_err_str)
        # Import here to avoid circular import
        import importlib

        t_matrix_models = importlib.import_module("rock_physics_open.t_matrix_models")

        if not hasattr(t_matrix_models, fcn):
            raise ValueError(fcn_err_str)
        t_matrix_fcn = getattr(t_matrix_models, fcn)

    else:  # Function must be recognised
        if not fcn:  # check that the function exists
            raise ValueError(
                "t-matrix inputs: function for T-Matrix model is not known"
            )
        t_matrix_fcn = fcn

    # 3: Check all inputs that should be one value per sample, expand perm and visco to match this.
    # frac_inc_ani and frac_inc_con can be single floats or vectors of the same length as the other vector inputs

    log_length = len(phi)

    # Check that the inputs that should have the same length actually do
    # dim_check_vector may throw an error message, modify it to show origin
    try:
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
            force_type=np.dtype(np.float64),
        )
    except ValueError:
        raise ValueError("t-matrix inputs: {}".format(str(sys.exc_info())))

    # 4: Scenario will override settings for alpha, v and tau
    if scenario:
        scenario = int(scenario)
        if scenario == 1:  # Mostly rounded pores
            alpha = np.array([0.9, 0.1])
            v = np.array([0.9, 0.1])
        elif scenario == 2:  # Dual porosity with little rounded pores
            alpha = np.array([0.58, 0.027])
            v = np.array([0.85, 0.15])
        elif scenario == 3:  # Mixed pores
            alpha = np.array([0.9, 0.1, 0.01])
            v = np.array([0.8, 0.19, 0.01])
        elif scenario == 4:  # Flat pores and cracks
            alpha = np.array([0.9, 0.1, 0.01, 0.001])
            v = np.array([0.689, 0.3, 0.01, 0.001])
        tau = np.ones_like(alpha) * 1.0e-7

    # 4a: Check alpha, v, tau
    # alpha and v is either 1D vector with inclusion aspect ratios and proportions applying to all samples or 2D arrays
    # where the first dimension must coincide with the log_length
    # tau should be a 1D vector with length equal to the number of inclusions per sample
    _assert_type(
        arg=alpha,
        exp_dtype=np.ndarray,
        err_str="alpha should be a 1D or 2D numpy array of matching size with v",
    )
    _assert_type(
        arg=v,
        exp_dtype=np.ndarray,
        err_str="v should be a 1D or 2D numpy array og matching size with alpha",
    )
    _assert_type(
        arg=tau,
        exp_dtype=(float, np.ndarray),
        err_str="tau should be a single float or 1D numpy array with length matching the number of inclusions",
    )

    alpha_shape = alpha.shape
    # First make sure that alpha and v have the same number of elements
    try:
        alpha, v = gen_utilities.dim_check_vector(
            (alpha, v), force_type=np.dtype(np.float64)
        )
    except ValueError:
        raise ValueError("t-matrix inputs: {}".format(str(sys.exc_info())))
    alpha = cast(Array1D[np.float64] | Array2D[np.float64], alpha.reshape(alpha_shape))
    v = cast(Array1D[np.float64] | Array2D[np.float64], v.reshape(alpha_shape))
    # If they are 2D arrays, the first dimension should match log_length
    if alpha.ndim == 2:
        if not (alpha_shape[0] == log_length or alpha_shape[0] == 1):
            raise ValueError(
                "t-matrix inputs: number of samples in alpha is not matching length of inputs"
            )
        if alpha_shape[0] == 1:
            alpha = np.tile(alpha, (log_length, 1))
            v = np.tile(v, (log_length, 1))

    elif alpha.ndim == 1:
        alpha = np.tile(alpha.reshape(1, alpha_shape[0]), (log_length, 1))
        v = np.tile(v.reshape(1, alpha_shape[0]), (log_length, 1))
    alpha = cast(Array2D[np.float64], alpha)
    v = cast(Array2D[np.float64], v)
    # Check that the number of elements in tau matches dim 1 of alpha
    if isinstance(tau, float):
        tau = np.ones(alpha_shape) * tau
    else:
        if not (tau.ndim == 1 and tau.shape[0] == alpha.shape[1]):
            raise ValueError(
                "t-matrix inputs: number of elements in tau is not matching number of inclusions"
            )

    # 5: Check pressure
    # Check that pressure either consists of at least two elements in ascending order or is None
    # in the case there is no pressure modelling
    if pressure is not None:
        _assert_type(
            arg=pressure,
            exp_dtype=(list, np.ndarray),
            err_str="pressure should be a list or numpy array of at least two elements in ascending order",
        )
        pressure = np.sort(np.array(pressure).flatten())
        if not pressure.shape[0] > 1:
            raise ValueError(
                "t-matrix inputs: if pressure is given, it should contain two or more values"
            )

    # 6: Determine case for connected/isolated and isotropic/anisotropic inclusions
    # Assume the most general case: mix of isotropic and anisotropic inclusions and connected and isolated inclusions
    ctrl_anisotropy: Literal[0, 1, 2] = 1
    if np.all(frac_inc_ani == 0):
        ctrl_anisotropy = 0  # All isotropic
        angle = 0.0
    elif np.all(frac_inc_ani == 1):
        ctrl_anisotropy = 2  # All anisotropic

    ctrl_connected: Literal[0, 1, 2] = 1
    # Create case based on amount of connected inclusions
    if np.all(frac_inc_con == 0):
        ctrl_connected = 0  # All isolated
    elif np.all(frac_inc_con == 1):
        ctrl_connected = 2  # All connected

    return (
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
        pressure,
        t_matrix_fcn,
        ctrl_connected,
        ctrl_anisotropy,
    )
