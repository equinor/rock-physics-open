import warnings

import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector


def gassmann(k_dry, por, k_fl, k_min):
    """
    Fluid substitution according to the Gassmann equation.

    Parameters
    ----------
    k_dry : np.ndarray
        Dry rock bulk modulus [Pa].
    por : np.ndarray
        Porosity [fraction].
    k_fl : np.ndarray
        Fluid bulk modulus.
    k_min : np.ndarray
        Mineral bulk modulus [Pa].

    Returns
    -------
    np.ndarray
        k_sat: bulk modulus for saturated rock [Pa]
    """
    k_dry, por, k_fl, k_min = dim_check_vector((k_dry, por, k_fl, k_min))

    idx = np.logical_or(k_dry == k_min, por == 0)
    k_sat = np.ones(k_dry.shape) * np.nan
    b = k_dry[~idx] / (k_min[~idx] - k_dry[~idx]) + k_fl[~idx] / (
        (k_min[~idx] - k_fl[~idx]) * por[~idx]
    )
    idx1 = b < 0
    if any(idx1):
        b[idx1] = np.nan

    k_sat[~idx] = b / (1 + b) * k_min[~idx]
    k_sat[idx] = k_min[idx]

    return k_sat


def gassmann2(k_sat_1, k_fl_1, k_fl_2, por, k_min):
    """
    Fluid substitution by Gassmann method with substitution of one fluid to another

    Parameters
    ----------
    k_sat_1 : np.ndarray
        bulk modulus for saturated rock with original fluid [Pa]
    k_fl_1 : np.ndarray
        bulk modulus for original fluid [Pa]
    k_fl_2 : np.ndarray
        bulk modulus for replaced fluid [Pa]
    por : np.ndarray
        porosity of rock [fraction]
    k_min : np.ndarray
        mineral bulk modulus of rock [Pa]

    Returns
    -------
    np.ndarray
        k_sat_2: bulk modulus of rock saturated with replaced fluid [Pa]
    """
    k_sat_1, k_fl_1, k_fl_2, por, k_min = dim_check_vector(
        (k_sat_1, k_fl_1, k_fl_2, por, k_min)
    )

    idx = np.any(
        np.array([k_sat_1 == k_min, por == 0, k_fl_1 == k_min, k_fl_2 == k_min]), axis=0
    )

    k_sat_2 = np.ones(k_sat_1.shape) * np.nan

    b = (
        k_fl_2[~idx] / (por[~idx] * (k_min[~idx] - k_fl_2[~idx]))
        - k_fl_1[~idx] / (por[~idx] * (k_min[~idx] - k_fl_1[~idx]))
        + k_sat_1[~idx] / (k_min[~idx] - k_sat_1[~idx])
    )

    idx1 = b < 0
    if any(idx1):
        warn_str = (
            "{0:d} unstable solution(s) to Gassmann equation, changed to NaN".format(
                np.sum(idx1)
            )
        )
        warnings.warn(warn_str, UserWarning)
        b[idx1] = np.nan

    k_sat_2[~idx] = b / (1 + b) * k_min[~idx]
    k_sat_2[idx] = k_sat_1[idx]

    return k_sat_2


def gassmann_dry(k_sat, por, k_fl, k_min):
    """
    Dry rock properties of saturated rock by Gassmann equation

    Parameters
    ----------
    k_sat : np.ndarray
        saturated rock bulk modulus [Pa]
    por : np.ndarray
        porosity of rock [fraction]
    k_fl : np.ndarray
        bulk modulus of fluid [Pa]
    k_min : np.ndarray
        bulk modulus of mineral [Pa]

    Returns
    -------
    np.ndarray
        k_dry: dry rock bulk modulus [Pa]
    """
    k_sat, por, k_fl, k_min = dim_check_vector((k_sat, por, k_fl, k_min))

    idx = np.any(np.array([k_sat == k_min, por == 0, k_fl == k_min]), axis=0)
    k_dry = np.ones(k_sat.shape)
    b = k_sat[~idx] / (k_min[~idx] - k_sat[~idx]) - k_fl[~idx] / (
        (k_min[~idx] - k_fl[~idx]) * por[~idx]
    )

    idx1 = b < 0
    if any(idx1):
        warn_str = (
            "{0:d} unstable solution(s) to Gassmann equation, changed to NaN".format(
                np.sum(idx1)
            )
        )
        warnings.warn(warn_str, UserWarning)
        b[idx1] = np.nan

    k_dry[~idx] = b / (1 + b) * k_min[~idx]
    k_dry[idx] = k_min[idx]

    return k_dry
