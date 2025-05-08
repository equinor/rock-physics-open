import numpy as np


def brie(s_gas, k_gas, s_brine, k_brine, s_oil, k_oil, e):
    """
    Brie function for effective bulk modulus of a mix of fluids.

    Parameters
    ----------
    s_gas : np.ndarray
        Gas saturation [ratio].
    k_gas : np.ndarray
        Gas bulk modulus [Pa].
    s_brine : np.ndarray
        Brine saturation [ratio].
    k_brine : np.ndarray
        Brine bulk modulus [Pa].
    s_oil : np.ndarray
        Oil saturation [ratio].
    k_oil : np.ndarray
        Oil bulk modulus [Pa].
    e : float or np.ndarray
        Exponent in Brie function [unitless].

    Returns
    -------
    np.ndarray
        k: effective bulk modulus [Pa].
    """

    # Reuss average for fluids, catch zero fluid saturations
    idx = s_brine + s_oil > 0
    k_liquid = np.zeros(k_brine.shape)
    k_liquid[idx] = (s_brine[idx] / k_brine[idx] + s_oil[idx] / k_oil[idx]) ** -1

    return (k_liquid - k_gas) * (1 - s_gas) ** e + k_gas


def wood(s1, k1, rho1, k2, rho2):
    """
    Wood effective fluid properties for a mix of two fluids.

    Parameters
    ----------
    s1 : np.ndarray
        Saturation of phase 1 [ratio].
    k1 : np.ndarray
        Bulk modulus of phase 1 [Pa].
    rho1 : np.ndarray
        Density of phase 1 [kg/m^3].
    k2 : np.ndarray
        Bulk modulus of phase 2 [Pa].
    rho2 : np.ndarray
        Density of phase 2 [kg/m^3].

    Returns
    -------
    tuple
        k, rho : np.ndarray.
        k: effective fluid bulk modulus [Pa], rho: effective fluid density [kg/m^3].
    """
    s2 = 1 - s1

    k = 1 / (s1 / k1 + s2 / k2)
    rho = s1 * rho1 + s2 * rho2

    return k, rho


def multi_wood(fractions, bulk_moduli):
    assert len(fractions) == len(bulk_moduli)
    sum_fractions = sum(fractions)
    ratio_sum = sum(
        saturation / bulk_modulus
        for (saturation, bulk_modulus) in zip(fractions, bulk_moduli)
    )
    return sum_fractions / ratio_sum
