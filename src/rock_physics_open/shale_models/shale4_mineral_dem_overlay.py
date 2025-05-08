import numpy as np

from rock_physics_open.equinor_utilities import std_functions
from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector

from .dem import dem_model


def shale_4_min_dem_overlay(
    k1,
    mu1,
    rho1,
    k2,
    mu2,
    rho2,
    k3,
    mu3,
    rho3,
    k4,
    mu4,
    rho4,
    k_fl,
    rho_fl,
    phi,
    f1,
    f2,
    prop_clay,
    asp,
):
    """
    Simple shale model with mixture of matrix minerals in Voigt-Reuss-Hill
    and DEM inclusion model for fluid filled porosity. The model is aimed at
    overlays in cross-plots where two input fractions are given, and the two
    remaining are given as a proportion.
    Fraction of carbonate and clay is implicit 1 - f1 - f2. f1 and f2 can be set to zero
    All k, mu values in [Pa], rho in [kg/m^3], f, phi, prop_clay, asp in [fraction].

    Parameters
    ----------
    k1, mu1, rho1 : np.ndarray
        Mineral 1 properties [Quartz and feldspar].
    f1 : np.ndarray
        Fraction of mineral 1.
    k2, mu2, rho2 : np.ndarray
        Mineral 2 properties [Kerogen].
    f2 : np.ndarray
        Fraction of mineral 2.
    k3, mu3, rho3 : np.ndarray
        Mineral 3 properties [Clay].
    k4, mu4, rho4 : np.ndarray
        Mineral 4 properties [Carbonates].
    prop_clay : float
        Range 0 - 1 of the fraction that is clay and carbonate.
    k_fl, rho_fl : np.ndarray
        Fluid properties.
    phi : np.ndarray
        Porosity.
    asp:
        Porosity aspect ratio.

    Returns
    -------
    tuple
        k, mu, rhob : (np.ndarray, np.ndarray, np.ndarray).
        k - effektive bulk modulus [Pa], mu - effective shear modulus [Pa], rhob - effective density [kg/m^3].
    """

    #   tol: DEM model calculation tolerance <= Set as a hardcoded value, not
    #   found to influence the results
    tol = 1e-6

    # Calculate effective mineral properties
    if np.any((f1 + f2) > 1.0):
        raise ValueError(f"{__file__}: fixed mineral fractions exceed 1.0")
    if np.any(np.logical_or(f1 < 0, f2 < 0)):
        raise ValueError(f"{__file__}: negative mineral fractions")

    f3 = prop_clay * (1.0 - f1 - f2)
    f4 = (1.0 - prop_clay) * (1.0 - f1 - f2)
    k_mat, mu_mat = std_functions.multi_voigt_reuss_hill(
        k1, mu1, f1, k2, mu2, f2, k3, mu3, f3, k4, mu4, f4
    )
    rho_mat = rho1 * f1 + rho2 * f2 + rho3 * f3 + rho4 * f4

    k_mat, mu_mat, rho_mat, k_fl, rho_fl, phi, asp = dim_check_vector(
        (k_mat, mu_mat, rho_mat, k_fl, rho_fl, phi, asp)
    )
    k, mu, rhob = dem_model(
        k_mat, mu_mat, rho_mat, k_fl, np.zeros(len(k_fl)), rho_fl, phi, asp, tol
    )

    return k, mu, rhob
