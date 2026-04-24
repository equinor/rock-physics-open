import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions
from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector

from .dem import dem_model


def shale_4_min_dem_overlay(
    k1: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    rho1: npt.NDArray[np.float64],
    k2: npt.NDArray[np.float64],
    mu2: npt.NDArray[np.float64],
    rho2: npt.NDArray[np.float64],
    k3: npt.NDArray[np.float64],
    mu3: npt.NDArray[np.float64],
    rho3: npt.NDArray[np.float64],
    k4: npt.NDArray[np.float64],
    mu4: npt.NDArray[np.float64],
    rho4: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    f1: npt.NDArray[np.float64],
    f2: npt.NDArray[np.float64],
    prop_clay: float,
    asp: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simple shale model with mixture of matrix minerals in Voigt-Reuss-Hill and DEM inclusion model for fluid filled porosity.

    The model is aimed at overlays in cross-plots where two input fractions are given, and the two remaining are given as a proportion.
    Fraction of carbonate and clay is implicit 1 - f1 - f2. f1 and f2 can be set to zero
    All k, mu values in [Pa], rho in [kg/m^3], f, phi, prop_clay, asp in [fraction].

    Parameters
    ----------
    k1
        Mineral 1 properties [Quartz and feldspar].
    mu1
        Mineral 1 properties [Quartz and feldspar].
    rho1
        Mineral 1 properties [Quartz and feldspar].
    k2
        Mineral 2 properties [Kerogen].
    mu2
        Mineral 2 properties [Kerogen].
    rho2
        Mineral 2 properties [Kerogen].
    k3
        Mineral 3 properties [Clay].
    mu3
        Mineral 3 properties [Clay].
    rho3
        Mineral 3 properties [Clay].
    k4
        Mineral 4 properties [Carbonates].
    mu4
        Mineral 4 properties [Carbonates].
    rho4
        Mineral 4 properties [Carbonates].
    k_fl
        Fluid properties.
    rho_fl
        Fluid properties.
    phi
        Porosity.
    f1
        Fraction of mineral 1.
    f2
        Fraction of mineral 2.
    prop_clay
        Range 0 - 1 of the fraction that is clay and carbonate.
    asp
        Porosity aspect ratio.

    Returns
    -------
    k
        Effective bulk modulus [Pa].
    mu
        Effective shear modulus [Pa].
    rhob
        Effective density [kg/m^3].

    Raises
    ------
    ValueError
        If sum of fixed mineral fractions exceeds 1.0 or if any fraction is negative.
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
        k1=k_mat,
        mu1=mu_mat,
        rho1=rho_mat,
        k2=k_fl,
        mu2=np.zeros(len(k_fl)),
        rho2=rho_fl,
        frac2=phi,
        asp2=asp,
        tol=tol,
    )

    return k, mu, rhob
