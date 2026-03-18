import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions


def hs_average(
    k1: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    rhob1: npt.NDArray[np.float64],
    k2: npt.NDArray[np.float64],
    mu2: npt.NDArray[np.float64],
    rhob2: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    BMix of two phases by Hashin-Shtrikman model. Derived properties are also returned.

    Parameters
    ----------
    k1
        Bulk modulus of phase 1 [Pa].
    mu1
        Shear modulus of phase 1 [Pa].
    rhob1
        Density of phase 1 [kg/m^3].
    k2
        Bulk modulus of phase 2 [Pa].
    mu2
        Shear modulus of phase 2 [Pa].
    rhob2
        Density of phase 2 [kg/m^3].
    f
        Fraction of phase 1 [fraction].

    Returns
    -------
    vp
        Compressional wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rhob
        Bulk density [kg/m^3].
    ai
        Acoustic impedance [m/s x kg/m^3].
    vp_vs
        Velocity ratio [ratio].
    k
        Bulk modulus [Pa].
    mu
        Shear modulus [Pa].
    """
    k, mu = std_functions.hashin_shtrikman_average(k1, mu1, k2, mu2, f)

    rhob = rhob1 * f + rhob2 * (1 - f)

    vp, vs, ai, vp_vs = std_functions.velocity(k, mu, rhob)

    return vp, vs, rhob, ai, vp_vs, k, mu
