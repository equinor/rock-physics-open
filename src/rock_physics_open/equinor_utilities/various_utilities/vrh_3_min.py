import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions


def min_3_voigt_reuss_hill(
    vp1: npt.NDArray[np.float64],
    vs1: npt.NDArray[np.float64],
    rhob1: npt.NDArray[np.float64],
    f1: npt.NDArray[np.float64],
    vp2: npt.NDArray[np.float64],
    vs2: npt.NDArray[np.float64],
    rhob2: npt.NDArray[np.float64],
    f2: npt.NDArray[np.float64],
    vp3: npt.NDArray[np.float64],
    vs3: npt.NDArray[np.float64],
    rhob3: npt.NDArray[np.float64],
    f3: npt.NDArray[np.float64],
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
    Mix of three phases by Voigt-Reuss-Hill model. The fractions should add up to 1 with input of vp, vs and rho.

    Parameters
    ----------
    vp1
        Pressure wave velocity of phase 1 [m/s].
    vs1
        Shear wave velocity of phase 1 [m/s].
    rhob1
        Bulk density of phase 1 [kg/m^3].
    f1
        Fraction of phase 1 [fraction].
    vp2
        Pressure wave velocity of phase 2 [m/s].
    vs2
        Shear wave velocity of phase 2 [m/s].
    rhob2
        Bulk density of phase 2 [kg/m^3].
    f2
        Fraction of phase 2 [fraction].
    vp3
        Pressure wave velocity of phase 3 [m/s].
    vs3
        Shear wave velocity of phase 3 [m/s].
    rhob3
        Bulk density of phase 3 [kg/m^3].
    f3
        Fraction of phase 3 [fraction].

    Returns
    -------
    vp
        Pressure wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rhob
        Bulk density [kg/m^3].
    ai
        Acoustic impedance [m/s x kg/m^3].
    vp_vs
        Velocity ratio [ratio].
    k
        Effective bulk modulus [Pa].
    mu
        Effective shear modulus [Pa].
    """
    k1, mu1 = std_functions.moduli(vp1, vs1, rhob1)
    k2, mu2 = std_functions.moduli(vp2, vs2, rhob2)
    k3, mu3 = std_functions.moduli(vp3, vs3, rhob3)

    # Normalise the fractions to make sure they sum to one
    tot = f1 + f2 + f3
    f1 = f1 / tot
    f2 = f2 / tot
    f3 = f3 / tot

    k, mu = std_functions.multi_voigt_reuss_hill(k1, mu1, f1, k2, mu2, f2, k3, mu3, f3)

    rhob = rhob1 * f1 + rhob2 * f2 + rhob3 * f3

    vp, vs, ai, vp_vs = std_functions.velocity(k, mu, rhob)

    return vp, vs, rhob, ai, vp_vs, k, mu
