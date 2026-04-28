from typing import cast, overload

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.various_utilities.types import ArrayAnyDOrFloat


@overload
def moduli(
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64] | float,
    rhob: npt.NDArray[np.float64] | float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def moduli(
    vp: npt.NDArray[np.float64] | float,
    vs: npt.NDArray[np.float64],
    rhob: npt.NDArray[np.float64] | float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def moduli(
    vp: npt.NDArray[np.float64] | float,
    vs: npt.NDArray[np.float64] | float,
    rhob: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def moduli(vp: float, vs: float, rhob: float) -> tuple[float, float]: ...
def moduli(
    vp: npt.NDArray[np.float64] | float,
    vs: npt.NDArray[np.float64] | float,
    rhob: npt.NDArray[np.float64] | float,
) -> tuple[npt.NDArray[np.float64] | float, npt.NDArray[np.float64] | float]:
    """
    Calculate isotropic moduli from velocity and density.

    Parameters
    ----------
    vp
        Pressure wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rhob
        Bulk density [kg/m^3].

    Returns
    -------
    k
        Bulk modulus [Pa].
    mu
        Shear modulus [Pa].
    """
    mu = vs**2 * rhob
    k = vp**2 * rhob - 4 / 3 * mu

    return k, mu


def velocity(
    k: ArrayAnyDOrFloat,
    mu: ArrayAnyDOrFloat,
    rhob: ArrayAnyDOrFloat,
) -> tuple[ArrayAnyDOrFloat, ArrayAnyDOrFloat, ArrayAnyDOrFloat, ArrayAnyDOrFloat]:
    """
    Calculate velocity, acoustic impedance and vp/vs ratio from elastic moduli and density.

    Parameters
    ----------
    k
        Bulk modulus [Pa].
    mu
        Shear modulus [Pa].
    rhob
        Bulk density [kg/m^3].

    Returns
    -------
    vp
        Pressure wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    ai
        Acoustic impedance [m/s x kg/m^3].
    vp_vs
        Velocity ratio [fraction].
    """
    vs = cast(ArrayAnyDOrFloat, (mu / rhob) ** 0.5)
    vp = cast(ArrayAnyDOrFloat, ((k + 4 / 3 * mu) / rhob) ** 0.5)
    ai = cast(ArrayAnyDOrFloat, vp * rhob)
    with np.errstate(divide="ignore", invalid="ignore"):
        # vs=0 is valid for Newtonian fluids (mu=0); vp/vs → inf is the correct result
        vp_vs = cast(ArrayAnyDOrFloat, vp / vs)

    return vp, vs, ai, vp_vs
