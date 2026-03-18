from typing import Literal

import numpy as np
import numpy.typing as npt


def _refl_models(
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64] | float = 2.0,
    mod: Literal["aki_richards", "smith_gidlow"] = "aki_richards",
) -> npt.NDArray[np.float64]:
    """Compute two-term reflection coefficients between layer pairs.

    Parameters
    ----------
    vp
        Compressional wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rho
        Bulk density [kg/m^3].
    theta
        Incidence angle [degree].
    k
        Background Vp/Vs ratio, by default 2.0.
    mod
        Reflection model, by default ``'aki_richards'``.

    Returns
    -------
    Reflection coefficient per interface [ratio].
    """
    theta = theta / 180 * np.pi
    r_vp = (vp[1:] - vp[0:-1]) / (vp[0:-1] + vp[1:])
    r_vs = (vs[1:] - vs[0:-1]) / (vs[0:-1] + vs[1:])
    r_rho = (rho[1:] - rho[0:-1]) / (rho[0:-1] + rho[1:])
    # Need to append one sample to make returned array of the same size - repeat the last value
    r_vp = np.append(r_vp, r_vp[-1])
    r_vs = np.append(r_vs, r_vs[-1])
    r_rho = np.append(r_rho, r_rho[-1])

    if mod == "aki_richards":
        reflect_coeff = (
            0.5 * (r_vp + r_rho)
            + (0.5 * r_vp - 2 * k**-2 * (2 * r_vs + r_rho)) * np.sin(theta) ** 2
            + 0.5 * r_vp * (np.tan(theta) ** 2 - np.sin(theta) ** 2)
        )
    else:
        reflect_coeff = (
            5 / 8 - 0.5 * k**-2 * np.sin(theta) ** 2 + 0.5 * np.tan(theta) ** 2
        ) * r_vp - 4 * k**-2 * np.sin(theta) ** 2 * r_vs

    return reflect_coeff


def aki_richards(
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64] | float = 2.0,
) -> npt.NDArray[np.float64]:
    """
    Linearised Zoeppritz equation according to Aki and Richards.

    Parameters
    ----------
    vp
        Pressure wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rho
        Density [kg/m^3].
    theta
        Angle of incident ray [radians].
    k
        Background vp/vs [unitless].

    Returns
    -------
    Reflection coefficient [unitless].
    """
    return _refl_models(vp, vs, rho, theta, k, mod="aki_richards")


def smith_gidlow(
    vp: npt.NDArray[np.float64],
    vs: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64] | float = 2.0,
) -> npt.NDArray[np.float64]:
    """
    Linearised Zoeppritz equation according to Smith and Gidlow.

    Parameters
    ----------
    vp
        Pressure wave velocity [m/s].
    vs
        Shear wave velocity [m/s].
    rho
        Density [kg/m^3].
    theta
        Angle of incident ray [radians].
    k
        Background vp/vs [unitless].

    Returns
    -------
    Reflection coefficient [unitless].
    """
    return _refl_models(vp, vs, rho, theta, k, mod="smith_gidlow")
