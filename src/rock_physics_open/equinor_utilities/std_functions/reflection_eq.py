import numpy as np


def _refl_models(vp, vs, rho, theta, k=2.0, mod="aki_richards"):
    """Calculate reflect coeff.

    Parameters
    ----------
    vp : np.ndarray
        vp array.
    vs : np.ndarray
        vs array.
    rho : np.ndarray
        rho array.
    theta : float, np.ndarray
        Theta value.
    k : float, optional
        By default 2.0.
    mod : str, optional
        By default 'aki_richards'.

    Returns
    -------
    np.ndarray
        Reflect coeff.
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


def aki_richards(vp, vs, rho, theta, k=2.0):
    """
    Linearised Zoeppritz equation according to Aki and Richards.

    Parameters
    ----------
    vp : np.ndarray
        Pressure wave velocity [m/s].
    vs : np.ndarray
        Shear wave velocity [m/s].
    rho : np.ndarray
        Density [kg/m^3].
    theta : np.ndarray
        Angle of incident ray [radians].
    k : float
        Background vp/vs [unitless].

    Returns
    -------
    refl_coeff : np.ndarray
        Reflection coefficient [unitless].
    """
    return _refl_models(vp, vs, rho, theta, k, mod="aki_richards")


def smith_gidlow(vp, vs, rho, theta, k=2.0):
    """
    Linearised Zoeppritz equation according to Smith and Gidlow.

    Parameters
    ----------
    vp : np.ndarray
        Pressure wave velocity [m/s].
    vs : np.ndarray
        Shear wave velocity [m/s].
    rho : np.ndarray
        Density [kg/m^3].
    theta : np.ndarray
        Angle of incident ray [radians].
    k : float
        Background vp/vs [unitless].

    Returns
    -------
    refl_coeff : np.ndarray
        Reflection coefficient [unitless].
    """

    return _refl_models(vp, vs, rho, theta, k, mod="smith_gidlow")
