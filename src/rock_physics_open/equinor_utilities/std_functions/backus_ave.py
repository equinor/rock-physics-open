import numpy as np


def backus_average(vp1, vs1, rho1, vp2, vs2, rho2, f1):
    """
    Backus average for a combination of two phases. The individual phases are isotropic
    but the resulting effective medium is not.

    Parameters
    ----------
    vp1 : np.ndarray
        Pressure wave velocity for phase 1 [m/s].
    vs1 : np.ndarray
        Shear wave velocity for phase 1 [m/s].
    rho1 : np.ndarray
        Density for phase 1 [kg/m^3].
    vp2 : np.ndarray
        Pressure wave velocity for phase 2 [m/s].
    vs2 : np.ndarray
        Shear wave velocity for phase 2 [m/s].
    rho2 : np.ndarray
        Density for phase 2 [kg/m^3].
    f1 : np.ndarray
        Fraction of phase 1.

    Returns
    -------
    tuple
        vpv, vsv, vph, vsh, rho : np.ndarray
        vpv: vertical pressure velocity, vsv: vertical shear velocity, vph: horizontal pressure velocity,
        vsh: horizontal shear velocity, rho: density.
    """

    a = (
        4 * f1 * rho1 * vs1**2 * (1 - vs1**2 / vp1**2)
        + 4 * (1 - f1) * rho2 * vs2**2 * (1 - (vs2 / vp2) ** 2)
        + (f1 * (1 - 2 * (vs1 / vp1) ** 2) + (1 - f1) * (1 - 2 * (vs2 / vp2) ** 2)) ** 2
        * (1 / (f1 / (rho1 * vp1**2) + (1 - f1) / (rho2 * vp2**2)))
    )

    c = 1 / (f1 / (rho1 * vp1**2) + (1 - f1) / (rho2 * vp2**2))
    d = 1 / (f1 / (rho1 * vs1**2) + (1 - f1) / (rho2 * vs2**2))

    m = f1 * rho1 * vs1**2 + (1 - f1) * rho2 * vs2**2

    rho = f1 * rho1 + (1 - f1) * rho2

    vpv = np.sqrt(c / rho)
    vsv = np.sqrt(d / rho)
    vph = np.sqrt(a / rho)
    vsh = np.sqrt(m / rho)

    return vpv, vsv, vph, vsh, rho
