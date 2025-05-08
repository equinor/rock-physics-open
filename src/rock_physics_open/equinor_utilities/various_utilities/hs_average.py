from rock_physics_open.equinor_utilities import std_functions


def hs_average(k1, mu1, rhob1, k2, mu2, rhob2, f):
    """
    BMix of two phases by Hashin-Shtrikman model. Derived properties are also returned.

    Parameters
    ----------
    k1 : np.ndarray
        k1 array.
    mu1 : np.ndarray
        mu1 array.
    rhob1 : np.ndarray
        rhob1 array.
    k2 : np.ndarray
        k2 array.
    mu2 : np.ndarray
        mu2 array.
    rhob2 : np.ndarray
        rhob2 array.
    f : float or np.ndarray
        f value or array.

    Returns
    -------
    tuple
        vp, vs, rhob, ai, vp_vs, k, mu : np.ndarray
        vp: compressional wave velocity [m/s], vs: shear wave velocity [m/s], ai: acoustic impedance [m/s x kg/m^3],
        vp_vs: velocity ratio [ratio], k: bulk modulus [Pa], mu: shear modulus [Pa]

    """

    k, mu = std_functions.hashin_shtrikman_average(k1, mu1, k2, mu2, f)

    rhob = rhob1 * f + rhob2 * (1 - f)

    vp, vs, ai, vp_vs = std_functions.velocity(k, mu, rhob)

    return vp, vs, rhob, ai, vp_vs, k, mu
