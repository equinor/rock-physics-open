def moduli(vp, vs, rhob):
    """
    Calculate isotropic moduli from velocity and density.

    Parameters
    ----------
    vp : np.ndarray or float
        Pressure wave velocity [m/s].
    vs : np.ndarray or float
        Shear wave velocity [m/s].
    rhob : np.ndarray or float
        Bulk density [kg/m^3].

    Returns
    -------
    tuple
        k, mu : (np.ndarray, np.ndarray) or (float, float).
        k: bulk modulus [Pa], mu: shear modulus [Pa].
    """
    mu = vs**2 * rhob
    k = vp**2 * rhob - 4 / 3 * mu

    return k, mu


def velocity(k, mu, rhob):
    """
    Calculate velocity, acoustic impedance and vp/vs ratio from elastic moduli and density.

    Parameters
    ----------
    k : np.ndarray, float
        Bulk modulus [Pa].
    mu : np.ndarray, float
        Shear modulus [Pa].
    rhob : np.ndarray, float
        Bulk density [kg/m^3].

    Returns
    -------
    tuple
        vp, vs, ai, vp_vs : np.ndarray.
        vp: pressure wave velocity [m/s], vs: shear wave velocity [m/s], ai: acoustic impedance [m/s x kg/m^3],
        vp_vs: velocity ratio [fraction].
    """
    vs = (mu / rhob) ** 0.5
    vp = ((k + 4 / 3 * mu) / rhob) ** 0.5
    ai = vp * rhob
    vp_vs = vp / vs

    return vp, vs, ai, vp_vs
