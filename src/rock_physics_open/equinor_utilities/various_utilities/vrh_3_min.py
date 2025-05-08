from rock_physics_open.equinor_utilities import std_functions


def min_3_voigt_reuss_hill(
    vp1, vs1, rhob1, f1, vp2, vs2, rhob2, f2, vp3, vs3, rhob3, f3
):
    """
    Mix of three phases by Voigt-Reuss-Hill model. The fractions should add up to 1 with input of vp, vs and rho.

    Parameters
    ----------
    vp1 : np.ndarray
        Pressure wave velocity of phase 1 [m/s].
    vs1 : np.ndarray
        Shear wave velocity of phase 1 [m/s].
    rhob1 : np.ndarray
        Bulk density of phase 1 [kg/m^3].
    f1 : np.ndarray
        Fraction of phase 1 [fraction].
    vp2 : np.ndarray
        Pressure wave velocity of phase 2 [m/s].
    vs2 : np.ndarray
        Shear wave velocity of phase 2 [m/s].
    rhob2 : np.ndarray
        Bulk density of phase 2 [kg/m^3].
    f2 : np.ndarray
        Fraction of phase 2 [fraction].
    vp3 : np.ndarray
        Pressure wave velocity of phase 3 [m/s].
    vs3 : np.ndarray
        Shear wave velocity of phase 3 [m/s].
    rhob3 : np.ndarray
        Bulk density of phase 3 [kg/m^3].
    f3 : np.ndarray
        Fraction of phase 3 [fraction].

    Returns
    -------
    tuple
        vp, vs, rhob, ai, vp_vs, k, mu : np.ndarray.
        vp: pressure wave velocity [m/s], vs: shear wave velocity [m/s], rhob: bulk density [kg/m^3],
        ai: acoustic impedance [m/s x kg/m^3], vp_vs: velocity ratio [ratio],
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa].
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
