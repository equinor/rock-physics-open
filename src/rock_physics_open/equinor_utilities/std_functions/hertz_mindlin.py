import numpy as np


def hertz_mindlin(k, mu, phi_c, p, shear_red, coord=None):
    """
    Hertz-Mindlin Pressure-induced moduli increase at critical porosity.

    Parameters
    ----------
    k : np.ndarray
        Bulk modulus of grain mineral [Pa].
    mu : np.ndarray
        Shear modulus of grain mineral [Pa].
    phi_c : np.ndarray
        Critical porosity [fraction].
    p : np.ndarray
        Effective pressure = lithostatic pressure - hydrostatic pressure [Pa].
    shear_red : float or np.ndarray
        Reduced shear factor, if set to 1.0, calculation reduces to standard Hertz-Mindlin equation.
    coord
        coordination number, i.e. the number of grain contacts per grain. If not provided a porosity based
        estimate is used.

    Returns
    -------
    tuple
        k_dry, mu_dry : np.ndarray.
        k_dry:	Bulk modulus [Pa] at effective pressure p,
        mu_dry:	Shear modulus [Pa] at effective pressure p.
    """
    n = 25.98805 * phi_c**2 - 43.7622 * phi_c + 21.6719 if coord is None else coord

    poiss = (3 * k - 2 * mu) / (2 * (3 * k + mu))

    a = ((3 * np.pi * (1 - poiss) * p) / (2 * n * (1 - phi_c) * mu)) ** (1 / 3)

    s_n = (4 * a * mu) / (1 - poiss)
    s_t = (8 * a * mu) / (2 - poiss)

    k_dry = (n * (1 - phi_c) / (12 * np.pi)) * s_n
    mu_dry = (n * (1 - phi_c) / (20 * np.pi)) * (s_n + 1.5 * s_t * shear_red)

    return k_dry, mu_dry
