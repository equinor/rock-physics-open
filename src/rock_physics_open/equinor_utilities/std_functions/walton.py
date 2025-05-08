import numpy as np


def walton_smooth(k, mu, phi, p_eff, coord=None):
    """
    Walton Smooth Pressure-induced moduli increase at critical porosity.

    Parameters
    ----------
    k : np.ndarray
        Bulk modulus of grain mineral [Pa].
    mu : np.ndarray
        Shear modulus of grain mineral [Pa].
    phi : np.ndarray
        Critical porosity [fraction].
    p_eff : np.ndarray
        Effective Pressure = (Lithostatic - Hydrostatic) pressure [Pa].
    coord : float or np.ndarray
        Coordination number, i.e. number of grain contract per grain. If not provided a porosity based estimate is
        used [unitless].

    Returns
    -------
    tuple
        k_dry , mu_dry : np.ndarray.
        k_dry	Bulk modulus at effective pressure p,
        mu_dry	Shear modulus at effective pressure p.
    """

    n = 25.98805 * phi**2 - 43.7622 * phi + 21.6719 if coord is None else coord

    pr_min = (3 * k - 2 * mu) / (2 * (3 * k + mu))
    k_dry_num = n**2 * (1 - phi) ** 2 * mu**2 * p_eff  # Numerator in walton expression
    k_dry_denom = 18 * np.pi**2 * (1 - pr_min) ** 2  # Denominator in Walton expression
    k_dry = (k_dry_num / k_dry_denom) ** (1 / 3)  # Bulk modulus
    mu_dry = 3 / 5 * k_dry

    return k_dry, mu_dry
