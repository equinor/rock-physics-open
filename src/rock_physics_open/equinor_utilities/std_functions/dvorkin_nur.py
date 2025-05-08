import numpy as np


def dvorkin_contact_cement(
    frac_cem, por0_sst, mu0_sst, k0_sst, mu0_cem, k0_cem, vs_red, c
):
    """
    Dvorkin-Nur contact cement model for estimation of elastic moduli.

    Parameters
    ----------
    frac_cem : numpy.ndarray
        Cement fraction of volume [ratio].
    por0_sst : numpy.ndarray
        Critical porosity of sand [ratio].
    mu0_sst : numpy.ndarray
        Mineral shear modulus of sand [Pa].
    k0_sst : numpy.ndarray
        Mineral bulk modulus of sand [Pa].
    mu0_cem : numpy.ndarray
        Mineral shear modulus of cement [Pa].
    k0_cem : numpy.ndarray
        Mineral bulk modulus of cement [Pa].
    vs_red : numpy.ndarray
        Shear modulus reduction factor [ratio].
    c : float
        Coordination number (grain contacts per grain) [unitless].

    Returns
    -------
    tuple
        k_cc, mu_cc : numpy.ndarray.
        k_cc: bulk modulus [Pa], mu_cc: shear modulus [Pa].
    """
    alpha = (2 * frac_cem / (3 * (1 - por0_sst))) ** 0.5
    poiss = (3 * k0_sst - 2 * mu0_sst) / (2 * (3 * k0_sst + mu0_sst))
    poiss_c = (3 * k0_cem - 2 * mu0_cem) / (2 * (3 * k0_cem + mu0_cem))
    a_an = (2 * mu0_cem / (np.pi * mu0_sst)) * (
        (1 - poiss) * (1 - poiss_c) / (1 - 2 * poiss_c)
    )
    a_at = mu0_cem / (np.pi * mu0_sst)

    a_t = (
        -1e-2
        * (2.26 * poiss**2 + 2.07 * poiss + 2.3)
        * a_at ** (0.079 * poiss**2 + 0.1754 * poiss - 1.342)
    )
    b_t = (0.0573 * poiss**2 + 0.0937 * poiss + 0.202) * a_at ** (
        0.0274 * poiss**2 + 0.0529 * poiss - 0.8765
    )
    c_t = (
        1e-4
        * (9.654 * poiss**2 + 4.945 * poiss + 3.1)
        * a_at ** (0.01867 * poiss**2 + 0.4011 * poiss - 1.8186)
    )

    s_t = a_t * alpha**2 + b_t * alpha + c_t

    c_n = 0.00024649 * a_an ** (-1.9864)
    b_n = 0.20405 * a_an ** (-0.89008)
    a_n = -0.024153 * a_an ** (-1.3646)

    s_n = a_n * alpha**2 + b_n * alpha + c_n

    m0_cem = k0_cem + 4 / 3 * mu0_cem
    k_cc = (1 / 6) * c * (1 - por0_sst) * m0_cem * s_n
    mu_cc = (3 / 5) * k_cc + vs_red * (3 / 20) * c * (1 - por0_sst) * mu0_cem * s_t

    return k_cc, mu_cc
