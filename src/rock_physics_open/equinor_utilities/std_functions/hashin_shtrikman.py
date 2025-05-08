import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector


def hashin_shtrikman(k1, mu1, k2, mu2, f):
    """
    Hashin-Sktrikman upper or lower according to ordering of phases.

    Parameters
    ----------
    k1 : np.ndarray
        Bulk modulus of phase 1 [Pa].
    mu1 : np.ndarray
        Shear modulus of phase 1 [Pa].
    k2 : np.ndarray
        Bulk modulus of phase 2 [Pa].
    mu2 : np.ndarray
        Shear modulus of phase 2 [Pa].
    f : np.ndarray
        Fraction of phase 1 [fraction].

    Returns
    -------
    tuple
        k, mu : np.ndarray.
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa].
    """
    k = k1 + (1 - f) * (k2 - k1) / (1 + (k2 - k1) * f * (k1 + 4 / 3 * mu1) ** -1)
    mu = mu1 + (1 - f) * (mu2 - mu1) / (
        1 + 2 * (mu2 - mu1) * f * (k1 + 2 * mu1) / (5 * mu1 * (k1 + 4 / 3 * mu1))
    )

    return k, mu


def hashin_shtrikman_average(k1, mu1, k2, mu2, f):
    """
    Average of Hashin-Shtrikman upper and lower bound.

    Parameters
    ----------
    k1 : np.ndarray
        Bulk modulus of phase 1 [Pa].
    mu1 : np.ndarray
        Shear modulus of phase 1 [Pa].
    k2 : np.ndarray
        Bulk modulus of phase 2 [Pa].
    mu2 : np.ndarray
        Shear modulus of phase 2 [Pa].
    f : np.ndarray
        Fraction of phase 1 [fraction].

    Returns
    -------
    tuple
        k_av, mu_av : np.ndarray.
        k_av: effective bulk modulus [Pa], mu_av: effective shear modulus [Pa]
    """
    k_hs1, mu_hs1 = hashin_shtrikman(k1, mu1, k2, mu2, f)
    k_hs2, mu_hs2 = hashin_shtrikman(k2, mu2, k1, mu1, 1 - f)

    k_av = (k_hs1 + k_hs2) / 2
    mu_av = (mu_hs1 + mu_hs2) / 2

    return k_av, mu_av


def hashin_shtrikman_walpole(k1, mu1, k2, mu2, f1, bound="lower"):
    """
    Hashin-Shtrikman upper bound is obtained when the stiffest material is
    termed 1 and vice versa for lower bound. Tricky in cases like Quartz -
    Calcite where the K and Mu have opposed values. HS - Walpole is
    generalised to regard highest and lowest values in each case. The default
    is to generate lower bound.

    Parameters
    ----------
    k1 : np.ndarray
        Bulk modulus of phase 1 [Pa].
    mu1 : np.ndarray
        Shear modulus of phase 1 [Pa].
    k2 : np.ndarray
        Bulk modulus of phase 2 [Pa].
    mu2 : np.ndarray
        Shear modulus of phase 2 [Pa].
    f1 : np.ndarray or float
        Fraction of phase 1 [fraction].
    bound: str
        'upper' or 'lower' selection of upper of lower bound of effective medium.
    Returns
    -------
    tuple
        k, mu : np.ndarray.
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa].
    """
    k1, mu1, k2, mu2, f1 = dim_check_vector((k1, mu1, k2, mu2, f1))
    if bound.lower() not in ["lower", "upper"]:
        raise ValueError(f'{__file__}: bound must be one of "lower" or "upper"')

    idx_k = k1 == k2
    idx_mu = mu1 == mu2
    f2 = 1 - f1

    if bound.lower() == "lower":
        k_m = np.minimum(k1, k2)
        mu_m = np.minimum(mu1, mu2)
    else:
        k_m = np.maximum(k1, k2)
        mu_m = np.maximum(mu1, mu2)

    k = np.zeros(k1.shape)
    mu = np.zeros(k1.shape)

    if np.any(idx_k):
        k[idx_k] = k1[idx_k]
    if np.any(~idx_k):
        k[~idx_k] = k1[~idx_k] + f2[~idx_k] / (
            (k2[~idx_k] - k1[~idx_k]) ** -1
            + f1[~idx_k] * (k1[~idx_k] + 4 / 3 * mu_m[~idx_k]) ** -1
        )
    if np.any(idx_mu):
        mu[idx_mu] = mu1[idx_mu]
    if np.any(~idx_mu):
        mu[~idx_mu] = mu1[~idx_mu] + f2[~idx_mu] / (
            (mu2[~idx_mu] - mu1[~idx_mu]) ** -1
            + f1[~idx_mu]
            * (
                mu1[~idx_mu]
                + mu_m[~idx_mu]
                / 6
                * (9 * k_m[~idx_mu] + 8 * mu_m[~idx_mu])
                / (k_m[~idx_mu] + 2 * mu_m[~idx_mu])
            )
            ** -1
        )

    return k, mu


def multi_hashin_shtrikman(*coeffs, mode="average"):
    """
    Hashin-Shtrikman effective medium calculation for multi-mineral case.

    Parameters
    ----------
    coeffs : np.ndarray
        Triplets of vectors with k (bulk modulus [Pa]), mu (shear modulus [Pa]) and fraction for each mineral.
    mode : str
        'average', 'upper' or 'lower'.

    Returns
    -------
    tuple
        k_hs, mu_hs : np.ndarray.
        k_hs, mu_hs - bulk modulus and shear modulus for effective medium [Pa].
    """
    if not len(coeffs) % 3 == 0:
        raise ValueError(
            "multi_hashin_shtrikman: inputs not vectors of k, mu and fraction for each mineral"
        )

    k_arr = np.array(coeffs[::3])
    mu_arr = np.array(coeffs[1::3])
    f = np.array(coeffs[2::3])

    if not np.all(np.around(np.sum(f, axis=0), decimals=6) == 1.0):
        raise ValueError("multi_hashin_shtrikman: all fractions do not add up to 1.0")

    if mode.lower() not in ["average", "upper", "lower"]:
        raise ValueError(
            'multi_hashin_shtrikman: mode is not one of "average", "upper" or "lower"'
        )

    k_min = np.min(k_arr, axis=0)
    k_max = np.max(k_arr, axis=0)
    mu_min = np.min(mu_arr, axis=0)
    mu_max = np.max(mu_arr, axis=0)

    k_hs_upper = np.sum(f / (k_arr + 4 / 3 * mu_max), axis=0) ** -1 - 4 / 3 * mu_max
    k_hs_lower = np.sum(f / (k_arr + 4 / 3 * mu_min), axis=0) ** -1 - 4 / 3 * mu_min

    zeta_max = mu_max / 6 * (9 * k_max + 8 * mu_max) / (k_max + 2 * mu_max)
    zeta_min = mu_min / 6 * (9 * k_min + 8 * mu_min) / (k_min + 2 * mu_min)

    mu_hs_upper = np.sum(f / (mu_arr + zeta_max), axis=0) ** -1 - zeta_max
    mu_hs_lower = np.sum(f / (mu_arr + zeta_min), axis=0) ** -1 - zeta_min

    if mode == "lower":
        return k_hs_lower, mu_hs_lower
    if mode == "upper":
        return k_hs_upper, mu_hs_upper
    k_hs = 0.5 * (k_hs_upper + k_hs_lower)
    mu_hs = 0.5 * (mu_hs_upper + mu_hs_lower)
    return k_hs, mu_hs
