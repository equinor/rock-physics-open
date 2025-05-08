import numpy as np


def voigt(k1, mu1, k2, mu2, f1):
    """
    Effective media calculation of a mixture of two phases using the Voigt bound.

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
    f1 : np.ndarray
        Fraction of phase 1 [fraction].

    Returns
    -------
    tuple
        k, mu : np.ndarray.
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa]
    """
    f2 = 1 - f1
    k_v = f1 * k1 + f2 * k2
    mu_v = f1 * mu1 + f2 * mu2

    return k_v, mu_v


def voigt_reuss_hill(k1, mu1, k2, mu2, f1):
    """
    Effective media calculation of a mixture of two phases using the Voigt-Reuss-Hill average.

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
    f1 : np.ndarray
        Fraction of phase 1 [fraction].

    Returns
    -------
    tuple
        k, mu : np.ndarray
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa]
    """
    k_v, mu_v = voigt(k1, mu1, k2, mu2, f1)
    k_r, mu_r = reuss(k1, mu1, k2, mu2, f1)

    k_vrh = (k_r + k_v) / 2
    mu_vrh = (mu_r + mu_v) / 2

    return k_vrh, mu_vrh


def multi_voigt_reuss_hill(*varargin):
    """
    Voigt-Reuss-Hill with multiple mineral input.

    Call function with [k, mu] = multi_voigt_reuss_hill(k1, mu1, f2, ....kn,
    mun, fn). Fractions must add up to 1.0. All inputs are assumed to be
    vectors with the same length.

    Parameters
    ----------
    varargin : np.ndarray

    Returns
    -------
    tuple
        k, mu : np.ndarray
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa].
    """
    k_min = np.array(varargin[::3])
    mu_min = np.array(varargin[1::3])
    f = np.array(varargin[2::3])

    k_voigt = np.sum(k_min * f, axis=0)
    mu_voigt = np.sum(mu_min * f, axis=0)

    k_reuss = 1 / np.sum(f / k_min, axis=0)
    mu_reuss = 1 / np.sum(f / mu_min, axis=0)

    k = (k_voigt + k_reuss) / 2
    mu = (mu_voigt + mu_reuss) / 2

    return k, mu


def reuss(k1, mu1, k2, mu2, f1):
    """
    Effective media calculation of a mixture of two phases using the Reuss bound.

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
    f1 : np.ndarray
        Fraction of phase 1 [fraction].

    Returns
    -------
    tuple
        k, mu : np.ndarray.
        k: effective bulk modulus [Pa], mu: effective shear modulus [Pa].
    """
    f2 = 1 - f1

    k_r = 1 / (f1 / k1 + f2 / k2)
    mu_r = 1 / (f1 / mu1 + f2 / mu2)

    return k_r, mu_r
