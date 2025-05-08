import numpy as np

from .pq import p_q_fcn


def self_consistent_approximation_model(
    k1, mu1, rho1, k2, mu2, rho2, frac1, asp1, asp2, tol
):
    """
    SCA - Effective elastic moduli using Berryman's Self-Consistent
    (Coherent Potential) Approximation method.

    Parameters
    ----------
    k1 : np.ndarray
        Bulk modulus of background matrix [Pa].
    mu1 : np.ndarray
        Shear modulus of background matrix [Pa].
    rho1 : np.ndarray
        Bulk density of background matrix [kg/m^3].
    k2 : np.ndarray
        Bulk modulus of inclusions [Pa].
    mu2 : np.ndarray
        Shear modulus of inclusions [Pa].
    rho2 : np.ndarray
        Bulk density of inclusions [kg/m^3].
    frac1 : np.ndarray
        Fraction of inclusions [fraction].
    asp1 : np.ndarray
        Aspect ratio of inclusions [ratio].
    asp2 : np.ndarray
        Aspect ratio of inclusions [ratio].
    tol: float
        Desired accuracy in the SCA iterations.

    Returns
    -------
    tuple
        k, mu, rho : (np.ndarray, np.ndarray, np.ndarray).
        k: effective medium bulk modulus [Pa], mu: effective medium shear modulus [Pa], rho: bulk density [kg/m^3].

    Comments
    --------
    Based on function by T. Mukerji, SRB, Stanford University, 1994.
    Ported to Python by Harald Flesche, Equinor 2015.
    """

    # Calculate fn and theta - independent of iteration
    f1 = frac1
    f2 = 1 - frac1

    k_sc = f1 * k1 + f2 * k2
    mu_sc = f1 * mu1 + f2 * mu2
    rhob = f1 * rho1 + f2 * rho2

    idx = np.logical_and(np.not_equal(f1, 0.0), np.not_equal(f1, 1.0))
    n_iter = 0
    k_new = np.zeros(np.sum(idx))
    delta = np.absolute(k_sc[idx] - k_new)
    # Express tolerance in terms of k1
    tol = tol * k1[idx]

    while np.any(delta > tol) and (n_iter < 3000):
        p1, q1 = p_q_fcn(k_sc[idx], mu_sc[idx], k1[idx], mu1[idx], asp1[idx])
        p2, q2 = p_q_fcn(k_sc[idx], mu_sc[idx], k2[idx], mu2[idx], asp2[idx])

        k_new = (f1[idx] * k1[idx] * p1 + f2[idx] * k2[idx] * p2) / (
            f1[idx] * p1 + f2[idx] * p2
        )
        mu_new = (f1[idx] * mu1[idx] * q1 + f2[idx] * mu2[idx] * q2) / (
            f1[idx] * q1 + f2[idx] * q2
        )

        delta = np.absolute(k_sc[idx] - k_new)
        k_sc[idx] = k_new
        mu_sc[idx] = mu_new

        n_iter = n_iter + 1

    # If all inclusions or all matrix - substitute with inclusion mineral properties
    idx = frac1 == 1.0
    if np.any(idx):
        k_sc[idx] = k1[idx]
        mu_sc[idx] = mu1[idx]
    idx = frac1 == 0.0
    if np.any(idx):
        k_sc[idx] = k2[idx]
        mu_sc[idx] = mu2[idx]

    return k_sc, mu_sc, rhob
