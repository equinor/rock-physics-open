import warnings

import numpy as np

from .pq import p_q_fcn


def kuster_toksoz_model(k1, mu1, rho1, k2, mu2, rho2, frac1, asp2):
    """Simplified Kuster-Toksoz model for single mineral inclusion with single aspect ratio.

    Parameters
    ----------
    k1 : np.ndarray
        Phase 1 bulk modulus [Pa].
    mu1 : np.ndarray
        Phase 1 shear modulus [Pa].
    rho1 : np.ndarray
        Phase 1 bulk density [kg/m^3].
    k2 : np.ndarray
        Phase 2 bulk modulus [Pa].
    mu2 : np.ndarray
        Phase 2 shear modulus [Pa].
    rho2 : np.ndarray
        Phase 2 bulk density [kg/m^3].
    frac1 : np.ndarray
        Fraction of phase 1 [fraction].
    asp2 : np.ndarray
        Aspect ratio for phase 2 inclusions [ratio].

    Returns
    -------
    tuple
        k_kt, mu_kt, rhob : (np.ndarray, np.ndarray, np.ndarray).
        effective media properties: k_kt: bulk modulus [Pa], mu_kt: shear modulus [Pa], rhob: bulk density [kg/m^3].
    """
    frac2 = 1.0 - frac1
    rhob = rho1 * frac1 + rho2 * frac2
    p, q = p_q_fcn(k1, mu1, k2, mu2, asp2)
    zeta = mu1 / 6.0 * (9.0 * k1 + 8.0 * mu1) / (k1 + 2.0 * mu1)

    k_kt = (k1 * (k1 + 4.0 / 3.0 * mu1) + 4.0 / 3.0 * mu1 * frac2 * (k2 - k1) * p) / (
        k1 + 4.0 / 3.0 * mu1 - frac2 * (k2 - k1) * p
    )
    mu_kt = (mu1 * (mu1 + zeta) + zeta * frac2 * (mu2 - mu1) * q) / (
        mu1 + zeta - frac2 * (mu2 - mu1) * q
    )

    # Non-physical situations can arise if there is too high volume fraction of inclusions with
    # low aspect ratio
    idx_neg = np.logical_or(k_kt < 0.0, mu_kt < 0.0)
    if np.any(idx_neg):
        k_kt[idx_neg] = np.nan
        mu_kt[idx_neg] = np.nan
        rhob[idx_neg] = np.nan
        warnings.warn(
            f"{__file__}: {np.sum(idx_neg)} non-physical solutions to Kuster-Toksöz equation"
        )

    return k_kt, mu_kt, rhob
