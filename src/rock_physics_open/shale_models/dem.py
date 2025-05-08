import sys

import numpy as np
from scipy.integrate import odeint

from .pq import p_q_fcn


def dem_model(k1, mu1, rho1, k2, mu2, rho2, frac2, asp2, tol):
    """
    DEM - Effective elastic moduli using Differential Effective Medium formulation.

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
    frac2 : np.ndarray
        Fraction of inclusions [fraction].
    asp2 : np.ndarray
        Aspect ratio of inclusions [ratio].
    tol: float
        Desired accuracy in the ODE solver.

    Returns
    -------
    tuple
        k, mu, rho : (np.ndarray, np.ndarray, np.ndarray).
        k: effective medium bulk modulus [Pa], mu: effective medium shear modulus [Pa], rho: bulk density [kg/m^3].

    Comments
    --------
    Written by T. Mukerji, SRB, Stanford University.
    Ported to Python by Harald Flesche, Equinor 2015.
    """

    # Make sure all log inputs are vectors, expand scalars, throw error if uneven
    # length and not scalars
    k = np.ones(k1.shape) * np.nan
    mu = np.ones(k1.shape) * np.nan
    rhob = rho1 * (1 - frac2) + rho2 * frac2

    # Check for the trivial cases: all inclusions or all matrix
    idx = np.logical_or(frac2 == 1.0, frac2 == 0.0)
    if np.any(idx):
        idx1 = frac2 == 1.0
        idx2 = frac2 == 0.0
        if np.any(idx1):
            k[idx1] = k2[idx1]
            mu[idx1] = mu2[idx1]
        if np.any(idx2):
            k[idx2] = k1[idx2]
            mu[idx2] = mu1[idx2]
    if np.all(idx):
        return k, mu, rhob

    # Need to run the DEM model for cases that are mixed
    idx1 = ~idx
    frac2 = frac2[idx1]

    # Only need to run the ODE on different initial conditions
    uni, idx_restore = np.unique(
        np.stack((k1[idx1], mu1[idx1], k2[idx1], mu2[idx1]), axis=1),
        return_inverse=True,
        axis=0,
    )
    k1, mu1, k2, mu2 = np.split(uni, 4, axis=1)
    # Must flatten arrays to avoid problems with dimensions
    k1 = k1.flatten()
    k2 = k2.flatten()
    mu1 = mu1.flatten()
    mu2 = mu2.flatten()

    k_vec = np.zeros(len(idx_restore))
    mu_vec = np.zeros(len(idx_restore))

    tot = k1.shape[0]

    # Must run ODE solver with values in increasing order, and must start at zero value
    i_x = np.argsort(frac2)
    frac2 = frac2[i_x]
    frac2 = np.insert(frac2, 0, 0)
    # Find reverse order
    i_y = np.argsort(i_x)

    # Too many data points create memory problems for ODE solver, set limit at lim = 1000
    lim = 1000
    runs = 1 if tot <= lim else tot // lim + 1
    for i in range(runs):
        start_i = i * lim
        end_i = min(tot, (i + 1) * lim)

        # Select part to run, cast to array in case it defaults to single value
        k1_tmp = np.array(k1[start_i:end_i])
        mu1_tmp = np.array(mu1[start_i:end_i])
        k2_tmp = np.array(k2[start_i:end_i])
        mu2_tmp = np.array(mu2[start_i:end_i])
        asp2_tmp = np.array(asp2[start_i:end_i])

        y0 = np.concatenate((k1_tmp, mu1_tmp))

        try:
            y_out = odeint(
                _demy_prime, y0, frac2, args=(k2_tmp, mu2_tmp, asp2_tmp), rtol=tol
            )
        except ValueError:
            raise ValueError(sys.exc_info())

        # Remove inserted zero-value in row 0, split output in k and mu
        # Pycharm reports wrong object type here, checked and found to be OK
        k_out, mu_out = np.split(y_out[1:, :], 2, axis=1)

        # Reorder rows back to original sequence of fraction values
        k_out = k_out[i_y, :]
        mu_out = mu_out[i_y, :]

        # Select correct column for the row values
        idx_pres_range = np.logical_and(idx_restore >= start_i, idx_restore < end_i)
        for j in range(len(idx_restore)):
            if idx_pres_range[j]:
                k_vec[j] = k_out[j, idx_restore[j] - start_i]
                mu_vec[j] = mu_out[j, idx_restore[j] - start_i]

    k[idx1] = k_vec
    mu[idx1] = mu_vec

    return k, mu, rhob


def _demy_prime(y, t, k2, mu2, asp2):
    """
    Used by DEM model.

    Parameters
    ----------
    y : np.ndarray
        y array.
    t : float
        t float.
    k2 : np.ndarray
        Bulk modulus of inclusions [Pa].
    mu2 : np.ndarray
        Shear modulus of inclusions [Pa].
    asp2 : np.ndarray
        Aspect ratio of inclusions [ratio].

    Returns
    -------
    np.ndarray
        k, mu array.

    Comments
    --------
    Written by T. Mukerji, Stanford University.
    Rewritten in Python by Harald Flesche, Equinor 2015.
    """
    # Input value vector consists of both k and mu values
    k, mu = np.split(y, 2)

    p, q = p_q_fcn(k, mu, k2, mu2, asp2)

    k_r_hs = (k2 - k) * p / (1 - t)
    mu_r_hs = (mu2 - mu) * q / (1 - t)

    return np.concatenate((k_r_hs, mu_r_hs))
