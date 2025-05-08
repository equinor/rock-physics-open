import numpy as np

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector

from .pq import p_q_fcn


def multi_sca(*args, tol):
    """SCA - Effective elastic moduli using Berryman's Self-Consistent
    (Coherent Potential Approximation) method.

    -----------------------------------------------

    Inputs for each phase:

    k:      mineral bulk modulus [Pa]

    mu:     mineral shear modulus [Pa]

    rho:    mineral density [kg/m^3]

    asp:    aspect ratio for inclusions [ratio]

    frac:   fraction of matrix that is made up of this mineral [fraction]

    Additional inputs:

    tol:    tolerance parameter (scalar) [unitless]

    The sum of all fractions must add to one, and it will be normalised to one if it differs.

    =======================================================

    Based on function by T. Mukerji, SRB, Stanford University, 1994.

    Ported to Python by Harald Flesche, Equinor 2015.

    Parameters
    ----------
    args : list or tuple
        List or tuple containing multiples of elastic properties as explained above, all numpy arrays.
    tol : float
        Tolerance for the SCA iterations.

    Returns
    -------
    tuple
        k_sc, mu_sc, rhob : (np.ndarray, np.ndarray, np.ndarray).
        effective medium properties k_sc: bulk modulus [Pa], mu_sc: shear modulus [Pa], rhob: bulk density [kg/m^3].
    """
    if len(args) % 5 != 0:
        raise ValueError(
            "Call function with (k_sc,mu_sc,rhob) = multi_sca((k1,mu1,rho1,asp1,frac1, ...,k_n,mu_n,rho_n,asp_n,"
            "frac_n,tol)). Fractions must add up to 1.0"
        )

    args = dim_check_vector(args)

    # Proceed without the parameter, all other inputs should be arrays

    n_mins = len(args) // 5

    # Sort inputs
    k_min = []
    mu_min = []
    rho_min = []
    frac = []
    asp = []

    for i in range(n_mins):
        k_min.append(args[5 * i])
        mu_min.append(args[5 * i + 1])
        rho_min.append(args[5 * i + 2])
        asp.append(args[5 * i + 3])
        frac.append(args[5 * i + 4])

    # Check and normalise frac
    tot_frac = np.sum(np.vstack(frac[:]), 0)
    if np.any(tot_frac != 1.0):
        for i in range(n_mins):
            frac[i] = frac[i] / tot_frac

    # Initiate k_sc and mu_sc with weighted sum of phases
    k_sc = np.sum(np.vstack(k_min[:]) * np.vstack(frac[:]), 0)
    mu_sc = np.sum(np.vstack(mu_min[:]) * np.vstack(frac[:]), 0)
    rhob = np.sum(np.vstack(rho_min[:]) * np.vstack(frac[:]), 0)

    idx = np.any(np.vstack(frac[:]) == 1.0, 0)
    if np.any(idx):
        # If any of the samples contain 100% of one phase - substitute the result
        # with the mineral properties of that phase
        for i in range(n_mins):
            idx1 = frac[i] == 1.0
            if np.any(idx1):
                k_sc[idx1] = k_min[i][idx1]
                mu_sc[idx1] = mu_min[i][idx1]
            # Continue with the real mixtures
            k_min[i] = k_min[i][~idx]
            mu_min[i] = mu_min[i][~idx]
            asp[i] = asp[i][~idx]
            frac[i] = frac[i][~idx]
    if not np.all(idx):
        niter = 0
        # Initiate the delta with a value that makes the while-loop run
        delta = k_sc[~idx]
        # Express tolerance in terms of k0
        tol = tol * k_min[0]

        # Iterate until delta is less than the tolerance and the loop has run for
        # less than 3000 iterations (normally far less)
        while np.any(np.logical_and(np.greater(delta, tol), np.less(niter, 3000))):
            p = []
            q = []
            for i in range(n_mins):
                p_tmp, q_tmp = p_q_fcn(
                    k_sc[~idx], mu_sc[~idx], k_min[i], mu_min[i], asp[i]
                )
                p.append(p_tmp)
                q.append(q_tmp)
            k_new = np.sum(
                np.vstack(frac[:]) * np.vstack(k_min[:]) * np.vstack(p[:]), 0
            ) / np.sum(np.vstack(frac[:]) * np.vstack(p[:]), 0)
            mu_new = np.sum(
                np.vstack(frac[:]) * np.vstack(mu_min[:]) * np.vstack(q[:]), 0
            ) / np.sum(np.vstack(frac[:]) * np.vstack(q[:]), 0)

            delta = np.absolute(k_sc[~idx] - k_new)
            k_sc[~idx] = k_new
            mu_sc[~idx] = mu_new

            niter = niter + 1

    return k_sc, mu_sc, rhob
