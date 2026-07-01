import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector

from .pq import p_q_fcn


def multi_sca(
    *args: npt.NDArray[np.float64],
    tol: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Estimate effective elastic moduli using a multi-phase SCA formulation.

    Uses Berryman's self-consistent (coherent potential approximation) method.
    Inputs are provided as repeated groups per phase in this order:
    `(k, mu, rho, asp, frac)`.

    For each phase tuple:
    - `k` is mineral bulk modulus [Pa].
    - `mu` is mineral shear modulus [Pa].
    - `rho` is mineral density [kg/m³].
    - `asp` is inclusion aspect ratio [ratio].
    - `frac` is the phase fraction in the matrix [fraction].

    The sum of all phase fractions must be 1.0 for each sample. If it differs,
    the fractions are normalized internally before iteration.

    Parameters
    ----------
    *args
        Repeated phase-property arrays in the order `(k, mu, rho, asp, frac)`
        for each phase.
    tol
        Scalar tolerance parameter for the SCA iterations [unitless].

    Returns
    -------
    k_sc
        Effective bulk modulus [Pa].
    mu_sc
        Effective shear modulus [Pa].
    rhob
        Effective density [kg/m³].

    Notes
    -----
    Based on a function by T. Mukerji (SRB, Stanford University, 1994) and
    ported to Python by Harald Flesche (Equinor, 2015).
    """
    if len(args) % 5 != 0:
        raise ValueError(
            "Call function with (k_sc,mu_sc,rhob) = multi_sca((k1,mu1,rho1,asp1,frac1, ...,k_n,mu_n,rho_n,asp_n,frac_n,tol)). Fractions must add up to 1.0"
        )

    args_ = dim_check_vector(args)

    # Proceed without the parameter, all other inputs should be arrays

    n_mins = len(args_) // 5

    # Sort inputs
    k_min: list[npt.NDArray[np.float64]] = []
    mu_min: list[npt.NDArray[np.float64]] = []
    rho_min: list[npt.NDArray[np.float64]] = []
    frac: list[npt.NDArray[np.float64]] = []
    asp: list[npt.NDArray[np.float64]] = []

    for i in range(n_mins):
        k_min.append(args_[5 * i])
        mu_min.append(args_[5 * i + 1])
        rho_min.append(args_[5 * i + 2])
        asp.append(args_[5 * i + 3])
        frac.append(args_[5 * i + 4])

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
        tol_ = tol * k_min[0]

        # Iterate until delta is less than the tolerance and the loop has run for
        # less than 3000 iterations (normally far less)
        while np.any(np.logical_and(np.greater(delta, tol_), np.less(niter, 3000))):
            p: list[npt.NDArray[np.float64]] = []
            q: list[npt.NDArray[np.float64]] = []
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
