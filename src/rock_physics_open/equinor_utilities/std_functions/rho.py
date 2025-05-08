from warnings import warn

import numpy as np


def rho_b(phi, rho_f, rho_mat):
    """
    Calculate bulk density from porosity, fluid density and matrix density.

    Parameters
    ----------
    phi: np.ndarray
        Porosity [fraction].
    rho_f: np.ndarray
        Fluid bulk density [kg/m^3].
    rho_mat: np.ndarray
        Matrix bulk density [kg/m^3].

    Returns
    -------
    np.ndarray
        rhob: bulk density [kg/m^3].
    """
    return phi * rho_f + (1 - phi) * rho_mat


def rho_m(frac_cem, phi, rho_cem, rho_min):
    """
    Calculates matrix density as a combination of cement fraction and minerals
    fracCem is defined as cement fraction relative to total volume.

    Parameters
    ----------
    frac_cem : np.ndarray
        Cement fraction [fraction].
    phi : np.ndarray
        Porosity [fraction].
    rho_cem : np.ndarray
         Cement density [kg/m^3].
    rho_min : np.ndarray
         Mineral density [kg/m^3].

    Returns
    -------
    np.ndarray
        rho_mat: matrix density [kg/m^3]
    """
    idx = np.logical_and(phi >= 0.0, phi < 1.0)
    if np.sum(idx) != len(phi):
        warn(
            f"{__file__}: phi out of range in {len(phi) - np.sum(idx)} sample",
        )

    rho_mat = np.ones_like(phi) * np.nan
    f_cem = frac_cem[idx] / (1 - phi[idx])

    rho_mat[idx] = f_cem * rho_cem[idx] + (1 - f_cem) * rho_min[idx]

    return rho_mat
