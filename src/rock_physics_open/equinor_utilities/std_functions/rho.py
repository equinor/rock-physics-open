from warnings import warn

import numpy as np
import numpy.typing as npt


def rho_b(
    phi: npt.NDArray[np.float64],
    rho_f: npt.NDArray[np.float64],
    rho_mat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate bulk density from porosity, fluid density and matrix density.

    Parameters
    ----------
    phi
        Porosity [fraction].
    rho_f
        Fluid bulk density [kg/m^3].
    rho_mat
        Matrix bulk density [kg/m^3].

    Returns
    -------
    Bulk density [kg/m^3].
    """
    return phi * rho_f + (1 - phi) * rho_mat


def rho_m(
    frac_cem: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculates matrix density as a combination of cement fraction and minerals.

    fracCem is defined as cement fraction relative to total volume.

    Parameters
    ----------
    frac_cem
        Cement fraction [fraction].
    phi
        Porosity [fraction].
    rho_cem
         Cement density [kg/m^3].
    rho_min
         Mineral density [kg/m^3].

    Returns
    -------
    Matrix density [kg/m^3].
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
