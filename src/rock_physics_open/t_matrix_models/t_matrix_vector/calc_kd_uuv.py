import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import Array2D, Array4D


def calc_kd_uuvv_vec(kd: Array4D[np.float64]) -> Array2D[np.float64]:
    """Returns the sum of dry k_uuvv.

    Parameters
    ----------
    kd : np.ndarray
        The dry K-tensor (n, 6,6,(numbers of inclusions)) matrix.

    Returns
    -------
    np.ndarray
        Summed elements.

    """
    return np.sum(kd[:, :3, :3, :], axis=(1, 2))
