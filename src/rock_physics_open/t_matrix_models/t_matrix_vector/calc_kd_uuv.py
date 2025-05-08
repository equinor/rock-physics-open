import numpy as np


def calc_kd_uuvv_vec(kd):
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
