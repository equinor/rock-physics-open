import numpy as np

from .post_prob import posterior_probability

NULL_CLASS = 0


def mahal_class(obs, class_mean, class_cov, class_id, thresh=np.inf):
    """
    Mahalanobis classification routine. All data points are assigned a class, unless a threshold is set

    Parameters
    ----------
    obs : np.ndarray
        An nxm array, where n is the number of samples and m is the number of variables.
    class_mean : np.ndarray
        A pxm array, where p is the number of classes and m is the number of variables.
    class_cov : np.ndarray
        A pxm array, where p is the number of classes and m is the number of variables.
    class_id : np.ndarray
        A p length vector, where p is the number of classes, containing class_id (integer numbers).
    thresh : float
        Unclassified threshold.

    Returns
    -------
    tuple
        mahal_class_arr, mahal_dist, mahal_pp : (np.ndarray, np.ndarray, np.ndarray).
        mahal_class_arr: nx1 vector. The classes are numbered 1 to m, and unclassified samples (with distance
        greater than thresh) are set to 0,
        mahal_dist:	nx1 vector with mahalanobis distance from the closest class centre to sample,
        mahal_pp: nx1 vector with posterior probability based on the distance to each class
    """

    # Find dimensions
    n = obs.shape[0]
    p = class_mean.shape[0]

    # Assign matrices
    dist = np.zeros((n, p))

    # Calculate distance for each class
    for i in range(p):
        cov_inv = np.linalg.inv(class_cov[:, :, i])
        delta = obs - class_mean[i, :]
        dist[:, i] = np.sqrt(np.einsum("nj,jk,nk->n", delta, cov_inv, delta))

    # Find the shortest distance, assign class, calculate posterior probability and
    # filter out observations with distance greater than the threshold
    mahal_class_arr = np.choose(np.argmin(dist, axis=1), class_id)
    mahal_dist = np.amin(dist, axis=1)
    mahal_pp = posterior_probability(mahal_dist, dist)
    d_idx = mahal_dist > thresh
    mahal_class_arr[d_idx] = NULL_CLASS

    return mahal_class_arr, mahal_dist, mahal_pp
