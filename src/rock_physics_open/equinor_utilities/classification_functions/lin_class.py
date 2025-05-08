import numpy as np

NULL_CLASS = 0


def lin_class(obs, class_mean, class_id, thresh=np.inf):
    """
    Linear classification routine. All data points are assigned a class, unless a threshold is set.

    Parameters
    ----------
    obs : np.ndarray
        An nxm array, where n is the number of samples and m is the number of features.
    class_mean : np.ndarray
        A pxm array, where p is the number of classes and m is the number of features.
    class_id : np.ndarray
        A p length vector, where p is the number of classes, containing class_id (integer numbers).
    thresh : float
        Unclassified threshold.

    Returns
    -------
    tuple
        lin_class_arr, lin_dist : (np.ndarray, np.ndarray).
        lin_class_arr: nx1 vector. The classes are numbered according to class_id,
        and unclassified samples (with distance greater than thresh) are set to 0,
        lin_dist: nx1 vector with linear distance from the closest class centre to each sample.
    """

    # Find dimensions
    n = obs.shape[0]
    p = class_mean.shape[0]

    # Assign matrices
    dist = np.zeros((n, p))

    # Calculate distance for each class
    for i in range(p):
        dist[:, i] = np.sqrt(np.sum((obs - class_mean[i, :]) ** 2, axis=1))

    # Find the shortest distance, assign class, filter out observations with distance
    # greater than the threshold
    lin_class_arr = np.choose(np.argmin(dist, axis=1), class_id)
    lin_dist = np.amin(dist, axis=1)
    lin_class_arr[lin_dist > thresh] = NULL_CLASS

    return lin_class_arr, lin_dist
