import numpy as np

NULL_CLASS = 0


def posterior_probability(min_dist, dist):
    """
    Posterior probability, which is defined as the exponential of minimum distance divided by the sum of the
    exponentials of distance to all classes.

    Parameters
    ----------
    min_dist : np.ndarray
        Minimum class distance according to some metric.
    dist : np.ndarray
        All class distances, each class in a column in a two-dimensional array.

    Returns
    -------
    np.ndarray
        Posterior probability array.
    """
    dist *= -1.0
    n_exp = np.exp(dist)
    d_sum = n_exp.sum(axis=1)
    return np.exp(-min_dist) / d_sum
