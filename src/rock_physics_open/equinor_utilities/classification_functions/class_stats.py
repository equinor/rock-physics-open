import numpy as np

NULL_CLASS = 0


def gen_class_stats(obs, class_val):
    """
    Generate statistics - mean, covariance and prior probability - for each
    class in the training data.	The observations are an n x m array, where n
    is the number of observations and m is the number of variables. With p
    classes the returned mean value will be an array of dimension p x m,
    covariance m x m x p and the class_id and prior probability p length vector.
    class_mean, class_cov, prior_prob, class_counts, class_id = gen_class_stats(obs, class_val).

    Parameters
    ----------
    obs : np.ndarray
        An nxm array of data samples (observations).
    class_val : np.ndarray
        n length vector with class ID of the observations. Assumed to
        be integer.

    Returns
    -------
    tuple
        class_mean, class_cov, prior_prob, class_counts, class_id : (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray).
    """

    n, m = obs.shape
    # Find number of classes. If class_val input is not integer, raise an exception
    if not (
        isinstance(class_val, np.ndarray)
        and issubclass(class_val.dtype.type, np.integer)
    ):
        raise ValueError(f"{__file__}: class values are not discrete numbers")

    class_id, class_counts = np.unique(class_val, return_counts=True)
    # Remove Null class
    idx_null = np.where(class_id == NULL_CLASS)
    class_id = np.delete(class_id, idx_null)
    class_counts = np.delete(class_counts, idx_null)
    p = class_id.shape[0]

    # Very simple prior probability - number of observations in each class
    # divided by total number of observations
    prior_prob = class_counts / n

    # Assign output arrays
    class_mean = np.zeros((p, m))
    class_cov = np.zeros((m, m, p))

    for i in range(len(class_id)):
        idx = class_val == class_id[i]
        class_mean[i, :] = np.mean(obs[idx, :], axis=0)
        class_cov[:, :, i] = np.cov(obs[idx, :], rowvar=False)

    return class_mean, class_cov, prior_prob, class_counts, class_id
