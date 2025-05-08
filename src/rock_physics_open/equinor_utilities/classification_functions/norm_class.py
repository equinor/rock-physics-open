import numpy as np

NULL_CLASS = 0


def norm_class(obs, class_mean, class_cov, prior_prob, class_id, thresh=np.inf):
    """
    Normal distribution classification routine. All data points are assigned a
    class, unless a threshold is set. The "dist" calculated here is the quadratic
    discriminant function according to a Bayes classification. This is a negative
    number, and the closest class has the smallest absolute value.

    Parameters
    ----------
    obs : np.ndarray
        An nxm array, where n is the number of samples and m is the number of variables.
    class_mean : np.ndarray
        A pxm array, where p is the number of classes and m is the number of variables.
    class_cov : np.ndarray
        A pxm array, where p is the number of classes and m is the number of variables.
    prior_prob : np.ndarray
        A p length vector, where p is the number of classes containing prior probabilities for each class.
    class_id : np.ndarray
        A p length vector, where p is the number of classes, containing class_id (integer numbers).
    thresh : float
        Unclassified threshold.


    Returns
    --------
    tuple
        norm_class_id, norm_dist, norm_pp : (np.ndarray, np.ndarray, np.ndarray).
        norm_class_id:	nx1 vector. The classes are numbered 1 to m, and unclassified
        samples (with absolute distance greater than thresh) are set to 0,
        norm_dist: nx1 vector with quadratic discriminant distance from the closest class centre to sample,
        norm_pp: nx1 vector with posterior probability based on the distance to each class.
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
        dist[:, i] = (
            -0.5 * np.log(np.linalg.det(class_cov[:, :, i]))
            - 0.5 * np.sqrt(np.einsum("nj,jk,nk->n", delta, cov_inv, delta))
            + np.log(prior_prob[i])
        )

    # The discrimination function ("dist") are negative numbers. Choose the one
    # with the smallest value as the closest class
    norm_class_id = np.choose(np.argmax(dist, axis=1), class_id)
    norm_dist = np.amax(dist, axis=1)
    norm_pp = -np.exp(norm_dist) / np.sum(np.exp(dist), axis=1)

    # Compare the absolute value of the discriminator with the threshold
    norm_class_id[np.abs(norm_dist) > thresh] = NULL_CLASS

    return norm_class_id, norm_dist, norm_pp
