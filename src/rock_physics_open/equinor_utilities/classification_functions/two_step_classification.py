from .class_stats import gen_class_stats
from .mahal_class import mahal_class

NULL_CLASS = 0


def gen_two_step_class_stats(obs, class_val, thresh):
    """
    The observations are an n x m array, where n
    is the number of observations and m is the number of variables. With p
    classes the returned mean value will be an array of dimension p x m,
    covariance m x m x p and the class_id and prior probability p length vector.

    Generate statistics - mean, covariance and prior probability - for each
    class in the training data.	Run a mahalanobis classification, and exclude
    values that have distance above the threshold. Generate class statistics again
    and return them.

    Parameters
    ----------
    obs : np.ndarray
        An nxm array, where n is the number of samples and m is the number of variables.
    class_val : np.ndarray
        A p length vector, where p is the number of classes, containing class_id (integer numbers).
    thresh : float
        Unclassified threshold.

    Returns
    -------
    tuple
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        class_mean, class_cov: statistics for each class after rejects;
        prior_prob, class_counts: based on number of observations in each class,
        find the prior probability of each class;
        class_id: label for each class;
        mahal_class_id: class id from mahalanobis classification.
    """
    mean_class_id, class_cov, _, _, class_id = gen_class_stats(obs, class_val)
    mahal_class_id = mahal_class(obs, mean_class_id, class_cov, class_id, thresh)[0]

    idx = mahal_class_id != NULL_CLASS
    mean_class_id, class_cov, prior_prob, class_counts, class_id = gen_class_stats(
        obs[idx], class_val[idx]
    )

    return mean_class_id, class_cov, prior_prob, class_counts, class_id, mahal_class_id
