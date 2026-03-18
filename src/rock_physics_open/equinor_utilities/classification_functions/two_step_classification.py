import numpy as np
import numpy.typing as npt

from .class_stats import gen_class_stats
from .mahal_class import mahal_class

NULL_CLASS = 0


def gen_two_step_class_stats(
    obs: npt.NDArray[np.float64],
    class_val: npt.NDArray[np.int64],
    thresh: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    """
    The observations are an n x m array, where n is the number of observations and m is the number of variables.

    With p classes the returned mean value will be an array of dimension p x m,
    covariance m x m x p and the class_id and prior probability p length vector.

    Generate statistics - mean, covariance and prior probability - for each
    class in the training data.	Run a mahalanobis classification, and exclude
    values that have distance above the threshold. Generate class statistics again
    and return them.

    Parameters
    ----------
    obs
        An nxm array, where n is the number of samples and m is the number of variables.
    class_val
        A p length vector, where p is the number of classes, containing class_id (integer numbers).
    thresh
        Unclassified threshold.

    Returns
    -------
    class_mean
        Mean values per class after rejects.
    class_cov
        Covariance matrices per class after rejects.
    prior_prob
        Prior probability per class based on observation counts after rejects.
    class_counts
        Number of observations per class after rejects.
    class_id
        Label for each class.
    mahal_class_id
        Class id from the initial Mahalanobis classification.
    """
    mean_class_id, class_cov, _, _, class_id = gen_class_stats(obs, class_val)
    mahal_class_id = mahal_class(obs, mean_class_id, class_cov, class_id, thresh)[0]

    idx = mahal_class_id != NULL_CLASS
    mean_class_id, class_cov, prior_prob, class_counts, class_id = gen_class_stats(
        obs[idx], class_val[idx]
    )

    return mean_class_id, class_cov, prior_prob, class_counts, class_id, mahal_class_id
