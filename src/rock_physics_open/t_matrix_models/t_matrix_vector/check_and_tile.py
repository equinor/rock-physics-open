import numpy as np


def check_and_tile(s, t):
    """Utility  - code that was repeated in the T-Matrix functions.

    Parameters
    ----------
    s : np.ndarray
        s parameter.
    t : np.ndarray
        t parameter.

    Returns
    -------
    tuple
        i2_i2, log_length, alpha_length : (np.ndarray, int, int).
        i2_i2: array with upper left matrices set to 1, log_length: array dimension, alpha_length: number of inclusions.
    """
    if not (
        s.ndim == 3
        and t.ndim == 4
        and np.all(np.array([s.shape[1], s.shape[2], t.shape[1]]) == t.shape[2])
        and s.shape[0] == t.shape[0]
    ):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    log_length = t.shape[0]
    alpha_length = t.shape[3]

    tmp = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    i2_i2 = np.tile(tmp.reshape((1, 6, 6)), (log_length, 1, 1))

    return i2_i2, log_length, alpha_length
