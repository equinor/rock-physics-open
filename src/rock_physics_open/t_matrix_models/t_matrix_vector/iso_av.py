import numpy as np


def iso_av_vec(t):
    """Returns a (nx6x6) matrix t_bar averaged over all the orientations (isotropy).


    t_bar = np.array([

        [c11, c12, c12, 0, 0, 0],

        [c12, c11, c12, 0, 0, 0],

        [c12, c12, c11, 0, 0, 0],

        [0, 0, 0, 2 * c44, 0, 0],

        [0, 0, 0, 0, 2 * c44, 0],

        [0, 0, 0, 0, 0, 2 * c44]

    ])


    Parameters
    ----------
    t : np.ndarray
        An nx6x6 matrix which has a HTI symmetry.


    Returns
    -------
    np.ndarray
        Averaged value.
    """
    t_bar = np.zeros(t.shape)

    lambda_var = (
        t[:, 0, 0] + t[:, 2, 2] + 5 * t[:, 0, 1] + 8 * t[:, 0, 2] - 2 * t[:, 3, 3]
    ) / 15
    mu = (
        7 * t[:, 0, 0]
        + 2 * t[:, 2, 2]
        - 5 * t[:, 0, 1]
        - 4 * t[:, 0, 2]
        + 6 * t[:, 3, 3]
    ) / 30
    c11 = lambda_var + 2 * mu
    c12 = lambda_var
    c44 = mu

    for i in range(3):
        t_bar[:, i, i] = c11
        t_bar[:, i + 3, i + 3] = 2 * c44
    for i in range(2):
        t_bar[:, 0, i + 1] = c12
        t_bar[:, 1, 2 * i] = c12
        t_bar[:, 2, i] = c12

    return t_bar
