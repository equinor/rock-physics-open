import numpy as np


def p_q_fcn(k, mu, k2, mu2, asp):
    """
    Geometric factors used in inclusion models.

    References
    ----------
    The rock physics handbook, Gary Mavko et al.

    Parameters
    ----------
    k : np.ndarray
        Bulk modulus of phase 1 [Pa].
    mu : np.ndarray
        Shear modulus of phase 1 [Pa].
    k2 : np.ndarray
        Bulk modulus of phase 2 [Pa].
    mu2 : np.ndarray
        Shear modulus of phase 2 [Pa].
    asp : np.ndarray
        Aspect ratio [ratio].

    Returns
    -------
    tuple
        p, q : (np.ndarray, np.ndarray).
        geometric factors p and q.
    """

    # Functions theta and fn defaults to 2/3 and -2/5 for asp == 1.0
    idx_oblate = np.less(asp, 1.0)
    idx_prolate = np.greater(asp, 1.0)
    theta = 2.0 / 3.0 * np.ones(asp.shape)
    fn = -2.0 / 5.0 * np.ones(asp.shape)

    if np.any(idx_oblate):
        theta[idx_oblate] = (
            asp[idx_oblate] / ((1 - asp[idx_oblate] ** 2) ** (3 / 2))
        ) * (
            np.arccos(asp[idx_oblate])
            - asp[idx_oblate] * np.sqrt(1 - asp[idx_oblate] ** 2)
        )
        fn[idx_oblate] = (asp[idx_oblate] ** 2 / (1 - asp[idx_oblate] ** 2)) * (
            3 * theta[idx_oblate] - 2
        )

    if np.any(idx_prolate):
        theta[idx_prolate] = (
            asp[idx_prolate] / ((asp[idx_prolate] ** 2 - 1) ** (3 / 2))
        ) * (
            asp[idx_prolate] * np.sqrt(asp[idx_prolate] ** 2 - 1)
            - np.arccosh(asp[idx_prolate])
        )
        fn[idx_prolate] = (asp[idx_prolate] ** 2 / (asp[idx_prolate] ** 2 - 1)) * (
            2 - 3 * theta[idx_prolate]
        )

    nu = (3 * k - 2 * mu) / (2 * (3 * k + mu))
    r = (1 - 2 * nu) / (2 * (1 - nu))
    a = mu2 / mu - 1
    b = (1 / 3) * (k2 / k - mu2 / mu)

    f1 = 1 + a * (
        (3 / 2) * (fn + theta) - r * ((3 / 2) * fn + (5 / 2) * theta - (4 / 3))
    )

    f2 = (
        1
        + a * (1 + (3 / 2) * (fn + theta) - (r / 2) * (3 * fn + 5 * theta))
        + b * (3 - 4 * r)
        + (a / 2)
        * (a + 3 * b)
        * (3 - 4 * r)
        * (fn + theta - r * (fn - theta + 2 * theta**2))
    )

    f3 = 1 + a * (1 - (fn + (3 / 2) * theta) + r * (fn + theta))

    f4 = 1 + (a / 4) * (fn + 3 * theta - r * (fn - theta))

    f5 = a * (-fn + r * (fn + theta - (4 / 3))) + b * theta * (3 - 4 * r)

    f6 = 1 + a * (1 + fn - r * (fn + theta)) + b * (1 - theta) * (3 - 4 * r)

    f7 = (
        2
        + (a / 4) * (3 * fn + 9 * theta - r * (3 * fn + 5 * theta))
        + b * theta * (3 - 4 * r)
    )

    f8 = a * (1 - 2 * r + (fn / 2) * (r - 1) + (theta / 2) * (5 * r - 3)) + b * (
        1 - theta
    ) * (3 - 4 * r)

    f9 = a * ((r - 1) * fn - r * theta) + b * theta * (3 - 4 * r)

    p = f1 / f2
    q = (1 / 5) * ((2 / f3) + (1 / f4) + ((f4 * f5 + f6 * f7 - f8 * f9) / (f2 * f4)))

    return p, q
