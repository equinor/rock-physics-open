import numpy as np

from .array_functions import array_matrix_mult


def g_tensor_vec(c0, s_0, alpha):
    """Returns the Eshelby green's tensor (nx6x6 array).

    Parameters
    ----------
    c0 : np.ndarray
        n stiffness tensors of the host material (nx6x6 array).
    s_0 : np.ndarray
        Inverse of stiffness tensor.
    alpha : float
        Aspect ratio for single inclusion.

    Returns
    -------
    np.ndarray
        g-tensor.

    Raises
    ------
    ValueError
        If mismatch in input dimension/shape.
    """
    if not (
        c0.ndim == 3
        and c0.shape[1] == c0.shape[2]
        and s_0.ndim == 3
        and s_0.shape[1] == s_0.shape[2]
    ):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    log_len = c0.shape[0]
    mu = c0[:, 3, 3] / 2
    kappa = c0[:, 0, 0] - (4 / 3) * mu
    pois_ratio = (3 * kappa - 2 * mu) / (2 * (3 * kappa + mu))

    s_r = np.zeros(c0.shape)

    s_11 = np.zeros(log_len)
    s_12 = np.zeros(log_len)
    s_13 = np.zeros(log_len)
    s_21 = np.zeros(log_len)
    s_22 = np.zeros(log_len)
    s_23 = np.zeros(log_len)
    s_31 = np.zeros(log_len)
    s_32 = np.zeros(log_len)
    s_33 = np.zeros(log_len)
    s_44 = np.zeros(log_len)
    s_55 = np.zeros(log_len)
    s_66 = np.zeros(log_len)

    if isinstance(alpha, float):
        alpha = np.ones(log_len) * alpha
    # Check for valid range of alpha - those outside are returned with zero matrices
    idx_in = (alpha >= 0) & (alpha < 1)
    idx_one = alpha == 1
    alpha_in = alpha[idx_in]
    pois_ratio_in = pois_ratio[idx_in]
    pois_ratio_one = pois_ratio[idx_one]

    if np.any(idx_in):
        q = (alpha_in / (1 - alpha_in**2) ** (3 / 2)) * (
            np.arccos(alpha_in) - alpha_in * (1 - alpha_in**2) ** (1 / 2)
        )
        s_11[idx_in] = (3 / (8 * (1 - pois_ratio_in))) * (
            alpha_in**2 / (alpha_in**2 - 1)
        ) + (1 / (4 * (1 - pois_ratio_in))) * (
            1 - 2 * pois_ratio_in - 9 / (4 * (alpha_in**2 - 1))
        ) * q
        s_33[idx_in] = (1 / (2 * (1 - pois_ratio_in))) * (
            1
            - 2 * pois_ratio_in
            + (3 * alpha_in**2 - 1) / (alpha_in**2 - 1)
            - (1 - 2 * pois_ratio_in + 3 * alpha_in**2 / (alpha_in**2 - 1)) * q
        )
        s_12[idx_in] = (1 / (4 * (1 - pois_ratio_in))) * (
            alpha_in**2 / (2 * (alpha_in**2 - 1))
            - (1 - 2 * pois_ratio_in + 3 / (4 * (alpha_in**2 - 1))) * q
        )
        s_13[idx_in] = (1 / (2 * (1 - pois_ratio_in))) * (
            -(alpha_in**2) / (alpha_in**2 - 1)
            + 0.5 * (3 * alpha_in**2 / (alpha_in**2 - 1) - (1 - 2 * pois_ratio_in)) * q
        )
        s_31[idx_in] = (1 / (2 * (1 - pois_ratio_in))) * (
            2 * pois_ratio_in
            - 1
            - 1 / (alpha_in**2 - 1)
            + (1 - 2 * pois_ratio_in + 3 / (2 * (alpha_in**2 - 1))) * q
        )
        s_66[idx_in] = (1 / (4 * (1 - pois_ratio_in))) * (
            alpha_in**2 / (2 * (alpha_in**2 - 1))
            + (1 - 2 * pois_ratio_in - 3 / (4 * (alpha_in**2 - 1))) * q
        )
        s_44[idx_in] = (1 / (4 * (1 - pois_ratio_in))) * (
            1
            - 2 * pois_ratio_in
            - (alpha_in**2 + 1) / (alpha_in**2 - 1)
            - 0.5
            * (1 - 2 * pois_ratio_in - (3 * (alpha_in**2 + 1)) / (alpha_in**2 - 1))
            * q
        )
        s_22[idx_in] = s_11[idx_in]
        s_21[idx_in] = s_12[idx_in]
        s_23[idx_in] = s_13[idx_in]
        s_32[idx_in] = s_31[idx_in]
        s_55[idx_in] = s_44[idx_in]
    if np.any(idx_one):
        s_11[idx_one] = (5 * pois_ratio_one - 1) / (15 * (1 - pois_ratio_one)) + (
            2 * (4 - 5 * pois_ratio_one)
        ) / (15 * (1 - pois_ratio_one))
        s_12[idx_one] = (5 * pois_ratio_one - 1) / (15 * (1 - pois_ratio_one))
        s_13[idx_one] = (5 * pois_ratio_one - 1) / (15 * (1 - pois_ratio_one))
        s_31[idx_one] = (5 * pois_ratio_one - 1) / (15 * (1 - pois_ratio_one))
        s_44[idx_one] = (4 - 5 * pois_ratio_one) / (15 * (1 - pois_ratio_one))
        s_22[idx_one] = s_11[idx_one]
        s_33[idx_one] = s_11[idx_one]
        s_21[idx_one] = s_12[idx_one]
        s_32[idx_one] = s_31[idx_one]
        s_23[idx_one] = s_13[idx_one]
        s_55[idx_one] = s_44[idx_one]
        s_66[idx_one] = s_44[idx_one]

    s_r[:, 0, 0] = s_11
    s_r[:, 1, 1] = s_22
    s_r[:, 2, 2] = s_33
    s_r[:, 3, 3] = 2 * s_44
    s_r[:, 4, 4] = 2 * s_55
    s_r[:, 5, 5] = 2 * s_66
    s_r[:, 0, 1] = s_12
    s_r[:, 0, 2] = s_13
    s_r[:, 1, 0] = s_21
    s_r[:, 1, 2] = s_23
    s_r[:, 2, 0] = s_31
    s_r[:, 2, 1] = s_32

    return array_matrix_mult(-s_r, s_0)
