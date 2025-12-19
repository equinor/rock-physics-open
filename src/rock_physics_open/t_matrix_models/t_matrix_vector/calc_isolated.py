from typing import Literal

import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import (
    Array1D,
    Array2D,
    Array3D,
)

from .array_functions import array_inverse, array_matrix_mult
from .g_tensor import g_tensor_vec
from .iso_av import iso_av_vec


def calc_isolated_part_vec(
    c0: Array3D[np.float64],
    s_0: Array3D[np.float64],
    kappa_f: Array1D[np.float64],
    alpha: Array2D[np.float64],
    v: Array2D[np.float64],
    case_iso: Literal[0, 1, 2],
    frac_ani: float,
) -> Array3D[np.float64]:
    """
    Returns the first order correction tensor: sum of the concentrations and
    t-matrices of the isolated porosity (6x6 matrix).
    case_iso = 0 : 100% isotropic,
    case_iso = 1 : mixed isotropic and anisotropic porosity,
    case_iso = 2 : 100% anisotropic porosity.

    Parameters
    ----------
    c0 : np.ndarray
        Stiffness tensor of the host material (6x6xn).
    s_0 :  np.ndarray
        Inverse of stiffness tensor.
    kappa_f : np.ndarray
        Bulk modulus of the fluid (n length vector).
    alpha : np.ndarray
        Aspect ratios of all the inclusions (1xnumber of inclusions) vector).
    v : np.ndarray
        Concentration of all the inclusions (1xnumber of inclusions) vector).
    case_iso : int
        Control parameter.
    frac_ani : float
        Fraction of anisotropic inclusions.

    Returns
    -------
    np.ndarray
        c1: correction tensor.

    Notes
    -----
    29.10.2020 HFLE: Simplification: alpha for isotropic and anisotropic part are the same. This has been default in
    Remy's code, but this function had assumption that half of the inclusions could have different aspect ratio
    13.11.2020 HFLE: In case of zero porosity, this routine returns a zero tensor, whereas the correct should have
    been to return the host material tensor - corrected.

    """
    if not (c0.ndim == 3 and s_0.ndim == 3):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    log_length = c0.shape[0]
    if v.ndim != 2:
        v = np.tile(v.reshape(1, v.shape[0]), (log_length, 1))

    if alpha.ndim == 1 and alpha.shape[0] != c0.shape[0]:
        alpha = np.tile(alpha.reshape(1, alpha.shape[0]), (log_length, 1))
    alpha_len = alpha.shape[1]

    cn = np.zeros_like(c0)
    cn[:, 0:3, 0:3] = np.tile(kappa_f.reshape((log_length, 1, 1)), (1, 3, 3))
    cn_d = cn - c0
    i4 = np.tile(np.eye(6).reshape(1, 6, 6), (log_length, 1, 1))
    c1 = np.zeros_like(c0)

    # Will need G tensor for each alpha
    g_arr = [g_tensor_vec(c0=c0, s_0=s_0, alpha=alpha[:, i]) for i in range(alpha_len)]

    if case_iso != 1:
        # if there is only isotropic or anisotropic inclusions
        for j in range(alpha_len):
            t = array_matrix_mult(
                cn_d, array_inverse(i4 - array_matrix_mult(g_arr[j], cn_d))
            )
            if case_iso != 2:
                t = iso_av_vec(t)
            c1 = c1 + v[:, j].reshape(log_length, 1, 1) * t
    else:
        # Isotropic and anisotropic part
        for j in range(alpha_len):
            t = array_matrix_mult(
                cn_d, array_inverse(i4 - array_matrix_mult(g_arr[j], cn_d))
            )
            t = iso_av_vec(t)
            c1 = c1 + ((1 - frac_ani) * v[:, j]).reshape(log_length, 1, 1) * t
        for j in range(alpha_len):
            t = array_matrix_mult(
                cn_d, array_inverse(i4 - array_matrix_mult(g_arr[j], cn_d))
            )
            c1 = c1 + (frac_ani * v[:, j]).reshape(log_length, 1, 1) * t

    idx_zero = np.sum(v, axis=1) == 0.0
    if np.any(idx_zero):
        c1[idx_zero] = c0[idx_zero]

    return c1
