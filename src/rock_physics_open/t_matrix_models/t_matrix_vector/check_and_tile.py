import numpy as np

from rock_physics_open.equinor_utilities.various_utilities.types import Array3D, Array4D


def check_and_tile(
    s: Array3D[np.float64],
    t: Array4D[np.float64],
) -> tuple[Array3D[np.float64], int, int]:
    """Utility  - code that was repeated in the T-Matrix functions.

    Parameters
    ----------
    s
        s parameter.
    t
        t parameter.

    Returns
    -------
    i2_i2
        Array with upper left matrices set to 1.
    log_length
        Array dimension.
    alpha_length
        Number of inclusions.

    Raises
    ------
    ValueError
        If input dimensions or shapes are inconsistent.
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
