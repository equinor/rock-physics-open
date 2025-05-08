import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities

# These routines are not finalised or used in any plugins yet
"""
	c11, c12, c13, c33, c44, c66 = c_ij_2_c_factors(cij)

	Transform a single stiffness tensor into components. VTI medium is assumed
	"""


def c_ij_2_c_factors(cij):
    """Transform a single stifness tensor into components. VTI medium is assumed

    Parameters
    ----------
    cij : np.ndarray
            A 6x6 matrix.

    Returns
    -------
    tuple
            (c11, c12, c13, c33, c44, c66).
    """
    if not isinstance(cij, np.ndarray):
        try:
            cij = np.array(cij, dtype=float)
        except ValueError:
            print("Input data can't be transformed into a NumPy array")
    try:
        num_samp = int(cij.size / 36)
        cij = cij.reshape((6, 6, num_samp))
        c11 = cij[0, 0, :].reshape(num_samp, 1)
        c12 = cij[0, 1, :].reshape(num_samp, 1)
        c13 = cij[0, 2, :].reshape(num_samp, 1)
        c33 = cij[2, 2, :].reshape(num_samp, 1)
        c44 = cij[3, 3, :].reshape(num_samp, 1)
        c66 = cij[5, 5, :].reshape(num_samp, 1)
        return c11, c12, c13, c33, c44, c66

    except ValueError:
        print("Input data is not a 6x6xN array")


def cfactors2cij(c11, c12, c13, c33, c44, c66):
    """Transform individual stiffness factors to stiffness tensor 6x6x(number of samples).

    Parameters
    ----------
    c11, c12, c13, c33, c44, c66 : np.ndarray
            All 1-dimensional of same length.

    Returns
    -------
    np.ndarray
            A 6xx6x(number of samples) stifness teensor.
    """
    c11, c12, c13, c33, c44, c66 = gen_utilities.dim_check_vector(
        (c11, c12, c13, c33, c44, c66)
    )

    num_samp = c11.shape[1]

    cij = np.zeros((6, 6, num_samp))
    cij[0, 0, :] = c11
    cij[0, 1, :] = c12
    cij[0, 2, :] = c13
    cij[1, 0, :] = c12
    cij[1, 1, :] = c11
    cij[1, 2, :] = c13
    cij[2, 0, :] = c13
    cij[2, 1, :] = c13
    cij[2, 1, :] = c33
    cij[3, 3, :] = c44
    cij[4, 4, :] = c44
    cij[5, 5, :] = c66

    return cij


def c_ij_2_thomsen(c, rho):
    """Thomsen parameter for weak anisotropy.

    Parameters
    ----------
    c : np.ndarray
            A (log of or single instance of) 6x6 elastic tensor.
    rho : np.ndarray
            Density - log of same length as c.

    Returns
    -------
    tuple
            alpha, beta, gamma, delta, epsilon.
    """
    # C matrix should be 6x6
    if not isinstance(c, np.ndarray):
        try:
            c = np.array(c, dtype=float)
        except ValueError:
            print("Input data can't be transformed into a NumPy array")
    try:
        num_samp = int(c.size / 36)
        c = c.reshape((6, 6, num_samp))
        rho = rho.reshape(num_samp, 1)
        alpha = np.sqrt(c[2, 2, :].reshape(num_samp, 1) / rho)
        beta = np.sqrt(c[3, 3, :].reshape(num_samp, 1) / rho)
        gamma = ((c[5, 5, :] - c[3, 3, :]) / (2 * c[3, 3, :])).reshape(num_samp, 1)
        epsilon = ((c[0, 0, :] - c[2, 2, :]) / (2 * c[2, 2, :])).reshape(num_samp, 1)
        delta = (
            ((c[0, 2, :] + c[3, 3, :]) ** 2 - (c[2, 2, :] - c[3, 3, :]) ** 2)
            / (2 * c[2, 2, :] * (c[2, 2, :] - c[3, 3, :]))
        ).reshape(num_samp, 1)

        return alpha, beta, epsilon, gamma, delta
    except ValueError:
        print("Input data is not a 6x6xN array")


def thomsen_2_c_ij(alpha, beta, gamma, delta, epsilon, rho):
    """Elastic stiffness. Assumptions:
            Thomsen's parameters apply for weak anisotropy in a transversly isotropic medium:

            c11 c12 c13  0   0   0

            c12 c11 c13  0   0   0

            c13 c13 c33  0   0   0

            0   0   0    c44 0   0

            0   0   0    0   c44 0

            0   0   0    0   0   c66

            Where c66 = 1/2(c11 - c12)

    Parameters
    ----------
    alpha, beta, gamma, delta, epsilon :
            Thomsen's parameters.
    rho :
            Bulk density.

    Returns
    -------
    tuple
            Elastic stiffness c11, c12, c13, c33, c44, c66.
    """
    alpha, beta, gamma, delta, epsilon = gen_utilities.dim_check_vector(
        (alpha, beta, gamma, delta, epsilon)
    )

    c33 = rho * alpha**2
    c44 = rho * beta**2
    c11 = c33 * (1 + 2 * epsilon)
    c66 = c44 * (1 + 2 * gamma)
    c12 = c11 - 2 * c66
    c13 = np.sqrt(2 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2) - c44

    return c11, c12, c13, c33, c44, c66
