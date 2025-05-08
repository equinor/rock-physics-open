import numpy as np


def velocity_vti_angles_vec(c_eff, rho_eff, angle):
    """Returns the P-velocity and  S-velocities.

    Parameters
    ----------
    c_eff : np.ndarray
        Effective stiffness tensor (nx6x6 matrix).
    rho_eff : np.ndarray
        Effective density.
    angle : np.ndarray or float
        The angle between the wave vector and the axis of symmetry.

    Returns
    -------
    tuple
        vp_out, vsv_out, vsh_out : (np.ndarray, np.ndarray, np.ndarray).
        vp_out, vsv_out, vsh_out : p-velocity, vertical polarisation s-velocity, horizontal polarisation s-velocity.
    """
    if not (
        c_eff.ndim == 3
        and rho_eff.ndim == 1
        and isinstance(angle, (float, int))
        and (c_eff.shape[0] == rho_eff.shape[0] and c_eff.shape[1] == c_eff.shape[2])
    ):
        raise ValueError("velocity_vti_angles: inconsistencies in input shapes")

    # vp45
    m45 = (
        (c_eff[:, 0, 0] - c_eff[:, 3, 3] / 2.0) * 0.5
        - (c_eff[:, 2, 2] - c_eff[:, 3, 3] / 2.0) * 0.5
    ) ** 2 + (c_eff[:, 0, 0] - c_eff[:, 3, 3] / 2.0) ** 2
    m45 = np.sqrt(m45)
    v2p = ((c_eff[:, 0, 0] + c_eff[:, 2, 2] + c_eff[:, 3, 3]) / 2.0 + m45) / (
        2.0 * rho_eff
    )
    sp = np.real(1.0 / (np.sqrt(v2p)))
    vp45 = 1.0 / sp

    # c11 = rho_eff * Vp90 ** 2
    c11 = np.real(c_eff[:, 0, 0])
    # c33 = rho_eff * Vp0 ** 2
    c33 = np.real(c_eff[:, 2, 2])
    # c44 = rho_eff * Vs0 ** 2
    c44 = np.real(c_eff[:, 3, 3] / 2.0)
    # c66 = rho_eff * Vs90 ** 2
    c66 = np.real(c_eff[:, 5, 5] / 2.0)

    c13 = -c44 + np.sqrt(
        4 * (rho_eff**2) * (vp45**4)
        - 2.0 * rho_eff * (vp45**2) * (c11 + c33 + 2.0 * c44)
        + (c11 + c44) * (c33 + c44)
    )

    rad_angle = (angle * np.pi) / 180.0

    m_real = (
        (c11 - c44) * (np.sin(rad_angle) ** 2) - (c33 - c44) * (np.cos(rad_angle) ** 2)
    ) ** 2 + ((c13 + c44) ** 2) * (np.sin(2.0 * rad_angle) ** 2)

    vp_out = np.sqrt(
        c11 * (np.sin(rad_angle) ** 2)
        + c33 * (np.cos(rad_angle) ** 2)
        + c44
        + np.sqrt(m_real)
    ) * np.sqrt(1.0 / (2.0 * rho_eff))

    vsv_out = np.sqrt(
        c11 * (np.sin(rad_angle) ** 2)
        + c33 * (np.cos(rad_angle) ** 2)
        + c44
        - np.sqrt(m_real)
    ) * np.sqrt(1.0 / (2.0 * rho_eff))

    vsh_out = np.sqrt(
        (c66 * (np.sin(rad_angle) ** 2) + c44 * (np.cos(rad_angle) ** 2)) / rho_eff
    )

    return vp_out, vsv_out, vsh_out
