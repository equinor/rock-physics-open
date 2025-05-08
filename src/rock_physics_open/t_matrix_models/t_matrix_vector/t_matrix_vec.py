import warnings

import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities

from .array_functions import array_inverse, array_matrix_mult
from .calc_c_eff import calc_c_eff_visco_vec
from .calc_isolated import calc_isolated_part_vec
from .calc_kd import calc_kd_vec
from .calc_kd_uuv import calc_kd_uuvv_vec
from .calc_pressure import calc_pressure_vec
from .calc_td import calc_td_vec
from .calc_x import calc_x_vec
from .iso_av import iso_av_vec
from .pressure_input import pressure_input_utility
from .velocity_vti_angles import velocity_vti_angles_vec


def t_matrix_porosity_vectorised(
    k_min,
    mu_min,
    rho_min,
    k_fl,
    rho_fl,
    phi,
    perm,
    visco,
    alpha,
    v,
    tau,
    frequency,
    angle,
    frac_inc_con,
    frac_inc_ani,
    pressure=None,
):
    """Vectorised version of T-Matrix, pure Python version - mainly intended for cases where it is wished to follow
    the entire process through and study intermediate results. The C++ implementation is significantly faster.

    Description of inputs:
    Mineral propeties (effective properties, assumed mixed).
    Fluid properties (effective properties, assume mixed).

    Parameters
    ----------
    k_min : np.ndarray
        N length array, bulk modulus of matrix/mineral [Pa].
    mu_min : np.ndarray
        N length array, shear modulus of matrix/mineral [Pa].
    rho_min : np.ndarray
        N length array, density of matrix/mineral [kg/m^3].
    k_fl : np.ndarray
        N length array, bulk modulus of fluid [Pa].
    rho_fl : np.ndarray
        N length array, density of fluid [kg/m^3].
    phi : np.ndarray
        N length array, porosity [fraction].
    perm : np.ndarray
        Single float or N length array, permeability [mD].
    visco : np.ndarray
        Single float or N length array, fluid viscosity [cP].
    alpha : np.ndarray
        M length vector, aspect ratio for inclusion sets [ratio].
    v : np.ndarray
        M length vector, fraction of porosity belonging to each inclusion set [fraction].
    tau : np.ndarray
        M length vector, relaxation time constant [s].
    frequency : float
        Single float, measurement frequency (seismic, sonic, ultrasonic range) [Hz].
    angle : float
        Single float, angle of symmetry plane (0 = HTI, 90 = VTI medium).
    frac_inc_con : float or np.ndarray
        Single float or N length array, fraction of inclusions that are connected.
    frac_inc_ani : float or np.ndaray
        Single float or N length array, fraction of inclusions that are anisotropic.
    pressure : np.ndarray, optional
        L length array (normally 2), by default None.

    Returns
    -------
    tuple
        Of type (np.ndarray, np.ndarray, np.ndarray, np.ndarray). Vertical P-wave velocity [m/s], Vsv: Vertical polarity S-wave velocity [m/s],
        Vsh: Horizontal polarity S-wave velocity [m/s], rho_b: bulk density [kg/m^3].
    """
    log_length = len(phi)
    # Check that the inputs that should have the same length actually do
    (
        k_min,
        mu_min,
        rho_min,
        k_fl,
        rho_fl,
        phi,
        perm,
        visco,
    ) = gen_utilities.dim_check_vector(
        (k_min, mu_min, rho_min, k_fl, rho_fl, phi, perm, visco)
    )

    # Conversion to SI units
    perm = perm * 0.986923e-15
    visco = visco * 1.0e-2

    # Alpha, v and tau should be of the same length
    (alpha, v, tau) = gen_utilities.dim_check_vector((alpha, v, tau))
    shape_len = alpha.shape[0]

    # Shape parameters go into a dict, duplicate here, can be changed with pressure effect
    shape_params = {
        "alpha_con": alpha,
        "alpha_iso": alpha,
        "v_con": v * frac_inc_con,
        "v_iso": v * (1 - frac_inc_con),
        "tau_con": tau,
        "tau_iso": tau,
    }
    # Create case based on amount of anisotropic inclusions
    if frac_inc_ani == 0:
        case = {"iso": 0}  # All isotropic
        angle = 0
    elif frac_inc_ani == 1:
        case = {"iso": 2}  # All anisotropic
    else:
        case = {"iso": 1}  # Mixed case

    # Create case based on amount of connected inclusions
    if frac_inc_con == 0:
        case["con"] = 0
    elif frac_inc_con == 1:
        # All connected
        case["con"] = 2
    else:
        # Mixed case
        case["con"] = 1

    pressure_steps = 1
    delta_pres = 0.0
    if pressure is not None:
        try:
            # Pressure should be an object with __len__ attribute, cast to np.array
            pressure = np.array(pressure)
            pressure_steps = len(pressure)
        except TypeError:
            w_str = "pressure input must be a vector of length > 1 with increasing effective pressures"
            warnings.warn(w_str)
        else:
            delta_pres = np.diff(pressure)

    # Predefine output vectors
    vp = np.zeros((log_length, pressure_steps))
    vs_v = np.zeros((log_length, pressure_steps))
    vs_h = np.zeros((log_length, pressure_steps))
    rho_b_est = np.zeros((log_length, pressure_steps))

    # Matrix properties needed
    c0, s0, gd = pressure_input_utility(k_min, mu_min, log_length)

    i4 = np.tile(np.eye(6).reshape(1, 6, 6), (log_length, 1, 1))

    # Vectorise v and alpha
    v_con = None
    alpha_con = None
    v_iso = None
    alpha_iso = None
    if case["con"] != 0:
        v_con = phi.reshape(log_length, 1) * shape_params["v_con"].reshape(1, shape_len)
        alpha_con = np.ones(v_con.shape) * shape_params["alpha_con"].reshape(
            1, shape_len
        )

    if case["con"] != 2:
        v_iso = phi.reshape(log_length, 1) * shape_params["v_iso"].reshape(1, shape_len)
        alpha_iso = np.ones(v_iso.shape) * shape_params["alpha_iso"].reshape(
            1, shape_len
        )

    for i in range(pressure_steps):
        # Check if v(j) > alpha(j)for maximum porosity. If true, set v(j) = alpha(j)/2 to make sure
        # the numbers of inclusions in the system is not violating the
        # approximations for effective medium theories.
        phi_con = 0.0
        phi_iso = 0.0
        if case["con"] != 0:
            idx_v_con = v_con * phi.reshape(log_length, 1) > alpha_con
            if np.any(idx_v_con):
                v_con[idx_v_con] = alpha_con[idx_v_con] / 2
            phi_con = np.sum(v_con, axis=1)
        if case["con"] != 2:
            idx_v_iso = v_iso * phi.reshape(log_length, 1) > alpha_iso
            if np.any(idx_v_iso):
                v_iso[idx_v_iso] = alpha_iso[idx_v_iso] / 2
            phi_iso = np.sum(v_iso, axis=1)
        # May seem unnecessary, but V-vectors can change with changing pressure
        phi_out = phi_con + phi_iso

        # Creating reference matrix
        vs_min = np.sqrt(mu_min / rho_min)

        if case["con"] == 0:
            # All isolated
            # Isolated part: calculated c1 tensor (sum over all the isolated t-matrices and concentrations
            c1 = calc_isolated_part_vec(
                c0, s0, k_fl, alpha_iso, v_iso, case["iso"], frac_inc_ani
            )
            c_eff = c0 + array_matrix_mult(
                c1, array_inverse(i4 + array_matrix_mult(gd, c1))
            )
            gamma = np.zeros_like(tau)
        else:
            kd = calc_kd_vec(c0, i4, s0, alpha_con)
            kd_uuvv = calc_kd_uuvv_vec(kd)
            gamma = (1 - k_fl / k_min).reshape(log_length, 1) + k_fl.reshape(
                log_length, 1
            ) * kd_uuvv
            if case["con"] == 1:
                # Mix
                # Isolated part: calculated c1 tensor (sum over all the isolated t-matrices and concentrations
                c1 = calc_isolated_part_vec(
                    c0, s0, k_fl, alpha_iso, v_iso, case["iso"], frac_inc_ani
                )
            else:  # case['con'] == 2
                # All connected - only c1 differs from the most general case
                c1 = np.zeros((log_length, 6, 6))
            # Connected part Calculate dry properties:
            td = calc_td_vec(c0, i4, s0, kd, alpha_con)
            # iso averaging the isotropic porosity
            td_bar = iso_av_vec(td)
            # Calculate the fluid effect
            x = calc_x_vec(s0, td)
            # iso averaging the isotropic porosity
            x_bar = iso_av_vec(x)
            # Frequency dependent stiffness
            c_eff = calc_c_eff_visco_vec(
                vs_min,
                perm,
                visco,
                v_con,
                gamma,
                tau,
                kd_uuvv,
                k_min,
                k_fl,
                c0,
                s0,
                c1,
                td,
                td_bar,
                x,
                x_bar,
                gd,
                frequency,
                frac_inc_ani,
            )
        # Effective density
        rho_b_est[:, i] = phi_out * rho_fl + (1 - phi_out) * rho_min
        vp[:, i], vs_v[:, i], vs_h[:, i] = velocity_vti_angles_vec(
            c_eff, rho_b_est[:, i], angle
        )

        if i != pressure_steps - 1:
            alpha_con, v_con, alpha_iso, v_iso, tau, gamma = calc_pressure_vec(
                alpha_con,
                alpha_iso,
                v_con,
                v_iso,
                c0,
                s0,
                gd,
                delta_pres[i],
                tau,
                gamma,
                k_fl,
                case["con"],
                frac_inc_ani,
            )

    return vp, vs_v, vs_h, rho_b_est
