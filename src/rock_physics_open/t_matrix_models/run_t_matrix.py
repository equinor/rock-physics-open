import numpy as np

from .parse_t_matrix_inputs import parse_t_matrix_inputs
from .t_matrix_vector import calc_pressure_vec, pressure_input_utility


def run_t_matrix(
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
    scenario=None,
    fcn=None,
):
    """Function to run T-Matrix in different flavours, with or without pressure steps included.
    A frontend to running T-Matrix model, including testing of all input parameters in the parse_t_matrix_inputs.

    Parameters
    ----------
    k_min : np.ndarray
        N length numpy array, mineral bulk modulus [Pa].
    mu_min : np.ndarray
        N length numpy array, mineral shear modulus [Pa].
    rho_min : np.ndarray
        N length numpy array, mineral density [kg/m^3].
    k_fl : np.ndarray
        N length numpy array, fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        N length numpy array, mineral density [kg/m^3].
    phi : np.ndarray or float
        N length numpy array, total porosity [ratio].
    perm : np.ndarray or float
        Float or N length numpy array, permeability [mD].
    visco : np.ndarray or float
        Float or N length numpy array, fluid viscosity [cP].
    alpha : np.ndarray
        M or NxM length numpy array, inclusion aspect ratio [ratio].
    v : np.ndarray
        M or NxM length numpy array, inclusion concentration [ratio].
    tau : np.ndarray
        M length numpy array, relaxation time [s].
    frequency : float
        Single float, signal frequency [Hz].
    angle : float
        Single float, angle of symmetry plane [degree].
    frac_inc_con : np.ndarray or float
        Single float or N length numpy array, fraction of connected inclusions [ratio].
    frac_inc_ani : np.ndarray or float
        Single float or N length numpy array, fraction of anisotropic inclusions [ratio].
    pressure : [type], optional
        > 1 value list or numpy array in ascending order, effective pressure [Pa], by default None.
    scenario : int, optional
        Pre-set scenarios for alpha, v and tau, by default None.
    fcn : callable, optional
        Function with which to run the T-Matrix model, by default None.

    Returns
    -------
    tuple
        vp, vsv, vsh, rho: (np.ndarray, np.ndarray, np.ndarray, np.ndarray).
    """
    # Check all input parameters and make sure that they are on expected format and shape
    (
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
        pressure,
        fcn,
        ctrl_connected,
        ctrl_anisotropy,
    ) = parse_t_matrix_inputs(
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
        pressure,
        scenario,
        fcn,
    )

    # If there are no pressure steps to consider the whole task can be assigned to the T-Matrix function
    if pressure is None:
        vp, vsv, vsh, rho = fcn(
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
        )
        return vp, vsv, vsh, rho

    # Pressure steps is mimicking depletion, and inclusion shape parameters can be changed
    # In Remy's implementation it is generally possible to have different alpha's and v's for connected and
    # isolated part, but this is not included directly in Calculo's implementation in C++. For the time being,
    # this possibility is not included here

    log_length = phi.shape[0]
    pressure_steps = pressure.shape[0]
    # It is the change from initial pressure (first value in the pressure vector) that is input to the function
    # that estimates matrix pressure sensitivity
    delta_pres = np.diff(pressure)

    # Predefine output vectors
    vp = np.zeros((log_length, pressure_steps))
    vs_v = np.zeros((log_length, pressure_steps))
    vs_h = np.zeros((log_length, pressure_steps))
    rho_b_est = np.zeros((log_length, pressure_steps))

    # Matrix properties needed as input to calc_pressure_vec
    c0, s0, gd = pressure_input_utility(k_min, mu_min, log_length)

    for i in range(pressure_steps):
        vp[:, i], vs_v[:, i], vs_h[:, i], rho_b_est[:, i] = fcn(
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
        )

        if i != pressure_steps - 1:
            # Need to check for extreme cases with all connected or all isolated
            if ctrl_connected != 0:
                v_con = v * (phi * frac_inc_con).reshape(log_length, 1)
            else:
                v_con = np.zeros(v.shape)
            if ctrl_connected != 2:
                v_iso = v * (phi * (1 - frac_inc_con)).reshape(log_length, 1)
            else:
                v_iso = np.zeros(v.shape)
            alpha_con, v_con, alpha_iso, v_iso, tau, gamma = calc_pressure_vec(
                alpha,
                alpha,
                v_con,
                v_iso,
                c0,
                s0,
                gd,
                delta_pres[i],
                tau,
                np.zeros_like(tau),
                k_fl,
                ctrl_connected,
                frac_inc_ani,
            )

            # Post-process outputs from calc_pressure_vec to match required inputs to the T-Matrix function
            # frac_inc_con, v and alpha are inputs that need to be updated
            # As mentioned: assume that there is no distinction between alpha_con and alpha_iso, that could only
            # happen if they differed from the start

            # v_con, alpha_con or v_iso, alpha_iso can be returned as None from calc_pressure_vec
            if ctrl_connected == 0:
                v_con = np.zeros_like(v_iso)
                alpha = alpha_iso
            elif ctrl_connected == 2:
                alpha = alpha_con
                v_iso = np.zeros_like(v_con)
            else:
                alpha = alpha_con
            # Don't divide by zero
            idx_zero = phi == 0
            if np.any(idx_zero):
                v = v_con + v_iso
                no_zero = np.sum(idx_zero)
                v[~idx_zero, :] = v[~idx_zero, :] / phi[~idx_zero].reshape(
                    log_length - no_zero, 1
                )
            else:
                v = (v_con + v_iso) / phi.reshape(log_length, 1)

            # frac_inc_con is close to a single value, only numerical difference in the range 10^-16, can safely
            # take the average over the alphas to make it match expected input shape
            frac_inc_con = np.mean(v_con / (v_con + v_iso), axis=1)

    # Return variables for each pressure step
    vp_out = []
    vs_v_out = []
    vs_h_out = []

    # Only one rho is needed - no change
    rho_out = rho_b_est[:, 0]

    for i in range(pressure_steps):
        vp_out.append(vp[:, i])
        vs_v_out.append(vs_v[:, i])
        vs_h_out.append(vs_h[:, i])
    return vp_out + vs_v_out + vs_h_out + [rho_out]
