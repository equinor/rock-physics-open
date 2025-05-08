from warnings import warn

import numpy as np

from rock_physics_open.equinor_utilities import std_functions

from .constant_cement_models import constant_cement_model_dry
from .friable_models import friable_model_dry
from .patchy_cement_model import constant_cement_model_pcm


def patchy_cement_pressure_fluid_substitution(
    k_min,
    mu_min,
    rho_min,
    k_cem,
    mu_cem,
    rho_cem,
    k_fl_old,
    rho_fl_old,
    k_fl_new,
    rho_fl_new,
    phi,
    p_eff_old,
    p_eff_new,
    vp_old,
    vs_old,
    rho_b_old,
    p_eff_low,
    frac_cem_up,
    frac_cem,
    shear_red,
    phi_c,
    coord_num_func,
    n,
    model_type="weight",
    phi_below_zero="disregard",
    phi_above_phi_c="snap",
    k_sat_above_k_min="disregard",
    above_upper_bound="snap",
    below_lower_bound="disregard",
):
    """
    Patchy cement model, developed by Per Avseth and Norunn Skjei. A sandstone model in which part of the
    matrix is pressure sensitive, and part is cemented.

    Control parameters with default value:
    model_type = 'weight': A sample by sample weight between cemented (upper) and friable (lower) bound are calculated.
        The alternative is 'cement_fraction', in which the weights are calculated from one constant cement model

    phi_below_zero='disregard': Values with negative porosity will either have their value snapped to zero ('snap')
    or the input value will be returned ('disregard'). These option should give the same
    result.

    phi_above_phi_c = 'disregard':	Values with porosity above critical porosity will either have their values snapped
    to critical porosity ('sbap') or the input value will be returned ('disregard').

    k_sat_above_k_min='disregard': Values with saturated bulk modulus above the mineral bulk modulus will have their
    value snapped to the mineral modulus ('snap') or the input value will be returned ('disregard').

    above_upper_bound='snap': 	Values with moduli above upper bound will either be snapped to upper bound ('snap')
    or the input value will be returned ('disregard').

    below_lower_bound='disregard': Values with moduli below lower bound will either be snapped to lower bound ('snap')
    or the input value returned ('disregard').

    Comment
    -------
    Based on program by Norunn Skjei.
    Translated to Python by Harald Flesche.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral density [kg/m^3]
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement density [kg/m^3]
    k_fl_old : np.ndarray
        Initial fluid bulk modulus for sandstone fluid [Pa].
    rho_fl_old : np.ndarray
        Initial fluid bulk density for sandstone fluid [kg/m^3].
    k_fl_new : np.ndarray
        Substituted fluid bulk modulus for sandstone fluid [Pa].
    rho_fl_new : np.ndarray
        Substituted fluid bulk density for sandstone fluid [kg/m^3].
    phi : np.ndarray
        Total porosity [fraction].
    p_eff_old : np.ndarray
        Initial effective pressure [Pa].
    p_eff_new : np.ndarray
        Substituted effective pressure [Pa].
    p_eff_low : float
        Lower bound effective pressure [Pa].
    frac_cem_up : float
        Upper bound cement volume fraction [fraction].
    frac_cem : cement fraction of constant cement model - representative for observed data. Must be lower than
        frac_cem_up
    shear_red : float
        Shear reduction factor for sandstone [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant.
    vp_old : np.ndarray
        Initial p-wave velocity [m/s].
    vs_old : np.ndarray
        Initial s-wave velocity [m/s].
    rho_b_old : np.ndarray
        Initial bulk density [kg/m^3].
    model_type : str
        Version of model, either 'weight' or 'cement_fraction'
    phi_below_zero : str
        Control for handling of negative porosity samples.
    phi_above_phi_c : str
        Control for handling of porosity samples above critical porosity.
    k_sat_above_k_min : str
        Control for handling of bulk modulus samples above mineral bulk modulus.
    above_upper_bound : str
        Control for handling of samples above the upper bound.
    below_lower_bound : str
        Control for handling of samples below the lower bound.

    Returns
    -------
    tuple
        vp_new, vs_new, rho_b_new, ai_new, vpvs_new, k_sat_new, mu_new, wk, wmu, idx_valid :
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vp_new :Saturated P-velocity [m/s] after fluid and pressure substitution,
        vs_new : Saturated S-velocity [m/s] after fluid and pressure substitution,
        rho_b_new : Saturated density [kg/m3] after fluid and pressure substitution,
        ai_new : Saturated acoustic impedance [kg/m3 x m/s] after fluid and pressure substitution,
        vpvs_new : Saturated Vp/Vs ratio [unitless] after fluid and pressure substitution,
        k_sat_new : New saturated rock bulk modulus [Pa],
        mu_new : New shear modulus [Pa],
        wk, wmu : weights for k and mu between upper and lower bound,
        idx_valid : samples fluid substitution is performed
    """

    # Original saturated bulk and shear modulus
    k_sat_old, mu_old = std_functions.moduli(vp_old, vs_old, rho_b_old)

    # Handling of samples that violate the assumptions of the Patchy Cement
    # substitution:
    #   1: phi <= 0
    #   2: phi > phi_c
    #   3: k_sat > k_min
    idx1 = _handle_exceptions_part_1(
        phi,
        phi_c,
        k_sat_old,
        mu_old,
        k_min,
        mu_min,
        phi_below_zero=phi_below_zero,
        phi_above_phi_c=phi_above_phi_c,
        k_sat_above_k_min=k_sat_above_k_min,
    )

    # k_dry for original pressure from Gassmann for valid samples
    k_dry_old = np.ones_like(phi) * np.nan
    k_dry_old[~idx1] = std_functions.gassmann_dry(
        k_sat_old[~idx1], phi[~idx1], k_fl_old[~idx1], k_min[~idx1]
    )

    # For common handling of zero-porosity point: calculate effective
    # properties using Hashin-Shtrikman
    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_min, mu_min, frac_cem_up, bound="lower"
    )

    # Lower bound for estimation of weight W (input moduli in Pa, pressure in Pa)
    k_low, mu_low = friable_model_dry(
        k_zero,
        mu_zero,
        phi,
        p_eff_low * np.ones_like(phi),
        phi_c,
        coord_num_func,
        n,
        shear_red,
    )

    # Upper bounds for estimation of weight W
    k_up, mu_up = constant_cement_model_dry(
        k_min, mu_min, k_cem, mu_cem, phi, frac_cem_up, phi_c, n, shear_red
    )

    # Handling of samples that violate the assumptions of the Patchy Cement
    # substitution:
    #   4: Model violation in Gassmann (k_dry < 0.0), friable model or constant cement
    #      model
    #   5: k_dry, mu below lower bound
    #   6: k_dry, mu above upper bound

    idx2 = _handle_exceptions_part_2(
        k_dry_old,
        mu_old,
        k_up,
        mu_up,
        k_low,
        mu_low,
        idx1,
        above_upper_bound=above_upper_bound,
        below_lower_bound=below_lower_bound,
    )
    idx_valid = np.logical_not(idx2)

    # Pressure sensitive model for initial pressure
    k_dry_p_init, mu_p_init = friable_model_dry(
        k_zero, mu_zero, phi, p_eff_old, phi_c, coord_num_func, n, shear_red
    )

    # Pressure sensitive model for final pressure
    k_dry_p_final, mu_p_final = friable_model_dry(
        k_zero, mu_zero, phi, p_eff_new, phi_c, coord_num_func, n, shear_red
    )

    # Weights - either on a sample basis or by a representative constant cement model
    if model_type == "weight":
        w_k = _calculate_weight(k_dry_old, k_low, k_up)
        w_mu = _calculate_weight(mu_old, mu_low, mu_up)
    else:
        k_cc, mu_cc = constant_cement_model_pcm(
            k_min,
            mu_min,
            k_cem,
            mu_cem,
            k_zero,
            mu_zero,
            phi,
            frac_cem,
            phi_c,
            n,
            shear_red,
        )[0:2]
        w_k = _calculate_weight(k_cc, k_low, k_up)
        w_mu = _calculate_weight(mu_cc, mu_low, mu_up)

    w_cem = 0.5 * (w_mu + w_k)
    k_mod_dry = np.nan * np.ones_like(phi)
    k_mod = np.nan * np.ones_like(phi)
    mu_mod = np.nan * np.ones_like(phi)
    rho_mod = np.nan * np.ones_like(phi)

    # Calculate modelled values - first for the valid samples
    k_mod_dry[idx_valid] = k_dry_p_init[idx_valid] + w_k[idx_valid] * (
        k_up[idx_valid] - k_low[idx_valid]
    )
    k_mod[idx_valid] = std_functions.gassmann(
        k_mod_dry[idx_valid], phi[idx_valid], k_fl_old[idx_valid], k_min[idx_valid]
    )
    mu_mod[idx_valid] = mu_p_init[idx_valid] + w_mu[idx_valid] * (
        mu_up[idx_valid] - mu_low[idx_valid]
    )
    rho_mod[idx_valid] = (
        phi[idx_valid] * rho_fl_old[idx_valid]
        + (1.0 - phi[idx_valid] - w_cem[idx_valid] * frac_cem_up) * rho_min[idx_valid]
        + w_cem[idx_valid] * frac_cem_up * rho_cem[idx_valid]
    )

    # Fill in input values for invalid samples
    k_mod[idx2] = k_sat_old[idx2]
    mu_mod[idx2] = mu_old[idx2]
    rho_mod[idx2] = rho_b_old[idx2]
    vp_mod, vs_mod = std_functions.velocity(k_mod, mu_mod, rho_mod)[0:2]

    # Calculate residuals. For invalid samples, the residuals will be set to zero
    vp_res = vp_old - vp_mod
    vs_res = vs_old - vs_mod
    rho_res = rho_b_old - rho_mod

    # Estimate dry moduli values according to initial dry values, changed
    # pressure and estimated pressure sensitivity
    k_dry_new = (
        k_dry_old
        * (k_dry_p_final + w_k * (k_up - k_dry_p_final))
        / (k_dry_p_init + w_k * (k_up - k_dry_p_init))
    )
    mu_new = (
        mu_old
        * (mu_p_final + w_mu * (mu_up - mu_p_final))
        / (mu_p_init + w_mu * (mu_up - mu_p_init))
    )

    # New k_sat for new pressure from Gassmann
    k_sat_new = np.ones_like(phi) * np.nan
    if np.any(idx_valid):
        k_sat_new[idx_valid] = std_functions.gassmann(
            k_dry_new[idx_valid], phi[idx_valid], k_fl_new[idx_valid], k_min[idx_valid]
        )
    k_sat_new[idx2] = k_sat_old[idx2]
    mu_new[idx2] = mu_old[idx2]

    # New saturated density
    rho_b_new = np.ones_like(phi) * np.nan
    rho_b_new[idx_valid] = rho_b_old[idx_valid] + phi[idx_valid] * (
        rho_fl_new[idx_valid] - rho_fl_old[idx_valid]
    )
    rho_b_new[idx2] = rho_b_old[idx2]

    # New saturated velocities and derived values
    vp_new, vs_new = std_functions.velocity(k_sat_new, mu_new, rho_b_new)[0:2]
    ai_new = vp_new * rho_b_new
    vpvs_new = vp_new / vs_new

    return (
        vp_new,
        vs_new,
        rho_b_new,
        ai_new,
        vpvs_new,
        w_k,
        w_mu,
        idx_valid,
        vp_res,
        vs_res,
        rho_res,
    )


def _handle_exceptions_part_1(
    phi_vec: np.ndarray,
    phi_c_const: float,
    k_sat: np.ndarray,
    mu: np.ndarray,
    k_min: np.ndarray,
    mu_min: np.ndarray,
    phi_below_zero: str = "disregard",
    phi_above_phi_c: str = "snap",
    k_sat_above_k_min: str = "disregard",
) -> np.ndarray:
    # Handling of samples that violate assumptions of the Patchy Cement substitution:
    #   1: phi < 0
    #   2: phi > phi_c
    #   3: k_sat > k_min, mu > mu_min

    # Handling of case 1:
    idx1 = phi_vec < 0.0
    if phi_below_zero == "snap":
        phi_vec[idx1] = 0.0
        idx1 = np.zeros(phi_vec.shape).astype(bool)
    elif phi_below_zero == "disregard":
        pass
    else:
        raise ValueError('unknown argument for parameter "phi_below_zero"')

    #  Handling of case 2:
    idx2 = phi_vec > phi_c_const
    if phi_above_phi_c == "snap":
        phi_vec[idx2] = phi_c_const
        idx2 = np.zeros(phi_vec.shape).astype(bool)
    elif phi_above_phi_c == "disregard":
        pass
    else:
        raise ValueError('unknown argument for parameter "phi_above_phi_c"')

    # Handling of case 3:
    idx3 = np.logical_or(k_sat > k_min, mu > mu_min)
    if k_sat_above_k_min == "snap":
        k_min[idx3] = k_sat[idx3]
        mu_min[idx3] = mu[idx3]
        idx3 = np.zeros(k_sat.shape).astype(bool)
    elif k_sat_above_k_min == "disregard":
        pass
    else:
        raise ValueError('unknown argument for parameter "k_sat_above_k_min"')
    return np.any(np.vstack((idx1, idx2, idx3)), axis=0)


def _handle_exceptions_part_2(
    k_dry: np.ndarray,
    mu: np.ndarray,
    k_up: np.ndarray,
    mu_up: np.ndarray,
    k_low: np.ndarray,
    mu_low: np.ndarray,
    idx: np.ndarray,
    above_upper_bound: str = "snap",
    below_lower_bound: str = "disregard",
) -> np.ndarray:
    # Handling of samples that violate the assumptions of the Patchy Cement
    # substitution:
    #   4: Model violation in Gassmann (k_dry < 0.0), friable model or constant cement
    #       model
    #   5: k_dry, mu below lower bound
    #   6: k_dry, mu above upper bound

    # Identification of case 4 and calculation of dry rock properties:
    # gassmann_dry returns dry property < 0 as NaN
    idx_nan = np.isnan(k_dry)

    # Can be non-physical cases in friable model
    idx_nan = np.logical_or(idx_nan, np.isnan(k_low))

    # Can be non-physical cases in constant cement model
    idx_nan = np.logical_or(idx_nan, np.isnan(k_up))

    # Handling of case 5:
    idx5k = np.zeros_like(k_dry).astype(bool)
    idx5mu = np.zeros_like(k_dry).astype(bool)
    np.less(k_dry, k_low, out=idx5k, where=~idx_nan)
    np.less(mu, mu_low, out=idx5mu, where=~idx_nan)
    if below_lower_bound == "snap":
        k_dry[idx5k] = k_low[idx5k]
        mu[idx5mu] = mu_low[idx5mu]
        idx5 = np.zeros(k_dry.shape).astype(bool)
    elif below_lower_bound == "disregard":
        idx5 = np.logical_or(idx5k, idx5mu)
    else:
        raise ValueError('unknown argument for parameter "below_lower_bound"')

    # Handling of case 6:
    if above_upper_bound == "snap":
        idx6k = np.zeros_like(k_dry).astype(bool)
        idx6mu = np.zeros_like(k_dry).astype(bool)
        np.greater(k_dry, k_up, out=idx6k, where=~idx_nan)
        np.greater(mu, mu_up, out=idx6mu, where=~idx_nan)
        k_dry[idx6k] = k_up[idx6k]
        mu[idx6mu] = mu_up[idx6mu]
        idx6 = np.zeros(k_dry.shape).astype(bool)
    elif above_upper_bound == "disregard":
        idx6 = np.logical_or(
            np.greater(k_dry, k_up, where=~idx_nan),
            np.greater(mu, mu_up, where=~idx_nan),
        )
    else:
        raise ValueError('unknown argument for parameter "above_upper_bound"')

    # Exception for all cases 1-6:
    return np.any(np.vstack((idx, idx5, idx6, idx_nan)), axis=0)


def _calculate_weight(
    dry_modulus: np.ndarray,
    low_modulus: np.ndarray,
    high_modulus: np.ndarray,
    force_full_range: str = False,
) -> np.ndarray:
    """
    Calculates a weight for a value between a lower and upper bound. Used for moduli


    Parameters
    ----------
    dry_modulus: value to be evaluated
    low_modulus: lower bound
    high_modulus: upper bound
    force_full_range: do not clip weight to a range of [0.0, 1.0]

    Returns
    -------
    weight: value between [0.0, 1.0]
    """
    idx = np.abs(high_modulus - low_modulus) < 2.0 * np.finfo(float).eps
    if np.any(idx):
        warn(
            f"weight estimation: high and low bound is identical for "
            f"{np.sum(high_modulus == low_modulus)} samples"
        )
    weight = np.ones_like(dry_modulus)
    weight[~idx] = (dry_modulus[~idx] - low_modulus[~idx]) / (
        high_modulus[~idx] - low_modulus[~idx]
    )
    # Catch cases outside the allowed range
    if not force_full_range:
        weight = np.clip(weight, 0.0, 1.0)
    return weight
