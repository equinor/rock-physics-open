import numpy as np

from rock_physics_open import sandstone_models
from rock_physics_open.equinor_utilities import gen_utilities, std_functions


def cemented_shaly_sand_sandy_shale_model(
    k_sst,
    mu_sst,
    rho_sst,
    k_cem,
    mu_cem,
    rho_cem,
    k_mud,
    mu_mud,
    rho_mud,
    k_fl_sst,
    rho_fl_sst,
    k_fl_mud,
    rho_fl_mud,
    phi,
    p_eff_mud,
    shale_frac,
    frac_cem,
    phi_c_sst,
    n_sst,
    shear_red_sst,
    phi_c_mud,
    phi_intr_mud,
    coord_num_func_mud,
    n_mud,
    shear_red_mud,
):
    """
    Model for mixing of cemented sand and friable shale.

    It is no point to use this model to calculate the shale response only,
    in that case Friable Model with shale parameters does the job

    The shale fluid should be brine.

    Shale fraction shaleFrac is in the range 0 to 1. For shaleFrac = 0 we
    have a pure sand end member with phi =< phi_c for sand. For shaleFrac = 1
    we have pure shale with phi = intrinsic porosity. For shaleFrac < phi_c
    the model is on the shaly sand trend, for shaleFrac > phi_c it is on the
    sandy shale trend.

    Parameters
    ----------
    k_sst : np.ndarray
        Sandstone matrix bulk modulus [Pa].
    mu_sst : np.ndarray
        Sandstone matrix shear modulus [Pa].
    rho_sst : np.ndarray
        Sandstone matrix bulk density [kg/m^3].
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Sandstone cement bulk density [kg/m^3].
    k_mud : np.ndarray
        Shale bulk modulus [Pa].
    mu_mud : np.ndarray
        Shale shear modulus [Pa].
    rho_mud : np.ndarray
        Shale bulk density [kg/m^3].
    k_fl_sst : np.ndarray
        Fluid bulk modulus for sandstone fluid [Pa].
    rho_fl_sst : np.ndarray
        Fluid bulk density for sandstone fluid [kg/m^3].
    k_fl_mud : np.ndarray
        Fluid bulk modulus for shale fluid [Pa].
    rho_fl_mud : np.ndarray
        Fluid bulk density for shale fluid[kg/m^3].
    phi : np.ndarray
        Total porosity [fraction].
    p_eff_mud : np.ndarray
        Effective pressure in mud [Pa].
    shale_frac : np.ndarray
        Shale fraction [fraction].
    frac_cem : np.ndarray
        Cement volume fraction [fraction].
    phi_c_sst : float
        Critical porosity for sandstone[fraction].
    phi_c_mud : float
        Critical porosity for mud [fraction].
    phi_intr_mud : float
        Intrinsic porosity for mud [fraction].
    n_sst : float
        Coordination number for sandstone [unitless].
    n_mud : float
        Coordination number for shale [unitless].
    coord_num_func_mud : str
        Indication if coordination number should be calculated from porosity or kept constant for shale.
    shear_red_sst : float
        Shear reduction factor for sandstone [fraction].
    shear_red_mud : float
        Shear reduction factor for mud [fraction].

    Returns
    -------
    tuple
        vp, vs, rho, ai, vpvs  : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        vp [m/s] and vs [m/s], bulk density [kg/m^3], ai [m/s x kg/m^3], vpvs [ratio] of saturated rock.
    """

    # Valid porosity values must not exceed (phi_c - frac_cem)
    # Assume that this only needs to be considered for the sandstone fraction
    (
        k_sst,
        mu_sst,
        rho_sst,
        k_cem,
        mu_cem,
        rho_cem,
        k_mud,
        mu_mud,
        rho_mud,
        k_fl_sst,
        rho_fl_sst,
        k_fl_mud,
        rho_fl_mud,
        phi,
        p_eff_mud,
        shale_frac,
        frac_cem,
    ) = gen_utilities.dim_check_vector(
        (
            k_sst,
            mu_sst,
            rho_sst,
            k_cem,
            mu_cem,
            rho_cem,
            k_mud,
            mu_mud,
            rho_mud,
            k_fl_sst,
            rho_fl_sst,
            k_fl_mud,
            rho_fl_mud,
            phi,
            p_eff_mud,
            shale_frac,
            frac_cem,
        )
    )
    (
        idx_phi,
        (
            k_sst,
            mu_sst,
            rho_sst,
            k_cem,
            mu_cem,
            rho_cem,
            k_mud,
            mu_mud,
            rho_mud,
            k_fl_sst,
            rho_fl_sst,
            k_fl_mud,
            rho_fl_mud,
            phi,
            p_eff_mud,
            shale_frac,
            frac_cem,
            _,
            _,
        ),
    ) = gen_utilities.filter_input_log(
        (
            k_sst,
            mu_sst,
            rho_sst,
            k_cem,
            mu_cem,
            rho_cem,
            k_mud,
            mu_mud,
            rho_mud,
            k_fl_sst,
            rho_fl_sst,
            k_fl_mud,
            rho_fl_mud,
            phi,
            p_eff_mud,
            shale_frac,
            frac_cem,
            phi_c_sst - frac_cem - phi,
            phi - phi_intr_mud,
        ),
        no_zero=False,
    )

    # Reduce range of porosity by frac_cem
    phi_c = phi_c_sst - frac_cem

    sandy_shale_idx = shale_frac > phi
    shaly_sand_idx = ~sandy_shale_idx

    # Fraction of silt in silt - shale trend
    frac_silt = (1 - shale_frac) / (1 - phi)
    # Fraction of sand in sand - silt trend
    frac_sand = 1 - shale_frac / phi

    # Shale properties for intrinsic porosity point NB!  The phi_intr_mud is
    # normally a parameter, but the assumption in Friable model is that it is a
    # log.  Make sure that it is of the same length as the other

    # Expand the needed variables from float to numpy array
    phi, phi_intr_mud = gen_utilities.dim_check_vector((phi, phi_intr_mud))

    vp_sat_mud, vs_sat_mud, rho_b_mud = sandstone_models.friable_model(
        k_mud,
        mu_mud,
        rho_mud,
        k_fl_mud,
        rho_fl_mud,
        phi_intr_mud,
        p_eff_mud,
        phi_c_mud,
        coord_num_func_mud,
        n_mud,
        shear_red_mud,
    )[0:3]
    k_sat_mud, mu_sat_mud = std_functions.moduli(vp_sat_mud, vs_sat_mud, rho_b_mud)

    # Calculate cemented zero-porosity sand
    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_sst, mu_sst, frac_cem, bound="lower"
    )
    rho_zero = rho_cem * frac_cem + (1 - frac_cem) * rho_sst

    # Silt end member
    k_silt, mu_silt = std_functions.hashin_shtrikman_walpole(
        k_sat_mud, mu_sat_mud, k_zero, mu_zero, phi_c
    )
    rho_silt = rho_b_mud * phi + rho_zero * (1 - phi)

    # Estimate the sand end member through the constant cement model with phi =
    # phiCSst <= maybe dubious to expand parameter to vector with assumption
    # that kSst has the correct size

    vp_sat_sst, vs_sat_sst, rho_sat_sst = sandstone_models.constant_cement_model(
        k_sst,
        mu_sst,
        rho_sst,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_sst,
        rho_fl_sst,
        phi,
        frac_cem,
        phi_c_sst,
        n_sst,
        shear_red_sst,
    )[0:3]
    k_sat_sst, mu_sat_sst = std_functions.moduli(vp_sat_sst, vs_sat_sst, rho_sat_sst)

    k = np.ones(shale_frac.shape) * np.nan
    mu = np.ones(shale_frac.shape) * np.nan
    rhob = np.ones(shale_frac.shape) * np.nan

    # Points on sandy shale trend
    k[sandy_shale_idx], mu[sandy_shale_idx] = std_functions.hashin_shtrikman_walpole(
        k_silt[sandy_shale_idx],
        mu_silt[sandy_shale_idx],
        k_sat_mud[sandy_shale_idx],
        mu_sat_mud[sandy_shale_idx],
        frac_silt[sandy_shale_idx],
    )

    rhob[sandy_shale_idx] = (
        rho_b_mud[sandy_shale_idx] * (1 - frac_silt[sandy_shale_idx])
        + rho_silt[sandy_shale_idx] * frac_silt[sandy_shale_idx]
    )

    # Points on shaly sand trend
    k[shaly_sand_idx], mu[shaly_sand_idx] = std_functions.hashin_shtrikman_walpole(
        k_sat_sst[shaly_sand_idx],
        mu_sat_sst[shaly_sand_idx],
        k_silt[shaly_sand_idx],
        mu_silt[shaly_sand_idx],
        frac_sand[shaly_sand_idx],
    )

    rhob[shaly_sand_idx] = (
        (1 - phi[shaly_sand_idx]) * rho_zero[shaly_sand_idx]
        + phi[shaly_sand_idx] * rho_fl_sst[shaly_sand_idx]
    ) * frac_sand[shaly_sand_idx] + (1 - frac_sand[shaly_sand_idx]) * rho_silt[
        shaly_sand_idx
    ]

    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rhob)

    # Restore original length
    vp, vs, rhob, ai, vpvs = gen_utilities.filter_output(
        idx_phi, (vp, vs, rhob, ai, vpvs)
    )

    return vp, vs, rhob, ai, vpvs
