import numpy as np

from rock_physics_open import sandstone_models
from rock_physics_open.equinor_utilities import gen_utilities, std_functions


def friable_shaly_sand_sandy_shale_model(
    k_sst,
    mu_sst,
    rho_sst,
    k_mud,
    mu_mud,
    rho_mud,
    k_fl_sst,
    rho_fl_sst,
    k_fl_mud,
    rho_fl_mud,
    phi,
    p_eff_sst,
    p_eff_mud,
    shale_frac,
    phi_c_sst,
    phi_c_mud,
    phi_intr_mud,
    coord_num_func_sst,
    n_sst,
    coord_num_func_mud,
    n_mud,
    shear_red_sst,
    shear_red_mud,
):
    """
    Model for mixing of friable sand and friable shale.

    It is no point to use this model to calculate the shale response only,
    in that case Friable Model with shale parameters does the job.

    The shale fluid should be brine.

    Shale fraction shaleFrac is in the range 0 to 1. For shaleFrac = 0 we
    have a pure sand end member with phi = phiC for sand. For shaleFrac = 1
    we have pure shale with phi = intrinsic porosity. For shaleFrac < phiC
    the model is on the shaly sand trend, for shaleFrac > phiC it is on the
    sandy shale trend.

    Parameters
    ----------
    k_sst : np.ndarray
        Sandstone bulk modulus [Pa].
    mu_sst : np.ndarray
        Sandstone shear modulus [Pa].
    rho_sst : np.ndarray
        Sandstone bulk density [kg/m^3].
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
    p_eff_sst : np.ndarray
        Effective pressure in sandstone [Pa].
    p_eff_mud : np.ndarray
        Effective pressure in mud [Pa].
    shale_frac : np.ndarray
        Shale fraction [fraction].
    phi_c_sst : float
        Critical porosity for sandstone [fraction].
    phi_c_mud : float
        Critical porosity for mud [fraction].
    phi_intr_mud : float
        Intrinsic shale porosity [fraction].
    coord_num_func_sst : str
        Indication if coordination number should be calculated from porosity or kept constant for sandstone.
    coord_num_func_mud : str
        Indication if coordination number should be calculated from porosity or kept constant for shale.
    n_sst : float
        Coordination number for sandstone [unitless].
    n_mud : float
        Coordination number for shale [unitless].
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

    # Filter out values of phi that are above phi_c, assumed only to apply for the sandstone
    (
        idx_phi,
        (
            k_sst,
            mu_sst,
            rho_sst,
            k_mud,
            mu_mud,
            rho_mud,
            k_fl_sst,
            rho_fl_sst,
            k_fl_mud,
            rho_fl_mud,
            phi,
            p_eff_sst,
            p_eff_mud,
            shale_frac,
            _,
            _,
        ),
    ) = gen_utilities.filter_input_log(
        (
            k_sst,
            mu_sst,
            rho_sst,
            k_mud,
            mu_mud,
            rho_mud,
            k_fl_sst,
            rho_fl_sst,
            k_fl_mud,
            rho_fl_mud,
            phi,
            p_eff_sst,
            p_eff_mud,
            shale_frac,
            phi_c_sst - phi,
            phi - phi_intr_mud,
        ),
        no_zero=False,
    )

    # Expand the needed variables from float to numpy array
    phi, phi_intr_mud = gen_utilities.dim_check_vector((phi, phi_intr_mud))

    sandy_shale_idx = shale_frac > phi
    shaly_sand_idx = ~sandy_shale_idx

    # Fraction of silt in silt - shale trend
    frac_silt = (1 - shale_frac) / (1 - phi)
    # Fraction of sand in sand - silt trend
    frac_sand = 1 - shale_frac / phi

    # Shale properties for intrinsic porosity point
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

    # Silt end member
    k_silt, mu_silt = std_functions.hashin_shtrikman_walpole(
        k_sat_mud, mu_sat_mud, k_sst, mu_sst, phi_c_sst
    )
    rho_silt = rho_b_mud * phi + rho_sst * (1 - phi)

    # Estimate the sand end member through the friable model with phi = phiC
    vp_sat_sst, vs_sat_sst, rho_sat_sst = sandstone_models.friable_model(
        k_sst,
        mu_sst,
        rho_sst,
        k_fl_sst,
        rho_fl_sst,
        phi,
        p_eff_sst,
        phi_c_sst,
        coord_num_func_sst,
        n_sst,
        shear_red_sst,
    )[0:3]
    k_sat_sst, mu_sat_sst = std_functions.moduli(vp_sat_sst, vs_sat_sst, rho_sat_sst)

    k = np.ones(shale_frac.shape) * np.nan
    mu = np.ones(shale_frac.shape) * np.nan
    rho = np.ones(shale_frac.shape) * np.nan

    # Points on sandy shale trend
    k[sandy_shale_idx], mu[sandy_shale_idx] = std_functions.hashin_shtrikman_walpole(
        k_silt[sandy_shale_idx],
        mu_silt[sandy_shale_idx],
        k_sat_mud[sandy_shale_idx],
        mu_sat_mud[sandy_shale_idx],
        frac_silt[sandy_shale_idx],
    )

    rho[sandy_shale_idx] = (
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

    rho[shaly_sand_idx] = (
        (1 - phi[shaly_sand_idx]) * rho_sst[shaly_sand_idx]
        + phi[shaly_sand_idx] * rho_fl_sst[shaly_sand_idx]
    ) * frac_sand[shaly_sand_idx] + (1 - frac_sand[shaly_sand_idx]) * rho_silt[
        shaly_sand_idx
    ]

    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rho)

    # Restore original array length
    vp, vs, rho, ai, vpvs = gen_utilities.filter_output(
        idx_phi, (vp, vs, rho, ai, vpvs)
    )

    return vp, vs, rho, ai, vpvs
