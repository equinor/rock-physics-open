from rock_physics_open import sandstone_models as sm
from rock_physics_open.equinor_utilities import std_functions


def unresolved_constant_cement_sand_shale_model(
    k_min_sst,
    mu_min_sst,
    rho_min_sst,
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
    phi_sst,
    phi_mud,
    p_eff_mud,
    shale_frac,
    frac_cem,
    phi_c_sst,
    phi_c_mud,
    n_sst,
    coord_num_func_mud,
    n_mud,
    shear_red_sst,
    shear_red_mud,
):
    """
    Model for silisiclastic rocks with alternating layers of cemented sand and friable shale, and in which the layers
    are not resolved by the investigating signal. Backus average is used to calculate the anisotropic effect of the
    alternating layers.

    Parameters
    ----------
    k_min_sst : np.ndarray
        Sandstone matrix bulk modulus [Pa].
    mu_min_sst : np.ndarray
        Sandstone matrix shear modulus [Pa].
    rho_min_sst : np.ndarray
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
    phi_sst : np.ndarray
        Sandstone porosity [fraction].
    phi_mud : np.ndarray
        Shale porosity [fraction].
    p_eff_mud : np.ndarray
        Effective pressure in mud [Pa].
    shale_frac : np.ndarray
        Shale fraction [fraction].
    frac_cem : float
        Cement volume fraction [fraction].
    phi_c_sst : float
        Critical porosity for sandstone [fraction].
    phi_c_mud : float
        Critical porosity for mud [fraction].
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
        vpv, vsv, vph, vsh, rho : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vertical p-wave velocity, vertical shear-wave velocity, horizontal p-wave velocity, horizontal shear-wave
        velocity (all [m/s]), bulk density [kg/m^3].
    """
    # Estimate the sand end member through the constant cement model
    vp_sst, vs_sst, rho_b_sst = sm.constant_cement_model(
        k_min_sst,
        mu_min_sst,
        rho_min_sst,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_sst,
        rho_fl_sst,
        phi_sst,
        frac_cem,
        phi_c_sst,
        n_sst,
        shear_red_sst,
    )[0:3]

    # Estimate the shale end member through the friable model
    vp_mud, vs_mud, rho_b_mud = sm.friable_model(
        k_mud,
        mu_mud,
        rho_mud,
        k_fl_mud,
        rho_fl_mud,
        phi_mud,
        p_eff_mud,
        phi_c_mud,
        coord_num_func_mud,
        n_mud,
        shear_red_mud,
    )[0:3]

    # Calculate Backus average for the effective medium
    vpv, vsv, vph, vsh, rho = std_functions.backus_average(
        vp_sst, vs_sst, rho_b_sst, vp_mud, vs_mud, rho_b_mud, 1.0 - shale_frac
    )

    return vpv, vsv, vph, vsh, rho
