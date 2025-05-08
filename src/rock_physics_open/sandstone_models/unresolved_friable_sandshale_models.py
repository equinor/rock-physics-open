from rock_physics_open import sandstone_models as sm
from rock_physics_open.equinor_utilities import std_functions


def unresolved_friable_sand_shale_model(
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
    phi_sst,
    phi_mud,
    p_eff_sst,
    p_eff_mud,
    shale_frac,
    phi_c_sst,
    phi_c_mud,
    coord_num_func_sst,
    n_sst,
    coord_num_func_mud,
    n_mud,
    shear_red_sst,
    shear_red_mud,
):
    """
    Model for siliciclastic rocks with alternating layers of friable sand and shale, and in which the layers are not
    resolved by the investigating signal. Backus average is used to calculate the anisotropic effect of the alternating
    layers.

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
        Fluid bulk density for shale fluid [kg/m^3].
    phi_sst : np.ndarray
        Sandstone porosity [fraction].
    phi_mud : np.ndarray
        Shale porosity [fraction].
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
        vpv, vsv, vph, vsh, rho : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vertical p-wave velocity, vertical shear-wave velocity, horizontal p-wave velocity, horizontal shear-wave
        velocity (all [m/s]), bulk density [kg/m^3].
    """

    # Estimate the sand and shale end members through the friable models
    vp_sst, vs_sst, rho_b_sst = sm.friable_model(
        k_sst,
        mu_sst,
        rho_sst,
        k_fl_sst,
        rho_fl_sst,
        phi_sst,
        p_eff_sst,
        phi_c_sst,
        coord_num_func_sst,
        n_sst,
        shear_red_sst,
    )[0:3]

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
